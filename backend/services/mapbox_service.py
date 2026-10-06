"""
Mapbox service: forward/reverse geocoding and driving-route directions.

Endpoints
---------
- Geocoding v6 forward/reverse (search/geocode/v6)
- Directions v5 (directions/v5/mapbox/{profile}) with alternatives

Design
------
- Async httpx client, shared and reused, with an explicit connect+read timeout.
- The access token is read from settings.MAPBOX_ACCESS_TOKEN and never logged,
  never returned in any payload, and never sent to the frontend.
- Errors are turned into typed exceptions, not crashes, mapped from Mapbox's
  documented contract:
    * geocoding/directions bodies carry a "code" (Ok, NoRoute, NoSegment,
      InvalidInput, ...) for HTTP < 500; "message" is human-readable.
    * HTTP 401/403 = auth/account; 404 = bad profile; 422 = invalid input;
      429 = rate limit; >=500 = upstream; network/timeout = connectivity.
  Callers (the /api/* routes) map these to clean HTTP responses.
- geocode() returns a ranked list of candidates (name/lat/lon/relevance/...),
  never a blind first pick, so the caller/UI can disambiguate.
- get_routes() returns >=1 route with GeoJSON LineString geometry; "no route"
  is a typed NoRouteError, not an empty success.

Storing results
---------------
Forward/reverse geocoding here defaults to temporary (permanent=False), the
correct choice for a live user search whose coordinates are not persisted. The
one place we DO persist geocodes (the historical waterlogging build) uses a
separate script that sets permanent=true.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import settings

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://api.mapbox.com/search/geocode/v6/{kind}"
DIRECTIONS_URL = "https://api.mapbox.com/directions/v5/mapbox/{profile}/{coords}"
DEFAULT_TIMEOUT_S = 8.0
VALID_PROFILES = ("driving", "driving-traffic", "walking", "cycling")

# Bias/limit results to Kolkata: (min_lon, min_lat, max_lon, max_lat).
KOLKATA_BBOX = (88.20, 22.40, 88.55, 22.75)
KOLKATA_CENTER = (88.3639, 22.5726)  # proximity bias (lon, lat)
# Geocoding v6 feature types (no "poi"; that's a v5 type). address/street give
# routable points; the rest let a user pick a neighbourhood or locality.
GEO_FEATURE_TYPES = "address,street,neighborhood,locality,place,district"


# ---------------------------------------------------------------- exceptions

class MapboxError(Exception):
    """Base class. `status` is an HTTP status hint for the API layer."""
    status = 502

    def __init__(self, message: str, *, code: Optional[str] = None, detail: Any = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.detail = detail

    def to_dict(self) -> Dict:
        d = {"error": type(self).__name__, "message": self.message}
        if self.code:
            d["mapbox_code"] = self.code
        return d


class MapboxNotConfiguredError(MapboxError):
    status = 503


class MapboxAuthError(MapboxError):
    status = 502  # our misconfig, not the client's fault; don't leak 401 downstream


class MapboxRateLimitError(MapboxError):
    status = 429


class InvalidInputError(MapboxError):
    status = 400


class NoRouteError(MapboxError):
    status = 404


class MapboxUnavailableError(MapboxError):
    status = 502


# ---------------------------------------------------------------- datatypes

@dataclass
class GeocodeCandidate:
    name: str
    full_address: Optional[str]
    lat: float
    lon: float
    feature_type: Optional[str]
    mapbox_id: Optional[str]
    # Candidates arrive already ranked (best first); v6 has no numeric relevance
    # score. match_confidence is set only for address-type results (Smart Match).
    match_confidence: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name, "full_address": self.full_address,
            "lat": round(self.lat, 6), "lon": round(self.lon, 6),
            "feature_type": self.feature_type, "mapbox_id": self.mapbox_id,
            "match_confidence": self.match_confidence, "context": self.context,
        }


@dataclass
class RouteStep:
    instruction: str
    distance_m: float
    duration_s: float
    name: str


@dataclass
class Route:
    index: int
    distance_m: float
    duration_s: float
    geometry: Dict[str, Any]          # GeoJSON LineString {"type","coordinates":[[lon,lat],...]}
    steps: List[RouteStep] = field(default_factory=list)
    weight_name: Optional[str] = None

    @property
    def coordinates(self) -> List[Tuple[float, float]]:
        return [(c[0], c[1]) for c in self.geometry.get("coordinates", [])]

    def to_dict(self, include_steps: bool = True) -> Dict:
        d = {
            "index": self.index,
            "distance_m": round(self.distance_m, 1),
            "distance_km": round(self.distance_m / 1000, 2),
            "duration_s": round(self.duration_s, 1),
            "duration_min": round(self.duration_s / 60, 1),
            "geometry": self.geometry,
        }
        if include_steps:
            d["steps"] = [{"instruction": s.instruction, "distance_m": round(s.distance_m, 1),
                           "duration_s": round(s.duration_s, 1), "name": s.name} for s in self.steps]
        return d


def _valid_lat_lon(lat: float, lon: float) -> bool:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and abs(lat) <= 90 and abs(lon) <= 180


# ------------------------------------------------------------------ service

class MapboxService:
    def __init__(self, token: Optional[str], timeout_s: float = DEFAULT_TIMEOUT_S,
                 client: Optional[httpx.AsyncClient] = None):
        self._token = token
        self._timeout = httpx.Timeout(timeout_s, connect=min(5.0, timeout_s))
        self._client = client
        self._owns_client = client is None
        self.last_success_ts: Optional[float] = None
        self.last_error: Optional[str] = None

    @property
    def configured(self) -> bool:
        return bool(self._token and self._token.strip())

    async def _get_client(self) -> httpx.AsyncClient:
        # Bind the owned client to its event loop; if the loop changed or closed
        # (e.g. a process-wide singleton reused across requests), recreate it.
        # A caller-injected client (tests) is used as-is.
        if self._client is None:
            import asyncio
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._client_loop = asyncio.get_event_loop()
        elif self._owns_client:
            import asyncio
            loop = asyncio.get_event_loop()
            if getattr(self, "_client_loop", None) is not loop or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=self._timeout)
                self._client_loop = loop
        return self._client

    async def aclose(self):
        if self._owns_client and self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def health(self) -> Dict:
        """Cached connectivity, from the last real call. No round-trip here."""
        return {"configured": self.configured, "last_success_ts": self.last_success_ts,
                "last_error": self.last_error}

    async def ping(self) -> Dict:
        """Explicit lightweight connectivity check (one cheap geocode)."""
        try:
            await self.geocode("Kolkata", limit=1)
            return {"connected": True, "last_success_ts": self.last_success_ts}
        except MapboxError as exc:
            return {"connected": False, "error": exc.message}

    async def _request(self, url: str, params: Dict, *, context: str) -> Dict:
        if not self.configured:
            raise MapboxNotConfiguredError("Mapbox access token is not configured")
        params = {**params, "access_token": self._token}
        client = await self._get_client()
        try:
            resp = await client.get(url, params=params)
        except httpx.TimeoutException as exc:
            self.last_error = f"{context} timed out"
            raise MapboxUnavailableError(f"Mapbox {context} timed out") from exc
        except httpx.HTTPError as exc:
            self.last_error = f"{context} connection failed"
            raise MapboxUnavailableError(f"Mapbox {context} connection failed: {exc}") from exc

        body = self._safe_json(resp)
        code = body.get("code") if isinstance(body, dict) else None
        msg = (body.get("message") if isinstance(body, dict) else None) or f"HTTP {resp.status_code}"

        if resp.status_code == 200 and code in (None, "Ok"):
            self.last_success_ts = time.time()
            self.last_error = None
            return body
        # HTTP 200 but a non-Ok body code (NoRoute / NoSegment)
        if resp.status_code == 200:
            if code in ("NoRoute", "NoSegment"):
                raise NoRouteError(msg or "No route found", code=code)
            raise MapboxError(msg, code=code)
        if resp.status_code in (401, 403):
            # Log server-side; surface a generic message so we never hint the token.
            logger.error("Mapbox %s auth error (HTTP %s, code=%s)", context, resp.status_code, code)
            raise MapboxAuthError("Mapbox request was not authorized (server configuration issue)", code=code)
        if resp.status_code == 404:
            raise InvalidInputError(msg or "Not found", code=code)
        if resp.status_code == 422:
            raise InvalidInputError(msg or "Invalid input", code=code)
        if resp.status_code == 429:
            raise MapboxRateLimitError("Mapbox rate limit exceeded", code=code)
        if resp.status_code >= 500:
            raise MapboxUnavailableError(f"Mapbox {context} upstream error (HTTP {resp.status_code})", code=code)
        raise MapboxError(msg, code=code)

    @staticmethod
    def _safe_json(resp: httpx.Response) -> Dict:
        try:
            data = resp.json()
            return data if isinstance(data, dict) else {"_raw": data}
        except ValueError:
            return {}

    # --------------------------------------------------------------- geocode

    async def geocode(self, query: str, *, limit: int = 5, country: str = "in",
                      bbox: Optional[Tuple[float, float, float, float]] = KOLKATA_BBOX,
                      proximity: Optional[Tuple[float, float]] = KOLKATA_CENTER,
                      permanent: bool = False) -> List[GeocodeCandidate]:
        """Forward geocode. Returns ranked candidates (possibly empty), never a blind pick."""
        q = (query or "").strip()
        if not q:
            raise InvalidInputError("query must not be empty")
        params: Dict[str, Any] = {"q": q, "limit": max(1, min(int(limit), 10)),
                                  "types": GEO_FEATURE_TYPES, "permanent": str(bool(permanent)).lower()}
        if country:
            params["country"] = country
        if bbox:
            params["bbox"] = ",".join(str(v) for v in bbox)
        if proximity:
            params["proximity"] = f"{proximity[0]},{proximity[1]}"
        body = await self._request(GEOCODE_URL.format(kind="forward"), params, context="geocode")
        return self._parse_candidates(body)

    async def reverse_geocode(self, lat: float, lon: float, *, limit: int = 1,
                              permanent: bool = False) -> List[GeocodeCandidate]:
        if not _valid_lat_lon(lat, lon):
            raise InvalidInputError("lat/lon out of range or not finite")
        params = {"longitude": float(lon), "latitude": float(lat),
                  "limit": max(1, min(int(limit), 5)), "permanent": str(bool(permanent)).lower()}
        body = await self._request(GEOCODE_URL.format(kind="reverse"), params, context="reverse geocode")
        return self._parse_candidates(body)

    @staticmethod
    def _parse_candidates(body: Dict) -> List[GeocodeCandidate]:
        out: List[GeocodeCandidate] = []
        for f in body.get("features", []):
            geom = f.get("geometry") or {}
            coords = geom.get("coordinates") or []
            if len(coords) < 2 or not _valid_lat_lon(coords[1], coords[0]):
                continue
            p = f.get("properties", {})
            ctx = p.get("context", {}) or {}
            out.append(GeocodeCandidate(
                name=p.get("name") or p.get("name_preferred") or (f.get("text") or "unknown"),
                full_address=p.get("full_address") or p.get("place_formatted"),
                lon=float(coords[0]), lat=float(coords[1]),
                feature_type=p.get("feature_type"),
                mapbox_id=p.get("mapbox_id"),
                match_confidence=(p.get("match_code") or {}).get("confidence"),
                context={k: (v.get("name") if isinstance(v, dict) else v)
                         for k, v in ctx.items() if k in ("place", "locality", "neighborhood", "postcode", "region")},
            ))
        return out

    # ------------------------------------------------------------- routing

    async def get_routes(self, origin: Tuple[float, float], destination: Tuple[float, float], *,
                         profile: str = "driving", alternatives: bool = True) -> List[Route]:
        """Driving routes between (lat, lon) origin and destination. >=1 route or NoRouteError."""
        if profile not in VALID_PROFILES:
            raise InvalidInputError(f"profile must be one of {VALID_PROFILES}")
        for name, pt in (("origin", origin), ("destination", destination)):
            if not (isinstance(pt, (tuple, list)) and len(pt) == 2 and _valid_lat_lon(pt[0], pt[1])):
                raise InvalidInputError(f"{name} must be a valid (lat, lon)")
        # Mapbox wants lon,lat;lon,lat
        coords = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
        params = {"alternatives": str(bool(alternatives)).lower(), "overview": "full",
                  "geometries": "geojson", "steps": "true"}
        body = await self._request(DIRECTIONS_URL.format(profile=profile, coords=coords),
                                   params, context="directions")
        routes = self._parse_routes(body)
        if not routes:
            raise NoRouteError("No route found between the given points", code=body.get("code"))
        return routes

    @staticmethod
    def _parse_routes(body: Dict) -> List[Route]:
        out: List[Route] = []
        for i, r in enumerate(body.get("routes", [])):
            geom = r.get("geometry")
            if not (isinstance(geom, dict) and geom.get("type") == "LineString" and len(geom.get("coordinates", [])) >= 2):
                continue
            steps = []
            for leg in r.get("legs", []):
                for s in leg.get("steps", []):
                    man = s.get("maneuver", {})
                    steps.append(RouteStep(
                        instruction=man.get("instruction", ""),
                        distance_m=float(s.get("distance", 0.0)),
                        duration_s=float(s.get("duration", 0.0)),
                        name=s.get("name", ""),
                    ))
            out.append(Route(index=i, distance_m=float(r.get("distance", 0.0)),
                             duration_s=float(r.get("duration", 0.0)), geometry=geom,
                             steps=steps, weight_name=r.get("weight_name")))
        return out


_service: Optional[MapboxService] = None


def get_mapbox_service() -> MapboxService:
    """Process-wide service. Token read once from settings."""
    global _service
    if _service is None:
        token = settings.MAPBOX_ACCESS_TOKEN.get_secret_value() if settings.MAPBOX_ACCESS_TOKEN else None
        _service = MapboxService(token)
    return _service

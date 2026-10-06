"""
Real-data endpoints (Task 18): health, weather, geocode, reverse-geocode,
route, route flood-risk, safe-route, data status.

These are additive. The legacy synthetic endpoints in main.py
(/api/predict, /api/predict/batch, /api/model/info, /api/nowcast, /api/drainage)
are untouched.

Every response that carries a prediction names its model version and prediction
type, so real_v1 output is never mistaken for the synthetic prototype and vice
versa. /api/safe-route never claims a route is "safe".
"""

import time
from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from services.flood_model import get_flood_model
from services.imd_service import get_imd_service
from services.mapbox_service import (
    MapboxError, MapboxNotConfiguredError, NoRouteError, get_mapbox_service,
)
from services.route_risk_service import get_route_risk_service
from services.model_registry import real_descriptor, synthetic_descriptor
from services.safe_route_service import pick_recommendation, risk_level_for_route, route_score
from services.spatial_service import get_spatial_service

router = APIRouter()


class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=256)
    full_name: Optional[str] = Field(None, max_length=120)


class LoginRequest(BaseModel):
    identifier: str = Field(..., min_length=1, max_length=254)
    password: str = Field(..., min_length=1, max_length=256)


class ProfileRequest(BaseModel):
    email: Optional[str] = Field(None, min_length=3, max_length=254)
    full_name: Optional[str] = Field(None, max_length=120)


def _bearer(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required",
                            headers={"WWW-Authenticate": "Bearer"})
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Bearer token required",
                            headers={"WWW-Authenticate": "Bearer"})
    return token


def _auth_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=409 if "registered" in str(exc) else 401,
                        detail=str(exc))


@router.post("/api/auth/signup", status_code=201)
def auth_signup(req: SignupRequest):
    from services.auth_service import AuthError, login, signup
    try:
        signup(req.username, req.email, req.password, req.full_name)
        return login(req.email, req.password)
    except AuthError as exc:
        raise _auth_error(exc)


@router.post("/api/auth/login")
def auth_login(req: LoginRequest):
    from services.auth_service import AuthError, login
    try:
        return login(req.identifier, req.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc),
                            headers={"WWW-Authenticate": "Bearer"})


@router.get("/api/auth/me")
def auth_me(authorization: Optional[str] = Header(None)):
    from services.auth_service import AuthError, me
    try:
        return {"user": me(_bearer(authorization))}
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc),
                            headers={"WWW-Authenticate": "Bearer"})


@router.put("/api/auth/profile")
def auth_profile(req: ProfileRequest, authorization: Optional[str] = Header(None)):
    from services.auth_service import AuthError, update_profile
    try:
        return {"user": update_profile(_bearer(authorization), req.email, req.full_name)}
    except AuthError as exc:
        status = 401 if "token" in str(exc) or "user" in str(exc) else 409
        raise HTTPException(status_code=status, detail=str(exc))


def _real_model_tag() -> dict:
    """The four section-40 fields (+ meaning), from the single registry source."""
    d = real_descriptor()
    return {"model_version": d["model_version"], "prediction_type": d["prediction_type"],
            "training_data_type": d["training_data_type"], "features_used": d["features_used"],
            "last_trained_at": d["last_trained_at"], "prediction_meaning": d["prediction_meaning"]}


def _mapbox_http(exc: MapboxError) -> HTTPException:
    return HTTPException(status_code=getattr(exc, "status", 502), detail=exc.to_dict())


# ----------------------------------------------------------------- health

@router.get("/api/health")
def health():
    return {"status": "ok", "service": "floodwatch", "time": int(time.time())}


# ----------------------------------------------------------------- geocode

@router.get("/api/geocode")
async def geocode(q: str = Query(..., min_length=1), limit: int = Query(5, ge=1, le=10)):
    svc = get_mapbox_service()
    try:
        cands = await svc.geocode(q, limit=limit)
    except MapboxError as exc:
        raise _mapbox_http(exc)
    return {"query": q, "candidates": [c.to_dict() for c in cands]}


@router.get("/api/reverse-geocode")
async def reverse_geocode(lat: float = Query(...), lon: float = Query(...)):
    svc = get_mapbox_service()
    try:
        cands = await svc.reverse_geocode(lat, lon)
    except MapboxError as exc:
        raise _mapbox_http(exc)
    return {"lat": lat, "lon": lon, "candidates": [c.to_dict() for c in cands]}


# ----------------------------------------------------------------- weather

@router.get("/api/weather")
async def weather(station: Optional[str] = None):
    svc = get_imd_service()
    wx = await svc.get_weather(station) if station else await svc.get_weather()
    return wx  # already a safe dict with available flag


@router.get("/api/location-risk")
async def location_risk(lat: float = Query(..., ge=-90, le=90),
                        lon: float = Query(..., ge=-180, le=180),
                        horizon_minutes: int = Query(0, ge=0, le=300),
                        simulated_rainfall_mm: Optional[float] = Query(None)):
    """Real flood-susceptibility for a single point (e.g. the user's location).

    Uses the same real pipeline as safe-route (DEM + KMC drainage + historical
    waterlogging + live IMD rainfall + real_v1) for ONE point. It is honest
    about coverage: outside the KMC-derived data, features are null and the
    response says so instead of inventing a number. No manual sliders.
    """
    from services.dem_service import get_dem_service
    dem = get_dem_service()
    spatial = get_spatial_service()
    model = get_flood_model()
    imd = get_imd_service()

    terrain = dem.get_terrain_features([(lat, lon)])[0]
    feats = spatial.get_spatial_features([(lat, lon)])[0]
    wx = await imd.get_weather()

    drain = feats.get("drain") or {}
    water = feats.get("water_body") or {}
    hist = feats.get("historical") or {}

    segment = {
        "lat": lat, "lng": lon,
        "elevation": terrain.get("elevation_m"),
        "slope": terrain.get("slope_percent"),
        "distance_to_drain_m": drain.get("distance_to_drain_m") if drain.get("found") else None,
        "pipe_diameter_mm": drain.get("pipe_diameter_mm") if drain.get("found") else None,
        "drain_capacity_estimated_m3s": drain.get("drain_capacity_estimated_m3s") if drain.get("found") else None,
        "distance_to_waterbody_m": water.get("distance_to_waterbody_m") if water.get("found") else None,
    }
    rainfall_mm = simulated_rainfall_mm if simulated_rainfall_mm is not None else wx.get("rainfall_24h")
    scored = model.score_segments([segment], rainfall_24h_mm=rainfall_mm, horizon_minutes=horizon_minutes)[0]

    within_kmc = hist.get("within_records_coverage")
    if scored.get("flood_probability") is None:
        coverage = "no_data"
    elif within_kmc:
        coverage = "full"
    else:
        coverage = "terrain_only"  # DEM works city-wide; drainage/history do not

    return {
        **_real_model_tag(),
        "model_status": model.status,
        "location": {"lat": lat, "lon": lon},
        "coverage": coverage,
        "within_kmc_data_area": within_kmc,
        "flood_probability": scored.get("flood_probability"),
        "susceptibility": scored.get("susceptibility"),
        "risk_level": scored.get("risk_level"),
        "elevation_m": terrain.get("elevation_m"),
        "slope_percent": terrain.get("slope_percent"),
        "nearest_drain_m": segment["distance_to_drain_m"],
        "pipe_diameter_mm": segment["pipe_diameter_mm"],
        "nearest_waterbody_m": segment["distance_to_waterbody_m"],
        "historical_waterlogging": hist.get("historical_waterlogging"),
        "historical_event_count": hist.get("historical_event_count"),
        "distance_to_historical_waterlogging_m": hist.get("distance_to_historical_waterlogging_m"),
        "nearest_historical_pocket": hist.get("nearest_record"),
        "rainfall": {"available": bool(wx.get("available")), "rainfall_24h_mm": rainfall_mm,
                     "station_name": wx.get("station_name"), "reason": wx.get("reason")},
        "disclaimer": ("Model-estimated flood susceptibility from limited real data (KMC pockets, "
                       "partial drainage, DEM). Not a guarantee. Outside Kolkata (KMC) there are no "
                       "local flood records, so risk cannot be assessed there."),
    }


# ------------------------------------------------------------- simulation

class SimPoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class SimulatePointsRequest(BaseModel):
    points: List[SimPoint] = Field(..., min_length=1, max_length=500)
    rainfall_mm_hr: float = Field(..., ge=0, le=300)
    duration_min: float = Field(..., ge=0, le=720)


def _drain_hazard(load: Optional[float], distance_m: Optional[float]) -> float:
    """0..0.9 extra hazard from an overloaded nearby drain. Starts at 75% load,
    saturates at 200%, fades out with distance (0 beyond 300 m)."""
    if load is None or distance_m is None:
        return 0.0
    overload = min(max((load - 0.75) / 1.25, 0.0), 1.0)
    proximity = max(0.0, 1.0 - distance_m / 300.0)
    return 0.9 * overload * proximity


@router.post("/api/simulate/points")
async def simulate_points(req: SimulatePointsRequest):
    """Scenario flood risk for many points in ONE call (the Simulation map).

    risk = 1 - (1 - susceptibility * rain_multiplier) * (1 - drain_hazard)
      susceptibility   real_v1 static score (terrain, drainage, water bodies)
      rain_multiplier  from scenario rainfall depth = intensity * duration
      drain_hazard     overload of the nearest sewer in the drainage simulation
    A scenario, not a forecast; live IMD weather is NOT used here."""
    from services.dem_service import get_dem_service
    from services.drainage_sim import load_ratio_for_segment, alert_for
    from services.flood_model import classify_risk, rainfall_multiplier

    pts = [(p.lat, p.lon) for p in req.points]
    terrains = get_dem_service().get_terrain_features(pts)
    spatials = get_spatial_service().get_spatial_features(pts)
    model = get_flood_model()
    depth_mm = req.rainfall_mm_hr * req.duration_min / 60.0

    segments, drains = [], []
    for (lat, lon), t, s in zip(pts, terrains, spatials):
        d = s.get("drain") or {}
        w = s.get("water_body") or {}
        found = d.get("found")
        drains.append(d if found else {})
        segments.append({
            "lat": lat, "lng": lon,
            "elevation": t.get("elevation_m"), "slope": t.get("slope_percent"),
            "distance_to_drain_m": d.get("distance_to_drain_m") if found else None,
            "pipe_diameter_mm": d.get("pipe_diameter_mm") if found else None,
            "drain_capacity_estimated_m3s": d.get("drain_capacity_estimated_m3s") if found else None,
            "distance_to_waterbody_m": w.get("distance_to_waterbody_m") if w.get("found") else None,
        })
    scored = model.score_segments(segments, rainfall_24h_mm=depth_mm, horizon_minutes=0)
    mult = rainfall_multiplier(depth_mm)
    if req.rainfall_mm_hr == 0 or req.duration_min == 0:
        mult = 0.3 * min(depth_mm / 10.0, 1.0)  # no rain in the scenario -> near-zero risk

    results = []
    for (lat, lon), s, d in zip(pts, scored, drains):
        sus = s.get("susceptibility")
        load = load_ratio_for_segment(d.get("segment_id"), req.rainfall_mm_hr, req.duration_min)
        hazard = _drain_hazard(load, d.get("distance_to_drain_m"))
        prob = None if sus is None else round(1 - (1 - min(sus * mult, 1.0)) * (1 - hazard), 4)
        results.append({
            "lat": lat, "lon": lon,
            "susceptibility": sus,
            "flood_probability": prob,
            "risk_level": classify_risk(prob),
            "nearest_drain_m": d.get("distance_to_drain_m"),
            "drain_load_pct": round(load * 100) if load is not None else None,
            "drain_alert": alert_for(load) if d else None,
        })
    return {
        **_real_model_tag(),
        "scenario": {"rainfall_mm_hr": req.rainfall_mm_hr, "duration_min": req.duration_min,
                     "total_rainfall_mm": round(depth_mm, 1)},
        "points": results,
        "disclaimer": "Scenario simulation from assumed rainfall, not a forecast.",
    }


@router.get("/api/simulate/drainage")
async def simulate_drainage(rainfall_mm_hr: float = Query(..., ge=0, le=300),
                            duration_min: float = Query(..., ge=0, le=720),
                            lat: Optional[float] = Query(None, ge=-90, le=90),
                            lon: Optional[float] = Query(None, ge=-180, le=180),
                            radius_m: Optional[float] = Query(None, gt=0, le=20000)):
    """Per-pipe load and GREEN/YELLOW/ORANGE/RED alert for a design storm.
    Summary counts cover pipes within radius_m of (lat, lon) when given."""
    from services.drainage_sim import DRAINAGE_GEOJSON, run_simulation
    if not DRAINAGE_GEOJSON.exists():
        raise HTTPException(status_code=503, detail="drainage network data is not available")
    return run_simulation(rainfall_mm_hr, duration_min, lat, lon, radius_m)


class RadiusRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: float = 30.0
    horizon_minutes: int = 60

@router.post("/api/flood/radius")
async def flood_radius(req: RadiusRequest):
    """Spatial 30km grid sampling for frontend Risk Map."""
    # We create a 5x5 grid around the center to keep the API fast and just pass it to the model.
    # In a full implementation, this would use spatial_service to sample an actual grid and return GeoJSON.
    from services.dem_service import get_dem_service
    dem = get_dem_service()
    spatial = get_spatial_service()
    model = get_flood_model()
    imd = get_imd_service()
    
    # Just generating a small cross pattern for now
    offsets = [
        (0, 0), (0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05),
        (0.1, 0.1), (-0.1, -0.1), (0.1, -0.1), (-0.1, 0.1)
    ]
    points = [(req.latitude + lat_off, req.longitude + lon_off) for lat_off, lon_off in offsets]
    
    terrains = dem.get_terrain_features(points)
    spatials = spatial.get_spatial_features(points)
    wx = await imd.get_weather()
    rainfall_mm = wx.get("rainfall_24h")
    
    segments = []
    for i, pt in enumerate(points):
        drain = spatials[i].get("drain") or {}
        water = spatials[i].get("water_body") or {}
        segments.append({
            "lat": pt[0], "lng": pt[1],
            "elevation": terrains[i].get("elevation_m"),
            "slope": terrains[i].get("slope_percent"),
            "distance_to_drain_m": drain.get("distance_to_drain_m") if drain.get("found") else None,
            "pipe_diameter_mm": drain.get("pipe_diameter_mm") if drain.get("found") else None,
            "drain_capacity_estimated_m3s": drain.get("drain_capacity_estimated_m3s") if drain.get("found") else None,
            "distance_to_waterbody_m": water.get("distance_to_waterbody_m") if water.get("found") else None,
        })
        
    scored = model.score_segments(segments, rainfall_24h_mm=rainfall_mm, horizon_minutes=req.horizon_minutes)
    
    features = []
    for i, s in enumerate(scored):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [points[i][1], points[i][0]]},
            "properties": {
                "flood_probability": s.get("flood_probability"),
                "risk_level": s.get("risk_level")
            }
        })
        
    return {
        "type": "FeatureCollection",
        "features": features
    }

# ------------------------------------------------------------------ routing

class LatLon(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class RouteRequest(BaseModel):
    origin: LatLon
    destination: LatLon
    profile: str = "driving"
    alternatives: bool = True


class SafeRouteRequest(BaseModel):
    # Either provide coordinates, or text to geocode server-side.
    origin: Optional[LatLon] = None
    destination: Optional[LatLon] = None
    origin_query: Optional[str] = None
    destination_query: Optional[str] = None
    profile: str = "driving"
    include_segments: bool = True
    horizon_minutes: int = 0


async def _resolve_point(pt: Optional[LatLon], query: Optional[str], label: str):
    if pt is not None:
        return (pt.lat, pt.lon), None
    if query:
        cands = await get_mapbox_service().geocode(query, limit=1)
        if not cands:
            raise HTTPException(status_code=404, detail=f"could not geocode {label}: {query!r}")
        return (cands[0].lat, cands[0].lon), cands[0].to_dict()
    raise HTTPException(status_code=422, detail=f"{label} requires coordinates or a query string")


@router.post("/api/route")
async def route(req: RouteRequest):
    svc = get_mapbox_service()
    try:
        routes = await svc.get_routes((req.origin.lat, req.origin.lon),
                                      (req.destination.lat, req.destination.lon),
                                      profile=req.profile, alternatives=req.alternatives)
    except NoRouteError as exc:
        raise HTTPException(status_code=404, detail=exc.to_dict())
    except MapboxError as exc:
        raise _mapbox_http(exc)
    return {"routes": [r.to_dict() for r in routes]}


@router.post("/api/route/flood-risk")
async def route_flood_risk(req: RouteRequest):
    """One route's per-segment flood-risk features + real_v1 scores."""
    svc = get_mapbox_service()
    try:
        routes = await svc.get_routes((req.origin.lat, req.origin.lon),
                                      (req.destination.lat, req.destination.lon),
                                      profile=req.profile, alternatives=False)
    except NoRouteError as exc:
        raise HTTPException(status_code=404, detail=exc.to_dict())
    except MapboxError as exc:
        raise _mapbox_http(exc)
    scored = await _score_route(routes[0])
    return {**_real_model_tag(), **scored}


@router.post("/api/safe-route")
async def safe_route(req: SafeRouteRequest):
    """Geocode -> alternatives -> sample -> risk engine -> real_v1 -> recommend.

    Never claims a route is "safe"; recommends the lower-risk option and warns
    when all routes are risky.
    """
    (o_lat, o_lon), o_info = await _resolve_point(req.origin, req.origin_query, "origin")
    (d_lat, d_lon), d_info = await _resolve_point(req.destination, req.destination_query, "destination")

    mb = get_mapbox_service()
    try:
        routes = await mb.get_routes((o_lat, o_lon), (d_lat, d_lon), profile=req.profile, alternatives=True)
    except NoRouteError as exc:
        raise HTTPException(status_code=404, detail=exc.to_dict())
    except MapboxNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=exc.to_dict())
    except MapboxError as exc:
        raise _mapbox_http(exc)

    scored_routes = []
    for r in routes:
        s = await _score_route(r, include_segments=req.include_segments, horizon_minutes=req.horizon_minutes)
        scored_routes.append(s)

    decision = pick_recommendation(scored_routes)
    model = get_flood_model()
    return {
        **_real_model_tag(),
        "model_status": model.status,
        "origin": {"lat": o_lat, "lon": o_lon, "resolved": o_info},
        "destination": {"lat": d_lat, "lon": d_lon, "resolved": d_info},
        "routes": scored_routes,
        **decision,
        "disclaimer": ("Flood risk is a relative, model-estimated susceptibility from limited real data "
                       "(KMC 2017 pockets, partial drainage). No route is guaranteed flood-free."),
    }


async def _score_route(route_obj, include_segments: bool = True, horizon_minutes: int = 0):
    """Assemble features for one Mapbox route and score with real_v1."""
    rr = get_route_risk_service()
    feats = await rr.assemble_route_features(route_obj.geometry)
    model = get_flood_model()
    seg_dicts = [s.to_dict() for s in feats.segments]
    rainfall_mm = feats.rainfall.get("rainfall_24h_mm")
    scored_segments = model.score_segments(seg_dicts, rainfall_24h_mm=rainfall_mm, horizon_minutes=horizon_minutes)
    score = route_score([s.get("flood_probability") for s in scored_segments])
    result = {
        "distance_m": round(route_obj.distance_m, 1),
        "distance_km": round(route_obj.distance_m / 1000, 2),
        "duration_min": round(route_obj.duration_s / 60, 1),
        "geometry": route_obj.geometry,
        "route_score": score,
        "risk_level": risk_level_for_route(score),
        "coverage": feats.coverage,
        "rainfall": feats.rainfall,
        "n_segments": len(scored_segments),
    }
    if include_segments:
        result["segments"] = scored_segments
    return result


# --------------------------------------------------------------- data status

@router.get("/api/data/status")
def data_status():
    """Per-source status with an explicit rolled-up `state` verdict (section 30).

    Connectivity for IMD/Mapbox uses each service's CACHED last-success signal;
    this endpoint does not make live round-trips. Use ?ping=1 to actively probe.
    """
    from services.dem_service import get_dem_service
    spatial = get_spatial_service()
    model = get_flood_model()
    imd = get_imd_service()
    mapbox = get_mapbox_service()
    rep = spatial.status_report()
    dem = get_dem_service().status_report()

    imd_health = imd.health()
    mb_health = mapbox.health()

    sources = {
        "dem": {"state": "LOADED" if dem["dem_status"] == "LOADED" else
                ("ERROR" if dem["dem_status"] == "ERROR" else "MISSING"), **dem},
        "drainage": _layer(rep["drainage_network"]),
        "water_bodies": _layer(rep["water_bodies"]),
        "pumping_stations": _layer(rep["pumping_stations"]),
        "historical_waterlogging": {"state": _hist_state(rep["historical_waterlogging"]),
                                    **rep["historical_waterlogging"]},
        "mapbox": {"state": _live_state(mapbox.configured, mb_health), **mb_health},
        "imd": {"state": _live_state(imd_health["configured"], imd_health), **imd_health},
    }
    return {
        "sources": sources,
        "model": {
            "active_version": model.metadata.get("model_version", "real_v1"),
            "state": "REAL" if model.status == "LOADED" else ("ERROR" if model.status == "ERROR" else "MISSING"),
            "status": model.status,
            "real_v1": {**real_descriptor(), "status": model.status},
            "synthetic_v1": {**synthetic_descriptor(), "status": "LOADED",
                             "note": "legacy physics-simulation model, served by /api/predict"},
        },
    }


@router.get("/api/data/status/ping")
async def data_status_ping():
    """Actively probe Mapbox and IMD connectivity (makes real calls)."""
    mapbox = get_mapbox_service()
    imd = get_imd_service()
    mb = await mapbox.ping()
    wx = await imd.get_weather()
    return {"mapbox": mb,
            "imd": {"connected": bool(wx.get("available")), "reason": wx.get("reason"),
                    "degraded_cause": wx.get("degraded_cause")}}


def _layer(rep: dict) -> dict:
    status = rep.get("status")
    state = "LOADED" if status == "LOADED" else ("ERROR" if status == "ERROR" else "MISSING")
    return {"state": state, "status": status, "features": rep.get("features")}


def _hist_state(rep: dict) -> str:
    status = rep.get("status")
    if status == "LOADED":
        return "LOADED"
    return "ERROR" if status == "ERROR" else "MISSING"


def _live_state(configured: bool, health: dict) -> str:
    """LIVE if a recent successful call is on record; else CONNECTED-unknown /
    ERROR / NOT_CONFIGURED. No round-trip is performed here."""
    if not configured:
        return "NOT_CONFIGURED"
    if health.get("last_success_ts"):
        return "LIVE"
    if health.get("last_error"):
        return "ERROR"
    return "UNKNOWN"  # configured but not yet exercised this run

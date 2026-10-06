"""
IMD (India Meteorological Department) weather service.

Auth (per the IMD API Portal User Guide):
  POST https://api.imd.gov.in/api/oauth/token.php   {email, password} -> {access_token, token_type, expires_in}
  Data calls send BOTH headers:
      X-API-KEY: <api key, bound to a registered static public IP>
      Authorization: Bearer <access_token>

Two hard realities, handled here rather than hidden:
  1. The API key is tied to a registered server IP. From any other machine
     (local dev, and possibly Render's shared egress) data calls return HTTP 403
     "IP address X not authorized". get_weather() surfaces this as a specific,
     non-crashing degraded result, not a generic failure.
  2. The IMD token is a NON-STANDARD 2-part JWT: segment 0 is JSON containing
     {"uid":..,"exp":..} and segment 1 is a binary signature. We read exp from
     segment 0 to cache the token until just before expiry, and fall back to the
     response's expires_in. Only ONE token is fetched per hour, not per request.

Endpoint field names (IMD API Reference) are mapped into a stable internal
schema by normalize_current_weather(). IMD only reports Past_24_hrs_Rainfall,
so rainfall_30m / rainfall_1h / rainfall_3h are None (never fabricated): IMD
does not publish sub-daily rainfall through this API. Short-horizon signal
comes from the district/station nowcast category instead.
"""

import asyncio
import base64
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

TOKEN_URL = "https://api.imd.gov.in/api/oauth/token.php"
BASE = "https://api.imd.gov.in/api/v1"
CURRENT_WX_URL = f"{BASE}/current_wx"
CITY_FORECAST_URL = f"{BASE}/cityforecast"
DISTRICT_NOWCAST_URL = f"{BASE}/districtnowcast"

DEFAULT_TIMEOUT_S = 10.0
TOKEN_REFRESH_SKEW_S = 120  # refresh this long before the token's exp
DEFAULT_TOKEN_LIFETIME_S = 3600

# IMD current_wx observations change roughly hourly, so a fresh result is
# reused for 30 min. After a failure we back off instead of retrying every
# request: 429 -> 5 min doubling to 1 h (or IMD's Retry-After); IP/auth
# rejection -> 15 min (retrying can't fix it until the IP is re-registered).
WX_CACHE_TTL_S = 1800
WX_FAILURE_CACHE_TTL_S = 120
# Hard local budget, checked BEFORE any network call, so this process can
# never be the one that trips IMD's limit (whatever the caller does).
IMD_MIN_CALL_INTERVAL_S = 60      # at most one IMD request (token or data) per minute
IMD_MAX_CALLS_PER_HOUR = 20
RATE_LIMIT_BACKOFF_START_S = 300
RATE_LIMIT_BACKOFF_MAX_S = 3600
AUTH_FAILURE_BACKOFF_S = 900
STALE_MAX_AGE_S = 6 * 3600  # serve last good reading (marked stale) up to 6 h old
OFFICIAL_RAIN_TTL_S = 3 * 3600  # official 24h rainfall updates once a day (08:30 IST)

# Token + last good reading survive `uvicorn --reload` restarts, so a restart
# doesn't spend a fresh token call and data call against IMD's quota.
CACHE_FILE = Path(__file__).resolve().parent.parent / ".cache" / "imd_cache.json"

# Kolkata defaults (Alipore observatory). Overridable per call.
KOLKATA_STATION_ID = "42809"
KOLKATA_DISTRICT_ID = "343"  # KOLKATA in district endpoints


class IMDError(Exception):
    """Base for IMD failures. Callers should still degrade, not crash."""


class IMDAuthError(IMDError):
    pass


class IMDIPNotAuthorizedError(IMDError):
    """API key's registered IP doesn't match this server's egress IP."""


class IMDNotConfiguredError(IMDError):
    pass


class IMDRateLimitError(IMDError):
    def __init__(self, msg: str, retry_after_s: Optional[float] = None):
        super().__init__(msg)
        self.retry_after_s = retry_after_s


class IMDLocalBudgetError(IMDRateLimitError):
    """We refused the call ourselves; IMD was never contacted."""


@dataclass
class _Token:
    value: str
    expires_at: float  # epoch seconds

    @property
    def valid(self) -> bool:
        return bool(self.value) and time.time() < (self.expires_at - TOKEN_REFRESH_SKEW_S)


def _decode_exp(token: str) -> Optional[float]:
    """Read exp from the IMD token's first segment (JSON: {'uid':..,'exp':..})."""
    try:
        seg = token.split(".")[0]
        data = json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))
        exp = data.get("exp")
        return float(exp) if exp is not None else None
    except Exception:
        return None


class IMDService:
    def __init__(self, api_key: Optional[str], email: Optional[str], password: Optional[str],
                 timeout_s: float = DEFAULT_TIMEOUT_S, client: Optional[httpx.AsyncClient] = None,
                 debug_log_raw: bool = False, cache_file: Optional[Path] = None):
        self._api_key = api_key
        self._email = email
        self._password = password
        self._timeout = httpx.Timeout(timeout_s, connect=min(5.0, timeout_s))
        self._client = client
        self._owns_client = client is None
        self._debug_log_raw = debug_log_raw

        self._token: Optional[_Token] = None
        self._lock = threading.Lock()
        self._auth_count = 0  # how many times we actually hit the token endpoint
        self._call_times: list = []  # timestamps of real IMD requests (last hour)
        self._enforce_budget = cache_file is not None  # production singleton only; tests use mocks
        self.last_success_ts: Optional[float] = None
        self.last_error: Optional[str] = None

        # Weather response cache. IMD observations update ~hourly, but every
        # /api/location-risk call asks for weather, so without this a map grid of
        # ~90 points fires ~90 IMD calls at once and trips IMD's 429 rate limit.
        self._wx_cache: Dict[str, tuple] = {}      # station -> (expires_at, result)
        self._wx_inflight: Dict[str, Any] = {}     # station -> asyncio.Task (coalesces concurrent misses)
        self._last_good: Dict[str, tuple] = {}     # station -> (fetched_at, result) for stale fallback
        self._rate_backoff_s = 0.0                 # current 429 backoff, doubles per consecutive 429
        self._blocked_until: Dict[str, tuple] = {}  # station -> (retry_at, degraded result)
        self._official_rain: Dict[str, tuple] = {}  # station -> (expires_at, mm, date)
        self._cache_file = Path(cache_file) if cache_file else None
        self._load_cache()

    @property
    def configured(self) -> bool:
        return all(bool(x and x.strip()) for x in (self._api_key, self._email, self._password))

    async def _get_client(self) -> httpx.AsyncClient:
        # Rebind an owned client if the event loop changed or closed (a
        # process-wide singleton reused across requests). Injected clients as-is.
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

    # --------------------------------------------------------------- auth

    async def get_jwt_token(self, force_refresh: bool = False) -> str:
        """Return a cached JWT, fetching a new one only when missing/expired.

        Thread-safe check of the cached token; the network fetch happens outside
        the lock so it doesn't block other coroutines.
        """
        if not self.configured:
            raise IMDNotConfiguredError("IMD credentials are not configured")
        with self._lock:
            if not force_refresh and self._token is not None and self._token.valid:
                return self._token.value
        token = await self._fetch_token()
        with self._lock:
            self._token = token
        return token.value

    async def _fetch_token(self) -> _Token:
        # Token and data calls share the budget, but a token fetch right before
        # its data call must not block that call: release the interval for it.
        self._spend_call()
        if self._call_times:
            self._call_times[-1] -= IMD_MIN_CALL_INTERVAL_S
        client = await self._get_client()
        try:
            resp = await client.post(TOKEN_URL, json={"email": self._email, "password": self._password},
                                     headers={"Content-Type": "application/json"})
        except httpx.HTTPError as exc:
            self.last_error = f"token request failed: {exc}"
            raise IMDError(self.last_error) from exc
        if resp.status_code in (401, 403):
            self.last_error = "IMD rejected credentials"
            raise IMDAuthError(self.last_error)
        if resp.status_code == 429:
            self.last_error = "token endpoint rate limited"
            raise IMDRateLimitError("IMD rate limit exceeded (token endpoint)")
        if resp.status_code >= 400:
            self.last_error = f"token endpoint HTTP {resp.status_code}"
            raise IMDError(self.last_error)
        try:
            body = resp.json()
        except ValueError:
            self.last_error = "token response was not JSON"
            raise IMDError(self.last_error)
        token = body.get("access_token")
        if not token:
            self.last_error = "token response had no access_token"
            raise IMDAuthError(self.last_error)
        exp = _decode_exp(token)
        if exp is None:
            lifetime = float(body.get("expires_in") or DEFAULT_TOKEN_LIFETIME_S)
            exp = time.time() + lifetime
        self._auth_count += 1
        logger.info("IMD token acquired (auth #%d), expires in %.0fs", self._auth_count, exp - time.time())
        new = _Token(value=token, expires_at=exp)
        self._token = new
        self._save_cache()
        return new

    # ------------------------------------------------------- persistence

    def _load_cache(self) -> None:
        if not self._cache_file or not self._cache_file.exists():
            return
        try:
            data = json.loads(self._cache_file.read_text())
        except (OSError, ValueError):
            return
        # Only trust a cached token issued for these credentials.
        if data.get("email") == self._email:
            tok = data.get("token") or {}
            if tok.get("value") and tok.get("expires_at"):
                self._token = _Token(value=tok["value"], expires_at=float(tok["expires_at"]))
            for station, (ts, result) in (data.get("last_good") or {}).items():
                self._last_good[station] = (float(ts), result)
            self._rate_backoff_s = float(data.get("rate_backoff_s") or 0)
            now = time.time()
            self._call_times = [float(t) for t in (data.get("call_times") or []) if now - float(t) < 3600]
            # Honour a backoff that was still running when the process restarted.
            for station, (until, result) in (data.get("blocked_until") or {}).items():
                if float(until) > time.time():
                    self._blocked_until[station] = (float(until), result)
                    self._wx_cache[station] = (float(until), result)

    def _save_cache(self) -> None:
        if not self._cache_file:
            return
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "email": self._email,
                "token": ({"value": self._token.value, "expires_at": self._token.expires_at}
                          if self._token else None),
                "last_good": {s: [ts, r] for s, (ts, r) in self._last_good.items()},
                "blocked_until": {s: [u, r] for s, (u, r) in self._blocked_until.items()},
                "rate_backoff_s": self._rate_backoff_s,
                "call_times": self._call_times[-IMD_MAX_CALLS_PER_HOUR:],
            }
            tmp = self._cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self._cache_file)
        except OSError as exc:
            logger.warning("could not write IMD cache: %s", exc)

    # ----------------------------------------------------------- requests

    def _spend_call(self) -> None:
        """Record one real IMD request, or refuse it locally if over budget."""
        if not self._enforce_budget:
            return
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 3600]
        if self._call_times and now - self._call_times[-1] < IMD_MIN_CALL_INTERVAL_S:
            wait = IMD_MIN_CALL_INTERVAL_S - (now - self._call_times[-1])
            raise IMDLocalBudgetError("local IMD budget: one call per minute", wait)
        if len(self._call_times) >= IMD_MAX_CALLS_PER_HOUR:
            wait = 3600 - (now - self._call_times[0])
            raise IMDLocalBudgetError(f"local IMD budget: {IMD_MAX_CALLS_PER_HOUR} calls/hour", wait)
        self._call_times.append(now)
        self._save_cache()

    async def _authed_get(self, url: str, params: Optional[Dict] = None) -> Any:
        token = await self.get_jwt_token()
        self._spend_call()
        client = await self._get_client()
        headers = {"X-API-KEY": self._api_key, "Authorization": f"Bearer {token}"}
        try:
            resp = await client.get(url, params=params or {}, headers=headers)
        except httpx.HTTPError as exc:
            self.last_error = f"request failed: {exc}"
            raise IMDError(self.last_error) from exc

        if resp.status_code == 401:
            # Token may have expired early; refresh once and retry.
            token = await self.get_jwt_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            try:
                resp = await client.get(url, params=params or {}, headers=headers)
            except httpx.HTTPError as exc:
                raise IMDError(f"request failed after token refresh: {exc}") from exc

        if resp.status_code == 403:
            msg = self._extract_message(resp) or "forbidden"
            if "not authorized" in msg.lower() or "ip" in msg.lower():
                self.last_error = f"IP not authorized: {msg}"
                raise IMDIPNotAuthorizedError(msg)
            self.last_error = f"forbidden: {msg}"
            raise IMDAuthError(msg)
        if resp.status_code == 401:
            self.last_error = "unauthorized after refresh"
            raise IMDAuthError("unauthorized (token rejected after refresh)")
        if resp.status_code == 429:
            self.last_error = "rate limited"
            retry_after = None
            try:
                retry_after = float(resp.headers.get("Retry-After"))
            except (TypeError, ValueError):
                pass
            raise IMDRateLimitError("IMD rate limit exceeded", retry_after)
        if resp.status_code >= 400:
            self.last_error = f"HTTP {resp.status_code}"
            raise IMDError(f"IMD HTTP {resp.status_code}")

        if self._debug_log_raw:
            logger.debug("IMD raw %s: %s", url, resp.text[:2000])
        try:
            data = resp.json()
        except ValueError as exc:
            raise IMDError("IMD response was not JSON") from exc
        self.last_success_ts = time.time()
        self.last_error = None
        return data

    @staticmethod
    def _extract_message(resp: httpx.Response) -> Optional[str]:
        try:
            body = resp.json()
            if isinstance(body, dict):
                return body.get("error") or body.get("message")
        except ValueError:
            return resp.text[:200] if resp.text else None
        return None

    # ------------------------------------------------------------ public

    async def get_current_weather(self, station_id: str = KOLKATA_STATION_ID) -> Dict:
        data = await self._authed_get(CURRENT_WX_URL, {"id": station_id})
        record = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
        return normalize_current_weather(record)

    async def get_rainfall(self, station_id: str = KOLKATA_STATION_ID) -> Dict:
        """Rainfall subset of current weather. Sub-daily buckets stay None (IMD has none)."""
        wx = await self.get_current_weather(station_id)
        return {k: wx.get(k) for k in ("rainfall_30m", "rainfall_1h", "rainfall_3h",
                                       "rainfall_24h", "timestamp", "source", "station")}

    async def get_city_forecast(self, station_id: str = KOLKATA_STATION_ID) -> Dict:
        data = await self._authed_get(CITY_FORECAST_URL, {"id": station_id})
        record = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
        return normalize_city_forecast(record)

    # ------------------------------------------------------- resilient API

    async def get_weather(self, station_id: str = KOLKATA_STATION_ID) -> Dict:
        """Cached + coalesced wrapper around _get_weather_uncached()."""
        cached = self._wx_cache.get(station_id)
        if cached and time.time() < cached[0]:
            return dict(cached[1])
        task = self._wx_inflight.get(station_id)
        if task is None:
            task = asyncio.ensure_future(self._get_weather_uncached(station_id))
            self._wx_inflight[station_id] = task
            try:
                result = await task
            finally:
                self._wx_inflight.pop(station_id, None)
            result = self._after_fetch(station_id, result)
            return dict(result)
        return dict(await task)

    def _after_fetch(self, station_id: str, result: Dict) -> Dict:
        """Pick the cache lifetime for this result and fall back to the last
        good reading (marked stale) when IMD failed."""
        now = time.time()
        cause = result.get("degraded_cause")
        ok = result.get("available") and cause is None

        if ok:
            self._rate_backoff_s = 0.0
            self._last_good[station_id] = (now, result)
            self._save_cache()
            ttl = WX_CACHE_TTL_S
        elif cause == "rate_limited":
            self._rate_backoff_s = (min(self._rate_backoff_s * 2, RATE_LIMIT_BACKOFF_MAX_S)
                                    if self._rate_backoff_s else RATE_LIMIT_BACKOFF_START_S)
            ttl = max(self._rate_backoff_s, result.pop("_retry_after_s", None) or 0)
            logger.warning("IMD rate limited; next attempt in %.0fs", ttl)
        elif cause == "local_budget":
            ttl = max(result.get("_retry_after_s") or 0, 5)  # no IMD contact -> no backoff growth
        elif cause in ("auth_or_ip_error", "ip_not_authorized", "auth", "not_configured"):
            ttl = AUTH_FAILURE_BACKOFF_S
        else:
            ttl = WX_FAILURE_CACHE_TTL_S
        result.pop("_retry_after_s", None)

        if not ok:
            good = self._last_good.get(station_id)
            if good and now - good[0] <= STALE_MAX_AGE_S:
                result = {**good[1], "available": True, "stale": True,
                          "fetched_at": good[0], "age_minutes": round((now - good[0]) / 60),
                          "reason": f"Showing last IMD reading ({round((now - good[0]) / 60)} min old): "
                                    f"{result.get('reason')}",
                          "degraded_cause": cause}
            result["next_retry_in_s"] = round(ttl)

        self._wx_cache[station_id] = (now + ttl, result)
        if not ok:
            self._blocked_until[station_id] = (now + ttl, result)
            self._save_cache()
        else:
            self._blocked_until.pop(station_id, None)
        return result

    async def _get_weather_uncached(self, station_id: str = KOLKATA_STATION_ID) -> Dict:
        """Current weather that NEVER raises. On failure returns available=False
        with a specific reason, so the route engine can degrade cleanly."""
        base = {"available": False, "source": "IMD", "station": station_id,
                "rainfall_30m": None, "rainfall_1h": None, "rainfall_3h": None,
                "rainfall_24h": None, "temperature": None, "humidity": None, "timestamp": None}
        try:
            wx = await self.get_current_weather(station_id)
            wx["available"] = True
            await self._apply_official_rainfall(station_id, wx)
            return wx
        # Never fabricate readings: fake rainfall would silently skew flood risk.
        except IMDNotConfiguredError as exc:
            return {**base, "reason": f"IMD not configured: {exc}", "degraded_cause": "not_configured"}
        except IMDIPNotAuthorizedError as exc:
            return {**base, "reason": f"IMD unavailable: IP not authorized ({exc})",
                    "degraded_cause": "ip_not_authorized"}
        except IMDAuthError as exc:
            return {**base, "reason": f"IMD unavailable: authentication failed ({exc})",
                    "degraded_cause": "auth"}
        except IMDLocalBudgetError as exc:
            return {**base, "reason": f"IMD call deferred: {exc}", "degraded_cause": "local_budget",
                    "_retry_after_s": exc.retry_after_s}
        except IMDRateLimitError as exc:
            return {**base, "reason": f"IMD unavailable: {exc}", "degraded_cause": "rate_limited",
                    "_retry_after_s": exc.retry_after_s}
        except IMDError as exc:
            return {**base, "reason": f"IMD unavailable: {exc}", "degraded_cause": "unavailable"}
        except Exception as exc:  # never let weather crash a route request
            logger.exception("unexpected IMD error")
            return {**base, "reason": f"unexpected error: {exc}", "degraded_cause": "unexpected"}

    async def _apply_official_rainfall(self, station_id: str, wx: Dict) -> None:
        """Combine current_wx's rolling "Last 24 hrs Rainfall" with the official
        Past_24_hrs_Rainfall from cityforecast (08:30 IST -> 08:30 IST).

        The official figure changes once a day, so it's cached for
        OFFICIAL_RAIN_TTL_S. Never raises: if cityforecast fails, the current_wx
        value is kept and labelled as such.
        """
        now = time.time()
        cached = self._official_rain.get(station_id)
        if not cached or now >= cached[0]:
            try:
                # Paired with the current_wx call just made: release the
                # per-minute interval (hourly cap still applies), as for tokens.
                if self._call_times:
                    self._call_times[-1] -= IMD_MIN_CALL_INTERVAL_S
                fc = await self.get_city_forecast(station_id)
                cached = (now + OFFICIAL_RAIN_TTL_S, fc.get("past_24h_rainfall"), fc.get("date"))
                self._official_rain[station_id] = cached
            except Exception as exc:  # keep current_wx value; don't fail the weather call
                logger.warning("IMD cityforecast rainfall unavailable: %s", exc)  # reuse expired value if any
        # The two IMD figures cover different windows: the official one is the
        # fixed 08:30->08:30 IST day, current_wx is a rolling 24h that includes
        # rain since 08:30 today. Either can miss rain the other caught (80.8 vs
        # 0 one day, NIL vs 24 mm the next), so use the larger for flood risk
        # and expose both.
        official = cached[1] if cached else None
        rolling = wx.get("rainfall_24h")
        wx["rainfall_24h_official"] = official
        wx["rainfall_24h_rolling"] = rolling
        period = (f"24h ending 08:30 IST {cached[2]}" if cached and cached[2] else "24h ending 08:30 IST")
        if official is not None and (rolling is None or official >= rolling):
            wx["rainfall_24h"] = official
            wx["rainfall_24h_source"] = "IMD observatory (cityforecast Past_24_hrs_Rainfall)"
            wx["rainfall_24h_period"] = period
        else:
            wx["rainfall_24h_source"] = "IMD current_wx (Last 24 hrs Rainfall, rolling)"
            wx["rainfall_24h_period"] = "rolling last 24h" if rolling is not None else None

    def health(self) -> Dict:
        return {
            "configured": self.configured,
            "token_cached": self._token is not None and self._token.valid,
            "auth_count": self._auth_count,
            "last_success_ts": self.last_success_ts,
            "last_error": self.last_error,
        }


# ------------------------------------------------------------ normalization

def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def _to_rain_mm(v: Any) -> Optional[float]:
    """IMD rainfall: numbers, or the bulletin codes "NIL" (no rain) / "TRACE"
    (< 0.1 mm). Both are real observations of ~0 mm, not missing data."""
    if isinstance(v, str) and v.strip().upper() in ("NIL", "TRACE", "TR"):
        return 0.0
    return _to_float(v)


def normalize_current_weather(rec: Dict) -> Dict:
    """Map IMD current_wx fields to the internal schema (missing -> None)."""
    return {
        "source": "IMD",
        "station": rec.get("Station Id") or rec.get("Station_Id") or rec.get("Station"),
        "station_name": rec.get("Station"),
        "timestamp": _combine_obs_time(rec.get("Date of Observation"),
                                       _obs_time(rec.get("Time of Observation") or rec.get("Time"))),
        "temperature": _to_float(rec.get("Temperature")),
        "feels_like": _to_float(rec.get("Feel Like")),
        "humidity": _to_float(rec.get("Humidity")),
        "mslp_hpa": _to_float(rec.get("M.S.L.P") or rec.get("Mean Sea Level Pressure")),
        "wind_speed_kmph": _to_float(rec.get("Wind Speed") or rec.get("Wind Speed KMPH")),
        "weather_message": rec.get("WEATHER_MESSAGE"),
        "wind_direction": rec.get("Wind Direction"),
        "weather_code": rec.get("Weather Code"),
        "nebulosity": _to_float(rec.get("Nebulosity")),
        "rainfall_24h": _to_rain_mm(rec.get("Last 24 hrs Rainfall")),
        # IMD current_wx does not report sub-daily rainfall. Never fabricated.
        "rainfall_30m": None, "rainfall_1h": None, "rainfall_3h": None,
        "rainfall_note": "IMD reports only last-24h rainfall; 30m/1h/3h are unavailable, not zero.",
    }


def normalize_city_forecast(rec: Dict) -> Dict:
    days = []
    days.append({"day": 1, "forecast": rec.get("Todays_Forecast"),
                 "max_temp": _to_float(rec.get("Todays_Forecast_Max_Temp")),
                 "min_temp": _to_float(rec.get("Todays_Forecast_Min_temp"))})
    for d in range(2, 8):
        days.append({"day": d, "forecast": rec.get(f"Day_{d}_Forecast"),
                     "max_temp": _to_float(rec.get(f"Day_{d}_Max_Temp")),
                     "min_temp": _to_float(rec.get(f"Day_{d}_Min_temp"))})
    return {
        "source": "IMD", "station": rec.get("Station_Code"), "station_name": rec.get("Station_Name"),
        "date": rec.get("Date"), "past_24h_rainfall": _to_rain_mm(rec.get("Past_24_hrs_Rainfall")),
        "humidity_0830": _to_float(rec.get("Relative_Humidity_at_0830")),
        "humidity_1730": _to_float(rec.get("Relative_Humidity_at_1730")),
        "forecast_days": days,
    }


def _obs_time(t: Optional[str]) -> Optional[str]:
    """IMD sometimes sends just the hour ("12"); normalise to HH:MM:SS."""
    if t is None:
        return None
    t = str(t).strip()
    if t.isdigit() and len(t) <= 2:
        return f"{int(t):02d}:00:00"
    return t or None


def _combine_obs_time(date: Optional[str], time_utc: Optional[str]) -> Optional[str]:
    if not date:
        return None
    return f"{date}T{time_utc}Z" if time_utc else date


_service: Optional[IMDService] = None


def get_imd_service() -> IMDService:
    global _service
    if _service is None:
        def val(s):
            return s.get_secret_value() if s else None
        _service = IMDService(
            api_key=val(settings.IMD_API_KEY),
            email=val(settings.IMD_EMAIL),
            password=val(settings.IMD_PASSWORD),
            debug_log_raw=settings.DEBUG_LOG_IMD_RAW,
            cache_file=CACHE_FILE,
        )
    return _service

import asyncio
import base64
import json
import time

import httpx
import pytest

from config import settings
from services.imd_service import (
    IMDAuthError, IMDIPNotAuthorizedError, IMDNotConfiguredError, IMDService,
    normalize_current_weather, normalize_city_forecast, _decode_exp,
)


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def imd_token(exp_epoch):
    """Build an IMD-style 2-part token: seg0 = JSON{uid,exp}, seg1 = binary sig."""
    seg0 = base64.urlsafe_b64encode(json.dumps({"uid": 5148, "exp": int(exp_epoch)}).encode()).rstrip(b"=").decode()
    seg1 = base64.urlsafe_b64encode(b"\xd5\x01\x02sig").rstrip(b"=").decode()
    return f"{seg0}.{seg1}"


def make_service(handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=10.0)
    return IMDService(api_key="key", email="e@x.gov.in", password="pw", client=client, **kw)


CURRENT_WX = {
    "Station Id": "42809", "Station": "KOLKATA (ALIPORE)", "Date of Observation": "2024-06-06",
    "Time of Observation": "07:00:00", "Temperature": "31.5", "Humidity": "89",
    "M.S.L.P": "1001.0", "Wind Speed": "12", "Wind Direction": "180", "Weather Code": "61",
    "Nebulosity": "6", "Last 24 hrs Rainfall": "23.4",
}


# ---------------------------------------------------------------- token exp

def test_decode_exp_from_imd_token():
    exp = time.time() + 3600
    assert _decode_exp(imd_token(exp)) == pytest.approx(int(exp), abs=1)


def test_decode_exp_bad_token_is_none():
    assert _decode_exp("not-a-token") is None
    assert _decode_exp("") is None


# ---------------------------------------------------------------- caching

def test_token_fetched_once_and_cached():
    calls = {"token": 0, "data": 0}

    def handler(request):
        if request.url.path.endswith("token.php"):
            calls["token"] += 1
            return httpx.Response(200, json={"access_token": imd_token(time.time() + 3600),
                                             "token_type": "Bearer", "expires_in": 3600})
        calls["data"] += 1
        return httpx.Response(200, json=[CURRENT_WX])
    svc = make_service(handler)

    async def go():
        await svc.get_current_weather()
        await svc.get_current_weather()
        await svc.get_current_weather()
    run(go())
    assert calls == {"token": 1, "data": 3}  # authenticated once, three data calls
    assert svc.health()["auth_count"] == 1


def test_expired_token_triggers_refresh():
    calls = {"token": 0}
    # First token already past its refresh skew; second is fresh.
    tokens = [imd_token(time.time() + 10), imd_token(time.time() + 3600)]

    def handler(request):
        if request.url.path.endswith("token.php"):
            t = tokens[min(calls["token"], len(tokens) - 1)]
            calls["token"] += 1
            return httpx.Response(200, json={"access_token": t, "token_type": "Bearer"})
        return httpx.Response(200, json=[CURRENT_WX])
    svc = make_service(handler)

    async def go():
        await svc.get_current_weather()  # fetch token 1 (expires within skew)
        await svc.get_current_weather()  # token 1 not valid -> fetch token 2
    run(go())
    assert calls["token"] == 2


def test_valid_token_not_refetched_within_expiry():
    calls = {"token": 0}

    def handler(request):
        if request.url.path.endswith("token.php"):
            calls["token"] += 1
            return httpx.Response(200, json={"access_token": imd_token(time.time() + 3600)})
        return httpx.Response(200, json=[CURRENT_WX])
    svc = make_service(handler)
    run(svc.get_current_weather())
    run(svc.get_rainfall())
    run(svc.get_current_weather())
    assert calls["token"] == 1


# --------------------------------------------------------------- happy path

def test_current_weather_normalized():
    def handler(request):
        if request.url.path.endswith("token.php"):
            return httpx.Response(200, json={"access_token": imd_token(time.time() + 3600)})
        assert request.headers["X-API-KEY"] == "key"
        assert request.headers["Authorization"].startswith("Bearer ")
        return httpx.Response(200, json=[CURRENT_WX])
    wx = run(make_service(handler).get_current_weather())
    assert wx["temperature"] == 31.5 and wx["humidity"] == 89.0
    assert wx["rainfall_24h"] == 23.4
    assert wx["rainfall_30m"] is None and wx["rainfall_1h"] is None and wx["rainfall_3h"] is None
    assert wx["timestamp"] == "2024-06-06T07:00:00Z"


def test_get_weather_wraps_available_true():
    def handler(request):
        if request.url.path.endswith("token.php"):
            return httpx.Response(200, json={"access_token": imd_token(time.time() + 3600)})
        return httpx.Response(200, json=[CURRENT_WX])
    wx = run(make_service(handler).get_weather())
    assert wx["available"] is True and wx["temperature"] == 31.5


def test_weather_uses_official_24h_rainfall_from_cityforecast():
    # current_wx said 0 mm on a day the observatory recorded 80.8 mm.
    def handler(request):
        if request.url.path.endswith("token.php"):
            return httpx.Response(200, json={"access_token": imd_token(time.time() + 3600)})
        if request.url.path.endswith("cityforecast"):
            return httpx.Response(200, json=[{"Station_Code": "42809", "Date": "2026-10-02",
                                              "Past_24_hrs_Rainfall": "80.80"}])
        return httpx.Response(200, json=[{**CURRENT_WX, "Last 24 hrs Rainfall": "0"}])
    wx = run(make_service(handler).get_weather())
    assert wx["available"] is True and wx["rainfall_24h"] == 80.8
    assert "08:30 IST 2026-10-02" in wx["rainfall_24h_period"]


def test_official_nil_does_not_hide_rolling_rain():
    # Next day: official 08:30 window reports NIL, but rain fell since 08:30.
    def handler(request):
        if request.url.path.endswith("token.php"):
            return httpx.Response(200, json={"access_token": imd_token(time.time() + 3600)})
        if request.url.path.endswith("cityforecast"):
            return httpx.Response(200, json=[{"Date": "2026-10-03", "Past_24_hrs_Rainfall": "NIL"}])
        return httpx.Response(200, json=[{**CURRENT_WX, "Last 24 hrs Rainfall": "24"}])
    wx = run(make_service(handler).get_weather())
    assert wx["rainfall_24h"] == 24.0
    assert wx["rainfall_24h_official"] == 0.0 and wx["rainfall_24h_rolling"] == 24.0
    assert "rolling" in wx["rainfall_24h_source"]


def test_weather_keeps_current_wx_rainfall_if_cityforecast_fails():
    def handler(request):
        if request.url.path.endswith("token.php"):
            return httpx.Response(200, json={"access_token": imd_token(time.time() + 3600)})
        if request.url.path.endswith("cityforecast"):
            return httpx.Response(503, text="down")
        return httpx.Response(200, json=[CURRENT_WX])
    wx = run(make_service(handler).get_weather())
    assert wx["available"] is True and wx["rainfall_24h"] == 23.4
    assert "current_wx" in wx["rainfall_24h_source"]


def test_city_forecast_normalized():
    rec = {"Station_Code": "42809", "Station_Name": "KOLKATA", "Date": "2024-06-06",
           "Past_24_hrs_Rainfall": "23.4", "Todays_Forecast": "Rain", "Todays_Forecast_Max_Temp": "33",
           "Day_2_Forecast": "Thunderstorm", "Day_2_Max_Temp": "32", "Day_2_Min_temp": "27"}

    def handler(request):
        if request.url.path.endswith("token.php"):
            return httpx.Response(200, json={"access_token": imd_token(time.time() + 3600)})
        return httpx.Response(200, json=[rec])
    fc = run(make_service(handler).get_city_forecast())
    assert fc["past_24h_rainfall"] == 23.4 and len(fc["forecast_days"]) == 7
    assert fc["forecast_days"][0]["forecast"] == "Rain"
    assert fc["forecast_days"][1]["forecast"] == "Thunderstorm" and fc["forecast_days"][1]["max_temp"] == 32.0


# ------------------------------------------------------- graceful degradation

def test_ip_not_authorized_degrades_specifically():
    def handler(request):
        if request.url.path.endswith("token.php"):
            return httpx.Response(200, json={"access_token": imd_token(time.time() + 3600)})
        return httpx.Response(403, json={"error": "IP address 152.59.167.158 not authorized"})
    wx = run(make_service(handler).get_weather())
    assert wx["available"] is False
    assert wx["degraded_cause"] == "ip_not_authorized"
    assert "not authorized" in wx["reason"]


def test_get_current_weather_raises_ip_error_directly():
    def handler(request):
        if request.url.path.endswith("token.php"):
            return httpx.Response(200, json={"access_token": imd_token(time.time() + 3600)})
        return httpx.Response(403, json={"error": "IP address X not authorized"})
    with pytest.raises(IMDIPNotAuthorizedError):
        run(make_service(handler).get_current_weather())


def test_bad_credentials_degrade_as_auth():
    def handler(request):
        return httpx.Response(401, json={"error": "invalid"})
    wx = run(make_service(handler).get_weather())
    assert wx["available"] is False and wx["degraded_cause"] == "auth"


def test_expired_token_401_refresh_then_success():
    state = {"data_calls": 0}

    def handler(request):
        if request.url.path.endswith("token.php"):
            return httpx.Response(200, json={"access_token": imd_token(time.time() + 3600)})
        if request.url.path.endswith("cityforecast"):
            return httpx.Response(200, json=[{}])
        state["data_calls"] += 1
        if state["data_calls"] == 1:
            return httpx.Response(401, json={"error": "expired"})
        return httpx.Response(200, json=[CURRENT_WX])
    wx = run(make_service(handler).get_weather())
    assert wx["available"] is True and state["data_calls"] == 2


def test_server_error_degrades_as_unavailable():
    def handler(request):
        if request.url.path.endswith("token.php"):
            return httpx.Response(200, json={"access_token": imd_token(time.time() + 3600)})
        return httpx.Response(503, text="upstream down")
    wx = run(make_service(handler).get_weather())
    assert wx["available"] is False and wx["degraded_cause"] == "unavailable"


def test_timeout_degrades_not_crash():
    def handler(request):
        if request.url.path.endswith("token.php"):
            return httpx.Response(200, json={"access_token": imd_token(time.time() + 3600)})
        raise httpx.ReadTimeout("slow", request=request)
    wx = run(make_service(handler).get_weather())
    assert wx["available"] is False and wx["degraded_cause"] == "unavailable"


def test_unconfigured_degrades():
    svc = IMDService(api_key=None, email=None, password=None)
    assert svc.configured is False
    wx = run(svc.get_weather())
    assert wx["available"] is False and "not configured" in wx["reason"]
    with pytest.raises(IMDNotConfiguredError):
        run(svc.get_jwt_token())


def test_token_response_without_access_token_is_auth_error():
    def handler(request):
        return httpx.Response(200, json={"token_type": "Bearer"})
    with pytest.raises(IMDAuthError):
        run(make_service(handler).get_jwt_token())


# --------------------------------------------------------------- normalizers

def test_normalize_handles_missing_and_bad_fields():
    n = normalize_current_weather({"Station": "X", "Temperature": "", "Humidity": "abc",
                                   "Last 24 hrs Rainfall": None})
    assert n["temperature"] is None and n["humidity"] is None and n["rainfall_24h"] is None
    assert n["rainfall_30m"] is None


def test_normalize_forecast_missing_days():
    fc = normalize_city_forecast({"Station_Code": "1"})
    assert len(fc["forecast_days"]) == 7 and all(d["forecast"] is None for d in fc["forecast_days"])


# ------------------------------------------------------------- live (opt-in)

_configured = all(getattr(settings, k) is not None for k in ("IMD_API_KEY", "IMD_EMAIL", "IMD_PASSWORD"))
live = pytest.mark.skipif(not _configured, reason="IMD credentials not configured")


@live
def test_live_token_generation_and_caching(loop):
    from services.imd_service import IMDService as S
    svc = S(api_key=settings.IMD_API_KEY.get_secret_value(),
            email=settings.IMD_EMAIL.get_secret_value(),
            password=settings.IMD_PASSWORD.get_secret_value())

    async def go():
        try:
            t1 = await svc.get_jwt_token()
            t2 = await svc.get_jwt_token()  # cached: no second auth
            wx = await svc.get_weather()    # may be ip_not_authorized off-server
            return t1, t2, wx, svc.health()
        finally:
            await svc.aclose()
    t1, t2, wx, health = loop.run_until_complete(go())
    assert t1 and t1 == t2
    assert health["auth_count"] == 1  # JWT cached across calls
    # Data may be unavailable from an unregistered IP; that must be honest, not a crash.
    if not wx["available"]:
        assert wx["degraded_cause"] in ("ip_not_authorized", "auth", "unavailable")
    else:
        assert wx["temperature"] is None or isinstance(wx["temperature"], float)

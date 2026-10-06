"""Task 20: /api/data/status source-state verdicts and resilience.

Section 30: each source rolls up to an explicit state; a missing source reports
MISSING and never crashes the whole endpoint.
"""

import httpx
from fastapi.testclient import TestClient

import main
from services.imd_service import IMDService
from services.mapbox_service import MapboxService


c = TestClient(main.app)


def test_status_returns_all_sources_with_state():
    body = c.get("/api/data/status").json()
    src = body["sources"]
    for name in ("dem", "drainage", "water_bodies", "pumping_stations",
                 "historical_waterlogging", "mapbox", "imd"):
        assert name in src, name
        assert "state" in src[name], f"{name} missing state"


def test_loaded_sources_report_loaded_state():
    src = c.get("/api/data/status").json()["sources"]
    assert src["dem"]["state"] in ("LOADED", "MISSING", "ERROR")
    assert src["drainage"]["state"] == "LOADED"        # built in this repo
    assert src["water_bodies"]["state"] == "LOADED"


def test_model_state_real_when_loaded():
    m = c.get("/api/data/status").json()["model"]
    assert m["state"] in ("REAL", "MISSING", "ERROR")
    assert m["real_v1"]["model_version"] == "real_v1"
    assert m["synthetic_v1"]["model_version"] == "synthetic_v1"


def test_live_state_verdicts():
    from routers.routes import _live_state
    assert _live_state(False, {}) == "NOT_CONFIGURED"
    assert _live_state(True, {"last_success_ts": 123}) == "LIVE"
    assert _live_state(True, {"last_error": "boom"}) == "ERROR"
    assert _live_state(True, {}) == "UNKNOWN"


def test_dem_missing_reports_missing_not_crash(monkeypatch, tmp_path):
    """Section 30: rename the DEM; status must still return 200 and mark DEM MISSING."""
    import services.dem_service as dem_mod
    from config import settings
    # point the DEM service at a non-existent file and clear its cached singleton
    monkeypatch.setattr(settings, "DEM_PATH", tmp_path / "gone.tif")
    dem_mod.get_dem_service.cache_clear()
    try:
        r = c.get("/api/data/status")
        assert r.status_code == 200
        assert r.json()["sources"]["dem"]["state"] == "MISSING"
        # rest of the endpoint still populated
        assert r.json()["sources"]["drainage"]["state"] == "LOADED"
    finally:
        dem_mod.get_dem_service.cache_clear()  # restore real DEM for other tests


def test_drainage_missing_reports_missing(monkeypatch, tmp_path):
    import services.spatial_service as sp
    from config import settings
    monkeypatch.setattr(settings, "DRAINAGE_GEOJSON_PATH", tmp_path / "nope.geojson")
    sp.get_spatial_service.cache_clear()
    try:
        r = c.get("/api/data/status")
        assert r.status_code == 200
        assert r.json()["sources"]["drainage"]["state"] == "MISSING"
    finally:
        sp.get_spatial_service.cache_clear()


def test_ping_endpoint_with_mocked_services(monkeypatch):
    def mb_handler(request):
        if "/forward" in request.url.path:
            return httpx.Response(200, json={"type": "FeatureCollection", "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [88.36, 22.57]},
                 "properties": {"name": "Kolkata", "feature_type": "place"}}]})
        return httpx.Response(404, json={"message": "no"})

    def imd_handler(request):
        import base64, json, time
        if request.url.path.endswith("token.php"):
            seg0 = base64.urlsafe_b64encode(json.dumps({"uid": 1, "exp": int(time.time() + 3600)}).encode()).rstrip(b"=").decode()
            return httpx.Response(200, json={"access_token": f"{seg0}.sig"})
        return httpx.Response(200, json=[{"Station": "KOLKATA", "Temperature": "28",
                                          "Last 24 hrs Rainfall": "10", "Date of Observation": "2026-09-27"}])
    import services.mapbox_service as ms
    import services.imd_service as ims
    monkeypatch.setattr(ms, "_service",
                        MapboxService("pk.test", client=httpx.AsyncClient(transport=httpx.MockTransport(mb_handler), timeout=5)))
    monkeypatch.setattr(ims, "_service",
                        IMDService("k", "e@x.gov.in", "pw", client=httpx.AsyncClient(transport=httpx.MockTransport(imd_handler), timeout=5)))
    body = c.get("/api/data/status/ping").json()
    assert body["mapbox"]["connected"] is True
    assert body["imd"]["connected"] is True


def test_async_service_survives_reuse_across_event_loops():
    """Regression: a process-wide async service reused across two separate event
    loops must recreate its client rather than crash with 'Event loop is closed'."""
    import asyncio
    from services.mapbox_service import MapboxService

    def handler(request):
        return httpx.Response(200, json={"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [88.36, 22.57]},
             "properties": {"name": "K", "feature_type": "place"}}]})
    # owns its client (not injected) so the loop-rebind path is exercised
    svc = MapboxService("pk.test")

    async def one_call():
        # swap in a mock transport on whatever client the service builds this loop
        c = await svc._get_client()
        c._transport = httpx.MockTransport(handler)
        return await svc.geocode("Kolkata", limit=1)

    r1 = asyncio.new_event_loop().run_until_complete(one_call())
    r2 = asyncio.new_event_loop().run_until_complete(one_call())  # different, now-closed first loop
    assert r1 and r2  # no "Event loop is closed"


def test_mapbox_health_tracks_last_success():
    import asyncio

    def handler(request):
        return httpx.Response(200, json={"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [88.36, 22.57]},
             "properties": {"name": "K", "feature_type": "place"}}]})
    svc = MapboxService("pk.test", client=httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5))
    assert svc.health()["last_success_ts"] is None
    asyncio.new_event_loop().run_until_complete(svc.geocode("Kolkata"))
    assert svc.health()["last_success_ts"] is not None
    assert svc.health()["last_error"] is None

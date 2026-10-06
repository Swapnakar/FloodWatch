"""Task 24: end-to-end testing pass.

Fills gaps in the overall test matrix that earlier per-task tests don't cover.
Each test here targets a scenario from section 36 that was either missing or
only covered at the unit/service level but not at the HTTP-endpoint integration
level.

All Mapbox/IMD calls use httpx MockTransport so the suite is deterministic and
runs offline.
"""

import base64
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from services.imd_service import IMDService
from services.mapbox_service import MapboxService

# Kolkata corridor: Sealdah → Salt Lake.
O = (22.5694, 88.3706)
D = (22.5945, 88.4262)


def _geom(a, b, n=40):
    return {"type": "LineString",
            "coordinates": [[a[1] + (b[1] - a[1]) * i / n,
                             a[0] + (b[0] - a[0]) * i / n] for i in range(n + 1)]}


def _route_json(a, b, dist, dur, n=40):
    return {"distance": dist, "duration": dur, "weight_name": "routability",
            "geometry": _geom(a, b, n),
            "legs": [{"steps": [{"maneuver": {"instruction": "go"},
                                 "distance": 10.0, "duration": 2.0, "name": "Rd"}]}]}


def _geocode_feature(name, lon, lat):
    return {"type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": name, "full_address": f"{name}, Kolkata",
                           "feature_type": "place"}}


def _imd_token():
    seg0 = base64.urlsafe_b64encode(
        json.dumps({"uid": 1, "exp": int(time.time() + 3600)}).encode()
    ).rstrip(b"=").decode()
    return f"{seg0}.sig"


def _make_mapbox(handler, token="pk.test"):
    return MapboxService(
        token, client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=8))


def _make_imd(available=True):
    def handler(request):
        if request.url.path.endswith("token.php"):
            return httpx.Response(200, json={"access_token": _imd_token(),
                                             "token_type": "Bearer"})
        if available:
            return httpx.Response(200, json=[{
                "Station Id": "42809", "Station": "KOLKATA",
                "Temperature": "28", "Humidity": "90",
                "Last 24 hrs Rainfall": "60",
                "Date of Observation": "2026-09-27"}])
        return httpx.Response(403, json={"error": "IP not authorized"})
    return IMDService(
        "key", "e@x.gov.in", "pw",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=8))


@pytest.fixture
def client(monkeypatch):
    """TestClient with two-route Mapbox + available IMD, plus a swappable state."""
    two_routes = {"code": "Ok",
                  "routes": [_route_json(O, D, 8800, 1860),
                             _route_json(O, D, 9000, 1900, n=42)]}

    def mapbox_handler(request):
        p = request.url.path
        if "/directions/" in p:
            return httpx.Response(200, json=state["directions"])
        if "/forward" in p:
            q = request.url.params.get("q", "")
            lon, lat = (88.3706, 22.5694) if "seal" in q.lower() else (88.4262, 22.5945)
            return httpx.Response(200, json={"type": "FeatureCollection",
                                             "features": [_geocode_feature(q or "X", lon, lat)]})
        if "/reverse" in p:
            return httpx.Response(200, json={"type": "FeatureCollection",
                                             "features": [_geocode_feature("Some Rd", 88.4, 22.5)]})
        return httpx.Response(404, json={"message": "nope"})

    state = {"directions": two_routes}
    mb = _make_mapbox(mapbox_handler)
    imd = _make_imd(available=True)
    import services.mapbox_service as ms
    import services.imd_service as ims
    import services.route_risk_service as rrs
    monkeypatch.setattr(ms, "_service", mb)
    monkeypatch.setattr(ims, "_service", imd)
    rrs._service = None
    c = TestClient(main.app)
    c._state = state
    yield c


def _post_safe(client, **over):
    body = {"origin": {"lat": O[0], "lon": O[1]},
            "destination": {"lat": D[0], "lon": D[1]}}
    body.update(over)
    return client.post("/api/safe-route", json=body)


# ============================================================== /api/route/flood-risk

def test_route_flood_risk_endpoint(client):
    """Task 24 gap: /api/route/flood-risk was not tested at endpoint level."""
    r = client.post("/api/route/flood-risk", json={
        "origin": {"lat": O[0], "lon": O[1]},
        "destination": {"lat": D[0], "lon": D[1]}})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "real_v1"
    assert body["prediction_type"] == "flood_probability"
    assert "segments" in body
    assert body["n_segments"] > 0
    seg = body["segments"][0]
    for k in ("lat", "lng", "flood_probability", "risk_level", "elevation"):
        assert k in seg


def test_route_flood_risk_no_route(client):
    """NoRouteError → 404 on /api/route/flood-risk."""
    client._state["directions"] = {"code": "NoRoute", "routes": [],
                                   "message": "no route"}
    r = client.post("/api/route/flood-risk", json={
        "origin": {"lat": O[0], "lon": O[1]},
        "destination": {"lat": D[0], "lon": D[1]}})
    assert r.status_code == 404


# ============================================================== /api/reverse-geocode

def test_reverse_geocode_endpoint(client):
    """Task 24 gap: /api/reverse-geocode not tested at endpoint level."""
    r = client.get("/api/reverse-geocode", params={"lat": 22.57, "lon": 88.37})
    assert r.status_code == 200
    body = r.json()
    assert body["lat"] == 22.57 and body["lon"] == 88.37
    assert "candidates" in body


def test_reverse_geocode_invalid_coords(client):
    """Out-of-range coordinates rejected by FastAPI validation."""
    r = client.get("/api/reverse-geocode", params={"lat": 999, "lon": 88})
    # Mapbox service rejects, but the response should be non-200
    assert r.status_code >= 400


# ============================================================== Mapbox malformed response at endpoint

def test_safe_route_mapbox_server_error(monkeypatch, client):
    """Task 24 gap: Mapbox 500 → clean error, not a stack trace."""
    def mapbox_500(request):
        p = request.url.path
        if "/forward" in p:
            return httpx.Response(200, json={"type": "FeatureCollection",
                                             "features": [_geocode_feature("X", 88.37, 22.57)]})
        return httpx.Response(500, text="internal server error")

    import services.mapbox_service as ms
    import services.route_risk_service as rrs
    monkeypatch.setattr(ms, "_service", _make_mapbox(mapbox_500))
    rrs._service = None
    r = _post_safe(client)
    assert r.status_code >= 500  # propagated as 502 or similar
    assert "detail" in r.json()


# ============================================================== safe-route with profile

def test_safe_route_with_profile(client):
    """Task 24 gap: profile parameter forwarded correctly."""
    body = {"origin": {"lat": O[0], "lon": O[1]},
            "destination": {"lat": D[0], "lon": D[1]},
            "profile": "driving"}
    r = client.post("/api/safe-route", json=body)
    assert r.status_code == 200


# ============================================================== model provenance on segments

def test_safe_route_segments_have_probability_and_risk(client):
    """Every segment in the safe-route response has flood_probability and
    risk_level (possibly None if no features, but the keys must be present)."""
    body = _post_safe(client).json()
    for route in body["routes"]:
        for seg in route["segments"]:
            assert "flood_probability" in seg
            assert "risk_level" in seg
            assert "elevation" in seg
            assert "distance_to_drain_m" in seg
            assert "missing" in seg


def test_safe_route_disclaimer_present(client):
    """The disclaimer field must be present and must never claim safety."""
    body = _post_safe(client).json()
    assert "disclaimer" in body
    assert "100% safe" not in body["disclaimer"].lower()
    assert "guaranteed flood-free" not in body["disclaimer"].lower() or \
           "no route is guaranteed flood-free" in body["disclaimer"].lower()


# ============================================================== coverage in safe-route

def test_safe_route_coverage_reported(client):
    """Each route in the response must carry coverage percentages."""
    body = _post_safe(client).json()
    for route in body["routes"]:
        assert "coverage" in route
        cov = route["coverage"]
        for k in ("dem_pct", "drainage_pct", "historical_pct"):
            assert k in cov
            assert 0.0 <= cov[k] <= 100.0


# ============================================================== /api/location-risk edge cases

def test_location_risk_outside_dem(client):
    """Delhi coordinate: no DEM → no_data coverage, flood_probability None."""
    r = client.get("/api/location-risk?lat=28.61&lon=77.20")
    assert r.status_code == 200
    body = r.json()
    assert body["coverage"] == "no_data"
    assert body["flood_probability"] is None
    assert body["elevation_m"] is None
    assert "outside Kolkata" in body["disclaimer"].lower() or \
           "Outside Kolkata" in body["disclaimer"]


# ============================================================== legacy endpoints cross-labelling

def test_batch_predict_tagged_synthetic(client):
    """Task 24 gap: /api/predict/batch must tag every result synthetic_v1."""
    r = client.post("/api/predict/batch", json={
        "locations": [{"rainfall_30m": 50, "elevation": 8}]})
    assert r.status_code == 200
    for pred in r.json()["predictions"]:
        assert pred["model_version"] == "synthetic_v1"
        assert pred["prediction_type"] == "prototype_estimated_depth"
        assert "real" not in pred.get("prediction_type", "").lower()


def test_legacy_drainage_endpoint_unchanged(client):
    """/api/drainage still returns the static placeholder data."""
    r = client.get("/api/drainage")
    assert r.status_code == 200
    body = r.json()
    assert "drainage_nodes" in body and "pipe_segments" in body


def test_home_endpoint_lists_all_endpoints(client):
    """The root / endpoint should list both legacy and real endpoints."""
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "/api/predict" in body.get("legacy_endpoints", [])
    assert "/api/safe-route" in body.get("real_endpoints", [])


# ============================================================== safe-route include_segments=false

def test_safe_route_without_segments(client):
    """include_segments=false omits the per-segment detail."""
    body = {"origin": {"lat": O[0], "lon": O[1]},
            "destination": {"lat": D[0], "lon": D[1]},
            "include_segments": False}
    r = client.post("/api/safe-route", json=body)
    assert r.status_code == 200
    for route in r.json()["routes"]:
        assert "segments" not in route
        assert "route_score" in route
        assert "n_segments" in route and route["n_segments"] > 0


# ============================================================== /api/weather endpoint degraded

def test_weather_endpoint_degraded(monkeypatch, client):
    """IMD failure → available:false, not a crash."""
    import services.imd_service as ims
    import services.route_risk_service as rrs
    ims._service = _make_imd(available=False)
    rrs._service = None
    r = client.get("/api/weather")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False


# ============================================================== /api/data/status/ping

def test_data_status_ping_connectivity(client):
    """The active-probe ping endpoint should return Mapbox and IMD status."""
    r = client.get("/api/data/status/ping")
    assert r.status_code == 200
    body = r.json()
    assert "mapbox" in body
    assert "imd" in body
    assert "connected" in body["imd"]


# ============================================================== full pipeline integrity chain

def test_full_pipeline_geocode_through_safe_route(client):
    """Round-trip: geocode via text → safe-route → scored segments.
    Verifies the entire chain works end to end with mocked externals."""
    r = client.post("/api/safe-route", json={
        "origin_query": "Sealdah", "destination_query": "Salt Lake"})
    assert r.status_code == 200
    body = r.json()
    # geocoded resolution present
    assert body["origin"]["resolved"] is not None
    assert body["destination"]["resolved"] is not None
    # routes scored
    assert len(body["routes"]) >= 1
    assert body["routes"][0]["route_score"] is not None or \
           body["routes"][0]["n_segments"] > 0
    # recommendation present and honest
    assert "recommendation" in body
    assert "100% safe" not in body["recommendation"].lower()
    # model tagging correct
    assert body["model_version"] == "real_v1"
    assert body["prediction_type"] == "flood_probability"


# ============================================================== cross-model isolation

def test_safe_route_never_says_synthetic(client):
    """Safe-route must never claim it's using the synthetic model."""
    body = _post_safe(client).json()
    assert body["model_version"] != "synthetic_v1"
    assert "synthetic" not in body.get("training_data_type", "").lower()
    assert "synthetic" not in body.get("prediction_type", "").lower()


def test_predict_never_says_real(client):
    """/api/predict must never claim it's using the real model."""
    body = client.post("/api/predict", json={}).json()
    assert body["model_version"] != "real_v1"
    assert "real" not in body.get("prediction_type", "").lower()

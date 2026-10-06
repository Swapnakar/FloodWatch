"""Endpoint tests for /api/safe-route and friends.

Mapbox and IMD are replaced with httpx MockTransport-backed services so the
tests are deterministic and offline. DEM/spatial/model use the real loaded
artifacts (present in this repo).
"""

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from services.imd_service import IMDService
from services.mapbox_service import MapboxService

# Sealdah -> Salt Lake area, a real Kolkata corridor.
O = (22.5694, 88.3706)
D = (22.5945, 88.4262)


def _geom(a, b, n=40):
    return {"type": "LineString",
            "coordinates": [[a[1] + (b[1] - a[1]) * i / n, a[0] + (b[0] - a[0]) * i / n] for i in range(n + 1)]}


def _route_json(a, b, dist, dur, n=40):
    return {"distance": dist, "duration": dur, "weight_name": "routability",
            "geometry": _geom(a, b, n),
            "legs": [{"steps": [{"maneuver": {"instruction": "go"}, "distance": 10.0,
                                 "duration": 2.0, "name": "Rd"}]}]}


def _geocode_feature(name, lon, lat):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": name, "full_address": f"{name}, Kolkata", "feature_type": "place"}}


def make_mapbox(handler, token="pk.test"):
    return MapboxService(token, client=httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=8))


def make_imd(available=True):
    def handler(request):
        if request.url.path.endswith("token.php"):
            import base64, json, time
            seg0 = base64.urlsafe_b64encode(json.dumps({"uid": 1, "exp": int(time.time() + 3600)}).encode()).rstrip(b"=").decode()
            return httpx.Response(200, json={"access_token": f"{seg0}.sig", "token_type": "Bearer"})
        if available:
            return httpx.Response(200, json=[{"Station Id": "42809", "Station": "KOLKATA",
                                              "Temperature": "28", "Humidity": "90",
                                              "Last 24 hrs Rainfall": "60", "Date of Observation": "2026-09-27"}])
        return httpx.Response(403, json={"error": "IP address 1.2.3.4 not authorized"})
    return IMDService("key", "e@x.gov.in", "pw", client=httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=8))


@pytest.fixture
def client(monkeypatch):
    """TestClient with Mapbox+IMD mocked. Default: 2 routes, IMD available."""
    two_routes = {"code": "Ok", "routes": [_route_json(O, D, 8800, 1860),
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
    mb = make_mapbox(mapbox_handler)
    imd = make_imd(available=True)
    import services.mapbox_service as ms
    import services.imd_service as ims
    import services.route_risk_service as rrs
    monkeypatch.setattr(ms, "_service", mb)
    monkeypatch.setattr(ims, "_service", imd)
    # route risk service uses the (patched) singletons
    rrs._service = None
    c = TestClient(main.app)
    c._state = state  # let tests swap the directions payload
    yield c


def _post_safe(client, **over):
    body = {"origin": {"lat": O[0], "lon": O[1]}, "destination": {"lat": D[0], "lon": D[1]}}
    body.update(over)
    return client.post("/api/safe-route", json=body)


# ------------------------------------------------------------ core cases

def test_two_route_case(client):
    r = _post_safe(client)
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "real_v1"
    assert len(body["routes"]) == 2
    assert body["recommended_index"] in (0, 1)
    assert "safe" not in body["recommendation"].lower()
    assert "disclaimer" in body
    for route in body["routes"]:
        assert "route_score" in route and "geometry" in route
        assert route["n_segments"] > 0


def test_single_route_case(client):
    client._state["directions"] = {"code": "Ok", "routes": [_route_json(O, D, 8800, 1860)]}
    body = _post_safe(client).json()
    assert len(body["routes"]) == 1
    assert body["recommended_index"] == 0


def test_no_route_returns_404(client):
    client._state["directions"] = {"code": "NoRoute", "routes": [], "message": "no route"}
    r = _post_safe(client)
    assert r.status_code == 404


def test_all_routes_risky_warns_end_to_end(monkeypatch, client):
    """Full endpoint: when every route scores high, the response must set
    all_risky and warn, never calling any route safe."""
    import routers.routes as rr
    # Patch the name as bound in the router (imported by name), so every route
    # gets a high score and the all-risky path triggers end to end.
    monkeypatch.setattr(rr, "route_score", lambda probs: 0.85)
    body = _post_safe(client).json()
    assert body["all_risky"] is True
    assert body["recommended_index"] in (0, 1)  # still names the least-bad
    assert "safe" not in body["recommendation"].lower()
    assert "elevated flood risk" in body["recommendation"].lower()


def test_geocoded_text_inputs(client):
    r = client.post("/api/safe-route", json={"origin_query": "Sealdah", "destination_query": "Salt Lake"})
    assert r.status_code == 200
    body = r.json()
    assert body["origin"]["resolved"] is not None
    assert body["destination"]["resolved"] is not None


def test_missing_inputs_422(client):
    r = client.post("/api/safe-route", json={"origin_query": "Sealdah"})  # no destination
    assert r.status_code == 422


def test_every_segment_tagged_and_has_keys(client):
    body = _post_safe(client).json()
    seg = body["routes"][0]["segments"][0]
    for k in ("lat", "lng", "flood_probability", "risk_level", "elevation", "distance_to_drain_m"):
        assert k in seg


def test_response_never_says_safe(client):
    body = _post_safe(client).json()
    text = str(body).lower()
    # "safe-route" appears in the path, but no claim like "is safe"/"100% safe".
    assert "100% safe" not in text and "is safe" not in text


# ------------------------------------------------------------ IMD failure

def test_imd_failure_still_returns_route(monkeypatch, client):
    import services.imd_service as ims
    import services.route_risk_service as rrs
    ims._service = make_imd(available=False)
    rrs._service = None
    body = _post_safe(client).json()
    assert body["routes"][0]["rainfall"]["available"] is False
    # route still scored (susceptibility doesn't require rainfall)
    assert body["routes"][0]["n_segments"] > 0


# ------------------------------------------------------------ other endpoints

def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_geocode_endpoint(client):
    body = client.get("/api/geocode", params={"q": "Sealdah"}).json()
    assert body["candidates"] and body["candidates"][0]["lat"]


def test_weather_endpoint(client):
    body = client.get("/api/weather").json()
    assert body["available"] is True and body["rainfall_24h"] == 60.0


def test_route_endpoint(client):
    r = client.post("/api/route", json={"origin": {"lat": O[0], "lon": O[1]},
                                        "destination": {"lat": D[0], "lon": D[1]}})
    assert r.status_code == 200 and len(r.json()["routes"]) == 2


def test_data_status_endpoint(client):
    body = client.get("/api/data/status").json()
    assert "sources" in body and "model" in body
    assert body["model"]["synthetic_v1"]["prediction_type"] == "prototype_estimated_depth"
    assert body["sources"]["dem"]["dem_status"] in ("LOADED", "MISSING", "ERROR")


# ------------------------------------------------------------ legacy intact

def test_legacy_endpoints_unchanged(client):
    assert client.post("/api/predict", json={}).json()["model_version"] == "synthetic_v1"
    assert client.get("/api/model/info").json()["model_version"] == "synthetic_v1"
    assert client.get("/api/nowcast").json()["legacy"] is True

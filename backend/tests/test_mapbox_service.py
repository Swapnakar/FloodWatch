import asyncio
import json

import httpx
import pytest

from config import settings
from services.mapbox_service import (
    InvalidInputError, MapboxAuthError, MapboxNotConfiguredError,
    MapboxRateLimitError, MapboxService, MapboxUnavailableError, NoRouteError,
)


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def run(coro):
    """Run a coroutine on a throwaway loop (mocked tests: no shared client)."""
    return asyncio.new_event_loop().run_until_complete(coro)


def make_service(handler, token="pk.test"):
    """MapboxService backed by a scripted MockTransport (no network)."""
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=8.0)
    return MapboxService(token, client=client)


# --------------------------------------------------------- response fixtures

def geocode_body(features):
    return {"type": "FeatureCollection", "features": features, "attribution": "© Mapbox"}


def feature(name, lon, lat, ftype="street", relevance=0.9, full="X, Kolkata"):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": name, "full_address": full, "feature_type": ftype,
                           "mapbox_id": "id." + name.replace(" ", ""),
                           "context": {"place": {"name": "Kolkata"}, "postcode": {"name": "700001"}}}}


def route_body(routes):
    return {"code": "Ok", "routes": routes, "waypoints": []}


def route(coords, distance=8800.0, duration=1860.0, with_steps=True):
    legs = [{"steps": [{"maneuver": {"instruction": "Head north"}, "distance": 100.0,
                        "duration": 20.0, "name": "AJC Bose Rd"}] if with_steps else []}]
    return {"distance": distance, "duration": duration, "weight_name": "routability",
            "geometry": {"type": "LineString", "coordinates": coords}, "legs": legs}


# ------------------------------------------------------------------ geocode

def test_geocode_returns_ranked_candidates():
    def handler(request):
        assert "/search/geocode/v6/forward" in str(request.url)
        assert request.url.params["permanent"] == "false"        # temporary by default
        assert "access_token" in request.url.params
        return httpx.Response(200, json=geocode_body([
            feature("Sealdah", 88.3706, 22.5694, relevance=0.99),
            feature("Sealdah Court", 88.371, 22.57, relevance=0.8)]))
    svc = make_service(handler)
    cands = run(svc.geocode("Sealdah"))
    assert [c.name for c in cands] == ["Sealdah", "Sealdah Court"]  # order preserved (already ranked)
    assert cands[0].lat == 22.5694 and cands[0].lon == 88.3706
    assert cands[0].to_dict()["context"]["place"] == "Kolkata"


def test_geocode_token_never_in_candidate_output():
    svc = make_service(lambda r: httpx.Response(200, json=geocode_body([feature("A", 88.4, 22.5)])), token="pk.SECRET")
    dumped = json.dumps([c.to_dict() for c in run(svc.geocode("A"))])
    assert "SECRET" not in dumped


def test_geocode_empty_query_rejected_without_network():
    called = []
    svc = make_service(lambda r: called.append(1) or httpx.Response(200, json=geocode_body([])))
    with pytest.raises(InvalidInputError):
        run(svc.geocode("   "))
    assert not called


def test_geocode_no_results_is_empty_list():
    svc = make_service(lambda r: httpx.Response(200, json=geocode_body([])))
    assert run(svc.geocode("nowhere-xyz")) == []


def test_address_match_confidence_captured():
    addr = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [88.4, 22.5]},
            "properties": {"name": "12 Park St", "feature_type": "address",
                           "match_code": {"confidence": "high"}}}
    svc = make_service(lambda r: httpx.Response(200, json=geocode_body([addr])))
    assert run(svc.geocode("12 Park St"))[0].match_confidence == "high"


def test_geocode_skips_malformed_features():
    body = geocode_body([{"type": "Feature", "geometry": {"type": "Point", "coordinates": [999, 999]},
                          "properties": {"name": "bad"}}, feature("Good", 88.4, 22.5)])
    svc = make_service(lambda r: httpx.Response(200, json=body))
    cands = run(svc.geocode("x"))
    assert [c.name for c in cands] == ["Good"]


# ---------------------------------------------------------- reverse geocode

def test_reverse_geocode_ok():
    def handler(request):
        assert "/search/geocode/v6/reverse" in str(request.url)
        assert request.url.params["longitude"] == "88.4"
        return httpx.Response(200, json=geocode_body([feature("Some Road", 88.4, 22.5)]))
    svc = make_service(handler)
    cands = run(svc.reverse_geocode(22.5, 88.4))
    assert cands[0].name == "Some Road"


@pytest.mark.parametrize("lat,lon", [(200, 88), (22.5, 500), (float("nan"), 88.4)])
def test_reverse_geocode_rejects_bad_coords(lat, lon):
    svc = make_service(lambda r: httpx.Response(200, json=geocode_body([])))
    with pytest.raises(InvalidInputError):
        run(svc.reverse_geocode(lat, lon))


# ------------------------------------------------------------------ routing

def test_get_routes_returns_alternatives_with_geometry():
    coords = [[88.37, 22.57], [88.40, 22.58], [88.42, 22.59]]
    svc = make_service(lambda r: httpx.Response(200, json=route_body([route(coords), route(coords[::-1], 9000.0)])))
    routes = run(svc.get_routes((22.5694, 88.3706), (22.5945, 88.4262)))
    assert len(routes) == 2
    assert routes[0].geometry["type"] == "LineString" and len(routes[0].coordinates) == 3
    d = routes[0].to_dict()
    assert d["distance_km"] == 8.8 and d["duration_min"] == 31.0 and d["steps"][0]["name"] == "AJC Bose Rd"


def test_get_routes_sends_correct_lonlat_order_and_params():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=route_body([route([[88.37, 22.57], [88.43, 22.59]])]))
    svc = make_service(handler)
    run(svc.get_routes((22.5694, 88.3706), (22.5945, 88.4262)))
    assert seen["path"].endswith("/88.3706,22.5694;88.4262,22.5945")  # lon,lat;lon,lat
    assert seen["params"]["geometries"] == "geojson" and seen["params"]["alternatives"] == "true"


def test_no_route_body_code_raises_no_route():
    svc = make_service(lambda r: httpx.Response(200, json={"code": "NoRoute", "routes": [],
                                                           "message": "no route"}))
    with pytest.raises(NoRouteError):
        run(svc.get_routes((22.5, 88.3), (22.6, 88.4)))


def test_no_segment_raises_no_route():
    svc = make_service(lambda r: httpx.Response(200, json={"code": "NoSegment", "routes": []}))
    with pytest.raises(NoRouteError):
        run(svc.get_routes((22.5, 88.3), (0.0, 0.0)))


def test_empty_routes_list_raises_no_route():
    svc = make_service(lambda r: httpx.Response(200, json=route_body([])))
    with pytest.raises(NoRouteError):
        run(svc.get_routes((22.5, 88.3), (22.6, 88.4)))


@pytest.mark.parametrize("bad", ["hovercraft", "", "DRIVING"])
def test_invalid_profile_rejected(bad):
    svc = make_service(lambda r: httpx.Response(200, json=route_body([])))
    with pytest.raises(InvalidInputError):
        run(svc.get_routes((22.5, 88.3), (22.6, 88.4), profile=bad))


@pytest.mark.parametrize("o,d", [((200, 88), (22, 88)), ((22, 88), (22, 500)), ((22,), (22, 88))])
def test_invalid_route_coords_rejected(o, d):
    svc = make_service(lambda r: httpx.Response(200, json=route_body([])))
    with pytest.raises(InvalidInputError):
        run(svc.get_routes(o, d))


# ------------------------------------------------------------- error mapping

@pytest.mark.parametrize("status,code,exc", [
    (401, "Not Authorized - Invalid Token", MapboxAuthError),
    (403, "Forbidden", MapboxAuthError),
    (422, "InvalidInput", InvalidInputError),
    (404, "ProfileNotFound", InvalidInputError),
    (429, None, MapboxRateLimitError),
    (500, None, MapboxUnavailableError),
    (503, None, MapboxUnavailableError),
])
def test_http_error_codes_map_to_typed_exceptions(status, code, exc):
    body = {"message": "x"} | ({"code": code} if code else {})
    svc = make_service(lambda r: httpx.Response(status, json=body))
    with pytest.raises(exc):
        run(svc.geocode("x"))


def test_auth_error_message_does_not_leak_token():
    svc = make_service(lambda r: httpx.Response(401, json={"message": "Invalid Token: pk.SECRET"}), token="pk.SECRET")
    try:
        run(svc.geocode("x"))
        assert False
    except MapboxAuthError as e:
        assert "SECRET" not in str(e) and e.status == 502


def test_timeout_maps_to_unavailable():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)
    with pytest.raises(MapboxUnavailableError):
        run(make_service(handler).geocode("x"))


def test_network_error_maps_to_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)
    with pytest.raises(MapboxUnavailableError):
        run(make_service(handler).get_routes((22.5, 88.3), (22.6, 88.4)))


def test_non_json_body_does_not_crash():
    svc = make_service(lambda r: httpx.Response(500, text="<html>gateway</html>"))
    with pytest.raises(MapboxUnavailableError):
        run(svc.geocode("x"))


def test_unconfigured_service_raises_before_network():
    called = []
    svc = MapboxService(token=None, client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: called.append(1) or httpx.Response(200, json=geocode_body([])))))
    assert svc.configured is False
    with pytest.raises(MapboxNotConfiguredError):
        run(svc.geocode("x"))
    assert not called


# ------------------------------------------------------------- live (opt-in)

live = pytest.mark.skipif(
    not (settings.MAPBOX_ACCESS_TOKEN and settings.MAPBOX_ACCESS_TOKEN.get_secret_value().strip()),
    reason="MAPBOX_ACCESS_TOKEN not configured",
)


def _live_service():
    """Fresh service with its own client (a live test owns one loop end-to-end)."""
    token = settings.MAPBOX_ACCESS_TOKEN.get_secret_value()
    return MapboxService(token)


@live
def test_live_geocode_sealdah(loop):
    async def go():
        svc = _live_service()
        try:
            return await svc.geocode("Sealdah Railway Station")
        finally:
            await svc.aclose()
    cands = loop.run_until_complete(go())
    assert cands, "expected at least one candidate"
    assert 22.4 <= cands[0].lat <= 22.7 and 88.2 <= cands[0].lon <= 88.5


@live
def test_live_routes_sealdah_to_saltlake(loop):
    async def go():
        svc = _live_service()
        try:
            o = (await svc.geocode("Sealdah, Kolkata"))[0]
            d = (await svc.geocode("Salt Lake Sector V, Kolkata"))[0]
            return await svc.get_routes((o.lat, o.lon), (d.lat, d.lon))
        finally:
            await svc.aclose()
    routes = loop.run_until_complete(go())
    assert len(routes) >= 1
    assert routes[0].distance_m > 0 and len(routes[0].coordinates) > 10


@live
def test_live_invalid_coordinates_clean_error(loop):
    async def go():
        svc = _live_service()
        try:
            # Bay of Bengal to a point on land: expect a clean NoRoute, not a crash.
            return await svc.get_routes((15.0, 88.0), (22.5726, 88.3639))
        finally:
            await svc.aclose()
    with pytest.raises((NoRouteError, InvalidInputError)):
        loop.run_until_complete(go())

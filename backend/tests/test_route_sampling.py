import math

import pytest
from pyproj import Transformer
from shapely.geometry import LineString

from config import settings
from services.route_sampling import METRIC_CRS, route_length_m, sample_route

TO_UTM = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
TO_LL = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)
# Kolkata origin in UTM 45N metres (~88.40 E, 22.52 N).
E0, N0 = 644_000.0, 2_491_000.0


def ll(e, n):
    lon, lat = TO_LL.transform(e, n)
    return lon, lat  # GeoJSON order


def line_utm(*offsets):
    """GeoJSON LineString through E0+de, N0+dn points."""
    return {"type": "LineString", "coordinates": [list(ll(E0 + de, N0 + dn)) for de, dn in offsets]}


def spacings(samples):
    return [samples[i + 1].distance_from_start_m - samples[i].distance_from_start_m
            for i in range(len(samples) - 1)]


# ------------------------------------------------------- spacing / counts

def test_750m_line_at_75_yields_11_points_endpoint_included():
    line = line_utm((0, 0), (750, 0))
    s = sample_route(line, spacing_m=75)
    assert len(s) == 11
    assert s[0].distance_from_start_m == 0
    assert s[-1].distance_from_start_m == pytest.approx(750, abs=0.01)
    assert all(abs(g - 75) < 0.01 for g in spacings(s))


def test_740m_line_includes_exact_endpoint_with_short_last_step():
    s = sample_route(line_utm((0, 0), (740, 0)), spacing_m=75)
    # 0,75,...,675 (10 pts) then exact 740 -> 11 points, last step 65 m
    assert len(s) == 11
    assert s[-1].distance_from_start_m == pytest.approx(740, abs=0.01)
    assert spacings(s)[-1] == pytest.approx(65, abs=0.02)


def test_exclude_endpoint_drops_partial_tail():
    s = sample_route(line_utm((0, 0), (740, 0)), spacing_m=75, include_endpoint=False)
    assert len(s) == 10
    assert s[-1].distance_from_start_m == pytest.approx(675, abs=0.01)
    assert all(abs(g - 75) < 0.01 for g in spacings(s))


def test_exact_multiple_not_duplicated_when_endpoint_included():
    s = sample_route(line_utm((0, 0), (300, 0)), spacing_m=75)
    dists = [round(x.distance_from_start_m, 3) for x in s]
    assert dists == [0.0, 75.0, 150.0, 225.0, 300.0]  # 300 appears once


def test_line_shorter_than_spacing_keeps_both_ends():
    s = sample_route(line_utm((0, 0), (40, 0)), spacing_m=75)
    assert [round(x.distance_from_start_m, 1) for x in s] == [0.0, 40.0]


# ---------------------------------------------------------- geometry fidelity

def test_samples_lie_on_a_diagonal_line():
    line = line_utm((0, 0), (300, 400))  # 500 m diagonal
    assert route_length_m(line) == pytest.approx(500, abs=0.05)
    s = sample_route(line, spacing_m=100)
    assert len(s) == 6  # 0,100,200,300,400,500
    # Each sample, reprojected to UTM, sits on the straight line at the right distance.
    for smp in s:
        e, n = TO_UTM.transform(smp.lng, smp.lat)
        t = smp.distance_from_start_m / 500
        assert e == pytest.approx(E0 + 300 * t, abs=0.5)
        assert n == pytest.approx(N0 + 400 * t, abs=0.5)


def test_metric_spacing_not_degree_spacing():
    # A due-east line: equal metre steps must NOT be equal longitude steps far
    # from where a naive degree interpolation would put them. Check spacing in metres.
    s = sample_route(line_utm((0, 0), (600, 0)), spacing_m=75)
    for a, b in zip(s[:-1], s[1:]):
        ea, na = TO_UTM.transform(a.lng, a.lat)
        eb, nb = TO_UTM.transform(b.lng, b.lat)
        assert math.hypot(eb - ea, nb - na) == pytest.approx(75, abs=0.05)


def test_multi_vertex_route_distance_is_cumulative():
    line = line_utm((0, 0), (300, 0), (300, 300))  # L-shape, 600 m
    assert route_length_m(line) == pytest.approx(600, abs=0.05)
    s = sample_route(line, spacing_m=150)
    assert [round(x.distance_from_start_m, 1) for x in s] == [0.0, 150.0, 300.0, 450.0, 600.0]
    # The 4th point (450 m) is 150 m up the vertical leg from the corner.
    e, n = TO_UTM.transform(s[3].lng, s[3].lat)
    assert e == pytest.approx(E0 + 300, abs=0.5) and n == pytest.approx(N0 + 150, abs=0.5)


# --------------------------------------------------------------- inputs

def test_accepts_shapely_and_coord_list():
    coords = [ll(E0, N0), ll(E0 + 150, N0)]
    from_list = sample_route(coords, spacing_m=75)
    from_shapely = sample_route(LineString(coords), spacing_m=75)
    assert len(from_list) == len(from_shapely) == 3


def test_default_spacing_from_settings():
    s = sample_route(line_utm((0, 0), (settings.ROUTE_SAMPLE_DISTANCE_M, 0)))
    assert len(s) == 2  # start + endpoint at exactly one spacing


def test_degenerate_zero_length_line_single_sample():
    s = sample_route(line_utm((0, 0), (0, 0)), spacing_m=75)
    assert len(s) == 1 and s[0].distance_from_start_m == 0.0


@pytest.mark.parametrize("bad", [
    {"type": "Point", "coordinates": [88.4, 22.5]},
    {"type": "LineString", "coordinates": [[88.4, 22.5]]},              # 1 point
    {"type": "LineString", "coordinates": [[88.4, 22.5], [200, 22.5]]},  # lon out of range
    {"type": "LineString", "coordinates": [[88.4, 22.5], [float("nan"), 22.5]]},
])
def test_invalid_geometry_rejected(bad):
    with pytest.raises(ValueError):
        sample_route(bad, spacing_m=75)


def test_nonpositive_spacing_rejected():
    with pytest.raises(ValueError):
        sample_route(line_utm((0, 0), (100, 0)), spacing_m=0)


def test_to_dict_shape():
    s = sample_route(line_utm((0, 0), (150, 0)), spacing_m=75)[0]
    d = s.to_dict()
    assert set(d) == {"lat", "lng", "distance_from_start_m"}
    assert 22.4 <= d["lat"] <= 22.7 and 88.2 <= d["lng"] <= 88.5


# ---------------------------------------------------- real Mapbox geometry

def _mapbox_available():
    tok = settings.MAPBOX_ACCESS_TOKEN
    return bool(tok and tok.get_secret_value().strip())


@pytest.mark.skipif(not _mapbox_available(), reason="MAPBOX_ACCESS_TOKEN not configured")
def test_real_mapbox_route_resamples_evenly():
    import asyncio
    from services.mapbox_service import MapboxService

    async def go():
        svc = MapboxService(settings.MAPBOX_ACCESS_TOKEN.get_secret_value())
        try:
            routes = await svc.get_routes((22.5694, 88.3706), (22.5945, 88.4262))
            return routes[0].geometry
        finally:
            await svc.aclose()
    geom = asyncio.new_event_loop().run_until_complete(go())
    length = route_length_m(geom)
    s = sample_route(geom, spacing_m=75)
    # count matches the documented convention: interior points + included endpoint
    expected_interior = int(math.ceil(length / 75 - 1e-6))
    assert len(s) == expected_interior + 1
    assert s[0].distance_from_start_m == 0
    assert s[-1].distance_from_start_m == pytest.approx(length, abs=0.5)
    # every interior gap is exactly 75 m; only the last may be shorter
    for g in spacings(s)[:-1]:
        assert g == pytest.approx(75, abs=0.1)
    assert spacings(s)[-1] <= 75 + 1e-6
    # all samples inside Kolkata
    assert all(22.4 <= smp.lat <= 22.75 and 88.2 <= smp.lng <= 88.55 for smp in s)

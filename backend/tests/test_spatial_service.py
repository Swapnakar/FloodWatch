import math

import geopandas as gpd
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon

from config import settings
from services import capacity_proxy as cp
from services.spatial_service import SpatialService

UTM = "EPSG:32645"
TO_LL = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
# Local origin inside Kolkata (UTM 45N metres)
E0, N0 = 590_000.0, 2_490_000.0


def ll(e, n):
    """UTM metres -> (lat, lon)."""
    lon, lat = TO_LL.transform(e, n)
    return lat, lon


def _write(tmp_path, name, rows, geoms):
    g = gpd.GeoDataFrame(rows, geometry=geoms, crs=UTM).to_crs("EPSG:4326")
    p = tmp_path / f"{name}.geojson"
    g.to_file(p, driver="GeoJSON")
    return p


@pytest.fixture
def svc(tmp_path):
    drains = _write(tmp_path, "drains", [
        {"segment_id": "a", "ward": 107, "conduit_type": "pipe", "pipe_diameter_mm": 300,
         "diameter_source": "legend_colour", "source": "test"},
        {"segment_id": "b", "ward": 107, "conduit_type": "pipe", "pipe_diameter_mm": None,
         "diameter_source": None, "source": "test"},
        {"segment_id": "c", "ward": 107, "conduit_type": "box_sewer", "pipe_diameter_mm": None,
         "diameter_source": None, "source": "test"},
    ], [
        LineString([(E0, N0), (E0 + 100, N0)]),                  # a: along y = N0
        LineString([(E0, N0 + 200), (E0 + 100, N0 + 200)]),      # b: along y = N0+200
        LineString([(E0 + 300, N0), (E0 + 300, N0 + 200)]),      # c: vertical at x = E0+300
    ])
    pumps = _write(tmp_path, "pumps", [{"name": "PS-T", "ward": 107, "source": "test"}],
                   [Point(E0 + 1000, N0)])
    water = _write(tmp_path, "water", [{"source": "test"}],
                   [Polygon([(E0 - 200, N0 - 200), (E0 - 100, N0 - 200), (E0 - 100, N0 - 100), (E0 - 200, N0 - 100)])])
    return SpatialService(drains, pumps, water, tmp_path / "none.csv",
                          drain_max_radius_m=500, pump_max_radius_m=3000, water_max_radius_m=1000)


# ------------------------------------------------------------ capacity proxy

def test_manning_capacity_matches_formula():
    d = 0.3
    expected = (1 / 0.013) * (math.pi * d * d / 4) * (d / 4) ** (2 / 3) * math.sqrt(0.001)
    assert cp.manning_full_pipe_capacity_m3s(300) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(0.0306, abs=5e-4)  # sanity: ~31 L/s


def test_manning_scales_with_d_to_8_3():
    q1, q2 = cp.manning_full_pipe_capacity_m3s(300), cp.manning_full_pipe_capacity_m3s(600)
    assert q2 / q1 == pytest.approx(2 ** (8 / 3), rel=1e-12)


@pytest.mark.parametrize("bad", [None, float("nan"), 0, -300, 50, 5000, "abc"])
def test_manning_rejects_unknown_or_implausible(bad):
    assert cp.manning_full_pipe_capacity_m3s(bad) is None


def test_capacity_fields_always_labelled():
    f = cp.capacity_fields(300, "pipe")
    assert f["drain_capacity_estimated_m3s"] == pytest.approx(0.0306, abs=5e-4)
    assert "ESTIMATED" in f["drain_capacity_note"] and "not measured" in f["drain_capacity_note"]
    assert cp.capacity_fields(None, "pipe")["drain_capacity_estimated_m3s"] is None
    assert cp.capacity_fields(1200, "box_sewer")["drain_capacity_estimated_m3s"] is None


# ------------------------------------------------------------ nearest drain

def test_nearest_drain_distance_and_diameter(svc):
    r = svc.find_nearest_drain(*ll(E0 + 50, N0 + 20))
    assert r["found"] and r["segment_id"] == "a"
    assert r["distance_to_drain_m"] == pytest.approx(20.0, abs=0.1)
    assert r["pipe_diameter_mm"] == 300
    assert r["drain_capacity_estimated_m3s"] == pytest.approx(cp.manning_full_pipe_capacity_m3s(300), abs=1e-4)
    assert r["within_mapped_area"] is True
    assert r["nearest_point"]["lat"] == pytest.approx(ll(E0 + 50, N0)[0], abs=1e-5)


def test_unknown_diameter_gives_no_capacity(svc):
    r = svc.find_nearest_drain(*ll(E0 + 50, N0 + 190))
    assert r["segment_id"] == "b" and r["pipe_diameter_mm"] is None
    assert r["drain_capacity_estimated_m3s"] is None


def test_box_sewer_gets_no_pipe_capacity(svc):
    r = svc.find_nearest_drain(*ll(E0 + 310, N0 + 100))
    assert r["segment_id"] == "c" and r["conduit_type"] == "box_sewer"
    assert r["drain_capacity_estimated_m3s"] is None


def test_far_point_is_no_coverage_not_fabricated(svc):
    r = svc.find_nearest_drain(*ll(E0 + 5000, N0 + 5000))
    assert r["found"] is False
    assert r["distance_to_drain_m"] is None and r["pipe_diameter_mm"] is None
    assert "no mapped drain within 500 m" in r["reason"]
    assert r["within_mapped_area"] is False


def test_just_outside_radius_is_no_coverage(tmp_path, svc):
    s = SpatialService(svc.drains.path, svc.pumps.path, svc.water.path, None, drain_max_radius_m=30)
    assert s.find_nearest_drain(*ll(E0 + 50, N0 + 29))["found"] is True
    assert s.find_nearest_drain(*ll(E0 + 50, N0 + 31))["found"] is False


def test_found_outside_mapped_area_is_flagged(svc):
    # 400 m below pipe a: inside radius, outside the ward footprint (hull + 50 m).
    r = svc.find_nearest_drain(*ll(E0 + 50, N0 - 400))
    assert r["found"] and r["within_mapped_area"] is False
    assert "upper bound" in r["note"]


def test_mapped_area_does_not_bridge_gaps_between_sheets(tmp_path):
    """Two sheets forming an 'L': the empty corner must NOT count as mapped
    (a convex hull would wrongly include it)."""
    rows, geoms = [], []
    for k in range(6):  # sheet A: horizontal pipes along the bottom
        rows.append({"segment_id": f"A{k}", "ward": 108, "source_sheet": "A", "conduit_type": "pipe",
                     "pipe_diameter_mm": 300, "diameter_source": "t", "source": "t"})
        geoms.append(LineString([(E0, N0 + 40 * k), (E0 + 1500, N0 + 40 * k)]))
    for k in range(6):  # sheet B: vertical pipes up the left side
        rows.append({"segment_id": f"B{k}", "ward": 108, "source_sheet": "B", "conduit_type": "pipe",
                     "pipe_diameter_mm": 300, "diameter_source": "t", "source": "t"})
        geoms.append(LineString([(E0 + 40 * k, N0 + 300), (E0 + 40 * k, N0 + 1500)]))
    p = _write(tmp_path, "L", rows, geoms)
    s = SpatialService(p, tmp_path / "x.geojson", tmp_path / "y.geojson", None, drain_max_radius_m=2000)
    corner = s.find_nearest_drain(*ll(E0 + 1000, N0 + 1000))
    assert corner["found"] is True                      # a pipe exists within 2 km...
    assert corner["within_mapped_area"] is False        # ...but the corner itself is unmapped
    assert s.find_nearest_drain(*ll(E0 + 700, N0 + 100))["within_mapped_area"] is True


@pytest.mark.parametrize("lat,lon", [(float("nan"), 88.4), (95.0, 88.4), (22.5, 200.0), ("x", 88.4)])
def test_invalid_coordinates(svc, lat, lon):
    r = svc.find_nearest_drain(lat, lon)
    assert r["found"] is False and r["reason"] == "invalid coordinates"


def test_batch_matches_single_and_keeps_order(svc):
    pts = [ll(E0 + 5000, N0), ll(E0 + 50, N0 + 20), (float("nan"), 0.0), ll(E0 + 310, N0 + 100)]
    batch = svc.find_nearest_drain_batch(pts)
    assert [b["segment_id"] for b in batch] == [None, "a", None, "c"]
    for p, b in zip(pts, batch):
        assert b == svc.find_nearest_drain(*p)


# ------------------------------------------------------ pumps / water / hist

def test_pumping_station(svc):
    r = svc.find_nearest_pumping_station(*ll(E0 + 400, N0))
    assert r["found"] and r["name"] == "PS-T"
    assert r["distance_to_pumping_station_m"] == pytest.approx(600.0, abs=0.1)
    assert svc.find_nearest_pumping_station(*ll(E0 + 9000, N0))["found"] is False


def test_waterbody_inside_and_near(svc):
    inside = svc.find_nearest_waterbody(*ll(E0 - 150, N0 - 150))
    assert inside["found"] and inside["inside_waterbody"] is True and inside["distance_to_waterbody_m"] == 0
    near = svc.find_nearest_waterbody(*ll(E0 - 50, N0 - 150))
    assert near["inside_waterbody"] is False
    assert near["distance_to_waterbody_m"] == pytest.approx(50.0, abs=0.1)


def test_historical_stub_returns_unknown_not_zero(svc):
    r = svc.find_historical_waterlogging(*ll(E0, N0))
    assert r["available"] is False
    assert r["historical_waterlogging"] is None and r["historical_event_count"] is None


def test_missing_layers_do_not_crash(tmp_path):
    s = SpatialService(tmp_path / "a.geojson", tmp_path / "b.geojson", tmp_path / "c.geojson", None)
    rep = s.status_report()
    assert rep["drainage_network"]["status"] == "MISSING"
    assert rep["pumping_stations"]["status"] == "MISSING"
    feats = s.get_spatial_features([ll(E0, N0)])[0]
    assert feats["drain"]["found"] is False and "missing" in feats["drain"]["reason"]
    assert feats["pumping_station"]["found"] is False
    assert feats["water_body"]["found"] is False


def test_wrong_crs_layer_reports_error(tmp_path):
    p = tmp_path / "utm.gpkg"
    gpd.GeoDataFrame({"segment_id": ["x"]}, geometry=[LineString([(E0, N0), (E0 + 10, N0)])], crs=UTM).to_file(p)
    s = SpatialService(p, tmp_path / "b.geojson", tmp_path / "c.geojson", None)
    assert s.status_report()["drainage_network"]["status"] == "ERROR"
    assert s.find_nearest_drain(*ll(E0, N0))["found"] is False


# ------------------------------------------------------------- real data

real = pytest.mark.skipif(not settings.DRAINAGE_GEOJSON_PATH.exists(), reason="GIS layers not built")


@pytest.fixture(scope="module")
def real_svc():
    return SpatialService(settings.DRAINAGE_GEOJSON_PATH, settings.PUMPING_STATIONS_GEOJSON_PATH,
                          settings.WATER_BODIES_GEOJSON_PATH, None,
                          drain_max_radius_m=500, pump_max_radius_m=3000, water_max_radius_m=1000)


@real
def test_real_point_on_known_segment(real_svc):
    g = real_svc.drains.gdf
    row = g[g["pipe_diameter_mm"].notna()].iloc[100]
    mid = row.geometry.interpolate(0.5, normalized=True)
    r = real_svc.find_nearest_drain(*ll(mid.x, mid.y))
    assert r["found"] and r["distance_to_drain_m"] < 1.0
    assert r["pipe_diameter_mm"] == int(row["pipe_diameter_mm"])
    assert r["within_mapped_area"] is True


@real
def test_real_sealdah_has_no_coverage_yet(real_svc):
    # Ward 36 isn't georeferenced yet; the nearest mapped ward is several km away.
    r = real_svc.find_nearest_drain(22.5675, 88.3700)
    assert r["found"] is False and r["distance_to_drain_m"] is None

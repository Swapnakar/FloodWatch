import csv
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from pyproj import Transformer
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_training_dataset as bt  # noqa: E402

from config import settings  # noqa: E402

TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
E0, N0 = 644_000.0, 2_491_000.0
TO_LL = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)


def ll(e, n):
    lon, lat = TO_LL.transform(e, n)
    return lat, lon


@pytest.fixture
def boundary():
    sq = Polygon([(E0 - 2000, N0 - 2000), (E0 + 2000, N0 - 2000),
                  (E0 + 2000, N0 + 2000), (E0 - 2000, N0 + 2000)])
    return gpd.GeoSeries([sq], crs="EPSG:32645").union_all()


def test_negatives_kept_away_from_positives(boundary):
    plat, plon = ll(E0, N0)
    positives = [{"point_id": "pos_0", "lat": plat, "lon": plon, "ward": "1",
                  "source": "kmc_2017_pocket", "label_source": "x"}]
    negs = bt.build_negatives(positives, boundary, seed=1)
    assert negs
    pos_utm = Point(E0, N0)
    for n in negs:
        x, y = TO_UTM.transform(n["lon"], n["lat"])
        assert pos_utm.distance(Point(x, y)) >= bt.NEG_MIN_DIST_M - 1e-6
        assert boundary.covers(Point(x, y))


def test_negatives_are_inside_boundary_only(boundary):
    positives = [{"point_id": "pos_0", "lat": ll(E0, N0)[0], "lon": ll(E0, N0)[1], "ward": "",
                  "source": "s", "label_source": "x"}]
    negs = bt.build_negatives(positives, boundary, seed=2)
    xs = [TO_UTM.transform(n["lon"], n["lat"]) for n in negs]
    assert all(boundary.covers(Point(x, y)) for x, y in xs)


# ---- real dataset assertions (require the built CSV) ------------------------

DATASET = bt.OUT


@pytest.fixture(scope="module")
def rows():
    if not DATASET.exists():
        pytest.skip("run scripts/build_training_dataset.py first")
    with DATASET.open() as f:
        return list(csv.DictReader(f))


def test_dataset_columns(rows):
    expected = set(bt.META_COLUMNS + bt.FEATURE_COLUMNS + [bt.LABEL])
    assert set(rows[0]) == expected


def test_label_is_binary_and_real(rows):
    labels = {r[bt.LABEL] for r in rows}
    assert labels == {"0", "1"}
    for r in rows:
        if r[bt.LABEL] == "1":
            assert r["label_source"] == "located_historical_pocket"
            assert r["source"] == "kmc_2017_pocket"
        else:
            assert r["label_source"] == "grid_no_nearby_pocket"


def test_class_balance_has_both_classes(rows):
    unique = {r["point_id"].rsplit("_", 1)[0] for r in rows}
    pos = {r["point_id"].rsplit("_", 1)[0] for r in rows if r[bt.LABEL] == "1"}
    assert len(pos) >= 1
    assert 0 < len(pos) < len(unique)  # not all one class


def test_terrain_features_present_everywhere(rows):
    for r in rows:
        assert r["elevation_m"] != "" and r["slope_percent"] != ""
        assert -50 <= float(r["elevation_m"]) <= 60


def test_missing_drainage_is_blank_not_zero(rows):
    # Invariant: no drain found => ALL drainage fields blank, never a fabricated 0.
    # (A point can be outside the mapped footprint yet still have a drain within
    # the 500 m search radius, flagged as an upper bound; that's real data.)
    no_drain = [r for r in rows if r["distance_to_drain_m"] == ""]
    assert no_drain, "expected rows with no mapped drain nearby"
    for r in no_drain:
        assert r["pipe_diameter_mm"] == ""
        assert r["drain_capacity_estimated_m3s"] == ""
    # and the majority of points have no drainage at all (option A: sparse feature)
    assert len(no_drain) > len(rows) * 0.5


def test_drainage_present_implies_all_drainage_fields(rows):
    for r in rows:
        if r["distance_to_drain_m"] != "":
            assert float(r["distance_to_drain_m"]) >= 0
            # capacity may be null (unknown diameter / box sewer), but if present it's positive
            if r["drain_capacity_estimated_m3s"] != "":
                assert float(r["drain_capacity_estimated_m3s"]) > 0


def test_rainfall_scenarios_expand_each_point(rows):
    scen = {r["rainfall_scenario"] for r in rows}
    assert scen == set(bt.RAINFALL_SCENARIOS)
    by_point = {}
    for r in rows:
        by_point.setdefault(r["point_id"].rsplit("_", 1)[0], set()).add(r["rainfall_scenario"])
    # every base point appears once per scenario
    assert all(v == set(bt.RAINFALL_SCENARIOS) for v in by_point.values())
    for r in rows:
        assert float(r["rainfall_scenario_mm"]) == bt.RAINFALL_SCENARIOS[r["rainfall_scenario"]]


def test_positives_lie_in_kmc(rows):
    b = gpd.read_file(settings.KMC_BOUNDARY_GEOJSON_PATH).to_crs("EPSG:32645").union_all()
    for r in rows:
        if r[bt.LABEL] == "1":
            x, y = TO_UTM.transform(float(r["lon"]), float(r["lat"]))
            assert b.buffer(50).covers(Point(x, y))

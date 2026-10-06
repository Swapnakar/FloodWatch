import csv
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from pyproj import Transformer
from shapely.geometry import Polygon

from config import settings
from services.historical_waterlogging import HistoricalWaterlogging
from services.spatial_service import SpatialService

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import geocode_waterlogging as gw  # noqa: E402

UTM = "EPSG:32645"
TO_LL = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
E0, N0 = 644_000.0, 2_491_000.0  # ~(22.52 N, 88.40 E), inside Kolkata
COLS = ["location_name", "ward", "latitude", "longitude", "source", "year", "severity", "notes"]


def ll(e, n):
    lon, lat = TO_LL.transform(e, n)
    return lat, lon


def _csv(tmp_path, rows, cols=COLS, name="wl.csv"):
    p = tmp_path / name
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return p


def _row(e, n, name="Test Lane", src="KMC test"):
    lat, lon = ll(e, n) if e is not None else ("", "")
    return {"location_name": name, "ward": "107", "latitude": lat, "longitude": lon, "source": src,
            "year": 2017, "severity": "listed_major_pocket", "notes": ""}


@pytest.fixture
def boundary(tmp_path):
    # 4 km square around the origin stands in for the KMC boundary.
    sq = Polygon([(E0 - 2000, N0 - 2000), (E0 + 2000, N0 - 2000), (E0 + 2000, N0 + 2000), (E0 - 2000, N0 + 2000)])
    p = tmp_path / "kmc.geojson"
    gpd.GeoDataFrame({"name": ["kmc"]}, geometry=[sq], crs=UTM).to_crs("EPSG:4326").to_file(p)
    return p


@pytest.fixture
def hist(tmp_path, boundary):
    p = _csv(tmp_path, [_row(E0, N0, "A"), _row(E0 + 100, N0, "B"), _row(E0 + 1500, N0, "C"),
                        _row(None, None, "Unlocated Lane")])
    return HistoricalWaterlogging(p, boundary, radius_m=250)


def test_point_within_radius_is_1(hist):
    r = hist.lookup(*ll(E0 + 50, N0 + 10))
    assert r["available"] is True
    assert r["historical_waterlogging"] == 1
    assert r["historical_event_count"] == 2  # A and B both within 250 m
    assert r["distance_to_historical_waterlogging_m"] == pytest.approx(51.0, abs=0.5)
    assert r["nearest_record"]["location_name"] in ("A", "B")


def test_point_far_away_is_0_with_distance(hist):
    r = hist.lookup(*ll(E0 - 1500, N0 + 1500))
    assert r["historical_waterlogging"] == 0
    assert r["historical_event_count"] == 0
    assert r["distance_to_historical_waterlogging_m"] is not None
    assert r["distance_to_historical_waterlogging_m"] > 250


def test_radius_edge(hist):
    assert hist.lookup(*ll(E0 - 249, N0))["historical_waterlogging"] == 1
    assert hist.lookup(*ll(E0 - 252, N0))["historical_waterlogging"] == 0


def test_outside_kmc_is_no_data_not_zero(hist):
    r = hist.lookup(*ll(E0 + 5000, N0))
    assert r["available"] is False
    assert r["historical_waterlogging"] is None and r["historical_event_count"] is None
    assert r["within_records_coverage"] is False
    assert "outside KMC" in r["reason"]


def test_unlocated_rows_counted_but_never_used(hist):
    s = hist.status_report()
    assert s["rows_total"] == 4 and s["rows_located"] == 3 and s["rows_unlocated"] == 1


def test_all_unlocated_gives_none_not_zero(tmp_path, boundary):
    h = HistoricalWaterlogging(_csv(tmp_path, [_row(None, None), _row(None, None)]), boundary)
    r = h.lookup(*ll(E0, N0))
    assert r["historical_waterlogging"] is None and r["available"] is False


def test_missing_csv_gives_none(tmp_path, boundary):
    h = HistoricalWaterlogging(tmp_path / "nope.csv", boundary)
    assert h.status_report()["status"] == "MISSING"
    assert h.lookup(*ll(E0, N0))["historical_waterlogging"] is None


def test_missing_columns_is_error(tmp_path, boundary):
    h = HistoricalWaterlogging(_csv(tmp_path, [_row(E0, N0)], cols=COLS[:4]), boundary)
    assert h.status_report()["status"] == "ERROR"
    assert "missing required columns" in h.status_report()["error"]


def test_rows_outside_kolkata_or_without_source_rejected(tmp_path, boundary):
    rows = [_row(E0, N0), {**_row(E0, N0), "latitude": 28.6, "longitude": 77.2}, _row(E0, N0, src="")]
    h = HistoricalWaterlogging(_csv(tmp_path, rows), boundary)
    s = h.status_report()
    assert s["rows_located"] == 1 and s["rows_rejected"] == 2


def test_no_boundary_warns(tmp_path):
    h = HistoricalWaterlogging(_csv(tmp_path, [_row(E0, N0)]), tmp_path / "none.geojson")
    r = h.lookup(*ll(E0 + 3000, N0))
    assert r["historical_waterlogging"] == 0 and "coverage_warning" in r


def test_invalid_coordinates(hist):
    assert hist.lookup(float("nan"), 88.4)["reason"] == "invalid coordinates"


def test_spatial_service_integration(tmp_path, boundary):
    p = _csv(tmp_path, [_row(E0, N0)])
    s = SpatialService(tmp_path / "d.geojson", tmp_path / "p.geojson", tmp_path / "w.geojson", p,
                       historical_radius_m=250, kmc_boundary_path=boundary)
    feats = s.get_spatial_features([ll(E0 + 10, N0), ll(E0 + 1000, N0)])
    assert [f["historical"]["historical_waterlogging"] for f in feats] == [1, 0]
    assert s.status_report()["historical_waterlogging"]["status"] == "LOADED"


# ------------------------------------------------------------ geocoder selection (offline)

class _B:  # boundary stub: covers everything
    def covers(self, _):
        return True


def _cand(lat, lon, half_deg=0.0005, name="x"):
    return {"lat": lat, "lon": lon, "bbox": (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg),
            "display_name": name}


def test_choose_accepts_single_precise_candidate():
    r = gw.choose([_cand(22.52, 88.39)], _B())
    assert r["status"] == "ok" and r["lat"] == 22.52


def test_choose_rejects_ambiguous():
    assert gw.choose([_cand(22.52, 88.39), _cand(22.58, 88.36)], _B())["status"] == "ambiguous"


def test_choose_rejects_vague_long_road():
    assert gw.choose([_cand(22.52, 88.39, half_deg=0.02)], _B())["status"] == "too_vague"


def test_choose_rejects_outside_kmc():
    class Out:
        def covers(self, _):
            return False
    assert gw.choose([_cand(22.52, 88.39)], Out())["status"] == "outside_kmc"
    assert gw.choose([], _B())["status"] == "not_found"


def test_choose_rejects_generic_street_name_match():
    # Real Mapbox behaviour: "Hossenpur 2nd Lane" returned an unrelated "2nd Lane".
    c = {**_cand(22.553, 88.403), "name": "2nd Lane", "kind": "street", "bbox": None}
    assert gw.choose([c], _B(), "Hossenpur 2nd Lane")["status"] == "name_mismatch"


def test_choose_accepts_transliteration_variant():
    c = {**_cand(22.5105, 88.4045), "name": "Hossainpur 2nd Lane", "kind": "street", "bbox": None}
    r = gw.choose([c], _B(), "Hossenpur 2nd Lane")
    assert r["status"] == "ok" and r["method_suffix"] == "street_centroid"


def test_choose_rejects_different_street_type():
    # Real case: "Ripon Lane" came back as Ripon Street; "Alipore Park Place" as Alipore Park Road.
    c = {**_cand(22.553, 88.354), "name": "Ripon Street", "kind": "street", "bbox": None}
    assert gw.choose([c], _B(), "Ripon Lane")["status"] == "name_mismatch"
    c2 = {**_cand(22.53, 88.33), "name": "Alipore Park Road", "kind": "street", "bbox": None}
    assert gw.choose([c2], _B(), "Alipore Park Place")["status"] == "name_mismatch"
    c3 = {**_cand(22.55, 88.35), "name": "Ripon Ln", "kind": "street", "bbox": None}
    assert gw.choose([c3], _B(), "Ripon Lane")["status"] == "ok"  # abbreviation is the same type


def _r(lat, lon, ward="40", borough="5"):
    return {"latitude": lat, "longitude": lon, "ward": ward, "borough": borough, "geocode_method": "x"}


def test_consistency_outlier_is_removed():
    rows = [_r(22.580, 88.360), _r(22.581, 88.361), _r(22.579, 88.362), _r(22.470, 88.300)]  # last: ~13 km off
    out = gw.consistency_outliers(rows)
    assert list(out) == [3] and "ward 40" in out[3]


def test_consistency_majority_wins_with_three_points():
    # Real ward 128 case: two Behala points agree, one north-Kolkata point is wrong.
    rows = [_r(22.625458, 88.39033, "128", "14"), _r(22.488561, 88.293703, "128", "14"),
            _r(22.492746, 88.293884, "128", "14")]
    assert list(gw.consistency_outliers(rows)) == [0]


def test_consistency_tie_sends_both_to_review():
    rows = [_r(22.58, 88.36, "12", "2"), _r(22.47, 88.30, "12", "2")]
    assert sorted(gw.consistency_outliers(rows)) == [0, 1]


def test_consistency_single_point_left_alone():
    assert gw.consistency_outliers([_r(22.58, 88.36, "12", "2")]) == {}


def test_choose_rejects_query_without_distinctive_name():
    assert gw.choose([_cand(22.52, 88.39)], _B(), "East Park")["status"] == "too_generic"
    assert gw.choose([_cand(22.52, 88.39)], _B(), "Village Road")["status"] == "too_generic"


def test_choose_rejects_city_level_feature():
    c = {**_cand(22.5568, 88.3547, half_deg=0.0001), "name": "Martinpara Kolkata", "kind": "place"}
    assert gw.choose([c], _B(), "Martinpara")["status"] == "too_vague"


def test_choose_neighbourhood_with_small_bbox_is_auto():
    c = {**_cand(22.51, 88.40), "name": "Martin Para", "kind": "neighborhood"}
    r = gw.choose([c], _B(), "Martinpara")
    assert r["status"] == "ok" and r["method_suffix"] == "auto"


@pytest.mark.parametrize("text,expected", [
    ("Akhil Mistri Ln. (from Pr-1 to 124)", "Akhil Mistri Lane"),
    ("Padma Pukur near Sukanta Park", "Padma Pukur"),
    ("Santosh Roy Road and James Long crossing towards Motilal Gupta Road", "Santosh Roy Road"),
    ("2A, 2B, 2C Chatu Babu Lane", "Chatu Babu Lane"),
    ("Elliot Lane (portion)", "Elliot Lane"),
    ("Muktaram Babu St.", "Muktaram Babu Street"),
    ("46, Middle Road (Bye Lane)", "Middle Road"),
    ("4,5 Netaji Nagar", "Netaji Nagar"),
    ("22 Bigha", "22 Bigha"),
    ("64 Pally", "64 Pally"),
    ("8 No. Sahid Nagar", "8 No. Sahid Nagar"),
])
def test_clean_query(text, expected):
    assert gw.clean_query(text) == expected


# ------------------------------------------------------------ real source PDF

PDF = settings.HISTORICAL_WATERLOGGING_PATH.parent / "water_logging_09_06_2017.pdf"


@pytest.mark.skipif(not PDF.exists(), reason="KMC 2017 action plan PDF not present")
def test_real_pdf_parses_all_boroughs_with_unbroken_numbering():
    import parse_kmc_waterlogging_pdf as pk
    rows = pk.parse(PDF)
    errors, counts = pk.check(rows)
    assert errors == []
    assert len(counts) == 16 and sum(counts.values()) == 347
    by = {(r["borough"], r["sn"]): r for r in rows}
    assert by[(1, 1)]["wards"] == [1] and by[(1, 1)]["location_text"] == "Gobinda Mondal Lane"
    assert by[(6, 15)]["wards"] == [61] and by[(6, 15)]["location_text"] == "Elliot Lane (portion)"
    assert by[(12, 12)]["wards"] == [108]
    assert by[(15, 9)]["wards"] == [138, 141]
    assert by[(16, 79)]["location_text"] == "Hanspukur"
    # wrapped multi-line rows are joined, not split into bogus rows
    assert by[(6, 2)]["location_text"] == "Jn. Of Jawaharlal Nehru Rd. and S. N. Banerje Road (near Big Bazar)"

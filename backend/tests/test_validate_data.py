"""validate_data.py must pass clean data and fail loudly, naming rows, on bad data."""

import json
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import validate_data as vd  # noqa: E402

LON, LAT = 88.39, 22.52  # inside ward 107/108 area


def _line(i, dx=0.001):
    return LineString([(LON + i * 0.0005, LAT), (LON + i * 0.0005 + dx, LAT + 0.0005)])


def _drainage(n=5, overrides=None):
    rows = []
    for i in range(n):
        r = {"segment_id": f"s{i}", "ward": 107, "source": "test", "conduit_type": "pipe",
             "pipe_diameter_mm": 300, "diameter_source": "legend_colour",
             "extraction_method": "test", "geometry": _line(i)}
        rows.append(r)
    for (i, col), val in (overrides or {}).items():
        rows[i][col] = val
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _spec(path, name="drainage_network"):
    for s in vd.default_specs():
        if s.name == name:
            s.path = path
            return s
    raise KeyError(name)


def _write(gdf, tmp_path, name="d.geojson"):
    p = tmp_path / name
    gdf.to_file(p, driver="GeoJSON")
    return p


def test_clean_layer_passes(tmp_path):
    r = vd.validate_layer(_spec(_write(_drainage(), tmp_path)))
    assert r.ok, r.errors
    assert r.rows == 5


def test_missing_file_fails(tmp_path):
    r = vd.validate_layer(_spec(tmp_path / "nope.geojson"))
    assert not r.ok and "file missing" in r.errors[0]


def test_projected_coordinates_fail_even_if_labelled_4326(tmp_path):
    # UTM metres written without a CRS member: GDAL reads it back as 4326.
    g = _drainage().to_crs("EPSG:32645")
    p = tmp_path / "utm.geojson"
    fc = json.loads(g.to_json())
    fc.pop("crs", None)
    p.write_text(json.dumps(fc))
    r = vd.validate_layer(_spec(p))
    assert not r.ok
    assert any("outside Kolkata bbox" in e for e in r.errors)


def test_wrong_declared_crs_fails(tmp_path):
    g = _drainage().to_crs("EPSG:32645")
    p = tmp_path / "utm.gpkg"
    g.to_file(p, driver="GPKG")
    r = vd.validate_layer(_spec(p))
    assert any("expected EPSG:4326" in e for e in r.errors)


def test_lat_lon_swapped_fails(tmp_path):
    g = _drainage()
    g.loc[2, "geometry"] = LineString([(LAT, LON), (LAT + 0.001, LON + 0.001)])
    r = vd.validate_layer(_spec(_write(g, tmp_path)))
    msg = next(e for e in r.errors if "outside Kolkata" in e)
    assert "2 (s2)" in msg  # names the bad row


def test_invalid_polygon_is_listed(tmp_path):
    bowtie = Polygon([(LON, LAT), (LON + .001, LAT + .001), (LON + .001, LAT), (LON, LAT + .001)])
    ok = Polygon([(LON, LAT), (LON + .001, LAT), (LON + .001, LAT + .001), (LON, LAT + .001)])
    g = gpd.GeoDataFrame({"source": ["t", "t"]}, geometry=[ok, bowtie], crs="EPSG:4326")
    r = vd.validate_layer(_spec(_write(g, tmp_path, "w.geojson"), "water_bodies"))
    msg = next(e for e in r.errors if "invalid geometry" in e)
    assert "Self-intersection" in msg and msg.endswith(": 1")


def test_wrong_geometry_type_fails(tmp_path):
    g = _drainage()
    g.loc[0, "geometry"] = Point(LON, LAT)
    r = vd.validate_layer(_spec(_write(g, tmp_path)))
    assert any("unexpected geometry types ['Point']" in e for e in r.errors)


def test_missing_required_column_fails(tmp_path):
    g = _drainage().drop(columns=["source"])
    r = vd.validate_layer(_spec(_write(g, tmp_path)))
    assert any("missing required columns: ['source']" in e for e in r.errors)


def test_null_required_value_fails(tmp_path):
    r = vd.validate_layer(_spec(_write(_drainage(overrides={(3, "ward"): None}), tmp_path)))
    assert any("null 'ward' in 1 rows: 3 (s3)" in e for e in r.errors)


def test_unknown_diameter_is_allowed_but_reported(tmp_path):
    g = _drainage(overrides={(1, "pipe_diameter_mm"): None, (1, "diameter_source"): None})
    r = vd.validate_layer(_spec(_write(g, tmp_path)))
    assert r.ok, r.errors
    assert r.stats["pipes_unknown_diameter"] == 1


def test_diameter_without_source_fails(tmp_path):
    r = vd.validate_layer(_spec(_write(_drainage(overrides={(0, "diameter_source"): None}), tmp_path)))
    assert any("diameter without diameter_source" in e for e in r.errors)


def test_implausible_diameter_fails(tmp_path):
    r = vd.validate_layer(_spec(_write(_drainage(overrides={(0, "pipe_diameter_mm"): 12}), tmp_path)))
    assert any("outside 100-3000" in e for e in r.errors)


def test_duplicates_fail(tmp_path):
    g = _drainage(overrides={(4, "segment_id"): "s0", (3, "geometry"): _line(2)})
    r = vd.validate_layer(_spec(_write(g, tmp_path)))
    assert any("duplicate 'segment_id'" in e for e in r.errors)
    assert any("exact duplicate geometries in 1 rows: 3" in e for e in r.errors)


def test_degenerate_line_fails(tmp_path):
    g = _drainage(overrides={(0, "geometry"): LineString([(LON, LAT), (LON + 1e-6, LAT)])})
    r = vd.validate_layer(_spec(_write(g, tmp_path)))
    assert any("shorter than 1.0 m" in e for e in r.errors)


def test_bad_ward_fails(tmp_path):
    r = vd.validate_layer(_spec(_write(_drainage(overrides={(0, "ward"): 999}), tmp_path)))
    assert any("ward outside KMC range" in e for e in r.errors)


def test_cli_exit_code_nonzero_on_failure(tmp_path, monkeypatch):
    bad = _spec(tmp_path / "missing.geojson")
    monkeypatch.setattr(vd, "default_specs", lambda: [bad])
    assert vd.main([]) == 1


REAL = vd.default_specs()


@pytest.mark.skipif(not all(s.path.exists() for s in REAL), reason="GIS layers not built yet")
def test_real_gis_layers_pass():
    results = vd.validate_all()
    assert all(r.ok for r in results), {r.name: r.errors for r in results}


@pytest.mark.skipif(not REAL[0].path.exists(), reason="drainage layer not built yet")
def test_corrupted_copy_of_real_drainage_fails_loudly(tmp_path):
    """Plan demo: corrupt a COPY of the real layer; each injected fault must be named.

    The real file is only read, never modified.
    """
    real = REAL[0].path
    before = real.read_bytes()
    g = gpd.read_file(real)
    ids = g["segment_id"]
    g.loc[10, "geometry"] = LineString([(22.52, 88.39), (22.521, 88.391)])  # lat/lon swapped
    g.loc[20, "ward"] = None                                               # missing ward
    g.loc[30, "pipe_diameter_mm"] = 9                                      # implausible diameter
    g.loc[40, "segment_id"] = ids[41]                                      # duplicate id
    r = vd.validate_layer(_spec(_write(g, tmp_path, "corrupt.geojson")))

    assert not r.ok
    errs = "\n".join(r.errors)
    assert "outside Kolkata bbox" in errs and f"10 ({ids[10]})" in errs
    assert f"null 'ward' in 1 rows: 20 ({ids[20]})" in errs
    assert f"outside 100-3000 in 1 rows: 30 ({ids[30]})" in errs
    assert f"duplicate 'segment_id' in 2 rows: 40 ({ids[41]}), 41 ({ids[41]})" in errs
    assert len(r.errors) == 4, r.errors  # nothing else in the real data is flagged
    assert real.read_bytes() == before

"""Coordinate-convention tests for the AutoCAD sheet GCP pipeline (no network)."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import extract_kmc_cad as cad  # noqa: E402

UTM = "EPSG:32645"
TO_LL = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
TO_3857 = Transformer.from_crs(UTM, "EPSG:3857", always_xy=True)


@pytest.mark.parametrize("rot", [0, 90, 180, 270])
def test_display_mapping_roundtrip(rot):
    W, H = 2384.0, 3370.0
    for x, y in [(0, 0), (W, H), (100.5, 2000.25), (W, 0)]:
        u, v = cad.page_to_display(x, y, W, H, rot)
        DW, DH = cad.display_size(W, H, rot)
        assert -1e-9 <= u <= DW + 1e-9 and -1e-9 <= v <= DH + 1e-9
        assert cad.display_to_page(u, v, W, H, rot) == pytest.approx((x, y))


def test_rotate_270_puts_content_plus_x_up():
    # /Rotate 270: content +x must display as "up" (north arrow check on sheets 36/104).
    W, H = 2384.0, 3370.0
    _, v0 = cad.page_to_display(100, 500, W, H, 270)
    _, v1 = cad.page_to_display(200, 500, W, H, 270)
    assert v1 > v0


def test_fit_similarity_recovers_known_transform():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 3000, (7, 2))
    s, th, t = 0.85, math.radians(92.0), np.array([590000.0, 2489000.0])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    dst = (s * (R @ src.T)).T + t
    s2, R2, t2 = cad.fit_similarity(src, dst)
    assert s2 == pytest.approx(s, rel=1e-9)
    assert np.allclose(R2, R) and np.allclose(t2, t)


def _write_points(path, px, utm, crs="4326", header=True):
    if crs == "4326":
        mx, my = TO_LL.transform(utm[:, 0], utm[:, 1])
        wkt = 'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563]],CS[ellipsoidal,2],AXIS["latitude",north],AXIS["longitude",east],ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",4326]]'
    else:
        mx, my = TO_3857.transform(utm[:, 0], utm[:, 1])
        wkt = "EPSG:3857"
    lines = [f"#CRS: {wkt}"] if header else []
    lines.append("mapX,mapY,sourceX,sourceY,enable,dX,dY,residual")
    for (X, Y), a, b in zip(px, mx, my):
        lines.append(f"{a},{b},{X},{-Y},1,0,0,0")  # QGIS stores pixel rows as negative Y
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("crs,header", [("4326", True), ("3857", True), ("3857", False)])
def test_read_points_formats(tmp_path, crs, header):
    px = np.array([[10.0, 20.0], [3000.0, 4000.0], [5000.0, 100.0]])
    utm = np.array([[590000.0, 2489000.0], [591000.0, 2488000.0], [590500.0, 2490000.0]])
    p = tmp_path / "w.points"
    _write_points(p, px, utm, crs=crs, header=header)
    got_px, got_utm, _ = cad.read_points(p)
    assert np.allclose(got_px, px)
    assert np.allclose(got_utm, utm, atol=0.01)


PREPARED = cad.CAD_DIR / "ward104_page.json"


@pytest.mark.skipif(not PREPARED.exists(), reason="run extract_kmc_cad.py prepare first")
@pytest.mark.parametrize("rot_offset_deg,expect", [(0.0, "ACCEPTED"), (25.0, "REJECTED")])
def test_apply_end_to_end_with_synthetic_gcps(tmp_path, monkeypatch, rot_offset_deg, expect):
    """Synthetic GCPs from a known transform must be recovered exactly and pass
    the gates; a transform contradicting the north arrow must be rejected."""
    page = json.loads(PREPARED.read_text())
    rep = page["report"]
    W, H = rep["page_size_pt"]
    rot = rep["page_rotate"]
    k = rep["render"]["px_per_pt"]
    DW, DH = cad.display_size(W, H, rot)
    s = rep["priors"]["m_per_pt_from_area"]
    th = math.radians(rep["priors"]["rotation_deg_from_north_arrow"] + rot_offset_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = np.array([591500.0, 2487900.0])

    page_pts = np.array([[400, 500], [2000, 600], [1900, 3000], [500, 2900], [1200, 1700], [800, 1200]], float)
    utm = (s * (R @ page_pts.T)).T + t
    px = []
    for x, y in page_pts:
        u, v = cad.page_to_display(x, y, W, H, rot)
        px.append((u * k, (DH - v) * k))
    gcp_dir = tmp_path / "gcp"
    gcp_dir.mkdir()
    _write_points(gcp_dir / "ward104.points", np.array(px), utm)

    monkeypatch.setattr(cad, "GCP_DIR", gcp_dir)
    monkeypatch.setattr(cad, "OUT", tmp_path)
    monkeypatch.setattr(cad.merge_kmc_layers, "osm_road_distance", lambda gdf: {"skipped": "test"})
    res = cad.apply_sheet("104_kalikapur.pdf", cad.SHEETS["104_kalikapur.pdf"])

    assert res["status"] == expect, res
    assert res["rms_m"] < 0.05
    assert res["m_per_pt"] == pytest.approx(s, rel=1e-6)
    if expect == "ACCEPTED":
        import geopandas as gpd
        g = gpd.read_file(tmp_path / "ward104_cad_pipes.geojson").to_crs(UTM)
        # First vertex of first segment lands exactly where the known transform puts it.
        x0, y0 = page["segments"][0]["page_pts"][0]
        want = s * (R @ np.array([x0, y0])) + t
        got = np.array(g.geometry.iloc[0].coords[0])
        assert np.hypot(*(got - want)) < 0.05
    else:
        assert any("north arrow" in e for e in res["errors"])
        assert not (tmp_path / "ward104_cad_pipes.geojson").exists()


def test_apply_waits_when_no_gcp_file(tmp_path, monkeypatch):
    if not PREPARED.exists():
        pytest.skip("run extract_kmc_cad.py prepare first")
    monkeypatch.setattr(cad, "GCP_DIR", tmp_path)
    res = cad.apply_sheet("104_kalikapur.pdf", cad.SHEETS["104_kalikapur.pdf"])
    assert res["status"] == "WAITING_FOR_GCPS"

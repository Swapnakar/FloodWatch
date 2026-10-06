"""
KMC AutoCAD ward sheets -> drainage GeoJSON, georeferenced by ground control points.

Two steps:

  prepare   Rebuild pipes from dashes (cad_pipes.py) and assign diameters, all in
            PDF page coordinates. Write a page-space JSON per sheet, plus a
            reference PNG with a verified page<->pixel mapping. You open the PNG
            in the QGIS Georeferencer to pick control points (see
            docs/KMC_CAD_GEOREFERENCING.md).

  apply     Read each sheet's QGIS .points file and fit a similarity transform
            (scale + rotation + translation, the right model for a CAD drawing).
            Check the fit against the sheet's own evidence: printed ward area
            -> scale, north arrow -> rotation. Also report leave-one-out error
            and distance to OSM roads, then write per-sheet GeoJSON and rebuild
            the merged data/gis layers.

Usage (from backend/):
    ../.venv/bin/python scripts/extract_kmc_cad.py prepare
    ../.venv/bin/python scripts/extract_kmc_cad.py apply
"""

import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from pypdf import PdfReader
from pyproj import CRS, Transformer
from scipy import ndimage
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cad_pipes import assemble_labels, assign_diameters, chain_dashes, resolve_chains  # noqa: E402
from pdf_vectors import read_page  # noqa: E402
from config import DATA_DIR  # noqa: E402
import merge_kmc_layers  # noqa: E402

RAW = DATA_DIR / "raw"
OUT = DATA_DIR / "processed" / "kmc_extracted"
CAD_DIR = OUT / "cad"
RENDER_DIR = CAD_DIR / "render"
GCP_DIR = CAD_DIR / "gcp"
UTM = "EPSG:32645"
TO_WGS84 = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
ACRE_M2 = 4046.8564224
RENDER_PX = 6000  # long side of the reference PNG

# Per-sheet layer semantics, established by inspecting the sheets:
#  generic_pipe_layers: pipes of ALL sizes share the layer, so diameter comes
#      only from map labels (e.g. Sealdah layer "225" carries 300/375/400 mm labels).
#  diameter_layers: layer name IS the diameter (Park Street: layer name
#      agreed with the printed label on 23/23 checked pipes).
SHEETS = {
    "036_Sealdah Station.pdf": {
        "ward": 36, "boundary_layer": "Z WARD BOUNDARY",
        "generic_pipe_layers": ["225", "300"],
        "diameter_layers": {"300 DIA": 300, "400 DIA": 400, "450 DIA": 450},
        "drain_layers": ["Z DRAIN"],
        "decode": None,
    },
    "104_kalikapur.pdf": {
        "ward": 104, "boundary_layer": "Z WARD BOUNDARY",
        "generic_pipe_layers": ["225"],
        "diameter_layers": {}, "drain_layers": [],
        "decode": None,
    },
    "061_Park Street.pdf": {
        "ward": 61, "boundary_layer": "ADMIN_BOUNDARY",
        "generic_pipe_layers": [],
        "diameter_layers": {"150": 150, "225": 225, "300": 300, "375": 375, "450": 450},
        # Manhole-to-manhole trunk/brick sewers are drawn in red on ROAD_NETWORK
        # (confirmed on the QA overlay: MH-34..MH-37, MH-17..MH-19). Only the red
        # strokes on that layer are sewers; its green/black strokes are roads.
        "styled_pipe_layers": [["ROAD_NETWORK", [1.0, 0.0, 0.0]]],
        "drain_layers": [],
        # Font without ToUnicode: control chars are ASCII shifted by -29, 0x91 is Ø.
        "decode": "shift29",
    },
    # Ward 67: TRUNK SEWERS ONLY. The red strokes on ROAD_NETWORK are the
    # existing trunk lines (e.g. the "EXISTING 1800 MM.Ø PIPE SEWER LINE" on Bose
    # Pukur Road). Branch sewers are green dashes indistinguishable by style from
    # the green road edges, sized in inches, and are NOT extracted.
    "067_kasba1.pdf": {
        "ward": 67, "boundary_layer": "ADMIN_BOUNDARY",
        "generic_pipe_layers": [], "diameter_layers": {},
        "styled_pipe_layers": [["ROAD_NETWORK", [1.0, 0.0, 0.0]]],
        "drain_layers": [],
        "decode": "shift29",
        "coverage_note": "trunk sewers only; branch sewer network not extracted",
    },
}

# Quality gates for the GCP fit (fixed before seeing any GCPs).
MIN_GCPS = 4
MAX_RMS_M = 5.0
MAX_LOO_M = 10.0
MAX_SCALE_VS_AREA = 0.10
MAX_ROT_VS_NORTH_DEG = 10.0
LABEL_MAX_DIST_PT = 30.0


def _decoder(kind):
    if kind == "shift29":
        return lambda s: "".join(chr(ord(c) + 29) if ord(c) < 0x20 else ("Ø" if c == "\x91" else c) for c in s)
    return None


# ------------------------------------------------------------ page <-> pixel

def page_to_display(x, y, W, H, rot):
    """Content coords -> displayed page coords (u right, v up) for /Rotate (clockwise)."""
    rot %= 360
    if rot == 0:
        return x, y
    if rot == 90:
        return y, W - x
    if rot == 180:
        return W - x, H - y
    if rot == 270:
        return H - y, x
    raise ValueError(rot)


def display_to_page(u, v, W, H, rot):
    rot %= 360
    if rot == 0:
        return u, v
    if rot == 90:
        return W - v, u
    if rot == 180:
        return W - u, H - v
    if rot == 270:
        return v, H - u
    raise ValueError(rot)


def display_size(W, H, rot):
    return (H, W) if rot % 180 else (W, H)


# ---------------------------------------------------------------- priors

def sheet_priors(pv, boundary_layer, rot):
    """Scale from printed ward area; rotation from 'north = displayed up'."""
    from PIL import ImageDraw
    acres = None
    for t in pv.texts:
        m = re.search(r"AREA\s*[:=\-]*\s*([\d.]+)\s*ACRE", t.text, re.I)
        if m:
            acres = float(m.group(1))
    W, H = int(pv.page_size[0]), int(pv.page_size[1])
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for p in pv.paths:
        if p.layer == boundary_layer:
            for sp in p.subpaths:
                d.line([(x, H - y) for x, y in sp], fill=255, width=2)
    a = np.array(img) > 0
    filled = ndimage.binary_fill_holes(ndimage.binary_dilation(a, iterations=12))
    inner = ndimage.binary_erosion(filled, iterations=12)
    lab, n = ndimage.label(inner)
    area_pt2 = float(ndimage.sum(inner, lab, range(1, n + 1)).max()) if n else 0.0
    m_per_pt = math.sqrt(acres * ACRE_M2 / area_pt2) if acres and area_pt2 else None
    # Content direction displayed as "up" has angle 90+rot (deg, CCW from +x).
    north_rot_deg = (90 - (90 + rot)) % 360  # rotation content->UTM putting that axis north
    return {"printed_area_acres": acres, "boundary_area_pt2": area_pt2,
            "m_per_pt_from_area": m_per_pt, "rotation_deg_from_north_arrow": north_rot_deg}


# ---------------------------------------------------------------- prepare

def auto_gap(dashes):
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    ends = [(sp[0], i) for i, (sp, _, _) in enumerate(dashes)] + [(sp[-1], i) for i, (sp, _, _) in enumerate(dashes)]
    pts = [Point(e[0]) for e in ends]
    tree = STRtree(pts)
    gaps = []
    for k, (xy, i) in enumerate(ends):
        ds = [pts[j].distance(pts[k]) for j in tree.query(pts[k].buffer(10)) if ends[j][1] != i]
        if ds:
            gaps.append(min(ds))
    return min(6.0, 1.25 * float(np.percentile(gaps, 95))) if gaps else 4.0


def prepare_sheet(name, cfg):
    reader = PdfReader(str(RAW / name))
    rot = int(reader.pages[0].get("/Rotate", 0) or 0) % 360
    pv = read_page(reader, 0)
    W, H = pv.page_size
    rep = {"sheet": name, "ward": cfg["ward"], "page_rotate": rot, "warnings": [],
           "coverage_note": cfg.get("coverage_note", "full sewer layer(s) extracted")}
    if pv.skipped_form_xobjects:
        rep["warnings"].append(f"{pv.skipped_form_xobjects} Form XObjects not parsed")

    groups = []  # (layers, kind, stroke filter or None)
    if cfg["generic_pipe_layers"]:
        groups.append((cfg["generic_pipe_layers"], "generic", None))
    if cfg["diameter_layers"]:
        groups.append((list(cfg["diameter_layers"]), "diameter_layer", None))
    for layer, rgb in cfg.get("styled_pipe_layers", []):
        groups.append(([layer], "generic_styled", tuple(float(v) for v in rgb)))
    if cfg["drain_layers"]:
        groups.append((cfg["drain_layers"], "drain", None))

    labels = assemble_labels([t for t in pv.texts if t.layer == "dia text"], decode=_decoder(cfg["decode"]))
    rep["labels"] = {"assembled": len(labels),
                     "by_diameter": dict(Counter(l.diameter_mm for l in labels).most_common())}

    all_chains = []
    for layers, kind, stroke_filter in groups:
        dashes = [(sp, p.layer, (p.stroke, round(p.width, 2))) for p in pv.paths
                  if p.layer in layers and p.stroked and not p.filled
                  and (stroke_filter is None or p.stroke == stroke_filter)
                  for sp in p.subpaths if len(sp) > 1]
        if not dashes:
            rep["warnings"].append(f"no strokes on layers {layers}")
            continue
        gap = auto_gap(dashes)
        chains = chain_dashes(dashes, max_gap=gap)
        for c in chains:
            c.kind = kind
        rep.setdefault("chaining", {})[kind] = {
            "layers": layers, "dashes": len(dashes), "max_gap_pt": round(gap, 2),
            "chains": len(chains), "single_dash_chains": sum(c.n_dashes == 1 for c in chains)}
        all_chains += chains

    # Labels are matched against every chain so the ambiguity check sees all pipes.
    rep["label_assignment"] = assign_diameters(all_chains, labels, max_dist=LABEL_MAX_DIST_PT)

    out, layer_checks = [], Counter()
    for c in resolve_chains(all_chains):
        kind = c.kind
        layer_dia = cfg["diameter_layers"].get(c.layer) if kind == "diameter_layer" else None
        label_dia = c.diameter_mm
        if layer_dia is not None:
            dia, src = layer_dia, "layer_name"
            if label_dia is not None:
                layer_checks["agree" if label_dia == layer_dia else "disagree"] += 1
        else:
            dia, src = label_dia, (c.diameter_source if label_dia is not None else None)
        length = sum(math.dist(c.pts[i], c.pts[i + 1]) for i in range(len(c.pts) - 1))
        if length < 5.0:  # isolated dash fragments / symbol bits
            continue
        out.append({
            "page_pts": [[round(x, 3), round(y, 3)] for x, y in c.pts],
            "source_layer": c.layer,
            "conduit_type": "drain" if kind == "drain" else "pipe",
            "pipe_diameter_mm": dia, "diameter_source": src,
            "label_diameter_mm": label_dia, "label_conflict_split": bool(c.label_conflict),
            "length_pt": round(length, 1),
        })
    tot = sum(o["length_pt"] for o in out)
    withd = sum(o["length_pt"] for o in out if o["pipe_diameter_mm"] is not None)
    rep["result"] = {
        "segments": len(out), "total_length_pt": round(tot),
        "length_with_diameter": round(withd / tot, 3) if tot else None,
        "by_source_layer": dict(Counter(o["source_layer"] for o in out)),
        "layer_name_vs_label": dict(layer_checks),
    }
    rep["priors"] = sheet_priors(pv, cfg["boundary_layer"], rot)
    rep["page_size_pt"] = [W, H]

    # Reference render + verified mapping
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    png = RENDER_DIR / f"ward{cfg['ward']:03d}.png"
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["qlmanage", "-t", "-s", str(RENDER_PX), "-o", td, str(RAW / name)],
                       check=True, capture_output=True)
        Path(td, name + ".png").replace(png)
    im = Image.open(png)
    DW, DH = display_size(W, H, rot)
    k = im.size[0] / DW
    if abs(im.size[1] / DH - k) > 0.01 * k:
        raise ValueError(f"{name}: render aspect mismatch {im.size} vs {DW}x{DH}")
    rep["render"] = {"png": str(png.relative_to(DATA_DIR.parent)), "px_per_pt": k, "size": im.size,
                     "mapping_check": verify_render(im, pv, cfg["boundary_layer"], W, H, rot, k)}
    if rep["render"]["mapping_check"]["hit_rate"] < 0.8:
        raise ValueError(f"{name}: render mapping check failed {rep['render']['mapping_check']}")

    CAD_DIR.mkdir(parents=True, exist_ok=True)
    (CAD_DIR / f"ward{cfg['ward']:03d}_page.json").write_text(json.dumps({"report": rep, "segments": out}))
    return rep


def verify_render(im, pv, boundary_layer, W, H, rot, k):
    """Project ward-boundary vectors to pixels; they should land on magenta pixels."""
    arr = np.asarray(im.convert("RGB")).astype(int)
    mag = (arr[:, :, 0] > 170) & (arr[:, :, 1] < 110) & (arr[:, :, 2] > 170)
    mag = ndimage.binary_dilation(mag, iterations=2)
    DW, DH = display_size(W, H, rot)
    pts = [pt for p in pv.paths if p.layer == boundary_layer for sp in p.subpaths for pt in sp]
    hits = total = 0
    for x, y in pts[:: max(1, len(pts) // 3000)]:
        u, v = page_to_display(x, y, W, H, rot)
        X, Y = int(u * k), int((DH - v) * k)
        if 0 <= Y < mag.shape[0] and 0 <= X < mag.shape[1]:
            total += 1
            hits += bool(mag[Y, X])
    return {"points": total, "hit_rate": round(hits / total, 3) if total else 0.0}


# ------------------------------------------------------------------ apply

def read_points(path):
    """QGIS Georeferencer .points file -> (pixel XY, map XY in UTM)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    crs = None
    if lines and lines[0].startswith("#CRS:"):
        wkt = lines[0][5:].strip()
        crs = CRS.from_user_input(wkt) if wkt else None
        lines = lines[1:]
    rows = list(csv.DictReader(lines))
    if crs is None:
        # Older QGIS writes no CRS line. Values beyond +-180 can't be degrees.
        big = any(abs(float(r["mapX"])) > 180 or abs(float(r["mapY"])) > 90 for r in rows)
        crs = CRS.from_epsg(3857 if big else 4326)
    to_utm = Transformer.from_crs(crs, UTM, always_xy=True)
    px, mp = [], []
    for r in rows:
        if str(r.get("enable", "1")).strip() in ("0", "false"):
            continue
        sx, sy = float(r["sourceX"]), float(r["sourceY"])
        mx, my = to_utm.transform(float(r["mapX"]), float(r["mapY"]))
        px.append((sx, -sy if sy < 0 else sy))  # QGIS stores pixel rows as negative Y
        mp.append((mx, my))
    return np.array(px), np.array(mp), crs.to_string()


def fit_similarity(src, dst):
    """Umeyama least squares: dst ~ s R src + t."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    cov = b.T @ a / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    t = md - s * R @ ms
    return s, R, t


def apply_sheet(name, cfg):
    ward = cfg["ward"]
    page = json.loads((CAD_DIR / f"ward{ward:03d}_page.json").read_text())
    rep, segs = page["report"], page["segments"]
    gcp = GCP_DIR / f"ward{ward:03d}.points"
    if not gcp.exists():
        return {"sheet": name, "status": "WAITING_FOR_GCPS", "expected_file": str(gcp)}
    W, H = rep["page_size_pt"]
    rot = rep["page_rotate"]
    k = rep["render"]["px_per_pt"]
    DW, DH = display_size(W, H, rot)
    px, mp, crs = read_points(gcp)
    res = {"sheet": name, "ward": ward, "gcps": len(px), "gcp_crs": crs, "errors": []}
    if len(px) < MIN_GCPS:
        res["errors"].append(f"need >= {MIN_GCPS} GCPs, got {len(px)}")
        res["status"] = "REJECTED"
        return res
    page_pts = np.array([display_to_page(X / k, DH - Y / k, W, H, rot) for X, Y in px])
    s, R, t = fit_similarity(page_pts, mp)
    pred = (s * (R @ page_pts.T)).T + t
    resid = np.hypot(*(pred - mp).T)
    loo = []
    for i in range(len(px)):
        m = np.arange(len(px)) != i
        si, Ri, ti = fit_similarity(page_pts[m], mp[m])
        loo.append(float(np.hypot(*((si * Ri @ page_pts[i]) + ti - mp[i]))))
    rot_deg = math.degrees(math.atan2(R[1, 0], R[0, 0])) % 360
    pr = rep["priors"]
    scale_dev = s / pr["m_per_pt_from_area"] - 1 if pr["m_per_pt_from_area"] else None
    rot_dev = ((rot_deg - pr["rotation_deg_from_north_arrow"] + 180) % 360) - 180
    res.update({
        "m_per_pt": round(s, 5), "implied_scale": round(s / (0.0254 / 72)),
        "rotation_deg": round(rot_deg, 3),
        "rms_m": round(float(np.sqrt((resid ** 2).mean())), 2), "max_resid_m": round(float(resid.max()), 2),
        "loo_max_m": round(max(loo), 2), "per_gcp_resid_m": [round(float(r), 2) for r in resid],
        "scale_vs_area_prior": round(scale_dev, 4) if scale_dev is not None else None,
        "rotation_vs_north_arrow_deg": round(rot_dev, 2),
    })
    if res["rms_m"] > MAX_RMS_M:
        res["errors"].append(f"RMS {res['rms_m']} m > {MAX_RMS_M}")
    if res["loo_max_m"] > MAX_LOO_M:
        res["errors"].append(f"leave-one-out max {res['loo_max_m']} m > {MAX_LOO_M} (a GCP is likely wrong)")
    if scale_dev is not None and abs(scale_dev) > MAX_SCALE_VS_AREA:
        res["errors"].append(f"scale deviates {scale_dev:+.1%} from printed ward area")
    if abs(rot_dev) > MAX_ROT_VS_NORTH_DEG:
        res["errors"].append(f"rotation deviates {rot_dev:+.1f} deg from north arrow")

    rows = []
    for sg in segs:
        P = np.array(sg["page_pts"])
        E = (s * (R @ P.T)).T + t
        lon, lat = TO_WGS84.transform(E[:, 0], E[:, 1])
        row = {k2: v for k2, v in sg.items() if k2 not in ("page_pts", "length_pt")}
        row.update({
            "ward": ward, "source_sheet": f"{name} p1",
            "source": "KMC ward drainage/sewer map (AutoCAD PDF)",
            "coverage_note": rep.get("coverage_note"),
            "extraction_method": "cad_vector_gcp_similarity",
            "georef_rms_m": res["rms_m"], "flow_arrow": None, "scheme": None,
            "geometry": LineString(zip(lon, lat)),
        })
        rows.append(row)
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    res["osm_check"] = merge_kmc_layers.osm_road_distance(gdf)
    res["status"] = "REJECTED" if res["errors"] else "ACCEPTED"
    if not res["errors"]:
        gdf.to_file(OUT / f"ward{ward:03d}_cad_pipes.geojson", driver="GeoJSON")
    return res


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "apply"):
        print(__doc__)
        return 2
    only = sys.argv[2:] or list(SHEETS)
    if sys.argv[1] == "prepare":
        reps = [prepare_sheet(n, SHEETS[n]) for n in only]
        (CAD_DIR / "prepare_report.json").write_text(json.dumps(reps, indent=2, default=str))
        for r in reps:
            print(json.dumps({k: v for k, v in r.items() if k not in ("page_size_pt",)}, indent=1, default=str))
        return 0
    reps = [apply_sheet(n, SHEETS[n]) for n in only]
    (CAD_DIR / "apply_report.json").write_text(json.dumps(reps, indent=2, default=str))
    print(json.dumps(reps, indent=1, default=str))
    merge_kmc_layers.main()
    return 1 if any(r.get("status") == "REJECTED" for r in reps) else 0


if __name__ == "__main__":
    sys.exit(main())

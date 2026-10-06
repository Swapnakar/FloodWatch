"""
Extract KMC sewer/drainage vectors from georeferenced (ArcMap GeoPDF) ward maps.

For each configured sheet this script:
  1. Reads the page's GeoPDF viewport (/VP /Measure /GEO) and fits an affine
     transform from page coordinates to UTM 45N using the embedded corner points
     (GPTS/LPTS). It checks fit residuals, north-up orientation, and the implied
     map scale against the scale printed on the sheet.
  2. Parses the sheet's own legend to map each line style (stroke colour and
     width) to a pipe diameter. Nothing is hardcoded per sheet. A style shared
     by two diameters is kept ambiguous and resolved only from on-map labels.
  3. Extracts every stroked path on the "Sewerage Network" layer. Any line style
     not in the legend is a hard error, never guessed. Pipes are cross-checked
     against the diameter labels printed on the map, and flow-direction arrows
     are detected.
  4. Extracts drainage pumping stations (symbol glyphs) and water bodies
     (polygonised from the sheet's water raster layer, using the legend colour).
  5. Writes per-sheet GeoJSON to data/processed/kmc_extracted/, then runs
     merge_kmc_layers.py to rebuild data/gis/*.geojson (EPSG:4326) from every
     passing sheet, both GeoPDF and AutoCAD.

Usage (from backend/):
    ../.venv/bin/python scripts/extract_kmc_geopdf.py
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from pypdf import PdfReader
from pypdf.generic import ContentStream, IndirectObject
from pyproj import Transformer
from rasterio.features import shapes as raster_shapes
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pdf_vectors import IDENTITY, PageVectors, _oc_names, mat_mul, read_page  # noqa: E402
from config import DATA_DIR  # noqa: E402

RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed" / "kmc_extracted"
GIS_DIR = DATA_DIR / "gis"

UTM = "EPSG:32645"  # WGS84 / UTM 45N, the CRS the GeoPDFs were produced in
TO_WGS84 = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
TO_UTM = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
PT_TO_M = 0.0254 / 72  # one PDF point in metres on paper

SHEETS = []
for _f in sorted(RAW_DIR.glob("*.pdf")):
    _m = re.match(r"^(\d{2,3})_", _f.name)
    if _m:
        _ward = int(_m.group(1))
        try:
            _reader = PdfReader(_f)
            for _p_idx in range(len(_reader.pages)):
                SHEETS.append({"file": _f.name, "page": _p_idx, "ward": _ward})
        except Exception:
            pass

SEWER_LAYER = "Sewerage Network"
SEWER_LABEL_LAYER = "Sewerage Network - Default"
PUMP_LAYER = "Drainage Pumping Station"
PUMP_LABEL_LAYER = "Drainage Pumping Station - Default"
IMAGE_LAYER = "Image"

# Quality gates (fail loudly if exceeded)
MAX_GEOREF_RESIDUAL_M = 15.0     # corner points in 108 have only 4 decimals (~11 m)
MAX_ROTATION_DEG = 3.0           # sheets are north-up
MAX_SCALE_MISMATCH = 0.05        # implied vs printed scale
MIN_LABEL_AGREEMENT = 0.90       # legend colour vs printed diameter labels

LABEL_MATCH_PT = 10.0            # label centre -> pipe distance to associate
ARROW_SNAP_PT = 0.75             # arrow vertex -> pipe endpoint distance
DUPLICATE_HAUSDORFF_M = 3.0      # cross-sheet duplicate pipe threshold
MIN_WATER_AREA_M2 = 25.0

Style = Tuple[Tuple[float, ...], float]


def style_of(p) -> Style:
    return (p.stroke, round(p.width, 2))


# --------------------------------------------------------------- georeference

def _res(o):
    return o.get_object() if isinstance(o, IndirectObject) else o


@dataclass
class Viewport:
    name: str
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 (as stored)
    coeffs_e: np.ndarray                      # E = a*x + b*y + c
    coeffs_n: np.ndarray
    residual_m: float
    rotation_deg: float
    m_per_pt: float
    wkt: str

    def contains(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.bbox
        return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return abs(x2 - x1) * abs(y2 - y1)

    def to_utm(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return (self.coeffs_e[0] * x + self.coeffs_e[1] * y + self.coeffs_e[2],
                self.coeffs_n[0] * x + self.coeffs_n[1] * y + self.coeffs_n[2])


def read_viewports(page) -> List[Viewport]:
    vps = []
    for vp in _res(page.get("/VP")) or []:
        vp = _res(vp)
        measure = _res(vp.get("/Measure"))
        if measure is None or str(measure.get("/Subtype")) != "/GEO":
            continue
        bbox = tuple(float(v) for v in vp["/BBox"])
        gpts = [float(v) for v in measure["/GPTS"]]
        lpts = [float(v) for v in measure.get("/LPTS", [0, 1, 0, 0, 1, 0, 1, 1])]
        gcs = _res(measure.get("/GCS"))
        wkt = str(gcs.get("/WKT", "")) if gcs is not None else ""
        if wkt and "UTM_Zone_45N" not in wkt:
            raise ValueError(f"unexpected GeoPDF CRS: {wkt[:80]}")
        x1, y1, x2, y2 = bbox
        xs, ys, es, ns = [], [], [], []
        for i in range(0, len(lpts), 2):
            u, v = lpts[i], lpts[i + 1]
            lat, lon = gpts[i], gpts[i + 1]      # GPTS are (lat, lon) pairs
            e, n = TO_UTM.transform(lon, lat)
            xs.append(x1 + u * (x2 - x1))
            ys.append(y1 + v * (y2 - y1))
            es.append(e)
            ns.append(n)
        A = np.column_stack([xs, ys, np.ones(len(xs))])
        ce, *_ = np.linalg.lstsq(A, np.array(es), rcond=None)
        cn, *_ = np.linalg.lstsq(A, np.array(ns), rcond=None)
        resid = np.hypot(A @ ce - es, A @ cn - ns)
        # Orientation: page +x should be east, page +y north (no mirroring).
        rotation = math.degrees(math.atan2(cn[0], ce[0]))
        det = ce[0] * cn[1] - ce[1] * cn[0]
        if det <= 0:
            raise ValueError("georeference is mirrored; LPTS/GPTS misread")
        m_per_pt = math.sqrt(det)
        vps.append(Viewport(
            name=str(vp.get("/Name", "")).replace("\x00", "").strip(),
            bbox=bbox, coeffs_e=ce, coeffs_n=cn,
            residual_m=float(resid.max()), rotation_deg=rotation,
            m_per_pt=m_per_pt, wkt=wkt,
        ))
    return vps


# --------------------------------------------------------------------- legend

DIA_RE = re.compile(r"^(\d{3,4})\s*mm\b", re.I)


def parse_legend(pv: PageVectors) -> Dict:
    """Map line styles to conduit types using the sheet's own legend."""
    title = next((t for t in pv.texts if t.text.strip().upper() == "LEGEND"), None)
    if title is None:
        raise ValueError("no LEGEND found on sheet")
    layer = title.layer
    rows = []  # (y, kind, value)
    water_fill = None
    for t in pv.texts:
        if t.layer != layer:
            continue
        m = DIA_RE.match(t.text)
        if m:
            rows.append((t, "pipe", int(m.group(1))))
        elif t.text.strip().lower() == "box sewer":
            rows.append((t, "box_sewer", None))
        elif t.text.strip().lower() == "water body":
            rows.append((t, "water", None))

    swatches = []
    for p in pv.paths:
        if p.layer != layer:
            continue
        pts = [pt for sp in p.subpaths for pt in sp]
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        swatches.append((p, min(xs), max(xs), min(ys), max(ys)))

    styles: Dict[Style, List] = defaultdict(list)
    for t, kind, value in rows:
        # Swatches sit just left of the label, vertically centred on its x-height.
        cy = t.y + 0.36 * t.font_size
        left = [s for s in swatches if s[2] <= t.x and t.x - s[2] < 25]
        if kind == "water":
            fills = [s for s in left if s[0].filled and s[3] - 4 <= cy <= s[4] + 4]
            if fills:
                water_fill = fills[0][0].fill
            continue
        row = [s for s in left if s[0].stroked and not s[0].filled
               and (s[4] - s[3]) < 0.5 and abs(s[3] - cy) < 0.4 * t.font_size]
        longest = max((s[2] - s[1] for s in row), default=0)
        # Swatch size varies by sheet; dash overlays are much shorter than the swatch.
        full = [s for s in row if (s[2] - s[1]) >= 0.8 * longest and longest > 5]
        if not full:
            raise ValueError(f"legend row {t.text!r} has no swatch")
        if kind == "pipe":
            # Full-length stroke only (arrowheads are filled, so already excluded).
            for s in full:
                styles[style_of(s[0])].append((kind, value))
        else:
            # Box sewer symbol = several stacked strokes, incl. short dash overlays.
            core = min(full, key=lambda s: s[0].width)
            for s in row:
                styles[style_of(s[0])].append((kind, "core" if s is core else "decoration"))

    style_map = {}
    for k, entries in styles.items():
        kinds = {e[0] for e in entries}
        if kinds == {"pipe"}:
            dias = sorted({e[1] for e in entries})
            style_map[k] = {"conduit_type": "pipe", "diameters": dias}
        elif kinds == {"box_sewer"}:
            # Box sewer = black casing + coloured core + dash overlays; only the
            # core stroke is used as geometry, the rest is symbol decoration.
            style_map[k] = {"conduit_type": "box_sewer", "diameters": [],
                            "geometry": any(e[1] == "core" for e in entries)}
        else:
            raise ValueError(f"legend style {k} maps to conflicting kinds {entries}")
    return {"styles": style_map, "water_fill": water_fill}


# ---------------------------------------------------------------- extraction

LABEL_RE = re.compile(r"^(\d{3,4})(?:\s+(KEIIP))?$", re.I)


def printed_scale(pv: PageVectors) -> Optional[int]:
    """Scale printed on the sheet, e.g. 'SCALE - 1:2,800' (may be split in two texts)."""
    for t in pv.texts:
        m = re.search(r"SCALE\s*[-:]?\s*1\s*:\s*([\d,]+)", t.text, re.I)
        if m:
            return int(m.group(1).replace(",", ""))
    anchors = [t for t in pv.texts if re.match(r"^\s*SCALE\b", t.text, re.I)]
    for t in pv.texts:
        m = re.match(r"^\s*1\s*:\s*([\d,]+)\s*$", t.text)
        if m and any(abs(a.y - t.y) < 2 * a.font_size and 0 <= t.x - a.x < 20 * a.font_size for a in anchors):
            return int(m.group(1).replace(",", ""))
    return None


def extract_sheet(sheet: Dict) -> Dict:
    path = RAW_DIR / sheet["file"]
    reader = PdfReader(str(path))
    page = reader.pages[sheet["page"]]
    pv = read_page(reader, sheet["page"])
    sheet_id = f"{sheet['file']} p{sheet['page'] + 1}"
    report = {"sheet": sheet_id, "ward": sheet["ward"], "errors": [], "warnings": []}

    if pv.skipped_form_xobjects:
        report["errors"].append(f"{pv.skipped_form_xobjects} Form XObjects not parsed")

    # ---- georeference
    vps = read_viewports(page)
    if not vps:
        raise ValueError(f"{sheet_id}: no GeoPDF viewport")
    sewer_paths = [p for p in pv.paths if SEWER_LAYER in p.layers and p.stroked]

    def vp_for(p):
        x, y = p.subpaths[0][0]
        return [v for v in vps if v.contains(x, y)]

    main = max(vps, key=lambda v: sum(1 for p in sewer_paths if v in vp_for(p)) + v.area * 1e-12)
    insets = [v for v in vps if v is not main]
    scale_implied = main.m_per_pt / PT_TO_M
    scale_printed = printed_scale(pv)
    report["georef"] = {
        "viewport": main.name, "max_corner_residual_m": round(main.residual_m, 2),
        "rotation_deg": round(main.rotation_deg, 3), "m_per_pt": round(main.m_per_pt, 4),
        "implied_scale": round(scale_implied), "printed_scale": scale_printed,
        "insets_excluded": [v.name for v in insets],
    }
    if main.residual_m > MAX_GEOREF_RESIDUAL_M:
        report["errors"].append(f"georef residual {main.residual_m:.1f} m")
    if abs(main.rotation_deg) > MAX_ROTATION_DEG:
        report["errors"].append(f"sheet rotated {main.rotation_deg:.2f} deg")
    if scale_printed:
        mismatch = abs(scale_implied - scale_printed) / scale_printed
        report["georef"]["scale_mismatch"] = round(mismatch, 4)
        if mismatch > MAX_SCALE_MISMATCH:
            report["errors"].append(f"implied scale 1:{scale_implied:.0f} vs printed 1:{scale_printed}")

    # ---- legend
    legend = parse_legend(pv)
    style_map = legend["styles"]
    report["legend"] = {f"{k[0]} w={k[1]}": v for k, v in style_map.items()}

    # ---- pipes (page coordinates)
    unknown = Counter()
    pipes = []
    outside_main = under_inset = decoration = 0
    for p in sewer_paths:
        if p.filled:  # flow arrowheads: stroked+filled black triangles
            continue
        if main not in vp_for(p):
            outside_main += 1
            continue
        if any(v.contains(*p.subpaths[0][0]) for v in insets):
            # Still main-frame data (inset frames carry no sewer layer); just counted.
            under_inset += 1
        key = style_of(p)
        info = style_map.get(key)
        if info is None:
            unknown[key] += 1
            continue
        if info["conduit_type"] == "box_sewer" and not info.get("geometry"):
            decoration += 1
            continue
        for sp in p.subpaths:
            pts = [pt for i, pt in enumerate(sp) if i == 0 or pt != sp[i - 1]]
            if len(pts) < 2:
                continue
            pipes.append({"page_pts": pts, "info": info, "style": key, "labels": []})
    if unknown:
        report["errors"].append(f"unmapped sewer line styles: {dict(unknown)}")

    # ---- flow arrows: small filled triangles whose apex touches a pipe end
    arrow_pts = []
    for p in pv.paths:
        if SEWER_LAYER in p.layers and p.filled:
            pts = [pt for sp in p.subpaths for pt in sp]
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            if max(xs) - min(xs) < 15 and max(ys) - min(ys) < 15:
                arrow_pts.extend(pts)
    arrow_tree = STRtree([Point(xy) for xy in arrow_pts]) if arrow_pts else None

    def near_arrow(xy):
        if arrow_tree is None:
            return False
        return len(arrow_tree.query(Point(xy).buffer(ARROW_SNAP_PT))) > 0

    for pp in pipes:
        start, end = pp["page_pts"][0], pp["page_pts"][-1]
        pp["flow_arrow"] = "end" if near_arrow(end) else ("start" if near_arrow(start) else "none")

    # ---- label cross-check
    lines_page = [LineString(pp["page_pts"]) for pp in pipes]
    tree = STRtree(lines_page)
    labels = [t for t in pv.texts if t.layer == SEWER_LABEL_LAYER]
    matched = 0
    for t in labels:
        m = LABEL_RE.match(t.text)
        if not m:
            continue
        # Approximate label centre: anchor + half the text length along its angle.
        half = 0.25 * t.font_size * len(t.text)
        a = math.radians(t.angle_deg)
        c = Point(t.x + half * math.cos(a), t.y + half * math.sin(a))
        idx = tree.query_nearest(c, max_distance=LABEL_MATCH_PT)
        if len(idx) == 0:
            continue
        pipes[int(idx[0])]["labels"].append((int(m.group(1)), bool(m.group(2))))
        matched += 1

    agree = disagree = resolved = unresolved = 0
    for pp in pipes:
        info = pp["info"]
        dias = info["diameters"]
        label_dias = Counter(d for d, _ in pp["labels"])
        label_dia = label_dias.most_common(1)[0][0] if label_dias else None
        pp["label_diameter_mm"] = label_dia
        pp["scheme"] = "KEIIP" if any(k for _, k in pp["labels"]) else None
        if info["conduit_type"] == "box_sewer":
            pp["pipe_diameter_mm"], pp["diameter_source"], pp["label_agrees"] = None, None, None
            continue
        if len(dias) == 1:
            pp["pipe_diameter_mm"], pp["diameter_source"] = dias[0], "legend_colour"
            if label_dia is not None:
                pp["label_agrees"] = label_dia == dias[0]
                agree += pp["label_agrees"]
                disagree += not pp["label_agrees"]
            else:
                pp["label_agrees"] = None
        else:
            # Colour shared by several diameters: only an on-map label may decide.
            if label_dia in dias:
                pp["pipe_diameter_mm"], pp["diameter_source"] = label_dia, "map_label"
                resolved += 1
            else:
                pp["pipe_diameter_mm"], pp["diameter_source"] = None, None
                pp["diameter_candidates_mm"] = dias
                unresolved += 1
            pp["label_agrees"] = None
    checked = agree + disagree
    rate = agree / checked if checked else None
    report["pipes"] = {
        "segments": len(pipes),
        "skipped_outside_main_frame": outside_main,
        "under_inset_frame_kept": under_inset,
        "box_sewer_decoration_strokes_skipped": decoration,
        "by_type": dict(Counter(pp["info"]["conduit_type"] for pp in pipes)),
        "labels_total": len(labels), "labels_matched_to_pipe": matched,
        "pipes_label_checked": checked, "label_agreement": round(rate, 4) if rate is not None else None,
        "ambiguous_colour_resolved_by_label": resolved,
        "ambiguous_colour_unresolved": unresolved,
        "flow_arrow": dict(Counter(pp["flow_arrow"] for pp in pipes)),
    }
    if rate is not None and rate < MIN_LABEL_AGREEMENT:
        report["errors"].append(f"legend/label agreement only {rate:.1%}")
    mism = Counter((pp["pipe_diameter_mm"], pp["label_diameter_mm"]) for pp in pipes if pp.get("label_agrees") is False)
    if mism:
        report["pipes"]["label_mismatches(legend,label)"] = {f"{a},{b}": n for (a, b), n in mism.most_common()}

    # ---- pumping stations
    glyphs = [t for t in pv.texts if t.layer == PUMP_LAYER and main.contains(t.x, t.y)]
    plabels = [t for t in pv.texts if t.layer == PUMP_LABEL_LAYER]
    stations = []
    for g in glyphs:
        if any(math.hypot(g.x - s["x"], g.y - s["y"]) < 3 for s in stations):
            continue
        lab = min(plabels, key=lambda t: math.hypot(t.x - g.x, t.y - g.y), default=None)
        name = lab.text if lab is not None and math.hypot(lab.x - g.x, lab.y - g.y) < 40 else None
        stations.append({"x": g.x, "y": g.y, "name": name})
    report["pumping_stations"] = len(stations)

    # ---- water bodies from the raster layer
    water_polys = extract_water(reader, page, legend["water_fill"], main)
    report["water_bodies"] = len(water_polys)

    # ---- to WGS84
    def page_line_to_wgs(pts):
        xs, ys = zip(*pts)
        e, n = main.to_utm(xs, ys)
        lon, lat = TO_WGS84.transform(e, n)
        return LineString(zip(lon, lat))

    base = {
        "ward": sheet["ward"], "source_sheet": sheet_id,
        "source": "KMC ward sewerage network map (ArcMap GeoPDF)",
        "extraction_method": "geopdf_vector",
        "georef_max_residual_m": round(main.residual_m, 1),
    }
    pipe_rows = []
    for pp in pipes:
        row = dict(base)
        row.update({
            "conduit_type": pp["info"]["conduit_type"],
            "pipe_diameter_mm": pp["pipe_diameter_mm"],
            "diameter_source": pp["diameter_source"],
            "diameter_candidates_mm": "|".join(map(str, pp.get("diameter_candidates_mm", []))) or None,
            "label_diameter_mm": pp["label_diameter_mm"],
            "label_agrees": pp["label_agrees"],
            "scheme": pp["scheme"],
            "flow_arrow": pp["flow_arrow"],
            "geometry": page_line_to_wgs(pp["page_pts"]),
        })
        pipe_rows.append(row)

    def page_xy_to_wgs(x, y):
        e, n = main.to_utm(x, y)
        return TO_WGS84.transform(float(e), float(n))

    station_rows = [dict(base, name=s["name"], geometry=Point(page_xy_to_wgs(s["x"], s["y"])),
                         position_note="symbol anchor; may be offset ~10 m from true location")
                    for s in stations]
    water_rows = [dict(base, extraction_method="geopdf_raster_polygonised", geometry=g)
                  for g in water_polys]
    for rows in (pipe_rows, station_rows, water_rows):
        for r in rows:
            if r is not base and "georef_max_residual_m" not in r:
                r["georef_max_residual_m"] = base["georef_max_residual_m"]

    return {"report": report, "pipes": pipe_rows, "stations": station_rows, "water": water_rows}


def extract_water(reader, page, water_fill, vp: Viewport) -> List:
    """Polygonise water-body pixels from the sheet's raster layer (legend colour)."""
    if water_fill is None:
        return []
    target = np.array([round(c * 255) for c in water_fill], dtype=np.int16)
    oc = _oc_names(page)
    ctm, st, mc, placed = IDENTITY, [], [], []
    for a, op in ContentStream(page.get_contents(), reader).operations:
        if op == b"q":
            st.append(ctm)
        elif op == b"Q":
            ctm = st.pop()
        elif op == b"cm":
            ctm = mat_mul(tuple(float(v) for v in a), ctm)
        elif op == b"BDC":
            mc.append(oc.get(str(a[1])) if len(a) > 1 and str(a[0]) == "/OC" else None)
        elif op == b"BMC":
            mc.append(None)
        elif op == b"EMC":
            mc.pop()
        elif op == b"Do" and IMAGE_LAYER in [s for s in mc if s]:
            placed.append((str(a[0]).lstrip("/"), ctm))
    if not placed:
        return []
    images = {im.name.rsplit(".", 1)[0].lstrip("/"): im.image for im in page.images}
    # Strips are axis-aligned (b = c = 0) and share x offset and width.
    if any(abs(m[1]) > 1e-9 or abs(m[2]) > 1e-9 for _, m in placed):
        raise ValueError("rotated raster strips not supported")
    x0, width_pt = placed[0][1][4], placed[0][1][0]
    wpx = images[placed[0][0]].size[0]
    sx = wpx / width_pt
    top = max(max(m[5], m[5] + m[3]) for _, m in placed)
    bottom = min(min(m[5], m[5] + m[3]) for _, m in placed)
    hpx = int(round((top - bottom) * sx))
    mask = np.zeros((hpx, wpx), dtype=np.uint8)
    for name, m in placed:
        arr = np.asarray(images[name].convert("RGB"), dtype=np.int16)
        hit = (np.abs(arr - target).max(axis=2) <= 3).astype(np.uint8)
        # PDF maps image row 0 to unit-square y=1. With d > 0 row 0 is at the
        # top of the placed box; with d < 0 (as ArcMap writes) the strip is
        # vertically flipped, so row 0 lands at the bottom.
        if m[3] < 0:
            hit = hit[::-1]
        strip_top = max(m[5], m[5] + m[3])
        r0 = int(round((top - strip_top) * sx))
        r1 = min(r0 + hit.shape[0], hpx)
        mask[r0:r1] |= hit[: r1 - r0]
    # pixel (col,row) -> page (x,y)
    from affine import Affine
    pix_to_page = Affine(1 / sx, 0, x0, 0, -1 / sx, top)
    polys = []
    for geom, val in raster_shapes(mask, mask=mask.astype(bool), transform=pix_to_page):
        if val != 1:
            continue
        poly_page = shape(geom)

        def to_utm(x, y, z=None):
            return vp.to_utm(x, y)

        poly_utm = shp_transform(to_utm, poly_page)
        if poly_utm.area < MIN_WATER_AREA_M2:
            continue
        poly_utm = poly_utm.simplify(0.5).buffer(0)
        polys.append(shp_transform(lambda x, y, z=None: TO_WGS84.transform(x, y), poly_utm))
    return polys


def main() -> int:
    import merge_kmc_layers

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    failed = False
    for sheet in SHEETS:
        try:
            res = extract_sheet(sheet)
            rep = res["report"]
        except Exception as e:
            print(f"Failed to process sheet {sheet}: {e}", file=sys.stderr)
            reports.append({"sheet": f"{sheet['file']} p{sheet['page']+1}", "errors": [str(e)]})
            failed = True
            continue
        reports.append(rep)
        print(json.dumps(rep, indent=2, default=str))
        stem = f"ward{sheet['ward']}_p{sheet['page'] + 1}"
        # Remove stale outputs first so a now-failing sheet can't linger in the merge.
        for old in OUT_DIR.glob(f"{stem}_*.geojson"):
            old.unlink()
        if rep["errors"]:
            failed = True
            print(f"!! {rep['sheet']}: NOT merged due to errors", file=sys.stderr)
            continue
        for kind, rows in (("pipes", res["pipes"]), ("pumping_stations", res["stations"]),
                           ("water_bodies", res["water"])):
            if rows:
                g = gpd.GeoDataFrame(rows, crs="EPSG:4326")
                if "label_agrees" in g:
                    g["label_agrees"] = g["label_agrees"].astype("boolean")
                g.to_file(OUT_DIR / f"{stem}_{kind}.geojson", driver="GeoJSON")

    (OUT_DIR / "extraction_report.json").write_text(json.dumps(reports, indent=2, default=str))
    merge_kmc_layers.main()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

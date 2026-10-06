"""
Data validation for FloodWatch GIS inputs (Task 8, early version; extended in Task 16).

Checks each runtime GIS layer in data/gis/ and FAILS LOUDLY (exit code 1)
listing the offending rows. Missing optional values, such as an unknown pipe
diameter, only produce warnings. Anything that would make downstream spatial
joins silently wrong is an error.

Per layer:
  - file exists and is readable
  - CRS is EPSG:4326 and coordinates really are lon/lat degrees. A GeoJSON with
    no CRS member is reported as 4326 by GDAL even if it holds UTM metres, so
    the Kolkata bounding-box check doubles as a CRS check.
  - geometry: no null, empty or invalid geometries; allowed geometry types only
  - every feature lies within the Kolkata bounding box
  - required attributes present, and non-null where required
  - attribute value ranges (pipe diameter, ward number)
  - duplicate IDs and exact duplicate geometries
  - degenerate geometry (near-zero-length lines, near-zero-area polygons)

Usage (from backend/):
    ../.venv/bin/python scripts/validate_data.py            # human-readable
    ../.venv/bin/python scripts/validate_data.py --json     # machine-readable
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
from shapely.validation import explain_validity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

# Generous Kolkata Metropolitan box (lon_min, lat_min, lon_max, lat_max).
KOLKATA_BBOX = (88.20, 22.40, 88.55, 22.75)
METRIC_CRS = "EPSG:32645"  # UTM 45N, for lengths/areas
MAX_LISTED = 25            # rows listed per problem before truncating


@dataclass
class LayerSpec:
    name: str
    path: Path
    geom_types: Tuple[str, ...]
    required: Tuple[str, ...]                  # columns that must exist
    required_non_null: Tuple[str, ...] = ()    # columns that must have no nulls
    id_column: Optional[str] = None
    min_length_m: Optional[float] = None
    min_area_m2: Optional[float] = None


def default_specs() -> List[LayerSpec]:
    gis = settings.DRAINAGE_GEOJSON_PATH.parent
    return [
        LayerSpec(
            name="drainage_network", path=settings.DRAINAGE_GEOJSON_PATH,
            geom_types=("LineString", "MultiLineString"),
            required=("segment_id", "ward", "source", "conduit_type", "pipe_diameter_mm",
                      "diameter_source", "extraction_method"),
            required_non_null=("segment_id", "ward", "source", "conduit_type", "extraction_method"),
            id_column="segment_id", min_length_m=1.0,
        ),
        LayerSpec(
            name="pumping_stations",
            path=getattr(settings, "PUMPING_STATIONS_GEOJSON_PATH", gis / "pumping_stations.geojson"),
            geom_types=("Point",),
            required=("ward", "source", "name"),
            required_non_null=("ward", "source"),
        ),
        LayerSpec(
            name="water_bodies",
            path=getattr(settings, "WATER_BODIES_GEOJSON_PATH", gis / "water_bodies.geojson"),
            geom_types=("Polygon", "MultiPolygon"),
            required=("source",),
            required_non_null=("source",),
            min_area_m2=10.0,
        ),
    ]


@dataclass
class LayerResult:
    name: str
    path: str
    rows: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _rows(idx: Sequence, gdf: gpd.GeoDataFrame, id_col: Optional[str]) -> str:
    """Human-readable row list: row index plus feature id where available."""
    idx = list(idx)
    labels = [f"{i}" + (f" ({gdf.at[i, id_col]})" if id_col and id_col in gdf and gdf.at[i, id_col] is not None else "")
              for i in idx[:MAX_LISTED]]
    more = f" … and {len(idx) - MAX_LISTED} more" if len(idx) > MAX_LISTED else ""
    return ", ".join(labels) + more


def validate_layer(spec: LayerSpec, bbox=KOLKATA_BBOX) -> LayerResult:
    res = LayerResult(name=spec.name, path=str(spec.path))
    if not spec.path.exists():
        res.errors.append(f"file missing: {spec.path}")
        return res
    try:
        gdf = gpd.read_file(spec.path)
    except Exception as exc:  # unreadable / malformed JSON
        res.errors.append(f"unreadable: {type(exc).__name__}: {exc}")
        return res
    gdf = gdf.reset_index(drop=True)
    res.rows = len(gdf)
    if res.rows == 0:
        res.errors.append("layer is empty")
        return res

    # ---- CRS
    if gdf.crs is None:
        res.errors.append("no CRS")
    elif gdf.crs.to_epsg() != 4326:
        res.errors.append(f"CRS is {gdf.crs.to_string()}, expected EPSG:4326")

    # ---- columns
    missing = [c for c in spec.required if c not in gdf.columns]
    if missing:
        res.errors.append(f"missing required columns: {missing}")
    for col in spec.required_non_null:
        if col in gdf.columns:
            nulls = gdf.index[gdf[col].isna()]
            if len(nulls):
                res.errors.append(f"null '{col}' in {len(nulls)} rows: {_rows(nulls, gdf, spec.id_column)}")

    # ---- geometry presence / type / validity
    geom = gdf.geometry
    null_g = gdf.index[geom.isna()]
    if len(null_g):
        res.errors.append(f"null geometry in {len(null_g)} rows: {_rows(null_g, gdf, spec.id_column)}")
    present = geom.notna()
    empty = gdf.index[present & geom.is_empty]
    if len(empty):
        res.errors.append(f"empty geometry in {len(empty)} rows: {_rows(empty, gdf, spec.id_column)}")
    usable = present & ~geom.is_empty
    bad_type = gdf.index[usable & ~geom.geom_type.isin(spec.geom_types)]
    if len(bad_type):
        types = sorted(set(geom[bad_type].geom_type))
        res.errors.append(f"unexpected geometry types {types} (allowed {list(spec.geom_types)}) "
                          f"in {len(bad_type)} rows: {_rows(bad_type, gdf, spec.id_column)}")
    invalid = gdf.index[usable & ~geom.is_valid]
    if len(invalid):
        reasons = sorted({explain_validity(geom[i]).split("[")[0].strip() for i in invalid[:MAX_LISTED]})
        res.errors.append(f"invalid geometry in {len(invalid)} rows ({'; '.join(reasons)}): "
                          f"{_rows(invalid, gdf, spec.id_column)}")

    # ---- coordinates: finite and inside Kolkata
    ok = gdf[usable]
    if len(ok):
        b = ok.geometry.bounds
        finite = np.isfinite(b.to_numpy()).all(axis=1)
        nonfinite = ok.index[~finite]
        if len(nonfinite):
            res.errors.append(f"non-finite coordinates in {len(nonfinite)} rows: {_rows(nonfinite, gdf, spec.id_column)}")
        w, s, e, n = bbox
        outside = ok.index[finite & ~((b.minx >= w) & (b.maxx <= e) & (b.miny >= s) & (b.maxy <= n))]
        if len(outside):
            res.errors.append(
                f"{len(outside)} rows outside Kolkata bbox {bbox} (wrong CRS, lat/lon swapped, or bad georeference): "
                f"{_rows(outside, gdf, spec.id_column)}")
        res.stats["bounds"] = [round(float(v), 5) for v in ok.total_bounds]

    # ---- degenerate geometry (metric), only on geometries that are otherwise fine
    good = gdf[usable & geom.is_valid & geom.geom_type.isin(spec.geom_types)]
    if len(good) and res.stats.get("bounds") and not any("outside Kolkata" in e for e in res.errors):
        metric = good.to_crs(METRIC_CRS).geometry
        if spec.min_length_m is not None:
            L = metric.length
            short = good.index[L < spec.min_length_m]
            if len(short):
                res.errors.append(f"{len(short)} lines shorter than {spec.min_length_m} m: {_rows(short, gdf, spec.id_column)}")
            res.stats["total_length_km"] = round(float(L.sum()) / 1000, 2)
        if spec.min_area_m2 is not None:
            A = metric.area
            tiny = good.index[A < spec.min_area_m2]
            if len(tiny):
                res.errors.append(f"{len(tiny)} polygons smaller than {spec.min_area_m2} m²: {_rows(tiny, gdf, spec.id_column)}")
            res.stats["total_area_ha"] = round(float(A.sum()) / 1e4, 2)

    # ---- duplicates
    if spec.id_column and spec.id_column in gdf.columns:
        dup = gdf.index[gdf[spec.id_column].duplicated(keep=False) & gdf[spec.id_column].notna()]
        if len(dup):
            res.errors.append(f"duplicate '{spec.id_column}' in {len(dup)} rows: {_rows(dup, gdf, spec.id_column)}")
    if usable.any():
        wkb = gdf.geometry[usable].apply(lambda g: g.normalize().wkb)
        dupg = wkb.index[wkb.duplicated(keep="first")]
        if len(dupg):
            res.errors.append(f"exact duplicate geometries in {len(dupg)} rows: {_rows(dupg, gdf, spec.id_column)}")

    # ---- layer-specific attribute checks
    if spec.name == "drainage_network":
        _check_drainage(gdf, spec, res)
    if "ward" in gdf.columns:
        w = gdf["ward"].dropna()
        bad = w.index[(w < 1) | (w > 144) | (w != np.floor(w))]
        if len(bad):
            res.errors.append(f"ward outside KMC range 1-144 in {len(bad)} rows: {_rows(bad, gdf, spec.id_column)}")
        res.stats["wards"] = sorted(int(v) for v in w.unique())
    return res


def _check_drainage(gdf, spec, res):
    if "pipe_diameter_mm" in gdf.columns:
        d = gdf["pipe_diameter_mm"]
        pipes = gdf["conduit_type"].eq("pipe") if "conduit_type" in gdf else d.notna()
        bad = gdf.index[d.notna() & ((d < 100) | (d > 3000))]
        if len(bad):
            res.errors.append(f"pipe_diameter_mm outside 100-3000 in {len(bad)} rows: {_rows(bad, gdf, spec.id_column)}")
        # Unknown diameter is allowed (never guessed), but its share is reported.
        if pipes.any():
            metric = gdf.to_crs(METRIC_CRS).geometry.length if res.stats.get("total_length_km") else None
            n_unknown = int((pipes & d.isna()).sum())
            res.stats["pipes_unknown_diameter"] = n_unknown
            if metric is not None:
                share = float(metric[pipes & d.isna()].sum() / max(metric[pipes].sum(), 1e-9))
                res.stats["pipe_length_unknown_diameter_pct"] = round(100 * share, 1)
                if share > 0.25:
                    res.warnings.append(f"{100 * share:.1f}% of pipe length has no diameter")
            elif n_unknown:
                res.warnings.append(f"{n_unknown} pipes have no diameter")
        res.stats["by_diameter_mm"] = {str(int(k)): int(v) for k, v in d.dropna().value_counts().sort_index().items()}
    if "conduit_type" in gdf.columns:
        allowed = {"pipe", "box_sewer", "drain"}
        bad = gdf.index[gdf["conduit_type"].notna() & ~gdf["conduit_type"].isin(allowed)]
        if len(bad):
            res.errors.append(f"conduit_type not in {sorted(allowed)} in {len(bad)} rows: {_rows(bad, gdf, spec.id_column)}")
        res.stats["by_conduit_type"] = {str(k): int(v) for k, v in gdf["conduit_type"].value_counts().items()}
    if "diameter_source" in gdf.columns and "pipe_diameter_mm" in gdf.columns:
        # A diameter must always say where it came from, and vice versa.
        a = gdf.index[gdf["pipe_diameter_mm"].notna() & gdf["diameter_source"].isna()]
        b = gdf.index[gdf["pipe_diameter_mm"].isna() & gdf["diameter_source"].notna()]
        if len(a):
            res.errors.append(f"diameter without diameter_source in {len(a)} rows: {_rows(a, gdf, spec.id_column)}")
        if len(b):
            res.errors.append(f"diameter_source set but no diameter in {len(b)} rows: {_rows(b, gdf, spec.id_column)}")


def validate_all(specs: Optional[List[LayerSpec]] = None) -> List[LayerResult]:
    return [validate_layer(s) for s in (specs or default_specs())]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = ap.parse_args(argv)
    results = validate_all()
    failed = any(not r.ok for r in results)
    if args.json:
        print(json.dumps([dict(r.__dict__, ok=r.ok) for r in results], indent=2, default=str))
    else:
        for r in results:
            print(f"[{'PASS' if r.ok else 'FAIL'}] {r.name}: {r.rows} features  ({r.path})")
            for k, v in r.stats.items():
                print(f"       {k}: {v}")
            for w in r.warnings:
                print(f"   WARN  {w}")
            for e in r.errors:
                print(f"   ERROR {e}")
        print("\nRESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

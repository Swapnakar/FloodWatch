"""
Build the REAL training dataset for the flood-probability classifier (Task 15, option A).

Label
-----
`historical_waterlogging` (1/0) is REAL, from the KMC 2017 "Major Water Logging
Pockets" list (data/historical/waterlogging_points.csv, geocoded in Task 10).
It is NOT fabricated. Points are only labelled where the label is trustworthy:

  positives: the located historical pockets themselves (label 1).
  negatives: a grid of points across the KMC area (label 0), each kept at least
             NEG_MIN_DIST_M from every located pocket, so we never label a known
             flood spot as safe. Points outside KMC are dropped (no KMC records
             there => unknown, not negative).

Features
--------
Per point, from the real services:
  elevation_m, slope_percent            (DEM, present city-wide)
  distance_to_waterbody_m               (KMC water bodies, wards 107/108 only -> often null)
  distance_to_drain_m, pipe_diameter_mm,
    drain_capacity_estimated_m3s         (drainage, wards 107/108 only -> mostly null)
  within_mapped_drainage                 (bool)
Drainage is a SPARSE optional feature (option A): the two digitised wards barely
overlap the pocket list, so most rows have null drainage. Terrain + water +
rainfall carry the signal. This is stated plainly and Task 17 will report how
much these features actually predict.

Rainfall
--------
IMD gives one live observation, not a per-point history, and the label is a
static list rather than dated events. So rainfall is added as documented
representative SCENARIOS (dry / moderate / heavy / extreme), each point emitted
once per scenario. This mirrors the old synthetic generator's approach, but the
LABEL stays real. Feature `rainfall_scenario_mm` is clearly a scenario, not an
observation. --no-rainfall-scenarios emits one row per point with rainfall null.

Output
------
data/processed/training_data.csv, plus a printed report: row count, class
balance, and per-feature completeness (% non-null).
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, settings  # noqa: E402
from services.dem_service import DEMService  # noqa: E402
from services.spatial_service import SpatialService  # noqa: E402

OUT = DATA_DIR / "processed" / "training_data.csv"
UTM = "EPSG:32645"

NEG_GRID_SPACING_M = 300.0
NEG_MIN_DIST_M = 300.0     # a negative must be at least this far from any pocket
HISTORICAL_RADIUS_M = 250.0  # positives use the same radius the runtime lookup uses

# Documented representative rainfall scenarios (mm over the event window).
# Not observations; used so the model sees a rainfall axis. Labels are unchanged.
RAINFALL_SCENARIOS = {"dry": 0.0, "moderate": 40.0, "heavy": 90.0, "extreme": 150.0}

FEATURE_COLUMNS = [
    "elevation_m", "slope_percent", "distance_to_waterbody_m",
    "distance_to_drain_m", "pipe_diameter_mm", "drain_capacity_estimated_m3s",
    "within_mapped_drainage", "rainfall_scenario_mm",
]
META_COLUMNS = ["point_id", "lat", "lon", "ward", "source", "rainfall_scenario", "label_source"]
LABEL = "historical_waterlogging"


def _services() -> Tuple[DEMService, SpatialService]:
    dem = DEMService(settings.DEM_PATH, settings.GEOID_GRID_PATH)
    spatial = SpatialService(
        settings.DRAINAGE_GEOJSON_PATH, settings.PUMPING_STATIONS_GEOJSON_PATH,
        settings.WATER_BODIES_GEOJSON_PATH, settings.HISTORICAL_WATERLOGGING_PATH,
        drain_max_radius_m=settings.DRAIN_MAX_SEARCH_RADIUS_M,
        historical_radius_m=HISTORICAL_RADIUS_M,
        kmc_boundary_path=settings.KMC_BOUNDARY_GEOJSON_PATH,
    )
    return dem, spatial


def load_positives() -> List[Dict]:
    path = settings.HISTORICAL_WATERLOGGING_PATH
    if not path.exists():
        raise SystemExit(f"historical file missing: {path}")
    pts = []
    with path.open(encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if not r.get("latitude") or not r.get("longitude"):
                continue
            pts.append({"point_id": f"pos_{i}", "lat": float(r["latitude"]), "lon": float(r["longitude"]),
                        "ward": (r.get("ward") or "").split(",")[0], "source": "kmc_2017_pocket",
                        "label_source": "located_historical_pocket"})
    if not pts:
        raise SystemExit("no located historical pockets; run scripts/geocode_waterlogging.py first")
    return pts


def dedupe_by_coord(points: List[Dict], places: int = 6) -> List[Dict]:
    """Collapse points sharing a coordinate (several pockets geocoded to one street
    centroid) to a single row, so identical positives aren't double-counted."""
    seen: Dict[Tuple[float, float], Dict] = {}
    for p in points:
        key = (round(p["lat"], places), round(p["lon"], places))
        seen.setdefault(key, p)
    return list(seen.values())


def build_negatives(positives: List[Dict], boundary, seed: int) -> List[Dict]:
    to_utm = gpd.GeoSeries([Point(p["lon"], p["lat"]) for p in positives], crs="EPSG:4326").to_crs(UTM)
    pos_tree = STRtree(list(to_utm.geometry))
    to_ll = __import__("pyproj").Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    minx, miny, maxx, maxy = boundary.bounds
    rng = np.random.default_rng(seed)
    negs = []
    k = 0
    xs = np.arange(minx, maxx, NEG_GRID_SPACING_M)
    ys = np.arange(miny, maxy, NEG_GRID_SPACING_M)
    for x in xs:
        for y in ys:
            # small jitter so negatives aren't a perfectly regular lattice
            px, py = x + rng.uniform(-60, 60), y + rng.uniform(-60, 60)
            p = Point(px, py)
            if not boundary.covers(p):
                continue
            near = pos_tree.query(p.buffer(NEG_MIN_DIST_M))
            if any(to_utm.geometry.iloc[int(j)].distance(p) < NEG_MIN_DIST_M for j in near):
                continue
            lon, lat = to_ll.transform(px, py)
            negs.append({"point_id": f"neg_{k}", "lat": lat, "lon": lon, "ward": "",
                         "source": "kmc_grid_sample", "label_source": "grid_no_nearby_pocket"})
            k += 1
    return negs


def extract(points: List[Dict], dem: DEMService, spatial: SpatialService) -> List[Dict]:
    coords = [(p["lat"], p["lon"]) for p in points]
    terrain = dem.get_terrain_features(coords)
    feats = spatial.get_spatial_features(coords)
    rows = []
    for p, t, f in zip(points, terrain, feats):
        drain = f["drain"]
        water = f["water_body"]
        rows.append({
            **p,
            "elevation_m": t["elevation_m"], "slope_percent": t["slope_percent"],
            "distance_to_waterbody_m": water.get("distance_to_waterbody_m") if water.get("found") else None,
            "distance_to_drain_m": drain.get("distance_to_drain_m") if drain.get("found") else None,
            "pipe_diameter_mm": drain.get("pipe_diameter_mm") if drain.get("found") else None,
            "drain_capacity_estimated_m3s": drain.get("drain_capacity_estimated_m3s") if drain.get("found") else None,
            "within_mapped_drainage": drain.get("within_mapped_area"),
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-negatives", type=int, default=0, help="cap negatives (0 = all)")
    ap.add_argument("--no-rainfall-scenarios", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    boundary = gpd.read_file(settings.KMC_BOUNDARY_GEOJSON_PATH).to_crs(UTM).union_all()
    dem, spatial = _services()

    positives = dedupe_by_coord(load_positives())
    negatives = build_negatives(positives, boundary, args.seed)
    if args.limit_negatives and len(negatives) > args.limit_negatives:
        rng = np.random.default_rng(args.seed)
        negatives = [negatives[i] for i in sorted(rng.choice(len(negatives), args.limit_negatives, replace=False))]

    pos_rows = extract(positives, dem, spatial)
    for r in pos_rows:
        r[LABEL] = 1
    neg_rows = extract(negatives, dem, spatial)
    for r in neg_rows:
        r[LABEL] = 0
    base_rows = pos_rows + neg_rows

    scenarios = {"none": None} if args.no_rainfall_scenarios else RAINFALL_SCENARIOS
    all_rows = []
    for r in base_rows:
        for name, mm in scenarios.items():
            row = dict(r)
            row["rainfall_scenario"] = name
            row["rainfall_scenario_mm"] = mm
            row["point_id"] = f"{r['point_id']}_{name}"
            all_rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    columns = META_COLUMNS + FEATURE_COLUMNS + [LABEL]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})

    _report(base_rows, all_rows, scenarios)
    print(f"\nwrote {OUT}")
    return 0


def _report(base_rows, all_rows, scenarios) -> None:
    n_pos = sum(r[LABEL] for r in base_rows)
    n_neg = len(base_rows) - n_pos
    print("=== training dataset report ===")
    print(f"unique points: {len(base_rows)}  (positives {n_pos}, negatives {n_neg}, "
          f"pos rate {100 * n_pos / len(base_rows):.1f}%)")
    print(f"rainfall scenarios: {list(scenarios)} -> total rows: {len(all_rows)}")
    print("feature completeness (% non-null over unique points):")
    for c in FEATURE_COLUMNS:
        if c == "rainfall_scenario_mm":
            continue
        non_null = sum(1 for r in base_rows if r.get(c) is not None)
        print(f"  {c:32} {100 * non_null / len(base_rows):5.1f}%")
    drained = [r for r in base_rows if r.get("distance_to_drain_m") is not None]
    print(f"points with drainage features: {len(drained)} "
          f"(positives among them: {sum(r[LABEL] for r in drained)})")
    if n_pos < 30:
        print("\nWARNING: very few positives. Metrics in Task 17 will be unstable; "
              "treat the model as a prototype and report confidence intervals.")


if __name__ == "__main__":
    sys.exit(main())

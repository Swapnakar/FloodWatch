"""
Validate the real training dataset before model training (Task 16).

Runs against data/processed/training_data.csv (from build_training_dataset.py)
and FAILS LOUDLY (exit 1) on any CRITICAL problem, so a broken dataset can never
reach Task 17. Softer concerns are warnings.

Checks
------
Schema
  - all expected meta/feature/label columns present, no unexpected columns
Label
  - label is strictly {0,1}; positives carry the real label_source; both classes present
Coordinates
  - every point is inside the Kolkata bbox (also catches a wrong CRS); finite
Feature integrity
  - terrain (elevation, slope) present for ~all points
  - drainage fields are all-or-nothing per row (never a partial/fabricated set)
  - value ranges: elevation, slope, diameter, distances, rainfall scenario
Class balance
  - both classes present; positive rate within a sane band (warn if extreme)
Leakage
  - CRITICAL: no negative lies within the historical radius of a positive
    (that would label a real flood spot as safe -> label noise + leakage)
  - CRITICAL: a base point (same coords) never appears with BOTH labels
  - duplicate exact coordinates within a class are reported (warning)
Scenario expansion
  - every base point appears exactly once per rainfall scenario

Usage (from backend/):
    ../.venv/bin/python scripts/validate_training_data.py [--json]
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from pyproj import Transformer
from shapely.geometry import Point
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_training_dataset as bt  # noqa: E402
from config import settings  # noqa: E402

TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
KOLKATA_BBOX = (88.20, 22.40, 88.55, 22.75)  # lon_min, lat_min, lon_max, lat_max
DATASET = bt.OUT

# Value ranges (fail loudly outside these).
RANGES = {
    "elevation_m": (-60, 60),
    "slope_percent": (0, 200),
    "distance_to_waterbody_m": (0, 5000),
    "distance_to_drain_m": (0, 5000),
    "pipe_diameter_mm": (100, 3000),
    "drain_capacity_estimated_m3s": (0, 1000),
    "rainfall_scenario_mm": (0, 500),
}
MIN_POSITIVES_WARN = 30
POS_RATE_BAND = (0.005, 0.60)  # outside -> warning
TERRAIN_MIN_COMPLETENESS = 0.95  # elevation/slope must cover ~all points


def _f(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _base_id(point_id: str) -> str:
    """Strip the trailing _<scenario> to recover the underlying point id."""
    pid = point_id
    for scen in list(bt.RAINFALL_SCENARIOS) + ["none"]:
        if pid.endswith("_" + scen):
            return pid[: -(len(scen) + 1)]
    return pid


class Report:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict = {}

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(path: Path = DATASET) -> Report:
    r = Report()
    if not path.exists():
        r.errors.append(f"dataset missing: {path} (run build_training_dataset.py)")
        return r
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        r.errors.append("dataset is empty")
        return r
    r.stats["total_rows"] = len(rows)

    # ---- schema
    expected = set(bt.META_COLUMNS + bt.FEATURE_COLUMNS + [bt.LABEL])
    got = set(rows[0])
    if got != expected:
        if expected - got:
            r.errors.append(f"missing columns: {sorted(expected - got)}")
        if got - expected:
            r.errors.append(f"unexpected columns: {sorted(got - expected)}")

    # ---- label
    labels = Counter(row[bt.LABEL] for row in rows)
    if set(labels) - {"0", "1"}:
        r.errors.append(f"label not strictly 0/1: values {sorted(labels)}")
    for row in rows:
        if row[bt.LABEL] == "1" and row.get("label_source") != "located_historical_pocket":
            r.errors.append(f"positive with wrong label_source at {row['point_id']}: {row.get('label_source')!r}")
            break

    # collapse to base points (dedupe scenario expansion)
    base: Dict[str, Dict] = {}
    scen_seen: Dict[str, set] = defaultdict(set)
    base_labels: Dict[str, set] = defaultdict(set)
    base_coords: Dict[str, set] = defaultdict(set)
    for row in rows:
        bid = _base_id(row["point_id"])
        base.setdefault(bid, row)
        scen_seen[bid].add(row.get("rainfall_scenario"))
        base_labels[bid].add(row.get(bt.LABEL))
        base_coords[bid].add((row.get("lat"), row.get("lon")))
    inconsistent = [bid for bid, s in base_labels.items() if len(s) > 1]
    if inconsistent:
        r.errors.append(f"{len(inconsistent)} points have inconsistent labels across their rows "
                        f"(e.g. {inconsistent[:3]})")
    moved = [bid for bid, s in base_coords.items() if len(s) > 1]
    if moved:
        r.errors.append(f"{len(moved)} points have differing coordinates across their rows (e.g. {moved[:3]})")
    n_base = len(base)
    n_pos = sum(1 for row in base.values() if row[bt.LABEL] == "1")
    n_neg = n_base - n_pos
    r.stats.update(unique_points=n_base, positives=n_pos, negatives=n_neg,
                   positive_rate=round(n_pos / n_base, 4) if n_base else 0)

    if n_pos == 0 or n_neg == 0:
        r.errors.append(f"need both classes; got positives={n_pos}, negatives={n_neg}")
    if n_pos < MIN_POSITIVES_WARN:
        r.warnings.append(f"only {n_pos} positives; model is a prototype, metrics will be unstable")
    if n_base:
        rate = n_pos / n_base
        if not (POS_RATE_BAND[0] <= rate <= POS_RATE_BAND[1]):
            r.warnings.append(f"positive rate {rate:.1%} outside {POS_RATE_BAND}")

    # ---- scenario expansion
    scen_set = set(bt.RAINFALL_SCENARIOS) if any(row.get("rainfall_scenario") in bt.RAINFALL_SCENARIOS
                                                 for row in rows) else {"none"}
    bad_expansion = [bid for bid, s in scen_seen.items() if s != scen_set]
    if bad_expansion:
        r.errors.append(f"{len(bad_expansion)} points not expanded across all scenarios "
                        f"{sorted(scen_set)} (e.g. {bad_expansion[:3]})")

    # ---- coordinates + ranges + feature integrity (per row)
    w, s, e, n = KOLKATA_BBOX
    outside = []
    terrain_complete = 0
    range_violations: Counter = Counter()
    partial_drainage = []
    for row in rows:
        lon, lat = _f(row["lon"]), _f(row["lat"])
        if lon is None or lat is None or not (w <= lon <= e and s <= lat <= n):
            outside.append(row["point_id"])
        if _f(row["elevation_m"]) is not None and _f(row["slope_percent"]) is not None:
            terrain_complete += 1
        for col, (lo, hi) in RANGES.items():
            v = _f(row.get(col))
            if v is not None and not (lo <= v <= hi):
                range_violations[col] += 1
        # drainage all-or-nothing: distance implies diameter-or-explained + capacity handling
        dd = _f(row.get("distance_to_drain_m"))
        pid_present = row.get("pipe_diameter_mm") not in (None, "")
        cap_present = row.get("drain_capacity_estimated_m3s") not in (None, "")
        if dd is None:
            if pid_present or cap_present:
                partial_drainage.append(row["point_id"])
        # capacity present but no diameter is impossible
        if cap_present and not pid_present:
            partial_drainage.append(row["point_id"])

    if outside:
        r.errors.append(f"{len(outside)} rows outside Kolkata bbox (wrong CRS or bad coords): {outside[:5]}")
    for col, cnt in range_violations.items():
        r.errors.append(f"{cnt} rows with {col} outside {RANGES[col]}")
    if partial_drainage:
        r.errors.append(f"{len(partial_drainage)} rows with partial/fabricated drainage fields: "
                        f"{partial_drainage[:5]}")
    r.stats["terrain_completeness"] = round(terrain_complete / len(rows), 4)
    if terrain_complete / len(rows) < TERRAIN_MIN_COMPLETENESS:
        r.errors.append(f"terrain completeness {terrain_complete / len(rows):.1%} < {TERRAIN_MIN_COMPLETENESS:.0%}")

    # feature completeness (over base points), for the report
    completeness = {}
    for col in bt.FEATURE_COLUMNS:
        if col == "rainfall_scenario_mm":
            continue
        nn = sum(1 for row in base.values() if row.get(col) not in (None, ""))
        completeness[col] = round(nn / n_base, 4) if n_base else 0
    r.stats["feature_completeness"] = completeness

    # ---- leakage / duplicate labels
    pos_pts, neg_pts, coord_key = [], [], defaultdict(set)
    for bid, row in base.items():
        lon, lat = _f(row["lon"]), _f(row["lat"])
        if lon is None or lat is None:
            continue
        x, y = TO_UTM.transform(lon, lat)
        (pos_pts if row[bt.LABEL] == "1" else neg_pts).append((Point(x, y), bid))
        coord_key[(round(x, 1), round(y, 1))].add(row[bt.LABEL])

    both = [k for k, v in coord_key.items() if v == {"0", "1"}]
    if both:
        r.errors.append(f"{len(both)} coordinates appear as BOTH positive and negative (label conflict)")

    if pos_pts and neg_pts:
        radius = getattr(settings, "HISTORICAL_WATERLOGGING_RADIUS_M", bt.HISTORICAL_RADIUS_M)
        tree = STRtree([p for p, _ in pos_pts])
        leaked = 0
        for np_, bid in neg_pts:
            for j in tree.query(np_.buffer(radius)):
                if pos_pts[int(j)][0].distance(np_) < radius:
                    leaked += 1
                    break
        r.stats["negatives_within_pos_radius"] = leaked
        if leaked:
            r.errors.append(f"{leaked} negatives lie within {radius:.0f} m of a positive "
                            f"(a real pocket labelled safe -> leakage)")

    # duplicate exact coords (warning: identical rows bias training slightly)
    exact_dups = len(rows) - len({(row["lat"], row["lon"], row["rainfall_scenario"]) for row in rows})
    if exact_dups:
        r.warnings.append(f"{exact_dups} exact duplicate (lat,lon,scenario) rows")

    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--path", type=Path, default=DATASET)
    args = ap.parse_args(argv)
    r = validate(args.path)
    if args.json:
        print(json.dumps({"ok": r.ok, "errors": r.errors, "warnings": r.warnings, "stats": r.stats},
                         indent=2, default=str))
    else:
        print(f"dataset: {args.path}")
        for k, v in r.stats.items():
            print(f"  {k}: {v}")
        for wmsg in r.warnings:
            print(f"  WARN  {wmsg}")
        for e in r.errors:
            print(f"  ERROR {e}")
        print("\nRESULT:", "PASS" if r.ok else "FAIL")
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())

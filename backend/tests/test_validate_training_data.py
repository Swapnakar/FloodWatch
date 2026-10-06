import csv
import sys
from pathlib import Path

import pytest
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_training_dataset as bt  # noqa: E402
import validate_training_data as vt  # noqa: E402

TO_LL = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)
E0, N0 = 644_000.0, 2_491_000.0
COLS = vt.bt.META_COLUMNS + vt.bt.FEATURE_COLUMNS + [vt.bt.LABEL]


def ll(e, n):
    lon, lat = TO_LL.transform(e, n)
    return lat, lon


def base_row(pid, e, n, label, **over):
    lat, lon = ll(e, n)
    row = {c: "" for c in COLS}
    row.update({
        "point_id": pid, "lat": f"{lat:.6f}", "lon": f"{lon:.6f}", "ward": "1",
        "source": "kmc_2017_pocket" if label == 1 else "kmc_grid_sample",
        "label_source": "located_historical_pocket" if label == 1 else "grid_no_nearby_pocket",
        "elevation_m": "8.0", "slope_percent": "1.5", "within_mapped_drainage": "False",
        bt.LABEL: str(label),
    })
    row.update({k: str(v) for k, v in over.items()})
    return row


def expand(base):
    """Expand base points across the rainfall scenarios, as the builder does."""
    out = []
    for b in base:
        for name, mm in bt.RAINFALL_SCENARIOS.items():
            r = dict(b)
            r["point_id"] = f"{b['point_id']}_{name}"
            r["rainfall_scenario"] = name
            r["rainfall_scenario_mm"] = str(mm)
            out.append(r)
    return out


def write(tmp_path, rows):
    p = tmp_path / "train.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return p


def clean_base():
    # 2 positives, 3 negatives, all far apart (>300 m); terrain filled.
    return [
        base_row("pos_0", E0, N0, 1),
        base_row("pos_1", E0 + 2000, N0, 1),
        base_row("neg_0", E0 + 1000, N0 + 1000, 0),
        base_row("neg_1", E0 - 1000, N0 - 1000, 0),
        base_row("neg_2", E0, N0 + 1500, 0),
    ]


def test_clean_dataset_passes(tmp_path):
    r = vt.validate(write(tmp_path, expand(clean_base())))
    assert r.ok, r.errors
    assert r.stats["positives"] == 2 and r.stats["negatives"] == 3


def test_missing_file_fails(tmp_path):
    r = vt.validate(tmp_path / "nope.csv")
    assert not r.ok and "missing" in r.errors[0]


def test_leakage_negative_near_positive_fails(tmp_path):
    base = clean_base()
    base.append(base_row("neg_close", E0 + 100, N0, 0))  # 100 m from pos_0
    r = vt.validate(write(tmp_path, expand(base)))
    assert not r.ok
    assert any("within" in e and "of a positive" in e for e in r.errors)


def test_label_conflict_same_coord_both_classes_fails(tmp_path):
    base = clean_base()
    # same coordinate as pos_0 but labelled negative
    base.append(base_row("neg_dup", E0, N0, 0))
    r = vt.validate(write(tmp_path, expand(base)))
    assert not r.ok
    assert any("BOTH positive and negative" in e for e in r.errors)


def test_only_one_class_fails(tmp_path):
    base = [base_row(f"neg_{i}", E0 + 800 * i, N0, 0) for i in range(4)]
    r = vt.validate(write(tmp_path, expand(base)))
    assert not r.ok and any("both classes" in e for e in r.errors)


def test_out_of_range_elevation_fails(tmp_path):
    base = clean_base()
    base[2]["elevation_m"] = "500"
    r = vt.validate(write(tmp_path, expand(base)))
    assert any("elevation_m outside" in e for e in r.errors)


def test_partial_drainage_fields_fail(tmp_path):
    base = clean_base()
    base[2]["drain_capacity_estimated_m3s"] = "0.03"  # capacity without diameter/distance
    r = vt.validate(write(tmp_path, expand(base)))
    assert any("partial/fabricated drainage" in e for e in r.errors)


def test_coordinates_outside_kolkata_fail(tmp_path):
    base = clean_base()
    base[0]["lat"], base[0]["lon"] = "28.61", "77.20"  # Delhi
    r = vt.validate(write(tmp_path, expand(base)))
    assert any("outside Kolkata bbox" in e for e in r.errors)


def test_positive_wrong_label_source_fails(tmp_path):
    base = clean_base()
    base[0]["label_source"] = "grid_no_nearby_pocket"
    r = vt.validate(write(tmp_path, expand(base)))
    assert any("wrong label_source" in e for e in r.errors)


def test_missing_terrain_fails(tmp_path):
    base = clean_base()
    for b in base:
        b["elevation_m"] = ""  # wipe elevation everywhere
    r = vt.validate(write(tmp_path, expand(base)))
    assert any("terrain completeness" in e for e in r.errors)


def test_per_row_label_flip_within_a_point_fails(tmp_path):
    # A single scenario-row of one point flipped to the other label: caught even
    # though the point's other rows are unchanged.
    rows = expand(clean_base())
    for row in rows:
        if row["point_id"] == "neg_0_heavy":
            row[bt.LABEL] = "1"
    r = vt.validate(write(tmp_path, rows))
    assert not r.ok and any("inconsistent labels" in e for e in r.errors)


def test_incomplete_scenario_expansion_fails(tmp_path):
    rows = expand(clean_base())
    rows = [r for r in rows if not (r["point_id"].startswith("pos_0") and r["rainfall_scenario"] == "extreme")]
    r = vt.validate(write(tmp_path, rows))
    assert any("not expanded across all scenarios" in e for e in r.errors)


def test_unexpected_column_fails(tmp_path):
    p = tmp_path / "x.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS + ["surprise"])
        w.writeheader()
        for row in expand(clean_base()):
            w.writerow({**row, "surprise": "1"})
    r = vt.validate(p)
    assert any("unexpected columns" in e for e in r.errors)


def test_few_positives_warns_not_fails(tmp_path):
    r = vt.validate(write(tmp_path, expand(clean_base())))
    assert r.ok and any("prototype" in w for w in r.warnings)


def test_cli_exit_code(tmp_path):
    base = clean_base()
    base[0]["elevation_m"] = "999"
    assert vt.main(["--path", str(write(tmp_path, expand(base)))]) == 1
    assert vt.main(["--path", str(write(tmp_path, expand(clean_base())))]) == 0


# ---- real dataset ----------------------------------------------------------

@pytest.mark.skipif(not bt.OUT.exists(), reason="run build_training_dataset.py first")
def test_real_training_data_passes():
    r = vt.validate(bt.OUT)
    assert r.ok, r.errors
    assert r.stats["positives"] >= 1 and r.stats["negatives"] >= 1
    assert r.stats["negatives_within_pos_radius"] == 0  # no leakage in the real set
    assert r.stats["terrain_completeness"] >= 0.95


@pytest.mark.skipif(not bt.OUT.exists(), reason="run build_training_dataset.py first")
def test_corrupted_copy_of_real_data_fails_loudly(tmp_path):
    """Corrupt a COPY of the real dataset; each injected fault is reported and
    the real file is never modified."""
    before = bt.OUT.read_bytes()
    with bt.OUT.open() as f:
        rows = list(csv.DictReader(f))
    # relabel one whole point negative->positive at a coordinate a real negative
    # occupies (label conflict) and push another elevation out of range.
    neg = next(r for r in rows if r[bt.LABEL] == "0")
    bad_pid = vt._base_id(neg["point_id"])
    for r in rows:
        if vt._base_id(r["point_id"]) == bad_pid:
            r[bt.LABEL] = "1"
            r["label_source"] = "grid_no_nearby_pocket"  # positive w/ wrong source
    rows[0]["elevation_m"] = "412"

    p = tmp_path / "corrupt.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    r = vt.validate(p)

    assert not r.ok
    errs = "\n".join(r.errors)
    assert "elevation_m outside" in errs
    assert "wrong label_source" in errs
    assert bt.OUT.read_bytes() == before  # real dataset untouched

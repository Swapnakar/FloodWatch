import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import train_real_model as trm  # noqa: E402


def synthetic_df(n_pos=40, n_neg=200, seed=0):
    """A separable-ish dataset: positives sit low + near water, negatives high + far.

    One row per location (as load_dataset produces). Spread across the grid so
    GroupKFold has multiple groups.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pos):
        rows.append({"lat": 22.50 + rng.uniform(0, 0.08), "lon": 88.35 + rng.uniform(0, 0.08),
                     "elevation_m": rng.uniform(3, 7), "slope_percent": rng.uniform(0, 2),
                     "distance_to_drain_m": "", "pipe_diameter_mm": "",
                     "drain_capacity_estimated_m3s": "", "distance_to_waterbody_m": rng.uniform(10, 120),
                     trm.LABEL: 1})
    for i in range(n_neg):
        rows.append({"lat": 22.50 + rng.uniform(0, 0.08), "lon": 88.35 + rng.uniform(0, 0.08),
                     "elevation_m": rng.uniform(9, 18), "slope_percent": rng.uniform(2, 8),
                     "distance_to_drain_m": "", "pipe_diameter_mm": "",
                     "drain_capacity_estimated_m3s": "", "distance_to_waterbody_m": rng.uniform(500, 3000),
                     trm.LABEL: 0})
    return pd.DataFrame(rows)


def test_spatial_groups_are_stable_and_grid_based():
    lats = np.array([22.50, 22.5001, 22.60])
    lons = np.array([88.40, 88.4001, 88.45])
    g = trm._spatial_groups(lats, lons, cell_m=500)
    assert g[0] == g[1]      # points ~10 m apart share a cell
    assert g[0] != g[2]      # points ~11 km apart do not


def test_train_and_evaluate_reports_all_metrics():
    df = synthetic_df()
    res = trm.train_and_evaluate(df)
    assert set(res["cv_summary"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    for m in res["cv_summary"].values():
        assert "mean" in m and "std" in m and len(m["per_fold"]) >= 2
    # separable data -> AUC clearly above chance
    assert res["cv_summary"]["roc_auc"]["mean"] > 0.7
    assert set(res["feature_importance"]) == set(trm.FEATURE_COLS)


def test_rainfall_not_a_feature():
    # Static-susceptibility model must not train on rainfall.
    assert "rainfall_scenario_mm" not in trm.FEATURE_COLS


def test_load_dataset_collapses_scenarios(tmp_path, monkeypatch):
    # Two locations, each duplicated across 4 scenarios -> 8 rows -> 2 after load.
    base = []
    for k, (lat, lon, lab) in enumerate([(22.51, 88.40, 1), (22.55, 88.42, 0)]):
        for scen, mm in [("dry", 0), ("moderate", 40), ("heavy", 90), ("extreme", 150)]:
            base.append({"point_id": f"p{k}_{scen}", "lat": lat, "lon": lon, "ward": "1",
                         "source": "s", "rainfall_scenario": scen, "label_source": "x",
                         "elevation_m": 8, "slope_percent": 1, "distance_to_waterbody_m": "",
                         "distance_to_drain_m": "", "pipe_diameter_mm": "",
                         "drain_capacity_estimated_m3s": "", "within_mapped_drainage": "False",
                         "rainfall_scenario_mm": mm, trm.LABEL: lab})
    p = tmp_path / "d.csv"
    pd.DataFrame(base).to_csv(p, index=False)
    monkeypatch.setattr(trm, "DATASET", p)
    df = trm.load_dataset()
    assert len(df) == 2  # collapsed to unique locations


def test_full_run_writes_model_and_metadata_without_touching_synthetic(tmp_path, monkeypatch):
    # Build a valid dataset CSV and point the trainer at it + a temp model dir.
    df = synthetic_df()
    ds = tmp_path / "training_data.csv"
    df.to_csv(ds, index=False)
    monkeypatch.setattr(trm, "DATASET", ds)
    monkeypatch.setattr(trm, "MODEL_DIR", tmp_path / "model")
    # skip the strict training-data validator (its schema differs from this minimal CSV)
    monkeypatch.setattr(trm, "load_dataset", lambda: trm.pd.read_csv(ds).drop_duplicates(["lat", "lon"]))
    import validate_training_data as vtd

    class OK:
        ok = True
        errors = []
    monkeypatch.setattr(vtd, "validate", lambda *_a, **_k: OK())

    assert trm.main() == 0
    model_path = tmp_path / "model" / "real_flood_probability_model.json"
    meta_path = tmp_path / "model" / "real_model_metadata.json"
    assert model_path.exists() and meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["model_version"] == "real_v1"
    assert meta["training_data_type"] == "real"
    assert meta["features_used"] == trm.FEATURE_COLS
    assert "prototype" in " ".join(meta["caveats"]).lower()

    # The loaded model predicts sane probabilities.
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(str(model_path))
    probs = m.predict_proba(df[trm.FEATURE_COLS].apply(pd.to_numeric, errors="coerce").values)[:, 1]
    assert probs.min() >= 0 and probs.max() <= 1


# ---- real trained model (opt-in on the built artifacts) --------------------

MODEL_DIR = trm.DATA_DIR.parent / "model"
REAL_META = MODEL_DIR / "real_model_metadata.json"
SYNTH_META = MODEL_DIR / "model_metadata.json"


@pytest.mark.skipif(not REAL_META.exists(), reason="run train_real_model.py first")
def test_real_metadata_is_honest():
    meta = json.loads(REAL_META.read_text())
    assert meta["model_version"] == "real_v1" and meta["training_data_type"] == "real"
    auc = meta["cv_metrics"]["roc_auc"]["mean"]
    assert 0.4 <= auc <= 0.95  # a real, modest number -- not the synthetic 0.97
    assert meta["unique_positives"] >= 1


@pytest.mark.skipif(not SYNTH_META.exists(), reason="synthetic metadata missing")
def test_synthetic_model_untouched():
    meta = json.loads(SYNTH_META.read_text())
    # The old synthetic metadata must be intact and clearly a different provenance.
    assert meta["probability_model"]["auc_roc"] == 0.9716
    assert "rainfall_1h" in meta["features"]


@pytest.mark.skipif(not (MODEL_DIR / "real_flood_probability_model.json").exists(),
                    reason="run train_real_model.py first")
def test_real_model_loads_and_predicts():
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(str(MODEL_DIR / "real_flood_probability_model.json"))
    meta = json.loads(REAL_META.read_text())
    X = np.array([[6.0, 1.0, np.nan, np.nan, np.nan, 50.0],      # low + near water
                  [16.0, 6.0, np.nan, np.nan, np.nan, 2000.0]])  # high + far
    assert X.shape[1] == len(meta["features_used"])
    probs = m.predict_proba(X)[:, 1]
    assert all(0 <= p <= 1 for p in probs)

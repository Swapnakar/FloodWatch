"""
Train the real flood-probability classifier (Task 17, version real_v1).

Label: historical_waterlogging (binary, from KMC 2017 pocket list).
Features: terrain (elevation, slope), drainage (sparse), rainfall_scenario_mm.
Split: spatial K-fold (GroupKFold by grid cell) so nearby points don't leak.

With only ~126 positives, a single hold-out set gives noisy metrics. So this
uses 5-fold spatial CV over all the data, and reports mean ± std for each metric.
A final model is trained on ALL data (no hold-out) for production serving, with
clear documentation that the metrics are cross-validated estimates, not hold-out.

The old synthetic models (flood_depth_model.json, flood_probability_model.json,
model_metadata.json) are NOT touched.

Output:
  model/real_flood_probability_model.json    XGBClassifier saved model
  model/real_model_metadata.json             version, features, metrics, caveats
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR  # noqa: E402

DATASET = DATA_DIR / "processed" / "training_data.csv"
MODEL_DIR = DATA_DIR.parent / "model"

# The label (historical_waterlogging) is a STATIC susceptibility label, not a
# dated event, so it is independent of rainfall. rainfall_scenario_mm is
# therefore NOT a training feature here (it carried 0 importance and only
# 4x-replicated each location). Rainfall belongs at inference time as a
# same-day severity multiplier, not in this static-susceptibility model.
FEATURE_COLS = [
    "elevation_m", "slope_percent",
    "distance_to_drain_m", "pipe_diameter_mm", "drain_capacity_estimated_m3s",
    "distance_to_waterbody_m",
]
LABEL = "historical_waterlogging"
SPATIAL_CELL_M = 500  # grid cell size for grouping in spatial CV
N_FOLDS = 5


def _spatial_groups(lats: np.ndarray, lons: np.ndarray, cell_m: float) -> np.ndarray:
    """Assign each point to a spatial grid cell for GroupKFold."""
    from pyproj import Transformer
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    xs, ys = to_utm.transform(lons, lats)
    gi = (xs // cell_m).astype(int)
    gj = (ys // cell_m).astype(int)
    # Encode 2D grid into a 1D group id.
    return gi * 1_000_000 + gj


def load_dataset():
    df = pd.read_csv(DATASET)
    df[LABEL] = df[LABEL].astype(int)
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # The label is static (rainfall-independent), so collapse the rainfall-scenario
    # expansion to ONE row per location. Otherwise every point is counted 4x.
    df = df.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    return df


def train_and_evaluate(df: pd.DataFrame) -> dict:
    # Coerce features to float here too (empty/sparse cells -> NaN), so the
    # function is correct regardless of how the caller built the frame.
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    y = df[LABEL].astype(int).to_numpy()
    groups = _spatial_groups(df["lat"].values, df["lon"].values, SPATIAL_CELL_M)

    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    scale = neg_count / max(pos_count, 1)

    xgb_params = {
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
        "scale_pos_weight": scale, "eval_metric": "logloss",
        "use_label_encoder": False, "n_jobs": 1, "random_state": 42,
        "tree_method": "hist",
    }

    # Cross-validated metrics (spatial folds).
    gkf = GroupKFold(n_splits=min(N_FOLDS, len(set(groups))))
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(set(y_te)) < 2:
            continue  # skip folds with only one class (small data)
        m = XGBClassifier(**xgb_params)
        m.fit(X_tr, y_tr)
        prob = m.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        fold_metrics.append({
            "fold": fold, "test_size": len(y_te),
            "test_positives": int(y_te.sum()), "test_negatives": int(len(y_te) - y_te.sum()),
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred, zero_division=0),
            "f1": f1_score(y_te, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_te, prob),
        })
    if not fold_metrics:
        raise RuntimeError("no valid folds (too few points or no class variation)")

    cv_summary = {}
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        vals = [fm[metric] for fm in fold_metrics]
        cv_summary[metric] = {"mean": round(float(np.mean(vals)), 4),
                              "std": round(float(np.std(vals)), 4),
                              "per_fold": [round(v, 4) for v in vals]}

    # Final model on ALL data for production serving.
    final = XGBClassifier(**xgb_params)
    final.fit(X, y)
    importances = {c: round(float(v), 4) for c, v in zip(FEATURE_COLS, final.feature_importances_)}

    # Confusion matrix on full refit (NOT a metric; just for shape inspection).
    full_pred = (final.predict_proba(X)[:, 1] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, full_pred).ravel()

    return {
        "model": final,
        "xgb_params": xgb_params,
        "fold_metrics": fold_metrics,
        "cv_summary": cv_summary,
        "feature_importance": importances,
        "refit_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def save(result: dict, df: pd.DataFrame):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_file = "real_flood_probability_model.json"
    result["model"].save_model(str(MODEL_DIR / model_file))

    unique = df.drop_duplicates(subset=["lat", "lon"])
    unique_pos = int(unique[LABEL].sum())
    unique_neg = len(unique) - unique_pos

    import datetime
    meta = {
        "model_version": "real_v1",
        "prediction_type": "flood_probability",
        "training_data_type": "real",
        "last_trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "training_data_source": "KMC 2017 Major Water Logging Pockets + DEM + KMC drainage",
        "prediction_meaning": "static flood susceptibility (probability a location is a waterlogging-prone spot), "
                              "NOT a same-day forecast",
        "label": LABEL,
        "label_source": "KMC 'Action Plan to Mitigate Flood, Cyclone & Water Logging 2017', Section C",
        "features_used": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "training_rows": len(df),
        "unique_locations": len(unique),
        "unique_positives": unique_pos,
        "unique_negatives": unique_neg,
        "class_balance": f"{unique_pos} positive / {unique_neg} negative (pos rate {unique_pos / len(unique):.1%})",
        "spatial_cv_folds": len(result["fold_metrics"]),
        "spatial_cv_cell_m": SPATIAL_CELL_M,
        "cv_metrics": result["cv_summary"],
        "feature_importance": result["feature_importance"],
        "refit_confusion_matrix": result["refit_confusion"],
        "xgb_params": result["xgb_params"],
        "model_file": model_file,
        "caveats": [
            f"Only {unique_pos} distinct positive locations. Metrics are cross-validated but unstable; "
            "treat this model as a prototype, not a production-quality classifier.",
            "Drainage features are sparse (~8% non-null). XGBoost handles the missing values; "
            "the model leans on terrain (elevation, slope) and distance to water.",
            "Label is 'listed major pocket', not a dated flood event. A positive means 'KMC listed "
            "this spot in 2017'; a negative means 'not listed in the covered area'.",
            "This is a static flood-SUSCEPTIBILITY model. It does not use rainfall (the label is "
            "rainfall-independent). Same-day rainfall is applied at inference as a severity multiplier.",
            "One row per location (rainfall-scenario expansion collapsed) so CV counts each place once.",
        ],
    }
    (MODEL_DIR / "real_model_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    if not DATASET.exists():
        print(f"ERROR: dataset not found at {DATASET}; run build_training_dataset.py first")
        return 1
    # Validate first.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import validate_training_data as vtd
    r = vtd.validate(DATASET)
    if not r.ok:
        print("ERROR: training data validation failed:")
        for e in r.errors:
            print(f"  {e}")
        return 1

    df = load_dataset()
    print(f"dataset: {len(df)} rows, {int(df[LABEL].sum())} positive, "
          f"{len(df) - int(df[LABEL].sum())} negative")
    result = train_and_evaluate(df)
    meta = save(result, df)

    print("\n=== real_v1 cross-validated metrics ===")
    for metric, vals in meta["cv_metrics"].items():
        print(f"  {metric:12} {vals['mean']:.4f} ± {vals['std']:.4f}  {vals['per_fold']}")
    print("\nfeature importance:")
    for feat, imp in sorted(meta["feature_importance"].items(), key=lambda kv: -kv[1]):
        print(f"  {feat:36} {imp:.4f}")
    print("\nrefit confusion matrix:", meta["refit_confusion_matrix"])
    print("\ncaveats:")
    for c in meta["caveats"]:
        print(f"  - {c}")
    print(f"\nsaved: {MODEL_DIR / meta['model_file']} + {MODEL_DIR / 'real_model_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

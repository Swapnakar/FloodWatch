"""
FloodWatch — XGBoost Model Training
=====================================
Trains two models:
  1. Water depth regressor  (XGBRegressor)
  2. Flood probability classifier (XGBClassifier → predict_proba)

Loads training_data.csv, trains, evaluates, and saves models.
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBRegressor, XGBClassifier


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "training_data.csv")
MODEL_DIR = SCRIPT_DIR

DEPTH_MODEL_PATH = os.path.join(MODEL_DIR, "flood_depth_model.json")
PROB_MODEL_PATH = os.path.join(MODEL_DIR, "flood_probability_model.json")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# ─────────────────────────────────────────────
# Feature columns (what the model sees)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "rainfall_30m",
    "rainfall_1h",
    "rainfall_3h",
    "horizon_minutes",
    "elevation",
    "slope",
    "imperviousness",
    "drain_capacity",
    "pipe_diameter",
    "distance_to_drain",
    "historical_floods",
]


def main():
    # ── Load data ──
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    X = df[FEATURE_COLS]
    y_depth = df["water_depth_cm"]
    y_flood = (df["water_depth_cm"] > 10).astype(int)  # binary: flood or not

    # ── Train/test split (same split for both models) ──
    X_train, X_test, y_depth_train, y_depth_test, y_flood_train, y_flood_test = (
        train_test_split(
            X,
            y_depth,
            y_flood,
            test_size=0.2,
            random_state=42,
        )
    )

    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # ═══════════════════════════════════════════
    # MODEL 1: Water Depth Regressor
    # ═══════════════════════════════════════════
    print("\n" + "=" * 50)
    print("Training Water Depth Regressor...")
    print("=" * 50)

    depth_model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    depth_model.fit(
        X_train,
        y_depth_train,
        eval_set=[(X_test, y_depth_test)],
        verbose=50,
    )

    depth_preds = depth_model.predict(X_test)
    depth_mae = mean_absolute_error(y_depth_test, depth_preds)
    depth_r2 = r2_score(y_depth_test, depth_preds)

    print(f"\n  MAE:  {depth_mae:.2f} cm")
    print(f"  R²:   {depth_r2:.4f}")

    # Save
    depth_model.save_model(DEPTH_MODEL_PATH)
    print(f"  Saved: {DEPTH_MODEL_PATH}")

    # ═══════════════════════════════════════════
    # MODEL 2: Flood Probability Classifier
    # ═══════════════════════════════════════════
    print("\n" + "=" * 50)
    print("Training Flood Probability Classifier...")
    print("=" * 50)

    prob_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    prob_model.fit(
        X_train,
        y_flood_train,
        eval_set=[(X_test, y_flood_test)],
        verbose=50,
    )

    prob_preds = prob_model.predict(X_test)
    prob_proba = prob_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_flood_test, prob_preds)
    auc = roc_auc_score(y_flood_test, prob_proba)

    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_flood_test, prob_preds, target_names=["No Flood", "Flood"]))

    # Save
    prob_model.save_model(PROB_MODEL_PATH)
    print(f"  Saved: {PROB_MODEL_PATH}")

    # ═══════════════════════════════════════════
    # Feature importances
    # ═══════════════════════════════════════════
    print("\n" + "=" * 50)
    print("Feature Importances (Depth Model)")
    print("=" * 50)

    importances = depth_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    feature_importance = {}
    for idx in sorted_idx:
        name = FEATURE_COLS[idx]
        score = float(importances[idx])
        feature_importance[name] = round(score, 4)
        bar = "█" * int(score * 50)
        print(f"  {name:<22} {score:.4f} {bar}")

    # ═══════════════════════════════════════════
    # Save metadata
    # ═══════════════════════════════════════════
    metadata = {
        "model_type": "XGBoost",
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "depth_model": {
            "file": "flood_depth_model.json",
            "mae_cm": round(depth_mae, 2),
            "r2": round(depth_r2, 4),
            "n_estimators": 500,
            "max_depth": 6,
        },
        "probability_model": {
            "file": "flood_probability_model.json",
            "accuracy": round(accuracy, 4),
            "auc_roc": round(auc, 4),
            "n_estimators": 300,
            "max_depth": 5,
        },
        "feature_importance": feature_importance,
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Metadata saved: {META_PATH}")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()

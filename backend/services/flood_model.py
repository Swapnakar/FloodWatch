"""
real_v1 flood-susceptibility model: load + per-segment inference.

Loads model/real_flood_probability_model.json (trained in Task 17) and scores
route segments from the features the route risk engine gathers.

Key honesty points:
  - This model predicts static SUSCEPTIBILITY (is this a waterlogging-prone
    spot), not a same-day forecast. Same-day rainfall is applied SEPARATELY as
    a severity multiplier here, not baked into the model.
  - A segment with no usable features (e.g. outside the DEM) gets
    flood_probability=None, never a fabricated score.
  - risk_level thresholds are documented and never promise "safe".
"""

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_FILE = MODEL_DIR / "real_flood_probability_model.json"
META_FILE = MODEL_DIR / "real_model_metadata.json"

# Must match FEATURE_COLS in scripts/train_real_model.py (order matters).
FEATURE_ORDER = [
    "elevation_m", "slope_percent",
    "distance_to_drain_m", "pipe_diameter_mm", "drain_capacity_estimated_m3s",
    "distance_to_waterbody_m",
]

# The route risk engine emits segment dicts with slightly different key names
# (elevation, slope) than the trained model's feature names. Map explicitly so a
# rename on either side fails a test rather than silently zeroing a feature.
SEGMENT_KEY_FOR_FEATURE = {
    "elevation_m": "elevation",
    "slope_percent": "slope",
    "distance_to_drain_m": "distance_to_drain_m",
    "pipe_diameter_mm": "pipe_diameter_mm",
    "drain_capacity_estimated_m3s": "drain_capacity_estimated_m3s",
    "distance_to_waterbody_m": "distance_to_waterbody_m",
}

# Risk bands on the (rainfall-adjusted) probability. Deliberately no "SAFE"
# band: the lowest level is LOW ("lower relative risk"), never "no risk".
RISK_BANDS = [(0.70, "HIGH"), (0.45, "ELEVATED"), (0.25, "MODERATE"), (0.0, "LOW")]

# Same-day rainfall severity multiplier applied to the static susceptibility.
# Susceptibility says WHERE floods; rainfall says WHETHER today is wet. A dry
# day scales risk down, a heavy day leaves it as-is. Documented, not a forecast.
RAINFALL_REFERENCE_MM = 90.0  # "heavy" day = full susceptibility


def rainfall_multiplier(rainfall_24h_mm: Optional[float]) -> float:
    """0.3 (dry) .. 1.0 (>= heavy). None -> 1.0 (no down-weighting when unknown)."""
    if rainfall_24h_mm is None:
        return 1.0
    try:
        mm = float(rainfall_24h_mm)
    except (TypeError, ValueError):
        return 1.0
    if mm < 0:
        return 1.0
    frac = min(mm / RAINFALL_REFERENCE_MM, 1.0)
    return 0.3 + 0.7 * frac


def classify_risk(prob: Optional[float]) -> Optional[str]:
    if prob is None:
        return None
    for threshold, label in RISK_BANDS:
        if prob >= threshold:
            return label
    return "LOW"


class FloodModel:
    def __init__(self, model_path: Path = MODEL_FILE, meta_path: Path = META_FILE):
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.status = "MISSING"
        self.error: Optional[str] = None
        self._model = None
        self.metadata: Dict = {}
        self._load()

    def _load(self):
        if not self.model_path.exists():
            self.error = f"real model not found: {self.model_path.name}"
            logger.warning(self.error)
            return
        try:
            from xgboost import XGBClassifier
            m = XGBClassifier(n_jobs=1)
            m.load_model(str(self.model_path))
            self._model = m
            if self.meta_path.exists():
                self.metadata = json.loads(self.meta_path.read_text())
            self.status = "LOADED"
        except Exception as exc:
            self.status = "ERROR"
            self.error = f"{type(exc).__name__}: {exc}"
            logger.error("real model load failed: %s", self.error)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def status_report(self) -> Dict:
        r = {"status": self.status, "model_file": self.model_path.name,
             "model_version": self.metadata.get("model_version", "real_v1" if self.loaded else None),
             "features_used": self.metadata.get("features_used", FEATURE_ORDER)}
        if self.error:
            r["error"] = self.error
        return r

    def _features_for(self, seg: Dict) -> List[float]:
        row = []
        for f in FEATURE_ORDER:
            # accept either the model feature name or the route-segment key name
            v = seg.get(f, seg.get(SEGMENT_KEY_FOR_FEATURE.get(f, f)))
            try:
                row.append(float(v) if v is not None else math.nan)
            except (TypeError, ValueError):
                row.append(math.nan)
        return row

    def score_segments(self, segments: List[Dict], rainfall_24h_mm: Optional[float] = None, horizon_minutes: int = 0) -> List[Dict]:
        """Return each segment with flood_probability (susceptibility x rainfall)
        and risk_level set. Segments lacking ALL features -> probability None.

        `segments` are dicts from RouteRiskService (feature keys per FEATURE_ORDER).
        """
        out = [dict(s) for s in segments]
        if not self.loaded or not out:
            for s in out:
                s["flood_probability"] = None
                s["risk_level"] = None
                s["susceptibility"] = None
            return out

        X = np.array([self._features_for(s) for s in out], dtype="float64")
        # A row with no usable features at all -> None (never fabricate).
        all_nan = np.all(np.isnan(X), axis=1)
        probs = self._model.predict_proba(X)[:, 1]
        mult = rainfall_multiplier(rainfall_24h_mm)
        
        # Simple horizon scaling: Risk increases by 10% per hour if it's currently raining, else decays or stays flat.
        horizon_factor = 1.0 + (horizon_minutes / 60.0) * 0.10
        
        for i, s in enumerate(out):
            if all_nan[i]:
                s["susceptibility"] = None
                s["flood_probability"] = None
                s["risk_level"] = None
                continue
            susceptibility = round(float(probs[i]), 4)
            adjusted = round(min(susceptibility * mult * horizon_factor, 1.0), 4)
            s["susceptibility"] = susceptibility
            s["flood_probability"] = adjusted
            s["risk_level"] = classify_risk(adjusted)
        return out


@lru_cache(maxsize=1)
def get_flood_model() -> FloodModel:
    return FloodModel()

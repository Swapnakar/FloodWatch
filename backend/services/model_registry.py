"""
Single source of truth for model version descriptors (Task 19, section 40).

Every API response that exposes model provenance uses these descriptors, so the
four required fields are always present and the two model types can NEVER be
mislabelled:

    model_version        "synthetic_v1" | "real_v1"
    training_data_type   "synthetic (physics simulation)" | "real (...)"
    features_used        list[str]
    last_trained_at      ISO-8601 UTC (from metadata, or the model file mtime)

The descriptors also carry prediction_type so a synthetic depth estimate is
never presented as a real flood-probability output, and vice versa.
"""

import datetime
import json
from pathlib import Path
from typing import Dict, Optional

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

SYNTHETIC_META = MODEL_DIR / "model_metadata.json"
SYNTHETIC_PROB_MODEL = MODEL_DIR / "flood_probability_model.json"
REAL_META = MODEL_DIR / "real_model_metadata.json"
REAL_MODEL = MODEL_DIR / "real_flood_probability_model.json"

SYNTHETIC_FEATURES = [
    "rainfall_30m", "rainfall_1h", "rainfall_3h", "horizon_minutes", "elevation",
    "slope", "imperviousness", "drain_capacity", "pipe_diameter",
    "distance_to_drain", "historical_floods",
]


def _file_mtime_iso(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return datetime.datetime.fromtimestamp(
        path.stat().st_mtime, datetime.timezone.utc).isoformat(timespec="seconds")


def _load(path: Path) -> Dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (ValueError, OSError):
        return {}


def synthetic_descriptor() -> Dict:
    """Legacy physics-simulation model. Its metadata predates last_trained_at,
    so that timestamp falls back to the model file's mtime."""
    meta = _load(SYNTHETIC_META)
    return {
        "model_version": "synthetic_v1",
        "prediction_type": "prototype_estimated_depth",
        "training_data_type": "synthetic (physics simulation)",
        "features_used": meta.get("features", SYNTHETIC_FEATURES),
        "last_trained_at": meta.get("last_trained_at") or _file_mtime_iso(SYNTHETIC_PROB_MODEL),
        "note": "Prototype trained on synthetic data; metrics are synthetic-vs-synthetic.",
    }


def real_descriptor() -> Dict:
    """Real-data flood-susceptibility model (real_v1). available=False when the
    model file is missing (never falsely reported as trained)."""
    meta = _load(REAL_META)
    available = REAL_MODEL.exists()
    return {
        "model_version": meta.get("model_version", "real_v1"),
        "prediction_type": meta.get("prediction_type", "flood_probability"),
        "training_data_type": meta.get("training_data_type", "real") if available else None,
        "features_used": meta.get("features_used", []),
        "last_trained_at": meta.get("last_trained_at") or (_file_mtime_iso(REAL_MODEL) if available else None),
        "prediction_meaning": meta.get("prediction_meaning",
                                       "static flood susceptibility x same-day rainfall; not a guarantee"),
        "available": available,
    }


REQUIRED_FIELDS = ("model_version", "training_data_type", "features_used", "last_trained_at")


def descriptor_for(version: str) -> Dict:
    if version == "synthetic_v1":
        return synthetic_descriptor()
    if version == "real_v1":
        return real_descriptor()
    raise ValueError(f"unknown model version: {version}")

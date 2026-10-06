import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from services.flood_model import (
    FloodModel, classify_risk, get_flood_model, rainfall_multiplier, FEATURE_ORDER,
)


def seg(**kw):
    # Use the model's own feature names; _features_for accepts these directly.
    base = {f: None for f in FEATURE_ORDER}
    base.update(kw)
    return base


# ---- rainfall multiplier ----------------------------------------------------

@pytest.mark.parametrize("mm,expected", [
    (None, 1.0), (0, 0.3), (90, 1.0), (200, 1.0), (45, 0.65), (-5, 1.0), ("x", 1.0),
])
def test_rainfall_multiplier(mm, expected):
    assert rainfall_multiplier(mm) == pytest.approx(expected, abs=1e-9)


# ---- risk banding -----------------------------------------------------------

@pytest.mark.parametrize("p,level", [
    (None, None), (0.9, "HIGH"), (0.7, "HIGH"), (0.5, "ELEVATED"),
    (0.3, "MODERATE"), (0.1, "LOW"), (0.0, "LOW"),
])
def test_classify_risk(p, level):
    assert classify_risk(p) == level


def test_no_safe_band_exists():
    # There must be no band that says "SAFE"/"NO RISK".
    from services.flood_model import RISK_BANDS
    assert all(label not in ("SAFE", "NONE", "NO_RISK") for _, label in RISK_BANDS)


# ---- scoring with the real model (if present) -------------------------------

real = pytest.mark.skipif(not FloodModel().loaded, reason="real_v1 model not trained")


@real
def test_segments_get_probability_and_risk():
    m = get_flood_model()
    out = m.score_segments([seg(elevation_m=6.0, slope_percent=1.0, distance_to_waterbody_m=40.0)],
                           rainfall_24h_mm=90)
    s = out[0]
    assert s["susceptibility"] is not None and 0 <= s["susceptibility"] <= 1
    assert 0 <= s["flood_probability"] <= 1 and s["risk_level"] in ("LOW", "MODERATE", "ELEVATED", "HIGH")


@real
def test_all_null_features_gives_none_probability():
    out = get_flood_model().score_segments([seg()], rainfall_24h_mm=90)
    assert out[0]["flood_probability"] is None and out[0]["risk_level"] is None


@real
def test_dry_day_scales_probability_down():
    m = get_flood_model()
    wet = m.score_segments([seg(elevation_m=6.0, slope_percent=1.0, distance_to_waterbody_m=40.0)], 90)[0]
    dry = m.score_segments([seg(elevation_m=6.0, slope_percent=1.0, distance_to_waterbody_m=40.0)], 0)[0]
    assert dry["susceptibility"] == wet["susceptibility"]      # same place
    assert dry["flood_probability"] < wet["flood_probability"]  # dry day -> lower risk


def test_missing_model_scores_none(tmp_path):
    m = FloodModel(model_path=tmp_path / "nope.json", meta_path=tmp_path / "nope_meta.json")
    assert m.status == "MISSING"
    out = m.score_segments([seg(elevation_m=6.0)], rainfall_24h_mm=90)
    assert out[0]["flood_probability"] is None and out[0]["risk_level"] is None


@real
def test_scores_from_route_segment_key_names():
    """Regression: the route engine emits 'elevation'/'slope', not the model's
    'elevation_m'/'slope_percent'. Scoring must still work via the key mapping."""
    route_segment = {"lat": 22.57, "lng": 88.40, "elevation": 6.0, "slope": 1.0,
                     "distance_to_waterbody_m": 40.0, "distance_to_drain_m": None,
                     "pipe_diameter_mm": None, "drain_capacity_estimated_m3s": None}
    out = get_flood_model().score_segments([route_segment], rainfall_24h_mm=60)
    assert out[0]["flood_probability"] is not None  # would be None if the mapping broke
    assert out[0]["risk_level"] in ("LOW", "MODERATE", "ELEVATED", "HIGH")

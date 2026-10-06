"""Task 19: model versioning metadata.

The four required fields from section 40 must be present and correct on every
endpoint that carries model provenance. Synthetic and real must NEVER be
cross-labelled.
"""

from fastapi.testclient import TestClient

import main
from services.model_registry import (
    REQUIRED_FIELDS, descriptor_for, real_descriptor, synthetic_descriptor,
)


REQUIRED = set(REQUIRED_FIELDS)


# --------------------------------------------------------- registry tests

def test_synthetic_descriptor_has_all_fields():
    d = synthetic_descriptor()
    assert REQUIRED <= set(d)
    assert d["model_version"] == "synthetic_v1"
    assert "synthetic" in d["training_data_type"].lower()
    assert "rainfall_1h" in d["features_used"]  # a known synthetic feature
    assert d["last_trained_at"] is not None


def test_real_descriptor_has_all_fields():
    d = real_descriptor()
    assert REQUIRED <= set(d)
    assert d["model_version"] == "real_v1"
    if d["available"]:
        assert "real" in d["training_data_type"].lower()
        assert "elevation_m" in d["features_used"]
        assert d["last_trained_at"] is not None
    else:
        assert d["training_data_type"] is None


def test_descriptors_never_share_model_version():
    assert synthetic_descriptor()["model_version"] != real_descriptor()["model_version"]


def test_features_lists_differ():
    assert set(synthetic_descriptor()["features_used"]) != set(real_descriptor()["features_used"])


def test_unknown_version_raises():
    import pytest
    with pytest.raises(ValueError):
        descriptor_for("imaginary_v3")


# --------------------------------------------------------- endpoint tests

c = TestClient(main.app)


def test_model_info_carries_all_four_fields():
    r = c.get("/api/model/info").json()
    assert REQUIRED <= set(r)
    assert r["model_version"] == "synthetic_v1"
    assert "synthetic" in r["training_data_type"].lower()
    assert isinstance(r["features_used"], list) and len(r["features_used"]) > 0
    assert r["last_trained_at"] is not None


def test_predict_response_tagged_synthetic():
    r = c.post("/api/predict", json={}).json()
    assert r["model_version"] == "synthetic_v1"
    assert r["prediction_type"] == "prototype_estimated_depth"


def test_nowcast_response_tagged_legacy():
    r = c.get("/api/nowcast").json()
    assert r["model_version"] == "legacy_threshold_table"
    assert r["legacy"] is True


def test_data_status_both_models_labelled():
    r = c.get("/api/data/status").json()["model"]
    syn = r["synthetic_v1"]
    real = r["real_v1"]
    for d in (syn, real):
        assert REQUIRED <= set(d)
    assert syn["model_version"] == "synthetic_v1" and syn["training_data_type"].startswith("synthetic")
    assert real["model_version"] == "real_v1"
    if real.get("available"):
        assert real["training_data_type"].startswith("real")
    # they must never share a version string
    assert syn["model_version"] != real["model_version"]


def test_synthetic_response_never_labelled_real():
    """Section 40: synthetic-model responses never mislabelled as real."""
    for r in (c.post("/api/predict", json={}).json(), c.get("/api/model/info").json()):
        assert r["model_version"] != "real_v1"
        assert "real" not in r.get("training_data_type", "").lower()

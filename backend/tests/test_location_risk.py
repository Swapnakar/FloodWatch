"""Tests for /api/location-risk: single-point real-data flood susceptibility.

Honesty: real prediction inside KMC, terrain-only just outside, no_data far away.
IMD is mocked so rainfall doesn't depend on the machine's rotating IP.
"""

import base64
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from services.imd_service import IMDService


@pytest.fixture
def client(monkeypatch):
    def imd_handler(request):
        if request.url.path.endswith("token.php"):
            seg0 = base64.urlsafe_b64encode(
                json.dumps({"uid": 1, "exp": int(time.time() + 3600)}).encode()).rstrip(b"=").decode()
            return httpx.Response(200, json={"access_token": f"{seg0}.sig"})
        return httpx.Response(200, json=[{"Station": "KOLKATA", "Temperature": "28",
                                          "Last 24 hrs Rainfall": "40", "Date of Observation": "2026-09-27"}])
    import services.imd_service as ims
    monkeypatch.setattr(ims, "_service",
                        IMDService("k", "e@x.gov.in", "pw",
                                   client=httpx.AsyncClient(transport=httpx.MockTransport(imd_handler), timeout=5)))
    return TestClient(main.app)


def _get(client, lat, lon):
    r = client.get(f"/api/location-risk?lat={lat}&lon={lon}")
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.skipif(not main.__dict__, reason="app import")
def test_kmc_point_full_coverage(client):
    b = _get(client, 22.5536, 88.3594)  # Park Street, near a pocket
    assert b["model_version"] == "real_v1"
    assert b["coverage"] == "full"
    assert b["within_kmc_data_area"] is True
    assert 0 <= b["flood_probability"] <= 1
    assert b["risk_level"] in ("LOW", "MODERATE", "ELEVATED", "HIGH")
    assert b["elevation_m"] is not None
    assert b["rainfall"]["available"] is True and b["rainfall"]["rainfall_24h_mm"] == 40.0


def test_outside_kmc_is_terrain_only(client):
    b = _get(client, 22.5800, 88.4200)  # Salt Lake, outside KMC
    assert b["coverage"] == "terrain_only"
    assert b["within_kmc_data_area"] is False
    assert b["historical_waterlogging"] is None      # no KMC records here
    assert b["elevation_m"] is not None              # DEM still covers it
    assert b["flood_probability"] is not None


def test_far_away_is_no_data(client):
    b = _get(client, 28.61, 77.20)  # Delhi
    assert b["coverage"] == "no_data"
    assert b["flood_probability"] is None
    assert b["risk_level"] is None
    assert b["elevation_m"] is None
    assert "Outside Kolkata" in b["disclaimer"] or "outside Kolkata" in b["disclaimer"]


def test_no_manual_inputs_required(client):
    # The endpoint takes ONLY lat/lon — no elevation/slope/imperviousness/etc.
    r = client.get("/api/location-risk?lat=22.55&lon=88.36")
    assert r.status_code == 200
    body = r.json()
    # response is derived, not echoing any user-supplied physical inputs
    assert "imperviousness" not in body
    assert "drain_capacity" not in body


@pytest.mark.parametrize("lat,lon", [(999, 88), (22.5, 999)])
def test_out_of_range_rejected(client, lat, lon):
    assert client.get(f"/api/location-risk?lat={lat}&lon={lon}").status_code == 422


def test_disclaimer_never_guarantees(client):
    b = _get(client, 22.5536, 88.3594)
    assert "not a guarantee" in b["disclaimer"].lower() or "not a guarantee" in b["prediction_meaning"].lower()

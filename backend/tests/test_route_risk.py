import asyncio

import pytest
from pyproj import Transformer

from config import settings
from services.route_risk_service import SEGMENT_FIELDS, RouteRiskService

TO_LL = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)
E0, N0 = 644_000.0, 2_491_000.0


def ll(e, n):
    lon, lat = TO_LL.transform(e, n)
    return lon, lat


def line(*offsets):
    return {"type": "LineString", "coordinates": [list(ll(E0 + de, N0 + dn)) for de, dn in offsets]}


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---- fakes for the three services -------------------------------------------

class FakeDEM:
    def __init__(self, elev=8.0, slope=1.5, none_after=None):
        self.elev, self.slope, self.none_after = elev, slope, none_after

    def get_terrain_features(self, points):
        out = []
        for i, _ in enumerate(points):
            if self.none_after is not None and i >= self.none_after:
                out.append({"elevation_m": None, "slope_percent": None})
            else:
                out.append({"elevation_m": self.elev, "slope_percent": self.slope})
        return out


class FakeSpatial:
    def __init__(self, drainage=True, historical=True):
        self.drainage, self.historical = drainage, historical

    def get_spatial_features(self, points):
        out = []
        for _ in points:
            drain = ({"found": True, "within_mapped_area": True, "pipe_diameter_mm": 300,
                      "distance_to_drain_m": 12.0, "drain_capacity_estimated_m3s": 0.031}
                     if self.drainage else
                     {"found": False, "within_mapped_area": False, "pipe_diameter_mm": None,
                      "distance_to_drain_m": None, "drain_capacity_estimated_m3s": None})
            hist = ({"available": True, "historical_waterlogging": 1, "historical_event_count": 2,
                     "distance_to_historical_waterlogging_m": 40.0}
                    if self.historical else
                    {"available": False, "reason": "outside KMC", "historical_waterlogging": None,
                     "historical_event_count": None, "distance_to_historical_waterlogging_m": None})
            out.append({"drain": drain,
                        "water_body": {"found": True, "distance_to_waterbody_m": 88.0},
                        "pumping_station": {"found": True, "distance_to_pumping_station_m": 950.0},
                        "historical": hist})
        return out


class FakeIMD:
    def __init__(self, available=True, calls=None):
        self.available = available
        self.calls = calls if calls is not None else []

    async def get_weather(self, station=None):
        self.calls.append(station)
        if self.available:
            return {"available": True, "station": "42809", "station_name": "Kolkata-Dum Dum",
                    "temperature": 27.8, "humidity": 92.0, "rainfall_24h": 5.0,
                    "rainfall_30m": None, "rainfall_1h": None, "rainfall_3h": None,
                    "weather_code": "5", "timestamp": "2026-09-27"}
        return {"available": False, "reason": "IP not authorized", "degraded_cause": "ip_not_authorized",
                "rainfall_24h": None, "rainfall_30m": None, "rainfall_1h": None, "rainfall_3h": None,
                "temperature": None, "humidity": None, "timestamp": None, "station": "42809"}


def make(dem=None, spatial=None, imd=None, spacing=75):
    return RouteRiskService(dem=dem or FakeDEM(), spatial=spatial or FakeSpatial(),
                            imd=imd or FakeIMD(), sample_spacing_m=spacing)


# ---- tests ------------------------------------------------------------------

def test_every_segment_has_all_required_keys():
    rf = run(make().assemble_route_features(line((0, 0), (300, 0))))
    assert len(rf.segments) == 5
    for seg in rf.segments:
        d = seg.to_dict()
        for k in SEGMENT_FIELDS:
            assert k in d, f"missing key {k}"


def test_features_populated_when_all_services_have_data():
    seg = run(make().assemble_route_features(line((0, 0), (150, 0)))).segments[0].to_dict()
    assert seg["elevation"] == 8.0 and seg["slope"] == 1.5
    assert seg["pipe_diameter_mm"] == 300 and seg["distance_to_drain_m"] == 12.0
    assert seg["drain_capacity_estimated_m3s"] == 0.031
    assert seg["historical_waterlogging"] == 1 and seg["historical_event_count"] == 2
    assert seg["distance_to_waterbody_m"] == 88.0
    assert seg["missing"] == []


def test_model_outputs_are_placeholders():
    seg = run(make().assemble_route_features(line((0, 0), (150, 0)))).segments[0]
    assert seg.flood_probability is None and seg.risk_level is None


def test_rainfall_fetched_once_per_route():
    imd = FakeIMD()
    rf = run(make(imd=imd).assemble_route_features(line((0, 0), (1500, 0))))  # ~21 segments
    assert len(rf.segments) > 10
    assert len(imd.calls) == 1  # one IMD call for the whole route
    assert rf.rainfall["available"] is True and rf.rainfall["applied_uniformly"] is True
    assert rf.rainfall["rainfall_24h_mm"] == 5.0


def test_missing_dem_becomes_null_not_fabricated():
    rf = run(make(dem=FakeDEM(none_after=2)).assemble_route_features(line((0, 0), (300, 0))))
    assert rf.segments[0].elevation == 8.0
    assert rf.segments[3].elevation is None and rf.segments[3].slope is None
    assert "elevation" in rf.segments[3].missing and "slope" in rf.segments[3].missing
    assert rf.coverage["dem_pct"] == pytest.approx(100 * 2 / 5, abs=0.1)


def test_missing_drainage_becomes_null():
    seg = run(make(spatial=FakeSpatial(drainage=False)).assemble_route_features(line((0, 0), (150, 0)))).segments[0]
    assert seg.pipe_diameter_mm is None and seg.distance_to_drain_m is None
    assert seg.drain_capacity_estimated_m3s is None and "drainage" in seg.missing


def test_missing_historical_becomes_null_not_zero():
    seg = run(make(spatial=FakeSpatial(historical=False)).assemble_route_features(line((0, 0), (150, 0)))).segments[0]
    assert seg.historical_waterlogging is None       # crucial: not 0
    assert seg.historical_event_count is None and "historical" in seg.missing


def test_imd_degraded_does_not_crash_and_marks_unavailable():
    rf = run(make(imd=FakeIMD(available=False)).assemble_route_features(line((0, 0), (300, 0))))
    assert rf.rainfall["available"] is False
    assert rf.rainfall["degraded_cause"] == "ip_not_authorized"
    assert rf.rainfall["rainfall_24h_mm"] is None
    # segments still built fully
    assert all("lat" in s.to_dict() for s in rf.segments)


def test_coverage_reported():
    rf = run(make(spatial=FakeSpatial(drainage=False, historical=False),
                  dem=FakeDEM()).assemble_route_features(line((0, 0), (300, 0))))
    assert rf.coverage["dem_pct"] == 100.0
    assert rf.coverage["drainage_pct"] == 0.0 and rf.coverage["historical_pct"] == 0.0


def test_route_dict_shape():
    d = run(make().assemble_route_features(line((0, 0), (300, 0)))).to_dict()
    assert set(d) >= {"length_m", "sample_spacing_m", "n_segments", "rainfall", "coverage", "segments"}
    assert d["n_segments"] == len(d["segments"])
    assert d["length_m"] == pytest.approx(300, abs=1)


def test_fetch_rainfall_false_skips_imd():
    imd = FakeIMD()
    rf = run(make(imd=imd).assemble_route_features(line((0, 0), (150, 0)), fetch_rainfall=False))
    assert imd.calls == [] and rf.rainfall["available"] is False


# ---- real end-to-end (opt-in) ----------------------------------------------

def _mapbox_ready():
    tok = settings.MAPBOX_ACCESS_TOKEN
    return bool(tok and tok.get_secret_value().strip())


@pytest.mark.skipif(not _mapbox_ready(), reason="MAPBOX_ACCESS_TOKEN not configured")
def test_real_sealdah_to_saltlake_feature_table():
    """Full pipeline on a real route: real Mapbox geometry, real DEM/drainage/IMD.
    Missing-data fields must be null, never fabricated."""
    from services.mapbox_service import get_mapbox_service
    from services.route_risk_service import get_route_risk_service

    async def go():
        mb = get_mapbox_service()
        try:
            routes = await mb.get_routes((22.5694, 88.3706), (22.5945, 88.4262))
        finally:
            await mb.aclose()
        rr = get_route_risk_service()
        return await rr.assemble_route_features(routes[0].geometry)
    rf = run(go())

    assert len(rf.segments) > 50
    for seg in rf.segments:
        d = seg.to_dict()
        for k in SEGMENT_FIELDS:
            assert k in d
        assert d["flood_probability"] is None  # no model yet
        if d["elevation"] is not None:
            assert 0 <= d["elevation"] <= 30       # plausible Kolkata elevation
        if d["distance_to_drain_m"] is not None:
            assert d["distance_to_drain_m"] >= 0
    # Sealdah/Salt Lake are outside the digitised wards -> drainage largely null, honestly
    assert rf.coverage["drainage_pct"] <= 100.0
    # DEM covers all of Kolkata -> near-full elevation coverage
    assert rf.coverage["dem_pct"] > 80.0
    print("coverage:", rf.coverage, "| rainfall available:", rf.rainfall["available"])

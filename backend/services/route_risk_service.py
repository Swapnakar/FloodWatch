"""
Route risk engine: assemble per-segment features for a route.

Wires together the services built in Tasks 5/9/10/12/13:
  - route_sampling: resample the route into points every ROUTE_SAMPLE_DISTANCE_M
  - dem_service:    elevation (EGM96 m) and slope (%) per point (batched)
  - spatial_service: nearest drain (+ Manning capacity proxy), pumping station,
                     water body, historical waterlogging per point (batched)
  - imd_service:    rainfall/weather, fetched ONCE per route (see below)

Rainfall is fetched once per request, not per point. A single route is a few
km long, far smaller than IMD's station spacing, so rainfall does not vary
meaningfully along it; one observation is applied to every segment and this
assumption is recorded in the response (rainfall.applied_uniformly = True).

Honesty rules (carried through from each service):
  - Every segment dict has ALL the keys in SEGMENT_FIELDS. Where a service has
    no data (outside the DEM, outside the mapped drainage, outside KMC for
    history, IMD degraded), the value is None -- never fabricated, never 0.
  - flood_probability and risk_level are None here. They are filled by the real
    model in Task 16 (wired in Task 18). This engine only gathers features.

The engine never raises for missing data; it raises only for a structurally
invalid route (caught and reported by the caller).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from services.dem_service import get_dem_service
from services.imd_service import get_imd_service
from services.route_sampling import RouteSample, route_length_m, sample_route
from services.spatial_service import get_spatial_service

logger = logging.getLogger(__name__)

# Keys guaranteed present on every segment (section 21).
SEGMENT_FIELDS = (
    "lat", "lng", "distance_from_start_m",
    "flood_probability", "risk_level",
    "elevation", "slope",
    "pipe_diameter_mm", "distance_to_drain_m", "drain_capacity_estimated_m3s",
    "historical_waterlogging",
)


@dataclass
class SegmentFeatures:
    lat: float
    lng: float
    distance_from_start_m: float
    # model outputs (placeholders until Task 16)
    flood_probability: Optional[float] = None
    risk_level: Optional[str] = None
    # terrain
    elevation: Optional[float] = None
    slope: Optional[float] = None
    # drainage
    pipe_diameter_mm: Optional[int] = None
    distance_to_drain_m: Optional[float] = None
    drain_capacity_estimated_m3s: Optional[float] = None
    within_mapped_drainage: Optional[bool] = None
    # water body / pumping station
    distance_to_waterbody_m: Optional[float] = None
    distance_to_pumping_station_m: Optional[float] = None
    # history
    historical_waterlogging: Optional[int] = None
    historical_event_count: Optional[int] = None
    distance_to_historical_waterlogging_m: Optional[float] = None
    # provenance: which inputs had no coverage at this point
    missing: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "lat": round(self.lat, 6), "lng": round(self.lng, 6),
            "distance_from_start_m": round(self.distance_from_start_m, 1),
            "flood_probability": self.flood_probability, "risk_level": self.risk_level,
            "elevation": self.elevation, "slope": self.slope,
            "pipe_diameter_mm": self.pipe_diameter_mm,
            "distance_to_drain_m": self.distance_to_drain_m,
            "drain_capacity_estimated_m3s": self.drain_capacity_estimated_m3s,
            "within_mapped_drainage": self.within_mapped_drainage,
            "distance_to_waterbody_m": self.distance_to_waterbody_m,
            "distance_to_pumping_station_m": self.distance_to_pumping_station_m,
            "historical_waterlogging": self.historical_waterlogging,
            "historical_event_count": self.historical_event_count,
            "distance_to_historical_waterlogging_m": self.distance_to_historical_waterlogging_m,
            "missing": self.missing,
        }
        return d


@dataclass
class RouteFeatures:
    segments: List[SegmentFeatures]
    length_m: float
    sample_spacing_m: float
    rainfall: Dict[str, Any]
    coverage: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length_m": round(self.length_m, 1),
            "sample_spacing_m": self.sample_spacing_m,
            "n_segments": len(self.segments),
            "rainfall": self.rainfall,
            "coverage": self.coverage,
            "segments": [s.to_dict() for s in self.segments],
        }


class RouteRiskService:
    def __init__(self, dem=None, spatial=None, imd=None,
                 sample_spacing_m: Optional[float] = None,
                 rainfall_station_id: Optional[str] = None):
        self._dem = dem
        self._spatial = spatial
        self._imd = imd
        self.sample_spacing_m = float(sample_spacing_m if sample_spacing_m is not None
                                      else settings.ROUTE_SAMPLE_DISTANCE_M)
        self._rainfall_station_id = rainfall_station_id

    # lazy singletons so tests can inject fakes
    @property
    def dem(self):
        return self._dem if self._dem is not None else get_dem_service()

    @property
    def spatial(self):
        return self._spatial if self._spatial is not None else get_spatial_service()

    @property
    def imd(self):
        return self._imd if self._imd is not None else get_imd_service()

    async def assemble_route_features(self, geometry, *, fetch_rainfall: bool = True) -> RouteFeatures:
        """Build the per-segment feature table for one route geometry (GeoJSON LineString)."""
        samples: List[RouteSample] = sample_route(geometry, spacing_m=self.sample_spacing_m)
        length_m = route_length_m(geometry)
        points: List[Tuple[float, float]] = [(s.lat, s.lng) for s in samples]

        # Rainfall once per route (I/O bound) concurrently with the CPU-bound
        # terrain+spatial batch (run in a thread so the event loop isn't blocked).
        rainfall_task = asyncio.create_task(self._get_rainfall()) if fetch_rainfall else None
        terrain, spatial = await asyncio.to_thread(self._gather_local_features, points)
        rainfall = await rainfall_task if rainfall_task is not None else self._rainfall_unfetched()

        segments: List[SegmentFeatures] = []
        cov = {"dem": 0, "drainage": 0, "historical": 0}
        for smp, t, sp in zip(samples, terrain, spatial):
            seg = self._build_segment(smp, t, sp)
            segments.append(seg)
            cov["dem"] += "elevation" not in seg.missing
            cov["drainage"] += "drainage" not in seg.missing
            cov["historical"] += "historical" not in seg.missing

        n = len(segments) or 1
        coverage = {
            "dem_pct": round(100 * cov["dem"] / n, 1),
            "drainage_pct": round(100 * cov["drainage"] / n, 1),
            "historical_pct": round(100 * cov["historical"] / n, 1),
            "note": "Percent of segments with real data for each source; the rest are null, not zero.",
        }
        return RouteFeatures(segments=segments, length_m=length_m,
                             sample_spacing_m=self.sample_spacing_m, rainfall=rainfall, coverage=coverage)

    # ------------------------------------------------------- gathering

    def _gather_local_features(self, points):
        """DEM + spatial batches (both CPU/IO-light, no event loop needed)."""
        terrain = self.dem.get_terrain_features(points)
        spatial = self.spatial.get_spatial_features(points)
        return terrain, spatial

    def _build_segment(self, smp: RouteSample, terrain: Dict, spatial: Dict) -> SegmentFeatures:
        seg = SegmentFeatures(lat=smp.lat, lng=smp.lng,
                              distance_from_start_m=smp.distance_from_start_m)

        seg.elevation = terrain.get("elevation_m")
        seg.slope = terrain.get("slope_percent")
        if seg.elevation is None:
            seg.missing.append("elevation")
        if seg.slope is None:
            seg.missing.append("slope")

        drain = spatial.get("drain") or {}
        seg.within_mapped_drainage = drain.get("within_mapped_area")
        if drain.get("found"):
            seg.pipe_diameter_mm = drain.get("pipe_diameter_mm")
            seg.distance_to_drain_m = drain.get("distance_to_drain_m")
            seg.drain_capacity_estimated_m3s = drain.get("drain_capacity_estimated_m3s")
        else:
            seg.missing.append("drainage")

        water = spatial.get("water_body") or {}
        if water.get("found"):
            seg.distance_to_waterbody_m = water.get("distance_to_waterbody_m")
        pump = spatial.get("pumping_station") or {}
        if pump.get("found"):
            seg.distance_to_pumping_station_m = pump.get("distance_to_pumping_station_m")

        hist = spatial.get("historical") or {}
        if hist.get("available"):
            seg.historical_waterlogging = hist.get("historical_waterlogging")
            seg.historical_event_count = hist.get("historical_event_count")
            seg.distance_to_historical_waterlogging_m = hist.get("distance_to_historical_waterlogging_m")
        else:
            seg.missing.append("historical")
        return seg

    # -------------------------------------------------------- rainfall

    async def _get_rainfall(self) -> Dict[str, Any]:
        station = self._rainfall_station_id
        wx = await self.imd.get_weather(station) if station else await self.imd.get_weather()
        return {
            "available": bool(wx.get("available")),
            "source": "IMD",
            "station": wx.get("station"),
            "station_name": wx.get("station_name"),
            "timestamp": wx.get("timestamp"),
            "rainfall_24h_mm": wx.get("rainfall_24h"),
            "rainfall_30m_mm": wx.get("rainfall_30m"),
            "rainfall_1h_mm": wx.get("rainfall_1h"),
            "rainfall_3h_mm": wx.get("rainfall_3h"),
            "temperature_c": wx.get("temperature"),
            "humidity_pct": wx.get("humidity"),
            "weather_code": wx.get("weather_code"),
            "applied_uniformly": True,
            "assumption": "One IMD observation applied to all segments; rainfall does not vary "
                          "meaningfully over a single short route.",
            "reason": wx.get("reason"),
            "degraded_cause": wx.get("degraded_cause"),
        }

    @staticmethod
    def _rainfall_unfetched() -> Dict[str, Any]:
        return {"available": False, "reason": "rainfall fetch skipped", "applied_uniformly": True}


_service: Optional[RouteRiskService] = None


def get_route_risk_service() -> RouteRiskService:
    global _service
    if _service is None:
        _service = RouteRiskService()
    return _service

"""
Spatial service: nearest drain, pumping station, water body and historical
waterlogging for a (lat, lon) point, using the KMC-derived GIS layers.

All distances are metres in UTM 45N (EPSG:32645). Every layer is reprojected
once at load, and nearest-neighbour queries use each layer's STRtree spatial
index.

Honesty rules
-------------
- Beyond the per-layer max search radius, the result is an explicit
  "no coverage" (found=False, values None). It is never a far-away "nearest".
- Drainage has only been digitised for a few KMC wards. `within_mapped_area`
  says whether the point lies inside a digitised sheet's footprint (concave
  hull of its pipes + 100 m). Outside it,
  a found distance is only the distance to the nearest *mapped* drain, which
  is an upper bound: unmapped drains may be closer.
- Missing layers never crash: queries return found=False with a reason, and
  status_report() shows MISSING or ERROR.
- Historical waterlogging comes from services/historical_waterlogging.py. It
  returns None (not 0) when the CSV is missing or the point is outside KMC,
  because 0 would assert "no history" where there is simply no data.
"""

import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import unary_union

from config import settings
from services.capacity_proxy import capacity_fields
from services.historical_waterlogging import HistoricalWaterlogging

logger = logging.getLogger(__name__)

METRIC_CRS = "EPSG:32645"
_TO_UTM = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
_TO_WGS84 = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)

# Mapped-area footprint = per-sheet CONCAVE hull of the pipes, grown by
# COVERAGE_BUFFER_M. A convex hull was tried and rejected: Ward 108 spans two
# sheets, and its convex hull (7.1 km2) bridged an unmapped gap. That flagged a
# point 285 m from any pipe as "mapped"; the concave hull covers ~3 km2.
COVERAGE_CONCAVE_RATIO = 0.2
COVERAGE_BUFFER_M = 100.0

LatLon = Tuple[float, float]


def _valid(lat, lon) -> bool:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and abs(lat) <= 90 and abs(lon) <= 180


def _opt(v):
    """numpy/pandas scalar -> plain Python; NaN/NA -> None."""
    if v is None:
        return None
    try:
        if v != v:  # NaN
            return None
    except (TypeError, ValueError):
        pass
    try:
        import pandas as pd
        if v is pd.NA:
            return None
    except ImportError:
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


class _Layer:
    def __init__(self, name: str, path: Path, geom_types: Sequence[str]):
        self.name = name
        self.path = Path(path)
        self.status = "MISSING"
        self.error: Optional[str] = None
        self.gdf: Optional[gpd.GeoDataFrame] = None
        if not self.path.exists():
            self.error = f"file not found: {self.path.name}"
            logger.warning("%s layer %s", name, self.error)
            return
        try:
            g = gpd.read_file(self.path)
            if g.crs is None or g.crs.to_epsg() != 4326:
                raise ValueError(f"expected EPSG:4326, got {g.crs}")
            g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
            bad = ~g.geom_type.isin(geom_types)
            if bad.any():
                raise ValueError(f"unexpected geometry types {sorted(set(g.geom_type[bad]))}")
            g = g.to_crs(METRIC_CRS).reset_index(drop=True)
            _ = g.sindex  # build the STRtree now, not on the first request
            self.gdf = g
            self.status = "LOADED"
        except Exception as exc:
            self.status = "ERROR"
            self.error = f"{type(exc).__name__}: {exc}"
            logger.error("%s layer failed to load: %s", name, self.error)

    @property
    def loaded(self) -> bool:
        return self.gdf is not None and len(self.gdf) > 0

    def report(self) -> Dict:
        r = {"status": self.status, "file": self.path.name,
             "features": int(len(self.gdf)) if self.gdf is not None else 0}
        if self.error:
            r["error"] = self.error
        return r

    def nearest(self, xs: np.ndarray, ys: np.ndarray, max_distance: float):
        """Nearest feature within max_distance per point: (tree_index or -1, distance or nan)."""
        idx = np.full(len(xs), -1, dtype=np.int64)
        dist = np.full(len(xs), np.nan)
        if not self.loaded or len(xs) == 0:
            return idx, dist
        pts = gpd.GeoSeries.from_xy(xs, ys, crs=METRIC_CRS)
        (inp, tree), d = self.gdf.sindex.nearest(
            pts, return_all=False, max_distance=max_distance, return_distance=True)
        idx[inp] = tree
        dist[inp] = d
        return idx, dist


class SpatialService:
    def __init__(
        self,
        drainage_path: Path,
        pumping_path: Path,
        water_path: Path,
        historical_path: Optional[Path] = None,
        drain_max_radius_m: float = 500.0,
        pump_max_radius_m: float = 3000.0,
        water_max_radius_m: float = 1000.0,
        historical_radius_m: float = 250.0,
        kmc_boundary_path: Optional[Path] = None,
    ):
        self.drain_max_radius_m = float(drain_max_radius_m)
        self.pump_max_radius_m = float(pump_max_radius_m)
        self.water_max_radius_m = float(water_max_radius_m)
        self.historical_radius_m = float(historical_radius_m)
        self.historical_path = Path(historical_path) if historical_path else None

        self.historical = HistoricalWaterlogging(self.historical_path, kmc_boundary_path, self.historical_radius_m)
        self.drains = _Layer("drainage_network", drainage_path, ("LineString", "MultiLineString"))
        self.pumps = _Layer("pumping_stations", pumping_path, ("Point",))
        self.water = _Layer("water_bodies", water_path, ("Polygon", "MultiPolygon"))

        self._coverage = None
        self.coverage_wards: List[int] = []
        if self.drains.loaded:
            g = self.drains.gdf
            key = "source_sheet" if "source_sheet" in g else ("ward" if "ward" in g else None)
            groups = g.groupby(key) if key else [(None, g)]
            hulls = []
            for _, grp in groups:
                lines = unary_union(list(grp.geometry))
                hulls.append(shapely.concave_hull(lines, ratio=COVERAGE_CONCAVE_RATIO).buffer(COVERAGE_BUFFER_M))
            self._coverage = unary_union(hulls)
            if "ward" in g:
                self.coverage_wards = sorted(int(w) for w in g["ward"].dropna().unique())

    # ---------------------------------------------------------------- status

    def status_report(self) -> Dict:
        return {
            "drainage_network": dict(self.drains.report(), mapped_wards=self.coverage_wards),
            "pumping_stations": self.pumps.report(),
            "water_bodies": self.water.report(),
            "historical_waterlogging": self.historical.status_report(),
            "search_radius_m": {"drain": self.drain_max_radius_m, "pumping_station": self.pump_max_radius_m,
                                "water_body": self.water_max_radius_m,
                                "historical_waterlogging": self.historical_radius_m},
        }

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _to_utm(points: Sequence[LatLon]):
        lats = np.array([float(p[0]) if _valid(*p) else np.nan for p in points])
        lons = np.array([float(p[1]) if _valid(*p) else np.nan for p in points])
        ok = np.isfinite(lats) & np.isfinite(lons)
        xs = np.full(len(points), np.nan)
        ys = np.full(len(points), np.nan)
        if ok.any():
            xs[ok], ys[ok] = _TO_UTM.transform(lons[ok], lats[ok])
        return xs, ys, ok

    def _in_mapped_area(self, x: float, y: float) -> bool:
        return bool(self._coverage is not None and self._coverage.covers(Point(x, y)))

    @staticmethod
    def _nearest_latlon(geom, x, y) -> Dict:
        from shapely.ops import nearest_points
        p = nearest_points(geom, Point(x, y))[0]
        lon, lat = _TO_WGS84.transform(p.x, p.y)
        return {"lat": round(lat, 6), "lon": round(lon, 6)}

    # --------------------------------------------------------------- drains

    def find_nearest_drain_batch(self, points: Sequence[LatLon]) -> List[Dict]:
        points = list(points)
        xs, ys, ok = self._to_utm(points)
        out: List[Dict] = []
        if not self.drains.loaded:
            reason = f"drainage layer {self.drains.status.lower()}"
            return [self._drain_none(reason if o else "invalid coordinates", None) for o in ok]
        idx, dist = self.drains.nearest(xs[ok], ys[ok], self.drain_max_radius_m)
        full_idx = np.full(len(points), -1, dtype=np.int64)
        full_dist = np.full(len(points), np.nan)
        full_idx[ok], full_dist[ok] = idx, dist
        g = self.drains.gdf
        for i in range(len(points)):
            if not ok[i]:
                out.append(self._drain_none("invalid coordinates", None))
                continue
            mapped = self._in_mapped_area(xs[i], ys[i])
            if full_idx[i] < 0:
                reason = (f"no mapped drain within {self.drain_max_radius_m:.0f} m"
                          + ("" if mapped else " (point is outside the digitised drainage footprint)"))
                out.append(self._drain_none(reason, mapped))
                continue
            row = g.iloc[int(full_idx[i])]
            dia = _opt(row.get("pipe_diameter_mm"))
            ctype = _opt(row.get("conduit_type")) or "pipe"
            rec = {
                "found": True,
                "distance_to_drain_m": round(float(full_dist[i]), 1),
                "within_mapped_area": mapped,
                "segment_id": _opt(row.get("segment_id")),
                "ward": _opt(row.get("ward")),
                "conduit_type": ctype,
                "pipe_diameter_mm": int(dia) if dia is not None else None,
                "diameter_source": _opt(row.get("diameter_source")),
                "source": _opt(row.get("source")),
                "nearest_point": self._nearest_latlon(row.geometry, xs[i], ys[i]),
            }
            rec.update(capacity_fields(rec["pipe_diameter_mm"], ctype))
            if not mapped:
                rec["note"] = ("Point is outside the digitised drainage footprint; unmapped drains may be closer. "
                               "Distance is an upper bound.")
            out.append(rec)
        return out

    def find_nearest_drain(self, lat: float, lon: float) -> Dict:
        return self.find_nearest_drain_batch([(lat, lon)])[0]

    @staticmethod
    def _drain_none(reason: str, mapped: Optional[bool]) -> Dict:
        rec = {"found": False, "reason": reason, "distance_to_drain_m": None,
               "within_mapped_area": mapped, "segment_id": None, "ward": None,
               "conduit_type": None, "pipe_diameter_mm": None, "diameter_source": None,
               "source": None, "nearest_point": None}
        rec.update(capacity_fields(None, None))
        return rec

    # ------------------------------------------------------ pumping stations

    def find_nearest_pumping_station_batch(self, points: Sequence[LatLon]) -> List[Dict]:
        points = list(points)
        xs, ys, ok = self._to_utm(points)
        empty = {"found": False, "distance_to_pumping_station_m": None, "name": None, "ward": None}
        if not self.pumps.loaded:
            return [dict(empty, reason=f"pumping station layer {self.pumps.status.lower()}" if o
                         else "invalid coordinates") for o in ok]
        idx, dist = self.pumps.nearest(xs[ok], ys[ok], self.pump_max_radius_m)
        out, j = [], 0
        for i in range(len(points)):
            if not ok[i]:
                out.append(dict(empty, reason="invalid coordinates"))
                continue
            k, d = idx[j], dist[j]
            j += 1
            if k < 0:
                out.append(dict(empty, reason=f"no mapped pumping station within {self.pump_max_radius_m:.0f} m"))
                continue
            row = self.pumps.gdf.iloc[int(k)]
            lon, lat = _TO_WGS84.transform(row.geometry.x, row.geometry.y)
            out.append({"found": True, "distance_to_pumping_station_m": round(float(d), 1),
                        "name": _opt(row.get("name")), "ward": _opt(row.get("ward")),
                        "location": {"lat": round(lat, 6), "lon": round(lon, 6)},
                        "position_note": _opt(row.get("position_note"))})
        return out

    def find_nearest_pumping_station(self, lat: float, lon: float) -> Dict:
        return self.find_nearest_pumping_station_batch([(lat, lon)])[0]

    # ----------------------------------------------------------- water bodies

    def find_nearest_waterbody_batch(self, points: Sequence[LatLon]) -> List[Dict]:
        points = list(points)
        xs, ys, ok = self._to_utm(points)
        empty = {"found": False, "distance_to_waterbody_m": None, "inside_waterbody": None}
        if not self.water.loaded:
            return [dict(empty, reason=f"water body layer {self.water.status.lower()}" if o
                         else "invalid coordinates") for o in ok]
        idx, dist = self.water.nearest(xs[ok], ys[ok], self.water_max_radius_m)
        out, j = [], 0
        for i in range(len(points)):
            if not ok[i]:
                out.append(dict(empty, reason="invalid coordinates"))
                continue
            k, d = idx[j], dist[j]
            j += 1
            if k < 0:
                out.append(dict(empty, reason=f"no mapped water body within {self.water_max_radius_m:.0f} m"))
                continue
            out.append({"found": True, "distance_to_waterbody_m": round(float(d), 1),
                        "inside_waterbody": bool(d == 0.0)})
        return out

    def find_nearest_waterbody(self, lat: float, lon: float) -> Dict:
        return self.find_nearest_waterbody_batch([(lat, lon)])[0]

    # ------------------------------------------------- historical waterlogging

    def find_historical_waterlogging_batch(self, points: Sequence[LatLon]) -> List[Dict]:
        return self.historical.lookup_batch(points)

    def find_historical_waterlogging(self, lat: float, lon: float) -> Dict:
        """historical_waterlogging (0/1 within radius), count, distance; None where no data."""
        return self.historical.lookup(lat, lon)

    # ------------------------------------------------------------- combined

    def get_spatial_features(self, points: Iterable[LatLon]) -> List[Dict]:
        """All spatial features per point (input order), for the route risk engine."""
        points = list(points)
        drains = self.find_nearest_drain_batch(points)
        pumps = self.find_nearest_pumping_station_batch(points)
        water = self.find_nearest_waterbody_batch(points)
        hist = self.find_historical_waterlogging_batch(points)
        return [{"drain": d, "pumping_station": p, "water_body": w, "historical": h}
                for d, p, w, h in zip(drains, pumps, water, hist)]


@lru_cache(maxsize=1)
def get_spatial_service() -> SpatialService:
    """Process-wide service, loaded once on first use."""
    return SpatialService(
        drainage_path=settings.DRAINAGE_GEOJSON_PATH,
        pumping_path=settings.PUMPING_STATIONS_GEOJSON_PATH,
        water_path=settings.WATER_BODIES_GEOJSON_PATH,
        historical_path=settings.HISTORICAL_WATERLOGGING_PATH,
        drain_max_radius_m=settings.DRAIN_MAX_SEARCH_RADIUS_M,
        pump_max_radius_m=settings.PUMP_MAX_SEARCH_RADIUS_M,
        water_max_radius_m=settings.WATERBODY_MAX_SEARCH_RADIUS_M,
        historical_radius_m=settings.HISTORICAL_WATERLOGGING_RADIUS_M,
        kmc_boundary_path=settings.KMC_BOUNDARY_GEOJSON_PATH,
    )


# Module-level wrappers (signatures per plan, lat/lon order).
def find_nearest_drain(lat: float, lon: float) -> Dict:
    return get_spatial_service().find_nearest_drain(lat, lon)


def find_nearest_pumping_station(lat: float, lon: float) -> Dict:
    return get_spatial_service().find_nearest_pumping_station(lat, lon)


def find_nearest_waterbody(lat: float, lon: float) -> Dict:
    return get_spatial_service().find_nearest_waterbody(lat, lon)


def find_historical_waterlogging(lat: float, lon: float) -> Dict:
    return get_spatial_service().find_historical_waterlogging(lat, lon)

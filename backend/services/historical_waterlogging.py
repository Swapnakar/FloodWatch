"""
Historical waterlogging records: loader and radius lookup.

CSV schema (data/historical/waterlogging_points.csv):
    required: location_name, ward, latitude, longitude, source, year, severity, notes
    optional: geocode_method, location_uncertainty_m, source_page, borough, source_sn

What a record means: one listing of a waterlogging location in a KMC source,
such as the 2017 action plan's "Major Water Logging Pockets". It is NOT one
flood event. So `historical_event_count` counts the source records within the
radius, which is the number of times a location near here was listed, not the
number of floods.

Honesty rules
-------------
- Rows without usable coordinates are kept for reporting but never used in
  lookups. They are counted in status_report() as unlocated.
- Records only cover the KMC area. Outside the KMC boundary the answer is
  "no data" (None), not 0. Salt Lake, Howrah and New Town have no KMC records.
- Inside KMC, 0 means "no listed record within the radius". That isn't proof a
  spot never floods, since the 2017 list only covers *major* pockets.
- If the file or boundary is missing, values are None with a reason.
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

logger = logging.getLogger(__name__)

METRIC_CRS = "EPSG:32645"
_TO_UTM = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
REQUIRED_COLUMNS = ("location_name", "ward", "latitude", "longitude", "source", "year", "severity", "notes")
KOLKATA_BBOX = (88.20, 22.40, 88.55, 22.75)  # lon_min, lat_min, lon_max, lat_max

LatLon = Tuple[float, float]


def _valid(lat, lon) -> bool:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and abs(lat) <= 90 and abs(lon) <= 180


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


class HistoricalWaterlogging:
    def __init__(self, csv_path: Optional[Path], boundary_path: Optional[Path], radius_m: float = 250.0):
        self.csv_path = Path(csv_path) if csv_path else None
        self.boundary_path = Path(boundary_path) if boundary_path else None
        self.radius_m = float(radius_m)
        self.status = "MISSING"
        self.error: Optional[str] = None
        self.rows_total = 0
        self.rows_unlocated = 0
        self.rows_rejected: List[str] = []
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.boundary = None
        self.boundary_status = "MISSING"
        self._load_boundary()
        self._load_csv()

    # ------------------------------------------------------------- loading

    def _load_boundary(self):
        if self.boundary_path is None or not self.boundary_path.exists():
            return
        try:
            b = gpd.read_file(self.boundary_path)
            if b.crs is None or b.crs.to_epsg() != 4326:
                raise ValueError(f"expected EPSG:4326, got {b.crs}")
            self.boundary = b.to_crs(METRIC_CRS).union_all()
            self.boundary_status = "LOADED"
        except Exception as exc:
            self.boundary_status = "ERROR"
            logger.error("KMC boundary failed to load: %s", exc)

    def _load_csv(self):
        if self.csv_path is None or not self.csv_path.exists():
            self.error = "historical waterlogging CSV not found"
            return
        try:
            df = pd.read_csv(self.csv_path, dtype={"ward": "string", "severity": "string", "notes": "string"})
        except Exception as exc:
            self.status, self.error = "ERROR", f"unreadable CSV: {type(exc).__name__}: {exc}"
            return
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            self.status, self.error = "ERROR", f"missing required columns: {missing}"
            return
        self.rows_total = len(df)
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        lon = pd.to_numeric(df["longitude"], errors="coerce")
        has = lat.notna() & lon.notna()
        self.rows_unlocated = int((~has).sum())
        w, s, e, n = KOLKATA_BBOX
        inside = has & lon.between(w, e) & lat.between(s, n)
        for i in df.index[has & ~inside]:
            self.rows_rejected.append(f"row {i + 2}: ({lat[i]}, {lon[i]}) outside Kolkata bbox")
        nosrc = df["source"].isna() | (df["source"].astype(str).str.strip() == "")
        for i in df.index[inside & nosrc]:
            self.rows_rejected.append(f"row {i + 2}: missing source")
        use = inside & ~nosrc
        if self.rows_rejected:
            logger.warning("historical CSV: %d rows rejected", len(self.rows_rejected))
        g = gpd.GeoDataFrame(df[use].copy(), geometry=gpd.points_from_xy(lon[use], lat[use]), crs="EPSG:4326")
        g = g.to_crs(METRIC_CRS).reset_index(drop=True)
        if len(g):
            _ = g.sindex
        self.gdf = g
        self.status = "LOADED"

    @property
    def usable(self) -> bool:
        return self.gdf is not None and len(self.gdf) > 0

    def status_report(self) -> Dict:
        r = {"status": self.status, "file": self.csv_path.name if self.csv_path else None,
             "rows_total": self.rows_total, "rows_located": int(len(self.gdf)) if self.gdf is not None else 0,
             "rows_unlocated": self.rows_unlocated, "rows_rejected": len(self.rows_rejected),
             "coverage_boundary": self.boundary_status, "radius_m": self.radius_m}
        if self.error:
            r["error"] = self.error
        if self.rows_rejected:
            r["rejected_examples"] = self.rows_rejected[:10]
        return r

    # -------------------------------------------------------------- lookup

    def _none(self, reason: str, covered: Optional[bool]) -> Dict:
        return {"available": False, "reason": reason, "historical_waterlogging": None,
                "historical_event_count": None, "distance_to_historical_waterlogging_m": None,
                "within_records_coverage": covered, "radius_m": self.radius_m, "nearest_record": None}

    def lookup_batch(self, points: Sequence[LatLon]) -> List[Dict]:
        points = list(points)
        out: List[Dict] = []
        if self.status != "LOADED":
            reason = self.error or f"historical data {self.status.lower()}"
            return [self._none(reason if _valid(*p) else "invalid coordinates", None) for p in points]
        for lat, lon in points:
            if not _valid(lat, lon):
                out.append(self._none("invalid coordinates", None))
                continue
            x, y = _TO_UTM.transform(float(lon), float(lat))
            pt = Point(x, y)
            covered = bool(self.boundary.covers(pt)) if self.boundary is not None else None
            if covered is False:
                out.append(self._none("outside KMC area: no KMC waterlogging records exist here", False))
                continue
            if not self.usable:
                out.append(self._none("no located historical records", covered))
                continue
            within = self.gdf.sindex.query(pt.buffer(self.radius_m), predicate="intersects")
            within = [int(i) for i in within if self.gdf.geometry.iloc[int(i)].distance(pt) <= self.radius_m]
            (_, tree), dist = self.gdf.sindex.nearest(gpd.GeoSeries([pt], crs=METRIC_CRS),
                                                      return_all=False, return_distance=True)
            k, d = int(tree[0]), float(dist[0])
            row = self.gdf.iloc[k]
            rec = {
                "available": True,
                "historical_waterlogging": 1 if within else 0,
                "historical_event_count": len(within),
                "distance_to_historical_waterlogging_m": round(d, 1),
                "within_records_coverage": covered,
                "radius_m": self.radius_m,
                "nearest_record": {
                    "location_name": _clean(row.get("location_name")), "ward": _clean(row.get("ward")),
                    "source": _clean(row.get("source")), "year": _clean(row.get("year")),
                    "severity": _clean(row.get("severity")),
                    "geocode_method": _clean(row.get("geocode_method")),
                    "location_uncertainty_m": _clean(row.get("location_uncertainty_m")),
                },
                "note": ("historical_event_count = number of source listings within the radius, "
                         "not number of flood events"),
            }
            if covered is None:
                rec["coverage_warning"] = "KMC boundary not loaded; 0 may mean 'no data' outside KMC"
            out.append(rec)
        return out

    def lookup(self, lat: float, lon: float) -> Dict:
        return self.lookup_batch([(lat, lon)])[0]

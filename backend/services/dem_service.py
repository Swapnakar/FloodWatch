"""
DEM service: terrain elevation and slope from the CartoDEM GeoTIFF.

Vertical datum
--------------
The CartoDEM tile stores heights above the WGS84 *ellipsoid* (inferred: raw
values in Kolkata are about -47 m, while the EGM96 geoid sits about -56.9 m
below the ellipsoid there). Every elevation this service returns is converted
per point to EGM96 orthometric height (roughly metres above mean sea level)
using PROJ and the bundled NGA EGM96 grid (data/geoid/us_nga_egm96_15.tif).
If that conversion is unavailable, elevation is returned as None. Raw
ellipsoidal heights are never passed off as above-sea-level values.

Sampling
--------
v1 uses nearest-pixel lookup (the pixel containing the point), not bilinear
interpolation. Bilinear sampling is a possible future improvement.

Slope
-----
Horn's 3x3 finite-difference method on the raw DEM surface, in percent
(100 * rise/run). The pixel size in metres is computed geodesically at each
point's latitude. Slope needs all 9 pixels of the 3x3 block to be valid. If any is
NoData, or the point is on the raster edge, slope is None. The geoid shift is
effectively constant over 90 m, so slope is unaffected by the datum.

CartoDEM is a surface model. In built-up areas, heights and slopes may include
buildings and trees, not just bare ground.

Missing data
------------
Out-of-bounds coordinates, invalid coordinates, and NoData pixels return None.
Values are never fabricated or filled in from neighbours.
"""

import logging
import math
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyproj
import rasterio
from pyproj import Geod, Transformer
from rasterio.windows import Window

from config import settings

logger = logging.getLogger(__name__)

VERTICAL_DATUM = "EGM96 orthometric height (m), converted per point from WGS84 ellipsoidal"
ASSUMED_SOURCE_DATUM = "WGS84 ellipsoidal height (inferred from geoid comparison)"
SAMPLING_METHOD = "nearest"
SLOPE_METHOD = "Horn 3x3 finite difference, percent"

LatLon = Tuple[float, float]

_GEOD = Geod(ellps="WGS84")


class DEMService:
    """Thread-safe reader over a single-band DEM GeoTIFF (EPSG:4326)."""

    def __init__(self, dem_path: Path, geoid_grid_path: Path):
        self.dem_path = Path(dem_path)
        self.geoid_grid_path = Path(geoid_grid_path)
        self._lock = threading.Lock()
        self._ds = None
        self._transformer: Optional[Transformer] = None

        self.dem_status = "MISSING"      # MISSING | LOADED | ERROR
        self.dem_error: Optional[str] = None
        self.geoid_status = "MISSING"    # MISSING | LOADED | ERROR
        self.geoid_error: Optional[str] = None

        self._open_dem()
        if self._ds is not None:
            self._init_geoid()

    # ------------------------------------------------------------------ setup

    def _open_dem(self) -> None:
        if not self.dem_path.exists():
            self.dem_error = f"DEM file not found: {self.dem_path.name}"
            logger.warning(self.dem_error)
            return
        try:
            ds = rasterio.open(self.dem_path)
            if ds.crs is None or ds.crs.to_epsg() != 4326:
                raise ValueError(f"expected EPSG:4326 DEM, got {ds.crs}")
            if ds.count < 1:
                raise ValueError("DEM has no bands")
            self._ds = ds
            self.dem_status = "LOADED"
        except Exception as exc:  # corrupt file, wrong CRS, etc.
            self.dem_status = "ERROR"
            self.dem_error = f"{type(exc).__name__}: {exc}"
            logger.error("DEM load failed: %s", self.dem_error)

    def _init_geoid(self) -> None:
        if not self.geoid_grid_path.exists():
            self.geoid_error = f"Geoid grid not found: {self.geoid_grid_path.name}"
            logger.warning(self.geoid_error)
            return
        try:
            pyproj.datadir.append_data_dir(str(self.geoid_grid_path.parent))
            # only_best=True: raise rather than silently fall back to a
            # "ballpark" operation that applies no vertical shift at all.
            t = Transformer.from_crs(
                "EPSG:4979", "EPSG:4326+5773", always_xy=True, only_best=True
            )
            # Sanity check at the DEM centre: for Kolkata the geoid shift is about
            # +57 m. A zero or non-finite shift means the grid isn't being applied.
            b = self._ds.bounds
            cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
            _, _, h = t.transform(cx, cy, 0.0)
            if not math.isfinite(h) or abs(h) < 0.01:
                raise RuntimeError(f"geoid shift check failed (got {h!r})")
            self._transformer = t
            self.geoid_status = "LOADED"
        except Exception as exc:
            self.geoid_status = "ERROR"
            self.geoid_error = f"{type(exc).__name__}: {exc}"
            logger.error("Geoid transformer init failed: %s", self.geoid_error)

    # ----------------------------------------------------------------- status

    @property
    def is_loaded(self) -> bool:
        return self.dem_status == "LOADED"

    def status_report(self) -> Dict:
        report = {
            "dem_status": self.dem_status,
            "dem_file": self.dem_path.name,
            "geoid_status": self.geoid_status,
            "elevation_available": self.is_loaded and self.geoid_status == "LOADED",
            "slope_available": self.is_loaded,
            "vertical_datum": VERTICAL_DATUM,
            "source_vertical_datum": ASSUMED_SOURCE_DATUM,
            "sampling_method": SAMPLING_METHOD,
            "slope_method": SLOPE_METHOD,
        }
        if self.dem_error:
            report["dem_error"] = self.dem_error
        if self.geoid_error:
            report["geoid_error"] = self.geoid_error
        if self._ds is not None:
            report["bounds_wgs84"] = list(self._ds.bounds)
            report["resolution_deg"] = list(self._ds.res)
        return report

    def close(self) -> None:
        with self._lock:
            if self._ds is not None:
                self._ds.close()
                self._ds = None
                self.dem_status = "MISSING"

    # ------------------------------------------------------------ core reads

    def _pixel_indices(self, points: Sequence[LatLon]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (rows, cols, inside_mask) for (lat, lon) points (nearest = containing pixel)."""
        lats = np.array([p[0] for p in points], dtype="float64")
        lons = np.array([p[1] for p in points], dtype="float64")
        valid = (
            np.isfinite(lats) & np.isfinite(lons)
            & (np.abs(lats) <= 90) & (np.abs(lons) <= 180)
        )
        inv = ~self._ds.transform
        with np.errstate(invalid="ignore"):
            cols = np.floor(inv.a * lons + inv.b * lats + inv.c)
            rows = np.floor(inv.d * lons + inv.e * lats + inv.f)
        inside = (
            valid
            & (rows >= 0) & (rows < self._ds.height)
            & (cols >= 0) & (cols < self._ds.width)
        )
        rows = np.where(inside, rows, 0).astype("int64")
        cols = np.where(inside, cols, 0).astype("int64")
        return rows, cols, inside

    def _read_window(self, rows: np.ndarray, cols: np.ndarray, pad: int):
        """Read one window covering all given pixels (+pad), clipped to the raster.

        Returns (data, row_off, col_off). `data` is a float64 array with NaN
        wherever the source is NoData or non-finite.
        """
        r0 = max(int(rows.min()) - pad, 0)
        c0 = max(int(cols.min()) - pad, 0)
        r1 = min(int(rows.max()) + pad + 1, self._ds.height)
        c1 = min(int(cols.max()) + pad + 1, self._ds.width)
        window = Window(c0, r0, c1 - c0, r1 - r0)
        masked = self._ds.read(1, window=window, masked=True)
        data = masked.astype("float64").filled(np.nan)
        data[~np.isfinite(data)] = np.nan
        return data, r0, c0

    def _to_orthometric(self, lats: np.ndarray, lons: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Ellipsoidal -> EGM96 orthometric. Returns NaN where conversion is impossible."""
        out = np.full(h.shape, np.nan)
        ok = np.isfinite(h)
        if self._transformer is None or not ok.any():
            return out
        _, _, H = self._transformer.transform(lons[ok], lats[ok], h[ok])
        H = np.asarray(H, dtype="float64")
        H[~np.isfinite(H)] = np.nan
        out[ok] = H
        return out

    # ------------------------------------------------------------- public API

    def get_terrain_features(self, points: Iterable[LatLon]) -> List[Dict]:
        """Elevation and slope for each (lat, lon). Missing data is None, never estimated.

        Returns one dict per input point, in input order:
            {"lat", "lon", "elevation_m", "slope_percent"}
        """
        pts = [(float(p[0]), float(p[1])) for p in points]
        results = [
            {"lat": lat, "lon": lon, "elevation_m": None, "slope_percent": None}
            for lat, lon in pts
        ]
        if not pts or self._ds is None:
            return results

        with self._lock:
            if self._ds is None:
                return results
            rows, cols, inside = self._pixel_indices(pts)
            if not inside.any():
                return results
            idx = np.nonzero(inside)[0]
            data, r0, c0 = self._read_window(rows[idx], cols[idx], pad=1)
            lr, lc = rows[idx] - r0, cols[idx] - c0

            # Elevation (nearest pixel), then per-point geoid conversion.
            h = data[lr, lc]
            lats = np.array([pts[i][0] for i in idx])
            lons = np.array([pts[i][1] for i in idx])
            H = self._to_orthometric(lats, lons, h)

            # Slope (Horn). Requires full 3x3 neighbourhood inside the raster.
            slope = np.full(len(idx), np.nan)
            full = (
                (rows[idx] >= 1) & (rows[idx] < self._ds.height - 1)
                & (cols[idx] >= 1) & (cols[idx] < self._ds.width - 1)
            )
            if full.any():
                fr, fc = lr[full], lc[full]
                z = {  # a b c / d e f / g h i, top row = north
                    k: data[fr + dr, fc + dc]
                    for k, (dr, dc) in {
                        "a": (-1, -1), "b": (-1, 0), "c": (-1, 1),
                        "d": (0, -1),                "f": (0, 1),
                        "g": (1, -1), "h": (1, 0), "i": (1, 1),
                    }.items()
                }
                res_x, res_y = self._ds.res
                flats, flons = lats[full], lons[full]
                _, _, dx = _GEOD.inv(flons, flats, flons + res_x, flats)
                _, _, dy = _GEOD.inv(flons, flats, flons, flats + res_y)
                dzdx = ((z["c"] + 2 * z["f"] + z["i"]) - (z["a"] + 2 * z["d"] + z["g"])) / (8 * np.asarray(dx))
                dzdy = ((z["g"] + 2 * z["h"] + z["i"]) - (z["a"] + 2 * z["b"] + z["c"])) / (8 * np.asarray(dy))
                # NaN propagates: any NoData neighbour -> NaN slope -> None.
                # Horn ignores the centre pixel, so require it explicitly: a
                # NoData point has no slope of its own.
                s = 100.0 * np.hypot(dzdx, dzdy)
                s[~np.isfinite(data[fr, fc])] = np.nan
                slope[full] = s

        for j, i in enumerate(idx):
            if np.isfinite(H[j]):
                results[i]["elevation_m"] = round(float(H[j]), 2)
            if np.isfinite(slope[j]):
                results[i]["slope_percent"] = round(float(slope[j]), 2)
        return results

    def get_elevation_batch(self, points: Iterable[LatLon]) -> List[Optional[float]]:
        return [r["elevation_m"] for r in self.get_terrain_features(points)]

    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        return self.get_elevation_batch([(lat, lon)])[0]

    def get_slope(self, lat: float, lon: float) -> Optional[float]:
        return self.get_terrain_features([(lat, lon)])[0]["slope_percent"]


@lru_cache(maxsize=1)
def get_dem_service() -> DEMService:
    """Process-wide DEM service, opened once on first use."""
    return DEMService(settings.DEM_PATH, settings.GEOID_GRID_PATH)


# Module-level convenience wrappers (signatures per the plan, lat/lon order).
def get_elevation(lat: float, lon: float) -> Optional[float]:
    return get_dem_service().get_elevation(lat, lon)


def get_elevation_batch(points: Iterable[LatLon]) -> List[Optional[float]]:
    return get_dem_service().get_elevation_batch(points)


def get_slope(lat: float, lon: float) -> Optional[float]:
    return get_dem_service().get_slope(lat, lon)


def get_terrain_features(points: Iterable[LatLon]) -> List[Dict]:
    return get_dem_service().get_terrain_features(points)

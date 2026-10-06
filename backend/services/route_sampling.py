"""
Resample a route geometry into evenly spaced points for per-segment scoring.

Given a route as a GeoJSON LineString in lon/lat (WGS84), produce points every
ROUTE_SAMPLE_DISTANCE_M metres along the line, each carrying lat, lng and
distance_from_start_m.

Why project first: interpolating in raw lon/lat degrees distorts spacing (a
degree of longitude is ~1.6x shorter than a degree of latitude in Kolkata), so
"every 0.001 deg" would not be even ground distance. We project the line to
UTM zone 45N (EPSG:32645), which covers Kolkata, interpolate at exact metre
intervals there, then convert each sample back to lon/lat.

Endpoint convention (documented): samples are placed at 0, d, 2d, ... up to and
INCLUDING the final endpoint. The last spacing may therefore be shorter than d.
A 750 m line at d=75 yields 11 points (0,75,...,750). A 740 m line yields 11
points too (0,75,...,675,740) because the exact end is always included. This
guarantees the destination is always represented and every sample lies on the
route. Set include_endpoint=False to drop the trailing partial point.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform as shp_transform

from config import settings

METRIC_CRS = "EPSG:32645"  # WGS84 / UTM 45N, covers Kolkata
_TO_UTM = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
_TO_WGS84 = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)

GeoJSONLine = Dict
Coord = Tuple[float, float]


@dataclass
class RouteSample:
    lat: float
    lng: float
    distance_from_start_m: float

    def to_dict(self) -> Dict:
        return {"lat": round(self.lat, 6), "lng": round(self.lng, 6),
                "distance_from_start_m": round(self.distance_from_start_m, 1)}


def _to_line_lonlat(geometry: Union[GeoJSONLine, LineString, Sequence[Coord]]) -> LineString:
    """Accept a GeoJSON LineString dict, a shapely LineString, or a [(lon,lat),...] list."""
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
    elif isinstance(geometry, dict):
        if geometry.get("type") != "LineString":
            raise ValueError(f"expected GeoJSON LineString, got type={geometry.get('type')!r}")
        coords = geometry.get("coordinates") or []
    else:
        coords = list(geometry)
    if len(coords) < 2:
        raise ValueError("route geometry needs at least 2 coordinates")
    for c in coords:
        if len(c) < 2 or not all(_finite(v) for v in c[:2]):
            raise ValueError(f"invalid coordinate in route geometry: {c!r}")
        lon, lat = float(c[0]), float(c[1])
        if abs(lon) > 180 or abs(lat) > 90:
            raise ValueError(f"coordinate out of lon/lat range: {c!r} (expected [lon, lat])")
    return LineString([(float(c[0]), float(c[1])) for c in coords])


def _finite(v) -> bool:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def sample_route(
    geometry: Union[GeoJSONLine, LineString, Sequence[Coord]],
    spacing_m: Optional[float] = None,
    include_endpoint: bool = True,
) -> List[RouteSample]:
    """Points every `spacing_m` along the route (default from settings).

    Distances are true ground metres (UTM). See the module docstring for the
    endpoint convention.
    """
    d = float(spacing_m) if spacing_m is not None else float(settings.ROUTE_SAMPLE_DISTANCE_M)
    if d <= 0:
        raise ValueError("spacing_m must be positive")

    line_ll = _to_line_lonlat(geometry)
    line_utm = shp_transform(lambda x, y, z=None: _TO_UTM.transform(x, y), line_ll)
    total = line_utm.length

    if total == 0:  # degenerate (all points identical): a single sample at the start
        lon, lat = line_ll.coords[0]
        return [RouteSample(lat=lat, lng=lon, distance_from_start_m=0.0)]

    # Distances 0, d, 2d, ... < total, then the exact endpoint (unless excluded).
    dists: List[float] = []
    x = 0.0
    while x < total - 1e-6:
        dists.append(x)
        x += d
    if include_endpoint:
        if not dists or abs(dists[-1] - total) > 1e-6:
            dists.append(total)
    elif not dists:
        dists.append(0.0)

    samples: List[RouteSample] = []
    for dist in dists:
        p_utm: Point = line_utm.interpolate(dist)
        lon, lat = _TO_WGS84.transform(p_utm.x, p_utm.y)
        samples.append(RouteSample(lat=lat, lng=lon, distance_from_start_m=dist))
    return samples


def route_length_m(geometry: Union[GeoJSONLine, LineString, Sequence[Coord]]) -> float:
    """Total ground length of the route in metres (UTM)."""
    line_ll = _to_line_lonlat(geometry)
    return shp_transform(lambda x, y, z=None: _TO_UTM.transform(x, y), line_ll).length

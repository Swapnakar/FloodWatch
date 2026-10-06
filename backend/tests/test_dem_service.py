import math

import numpy as np
import pytest
import rasterio
from pyproj import Geod
from rasterio.transform import from_origin

from config import settings
from services.dem_service import DEMService

RES = 1 / 3600  # 1 arc-second, same as CartoDEM
NODATA = -32768.0
# Small synthetic tile inside Kolkata so the real EGM96 shift applies.
WEST, NORTH = 88.30, 22.60
SIZE = 60  # pixels
GEOID = settings.GEOID_GRID_PATH

SEALDAH = (22.5675, 88.3700)
DELHI = (28.6139, 77.2090)


def _write_dem(path, data):
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32", crs="EPSG:4326",
        transform=from_origin(WEST, NORTH, RES, RES), nodata=NODATA,
    ) as dst:
        dst.write(data.astype("float32"), 1)


def _pixel_center(row, col):
    return NORTH - (row + 0.5) * RES, WEST + (col + 0.5) * RES


def _geoid_n(lat, lon):
    """Independent EGM96 undulation: bilinear read of the grid (pixel-is-point)."""
    with rasterio.open(GEOID) as g:
        # Grid bounds are half a cell outside the node lattice; nodes at (-180 + k*0.25).
        fx = (lon - (g.bounds.left + g.res[0] / 2)) / g.res[0]
        fy = ((g.bounds.top - g.res[1] / 2) - lat) / g.res[1]
        c0, r0 = int(math.floor(fx)), int(math.floor(fy))
        w = g.read(1, window=((r0, r0 + 2), (c0, c0 + 2))).astype(float)
        tx, ty = fx - c0, fy - r0
        return (w[0, 0] * (1 - tx) * (1 - ty) + w[0, 1] * tx * (1 - ty)
                + w[1, 0] * (1 - tx) * ty + w[1, 1] * tx * ty)


@pytest.fixture
def plane_dem(tmp_path):
    """Tilted plane: raw (ellipsoidal) height rises 0.3 m per column eastward."""
    cols = np.arange(SIZE)
    data = np.tile(-50.0 + 0.3 * cols, (SIZE, 1))
    data[20, 20] = NODATA
    path = tmp_path / "plane.tif"
    _write_dem(path, data)
    svc = DEMService(path, GEOID)
    yield svc, data
    svc.close()


# ------------------------------------------------------------ synthetic tests

def test_loads_and_reports_status(plane_dem):
    svc, _ = plane_dem
    report = svc.status_report()
    assert report["dem_status"] == "LOADED"
    assert report["geoid_status"] == "LOADED"
    assert report["sampling_method"] == "nearest"


def test_elevation_is_geoid_converted_per_point(plane_dem):
    svc, data = plane_dem
    lat, lon = _pixel_center(10, 30)
    expected = data[10, 30] - _geoid_n(lat, lon)  # H = h - N
    assert svc.get_elevation(lat, lon) == pytest.approx(expected, abs=0.02)


def test_nearest_pixel_lookup(plane_dem):
    svc, data = plane_dem
    lat, lon = _pixel_center(10, 30)
    # Anywhere inside the same pixel returns that pixel's value.
    nudged = svc.get_elevation(lat + RES * 0.4, lon - RES * 0.4)
    assert nudged == pytest.approx(svc.get_elevation(lat, lon), abs=0.01)


def test_slope_matches_known_gradient(plane_dem):
    svc, _ = plane_dem
    lat, lon = _pixel_center(40, 40)
    _, _, dx = Geod(ellps="WGS84").inv(lon, lat, lon + RES, lat)
    assert svc.get_slope(lat, lon) == pytest.approx(100 * 0.3 / dx, rel=1e-3)


def test_nodata_cell_returns_none(plane_dem):
    svc, _ = plane_dem
    lat, lon = _pixel_center(20, 20)
    assert svc.get_elevation(lat, lon) is None
    assert svc.get_slope(lat, lon) is None


def test_slope_none_when_neighbour_is_nodata(plane_dem):
    svc, _ = plane_dem
    lat, lon = _pixel_center(21, 21)  # diagonal neighbour of the NoData cell
    assert svc.get_elevation(lat, lon) is not None
    assert svc.get_slope(lat, lon) is None


def test_edge_pixel_has_elevation_but_no_slope(plane_dem):
    svc, _ = plane_dem
    lat, lon = _pixel_center(0, 30)
    assert svc.get_elevation(lat, lon) is not None
    assert svc.get_slope(lat, lon) is None


@pytest.mark.parametrize("lat,lon", [
    (NORTH + 0.01, WEST + 0.005),        # north of tile
    (NORTH - 0.005, WEST - 0.01),        # west of tile
    DELHI,
    (float("nan"), 88.31),
    (95.0, 88.31),
])
def test_out_of_bounds_or_invalid_returns_none(plane_dem, lat, lon):
    svc, _ = plane_dem
    assert svc.get_elevation(lat, lon) is None
    assert svc.get_slope(lat, lon) is None


def test_batch_preserves_order_with_mixed_coverage(plane_dem):
    svc, _ = plane_dem
    inside = _pixel_center(5, 5)
    nodata = _pixel_center(20, 20)
    out = svc.get_terrain_features([DELHI, inside, nodata, inside])
    assert [r["elevation_m"] is None for r in out] == [True, False, True, False]
    assert (out[1]["lat"], out[1]["lon"]) == inside
    assert svc.get_elevation_batch([]) == []


def test_missing_dem_file_does_not_raise(tmp_path):
    svc = DEMService(tmp_path / "nope.tif", GEOID)
    assert svc.status_report()["dem_status"] == "MISSING"
    assert svc.get_elevation(*SEALDAH) is None
    assert svc.get_slope(*SEALDAH) is None


def test_corrupt_dem_file_reports_error(tmp_path):
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"not a tiff")
    svc = DEMService(bad, GEOID)
    assert svc.status_report()["dem_status"] == "ERROR"
    assert svc.get_elevation(*SEALDAH) is None


def test_missing_geoid_withholds_elevation_but_keeps_slope(tmp_path):
    data = np.tile(-50.0 + 0.3 * np.arange(SIZE), (SIZE, 1))
    path = tmp_path / "plane.tif"
    _write_dem(path, data)
    svc = DEMService(path, tmp_path / "no_geoid.tif")
    lat, lon = _pixel_center(30, 30)
    report = svc.status_report()
    assert report["geoid_status"] == "MISSING"
    assert report["elevation_available"] is False
    # Never return raw ellipsoidal heights as if they were above sea level.
    assert svc.get_elevation(lat, lon) is None
    assert svc.get_slope(lat, lon) is not None


# ---------------------------------------------------------- real CartoDEM tests

real_dem = pytest.mark.skipif(
    not settings.DEM_PATH.exists(), reason="CartoDEM GeoTIFF not present in backend/data/dem/"
)


@pytest.fixture(scope="module")
def real_svc():
    svc = DEMService(settings.DEM_PATH, GEOID)
    yield svc
    svc.close()


@real_dem
def test_real_sealdah_elevation_plausible(real_svc):
    elev = real_svc.get_elevation(*SEALDAH)
    assert elev is not None and 2 <= elev <= 15


@real_dem
def test_real_value_is_raw_pixel_minus_geoid(real_svc):
    with rasterio.open(settings.DEM_PATH) as ds:
        raw = float(next(ds.sample([(SEALDAH[1], SEALDAH[0])]))[0])
    assert real_svc.get_elevation(*SEALDAH) == pytest.approx(raw - _geoid_n(*SEALDAH), abs=0.02)


@real_dem
def test_real_outside_raster_returns_none(real_svc):
    assert real_svc.get_elevation(*DELHI) is None
    assert real_svc.get_slope(*DELHI) is None


@real_dem
def test_real_slope_is_finite_non_negative(real_svc):
    s = real_svc.get_slope(*SEALDAH)
    assert s is not None and s >= 0

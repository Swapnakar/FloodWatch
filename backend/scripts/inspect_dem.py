"""
One-off DEM inspection for the Task 4 data checkpoint.

Prints CRS, bounds, resolution, dtype and NoData value of the CartoDEM GeoTIFF,
then checks that the raster actually covers Kolkata and how much of the Kolkata
window is NoData. Exits non-zero if the file is missing, unreadable, or does not
cover Kolkata.

Usage (from backend/):
    ../.venv/bin/python scripts/inspect_dem.py [path/to/dem.tif]
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

# Rough Kolkata extent (lon/lat, EPSG:4326)
KOLKATA_BOUNDS = (88.2, 22.4, 88.5, 22.7)  # (west, south, east, north)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.DEM_PATH
    print(f"DEM path: {path}")
    if not path.exists():
        print("FAIL: file not found. Place the GeoTIFF in backend/data/dem/.")
        return 1

    try:
        ds = rasterio.open(path)
    except rasterio.errors.RasterioIOError as exc:
        print(f"FAIL: could not open raster: {exc}")
        return 1

    with ds:
        print(f"Driver:       {ds.driver}")
        print(f"CRS:          {ds.crs}")
        print(f"Size:         {ds.width} x {ds.height} px, {ds.count} band(s)")
        print(f"Dtype:        {ds.dtypes[0]}")
        print(f"NoData:       {ds.nodata}")
        print(f"Resolution:   {ds.res[0]:.8f} x {ds.res[1]:.8f} (CRS units)")
        print(f"Bounds (CRS): {tuple(round(v, 6) for v in ds.bounds)}")

        if ds.crs is None:
            print("FAIL: raster has no CRS; cannot confirm location.")
            return 1

        wgs84 = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
        print(f"Bounds (WGS84 lon/lat): W={wgs84[0]:.4f} S={wgs84[1]:.4f} "
              f"E={wgs84[2]:.4f} N={wgs84[3]:.4f}")

        w, s, e, n = KOLKATA_BOUNDS
        covers = wgs84[0] <= w and wgs84[1] <= s and wgs84[2] >= e and wgs84[3] >= n
        print(f"Covers Kolkata box {KOLKATA_BOUNDS}: {'YES' if covers else 'NO'}")
        if not covers:
            print("FAIL: raster does not fully contain the Kolkata extent.")
            return 1

        # Read just the Kolkata window and summarise it.
        kb = transform_bounds("EPSG:4326", ds.crs, *KOLKATA_BOUNDS)
        window = from_bounds(*kb, transform=ds.transform).round_offsets().round_lengths()
        data = ds.read(1, window=window, masked=True)
        total = data.size
        nodata_pct = 100.0 * np.ma.count_masked(data) / total if total else 100.0
        print(f"Kolkata window: {data.shape[1]} x {data.shape[0]} px, "
              f"NoData {nodata_pct:.2f}%")
        if data.count():
            print(f"Kolkata elevation: min={data.min():.1f} "
                  f"median={np.ma.median(data):.1f} max={data.max():.1f}")

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

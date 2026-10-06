"""
Fetch the Kolkata Municipal Corporation area boundary from OpenStreetMap.

OSM relation 9381363 ("Kolkata", boundary=administrative, admin_level=8) is the
city/KMC area. The script assembles its outer ways into a polygon and checks
the area against KMC's published ~205 km². It writes data/gis/kmc_boundary.geojson.

Why this is needed: the KMC waterlogging list only covers KMC. Outside this
boundary (Salt Lake/Bidhannagar, Howrah, New Town…) "no listed pocket nearby"
means "no data", not "no waterlogging".

Data (c) OpenStreetMap contributors, ODbL 1.0.
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR  # noqa: E402

REL_ID = 9381363
OUT = DATA_DIR / "gis" / "kmc_boundary.geojson"
EXPECTED_KM2 = (180, 230)


ENDPOINTS = ("https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter")


def overpass(query: str, attempts: int = 3) -> dict:
    """POST to Overpass with retry/backoff across mirrors (public servers 429/504 under load)."""
    import time
    last = None
    for k in range(attempts):
        for url in ENDPOINTS:
            try:
                req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": query}).encode(),
                                             headers={"User-Agent": "FloodWatch-boundary/0.1"})
                return json.load(urllib.request.urlopen(req, timeout=180))
            except Exception as exc:  # HTTPError 429/504, timeouts, resets
                last = exc
                print(f"  overpass {url} failed ({exc}); retrying")
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"Overpass unavailable after {attempts} rounds: {last}")


def main() -> int:
    q = f"[out:json][timeout:120];rel({REL_ID});out geom;"
    els = overpass(q)["elements"]
    if not els:
        print(f"ERROR relation {REL_ID} not found")
        return 1
    rel = els[0]
    tags = rel.get("tags", {})
    if tags.get("name") != "Kolkata" or tags.get("admin_level") != "8":
        print("ERROR unexpected relation tags", tags)
        return 1
    outers = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
              for m in rel["members"] if m.get("type") == "way" and m.get("role") in ("outer", "")
              and len(m.get("geometry", [])) > 1]
    polys = list(polygonize(unary_union(outers)))
    if not polys:
        print("ERROR could not assemble boundary polygon")
        return 1
    geom = unary_union(polys)
    g = gpd.GeoDataFrame([{"name": "Kolkata Municipal Corporation", "osm_relation": REL_ID,
                           "source": "OpenStreetMap (c) OpenStreetMap contributors, ODbL 1.0",
                           "wikidata": tags.get("wikidata")}], geometry=[geom], crs="EPSG:4326")
    km2 = float(g.to_crs("EPSG:32645").area.iloc[0]) / 1e6
    print(f"boundary: {len(polys)} polygon part(s), area {km2:.1f} km2, bounds {[round(v, 4) for v in g.total_bounds]}")
    if not (EXPECTED_KM2[0] <= km2 <= EXPECTED_KM2[1]):
        print(f"ERROR area {km2:.1f} km2 outside plausible KMC range {EXPECTED_KM2}")
        return 1
    g.to_file(OUT, driver="GeoJSON")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

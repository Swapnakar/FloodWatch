"""
Merge per-sheet KMC extraction outputs into the runtime GIS layers.

Inputs (data/processed/kmc_extracted/):
    ward*_p*_pipes.geojson            GeoPDF sheets (extract_kmc_geopdf.py)
    ward*_cad_pipes.geojson           AutoCAD sheets that passed the GCP gates
    ward*_p*_pumping_stations.geojson
    ward*_p*_water_bodies.geojson
Outputs (data/gis/, EPSG:4326):
    drainage_network.geojson, pumping_stations.geojson, water_bodies.geojson

Only files written by a passing extraction exist in the input folder, so
everything found is merged. Pipes duplicated across overlapping sheets are
dropped (the first sheet in sorted order wins).
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR  # noqa: E402

OUT = DATA_DIR / "processed" / "kmc_extracted"
GIS = DATA_DIR / "gis"
UTM = "EPSG:32645"
DUPLICATE_HAUSDORFF_M = 3.0


def dedupe_lines(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, int]:
    """Drop pipes that duplicate a pipe from an EARLIER sheet; same-sheet parallels are real."""
    utm = gdf.to_crs(UTM)
    keep: List[int] = []
    prior: List = []
    dropped = 0
    for sheet in list(dict.fromkeys(utm["source_sheet"])):
        idx = np.nonzero((utm["source_sheet"] == sheet).to_numpy())[0]
        tree = STRtree(prior) if prior else None
        accepted = []
        for i in idx:
            g = utm.geometry.iloc[i]
            dup = tree is not None and any(
                g.hausdorff_distance(prior[j]) < DUPLICATE_HAUSDORFF_M
                for j in tree.query(g.buffer(DUPLICATE_HAUSDORFF_M)))
            if dup:
                dropped += 1
            else:
                keep.append(int(i))
                accepted.append(g)
        prior.extend(accepted)
    return gdf.iloc[keep].reset_index(drop=True), dropped


def osm_road_distance(gdf: gpd.GeoDataFrame, step_m: float = 10.0) -> dict:
    """Independent check: distance from pipe samples to OSM highway centrelines.

    Sends only a bounding box to the public Overpass API.
    """
    w, s, e, n = gdf.total_bounds
    cache = OUT / "osm_cache"
    cache.mkdir(parents=True, exist_ok=True)
    key = cache / f"roads_{s:.4f}_{w:.4f}_{n:.4f}_{e:.4f}.json"
    if key.exists():
        lines = json.loads(key.read_text())
    else:
        q = f'[out:json][timeout:90];way["highway"]({s - .002},{w - .002},{n + .002},{e + .002});out geom;'
        req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                     data=urllib.parse.urlencode({"data": q}).encode(),
                                     headers={"User-Agent": "FloodWatch-georef-check/0.1"})
        els = json.load(urllib.request.urlopen(req, timeout=120))["elements"]
        lines = [[(p["lon"], p["lat"]) for p in el["geometry"]] for el in els if len(el.get("geometry", [])) > 1]
        key.write_text(json.dumps(lines))
    roads = gpd.GeoSeries([LineString(l) for l in lines], crs=4326).to_crs(UTM)
    tree = STRtree(list(roads.geometry))
    d = []
    for g in gdf.to_crs(UTM).geometry:
        for t in np.arange(0, g.length + 1e-9, step_m):
            p = g.interpolate(t)
            d.append(p.distance(roads.geometry.iloc[int(tree.query_nearest(p)[0])]))
    d = np.array(d)
    return {"samples": int(len(d)), "median_m": round(float(np.median(d)), 2),
            "p90_m": round(float(np.percentile(d, 90)), 2), "within_15m": round(float(np.mean(d < 15)), 3)}


def _read_all(pattern: str) -> List[gpd.GeoDataFrame]:
    return [gpd.read_file(p) for p in sorted(OUT.glob(pattern))]


def main() -> int:
    GIS.mkdir(parents=True, exist_ok=True)
    parts = _read_all("ward*_p*_pipes.geojson") + _read_all("ward*_cad_pipes.geojson")
    if parts:
        pipes = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
        pipes, dropped = dedupe_lines(pipes)
        if "segment_id" in pipes:
            pipes = pipes.drop(columns="segment_id")
        pipes.insert(0, "segment_id", [f"kmc_w{int(w)}_{i:05d}" for i, w in enumerate(pipes["ward"])])
        for col in ("pipe_diameter_mm", "label_diameter_mm"):
            if col in pipes:
                pipes[col] = pipes[col].astype("Int64")
        for col in ("label_agrees", "label_conflict_split"):
            if col in pipes:
                pipes[col] = pipes[col].astype("boolean")
        pipes.to_file(GIS / "drainage_network.geojson", driver="GeoJSON")
        print(f"drainage_network.geojson: {len(pipes)} segments from wards "
              f"{sorted(pipes['ward'].unique().tolist())} ({dropped} cross-sheet duplicates dropped)")
    st = _read_all("ward*_pumping_stations.geojson")
    if st:
        s = gpd.GeoDataFrame(pd.concat(st, ignore_index=True), crs="EPSG:4326")
        s.to_file(GIS / "pumping_stations.geojson", driver="GeoJSON")
        print(f"pumping_stations.geojson: {len(s)} points")
    wb = _read_all("ward*_water_bodies.geojson")
    if wb:
        w = gpd.GeoDataFrame(pd.concat(wb, ignore_index=True), crs="EPSG:4326")
        wards = ",".join(sorted({str(int(x)) for x in w["ward"]}))
        merged = w.to_crs(UTM).dissolve().explode(index_parts=False).reset_index(drop=True)
        merged = merged[["geometry"]].to_crs("EPSG:4326")
        merged["source"] = "KMC ward sewerage network map (ArcMap GeoPDF), water raster"
        merged["extraction_method"] = "geopdf_raster_polygonised"
        merged["wards"] = wards
        merged.to_file(GIS / "water_bodies.geojson", driver="GeoJSON")
        print(f"water_bodies.geojson: {len(merged)} polygons")

    # Always validate what was just written; a bad merge must not pass silently.
    import validate_data
    print("\n--- validate_data ---")
    return validate_data.main([])


if __name__ == "__main__":
    sys.exit(main())

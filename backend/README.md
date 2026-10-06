# FloodWatch backend

FastAPI service. Legacy endpoints (`/api/predict`, `/api/predict/batch`, `/api/model/info`, `/api/nowcast`, `/api/drainage`) are served by `main.py` and currently use the synthetic prototype model (`synthetic_v1`).

## Local setup

```sh
python3 -m venv .venv                      # from repo root; .venv/ is gitignored
.venv/bin/pip install -r backend/requirements-dev.txt
cp backend/.env.example backend/.env       # then fill in real values
cd backend
../.venv/bin/uvicorn main:app --reload
../.venv/bin/python -m pytest              # run from backend/
```

Settings live in `config.py` (Pydantic Settings). Secrets have no defaults: if unset they are `None` and the dependent service reports itself unavailable instead of crashing the app. Relative data paths resolve against `backend/`.

## Data layout

| Directory | Contents | Committed? |
|---|---|---|
| `data/dem/` | CartoDEM GeoTIFF (`P5_PAN_CD_N22_000_E088_000_DEM_30m.tif`) | No (`*.tif` ignored) |
| `data/geoid/` | `us_nga_egm96_15.tif`, PROJ's EGM96 geoid grid (2.7 MB, public domain, NGA; from cdn.proj.org, sha256 `db493027…e78d`). Used to convert CartoDEM's ellipsoidal heights to EGM96 heights above sea level | Yes |
| `data/raw/` | KMC drainage PDFs / Word docs | No |
| `data/gis/` | `drainage_network.geojson`, `pumping_stations.geojson`, `water_bodies.geojson`, rebuilt by `scripts/merge_kmc_layers.py` from the KMC extraction (wards 107/108 via `extract_kmc_geopdf.py`; AutoCAD wards via `extract_kmc_cad.py`, see `docs/KMC_CAD_GEOREFERENCING.md`) | Yes |
| `data/historical/` | `water_logging_09_06_2017.pdf` (KMC 2017 action plan, the source); `kmc_2017_pockets_raw.csv` (347 pockets parsed from Section C by `scripts/parse_kmc_waterlogging_pdf.py`); `waterlogging_points.csv` (runtime file, built by `scripts/geocode_waterlogging.py`); optional `manual_coordinates.csv` (coordinates you verified) | PDF and raw CSV: yes. `waterlogging_points.csv`, `geocode_cache.json` and `geocode_review.csv`: **no**, because Mapbox permanent-geocoding results may not be redistributed and the repo is public. Deploy them to Render as a Secret File |
| `data/processed/` | `training_data.csv` built by `scripts/build_training_dataset.py` (label = real KMC `historical_waterlogging`; terrain/water/rainfall features; drainage sparse). Gitignored (derived from geocoded coordinates). KMC extraction artefacts under `kmc_extracted/` | Mixed; see .gitignore |

## Validating GIS data

```sh
cd backend && ../.venv/bin/python scripts/validate_data.py   # exit 1 + row list on any problem
```

It checks CRS, geometry validity and type, the Kolkata bounding box (which also catches projected coordinates labelled as 4326), required attributes, value ranges, and duplicates. It runs automatically at the end of every `merge_kmc_layers.py` rebuild.

The training dataset has its own validator (run before training):

```sh
cd backend && ../.venv/bin/python scripts/validate_training_data.py   # exit 1 on any critical problem
```

It checks schema, binary real labels, coordinate/CRS sanity, value ranges, drainage all-or-nothing per row, class balance, per-point label consistency, and leakage (no negative within the historical radius of a positive; no coordinate labelled both classes).

## System dependencies

None needed locally or on Render as long as pip installs the prebuilt wheels. `rasterio`, `pyogrio` (geopandas' IO engine), `shapely`, and `pyproj` ship wheels that bundle GDAL, GEOS, and PROJ. Verified locally on macOS arm64 / Python 3.9 (rasterio 1.4.3 with GDAL 3.9.3, pyproj 3.6.1 with PROJ 9.3.0).

If pip ever falls back to building from source (you'll see it compiling, or errors mentioning `gdal-config` or `proj`), the Python version has no matching wheel. Fix the Python version rather than installing system GDAL.

## Models

Two models coexist, honestly labelled and never conflated:

| | synthetic_v1 (legacy) | real_v1 |
|---|---|---|
| Files | `model/flood_{depth,probability}_model.json`, `model_metadata.json` | `model/real_flood_probability_model.json`, `real_model_metadata.json` |
| Provenance | physics simulation (`data/generate_training_data.py`) | KMC 2017 pocket list + DEM + drainage |
| Label | simulated flood depth/prob | `historical_waterlogging` (listed pocket) |
| Prediction | prototype estimated depth | static flood susceptibility |
| ROC-AUC | 0.97 (random hold-out, synthetic-vs-synthetic) | ~0.60 ± 0.02 (spatial 5-fold CV) |

`real_v1` is a prototype: 126 positive locations, sparse drainage, a static
susceptibility label (not a dated event, and rainfall-independent). Its ~0.60 AUC
is the honest signal in the real data, not a regression from the synthetic 0.97.
Retrain with `scripts/train_real_model.py` (validates the data first, then
spatial-CV, then refits on all points).

## IMD weather API

`services/imd_service.py` talks to `api.imd.gov.in`. Auth is a JWT from
`POST /api/oauth/token.php` (cached in memory for its ~1h lifetime; the token's
`exp` is read from its first segment), sent with the `X-API-KEY` on every data
call.

**The API key is bound to a registered static public IP.** Calls from any other
machine return `403 "IP address X not authorized"`. Locally this is expected:
`get_weather()` returns `available: false` with `degraded_cause:
"ip_not_authorized"` instead of crashing. To get live data, register the
server's egress IP on the IMD portal (a Task 25 deployment step). IMD publishes
only last-24h rainfall through this API, so `rainfall_30m/1h/3h` are always
`None`, never fabricated.

## Render deployment notes

- Python version: `/.python-version` pins Render to 3.12. `pyproj==3.6.1` (the newest release that still supports the local Python 3.9) has no wheels for 3.13+, and Render's current default for new services is 3.14. A `PYTHON_VERSION` env var in the Render dashboard overrides this file, so don't set one unless it's 3.12.x.
- Unverified: the wheel-based install has not been checked against a real Render build yet. Confirm it in the build logs on the first deploy (Task 25).
- DEM on Render: the GeoTIFF is gitignored, so a git-based deploy won't include it. Before deploying the route features, pick one: commit a clipped Kolkata-extent copy, use Git LFS, or download it during the build.
- Secrets (`IMD_*`, `MAPBOX_ACCESS_TOKEN`) go in the Render Environment settings, never in the repo.
- IMD reachability from Render's shared outbound IPs is unverified. If IMD allowlists IPs, a static outbound IP may be required.

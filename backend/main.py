import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from xgboost import XGBRegressor, XGBClassifier

app = FastAPI(title="FloodWatch API")


# =========================
# CORS
# =========================

# NOTE: allow_origins=["*"] is dev-only. Task 25 restricts this to the deployed
# frontend origin before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# REAL-DATA ROUTES (Task 18)
# Additive; legacy endpoints below are unchanged.
# =========================

from routers.routes import router as real_router  # noqa: E402
from routers.auth import router as auth_router    # noqa: E402

app.include_router(real_router)
app.include_router(auth_router)


# =========================
# LOAD XGBOOST MODELS
# =========================

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

depth_model = XGBRegressor(n_jobs=1)
depth_model.load_model(os.path.join(MODEL_DIR, "flood_depth_model.json"))

prob_model = XGBClassifier(n_jobs=1)
prob_model.load_model(os.path.join(MODEL_DIR, "flood_probability_model.json"))

# Load metadata
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
model_metadata = {}
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        model_metadata = json.load(f)

FEATURE_COLS = [
    "rainfall_30m",
    "rainfall_1h",
    "rainfall_3h",
    "horizon_minutes",
    "elevation",
    "slope",
    "imperviousness",
    "drain_capacity",
    "pipe_diameter",
    "distance_to_drain",
    "historical_floods",
]


# =========================
# REQUEST / RESPONSE MODELS
# =========================

class PredictRequest(BaseModel):
    rainfall_30m: float = Field(default=50, description="30-min rainfall intensity (mm/hr)")
    rainfall_1h: float = Field(default=50, description="1-hr rainfall intensity (mm/hr)")
    rainfall_3h: float = Field(default=100, description="3-hr accumulated rainfall (mm)")
    horizon_minutes: int = Field(default=60, description="Forecast horizon in minutes")
    elevation: float = Field(default=8.0, description="Ground elevation (m)")
    slope: float = Field(default=1.0, description="Terrain slope (%)")
    imperviousness: float = Field(default=80, description="Surface imperviousness (%)")
    drain_capacity: float = Field(default=500, description="Drainage capacity (m³/hr)")
    pipe_diameter: float = Field(default=1.0, description="Drain pipe diameter (m)")
    distance_to_drain: float = Field(default=15, description="Distance to nearest drain (m)")
    historical_floods: int = Field(default=3, description="Past flood events in last 10 years")


class BatchPredictRequest(BaseModel):
    locations: list[PredictRequest]


def classify_risk(probability: float) -> str:
    """Convert flood probability to risk level."""
    if probability >= 0.80:
        return "CRITICAL"
    elif probability >= 0.60:
        return "HIGH"
    elif probability >= 0.35:
        return "MODERATE"
    else:
        return "LOW"


def risk_color(risk: str) -> str:
    """Map risk level to a hex color."""
    return {
        "CRITICAL": "#d62828",
        "HIGH": "#f97316",
        "MODERATE": "#eab308",
        "LOW": "#22c55e",
    }.get(risk, "#94a3b8")


def predict_single(req: PredictRequest) -> dict:
    """Run XGBoost inference for a single location."""
    features = pd.DataFrame([{
        col: getattr(req, col) for col in FEATURE_COLS
    }])

    # Depth prediction
    depth = float(depth_model.predict(features)[0])
    depth = max(0.0, round(depth, 1))

    # Probability prediction
    probability = float(prob_model.predict_proba(features)[0][1])
    probability = round(probability, 3)

    risk = classify_risk(probability)

    return {
        "model_version": "synthetic_v1",
        "prediction_type": "prototype_estimated_depth",
        "water_depth_cm": depth,
        "flood_probability": probability,
        "risk_level": risk,
        "risk_color": risk_color(risk),
        "confidence": {
            "model": "XGBoost (synthetic/physics-simulation training data)",
            "features_used": len(FEATURE_COLS),
        },
    }


# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {
        "message": "FloodWatch Backend is running!",
        "legacy_model": "synthetic_v1 (physics-simulation prototype)",
        "legacy_endpoints": [
            "/api/predict",
            "/api/predict/batch",
            "/api/nowcast",
            "/api/drainage",
            "/api/manholes",
            "/api/model/info",
        ],
        "real_endpoints": [
            "/api/health",
            "/api/weather",
            "/api/geocode",
            "/api/reverse-geocode",
            "/api/route",
            "/api/route/flood-risk",
            "/api/safe-route",
            "/api/data/status",
        ],
    }


# =========================
# PREDICT (SINGLE)
# =========================

@app.post("/api/predict")
def predict(req: PredictRequest):
    result = predict_single(req)
    result["input"] = req.model_dump()
    return result


# =========================
# PREDICT (BATCH)
# =========================

@app.post("/api/predict/batch")
def predict_batch(req: BatchPredictRequest):
    results = []
    for loc in req.locations:
        result = predict_single(loc)
        result["input"] = loc.model_dump()
        results.append(result)
    return {"predictions": results}


# =========================
# MODEL INFO
# =========================

@app.get("/api/model/info")
def model_info():
    from services.model_registry import synthetic_descriptor
    return {
        **synthetic_descriptor(),  # model_version, training_data_type, features_used, last_trained_at
        "model_type": model_metadata.get("model_type", "XGBoost"),
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "depth_model": model_metadata.get("depth_model", {}),
        "probability_model": model_metadata.get("probability_model", {}),
        "feature_importance": model_metadata.get("feature_importance", {}),
    }


# =========================
# NOWCAST (LEGACY)
# =========================

@app.get("/api/nowcast")
def nowcast(
    rainfall: float = 50,
    hours: int = 1
):
    """LEGACY endpoint. Simple rainfall->risk threshold table, not the real
    model. Kept for backward compatibility; prefer /api/safe-route and
    /api/weather. Response is tagged legacy so clients don't mistake it for
    real_v1 output."""

    # Flood risk calculation

    if rainfall < 20:
        risk = "LOW"
        probability = 15

    elif rainfall < 50:
        risk = "MODERATE"
        probability = 40

    elif rainfall < 80:
        risk = "HIGH"
        probability = 70

    else:
        risk = "CRITICAL"
        probability = 90


    return {
        "legacy": True,
        "model_version": "legacy_threshold_table",
        "rainfall_mm_per_hr": rainfall,
        "forecast_hours": hours,
        "flood_probability": probability,
        "risk_level": risk
    }


# =========================
# DRAINAGE
# =========================

DRAINAGE_GEOJSON = os.path.join(os.path.dirname(__file__), "data", "gis", "drainage_network.geojson")


@app.get("/api/drainage")
def drainage_network():
    if os.path.exists(DRAINAGE_GEOJSON):
        with open(DRAINAGE_GEOJSON, "r") as f:
            return json.load(f)
    return {"type": "FeatureCollection", "features": []}


# =========================
# MANHOLES (inferred from pipe network nodes)
# =========================

_manhole_cache = {"mtime": None, "data": None}


def _build_manholes() -> dict:
    """KMC sheets give no surveyed manhole layer, so manholes are inferred at
    pipe-network nodes (segment endpoints, snapped to ~1 m). Sewer manholes sit
    at exactly these places: junctions, direction/diameter changes and line ends.
    Every feature is tagged inferred=True so it is never shown as survey data."""
    with open(DRAINAGE_GEOJSON) as f:
        segments = json.load(f).get("features", [])

    nodes = {}
    for seg in segments:
        coords = (seg.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        props = seg.get("properties") or {}
        for lon, lat, *_ in (coords[0], coords[-1]):
            key = (round(lon, 5), round(lat, 5))  # ~1 m snap
            n = nodes.setdefault(key, {"lon": 0.0, "lat": 0.0, "count": 0,
                                       "diameters": set(), "ward": props.get("ward")})
            n["lon"] += lon
            n["lat"] += lat
            n["count"] += 1
            if props.get("pipe_diameter_mm"):
                n["diameters"].add(props["pipe_diameter_mm"])

    kind_for = lambda deg: "junction" if deg >= 3 else ("joint" if deg == 2 else "terminal")
    features = []
    for i, n in enumerate(nodes.values()):
        lon, lat = n["lon"] / n["count"], n["lat"] / n["count"]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]},
            "properties": {
                "manhole_id": f"MH-{i + 1:05d}",
                "kind": kind_for(n["count"]),
                "connected_pipes": n["count"],
                "max_diameter_mm": max(n["diameters"]) if n["diameters"] else None,
                "ward": n["ward"],
                "inferred": True,
                "source": "Inferred from KMC sewer network nodes",
            },
        })
    return {"type": "FeatureCollection", "features": features}


@app.get("/api/manholes")
def manholes():
    if not os.path.exists(DRAINAGE_GEOJSON):
        return {"type": "FeatureCollection", "features": []}
    mtime = os.path.getmtime(DRAINAGE_GEOJSON)
    if _manhole_cache["mtime"] != mtime:
        _manhole_cache["data"] = _build_manholes()
        _manhole_cache["mtime"] = mtime
    return _manhole_cache["data"]
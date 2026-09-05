import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from xgboost import XGBRegressor, XGBClassifier

app = FastAPI(title="FloodWatch API")


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "water_depth_cm": depth,
        "flood_probability": probability,
        "risk_level": risk,
        "risk_color": risk_color(risk),
        "confidence": {
            "model": "XGBoost",
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
        "model": "XGBoost",
        "endpoints": [
            "/api/predict",
            "/api/predict/batch",
            "/api/nowcast",
            "/api/drainage",
            "/api/model/info",
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
    return {
        "model_type": model_metadata.get("model_type", "XGBoost"),
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "depth_model": model_metadata.get("depth_model", {}),
        "probability_model": model_metadata.get("probability_model", {}),
        "feature_importance": model_metadata.get("feature_importance", {}),
    }


# =========================
# NOWCAST (legacy fallback)
# =========================

@app.get("/api/nowcast")
def nowcast(
    rainfall: float = 50,
    hours: int = 1
):

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
        "rainfall_mm_per_hr": rainfall,
        "forecast_hours": hours,
        "flood_probability": probability,
        "risk_level": risk
    }


# =========================
# DRAINAGE
# =========================

@app.get("/api/drainage")
def drainage_network():

    return {
        "drainage_nodes": 24,
        "pipe_segments": 31,
        "overcapacity_nodes": 3,
        "network_loading": 52
    }
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
# HOME
# =========================

@app.get("/")
def home():
    return {
        "message": "FloodWatch Backend is running!"
    }


# =========================
# NOWCAST
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
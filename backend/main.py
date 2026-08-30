from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FloodWatch API")

# Allow React frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "FloodWatch Backend is running!"
    }


@app.get("/api/nowcast")
def nowcast(rainfall: float = 50, hours: int = 1):
    """
    Simple flood prediction logic.
    """

    if rainfall < 20:
        risk = "LOW"
    elif rainfall < 50:
        risk = "MODERATE"
    elif rainfall < 80:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    return {
        "rainfall": rainfall,
        "forecast_hours": hours,
        "flood_risk": risk
    }


@app.get("/api/drainage")
def drainage_network():
    return {
        "drainage_nodes": 24,
        "pipe_segments": 31,
        "overcapacity_nodes": 3,
        "network_loading": 52
    }
"""
FloodWatch — Synthetic Training Data Generator
================================================
Generates physics-informed training data for XGBoost flood prediction.

Approach:
  - Divides Kolkata into a 500m × 500m grid (~200 cells)
  - Each cell gets realistic terrain/drainage features
  - Runs 7 rainfall scenarios × 4 forecast horizons per cell
  - Computes water depth using simplified urban hydrology:
      * Rational method for runoff volume
      * Drainage removal from pipe capacity
      * Excess water → depth from cell geometry
  - Computes flood probability from depth thresholds

Output: training_data.csv
"""

import csv
import math
import os
import random

random.seed(42)

# ─────────────────────────────────────────────
# Kolkata bounding box (approximate urban core)
# ─────────────────────────────────────────────
LAT_MIN, LAT_MAX = 22.48, 22.62
LNG_MIN, LNG_MAX = 88.28, 88.44

CELL_SIZE_M = 500  # 500m × 500m grid
CELL_AREA_M2 = CELL_SIZE_M * CELL_SIZE_M  # 250,000 m²

# Convert degrees to meters (approx at Kolkata's latitude)
DEG_LAT_TO_M = 111_320
DEG_LNG_TO_M = 111_320 * math.cos(math.radians(22.55))

lat_steps = int((LAT_MAX - LAT_MIN) * DEG_LAT_TO_M / CELL_SIZE_M)
lng_steps = int((LNG_MAX - LNG_MIN) * DEG_LNG_TO_M / CELL_SIZE_M)

# ─────────────────────────────────────────────
# Rainfall scenarios (mm/hr)
# ─────────────────────────────────────────────
RAINFALL_SCENARIOS = [20, 40, 60, 80, 100, 120, 150]

# ─────────────────────────────────────────────
# Forecast horizons (minutes)
# ─────────────────────────────────────────────
HORIZONS = [30, 60, 120, 180]


def generate_cell_features(lat: float, lng: float) -> dict:
    """
    Generate realistic terrain and drainage features for a grid cell.
    Kolkata is very flat (3–15m elevation) with heavy urbanisation.
    """
    # Distance from river (Hooghly on the west)
    dist_from_river = (lng - LNG_MIN) / (LNG_MAX - LNG_MIN)

    # Elevation: lower near the river, slightly higher inland
    # Kolkata is notoriously flat — most areas 3–12m
    base_elevation = 3.0 + dist_from_river * 9.0
    elevation = base_elevation + random.gauss(0, 1.5)
    elevation = max(2.0, min(15.0, elevation))

    # Slope: very flat city
    slope = random.uniform(0.1, 3.0)

    # Imperviousness: higher in dense urban core (center of bounding box)
    lat_center = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) - 0.5
    lng_center = (lng - LNG_MIN) / (LNG_MAX - LNG_MIN) - 0.5
    dist_from_center = math.sqrt(lat_center**2 + lng_center**2)
    imperviousness = 95 - dist_from_center * 60 + random.gauss(0, 5)
    imperviousness = max(45.0, min(98.0, imperviousness))

    # Drain capacity: better infrastructure in central areas
    drain_capacity = 800 - dist_from_center * 800 + random.gauss(0, 100)
    drain_capacity = max(150.0, min(1200.0, drain_capacity))

    # Pipe diameter (meters): correlates with drain capacity
    pipe_diameter = 0.3 + (drain_capacity / 1200) * 1.5 + random.gauss(0, 0.1)
    pipe_diameter = max(0.3, min(1.8, pipe_diameter))

    # Distance to nearest drain (meters)
    distance_to_drain = 5 + (1 - drain_capacity / 1200) * 45 + random.gauss(0, 5)
    distance_to_drain = max(2.0, min(60.0, distance_to_drain))

    # Historical flood frequency (events in last 10 years)
    # Low elevation + high imperviousness + low drain capacity → more floods
    flood_score = (
        (15 - elevation) / 12 * 0.4
        + imperviousness / 100 * 0.3
        + (1 - drain_capacity / 1200) * 0.3
    )
    historical_floods = int(flood_score * 15 + random.gauss(0, 1.5))
    historical_floods = max(0, min(20, historical_floods))

    return {
        "lat": round(lat, 5),
        "lng": round(lng, 5),
        "elevation": round(elevation, 1),
        "slope": round(slope, 2),
        "imperviousness": round(imperviousness, 1),
        "drain_capacity": round(drain_capacity, 0),
        "pipe_diameter": round(pipe_diameter, 2),
        "distance_to_drain": round(distance_to_drain, 1),
        "historical_floods": historical_floods,
    }


def compute_water_depth(
    rainfall_1h: float,
    horizon_min: int,
    features: dict,
) -> float:
    """
    Simplified physics-based water depth computation.

    Uses:
      - Rational method: Q = C × i × A
        where C = runoff coefficient (from imperviousness),
              i = rainfall intensity (m/s),
              A = cell area (m²)
      - Drainage removal: based on drain capacity
      - Excess volume → depth over the cell

    Horizon effect: longer horizon = more accumulated rainfall
    and more drainage time, but also more total inflow.
    """
    # --- Runoff coefficient from imperviousness ---
    # Highly impervious → C ≈ 0.90–0.95
    # Permeable areas → C ≈ 0.30–0.50
    C = 0.30 + (features["imperviousness"] / 100) * 0.65

    # --- Rainfall over the forecast horizon ---
    duration_hr = horizon_min / 60.0
    # Rainfall intensity can vary; use IDF-like decay for longer durations
    # (shorter durations are more intense per unit time)
    intensity_factor = 1.0 / (1.0 + 0.3 * (duration_hr - 1.0))
    effective_intensity = rainfall_1h * intensity_factor  # mm/hr

    # Total rainfall depth over duration (mm)
    total_rainfall_mm = effective_intensity * duration_hr

    # --- Runoff volume (m³) ---
    runoff_mm = total_rainfall_mm * C
    runoff_m3 = runoff_mm / 1000.0 * CELL_AREA_M2

    # --- Drainage removal (m³) ---
    # Drain capacity is m³/hr; over the duration, drains remove water
    # But drainage efficiency drops as system loads up
    drain_efficiency = max(
        0.2,
        1.0 - (rainfall_1h / 200.0)  # efficiency drops at high rainfall
    )
    drainage_removal_m3 = (
        features["drain_capacity"] * duration_hr * drain_efficiency
    )

    # --- Distance to drain penalty ---
    # Further from drain → slower removal
    drain_distance_factor = max(0.3, 1.0 - features["distance_to_drain"] / 100)
    drainage_removal_m3 *= drain_distance_factor

    # --- Excess water ---
    excess_m3 = max(0, runoff_m3 - drainage_removal_m3)

    # --- Depth (cm) ---
    # Water accumulates in low-lying areas.
    # Lower elevation → water pools more (topographic factor)
    topo_factor = max(0.5, 2.0 - features["elevation"] / 10.0)

    # Slope: steeper → water flows away faster
    slope_factor = max(0.3, 1.0 - features["slope"] / 5.0)

    depth_m = (excess_m3 / CELL_AREA_M2) * topo_factor * slope_factor

    # Convert to cm
    depth_cm = depth_m * 100

    # Historical flood frequency adds a small bias
    # (areas that flood often likely have poor local drainage / low spots)
    historical_bias = features["historical_floods"] * 0.5
    depth_cm += historical_bias

    # Add small noise for realism
    depth_cm += random.gauss(0, 1.5)
    depth_cm = max(0.0, round(depth_cm, 1))

    return depth_cm


def compute_flood_probability(depth_cm: float) -> float:
    """
    Convert water depth to flood probability using a sigmoid.
    Threshold: ~10 cm marks the onset of significant urban flooding.
    """
    # Sigmoid centered around 10 cm
    x = (depth_cm - 10) / 5.0
    prob = 1.0 / (1.0 + math.exp(-x))
    return round(prob, 3)


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "training_data.csv")

    fieldnames = [
        "lat",
        "lng",
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
        "water_depth_cm",
        "flood_probability",
    ]

    rows = []
    cell_count = 0

    for i in range(lat_steps):
        for j in range(lng_steps):
            lat = LAT_MIN + (i + 0.5) * CELL_SIZE_M / DEG_LAT_TO_M
            lng = LNG_MIN + (j + 0.5) * CELL_SIZE_M / DEG_LNG_TO_M

            features = generate_cell_features(lat, lng)
            cell_count += 1

            for rainfall_1h in RAINFALL_SCENARIOS:
                # Derive sub-hourly and multi-hour rainfall from 1h intensity
                rainfall_30m = round(rainfall_1h * random.uniform(1.1, 1.4), 1)
                rainfall_3h = round(rainfall_1h * random.uniform(2.2, 3.0), 1)

                for horizon in HORIZONS:
                    depth = compute_water_depth(rainfall_1h, horizon, features)
                    prob = compute_flood_probability(depth)

                    row = {
                        "lat": features["lat"],
                        "lng": features["lng"],
                        "rainfall_30m": rainfall_30m,
                        "rainfall_1h": rainfall_1h,
                        "rainfall_3h": rainfall_3h,
                        "horizon_minutes": horizon,
                        "elevation": features["elevation"],
                        "slope": features["slope"],
                        "imperviousness": features["imperviousness"],
                        "drain_capacity": features["drain_capacity"],
                        "pipe_diameter": features["pipe_diameter"],
                        "distance_to_drain": features["distance_to_drain"],
                        "historical_floods": features["historical_floods"],
                        "water_depth_cm": depth,
                        "flood_probability": prob,
                    }
                    rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} training samples from {cell_count} grid cells")
    print(f"Rainfall scenarios: {len(RAINFALL_SCENARIOS)}")
    print(f"Forecast horizons: {len(HORIZONS)}")
    print(f"Output: {output_path}")

    # Print sample statistics
    depths = [r["water_depth_cm"] for r in rows]
    probs = [r["flood_probability"] for r in rows]
    flood_count = sum(1 for d in depths if d > 10)

    print(f"\n--- Dataset Statistics ---")
    print(f"Depth range: {min(depths):.1f} – {max(depths):.1f} cm")
    print(f"Mean depth: {sum(depths)/len(depths):.1f} cm")
    print(f"Flood events (>10cm): {flood_count} ({flood_count/len(rows)*100:.1f}%)")
    print(f"Mean probability: {sum(probs)/len(probs):.3f}")


if __name__ == "__main__":
    main()

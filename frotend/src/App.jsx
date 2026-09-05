import { useState } from "react";
import {
  MapContainer,
  TileLayer,
  Circle,
  Popup,
  Polyline,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./App.css";
import { icon } from "leaflet";


// =========================
// DEFAULT LOCATION DATA
// =========================

const DEFAULT_LOCATIONS = [
  {
    name: "Sealdah",
    lat: 22.565,
    lng: 88.371,
    elevation: 7.2,
    slope: 0.8,
    imperviousness: 92,
    drain_capacity: 350,
    pipe_diameter: 0.9,
    distance_to_drain: 8,
    historical_floods: 8,
  },
  {
    name: "EM Bypass",
    lat: 22.535,
    lng: 88.397,
    elevation: 8.5,
    slope: 1.2,
    imperviousness: 85,
    drain_capacity: 450,
    pipe_diameter: 1.0,
    distance_to_drain: 12,
    historical_floods: 5,
  },
  {
    name: "Howrah",
    lat: 22.595,
    lng: 88.263,
    elevation: 6.5,
    slope: 0.6,
    imperviousness: 88,
    drain_capacity: 380,
    pipe_diameter: 0.8,
    distance_to_drain: 10,
    historical_floods: 7,
  },
  {
    name: "Salt Lake",
    lat: 22.58,
    lng: 88.42,
    elevation: 10.2,
    slope: 1.8,
    imperviousness: 72,
    drain_capacity: 600,
    pipe_diameter: 1.2,
    distance_to_drain: 18,
    historical_floods: 2,
  },
  {
    name: "Esplanade",
    lat: 22.565,
    lng: 88.35,
    elevation: 9.0,
    slope: 1.4,
    imperviousness: 78,
    drain_capacity: 550,
    pipe_diameter: 1.1,
    distance_to_drain: 15,
    historical_floods: 3,
  },
  {
    name: "Park Street",
    lat: 22.553,
    lng: 88.352,
    elevation: 9.5,
    slope: 1.5,
    imperviousness: 75,
    drain_capacity: 520,
    pipe_diameter: 1.0,
    distance_to_drain: 20,
    historical_floods: 2,
  },
];

const BACKEND_URL = "https://floodwatch-x33s.onrender.com";
// For local development, use: "http://localhost:8000"

const RISK_COLORS = {
  CRITICAL: "#d62828",
  HIGH: "#f97316",
  MODERATE: "#eab308",
  LOW: "#22c55e",
};


function App() {
  // ── Rainfall & forecast controls ──
  const [rainfall, setRainfall] = useState(50);
  const [leadTime, setLeadTime] = useState("1 hour");

  // ── Terrain / drainage feature controls ──
  const [elevation, setElevation] = useState(8.0);
  const [slope, setSlope] = useState(1.0);
  const [imperviousness, setImperviousness] = useState(80);
  const [drainCapacity, setDrainCapacity] = useState(500);
  const [pipeDiameter, setPipeDiameter] = useState(1.0);
  const [distanceToDrain, setDistanceToDrain] = useState(15);
  const [historicalFloods, setHistoricalFloods] = useState(3);

  // ── Prediction state ──
  const [prediction, setPrediction] = useState(null);
  const [batchPredictions, setBatchPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [routeMessage, setRouteMessage] = useState(
    "No route analysis requested",
  );

  // ── Toggle for feature panel ──
  const [showFeatures, setShowFeatures] = useState(false);

  // ── Model info ──
  const [modelInfo, setModelInfo] = useState(null);


  // =========================
  // DERIVED LOCATIONS (with predictions)
  // =========================

  const locations = DEFAULT_LOCATIONS.map((loc, idx) => {
    const pred = batchPredictions?.predictions?.[idx];
    return {
      ...loc,
      depth: pred ? pred.water_depth_cm : "--",
      risk: pred ? pred.risk_level : "MODERATE",
      probability: pred ? Math.round(pred.flood_probability * 100) : null,
      color: pred
        ? (RISK_COLORS[pred.risk_level] || "#94a3b8")
        : "#eab308",
      capacity: pred
        ? Math.round(100 - pred.water_depth_cm * 1.5)
        : 65,
      status: pred
        ? (pred.water_depth_cm > 25 ? "Overloaded" : "Operating")
        : "Operating",
    };
  });


  // =========================
  // XGBOOST PREDICTION
  // =========================

  async function runNowcast() {
    setLoading(true);
    setError("");
    setPrediction(null);
    setBatchPredictions(null);

    const horizonMap = {
      "30 minutes": 30,
      "1 hour": 60,
      "2 hours": 120,
      "3 hours": 180,
    };
    const horizonMinutes = horizonMap[leadTime] || 60;

    // Build the request for the "control panel" location
    const singleRequest = {
      rainfall_30m: Math.round(rainfall * 1.2),
      rainfall_1h: rainfall,
      rainfall_3h: Math.round(rainfall * 2.5),
      horizon_minutes: horizonMinutes,
      elevation,
      slope,
      imperviousness,
      drain_capacity: drainCapacity,
      pipe_diameter: pipeDiameter,
      distance_to_drain: distanceToDrain,
      historical_floods: historicalFloods,
    };

    // Build batch request for all Kolkata locations
    const batchLocations = DEFAULT_LOCATIONS.map((loc) => ({
      rainfall_30m: Math.round(rainfall * 1.2),
      rainfall_1h: rainfall,
      rainfall_3h: Math.round(rainfall * 2.5),
      horizon_minutes: horizonMinutes,
      elevation: loc.elevation,
      slope: loc.slope,
      imperviousness: loc.imperviousness,
      drain_capacity: loc.drain_capacity,
      pipe_diameter: loc.pipe_diameter,
      distance_to_drain: loc.distance_to_drain,
      historical_floods: loc.historical_floods,
    }));

    try {
      // Run both requests in parallel
      const [singleRes, batchRes] = await Promise.all([
        fetch(`${BACKEND_URL}/api/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(singleRequest),
        }),
        fetch(`${BACKEND_URL}/api/predict/batch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ locations: batchLocations }),
        }),
      ]);

      if (!singleRes.ok) throw new Error(`Single predict: HTTP ${singleRes.status}`);
      if (!batchRes.ok) throw new Error(`Batch predict: HTTP ${batchRes.status}`);

      const singleData = await singleRes.json();
      const batchData = await batchRes.json();

      setPrediction(singleData);
      setBatchPredictions(batchData);
    } catch (err) {
      console.error("Backend error:", err);
      setError(
        `Could not connect to backend. ${err.message}`,
      );
    } finally {
      setLoading(false);
    }
  }


  // =========================
  // FETCH MODEL INFO
  // =========================

  async function fetchModelInfo() {
    try {
      const res = await fetch(`${BACKEND_URL}/api/model/info`);
      const data = await res.json();
      setModelInfo(data);
    } catch {
      console.error("Could not fetch model info");
    }
  }


  // =========================
  // SAFE ROUTE (uses predictions)
  // =========================

  function findSafeRoute() {
    if (batchPredictions?.predictions) {
      const safest = batchPredictions.predictions
        .map((p, i) => ({ ...p, name: DEFAULT_LOCATIONS[i].name }))
        .sort((a, b) => a.water_depth_cm - b.water_depth_cm)[0];

      setRouteMessage(
        `Suggested safer corridor: ${safest.name}. Predicted water depth: ${safest.water_depth_cm} cm (${Math.round(safest.flood_probability * 100)}% flood probability).`,
      );
    } else {
      setRouteMessage(
        "Run the nowcast first to get AI-powered route recommendations.",
      );
    }
  }


  // =========================
  // STATS
  // =========================

  const avgDepth = batchPredictions?.predictions
    ? (batchPredictions.predictions.reduce((s, p) => s + p.water_depth_cm, 0)
       / batchPredictions.predictions.length).toFixed(1)
    : "--";

  const criticalCount = batchPredictions?.predictions
    ? batchPredictions.predictions.filter((p) => p.risk_level === "CRITICAL").length
    : 0;

  const highCount = batchPredictions?.predictions
    ? batchPredictions.predictions.filter((p) => p.risk_level === "HIGH").length
    : 0;


  return (
    <div className="app">
      {/* ================= HEADER ================= */}

      <header className="header">
        <div>
          <h1>
            FLOOD<span>WATCH</span>
          </h1>

          <p>Physics-Informed AI Flood Nowcasting • XGBoost</p>
        </div>

        <div className="header-right">
          <div className="model-badge" onClick={fetchModelInfo}>
            🤖 XGBoost Model
          </div>

          <div className="live-status">
            <span className="live-dot"></span>
            LIVE AI PREDICTION
          </div>
        </div>
      </header>

      <main className="container">
        {/* ================= NOWCAST CONTROL ================= */}

        <section className="card nowcast-card">
          <div className="section-heading">
            <div>
              <h2>⚡ AI Nowcast Control</h2>
              <p>Multi-feature XGBoost flood prediction • 0–3 hour window</p>
            </div>

            <span className="updated">
              Updated {new Date().toLocaleTimeString()}
            </span>
          </div>

          {/* Primary controls */}
          <div className="control-row">
            <div className="rain-control">
              <div className="control-label">
                <span>🌧️ Rainfall Intensity</span>
                <strong>{rainfall} mm/hr</strong>
              </div>

              <input
                type="range"
                min="0"
                max="150"
                value={rainfall}
                onChange={(e) => setRainfall(Number(e.target.value))}
              />
            </div>

            <div className="forecast-control">
              <label>Forecast Lead Time</label>

              <select
                value={leadTime}
                onChange={(e) => setLeadTime(e.target.value)}
              >
                <option>30 minutes</option>
                <option>1 hour</option>
                <option>2 hours</option>
                <option>3 hours</option>
              </select>
            </div>

            <button
              className="run-button"
              onClick={runNowcast}
              disabled={loading}
            >
              {loading ? "⏳ PREDICTING..." : "⚡ RUN AI PREDICTION"}
            </button>
          </div>

          {/* Feature toggle */}
          <button
            className="toggle-features"
            onClick={() => setShowFeatures(!showFeatures)}
          >
            {showFeatures ? "▲ Hide" : "▼ Show"} Terrain & Drainage Features
          </button>

          {/* Advanced feature controls */}
          {showFeatures && (
            <div className="features-grid">
              <FeatureSlider
                label="🏔️ Elevation"
                value={elevation}
                min={2}
                max={15}
                step={0.1}
                unit="m"
                onChange={setElevation}
              />
              <FeatureSlider
                label="📐 Slope"
                value={slope}
                min={0.1}
                max={5}
                step={0.1}
                unit="%"
                onChange={setSlope}
              />
              <FeatureSlider
                label="🏗️ Imperviousness"
                value={imperviousness}
                min={30}
                max={98}
                step={1}
                unit="%"
                onChange={setImperviousness}
              />
              <FeatureSlider
                label="🚰 Drain Capacity"
                value={drainCapacity}
                min={100}
                max={1200}
                step={10}
                unit="m³/hr"
                onChange={setDrainCapacity}
              />
              <FeatureSlider
                label="⭕ Pipe Diameter"
                value={pipeDiameter}
                min={0.3}
                max={1.8}
                step={0.1}
                unit="m"
                onChange={setPipeDiameter}
              />
              <FeatureSlider
                label="📏 Distance to Drain"
                value={distanceToDrain}
                min={2}
                max={60}
                step={1}
                unit="m"
                onChange={setDistanceToDrain}
              />
              <FeatureSlider
                label="📊 Historical Floods"
                value={historicalFloods}
                min={0}
                max={20}
                step={1}
                unit="events"
                onChange={setHistoricalFloods}
              />
            </div>
          )}
        </section>


        {/* ================= AI PREDICTION RESULT ================= */}

        {prediction && (
          <section className="card dual-prediction">
            <h2>🤖 XGBoost Prediction</h2>

            <div className="prediction-dual">
              {/* Flood probability gauge */}
              <div className="gauge-card">
                <div className="gauge-label">Flood Probability</div>
                <div
                  className="gauge-ring"
                  style={{
                    "--progress": `${Math.round(prediction.flood_probability * 100)}%`,
                    "--color": prediction.risk_color || RISK_COLORS[prediction.risk_level],
                  }}
                >
                  <div className="gauge-value">
                    {Math.round(prediction.flood_probability * 100)}%
                  </div>
                </div>
                <div
                  className="gauge-risk"
                  style={{
                    color: prediction.risk_color || RISK_COLORS[prediction.risk_level],
                  }}
                >
                  {prediction.risk_level}
                </div>
              </div>

              {/* Water depth indicator */}
              <div className="depth-card">
                <div className="depth-label">Predicted Water Depth</div>
                <div className="depth-visual">
                  <div
                    className="depth-fill"
                    style={{
                      height: `${Math.min(100, prediction.water_depth_cm * 2)}%`,
                      background: prediction.risk_color || RISK_COLORS[prediction.risk_level],
                    }}
                  ></div>
                  <div className="depth-value">
                    {prediction.water_depth_cm} cm
                  </div>
                </div>
                <div className="depth-scale">
                  <span>0 cm</span>
                  <span>50 cm</span>
                </div>
              </div>

              {/* Model confidence */}
              <div className="confidence-card">
                <div className="confidence-label">Model Details</div>
                <div className="confidence-items">
                  <div>
                    <span>Model</span>
                    <strong>{prediction.confidence?.model || "XGBoost"}</strong>
                  </div>
                  <div>
                    <span>Features Used</span>
                    <strong>{prediction.confidence?.features_used || 11}</strong>
                  </div>
                  <div>
                    <span>Rainfall Input</span>
                    <strong>{rainfall} mm/hr</strong>
                  </div>
                  <div>
                    <span>Horizon</span>
                    <strong>{leadTime}</strong>
                  </div>
                </div>
              </div>
            </div>

            {/* Raw response */}
            <details className="raw-response">
              <summary>View raw XGBoost response</summary>
              <pre>{JSON.stringify(prediction, null, 2)}</pre>
            </details>
          </section>
        )}

        {error && (
          <section className="card backend-result">
            <div className="backend-error">⚠️ {error}</div>
          </section>
        )}


        {/* ================= KPI CARDS ================= */}

        <section className="stats-grid">
          <StatCard icon="🌧️" title="RAINFALL" value={`${rainfall} mm/hr`} />
          <StatCard icon="💧" title="AVG. WATER DEPTH" value={`${avgDepth} cm`} />
          <StatCard
            icon="🚨"
            title="CRITICAL ZONES"
            value={String(criticalCount)}
            type="critical"
          />
          <StatCard
            icon="⚠️"
            title="HIGH RISK ZONES"
            value={String(highCount)}
            type="warning"
          />
        </section>


        {/* ================= MAP + SIDEBAR ================= */}

        <section className="dashboard-grid">
          {/* MAP */}

          <div className="card map-card">
            <div className="map-header">
              <div>
                <h2>Flood Risk Map</h2>

                <p>
                  {batchPredictions
                    ? "XGBoost-predicted street-level inundation • Kolkata"
                    : "Predicted street-level inundation • Kolkata"}
                </p>
              </div>

              <div className="legend">
                <span>
                  <i className="low"></i>
                  Low
                </span>

                <span>
                  <i className="moderate"></i>
                  Moderate
                </span>

                <span>
                  <i className="high"></i>
                  High
                </span>

                <span>
                  <i className="critical"></i>
                  Critical
                </span>
              </div>
            </div>

            <div className="map-wrapper">
              <MapContainer
                center={[22.5726, 88.3639]}
                zoom={9}
                scrollWheelZoom={true}
                className="map"
              >
                <TileLayer
                  attribution="&copy; OpenStreetMap contributors"
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                {locations.map((location) => (
                  <Circle
                    key={location.name}
                    center={[location.lat, location.lng]}
                    radius={1200}
                    pathOptions={{
                      color: location.color,
                      fillColor: location.color,
                      fillOpacity: 0.25,
                    }}
                  >
                    <Popup>
                      <strong>{location.name}</strong>
                      <br />
                      Water depth: {location.depth} cm
                      <br />
                      Risk: {location.risk}
                      {location.probability !== null && (
                        <>
                          <br />
                          Flood probability: {location.probability}%
                        </>
                      )}
                    </Popup>
                  </Circle>
                ))}

                <Polyline
                  positions={[
                    [22.565, 88.371],
                    [22.58, 88.42],
                    [22.535, 88.397],
                    [22.553, 88.352],
                    [22.565, 88.35],
                  ]}
                  pathOptions={{
                    color: "#2563eb",
                    weight: 4,
                    dashArray: "8 8",
                  }}
                />
              </MapContainer>
            </div>

            <div className="map-footer">
              <span className="blue-dot"></span>
              Blue dashed lines represent the drainage network.
              Colored zones represent {batchPredictions ? "XGBoost-predicted" : "predicted"} surface inundation.
            </div>
          </div>

          {/* ================= SIDEBAR ================= */}

          <aside className="sidebar">
            {/* HOTSPOTS */}

            <div className="card sidebar-card">
              <h2>🚨 Flood Hotspots</h2>

              <div className="hotspot-list">
                {[...locations]
                  .sort((a, b) => {
                    const dA = typeof a.depth === "number" ? a.depth : 0;
                    const dB = typeof b.depth === "number" ? b.depth : 0;
                    return dB - dA;
                  })
                  .slice(0, 5)
                  .map((location) => (
                    <div className="hotspot" key={location.name}>
                      <div>
                        <strong>{location.name}</strong>

                        <small>
                          {location.probability !== null
                            ? `Flood prob: ${location.probability}%`
                            : `Elevation: ${location.elevation}m`}
                        </small>
                      </div>

                      <strong
                        className="depth"
                        style={{
                          color: location.color,
                        }}
                      >
                        {location.depth}{typeof location.depth === "number" ? " cm" : ""}
                      </strong>
                    </div>
                  ))}
              </div>
            </div>

            {/* DRAINAGE NETWORK */}

            <div className="card sidebar-card">
              <h2>🔵 Drainage Network</h2>

              <div className="network-stats">
                <div>
                  <span>Drainage nodes</span>
                  <strong>24</strong>
                </div>

                <div>
                  <span>Pipe segments</span>
                  <strong>31</strong>
                </div>

                <div>
                  <span>Overcapacity nodes</span>
                  <strong>
                    {batchPredictions?.predictions
                      ? batchPredictions.predictions.filter(
                          (p) => p.water_depth_cm > 25,
                        ).length
                      : 3}
                  </strong>
                </div>
              </div>

              <div className="progress">
                <div
                  style={{
                    width: batchPredictions?.predictions
                      ? `${Math.min(
                          100,
                          (batchPredictions.predictions.reduce(
                            (s, p) => s + p.water_depth_cm,
                            0,
                          ) /
                            batchPredictions.predictions.length) *
                            3,
                        )}%`
                      : "52%",
                  }}
                ></div>
              </div>

              <small className="loading-text">Estimated network loading</small>
            </div>

            {/* SAFE ROUTE */}

            <div className="card sidebar-card route-card">
              <h2>🗺️ Safe Route Advisor</h2>

              <p>
                AI-powered route recommendation using XGBoost depth predictions.
              </p>

              <button onClick={findSafeRoute}>FIND SAFE ROUTE</button>

              <div className="route-message">{routeMessage}</div>
            </div>
          </aside>
        </section>


        {/* ================= TABLE ================= */}

        <section className="card table-card">
          <div className="table-heading">
            <div>
              <h2>Street-Level Prediction</h2>

              <p>
                Forecast +{leadTime} • Rainfall: {rainfall} mm/hr
                {batchPredictions ? " • XGBoost AI" : ""}
              </p>
            </div>

            <span className={batchPredictions ? "model-tag" : "prototype"}>
              {batchPredictions ? "XGBOOST MODEL" : "PROTOTYPE DATA"}
            </span>
          </div>

          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Location</th>
                  <th>Predicted Depth</th>
                  <th>Flood Probability</th>
                  <th>Risk</th>
                  <th>Drainage Status</th>
                </tr>
              </thead>

              <tbody>
                {locations.map((location) => (
                  <tr key={location.name}>
                    <td className="location-name">{location.name}</td>

                    <td>
                      <strong>
                        {location.depth}
                        {typeof location.depth === "number" ? " cm" : ""}
                      </strong>
                    </td>

                    <td>
                      {location.probability !== null ? (
                        <div className="prob-bar-cell">
                          <div className="prob-bar">
                            <div
                              className="prob-fill"
                              style={{
                                width: `${location.probability}%`,
                                background: location.color,
                              }}
                            ></div>
                          </div>
                          <span>{location.probability}%</span>
                        </div>
                      ) : (
                        "--"
                      )}
                    </td>

                    <td>
                      <span className={`risk ${location.risk.toLowerCase()}`}>
                        {location.risk}
                      </span>
                    </td>

                    <td>
                      <span
                        className={
                          location.status === "Overloaded"
                            ? "status overloaded"
                            : "status operating"
                        }
                      >
                        {location.status === "Overloaded"
                          ? "⚠ Overloaded"
                          : "✓ Operating"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>


        {/* ================= MODEL INFO ================= */}

        {modelInfo && (
          <section className="card model-info-card">
            <h2>🧠 Model Information</h2>

            <div className="model-metrics">
              <div className="metric">
                <span>Depth Model MAE</span>
                <strong>{modelInfo.depth_model?.mae_cm ?? "--"} cm</strong>
              </div>
              <div className="metric">
                <span>Depth Model R²</span>
                <strong>{modelInfo.depth_model?.r2 ?? "--"}</strong>
              </div>
              <div className="metric">
                <span>Probability AUC</span>
                <strong>{modelInfo.probability_model?.auc_roc ?? "--"}</strong>
              </div>
              <div className="metric">
                <span>Features</span>
                <strong>{modelInfo.n_features ?? 11}</strong>
              </div>
            </div>

            {modelInfo.feature_importance && (
              <div className="importance-chart">
                <h3>Feature Importance</h3>
                {Object.entries(modelInfo.feature_importance)
                  .sort(([, a], [, b]) => b - a)
                  .map(([feature, score]) => (
                    <div className="importance-row" key={feature}>
                      <span className="feat-name">{feature}</span>
                      <div className="feat-bar-bg">
                        <div
                          className="feat-bar"
                          style={{ width: `${score * 100}%` }}
                        ></div>
                      </div>
                      <span className="feat-score">{(score * 100).toFixed(1)}%</span>
                    </div>
                  ))}
              </div>
            )}
          </section>
        )}
      </main>


      {/* ================= FOOTER ================= */}

      <footer>
        <div>
          <strong>Urban Flood Nowcasting System</strong>
          <br />
          Physics-Informed XGBoost • Drainage + Rainfall + Terrain Intelligence
        </div>

        <span>SIH26085 • 0–3 Hour AI Forecast Window</span>
      </footer>
    </div>
  );
}


/* ================= STAT CARD ================= */

function StatCard({ icon, title, value, type }) {
  return (
    <div className={`stat-card ${type || ""}`}>
      <div className="stat-icon">{icon}</div>

      <div>
        <span>{title}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}


/* ================= FEATURE SLIDER ================= */

function FeatureSlider({ label, value, min, max, step, unit, onChange }) {
  return (
    <div className="feature-slider">
      <div className="feature-label">
        <span>{label}</span>
        <strong>
          {value} {unit}
        </strong>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export default App;

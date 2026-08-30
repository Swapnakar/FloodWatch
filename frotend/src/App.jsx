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

function App() {
  const [rainfall, setRainfall] = useState(50);
  const [leadTime, setLeadTime] = useState("1 hour");
  const [routeMessage, setRouteMessage] = useState(
    "No route analysis requested",
  );
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const locations = [
    {
      name: "Sealdah",
      depth: 54,
      risk: "CRITICAL",
      capacity: 48,
      status: "Overloaded",
      lat: 22.565,
      lng: 88.371,
      color: "#d62828",
    },
    {
      name: "EM Bypass",
      depth: 47,
      risk: "HIGH",
      capacity: 58,
      status: "Overloaded",
      lat: 22.535,
      lng: 88.397,
      color: "#f97316",
    },
    {
      name: "Howrah",
      depth: 43,
      risk: "HIGH",
      capacity: 55,
      status: "Overloaded",
      lat: 22.595,
      lng: 88.263,
      color: "#f97316",
    },
    {
      name: "Salt Lake",
      depth: 28,
      risk: "MODERATE",
      capacity: 65,
      status: "Operating",
      lat: 22.58,
      lng: 88.42,
      color: "#eab308",
    },
    {
      name: "Esplanade",
      depth: 22,
      risk: "MODERATE",
      capacity: 75,
      status: "Operating",
      lat: 22.565,
      lng: 88.35,
      color: "#eab308",
    },
    {
      name: "Park Street",
      depth: 20,
      risk: "MODERATE",
      capacity: 72,
      status: "Operating",
      lat: 22.553,
      lng: 88.352,
      color: "#eab308",
    },
  ];
  const runNowcast = async () => {
  setLoading(true);

  try {
    const hours = Number(leadTime.split(" ")[0]);

    const response = await fetch(
      `http://localhost:8000/api/nowcast?rainfall=${rainfall}&hours=${hours}`
    );

    if (!response.ok) {
      throw new Error("Backend request failed");
    }

    const data = await response.json();

    setPrediction(data);

    setRouteMessage(
      `Nowcast generated for ${rainfall} mm/hr rainfall with ${leadTime} forecast.`
    );

  } catch (error) {
    console.error(error);

    setRouteMessage(
      "Unable to connect to FloodWatch backend."
    );

  } finally {
    setLoading(false);
  }
};
  const findSafeRoute = () => {
    setRouteMessage(
      "Suggested safer corridor: Park Street. Predicted water depth: 20 cm.",
    );
  };

  return (
    <div className="app">
      {/* HEADER */}
      <header className="header">
        <div>
          <h1>
            FLOOD<span>WATCH</span>
          </h1>
          <p>Urban Flood Nowcasting & Drainage Intelligence</p>
        </div>

        <div className="live-status">
          <span className="live-dot"></span>
          LIVE SIMULATION
        </div>
      </header>

      <main className="container">
        {/* NOWCAST CONTROL */}
        <section className="card nowcast-card">
          <div className="section-heading">
            <div>
              <h2>Nowcast Control</h2>
              <p>0–3 hour urban flood prediction</p>
            </div>

            <span className="updated">
              Updated {new Date().toLocaleTimeString()}
            </span>
          </div>

          <div className="control-row">
            <div className="rain-control">
              <div className="control-label">
                <span>Rainfall Intensity</span>
                <strong>{rainfall} mm/hr</strong>
              </div>

              <input
                type="range"
                min="0"
                max="120"
                value={rainfall}
                onChange={(e) => setRainfall(e.target.value)}
              />
            </div>

            <div className="forecast-control">
              <label>Forecast Lead Time</label>

              <select
                value={leadTime}
                onChange={(e) => setLeadTime(e.target.value)}
              >
                <option>1 hour</option>
                <option>2 hours</option>
                <option>3 hours</option>
              </select>
            </div>

            <button className="run-button" 
            onClick={runNowcast}>
              ⚡RUN NOWCAST
            </button>
          </div>
        </section>

        {/* KPI CARDS */}
        <section className="stats-grid">
          <StatCard icon="🌧️" title="RAINFALL" value={`${rainfall} mm/hr`} />

          <StatCard icon="💧" title="AVG. WATER DEPTH" value="36 cm" />

          <StatCard
            icon="🚨"
            title="CRITICAL ZONES"
            value="1"
            type="critical"
          />

          <StatCard
            icon="⚠️"
            title="HIGH RISK ZONES"
            value="2"
            type="warning"
          />
        </section>
        {prediction && (
  <section className="card prediction-result">

    <div>
      <h2>🌊Prediction</h2>

      <p>
        Rainfall:{" "}
        <strong>
          {prediction.rainfall_mm_per_hr} mm/hr
        </strong>
      </p>

      <p>
        Forecast:{" "}
        <strong>
          {prediction.forecast_hours} hour(s)
        </strong>
      </p>

      <p>
        Flood Probability:{" "}
        <strong>
          {prediction.flood_probability}%
        </strong>
      </p>

      <p>
        Risk Level:{" "}
        <strong>
          {prediction.risk_level}
        </strong>
      </p>
    </div>

  </section>
)}
        {/* MAP + SIDEBAR */}
        <section className="dashboard-grid">
          {/* MAP */}
          <div className="card map-card">
            <div className="map-header">
              <div>
                <h2>Flood Risk Map</h2>
                <p>Predicted street-level inundation • Kolkata</p>
              </div>

              <div className="legend">
                <span>
                  <i className="low"></i> Low
                </span>
                <span>
                  <i className="moderate"></i> Moderate
                </span>
                <span>
                  <i className="high"></i> High
                </span>
                <span>
                  <i className="critical"></i> Critical
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

                {/* FLOOD ZONES */}
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
                    </Popup>
                  </Circle>
                ))}

                {/* PROTOTYPE DRAINAGE NETWORK */}
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
              Blue dashed lines represent the prototype drainage network.
              Colored zones represent predicted surface inundation.
            </div>
          </div>

          {/* RIGHT SIDEBAR */}
          <aside className="sidebar">
            {/* HOTSPOTS */}
            <div className="card sidebar-card">
              <h2>🚨 Flood Hotspots</h2>

              <div className="hotspot-list">
                {locations.slice(0, 5).map((location) => (
                  <div className="hotspot" key={location.name}>
                    <div>
                      <strong>{location.name}</strong>
                      <small>Drain capacity {location.capacity}%</small>
                    </div>

                    <strong className="depth">{location.depth} cm</strong>
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
                  <strong>3</strong>
                </div>
              </div>

              <div className="progress">
                <div></div>
              </div>

              <small className="loading-text">Estimated network loading</small>
            </div>

            {/* SAFE ROUTE */}
            <div className="card sidebar-card route-card">
              <h2>🗺️ Safe Route Advisor</h2>

              <p>
                Find a lower-risk route for emergency services and commuters.
              </p>

              <button onClick={findSafeRoute}>FIND SAFE ROUTE</button>

              <div className="route-message">{routeMessage}</div>
            </div>
          </aside>
        </section>

        {/* TABLE */}
        <section className="card table-card">
          <div className="table-heading">
            <div>
              <h2>Street-Level Prediction</h2>

              <p>
                Forecast +{leadTime.replace(" hour", "")} hour • Rainfall:{" "}
                {rainfall} mm/hr
              </p>
            </div>

            <span className="prototype">PROTOTYPE DATA</span>
          </div>

          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Location</th>
                  <th>Predicted Depth</th>
                  <th>Risk</th>
                  <th>Drain Capacity</th>
                  <th>Drainage Status</th>
                </tr>
              </thead>

              <tbody>
                {locations.map((location) => (
                  <tr key={location.name}>
                    <td className="location-name">{location.name}</td>

                    <td>
                      <strong>{location.depth} cm</strong>
                    </td>

                    <td>
                      <span className={`risk ${location.risk.toLowerCase()}`}>
                        {location.risk}
                      </span>
                    </td>

                    <td>{location.capacity}%</td>

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
      </main>

      {/* FOOTER */}
      <footer>
        <div>
          <strong>Urban Flood Nowcasting System</strong>
          <br />
          Drainage + Rainfall + Terrain Intelligence
        </div>

        <span>Prototype • 0–3 Hour Forecast Window</span>
      </footer>
    </div>
  );
}

/* STAT CARD COMPONENT */
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

export default App;

import { useEffect, useState, useRef, useCallback } from "react";

import { ApiError, locationRisk, reverseGeocode } from "../services/api.js";
import "./LocationRisk.css";

const RISK_LABEL = {
  HIGH: "High flood risk",
  ELEVATED: "Elevated flood risk",
  MODERATE: "Moderate flood risk",
  LOW: "Lower flood risk",
};
const RISK_COLOR = {
  HIGH: "#dc2626",
  ELEVATED: "#f97316",
  MODERATE: "#eab308",
  LOW: "#22c55e",
};

const COVERAGE_NOTE = {
  full: "Assessed from terrain, drainage, historical waterlogging and rainfall.",
  terrain_only:
    "Outside Kolkata's mapped drainage and flood records — estimated from terrain only. Treat as indicative.",
  no_data:
    "No local flood data for this location. FloodWatch only has real data for Kolkata (KMC).",
};

// AI streaming text builder — creates the full analysis string from prediction data
function buildAnalysisText(data, coords, horizonMinutes, locationName) {
  const lines = [];
  const level = data?.risk_level;
  const label = RISK_LABEL[level] || "Unknown risk";
  const prob = data.flood_probability != null ? (data.flood_probability * 100).toFixed(0) : null;
  
  lines.push(`📍 Location: ${locationName || `${coords.lat.toFixed(4)}, ${coords.lon.toFixed(4)}`}`);
  lines.push(`⏱ Prediction window: ${horizonMinutes} minutes`);
  lines.push("");
  
  lines.push(`🔬 Risk Assessment: ${label}`);
  if (prob != null) lines.push(`   Flood probability: ${prob}%`);
  if (data.water_depth_cm != null) lines.push(`   Estimated depth: ${data.water_depth_cm} cm`);
  lines.push("");
  
  lines.push("📊 Terrain Analysis:");
  if (data.elevation_m != null) lines.push(`   Elevation: ${data.elevation_m} m`);
  if (data.slope_percent != null) lines.push(`   Slope gradient: ${data.slope_percent}%`);
  if (data.nearest_drain_m != null) lines.push(`   Nearest drain: ${data.nearest_drain_m} m`);
  lines.push("");
  
  lines.push("🗂 Historical Context:");
  if (data.historical_waterlogging != null) {
    lines.push(`   Waterlogging records: ${data.historical_waterlogging ? `Yes (${data.historical_event_count || 0} nearby events)` : "None recorded nearby"}`);
  }
  if (data.nearest_historical_pocket?.location_name) {
    lines.push(`   Nearest waterlogging spot: ${data.nearest_historical_pocket.location_name}`);
    if (data.distance_to_historical_waterlogging_m != null) {
      lines.push(`   Distance: ${Math.round(data.distance_to_historical_waterlogging_m)} m`);
    }
  }
  lines.push("");
  
  lines.push("🌧 Rainfall Data:");
  if (data.rainfall?.available && data.rainfall.rainfall_24h_mm != null) {
    lines.push(`   24h rainfall: ${data.rainfall.rainfall_24h_mm} mm`);
  } else {
    lines.push("   24h rainfall: Currently unavailable");
  }
  lines.push("");
  
  const coverageNote = COVERAGE_NOTE[data.coverage] || "";
  if (coverageNote) lines.push(`ℹ️ ${coverageNote}`);
  if (data.disclaimer) lines.push(`⚠️ ${data.disclaimer}`);
  
  return lines.join("\n");
}


export default function LocationRisk({ onPredict }) {
  const [state, setState] = useState({ status: "idle" });
  const [data, setData] = useState(null);
  const [coords, setCoords] = useState(null);
  const [horizonMinutes, setHorizonMinutes] = useState(60);
  const [locationName, setLocationName] = useState(null);
  
  // AI streaming animation state
  const [streamedText, setStreamedText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [showCursor, setShowCursor] = useState(false);
  const streamRef = useRef(null);
  const resultRef = useRef(null);

  // Reverse geocode to get location name
  const fetchLocationName = useCallback(async (lat, lon) => {
    try {
      const { data: geoData } = await reverseGeocode(lat, lon);
      if (geoData?.candidates?.[0]) {
        const c = geoData.candidates[0];
        const ctx = c.context || {};
        const parts = [ctx.locality || ctx.neighborhood || c.name, ctx.place || ctx.region].filter(Boolean);
        setLocationName(parts.length > 0 ? parts.join(", ") : c.name);
      } else if (geoData?.display_name) {
        const parts = geoData.display_name.split(",");
        setLocationName(parts.slice(0, 2).join(",").trim());
      } else if (geoData?.features?.[0]?.place_name) {
        setLocationName(geoData.features[0].place_name.split(",").slice(0, 2).join(",").trim());
      }
    } catch {
      setLocationName(`${lat.toFixed(4)}, ${lon.toFixed(4)}`);
    }
  }, []);

  // Stream text character by character like Google AI
  const streamAnalysis = useCallback((fullText) => {
    if (streamRef.current) clearInterval(streamRef.current);
    setStreamedText("");
    setIsStreaming(true);
    setShowCursor(true);

    let idx = 0;
    const charsPerTick = 2;
    const tickMs = 12;

    streamRef.current = setInterval(() => {
      idx += charsPerTick;
      if (idx >= fullText.length) {
        setStreamedText(fullText);
        setIsStreaming(false);
        clearInterval(streamRef.current);
        streamRef.current = null;
        // Keep cursor blinking for a moment after done
        setTimeout(() => setShowCursor(false), 1500);
      } else {
        setStreamedText(fullText.slice(0, idx));
      }
    }, tickMs);
  }, []);

  function predict() {
    if (!coords) return;
    assess(coords.lat, coords.lon, horizonMinutes);
    onPredict?.(horizonMinutes); // also refresh the map grid
  }

  async function assess(lat, lon, horizon = horizonMinutes) {
    setState({ status: "loading" });
    setCoords({ lat, lon });
    setStreamedText("");
    setIsStreaming(false);
    try {
      const { data: riskData } = await locationRisk(lat.toFixed(6), lon.toFixed(6), horizon);
      setData(riskData);
      setState({ status: "done" });
      
      // Build and stream the analysis text
      const analysisText = buildAnalysisText(riskData, { lat, lon }, horizon, locationName);
      streamAnalysis(analysisText);
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : "Could not reach the service.";
      setState({ status: "error", message: typeof msg === "string" ? msg : "Failed." });
    }
  }

  function locate() {
    if (!("geolocation" in navigator)) {
      setState({ status: "denied", message: "Geolocation is not available in this browser." });
      return;
    }
    setState({ status: "locating" });
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        // Only detect the location here. Prediction runs when the user clicks Predict.
        setCoords({ lat: pos.coords.latitude, lon: pos.coords.longitude });
        fetchLocationName(pos.coords.latitude, pos.coords.longitude);
        setState({ status: "ready" });
      },
      (err) => setState({ status: "denied", message: err.message || "Location permission denied." }),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 },
    );
  }

  useEffect(() => {
    locate();
    return () => { if (streamRef.current) clearInterval(streamRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-scroll the streaming result
  useEffect(() => {
    if (resultRef.current && isStreaming) {
      resultRef.current.scrollTop = resultRef.current.scrollHeight;
    }
  }, [streamedText, isStreaming]);

  const level = data?.risk_level;
  const color = RISK_COLOR[level] || "#93939f";

  return (
    <div className="locrisk">
      {/* Header */}
      <div className="locrisk__head">
        <div className="locrisk__title-row">
          <div className="locrisk__icon-wrap">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z" />
              <circle cx="12" cy="10" r="3" />
            </svg>
          </div>
          <div>
            <h3>Nowcast Prediction</h3>
            {locationName && state.status === "done" && (
              <span className="locrisk__location-name">{locationName}</span>
            )}
          </div>
        </div>
        
        <p className="locrisk__subtitle">Auto-detected location • Select prediction window</p>
        
        <div className="locrisk__controls">
          <div className="locrisk__time-buttons">
            <button 
              className={`time-btn ${horizonMinutes === 30 ? 'active' : ''}`} 
              onClick={() => setHorizonMinutes(30)}
            >
              30m
            </button>
            <button 
              className={`time-btn ${horizonMinutes === 60 ? 'active' : ''}`} 
              onClick={() => setHorizonMinutes(60)}
            >
              1h
            </button>
            <button 
              className={`time-btn ${horizonMinutes === 120 ? 'active' : ''}`} 
              onClick={() => setHorizonMinutes(120)}
            >
              2h
            </button>
            <button 
              className={`time-btn ${horizonMinutes === 180 ? 'active' : ''}`} 
              onClick={() => setHorizonMinutes(180)}
            >
              3h
            </button>
          </div>

          <button
            className="locrisk__predict-btn"
            onClick={predict}
            disabled={!coords || state.status === "loading"}
            title="Run flood risk prediction at your current location"
          >
            {state.status === "loading" ? (
              <>
                <span className="locrisk__spinner"></span>
                Analyzing...
              </>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                </svg>
                Predict
              </>
            )}
          </button>
        </div>
      </div>

      {/* States */}
      {state.status === "locating" && (
        <div className="locrisk__loading-state">
          <div className="locrisk__pulse-ring"></div>
          <span>Detecting your location…</span>
        </div>
      )}

      {state.status === "ready" && (
        <div className="locrisk__loading-state">
          <span>Location detected. Choose a prediction window and click Predict.</span>
        </div>
      )}

      {state.status === "loading" && !streamedText && (
        <div className="locrisk__loading-state">
          <div className="locrisk__shimmer-bar"></div>
          <div className="locrisk__shimmer-bar short"></div>
          <div className="locrisk__shimmer-bar shorter"></div>
        </div>
      )}

      {state.status === "denied" && (
        <div className="locrisk__denied">
          <p>{state.message}</p>
          <button type="button" className="locrisk__retry-btn" onClick={locate}>
            Try again
          </button>
        </div>
      )}

      {state.status === "error" && (
        <div className="locrisk__error-msg">{state.message}</div>
      )}

      {/* Risk badge (always visible when done) */}
      {state.status === "done" && data && (
        <div className="locrisk__risk-badge" style={{ borderColor: color }}>
          <span className="locrisk__risk-dot" style={{ background: color }}></span>
          <span className="locrisk__risk-text" style={{ color }}>
            {RISK_LABEL[level] || "Risk unknown"}
          </span>
          {data.flood_probability != null && (
            <span className="locrisk__risk-prob">
              {(data.flood_probability * 100).toFixed(0)}%
            </span>
          )}
        </div>
      )}

      {/* AI-style streaming prediction output */}
      {(streamedText || isStreaming) && (
        <div className="locrisk__ai-output" ref={resultRef}>
          <div className="locrisk__ai-header">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
            <span>FLOODEXA AI Analysis</span>
            {isStreaming && <span className="locrisk__streaming-badge">Streaming</span>}
          </div>
          <pre className="locrisk__ai-text">
            {streamedText}
            {showCursor && <span className="locrisk__cursor">│</span>}
          </pre>
        </div>
      )}

      {/* Coords */}
      {coords && state.status === "done" && (
        <div className="locrisk__coords">
          {coords.lat.toFixed(4)}, {coords.lon.toFixed(4)}
        </div>
      )}
    </div>
  );
}

import React from "react";

const RISK_LABEL = {
  HIGH: "High flood risk",
  ELEVATED: "Elevated flood risk",
  MODERATE: "Moderate flood risk",
  LOW: "Lower flood risk",
};

const RISK_COLOR = {
  HIGH: "#c62222",
  ELEVATED: "#e2670c",
  MODERATE: "#b98600",
  LOW: "#1f8f4e",
};

function riskColor(level) {
  return RISK_COLOR[level] || "#adadad";
}

function computeRouteMetrics(route) {
  const segments = route.segments || [];
  const probs = segments
    .map(s => (s.flood_probability != null ? s.flood_probability : s.susceptibility))
    .filter(p => p != null && !isNaN(p));

  const avg = probs.length > 0 
    ? (probs.reduce((sum, p) => sum + p, 0) / probs.length)
    : (route.route_score != null ? route.route_score : null);

  const max = probs.length > 0 
    ? Math.max(...probs) 
    : (route.route_score != null ? route.route_score : null);
  
  // Count segments with risk > 0.25 as exposed
  const riskySegments = segments.filter(
    s => {
      const p = s.flood_probability != null ? s.flood_probability : s.susceptibility;
      return (p != null && p > 0.25) || (s.risk_level && s.risk_level !== "LOW");
    }
  );
  const totalDistanceM = route.distance_m || (route.distance_km ? route.distance_km * 1000 : 0);
  const segmentLength = segments.length > 0 ? totalDistanceM / segments.length : 0;
  const exposureDistance = Math.round(riskySegments.length * segmentLength);

  return {
    averageRisk: avg,
    maxRisk: max,
    exposureDistance: segments.length > 0 ? exposureDistance : (avg != null && avg > 0.3 ? Math.round(totalDistanceM * 0.2) : 0),
  };
}

export default function RouteRiskPanel({ result, selectedRouteIndex, onSelectRoute }) {
  if (!result || !result.routes || result.routes.length === 0) return null;

  return (
    <div className="route-risk-panel">
      {result.all_risky && (
        <div className="sra__warn" style={{ padding: "12px", background: "var(--red-light)", color: "var(--red)", borderRadius: "8px", marginBottom: "16px", fontSize: "14px", fontWeight: 500 }}>
          ⚠ No low-risk alternative found. All available routes contain elevated flood risk.
        </div>
      )}
      
      <ul className="sra__routes" style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: "12px" }}>
        {result.routes.map((route, i) => {
          const level = route.risk_level;
          const recommended = i === result.recommended_index;
          const selected = i === selectedRouteIndex;
          const metrics = computeRouteMetrics(route);
          
          let tradeoffText = "";
          if (i > 0 && result.routes[0]) {
            const baseRoute = result.routes[0];
            const diffMin = route.duration_min - baseRoute.duration_min;
            const diffKm = (route.distance_km - baseRoute.distance_km).toFixed(1);
            if (diffMin > 0 || diffKm > 0) {
              tradeoffText = ` (+${diffMin > 0 ? `${diffMin} min` : ""}${diffMin > 0 && diffKm > 0 ? ", " : ""}${diffKm > 0 ? `${diffKm} km` : ""})`;
            }
          }

          return (
            <li
              key={i}
              className={`sra__route ${recommended ? "is-recommended" : ""} ${selected ? "is-selected" : ""}`}
              onClick={() => onSelectRoute(i)}
              style={{
                padding: "16px",
                border: `2px solid ${selected ? "var(--primary)" : "var(--hairline)"}`,
                borderRadius: "12px",
                cursor: "pointer",
                background: recommended ? "var(--canvas-alt)" : "var(--canvas)",
                transition: "all 0.2s"
              }}
            >
              <div className="sra__route-top" style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                <span className="sra__route-name" style={{ fontWeight: 600, fontSize: "15px" }}>
                  Route {i + 1}
                </span>
                <span className="sra__risk" style={{ color: riskColor(level), fontSize: "14px", fontWeight: 600, display: "flex", alignItems: "center", gap: "6px" }}>
                  <i className="swatch" style={{ background: riskColor(level), width: "10px", height: "10px", borderRadius: "50%", display: "inline-block" }} />
                  {level}
                </span>
              </div>
              
              {recommended && (
                <div style={{ color: "var(--primary)", fontSize: "12px", fontWeight: 600, marginBottom: "8px" }}>
                  Among the available candidate routes, this route has the lower predicted flood risk.
                </div>
              )}

              <div className="sra__route-meta num" style={{ fontSize: "14px", color: "var(--ink-muted-48)", marginBottom: "8px" }}>
                {route.distance_km} km · {route.duration_min} min
                <span style={{ color: "var(--slate)" }}>{tradeoffText}</span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "8px", fontSize: "12px", background: "var(--canvas)", padding: "10px", borderRadius: "8px", border: "1px solid var(--hairline)" }}>
                <div>
                  <div style={{ color: "var(--slate)" }}>Average Risk</div>
                  <div style={{ fontWeight: 600 }}>
                    {metrics.averageRisk != null ? `${(metrics.averageRisk * 100).toFixed(0)}%` : "—"}
                  </div>
                </div>
                <div>
                  <div style={{ color: "var(--slate)" }}>Maximum Risk</div>
                  <div style={{ fontWeight: 600 }}>
                    {metrics.maxRisk != null ? `${(metrics.maxRisk * 100).toFixed(0)}%` : "—"}
                  </div>
                </div>
                <div>
                  <div style={{ color: "var(--slate)" }}>Flood Exposure</div>
                  <div style={{ fontWeight: 600 }}>
                    {metrics.exposureDistance != null ? `${metrics.exposureDistance} m` : "—"}
                  </div>
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

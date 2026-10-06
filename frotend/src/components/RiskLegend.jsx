import React from "react";

const RISK_COLORS = {
  CRITICAL: "#d62828",
  HIGH: "#f97316",
  ELEVATED: "#f97316",
  MODERATE: "#eab308",
  LOW: "#22c55e",
  UNKNOWN: "#94a3b8"
};

export default function RiskLegend() {
  return (
    <div className="risk-legend" style={{ background: "var(--canvas)", padding: "16px", borderRadius: "12px", boxShadow: "0 2px 12px rgba(0,0,0,0.05)", border: "1px solid var(--hairline)" }}>
      <h4 style={{ margin: "0 0 12px", fontSize: "14px", color: "var(--slate)" }}>FLOOD RISK</h4>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "12px", fontSize: "13px" }}>
        {Object.entries(RISK_COLORS).filter(([k]) => k !== "ELEVATED").map(([label, color]) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ display: "block", width: "12px", height: "12px", borderRadius: "50%", background: color }} />
            <span>{label.charAt(0).toUpperCase() + label.slice(1).toLowerCase()}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

import React, { useState } from 'react';
import './SimulationPanel.css';

export const ALERT_COLORS = {
  RED: "#dc2626",
  ORANGE: "#f97316",
  YELLOW: "#eab308",
  GREEN: "#22c55e",
  UNKNOWN: "#94a3b8",
};

const PRESETS = [
  { label: "Light", rainfall: 10, duration: 60 },
  { label: "Heavy", rainfall: 50, duration: 60 },
  { label: "Very heavy", rainfall: 100, duration: 120 },
  { label: "Extreme", rainfall: 150, duration: 180 },
];

const MODES = [
  { id: "flood", label: "Flood risk" },
  { id: "drainage", label: "Drainage network" },
];

export default function SimulationPanel({ onSimulate, isLoading, mode, onModeChange, drainageResult, error }) {
  const [rainfall, setRainfall] = useState(50);
  const [duration, setDuration] = useState(60);

  const run = (r = rainfall, d = duration) => onSimulate({ rainfall: Number(r), duration: Number(d) });

  const handleSubmit = (e) => {
    e.preventDefault();
    run();
  };

  const handleReset = () => {
    setRainfall(50);
    setDuration(60);
    onSimulate(null);
  };

  const applyPreset = (p) => {
    setRainfall(p.rainfall);
    setDuration(p.duration);
    run(p.rainfall, p.duration);
  };

  const totalMm = Math.round((rainfall * duration) / 60);
  const summary = drainageResult?.summary;

  return (
    <div className="panel sim-panel">
      <div className="panel-header">
        <h3 className="panel-title">Model Simulation</h3>
        <p className="panel-desc">
          {mode === "drainage"
            ? "See which sewer pipes overload for a given storm"
            : "Test flood predictions with custom rainfall"}
        </p>
      </div>

      <div className="sim-modes" role="tablist" aria-label="Simulation type">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            role="tab"
            aria-selected={mode === m.id}
            className={`sim-mode ${mode === m.id ? "active" : ""}`}
            onClick={() => onModeChange(m.id)}
            disabled={isLoading}
          >
            {m.label}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="sim-form">
        <div className="sim-presets">
          {PRESETS.map((p) => (
            <button key={p.label} type="button" className="sim-preset" onClick={() => applyPreset(p)} disabled={isLoading}>
              {p.label}
              <span>{p.rainfall} mm/hr · {p.duration} min</span>
            </button>
          ))}
        </div>

        <div className="sim-group">
          <label className="sim-label" htmlFor="sim-rainfall">Rainfall Intensity (mm/hr)</label>
          <div className="sim-input-wrapper">
            <input
              id="sim-rainfall"
              type="range" min="0" max="200" step="5"
              value={rainfall}
              onChange={(e) => setRainfall(Number(e.target.value))}
              className="sim-slider"
            />
            <span className="sim-value">{rainfall} mm/hr</span>
          </div>
        </div>

        <div className="sim-group">
          <label className="sim-label" htmlFor="sim-duration">Duration (minutes)</label>
          <div className="sim-input-wrapper">
            <input
              id="sim-duration"
              type="range" min="15" max="360" step="15"
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="sim-slider"
            />
            <span className="sim-value">{duration} min</span>
          </div>
        </div>

        <p className="sim-total">Total rainfall in this scenario: <strong>{totalMm} mm</strong></p>

        <div className="sim-actions">
          <button type="button" className="sim-btn secondary" onClick={handleReset} disabled={isLoading}>
            {mode === "drainage" ? "Clear" : "Reset Live"}
          </button>
          <button type="submit" className="sim-btn primary" disabled={isLoading}>
            {isLoading ? "Simulating..." : "Run Simulation"}
          </button>
        </div>
      </form>

      {error && <p className="sim-error" role="alert">{error}</p>}

      {mode === "drainage" && summary && (
        <div className="sim-result" aria-live="polite">
          <div className="sim-alert" style={{ borderColor: ALERT_COLORS[summary.overall_alert], background: `${ALERT_COLORS[summary.overall_alert]}14` }}>
            <span className="sim-alert-dot" style={{ background: ALERT_COLORS[summary.overall_alert] }} />
            <div>
              <strong style={{ color: ALERT_COLORS[summary.overall_alert] }}>{summary.overall_alert} ALERT</strong>
              <div className="sim-alert-text">{summary.overall_meaning}</div>
            </div>
          </div>

          <p className="sim-label" style={{ margin: "12px 0 8px" }}>
            {summary.pipes_assessed} pipes within 1 km of you
          </p>
          <div className="sim-counts">
            {["RED", "ORANGE", "YELLOW", "GREEN"].map((lvl) => (
              <div key={lvl} className="sim-count" style={{ borderTopColor: ALERT_COLORS[lvl] }}>
                <span className="sim-count-value">{summary.counts[lvl]}</span>
                <span className="sim-count-label">{drainageResult.alert_meaning[lvl]}</span>
              </div>
            ))}
          </div>
          {summary.estimated_overflow_m3 > 0 && (
            <p className="sim-total" style={{ marginTop: 12 }}>
              Estimated overflow nearby: <strong>{summary.estimated_overflow_m3.toLocaleString()} m³</strong>
            </p>
          )}

          <details className="sim-assumptions">
            <summary>Assumptions</summary>
            <ul>
              {drainageResult.assumptions.map((a) => <li key={a}>{a}</li>)}
            </ul>
          </details>
        </div>
      )}
    </div>
  );
}

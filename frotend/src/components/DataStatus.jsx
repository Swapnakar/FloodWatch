import { useCallback, useEffect, useState } from "react";

import { getDataStatus, pingDataStatus } from "../services/api.js";

// Colour each state honestly: green = working, amber = unknown/not yet checked,
// red = error/missing.
const STATE_TONE = {
  LOADED: "ok",
  LIVE: "ok",
  REAL: "ok",
  CONNECTED: "ok",
  UNKNOWN: "warn",
  NOT_CONFIGURED: "warn",
  MISSING: "err",
  ERROR: "err",
};

const SOURCE_LABEL = {
  dem: "Elevation (DEM)",
  drainage: "Drainage network",
  water_bodies: "Water bodies",
  pumping_stations: "Pumping stations",
  historical_waterlogging: "Historical waterlogging",
  mapbox: "Mapbox (routing)",
  imd: "IMD (rainfall)",
};

function Dot({ state }) {
  const tone = STATE_TONE[state] || "warn";
  return <i className={`ds__dot ds__dot--${tone}`} title={state} />;
}

export default function DataStatus() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [pinging, setPinging] = useState(false);

  const load = useCallback(async () => {
    try {
      const { data } = await getDataStatus();
      setStatus(data);
      setError("");
    } catch {
      setError("Backend unreachable — status unavailable.");
    }
  }, []);

  useEffect(() => {
    let alive = true;
    const run = () => alive && load();
    run();
    const t = setInterval(run, 30000); // refresh every 30s
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [load]);

  async function ping() {
    setPinging(true);
    try {
      await pingDataStatus(); // actively probe Mapbox + IMD
      await load();
    } catch {
      // load() already handles the message
    } finally {
      setPinging(false);
    }
  }

  return (
    <div className="ds" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div className="ds__head" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--hairline)", paddingBottom: "12px" }}>
        <h3 style={{ margin: 0, fontSize: "16px", color: "var(--primary)" }}>Data sources</h3>
        <button
          type="button"
          style={{ background: "var(--canvas-alt)", border: "1px solid var(--hairline)", padding: "6px 12px", borderRadius: "16px", fontSize: "12px", cursor: "pointer" }}
          onClick={ping}
          disabled={pinging}
        >
          {pinging ? "Checking…" : "Check connectivity"}
        </button>
      </div>

      {error && <div style={{ color: "var(--red)", fontSize: "13px" }}>{error}</div>}

      {status && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "16px" }}>
          {Object.entries(status.sources).map(([key, src]) => {
            const tone = STATE_TONE[src.state] || "warn";
            const color = tone === "ok" ? "var(--green)" : tone === "err" ? "var(--red)" : "var(--orange)";
            return (
              <div key={key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", background: "var(--canvas-alt)", padding: "10px 12px", borderRadius: "8px", border: "1px solid var(--hairline)" }}>
                <span style={{ fontSize: "13px", fontWeight: 500, color: "var(--primary)" }}>{SOURCE_LABEL[key] || key}</span>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  {typeof src.features === "number" && (
                    <span style={{ fontSize: "11px", color: "var(--muted)" }}>{src.features}</span>
                  )}
                  <span style={{ fontSize: "10px", fontWeight: 700, padding: "2px 6px", borderRadius: "12px", background: `color-mix(in srgb, ${color} 15%, transparent)`, color }}>
                    {src.state}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {status && (
        <div style={{ marginTop: "8px", fontSize: "12px", color: "var(--muted)", display: "flex", justifyContent: "space-between" }}>
          <span>Active model: <strong>{status.model.active_version}</strong> ({status.model.state})</span>
          <span>Live connectivity is cached from the last call.</span>
        </div>
      )}
    </div>
  );
}

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, geocode, safeRoute } from "../services/api.js";
import RouteRiskPanel from "./RouteRiskPanel";

// Honest risk vocabulary. There is deliberately no "SAFE" — the lowest band is
// "Lower risk", never a guarantee. Mirrors the backend's risk_level values.
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

// Debounced geocoding autocomplete for one endpoint field.
function useGeocodeSuggestions() {
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const timer = useRef(null);
  const seq = useRef(0);

  const query = useCallback((text) => {
    if (timer.current) clearTimeout(timer.current);
    const trimmed = (text || "").trim();
    if (trimmed.length < 3) {
      setSuggestions([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    const mine = ++seq.current;
    timer.current = setTimeout(async () => {
      try {
        const { data } = await geocode(trimmed, 5);
        if (mine === seq.current) setSuggestions(data.candidates || []);
      } catch {
        if (mine === seq.current) setSuggestions([]);
      } finally {
        if (mine === seq.current) setLoading(false);
      }
    }, 350);
  }, []);

  const clear = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    seq.current++; // invalidate any in-flight request
    setSuggestions([]);
    setLoading(false);
  }, []);

  useEffect(() => () => timer.current && clearTimeout(timer.current), []);
  return { suggestions, loading, query, clear };
}

function EndpointField({ id, label, value, chosen, onText, onPick }) {
  const { suggestions, loading, query, clear } = useGeocodeSuggestions();
  const [open, setOpen] = useState(false);

  return (
    <div className="sra__field">
      <label htmlFor={id} className="t-label">
        {label}
      </label>
      <input
        id={id}
        className="sra__input"
        type="text"
        autoComplete="off"
        value={value}
        placeholder="Type a place in Kolkata"
        onChange={(e) => {
          onText(e.target.value);
          query(e.target.value);
          setOpen(true);
        }}
        onFocus={() => value && setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
      />
      {chosen && (
        <span className="sra__chosen num">
          {chosen.lat.toFixed(4)}, {chosen.lon.toFixed(4)}
        </span>
      )}
      {open && (suggestions.length > 0 || loading) && (
        <ul className="sra__suggest">
          {loading && <li className="sra__suggest-empty">Searching…</li>}
          {suggestions.map((s) => (
            <li key={s.mapbox_id || `${s.lat},${s.lon}`}>
              <button
                type="button"
                className="sra__suggest-item"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  onPick(s);
                  clear();
                  setOpen(false);
                }}
              >
                <strong>{s.name}</strong>
                {s.full_address && <small>{s.full_address}</small>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function SafeRouteAdvisor({ horizon = 0, onResult, selectedRouteIndex, onSelectRoute }) {
  const [originText, setOriginText] = useState("");
  const [destText, setDestText] = useState("");
  const [originPick, setOriginPick] = useState(null);
  const [destPick, setDestPick] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // Auto-submit when both points are picked
  useEffect(() => {
    if (originPick && destPick) {
      findRoute();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [originPick, destPick]);

  async function findRoute() {
    if (!originText.trim() || !destText.trim()) {
      setError("Enter both a start and a destination.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    onResult?.(null);
    try {
      // Prefer an explicitly picked coordinate; otherwise send the text for
      // server-side geocoding. Never uses hardcoded locations.
      const payload = { horizon_minutes: horizon };
      if (originPick) payload.origin = { lat: originPick.lat, lon: originPick.lon };
      else payload.origin_query = originText.trim();
      if (destPick) payload.destination = { lat: destPick.lat, lon: destPick.lon };
      else payload.destination_query = destText.trim();

      const { data } = await safeRoute(payload);
      setResult(data);
      onResult?.(data);
    } catch (err) {
      // err.detail is already a normalized string from api.js.
      const msg =
        err instanceof ApiError
          ? err.detail
          : "Could not reach the routing service.";
      setError(typeof msg === "string" && msg ? msg : "Routing failed.");
    } finally {
      setLoading(false);
    }
  }

  const recommendedIndex = result?.recommended_index;

  return (
    <div className="sra">
      <div className="sra__head">
        <h3>Safe route advisor</h3>
        <p>
          Compares real driving routes by flood risk. Uses live geocoding and the
          real-data model — no route is ever called flood-free.
        </p>
      </div>

      <div className="sra__form">
        <EndpointField
          id="sra-origin"
          label="Start"
          value={originText}
          chosen={originPick}
          onText={(t) => {
            setOriginText(t);
            setOriginPick(null);
          }}
          onPick={(s) => {
            setOriginPick(s);
            setOriginText(s.name);
          }}
        />
        <EndpointField
          id="sra-dest"
          label="Destination"
          value={destText}
          chosen={destPick}
          onText={(t) => {
            setDestText(t);
            setDestPick(null);
          }}
          onPick={(s) => {
            setDestPick(s);
            setDestText(s.name);
          }}
        />
        <button
          type="button"
          className="pill pill--solid sra__go"
          onClick={findRoute}
          disabled={loading}
        >
          {loading ? "Finding route…" : "Find route"}
        </button>
      </div>

      {error && <div className="sra__error">{error}</div>}

      {result && (
        <RouteRiskPanel 
          result={result} 
          selectedRouteIndex={selectedRouteIndex} 
          onSelectRoute={onSelectRoute} 
        />
      )}
    </div>
  );
}

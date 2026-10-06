// Centralized backend access for FloodWatch.
//
// Replaces the ad-hoc fetches scattered in App.jsx. Every call goes through
// requestJSON(), which tries the configured backends in order (local first in
// dev, then the deployed URL) and reports which one answered. Callers can tell
// a real backend response apart from a failure, so the UI never presents a
// client-side fallback as if it were the real model (the pre-existing
// "Hydrological XGBoost Engine" mislabeling bug).

const REMOTE_BACKEND_URL = "https://floodwatch-x33s.onrender.com";
const LOCAL_BACKEND_URL = "http://127.0.0.1:8000";

// Allow an explicit override at build time (VITE_API_URL); otherwise try
// local then remote so `npm run dev` hits a local server first.
const ENV_URL =
  typeof import.meta !== "undefined" && import.meta.env
    ? import.meta.env.VITE_API_URL
    : undefined;

export const BACKEND_CANDIDATES = ENV_URL ? [ENV_URL] : [LOCAL_BACKEND_URL, REMOTE_BACKEND_URL];

const DEFAULT_TIMEOUT_MS = 12000;

export class ApiError extends Error {
  constructor(message, { status, body, detail } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
    // Always a human-readable string, whatever shape the backend used
    // (FastAPI detail can be a string, a dict, or a validation list).
    this.detail = normalizeDetail(detail) || message;
  }
}

// FastAPI errors arrive as {detail: <string | object | array>}. Mapbox errors
// wrap {error, message, mapbox_code}. Reduce any of these to one clear line so
// the UI never renders an object (which crashes React) or shows a blank error.
function normalizeDetail(detail) {
  if (detail == null) return null;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // pydantic validation errors: [{loc, msg, type}, ...]
    return detail
      .map((e) => (e && e.msg ? `${(e.loc || []).slice(1).join(".")} ${e.msg}`.trim() : String(e)))
      .join("; ");
  }
  if (typeof detail === "object") {
    return detail.message || detail.error || detail.reason || JSON.stringify(detail);
  }
  return String(detail);
}

// Try each candidate backend until one responds (2xx or a structured 4xx).
// Returns { data, backend }. Throws ApiError only when NO backend is reachable
// or all return 5xx, so callers can cleanly distinguish "backend said no"
// (4xx with detail) from "no backend at all".
async function requestJSON(path, { method = "GET", body, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  let lastError = null;
  for (const base of BACKEND_CANDIDATES) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(`${base}${path}`, {
        method,
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
      clearTimeout(timer);

      let data = null;
      try {
        data = await res.json();
      } catch {
        data = null;
      }

      if (res.ok) {
        return { data, backend: base };
      }
      // A 4xx with a JSON body is a definitive answer from THIS backend
      // (e.g. no route found, bad input) — surface it, don't try the next.
      if (res.status >= 400 && res.status < 500) {
        throw new ApiError(`Request failed (${res.status})`, {
          status: res.status,
          body: data,
          detail: data ? (data.detail ?? data.message ?? data) : null,
        });
      }
      // 5xx: remember and try the next candidate.
      lastError = new ApiError(`Server error (${res.status})`, { status: res.status, body: data });
    } catch (err) {
      clearTimeout(timer);
      if (err instanceof ApiError) throw err; // definitive 4xx
      lastError = err; // network/timeout: try next candidate
    }
  }
  throw new ApiError("No backend reachable", { detail: lastError ? String(lastError.message || lastError) : null });
}

// ---- endpoints ----

export function getHealth() {
  return requestJSON("/api/health");
}

export function geocode(query, limit = 5) {
  return requestJSON(`/api/geocode?q=${encodeURIComponent(query)}&limit=${limit}`);
}

export function reverseGeocode(lat, lon) {
  return requestJSON(`/api/reverse-geocode?lat=${lat}&lon=${lon}`);
}

export function getWeather() {
  return requestJSON("/api/weather");
}

export function locationRisk(lat, lon, horizonMinutes = 0, simulatedRainfall = null) {
  let url = `/api/location-risk?lat=${lat}&lon=${lon}&horizon_minutes=${horizonMinutes}`;
  if (simulatedRainfall !== null) {
    url += `&simulated_rainfall_mm=${simulatedRainfall}`;
  }
  return requestJSON(url);
}

export function getDataStatus() {
  return requestJSON("/api/data/status");
}

export function pingDataStatus() {
  return requestJSON("/api/data/status/ping", { timeoutMs: 20000 });
}

export function getModelInfo() {
  return requestJSON("/api/model/info");
}

// Safe route: origin/destination may be coordinates or text queries.
export function safeRoute({ originQuery, destinationQuery, origin, destination, profile = "driving", horizon_minutes = 60 }) {
  const body = { profile, include_segments: true, horizon_minutes };
  if (origin) body.origin = origin;
  if (destination) body.destination = destination;
  if (originQuery) body.origin_query = originQuery;
  if (destinationQuery) body.destination_query = destinationQuery;
  return requestJSON("/api/safe-route", { method: "POST", body, timeoutMs: 25000 });
}

// Legacy synthetic-model endpoints (kept for the existing nowcast UI).
export function predict(input) {
  return requestJSON("/api/predict", { method: "POST", body: input });
}

export function predictBatch(locations) {
  return requestJSON("/api/predict/batch", { method: "POST", body: { locations } });
}

export function getDrainage() {
  return requestJSON("/api/drainage");
}

export function getManholes() {
  return requestJSON("/api/manholes");
}

// Scenario flood risk for many points in one request.
export function simulatePoints(points, rainfallMmHr, durationMin) {
  return requestJSON("/api/simulate/points", {
    method: "POST",
    body: { points, rainfall_mm_hr: rainfallMmHr, duration_min: durationMin },
    timeoutMs: 30000,
  });
}

// Per-pipe drainage load + GREEN/YELLOW/ORANGE/RED alerts for a design storm.
export function simulateDrainage(rainfallMmHr, durationMin, lat, lon, radiusM) {
  const q = new URLSearchParams({ rainfall_mm_hr: rainfallMmHr, duration_min: durationMin });
  if (lat != null && lon != null && radiusM) {
    q.set("lat", lat);
    q.set("lon", lon);
    q.set("radius_m", radiusM);
  }
  return requestJSON(`/api/simulate/drainage?${q}`, { timeoutMs: 30000 });
}

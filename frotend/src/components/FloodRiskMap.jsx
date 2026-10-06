import { useEffect, useRef } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";

// Public Mapbox token (pk.) — required by Mapbox GL JS. Set VITE_MAPBOX_TOKEN
// in frotend/.env.local. Without it the map cannot render (Mapbox GL has no
// tile fallback), so we show a clear message instead of a blank canvas.
const MAPBOX_TOKEN =
  typeof import.meta !== "undefined" && import.meta.env
    ? import.meta.env.VITE_MAPBOX_TOKEN
    : undefined;
const HAS_TOKEN = Boolean(MAPBOX_TOKEN && MAPBOX_TOKEN.startsWith("pk."));

const KOLKATA = [88.3639, 22.5726]; // [lng, lat]

// Per-segment risk colours (green -> red), matching backend risk_level.
const RISK_COLOR = {
  HIGH: "#c62222",
  ELEVATED: "#e2670c",
  MODERATE: "#b98600",
  LOW: "#1f8f4e",
};
const UNKNOWN_COLOR = "#8a8a8a";

function segColor(level) {
  return RISK_COLOR[level] || UNKNOWN_COLOR;
}

// Build a FeatureCollection of coloured line pieces: each geometry edge takes
// the risk colour of the sample bounding it. Recommended route is opaque/thick,
// alternatives dimmed.
function routeSegmentFeatures(routes, recommendedIndex) {
  const features = [];
  routes.forEach((route, ri) => {
    const coords = route.geometry?.coordinates || [];
    const segs = route.segments || [];
    if (coords.length < 2) return;
    const n = coords.length - 1;
    const s = segs.length || 1;
    for (let i = 0; i < n; i++) {
      const seg = segs[Math.min(segs.length - 1, Math.floor((i / n) * s))] || {};
      features.push({
        type: "Feature",
        properties: {
          color: segColor(seg.risk_level),
          recommended: ri === recommendedIndex,
        },
        geometry: { type: "LineString", coordinates: [coords[i], coords[i + 1]] },
      });
    }
  });
  return { type: "FeatureCollection", features };
}

// Clickable sample points on the recommended route (one dot per scored segment).
function routePointFeatures(routes, recommendedIndex) {
  const route = routes[recommendedIndex];
  const features = [];
  (route?.segments || []).forEach((seg, i) => {
    if (seg.flood_probability == null) return;
    features.push({
      type: "Feature",
      properties: {
        color: segColor(seg.risk_level),
        index: i + 1,
        prob: Math.round(seg.flood_probability * 100),
        risk: seg.risk_level || "--",
        elevation: seg.elevation,
        slope: seg.slope,
        drain: seg.distance_to_drain_m,
        pipe: seg.pipe_diameter_mm,
        hist: seg.historical_waterlogging,
      },
      geometry: { type: "Point", coordinates: [seg.lng, seg.lat] },
    });
  });
  return { type: "FeatureCollection", features };
}

function nowcastCircleFeatures(locations, riskColorFor) {
  return {
    type: "FeatureCollection",
    features: (locations || []).map((loc) => ({
      type: "Feature",
      properties: {
        color: riskColorFor ? riskColorFor(loc.risk) : UNKNOWN_COLOR,
        name: loc.name,
        depth: loc.depth,
        probability: loc.probability,
        elevation: loc.elevation,
      },
      geometry: { type: "Point", coordinates: [loc.lng, loc.lat] },
    })),
  };
}

function popupHTML(p) {
  const row = (label, value) =>
    `<div><span>${label}</span><strong>${value}</strong></div>`;
  const na = (v, unit = "") => (v == null ? "no data" : `${v}${unit}`);
  return (
    `<div class="pop"><b>Segment ${p.index}</b>` +
    row("Flood probability", `${p.prob}%`) +
    row("Risk", p.risk) +
    row("Elevation", na(p.elevation, " m")) +
    row("Slope", na(p.slope, "%")) +
    row("Nearest drain", na(p.drain, " m")) +
    row("Pipe diameter", na(p.pipe, " mm")) +
    row(
      "Historical waterlogging",
      p.hist == null ? "no data" : p.hist ? "yes" : "no",
    ) +
    `</div>`
  );
}

export default function FloodRiskMap({ safeRoute, fallbackLocations, riskColorFor }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const readyRef = useRef(false);

  // Create the map once.
  useEffect(() => {
    if (!HAS_TOKEN || mapRef.current || !containerRef.current) return;
    mapboxgl.accessToken = MAPBOX_TOKEN;
    const map = new mapboxgl.Map({
      container: containerRef.current,
      style: "mapbox://styles/mapbox/streets-v12",
      center: KOLKATA,
      zoom: 11,
    });
    map.addControl(new mapboxgl.NavigationControl({ showCompass: false }), "top-right");
    map.on("load", () => {
      readyRef.current = true;
      // sources
      map.addSource("route-lines", { type: "geojson", data: emptyFC() });
      map.addSource("route-points", { type: "geojson", data: emptyFC() });
      map.addSource("nowcast", { type: "geojson", data: emptyFC() });
      // nowcast risk circles (shown before a route)
      map.addLayer({
        id: "nowcast-circles", type: "circle", source: "nowcast",
        paint: {
          "circle-radius": 10,
          "circle-color": ["get", "color"],
          "circle-opacity": 0.25,
          "circle-stroke-color": ["get", "color"],
          "circle-stroke-width": 1.5,
        },
      });
      // route segment lines, coloured per segment
      map.addLayer({
        id: "route-lines", type: "line", source: "route-lines",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": ["get", "color"],
          "line-width": ["case", ["get", "recommended"], 6, 3],
          "line-opacity": ["case", ["get", "recommended"], 0.95, 0.4],
        },
      });
      // clickable sample points on the recommended route
      map.addLayer({
        id: "route-points", type: "circle", source: "route-points",
        paint: {
          "circle-radius": 5,
          "circle-color": ["get", "color"],
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 1,
        },
      });
      map.on("click", "route-points", (e) => {
        const f = e.features[0];
        new mapboxgl.Popup({ closeButton: true })
          .setLngLat(f.geometry.coordinates)
          .setHTML(popupHTML(f.properties))
          .addTo(map);
      });
      map.on("mouseenter", "route-points", () => (map.getCanvas().style.cursor = "pointer"));
      map.on("mouseleave", "route-points", () => (map.getCanvas().style.cursor = ""));
      // paint whatever data is already in props
      syncData();
    });
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
      readyRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push data whenever props change.
  useEffect(() => {
    syncData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [safeRoute, fallbackLocations]);

  function syncData() {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;
    const routes = safeRoute?.routes || null;
    if (routes && routes.length) {
      const rec = safeRoute.recommended_index ?? 0;
      map.getSource("route-lines").setData(routeSegmentFeatures(routes, rec));
      map.getSource("route-points").setData(routePointFeatures(routes, rec));
      map.getSource("nowcast").setData(emptyFC());
      // fit to the route
      const b = new mapboxgl.LngLatBounds();
      routes.forEach((r) => (r.geometry?.coordinates || []).forEach((c) => b.extend(c)));
      if (!b.isEmpty()) map.fitBounds(b, { padding: 48, duration: 600 });
    } else {
      map.getSource("route-lines").setData(emptyFC());
      map.getSource("route-points").setData(emptyFC());
      map.getSource("nowcast").setData(nowcastCircleFeatures(fallbackLocations, riskColorFor));
    }
  }

  if (!HAS_TOKEN) {
    return (
      <div className="mapframe mapframe--empty">
        <div className="map-token-msg">
          <strong>Map needs a Mapbox token.</strong>
          <span>
            Add a public token (pk.) as <code>VITE_MAPBOX_TOKEN</code> in
            {" "}<code>frotend/.env.local</code> to render the flood-risk map.
          </span>
        </div>
      </div>
    );
  }

  return <div className="mapframe" ref={containerRef} />;
}

function emptyFC() {
  return { type: "FeatureCollection", features: [] };
}

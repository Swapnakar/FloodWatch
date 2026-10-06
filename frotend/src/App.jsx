import { useState, useEffect, useCallback, useRef } from "react";
import Map, { Source, Layer, Marker, Popup as MapboxPopup } from "react-map-gl/mapbox";
import "mapbox-gl/dist/mapbox-gl.css";
import "./App.css";
import SafeRouteAdvisor from "./components/SafeRouteAdvisor";
import WeatherCard from "./components/WeatherCard";
import LocationRisk from "./components/LocationRisk";
import DataStatus from "./components/DataStatus";
import RiskLegend from "./components/RiskLegend";
import Sidebar from "./components/Sidebar";
import SimulationPanel, { ALERT_COLORS } from "./components/SimulationPanel";
import {
  locationRisk, reverseGeocode, getDrainage, getManholes, simulatePoints, simulateDrainage, ApiError,
} from "./services/api.js";

const MANHOLE_COLORS = { junction: "#7c3aed", joint: "#475569", terminal: "#94a3b8" };

// Haversine distance in metres.
function distanceM(lat1, lng1, lat2, lng2) {
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return 2 * 6371000 * Math.asin(Math.sqrt(a));
}


const RISK_COLORS = {
  CRITICAL: "#d62828",
  HIGH: "#f97316",
  ELEVATED: "#f97316",
  MODERATE: "#eab308",
  LOW: "#22c55e",
};

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;

// ── Radius prediction grid ──
const GRID_RADIUS_M = 1000;   // circle radius around the user
const GRID_SPACING_M = 200;   // distance between neighbouring dots
const GRID_CONCURRENCY = 8;   // parallel /api/location-risk requests
const M_PER_DEG_LAT = 111320;

// Hexagonal grid of points clipped to a circle (denser, more even than a square grid).
function circlePoints(centerLat, centerLng, radiusM, spacingM) {
  const mPerDegLng = M_PER_DEG_LAT * Math.cos((centerLat * Math.PI) / 180);
  const rowStep = spacingM * Math.sqrt(3) / 2;
  const rows = Math.floor(radiusM / rowStep);
  const points = [];
  for (let r = -rows; r <= rows; r++) {
    const dy = r * rowStep;
    const xOffset = (Math.abs(r) % 2) * (spacingM / 2);
    const cols = Math.ceil(radiusM / spacingM) + 1;
    for (let c = -cols; c <= cols; c++) {
      const dx = c * spacingM + xOffset;
      if (dx * dx + dy * dy > radiusM * radiusM) continue;
      points.push({
        lat: centerLat + dy / M_PER_DEG_LAT,
        lng: centerLng + dx / mPerDegLng,
      });
    }
  }
  return points;
}

// GeoJSON polygon approximating the radius circle, for drawing the outline.
function circlePolygon(centerLat, centerLng, radiusM, steps = 64) {
  const mPerDegLng = M_PER_DEG_LAT * Math.cos((centerLat * Math.PI) / 180);
  const ring = [];
  for (let i = 0; i <= steps; i++) {
    const a = (i / steps) * 2 * Math.PI;
    ring.push([
      centerLng + (radiusM * Math.cos(a)) / mPerDegLng,
      centerLat + (radiusM * Math.sin(a)) / M_PER_DEG_LAT,
    ]);
  }
  return { type: "Feature", geometry: { type: "Polygon", coordinates: [ring] }, properties: {} };
}

function App() {
  // ── Location & Dynamic Grid ──
  const [currentLocation, setCurrentLocation] = useState(null);
  const [locationName, setLocationName] = useState(null);
  const [dynamicLocations, setDynamicLocations] = useState([]);
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [drainageData, setDrainageData] = useState(null);
  const [manholeData, setManholeData] = useState(null);
  const [selectedManhole, setSelectedManhole] = useState(null);
  const drainageMapRef = useRef(null);
  const [activeTab, setActiveTab] = useState('prediction');

  // ── Route state (only for route tab) ──
  const [routeResult, setRouteResult] = useState(null);
  const [selectedRouteIndex, setSelectedRouteIndex] = useState(0);
  const [selectedSegment, setSelectedSegment] = useState(null);
  const routeMapRef = useRef(null);

  // ── Simulation state (only for simulation tab) ──
  const [isSimulating, setIsSimulating] = useState(false);
  const [simLocations, setSimLocations] = useState([]);
  const [selectedSimLocation, setSelectedSimLocation] = useState(null);
  const [simMode, setSimMode] = useState("flood");          // "flood" | "drainage"
  const [drainageSim, setDrainageSim] = useState(null);     // /api/simulate/drainage result
  const [selectedSimPipe, setSelectedSimPipe] = useState(null);
  const [simError, setSimError] = useState(null);

  // ── Fetch Drainage Network ──
  useEffect(() => {
    getDrainage().then(res => setDrainageData(res.data)).catch(console.error);
    getManholes().then(res => setManholeData(res.data)).catch(console.error);
  }, []);

  // ── Auto-fit bounds when route changes ──
  useEffect(() => {
    if (activeTab !== 'route') return;
    if (routeResult?.routes?.[selectedRouteIndex]?.geometry?.coordinates?.length > 0) {
      const coords = routeResult.routes[selectedRouteIndex].geometry.coordinates;
      let minLng = 180, maxLng = -180, minLat = 90, maxLat = -90;
      coords.forEach(([lng, lat]) => {
        if (lng < minLng) minLng = lng;
        if (lng > maxLng) maxLng = lng;
        if (lat < minLat) minLat = lat;
        if (lat > maxLat) maxLat = lat;
      });
      routeMapRef.current?.fitBounds(
        [[minLng, minLat], [maxLng, maxLat]],
        { padding: 60, duration: 1000 }
      );
    }
  }, [routeResult, selectedRouteIndex, activeTab]);

  // ── Auto-Detect Location & Generate Grid ──
  useEffect(() => {
    if (!("geolocation" in navigator)) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        setCurrentLocation({ lat: latitude, lng: longitude });
        
        // Reverse Geocode for Navbar
        reverseGeocode(latitude, longitude).then((res) => {
          const geoData = res.data;
          // Our backend returns Mapbox-style { candidates: [{name, context: {locality, place, region}}] }
          if (geoData?.candidates?.[0]) {
            const c = geoData.candidates[0];
            const ctx = c.context || {};
            // Build a human-readable name: "Locality, City" or "Neighborhood, City"
            const parts = [ctx.locality || ctx.neighborhood || c.name, ctx.place || ctx.region].filter(Boolean);
            setLocationName(parts.length > 0 ? parts.join(", ") : c.name);
          } else if (geoData?.display_name) {
            setLocationName(geoData.display_name.split(",").slice(0, 2).join(",").trim());
          } else if (geoData?.features?.[0]?.place_name) {
            setLocationName(geoData.features[0].place_name.split(",").slice(0, 2).join(",").trim());
          } else {
            setLocationName(`${latitude.toFixed(4)}, ${longitude.toFixed(4)}`);
          }
        }).catch(() => setLocationName(`${latitude.toFixed(4)}, ${longitude.toFixed(4)}`));

      },
      (err) => console.error("Geolocation error:", err),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
    );
  }, []);

  const generateRadiusGrid = async (centerLat, centerLng, simRainfall = null, simDuration = null) => {
    const horizon = simDuration || 60;
    const points = circlePoints(centerLat, centerLng, GRID_RADIUS_M, GRID_SPACING_M);

    const fetchPoint = ({ lat: plat, lng: plng }) =>
      locationRisk(plat, plng, horizon, simRainfall).then(res => {
        const data = res.data;
        return {
          name: `Lat ${plat.toFixed(4)}, Lng ${plng.toFixed(4)}`,
          lat: plat,
          lng: plng,
          depth: data.water_depth_cm || 0,
          risk: data.risk_level || "LOW",
          probability: data.flood_probability ? Math.round(data.flood_probability * 100) : 0,
          color: RISK_COLORS[data.risk_level] || "#22c55e",
        };
      }).catch(() => null);

    try {
      // Limit concurrency so ~80 points don't flood the backend at once.
      const results = new Array(points.length);
      let next = 0;
      const worker = async () => {
        while (next < points.length) {
          const i = next++;
          results[i] = await fetchPoint(points[i]);
        }
      };
      await Promise.all(Array.from({ length: GRID_CONCURRENCY }, worker));
      return results.filter(r => r !== null);
    } catch {
      return [];
    }
  };

  const radiusCircle = currentLocation
    ? circlePolygon(currentLocation.lat, currentLocation.lng, GRID_RADIUS_M)
    : null;

  // Manholes closest to the user, nearest first.
  const nearbyManholes = (() => {
    if (!currentLocation || !manholeData?.features) return [];
    return manholeData.features
      .map((f) => {
        const [lng, lat] = f.geometry.coordinates;
        return { ...f.properties, lat, lng,
          distance: distanceM(currentLocation.lat, currentLocation.lng, lat, lng) };
      })
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 5);
  })();

  const locateManhole = (mh) => {
    setSelectedManhole(mh);
    drainageMapRef.current?.flyTo({ center: [mh.lng, mh.lat], zoom: 18, duration: 1200 });
  };

  const handleDrainageMapClick = (e) => {
    const f = e.features?.[0];
    if (!f) { setSelectedManhole(null); return; }
    const [lng, lat] = f.geometry.coordinates;
    const distance = currentLocation
      ? distanceM(currentLocation.lat, currentLocation.lng, lat, lng) : null;
    setSelectedManhole({ ...f.properties, lat, lng, distance });
  };

  // Prediction grid runs only when the user clicks Predict (no auto-predict).
  const [isGridLoading, setIsGridLoading] = useState(false);
  const handlePredictGrid = async (horizonMinutes) => {
    if (!currentLocation) return;
    setIsGridLoading(true);
    setSelectedLocation(null);
    try {
      setDynamicLocations(
        await generateRadiusGrid(currentLocation.lat, currentLocation.lng, null, horizonMinutes),
      );
    } finally {
      setIsGridLoading(false);
    }
  };

  // Scenario flood risk for the whole radius grid in ONE request.
  const runFloodSimulation = async ({ rainfall, duration }) => {
    const points = circlePoints(currentLocation.lat, currentLocation.lng, GRID_RADIUS_M, GRID_SPACING_M)
      .map((p) => ({ lat: p.lat, lon: p.lng }));
    const { data } = await simulatePoints(points, rainfall, duration);
    return data.points
      .filter((p) => p.risk_level)
      .map((p) => ({
        name: `Lat ${p.lat.toFixed(4)}, Lng ${p.lon.toFixed(4)}`,
        lat: p.lat,
        lng: p.lon,
        risk: p.risk_level,
        probability: Math.round((p.flood_probability || 0) * 100),
        drainLoad: p.drain_load_pct,
        drainAlert: p.drain_alert,
        color: RISK_COLORS[p.risk_level] || "#22c55e",
      }));
  };

  const handleSimulate = async (params) => {
    if (!currentLocation) return;
    setSimError(null);
    setSelectedSimLocation(null);
    setSelectedSimPipe(null);
    if (!params) {
      // Reset: back to live data / clear the drainage scenario.
      setSimLocations([]);
      setDrainageSim(null);
      return;
    }
    setIsSimulating(true);
    try {
      if (simMode === "drainage") {
        const { data } = await simulateDrainage(
          params.rainfall, params.duration, currentLocation.lat, currentLocation.lng, GRID_RADIUS_M,
        );
        setDrainageSim(data);
      } else {
        setSimLocations(await runFloodSimulation(params));
      }
    } catch (err) {
      const detail = err instanceof ApiError ? err.detail : null;
      setSimError(typeof detail === "string" ? detail : "Simulation failed. Is the backend running?");
    } finally {
      setIsSimulating(false);
    }
  };

  const handleSimMapClick = (e) => {
    const f = e.features?.[0];
    if (!f) { setSelectedSimPipe(null); return; }
    setSelectedSimPipe({ ...f.properties, lng: e.lngLat.lng, lat: e.lngLat.lat });
  };


  return (
    <div className="app-layout">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <div className="app-content-area">
        {/* ================= HEADER ================= */}
        <header className="header">
          <div className="header-left">
            <h1>
              FLOOD<span>EXA</span>
            </h1>
          </div>
          <div className="header-right">
            <span className="detected-location">
              📍 {locationName || "Detecting location..."}
            </span>
          </div>
        </header>

        <main className="app-main">
          {/* ================= WEATHER ================= */}
          <section className="dashboard-top-row">
            <div className="panel weather-panel" style={{ padding: "16px 24px", minHeight: "auto" }}>
              <WeatherCard />
            </div>
          </section>

          {/* ================= TAB CONTENT ================= */}

          {/* ── PREDICTION TAB ── */}
          {activeTab === 'prediction' && (
            <section className="tab-content">
              <div className="tab-split-layout">
                <div className="tab-panel-col">
                  <div className="panel">
                    {currentLocation && (
                      <LocationRisk lat={currentLocation.lat} lng={currentLocation.lng} onPredict={handlePredictGrid} />
                    )}
                  </div>
                </div>
                <div className="tab-map-col">
                  <div className="panel tab-map-wrapper">
                    <div className="map-header-content">
                      <h4 className="panel-title">Flood Risk Map</h4>
                      <p className="panel-desc">
                        {isGridLoading
                          ? "Predicting flood risk around you…"
                          : dynamicLocations.length > 0
                            ? "Dynamic risk assessment around your location (1km radius)"
                            : "Click Predict to assess flood risk around your location (1km radius)"}
                      </p>
                    </div>
                    {currentLocation && (
                      <div className="tab-map-container">
                        <Map
                          key="prediction-map"
                          initialViewState={{
                            longitude: currentLocation.lng,
                            latitude: currentLocation.lat,
                            zoom: 14
                          }}
                          style={{ width: "100%", height: "100%", borderRadius: "12px" }}
                          mapStyle="mapbox://styles/mapbox/streets-v12"
                          mapboxAccessToken={MAPBOX_TOKEN}
                        >
                          {/* Current location pin */}
                          <Marker longitude={currentLocation.lng} latitude={currentLocation.lat} anchor="center">
                            <div style={{
                              width: '18px', height: '18px',
                              borderRadius: '50%', background: '#0066cc',
                              border: '3px solid #fff', boxShadow: '0 0 12px rgba(0,102,204,0.5)',
                              position: 'relative'
                            }}>
                              <div style={{
                                position: 'absolute', top: '-6px', left: '-6px',
                                width: '30px', height: '30px',
                                borderRadius: '50%', border: '2px solid rgba(0,102,204,0.3)',
                                animation: 'pulse-ring 2s ease-out infinite'
                              }} />
                            </div>
                          </Marker>

                          {/* Prediction radius */}
                          {radiusCircle && (
                            <Source id="radius-pred" type="geojson" data={radiusCircle}>
                              <Layer id="radius-pred-fill" type="fill" paint={{ "fill-color": "#0066cc", "fill-opacity": 0.06 }} />
                              <Layer id="radius-pred-line" type="line" paint={{ "line-color": "#0066cc", "line-width": 2, "line-dasharray": [2, 2], "line-opacity": 0.7 }} />
                            </Source>
                          )}

                          {/* Risk markers */}
                          {dynamicLocations.map((location) => (
                            <Marker key={location.name} longitude={location.lng} latitude={location.lat} anchor="center">
                              <div
                                style={{
                                  width: '14px', height: '14px', borderRadius: '50%',
                                  backgroundColor: location.color, opacity: 0.8,
                                  border: `2px solid #fff`, cursor: 'pointer'
                                }}
                                onClick={(e) => { e.stopPropagation(); setSelectedLocation(location); }}
                              />
                            </Marker>
                          ))}

                          {selectedLocation && (
                            <MapboxPopup
                              longitude={selectedLocation.lng} latitude={selectedLocation.lat}
                              anchor="bottom" onClose={() => setSelectedLocation(null)} closeOnClick={false}
                            >
                              <div style={{color: '#17171c', fontSize: '13px'}}>
                                <strong>{selectedLocation.name}</strong><br />
                                Water depth: {selectedLocation.depth} cm<br />
                                Risk: <strong style={{color: selectedLocation.color}}>{selectedLocation.risk}</strong><br />
                                Probability: {selectedLocation.probability}%
                              </div>
                            </MapboxPopup>
                          )}

                          {/* Drainage Layer */}
                          {drainageData && (
                            <Source id="drainage-pred" type="geojson" data={drainageData}>
                              <Layer id="drainage-pred-line" type="line" paint={{
                                "line-color": "#0ea5e9", "line-width": 1.5, "line-opacity": 0.4
                              }} />
                            </Source>
                          )}
                        </Map>
                        <RiskLegend />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* ── SAFE ROUTE TAB ── */}
          {activeTab === 'route' && (
            <section className="tab-content">
              <div className="tab-split-layout">
                <div className="tab-panel-col">
                  <div className="panel">
                    <SafeRouteAdvisor 
                      horizon={60} 
                      onResult={(res) => { setRouteResult(res); setSelectedRouteIndex(0); setSelectedSegment(null); }} 
                      selectedRouteIndex={selectedRouteIndex}
                      onSelectRoute={setSelectedRouteIndex}
                    />
                  </div>
                </div>
                <div className="tab-map-col">
                  <div className="panel tab-map-wrapper">
                    <div className="map-header-content">
                      <h4 className="panel-title">Route Map</h4>
                      <p className="panel-desc">Visualize route flood risk in real-time</p>
                    </div>
                    <div className="tab-map-container">
                      <Map
                        ref={routeMapRef}
                        key="route-map"
                        initialViewState={{
                          longitude: currentLocation?.lng || 88.3639,
                          latitude: currentLocation?.lat || 22.5726,
                          zoom: 13
                        }}
                        style={{ width: "100%", height: "100%", borderRadius: "12px" }}
                        mapStyle="mapbox://styles/mapbox/streets-v12"
                        mapboxAccessToken={MAPBOX_TOKEN}
                      >
                        {/* Route lines */}
                        {routeResult?.routes?.map((route, idx) => {
                          const isSelected = idx === selectedRouteIndex;
                          return (
                            <Source key={`route-${idx}`} id={`route-${idx}`} type="geojson" data={route.geometry}>
                              <Layer
                                id={`route-line-${idx}`}
                                type="line"
                                layout={{ "line-join": "round", "line-cap": "round" }}
                                paint={{
                                  "line-color": isSelected ? "#0066cc" : "#d9d9dd",
                                  "line-width": isSelected ? 5 : 3,
                                  "line-opacity": isSelected ? 1 : 0.5
                                }}
                              />
                            </Source>
                          );
                        })}

                        {/* Segment risk markers */}
                        {routeResult?.routes?.[selectedRouteIndex]?.segments?.map((segment, idx) => (
                          <Marker key={`seg-${idx}`} longitude={segment.lng} latitude={segment.lat} anchor="center">
                            <div
                              style={{
                                width: '12px', height: '12px', borderRadius: '50%',
                                backgroundColor: RISK_COLORS[segment.risk_level] || "#94a3b8",
                                border: `1px solid #fff`, cursor: 'pointer',
                                boxShadow: '0 0 4px rgba(0,0,0,0.3)'
                              }}
                              onClick={(e) => { e.stopPropagation(); setSelectedSegment(segment); }}
                            />
                          </Marker>
                        ))}
                        
                        {/* Origin/Destination markers */}
                        {routeResult?.routes?.[selectedRouteIndex]?.geometry?.coordinates && (
                          <>
                            <Marker
                              longitude={routeResult.routes[selectedRouteIndex].geometry.coordinates[0][0]}
                              latitude={routeResult.routes[selectedRouteIndex].geometry.coordinates[0][1]}
                              anchor="bottom"
                            >
                              <div style={{ background: '#22c55e', color: 'white', padding: '2px 8px', borderRadius: '6px', fontSize: '11px', fontWeight: 'bold' }}>START</div>
                            </Marker>
                            <Marker
                              longitude={routeResult.routes[selectedRouteIndex].geometry.coordinates.at(-1)[0]}
                              latitude={routeResult.routes[selectedRouteIndex].geometry.coordinates.at(-1)[1]}
                              anchor="bottom"
                            >
                              <div style={{ background: '#d62828', color: 'white', padding: '2px 8px', borderRadius: '6px', fontSize: '11px', fontWeight: 'bold' }}>END</div>
                            </Marker>
                          </>
                        )}

                        {selectedSegment && (
                          <MapboxPopup
                            longitude={selectedSegment.lng} latitude={selectedSegment.lat}
                            anchor="bottom" onClose={() => setSelectedSegment(null)} closeOnClick={false}
                          >
                            <div style={{color: '#17171c', fontSize: '12px', lineHeight: '1.4'}}>
                              <strong>Flood Risk Segment</strong><br />
                              Probability: {selectedSegment.flood_probability != null ? `${(selectedSegment.flood_probability * 100).toFixed(0)}%` : "N/A"}<br />
                              Risk: <strong>{selectedSegment.risk_level}</strong><br />
                              Elevation: {selectedSegment.elevation != null ? `${selectedSegment.elevation} m` : "—"}<br />
                              Slope: {selectedSegment.slope != null ? `${selectedSegment.slope}%` : "—"}<br />
                              Drain dist: {selectedSegment.distance_to_drain_m != null ? `${selectedSegment.distance_to_drain_m} m` : "—"}
                            </div>
                          </MapboxPopup>
                        )}

                        {/* Show drainage on route map too */}
                        {drainageData && (
                          <Source id="drainage-route" type="geojson" data={drainageData}>
                            <Layer id="drainage-route-line" type="line" paint={{
                              "line-color": "#0ea5e9", "line-width": 1.5, "line-opacity": 0.3
                            }} />
                          </Source>
                        )}
                      </Map>

                      {!routeResult && (
                        <div className="map-placeholder-msg">
                          Enter start and destination to see the route on the map
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* ── SIMULATION TAB ── */}
          {activeTab === 'simulation' && (
            <section className="tab-content">
              <div className="tab-split-layout">
                <div className="tab-panel-col">
                  <SimulationPanel
                    onSimulate={handleSimulate}
                    isLoading={isSimulating}
                    mode={simMode}
                    onModeChange={(m) => { setSimMode(m); setSimError(null); setSelectedSimLocation(null); setSelectedSimPipe(null); }}
                    drainageResult={drainageSim}
                    error={simError}
                  />
                </div>
                <div className="tab-map-col">
                  <div className="panel tab-map-wrapper">
                    <div className="map-header-content">
                      <h4 className="panel-title">
                        {simMode === "drainage" ? "Drainage Network Simulation" : "Simulation Map"}
                      </h4>
                      <p className="panel-desc">
                        {simMode === "drainage"
                          ? (drainageSim
                              ? `Pipe load for ${drainageSim.scenario.rainfall_mm_hr} mm/hr over ${drainageSim.scenario.duration_min} min. Click a pipe for details`
                              : "Pick a storm and run the simulation to see pipe alerts")
                          : (simLocations.length > 0
                              ? "Simulated flood risk based on custom rainfall parameters"
                              : "Showing live risk. Run a simulation to see predicted flood impact")}
                      </p>
                    </div>
                    {currentLocation && (
                      <div className="tab-map-container">
                        <Map
                          key="simulation-map"
                          interactiveLayerIds={simMode === "drainage" && drainageSim ? ["drainsim-line"] : []}
                          onClick={simMode === "drainage" ? handleSimMapClick : undefined}
                          initialViewState={{
                            longitude: currentLocation.lng,
                            latitude: currentLocation.lat,
                            zoom: 14
                          }}
                          style={{ width: "100%", height: "100%", borderRadius: "12px" }}
                          mapStyle="mapbox://styles/mapbox/streets-v12"
                          mapboxAccessToken={MAPBOX_TOKEN}
                        >
                          {/* Current location pin */}
                          <Marker longitude={currentLocation.lng} latitude={currentLocation.lat} anchor="center">
                            <div style={{
                              width: '18px', height: '18px',
                              borderRadius: '50%', background: '#0066cc',
                              border: '3px solid #fff', boxShadow: '0 0 12px rgba(0,102,204,0.5)'
                            }} />
                          </Marker>

                          {/* Simulation radius */}
                          {radiusCircle && (
                            <Source id="radius-sim" type="geojson" data={radiusCircle}>
                              <Layer id="radius-sim-fill" type="fill" paint={{ "fill-color": "#0066cc", "fill-opacity": 0.06 }} />
                              <Layer id="radius-sim-line" type="line" paint={{ "line-color": "#0066cc", "line-width": 2, "line-dasharray": [2, 2], "line-opacity": 0.7 }} />
                            </Source>
                          )}

                          {/* Drainage simulation: pipes coloured by alert */}
                          {simMode === "drainage" && drainageSim && (
                            <Source id="drainsim" type="geojson" data={drainageSim.network}>
                              <Layer id="drainsim-line" type="line"
                                layout={{ "line-cap": "round", "line-join": "round", "line-sort-key": ["match", ["get", "alert"], "RED", 4, "ORANGE", 3, "YELLOW", 2, 1] }}
                                paint={{
                                  "line-color": ["match", ["get", "alert"],
                                    "RED", ALERT_COLORS.RED,
                                    "ORANGE", ALERT_COLORS.ORANGE,
                                    "YELLOW", ALERT_COLORS.YELLOW,
                                    "GREEN", ALERT_COLORS.GREEN,
                                    ALERT_COLORS.UNKNOWN],
                                  "line-width": ["interpolate", ["linear"], ["zoom"], 12, 2, 16, 5],
                                  "line-opacity": ["case", ["get", "in_area"], 0.95, 0.45],
                                }} />
                            </Source>
                          )}

                          {selectedSimPipe && (
                            <MapboxPopup
                              longitude={selectedSimPipe.lng} latitude={selectedSimPipe.lat}
                              anchor="bottom" onClose={() => setSelectedSimPipe(null)} closeOnClick={false}
                            >
                              <div style={{ color: '#17171c', fontSize: '13px', lineHeight: 1.5 }}>
                                <strong style={{ color: ALERT_COLORS[selectedSimPipe.alert] }}>{selectedSimPipe.alert} ALERT</strong><br />
                                {drainageSim?.alert_meaning?.[selectedSimPipe.alert]}<br />
                                Pipe: {selectedSimPipe.segment_id}{selectedSimPipe.pipe_diameter_mm ? ` (${selectedSimPipe.pipe_diameter_mm} mm)` : ""}<br />
                                {selectedSimPipe.load_pct != null && (<>Load: {selectedSimPipe.load_pct}% of capacity<br /></>)}
                                Inflow: {selectedSimPipe.inflow_m3s} m³/s
                                {selectedSimPipe.capacity_m3s ? ` / capacity ${selectedSimPipe.capacity_m3s} m³/s` : ""}<br />
                                {selectedSimPipe.overflow_m3 > 0 && (<>Overflow: {selectedSimPipe.overflow_m3} m³<br /></>)}
                                Ward: {selectedSimPipe.ward}
                              </div>
                            </MapboxPopup>
                          )}

                          {/* Simulation risk markers */}
                          {simMode === "flood" && (simLocations.length > 0 ? simLocations : dynamicLocations).map((location) => (
                            <Marker key={location.name} longitude={location.lng} latitude={location.lat} anchor="center">
                              <div
                                style={{
                                  width: '14px', height: '14px', borderRadius: '50%',
                                  backgroundColor: location.color, opacity: 0.8,
                                  border: `2px solid #fff`, cursor: 'pointer'
                                }}
                                onClick={(e) => { e.stopPropagation(); setSelectedSimLocation(location); }}
                              />
                            </Marker>
                          ))}

                          {selectedSimLocation && (
                            <MapboxPopup
                              longitude={selectedSimLocation.lng} latitude={selectedSimLocation.lat}
                              anchor="bottom" onClose={() => setSelectedSimLocation(null)} closeOnClick={false}
                            >
                              <div style={{color: '#17171c', fontSize: '13px'}}>
                                <strong>{selectedSimLocation.name}</strong><br />
                                Risk: <strong style={{color: selectedSimLocation.color}}>{selectedSimLocation.risk}</strong><br />
                                Probability: {selectedSimLocation.probability}%
                                {selectedSimLocation.drainLoad != null && (
                                  <><br />Nearest drain load: <strong style={{ color: ALERT_COLORS[selectedSimLocation.drainAlert] }}>
                                    {selectedSimLocation.drainLoad}%</strong></>
                                )}
                              </div>
                            </MapboxPopup>
                          )}

                          {/* Drainage Layer (plain, flood mode only) */}
                          {simMode === "flood" && drainageData && (
                            <Source id="drainage-sim" type="geojson" data={drainageData}>
                              <Layer id="drainage-sim-line" type="line" paint={{
                                "line-color": "#0ea5e9", "line-width": 2, "line-opacity": 0.5
                              }} />
                            </Source>
                          )}
                        </Map>
                        {simMode === "drainage" ? (
                          <div className="risk-legend" style={{ background: "var(--canvas)", padding: "16px", borderRadius: "12px", boxShadow: "0 2px 12px rgba(0,0,0,0.05)", border: "1px solid var(--hairline)" }}>
                            <h4 style={{ margin: "0 0 12px", fontSize: "14px", color: "var(--slate)" }}>DRAIN ALERT</h4>
                            <div style={{ display: "flex", flexWrap: "wrap", gap: "12px", fontSize: "13px" }}>
                              {[["GREEN", "< 75%"], ["YELLOW", "75–100%"], ["ORANGE", "100–150%"], ["RED", "> 150%"]].map(([lvl, range]) => (
                                <div key={lvl} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                  <span style={{ display: "block", width: "18px", height: "4px", borderRadius: "2px", background: ALERT_COLORS[lvl] }} />
                                  <span>{lvl.charAt(0) + lvl.slice(1).toLowerCase()} ({range} load)</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        ) : (
                          <RiskLegend />
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* ── DRAINAGE TAB ── */}
          {activeTab === 'drainage' && (
            <section className="tab-content">
              <div className="tab-split-layout">
                <div className="tab-panel-col">
                  <div className="panel drainage-info-panel">
                    <h3 className="panel-title" style={{ color: "var(--primary)", marginBottom: "16px" }}>Drainage Network</h3>
                    <p className="panel-desc">KMC's mapped drainage and canal network around your current location. Drainage proximity is a key input for the flood prediction model.</p>
                    
                    <div className="drainage-stats">
                      <div className="drainage-stat-card">
                        <span className="drainage-stat-label">Total Drains</span>
                        <span className="drainage-stat-value">
                          {drainageData?.features?.length || "—"}
                        </span>
                      </div>
                      <div className="drainage-stat-card">
                        <span className="drainage-stat-label">Network Type</span>
                        <span className="drainage-stat-value">KMC Canal</span>
                      </div>
                      <div className="drainage-stat-card">
                        <span className="drainage-stat-label">Data Status</span>
                        <span className="drainage-stat-value" style={{ color: drainageData ? "#22c55e" : "#d62828" }}>
                          {drainageData ? "Loaded" : "Unavailable"}
                        </span>
                      </div>
                      <div className="drainage-stat-card">
                        <span className="drainage-stat-label">Manholes</span>
                        <span className="drainage-stat-value">
                          {manholeData?.features?.length || "—"}
                        </span>
                      </div>
                    </div>

                    {nearbyManholes.length > 0 && (
                      <div style={{ marginTop: "20px" }}>
                        <strong style={{ color: "var(--ink)", fontSize: "14px" }}>Nearest manholes</strong>
                        <ul style={{ listStyle: "none", margin: "8px 0 0 0", padding: 0 }}>
                          {nearbyManholes.map((mh) => (
                            <li key={mh.manhole_id}>
                              <button
                                type="button"
                                onClick={() => locateManhole(mh)}
                                aria-label={`Show ${mh.manhole_id} on map`}
                                style={{
                                  width: "100%", display: "flex", justifyContent: "space-between", alignItems: "center",
                                  gap: "8px", padding: "8px 10px", marginBottom: "6px", cursor: "pointer",
                                  background: selectedManhole?.manhole_id === mh.manhole_id ? "var(--canvas-parchment)" : "transparent",
                                  border: "1px solid var(--hairline)", borderRadius: "8px", fontSize: "13px", color: "var(--ink)",
                                  textAlign: "left",
                                }}
                              >
                                <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                  <span style={{ width: 10, height: 10, borderRadius: "50%", background: MANHOLE_COLORS[mh.kind] }} />
                                  {mh.manhole_id} · {mh.kind}
                                </span>
                                <span style={{ color: "var(--ink-muted-48)" }}>{Math.round(mh.distance)} m</span>
                              </button>
                            </li>
                          ))}
                        </ul>
                        <p style={{ fontSize: "12px", color: "var(--ink-muted-48)", margin: "6px 0 0 0" }}>
                          Manhole positions are inferred from pipe junctions and ends in the KMC sewer maps, not surveyed.
                        </p>
                      </div>
                    )}

                    <div style={{ marginTop: "20px", padding: "16px", background: "var(--canvas-parchment)", borderRadius: "12px", fontSize: "13px", color: "var(--ink-muted-48)" }}>
                      <strong style={{ color: "var(--ink)" }}>How drainage affects flood risk:</strong>
                      <ul style={{ margin: "8px 0 0 0", paddingLeft: "16px", lineHeight: "1.8" }}>
                        <li>Points closer to drains have better water removal capacity</li>
                        <li>Pipe diameter and drain capacity factor into risk calculations</li>
                        <li>Historical waterlogging correlates with poor drainage areas</li>
                        <li>The model uses distance-to-drain as a key prediction feature</li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div className="tab-map-col">
                  <div className="panel tab-map-wrapper">
                    <div className="map-header-content">
                      <h4 className="panel-title">Drainage Network Map</h4>
                      <p className="panel-desc">KMC drainage pipes and manholes. Click a manhole to see its location</p>
                    </div>
                    {currentLocation && (
                      <div className="tab-map-container">
                        <Map
                          key="drainage-map"
                          ref={drainageMapRef}
                          interactiveLayerIds={manholeData ? ["manhole-points"] : []}
                          onClick={handleDrainageMapClick}
                          cursor="auto"
                          initialViewState={{
                            longitude: currentLocation.lng,
                            latitude: currentLocation.lat,
                            zoom: 14
                          }}
                          style={{ width: "100%", height: "100%", borderRadius: "12px" }}
                          mapStyle="mapbox://styles/mapbox/light-v11"
                          mapboxAccessToken={MAPBOX_TOKEN}
                        >
                          {/* Current location pin */}
                          <Marker longitude={currentLocation.lng} latitude={currentLocation.lat} anchor="center">
                            <div style={{
                              width: '18px', height: '18px',
                              borderRadius: '50%', background: '#0066cc',
                              border: '3px solid #fff', boxShadow: '0 0 12px rgba(0,102,204,0.5)'
                            }} />
                          </Marker>

                          {/* Drainage Layer — prominent */}
                          {drainageData && (
                            <Source id="drainage-main" type="geojson" data={drainageData}>
                              <Layer id="drainage-main-line" type="line" paint={{
                                "line-color": "#0ea5e9",
                                "line-width": 3,
                                "line-opacity": 0.8
                              }} />
                            </Source>
                          )}

                          {/* Manholes (inferred network nodes) */}
                          {manholeData && (
                            <Source id="manholes" type="geojson" data={manholeData}>
                              <Layer id="manhole-points" type="circle" minzoom={13} paint={{
                                "circle-color": ["match", ["get", "kind"],
                                  "junction", MANHOLE_COLORS.junction,
                                  "joint", MANHOLE_COLORS.joint,
                                  MANHOLE_COLORS.terminal],
                                "circle-radius": ["interpolate", ["linear"], ["zoom"], 13, 2, 16, 5, 19, 9],
                                "circle-stroke-color": "#ffffff",
                                "circle-stroke-width": 1,
                              }} />
                            </Source>
                          )}

                          {/* Selected manhole: highlighted pin + location popup */}
                          {selectedManhole && (
                            <>
                              <Marker longitude={selectedManhole.lng} latitude={selectedManhole.lat} anchor="center">
                                <div style={{
                                  width: 22, height: 22, borderRadius: "50%",
                                  border: `3px solid ${MANHOLE_COLORS[selectedManhole.kind] || "#7c3aed"}`,
                                  background: "rgba(124,58,237,0.15)", animation: "pulse-ring 2s ease-out infinite",
                                }} />
                              </Marker>
                              <MapboxPopup
                                longitude={selectedManhole.lng} latitude={selectedManhole.lat}
                                anchor="bottom" offset={14} closeOnClick={false}
                                onClose={() => setSelectedManhole(null)}
                              >
                                <div style={{ color: "#17171c", fontSize: "13px", lineHeight: 1.5 }}>
                                  <strong>{selectedManhole.manhole_id}</strong> ({selectedManhole.kind})<br />
                                  Location: {selectedManhole.lat.toFixed(6)}, {selectedManhole.lng.toFixed(6)}<br />
                                  Connected pipes: {selectedManhole.connected_pipes}<br />
                                  {selectedManhole.max_diameter_mm != null && (<>Max pipe: {selectedManhole.max_diameter_mm} mm<br /></>)}
                                  {selectedManhole.ward != null && (<>Ward: {selectedManhole.ward}<br /></>)}
                                  {selectedManhole.distance != null && (<>Distance from you: {Math.round(selectedManhole.distance)} m<br /></>)}
                                  <a
                                    href={`https://www.google.com/maps/dir/?api=1&destination=${selectedManhole.lat},${selectedManhole.lng}`}
                                    target="_blank" rel="noopener noreferrer"
                                  >
                                    Directions
                                  </a>
                                  <div style={{ fontSize: "11px", color: "#6b7280", marginTop: 4 }}>Inferred position</div>
                                </div>
                              </MapboxPopup>
                            </>
                          )}
                        </Map>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* ── CONNECTIVITY TAB (no map) ── */}
          {activeTab === 'connectivity' && (
            <section className="tab-content">
              <div className="connectivity-layout">
                <div className="panel connectivity-panel">
                  <DataStatus />
                </div>
              </div>
            </section>
          )}
        </main>
      </div>
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

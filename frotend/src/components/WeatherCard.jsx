import { useEffect, useState } from "react";
import { getWeather } from "../services/api.js";
import "./WeatherCard.css";

export default function WeatherCard() {
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchWeather() {
      try {
        const response = await getWeather();
        const data = response.data;
        if (data && data.available) {
          setWeather(data);
        } else {
          setError(data?.reason || "Weather data unavailable");
        }
      } catch (err) {
        setError(err.message || "Failed to fetch weather");
      } finally {
        setLoading(false);
      }
    }
    fetchWeather();
  }, []);

  if (loading) {
    return (
      <div className="wc wc--loading">
        <div className="wc__shimmer"></div>
        <div className="wc__shimmer short"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="wc wc--error">
        <div className="wc__error-icon">⚠</div>
        <h4>Weather Unavailable</h4>
        <p>{error}</p>
      </div>
    );
  }

  if (!weather) return null;

  // Determine weather icon based on code or conditions
  const getWeatherEmoji = () => {
    if (weather.rainfall_24h > 10) return "🌧";
    if (weather.rainfall_24h > 0) return "🌦";
    if (weather.humidity > 80) return "☁️";
    return "☀️";
  };

  return (
    <div className="wc">
      <div className="wc__header">
        <div className="wc__location">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z" />
            <circle cx="12" cy="10" r="3" />
          </svg>
          <span>{weather.station_name || "Kolkata"}</span>
        </div>
        <div className="wc__source">{weather.source || "IMD Live"}</div>
      </div>

      <div className="wc__main">
        <div className="wc__temp-block">
          <span className="wc__emoji">{getWeatherEmoji()}</span>
          <span className="wc__temp">{weather.temperature != null ? `${weather.temperature}°` : "--"}</span>
        </div>

        <div className="wc__details">
          <div className="wc__detail">
            <span className="wc__detail-val">{weather.humidity != null ? `${weather.humidity}%` : "--"}</span>
            <span className="wc__detail-lbl">Humidity</span>
          </div>
          <div className="wc__detail" title={weather.rainfall_24h_period || undefined}>
            <span className="wc__detail-val">{weather.rainfall_24h != null ? `${weather.rainfall_24h}mm` : "--"}</span>
            <span className="wc__detail-lbl">24h Rain</span>
          </div>
          {weather.wind_speed_kmph != null && (
            <div className="wc__detail">
              <span className="wc__detail-val">{weather.wind_speed_kmph} km/h</span>
              <span className="wc__detail-lbl">Wind</span>
            </div>
          )}
        </div>
      </div>

      <div className="wc__timestamp">
        {weather.timestamp || new Date().toLocaleDateString()}
      </div>
    </div>
  );
}

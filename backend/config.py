"""
FloodWatch backend configuration.

All settings are read from environment variables (or backend/.env for local
development). Secrets have NO default values: when unset they are None, and the
service that needs them must report itself as unavailable rather than crash the
whole app. This keeps the legacy endpoints (/api/predict, etc.) working even when
IMD or Mapbox credentials are not configured.

Relative paths are resolved against the backend/ directory, not the current
working directory, so the app behaves the same whether it is started from the
repo root or from backend/.

Written for Python 3.9 compatibility (Optional[...] rather than `X | None`).
"""

from pathlib import Path
from typing import List, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ---- Secrets (no defaults; None means "not configured") ----
    IMD_API_KEY: Optional[SecretStr] = None
    IMD_EMAIL: Optional[SecretStr] = None
    IMD_PASSWORD: Optional[SecretStr] = None
    MAPBOX_ACCESS_TOKEN: Optional[SecretStr] = None

    # ---- Data paths (dev defaults inside backend/data) ----
    DEM_PATH: Path = DATA_DIR / "dem" / "P5_PAN_CD_N22_000_E088_000_DEM_30m.tif"
    # PROJ EGM96 geoid grid (public domain, NGA) used to convert CartoDEM's
    # WGS84-ellipsoidal heights to EGM96 orthometric heights (~metres above MSL).
    GEOID_GRID_PATH: Path = DATA_DIR / "geoid" / "us_nga_egm96_15.tif"
    DRAINAGE_GEOJSON_PATH: Path = DATA_DIR / "gis" / "drainage_network.geojson"
    PUMPING_STATIONS_GEOJSON_PATH: Path = DATA_DIR / "gis" / "pumping_stations.geojson"
    WATER_BODIES_GEOJSON_PATH: Path = DATA_DIR / "gis" / "water_bodies.geojson"
    HISTORICAL_WATERLOGGING_PATH: Path = DATA_DIR / "historical" / "waterlogging_points.csv"
    # Area the waterlogging records cover (KMC). Outside it, "no record" = no data.
    KMC_BOUNDARY_GEOJSON_PATH: Path = DATA_DIR / "gis" / "kmc_boundary.geojson"
    # ---- MongoDB ----
    MONGODB_URI: str = Field(default="mongodb://localhost:27017/floodwatch")

    # ---- Auth ----
    AUTH_JWT_SECRET: SecretStr = SecretStr("floodwatch-dev-auth-secret-change-me")
    AUTH_JWT_ALGORITHM: str = "HS256"
    AUTH_JWT_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # ---- Model ----
    # Stays "synthetic_v1" until the real-data model (real_v1) is trained and verified.
    MODEL_VERSION: str = "synthetic_v1"

    # ---- Route / spatial tuning ----
    ROUTE_SAMPLE_DISTANCE_M: float = Field(default=75.0, gt=0)
    HISTORICAL_WATERLOGGING_RADIUS_M: float = Field(default=250.0, gt=0)
    DRAIN_MAX_SEARCH_RADIUS_M: float = Field(default=500.0, gt=0)
    PUMP_MAX_SEARCH_RADIUS_M: float = Field(default=3000.0, gt=0)
    WATERBODY_MAX_SEARCH_RADIUS_M: float = Field(default=1000.0, gt=0)

    # ---- Diagnostics ----
    # Raw IMD payloads may contain account details; keep off unless debugging.
    DEBUG_LOG_IMD_RAW: bool = False

    @field_validator(
        "DEM_PATH",
        "GEOID_GRID_PATH",
        "DRAINAGE_GEOJSON_PATH",
        "PUMPING_STATIONS_GEOJSON_PATH",
        "WATER_BODIES_GEOJSON_PATH",
        "HISTORICAL_WATERLOGGING_PATH",
        "KMC_BOUNDARY_GEOJSON_PATH",
        mode="after",
    )
    @classmethod
    def _resolve_relative_to_backend(cls, value: Path) -> Path:
        return value if value.is_absolute() else (BACKEND_DIR / value).resolve()

    def missing_secrets(self) -> List[str]:
        """Names of secret settings that are not configured (values never exposed)."""
        names = ["IMD_API_KEY", "IMD_EMAIL", "IMD_PASSWORD", "MAPBOX_ACCESS_TOKEN"]
        return [
            n for n in names
            if getattr(self, n) is None or not getattr(self, n).get_secret_value().strip()
        ]


settings = Settings()

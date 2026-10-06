from config import BACKEND_DIR, Settings

SECRET_VARS = {
    "IMD_API_KEY": "test-imd-key",
    "IMD_EMAIL": "tester@example.com",
    "IMD_PASSWORD": "test-password",
    "MAPBOX_ACCESS_TOKEN": "pk.test-token",
}


def _settings(**kwargs) -> Settings:
    # _env_file=None: ignore any developer backend/.env so tests are deterministic.
    return Settings(_env_file=None, **kwargs)


def test_loads_with_required_env_vars(monkeypatch):
    for key, value in SECRET_VARS.items():
        monkeypatch.setenv(key, value)
    s = _settings()
    assert s.MAPBOX_ACCESS_TOKEN.get_secret_value() == "pk.test-token"
    assert s.missing_secrets() == []


def test_route_sample_distance_defaults_to_75(monkeypatch):
    monkeypatch.delenv("ROUTE_SAMPLE_DISTANCE_M", raising=False)
    assert _settings().ROUTE_SAMPLE_DISTANCE_M == 75


def test_missing_secrets_are_none_not_crash(monkeypatch):
    for key in SECRET_VARS:
        monkeypatch.delenv(key, raising=False)
    s = _settings()
    assert s.IMD_API_KEY is None
    assert set(s.missing_secrets()) == set(SECRET_VARS)


def test_secrets_not_exposed_in_repr(monkeypatch):
    for key, value in SECRET_VARS.items():
        monkeypatch.setenv(key, value)
    text = repr(_settings())
    for value in SECRET_VARS.values():
        assert value not in text


def test_relative_paths_resolve_against_backend(monkeypatch):
    monkeypatch.setenv("DEM_PATH", "data/dem/x.tif")
    assert _settings().DEM_PATH == (BACKEND_DIR / "data/dem/x.tif").resolve()


def test_default_model_version_is_synthetic(monkeypatch):
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    assert _settings().MODEL_VERSION == "synthetic_v1"

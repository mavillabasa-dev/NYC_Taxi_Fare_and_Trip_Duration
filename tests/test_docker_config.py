# tests/test_docker_config.py — Validation tests for Docker and Docker Compose configuration (T-114).
from pathlib import Path
import yaml
import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def compose_config(repo_root) -> dict:
    compose_path = repo_root / "docker-compose.yml"
    assert compose_path.exists(), f"docker-compose.yml not found at {compose_path}"
    with open(compose_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_docker_compose_v2_specification(compose_config):
    """Assert that obsolete version key is removed and services exist."""
    assert "version" not in compose_config, "Obsolete top-level 'version' key must be removed for Compose v2"
    assert "services" in compose_config, "Missing 'services' in docker-compose.yml"
    services = compose_config["services"]
    assert "api" in services, "Missing 'api' service in docker-compose.yml"
    assert "dashboard" in services, "Missing 'dashboard' service in docker-compose.yml"
    assert "trainer" in services, "Missing 'trainer' service in docker-compose.yml"


def test_api_service_config(compose_config):
    """Validate api service configuration, ports, healthcheck, and model volume mount."""
    api = compose_config["services"]["api"]
    assert api["build"]["context"] == "./api"
    assert api["build"]["dockerfile"] == "Dockerfile"
    assert "8000:8000" in api["ports"]
    assert ".env" in api.get("env_file", [])

    # Check models volume mount
    volumes = api.get("volumes", [])
    assert any("./models:/app/models" in v for v in volumes), "api service must mount ./models:/app/models"

    # Check healthcheck
    healthcheck = api.get("healthcheck", {})
    assert "test" in healthcheck, "api service must have a healthcheck test"
    assert "/health" in str(healthcheck["test"]), "Healthcheck must target the /health endpoint"


def test_dashboard_service_config(compose_config):
    """Validate dashboard service configuration, environment, dataset volume, and dependency on healthy api."""
    dashboard = compose_config["services"]["dashboard"]
    assert dashboard["build"]["context"] == "./ui"
    assert dashboard["build"]["dockerfile"] == "Dockerfile"
    assert "8501:8501" in dashboard["ports"]

    env = dashboard.get("environment", [])
    assert "API_URL=http://api:8000" in env, "dashboard must point to http://api:8000"
    assert "DATASET_DIR=/app/dataset" in env, "dashboard must set DATASET_DIR=/app/dataset"

    # Check dataset volume mount
    volumes = dashboard.get("volumes", [])
    assert any("./dataset:/app/dataset" in v for v in volumes), "dashboard service must mount ./dataset:/app/dataset"

    # Check dependency on api health
    depends_on = dashboard.get("depends_on", {})
    assert "api" in depends_on, "dashboard must depend on api"
    if isinstance(depends_on, dict):
        assert depends_on["api"].get("condition") == "service_healthy", "dashboard must wait for api condition service_healthy"


def test_trainer_service_config(compose_config):
    """Validate trainer profile isolation and volume mounts."""
    trainer = compose_config["services"]["trainer"]
    assert "train" in trainer.get("profiles", []), "trainer service must be registered under profiles: ['train']"
    assert trainer["build"]["dockerfile"] == "Dockerfile.train"
    assert trainer.get("restart") == "no", "trainer service must have restart: 'no'"

    volumes = trainer.get("volumes", [])
    assert any("./dataset:/workspace/dataset" in v for v in volumes), "trainer must mount ./dataset"
    assert any("./models:/workspace/models" in v for v in volumes), "trainer must mount ./models"


def test_api_requirements_pinned(repo_root):
    """Assert all runtime dependencies in api/requirements.txt are pinned with exact versions."""
    req_path = repo_root / "api" / "requirements.txt"
    assert req_path.exists()
    lines = [
        line.strip()
        for line in req_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) >= 6
    for line in lines:
        assert "==" in line, f"Dependency '{line}' in api/requirements.txt must be pinned with '=='"
    assert any("lightgbm==" in line for line in lines), "api/requirements.txt must pin lightgbm"


def test_ui_requirements_pinned(repo_root):
    """Assert all frontend dependencies in ui/requirements.txt are pinned with exact versions."""
    req_path = repo_root / "ui" / "requirements.txt"
    assert req_path.exists()
    lines = [
        line.strip()
        for line in req_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) >= 4
    for line in lines:
        assert "==" in line, f"Dependency '{line}' in ui/requirements.txt must be pinned with '=='"


def test_dockerfiles_exist_and_use_python311(repo_root):
    """Assert that all Dockerfiles exist and use python:3.11-slim base image."""
    for dockerfile_path in [
        repo_root / "api" / "Dockerfile",
        repo_root / "ui" / "Dockerfile",
        repo_root / "Dockerfile.train",
    ]:
        assert dockerfile_path.exists(), f"Missing {dockerfile_path.name}"
        content = dockerfile_path.read_text()
        assert "python:3.11-slim" in content, f"{dockerfile_path.name} must use python:3.11-slim"

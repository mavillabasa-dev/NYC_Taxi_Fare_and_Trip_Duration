# T-114 — Containerisation and Docker Configuration Report

This document details the multi-container architecture, pinned runtime dependencies, Compose profiles, healthcheck contracts, and operational instructions for ticket **T-114**.

---

## 1. Multi-Container System Architecture

The deployment is containerized using **Docker Compose v2** into distinct service tiers separating real-time serving from on-demand training:

```mermaid
graph TD
    subgraph Host Machine
        DotEnv[".env (Environment Configuration)"]
        ModelsDir["./models (Trained Artifacts: model.pkl)"]
        DatasetDir["./dataset (Parquet Data & Shapefiles)"]
    end

    subgraph Docker Network: default
        API["API Service (FastAPI :8000)<br/>• Python 3.11-slim<br/>• Healthcheck: GET /health<br/>• Zero src/ dependency"]
        UI["Dashboard Service (Streamlit :8501)<br/>• Python 3.11-slim<br/>• Waits for healthy API<br/>• Choropleth & Point Maps"]
        Trainer["Trainer Service (One-Shot Task)<br/>• Profile: 'train'<br/>• Python 3.11-slim + C++ toolchain<br/>• Runs full ingestion & modeling pipeline"]
    end

    DotEnv --> API
    ModelsDir -->|bind mount :ro| API
    DatasetDir -->|bind mount :ro| UI
    ModelsDir <-->|bind mount :rw| Trainer
    DatasetDir <-->|bind mount :rw| Trainer
    API -->|API_URL=http://api:8000| UI
```

---

## 2. Services Specification

### 1. `api` Service (Serving Backend)
* **Build Context**: `./api` (using [api/Dockerfile](../api/Dockerfile)).
* **Base Image**: `python:3.11-slim` with `libgomp1` for LightGBM OpenMP parallelization.
* **Port**: `8000:8000`.
* **Volume Mount**: `./models:/app/models:ro`.
* **Healthcheck**: Polling `GET /health` every 10s (`test: ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\""]`).
* **Runtime Dependencies**: Strictly pinned in [api/requirements.txt](../api/requirements.txt):
  ```txt
  fastapi==0.115.6
  uvicorn[standard]==0.34.0
  pydantic==2.10.4
  scikit-learn==1.6.1
  pandas==2.2.3
  numpy==2.2.1
  lightgbm==4.7.0
  ```

### 2. `dashboard` Service (Streamlit Frontend)
* **Build Context**: `./ui` (using [ui/Dockerfile](../ui/Dockerfile)).
* **Base Image**: `python:3.11-slim`.
* **Port**: `8501:8501`.
* **Environment**: `API_URL=http://api:8000`, `DATASET_DIR=/app/dataset`.
* **Volume Mount**: `./dataset:/app/dataset:ro`.
* **Startup Sequencing**: `depends_on: api: condition: service_healthy` ensures Streamlit only initializes when FastAPI has successfully unpickled `models/model.pkl` and is healthy.
* **Dependencies**: Strictly pinned in [ui/requirements.txt](../ui/requirements.txt):
  ```txt
  streamlit==1.41.1
  requests==2.32.3
  plotly==5.24.1
  pandas==2.2.3
  ```

### 3. `trainer` Service (On-Demand Ephemeral Pipeline)
* **Profile**: Registered under `profiles: ["train"]` with `restart: "no"`.
* **Build Context**: `.` (using [Dockerfile.train](../Dockerfile.train)).
* **Volume Mounts**: `./dataset:/workspace/dataset` and `./models:/workspace/models`.
* **Entrypoint**: Runs [scripts/run_training_pipeline.py](../scripts/run_training_pipeline.py) (Ingestion $\to$ Preprocessing $\to$ Feature Pipeline $\to$ Model Selection $\to$ Isolation Acceptance Test).

---

## 3. Operational Commands & Workflow

### 1. Prerequisites
Copy `.env.original` to `.env` if not already present:
```bash
cp .env.original .env
```

### 2. Build & Run Serving Platform
```bash
# 1. Build images
docker compose build
# (or shortcut: make build)

# 2. Start services (API on :8000, Dashboard on :8501)
docker compose up
# (or shortcut: make run)

# 3. Stop services
docker compose down
# (or shortcut: make down)
```

### 3. Run One-Shot ML Training Pipeline in Docker
To retrain and export new model artifacts headlessly inside a clean container:
```bash
docker compose --profile train run --rm trainer
```

### 4. Model Hot-Swap Verification
Because `./models` is mounted as a volume into `/app/models`:
1. When retraining completes (either on host or via the `trainer` container), `models/model.pkl` is updated on the host filesystem.
2. Restarting the `api` container (`docker compose restart api`) immediately loads the new model artifact with **zero image rebuilds required**.

---

## 4. Verification & Acceptance
All Docker and Compose configurations are validated by automated unit tests in [tests/test_docker_config.py](../tests/test_docker_config.py):
* Compose v2 specification conformity.
* Dependency pinning verification.
* Healthcheck wiring verification.
* Profile isolation verification.

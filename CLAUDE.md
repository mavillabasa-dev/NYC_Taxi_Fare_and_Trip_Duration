# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This repository is a **scaffold**. Every `.py` file under [src/](src/) and [api/](api/) contains a single Spanish comment describing its intended purpose and nothing else; the three notebooks contain only a title cell. There is no working code, no trained model, and no dataset yet. Expect to write implementations from scratch rather than extend existing ones.

Language split: the placeholder comments in `src/` and `api/` are **Spanish** — match that when editing those files. [README.md](README.md) is **English**. Files are UTF-8 — on Windows/PowerShell pass `-Encoding utf8` explicitly when writing via shell redirection, or accented characters will be mangled.

## Commands

```
make build    # docker compose build
make run      # docker compose up  (requires .env — see below)
make test     # pytest
make down     # docker compose down
```

`make run` fails without a `.env` file: `docker-compose.yml` declares `env_file: .env`, but `.env` is gitignored. Copy the template first:

```
Copy-Item .env.original .env
```

Run a single test: `pytest tests/test_integration.py::test_name` (or `pytest -k <expr>`).

There is no lint/format configuration and no root `requirements.txt`. `pytest` is **not** in [api/requirements.txt](api/requirements.txt), so `make test` requires it installed separately in the local environment. Training-side dependencies (LightGBM, XGBoost, an MLP framework) are not declared anywhere yet. T-101 calls for a root `requirements.txt` plus a `requirements-dev.txt` carrying `pytest`; create both rather than adding to `api/requirements.txt`, which is the container's runtime manifest and should stay minimal.

`api/requirements.txt` is also unpinned and currently lists only `scikit-learn` and `pandas`. Whichever library wins model selection in T-109 must be added there, or the container will fail to unpickle the artifact at startup.

Python 3.11 (pinned by [api/Dockerfile](api/Dockerfile) and the notebook kernels).

## Architecture

Two **independent** Python trees that never import each other:

- **[src/](src/)** — offline pipeline, runs on the host only. `config.py` (paths/constants) → `data_utils.py` (loading) → `preprocessing.py` (cleaning/transforms) → `train.py` (fit + evaluate, writes the model artifact).
- **[api/](api/)** — online serving, runs in the container. `main.py` (FastAPI entrypoint) mounts `app/model/router.py` (endpoints), which calls `app/model/services.py` (load model, predict) using the Pydantic contracts in `app/model/schema.py`. `settings.py` reads environment config.

The Docker build context is `./api`, so **`src/` is not present at runtime**. The only interface between training and serving is the serialized model file: `docker-compose.yml` bind-mounts `./models` to `/app/models`, and `MODEL_PATH=models/model.pkl` resolves against `WORKDIR /app`. Anything `src/train.py` needs at inference time (encoders, scalers, feature ordering) must therefore be serialized into that artifact or written alongside it in `models/` — it cannot be imported from `src/`.

Inside the image the API package root is flattened: `COPY . .` from `./api` puts `main.py` at `/app/main.py`, which is why the CMD is `uvicorn main:app`. Imports inside `api/` must be written to work with `api/` as the top level (`from app.model.router import ...`), not `from api.app...`. Tests run from the repo root see a different layout — [api/tests/](api/tests/) needs `sys.path`/`conftest.py` handling or root-relative imports to collect there.

`dataset/` and `models/` hold only `.gitkeep`; `*.parquet`, `*.csv`, `*.pkl`, `*.h5` are gitignored. Data and artifacts are never committed — treat both directories as locally-populated.

The Dockerfile hardcodes `--host 0.0.0.0 --port 8000`, so `API_HOST`/`API_PORT` from `.env` only take effect if `settings.py`/`main.py` read them for local (non-container) runs.

## Domain constraints

These are properties of the data, not preferences. Getting them wrong silently produces a model that scores well and is worthless. [README.md](README.md) states each as ticket acceptance criteria.

**There are no coordinates.** TLC removed latitude/longitude in mid-2016. The 2022 yellow taxi parquet has `PULocationID` / `DOLocationID` — taxi zone IDs, 1–265. Any source describing the model as taking "pickup and dropoff coordinates" (including [intro.md](intro.md)) predates that change. Coordinates exist only as zone centroids derived from the Taxi Zone Shapefile, which makes the shapefile a required input alongside the trip parquet and the zone lookup CSV.

**Most columns are post-trip.** The target is predicted at the *start* of the ride, so `tpep_dropoff_datetime` (which defines the duration target), `total_amount`, `tip_amount`, `tolls_amount`, `extra`, `mta_tax`, `improvement_surcharge`, `congestion_surcharge`, `airport_fee`, `payment_type` and `store_and_fwd_flag` are all banned as features. `fare_amount` is a target, never an input. The README carries the full allowed/banned table.

**Split by time, not at random.** Training uses a single month (May 2022); a random split leaks across time. The split boundary belongs in `src/config.py`.

**`trip_distance` is an accepted approximation.** It is the metered distance of the *completed* trip, so using it as an input is mildly optimistic — production would supply a routing estimate. Keep the assumption documented rather than quietly relying on it.

## Workflow

[README.md](README.md) defines the project as 19 tickets (T-101 … T-119) with a dependency DAG and per-ticket acceptance criteria. Build order: setup/research → dataset ingestion → EDA → preprocessing → feature engineering → the three model families **in parallel** (baselines, gradient boosting, MLP) → model selection → API + dashboard → tests/benchmarking/containerization → peer review → docs. Check the ticket a change belongs to and whether its upstream dependencies are actually implemented before building on them.

Ticket IDs are stable and not contiguous in execution order: T-116 (dataset ingestion) runs early, before T-103; T-117 and T-118 are optional and off the critical path; T-119 (peer review) runs just before T-115. The README's directory is grouped by phase for that reason.

Notebooks are numbered to match that flow (`01_EDA` → `02_preprocessing` → `03_feature_engineering` → `04_model_experiments`). They are exploration surfaces; production logic belongs in `src/`.

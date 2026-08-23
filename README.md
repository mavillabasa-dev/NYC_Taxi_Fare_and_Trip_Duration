# NYC Taxi Fare and Trip Duration Prediction

Predict the **fare** and **duration** of a New York City yellow taxi ride using only
information available *at the start of the trip*.

---

## Dataset

| Item | Value |
|------|-------|
| Source | NYC TLC Trip Record Data |
| Record type | Yellow Taxi Trip Records |
| Period | **May 2022** (single month, PARQUET) |
| Auxiliary | Taxi Zone Lookup Table (CSV), Taxi Zone Shapefile |

> [!WARNING]
> **The 2022 dataset contains no latitude/longitude columns.** TLC removed raw
> coordinates in mid-2016 and replaced them with `PULocationID` / `DOLocationID`
> (taxi zone IDs, 1–265). Any reference to "pickup and dropoff coordinates" in the
> project brief must be read as *zone IDs*. Coordinates can only be obtained by
> deriving zone centroids from the Taxi Zone Shapefile — which is why the shapefile
> is a **required input**, not a reference document.

### Feature contract (leakage control)

The parquet ships many columns that are only known *after* the ride ends. These are
banned as features across every model ticket.

| Allowed at prediction time | Banned (post-trip / target leakage) |
|---|---|
| `tpep_pickup_datetime` | `tpep_dropoff_datetime` (source of the duration target) |
| `PULocationID`, `DOLocationID` | `fare_amount` (fare target), `total_amount` |
| `passenger_count` | `tip_amount`, `tolls_amount`, `extra`, `mta_tax` |
| `RatecodeID` | `improvement_surcharge`, `congestion_surcharge`, `airport_fee` |
| `trip_distance` ¹ | `payment_type`, `store_and_fwd_flag` |
| `VendorID` | |

¹ `trip_distance` is the *metered* distance of the completed trip. In production you
would only have a routing estimate. Using it is a documented, accepted approximation —
state the assumption in T-105 rather than pretending it is leak-free.

**Targets:** `fare_amount`, and `duration_minutes = tpep_dropoff_datetime − tpep_pickup_datetime`.

---

## Architecture constraints

Two independent Python trees that never import each other. See [CLAUDE.md](CLAUDE.md)
for detail. The three constraints that shape the tickets:

1. **`src/` is not in the Docker build context.** The build context is `./api`, so the
   serialized model artifact is the *only* interface between training and serving.
   Encoders, scalers and feature ordering must be inside the artifact.
2. **The container must be able to unpickle the artifact.** Whichever library wins
   model selection (LightGBM, XGBoost, …) has to appear in `api/requirements.txt`.
3. **`dataset/` and `models/` are gitignored.** Both are populated locally; every
   step that produces them must be reproducible from a script.

---

## Ticket Directory

Tickets are listed in **execution order**. IDs are stable — T-116…T-119 were added
later and deliberately not renumbered.

### Phase 1 — Foundation

| Ticket | Category | Description |
|--------|----------|-------------|
| T-101 | `SETUP` | Repository setup and modular structure |
| T-102 | `RESEARCH` | Literature review summary and schemas |
| T-116 | `DATA` | Reproducible dataset ingestion |

### Phase 2 — Data

| Ticket | Category | Description |
|--------|----------|-------------|
| T-103 | `EDA` | Exploratory data analysis notebook |
| T-104 | `DATA` | Preprocessing and data cleaning module |
| T-105 | `DATA` | Feature engineering pipeline |

### Phase 3 — Modelling (parallel)

| Ticket | Category | Description |
|--------|----------|-------------|
| T-106 | `ML` | Baseline regressors (linear and trees) |
| T-107 | `ML` | Gradient-boosting ensembles (LightGBM and XGBoost) |
| T-108 | `ML` | Multi-layer perceptron (MLP) evaluation |
| T-109 | `EVAL` | Model selection, benchmarking and evaluation |

### Phase 4 — Serving

| Ticket | Category | Description |
|--------|----------|-------------|
| T-110 | `API` | Real-time prediction backend API |
| T-111 | `FRONTEND` | Interactive visual dashboard and demo |
| T-112 | `TEST` | Unit and integration test suite |
| T-113 | `EVAL` | API benchmarking and latency optimisation |
| T-114 | `DEVOPS` | Containerisation and Docker configuration |

### Phase 5 — Delivery

| Ticket | Category | Description |
|--------|----------|-------------|
| T-119 | `REVIEW` | Peer preview and feedback round |
| T-115 | `SETUP` | Final documentation and presentation prep |

### Optional (off the critical path)

| Ticket | Category | Description |
|--------|----------|-------------|
| T-117 | `OPTIONAL` | Weather enrichment via third-party API |
| T-118 | `OPTIONAL` | Taxi demand prediction by region |

---

## Dependency Flow

```mermaid
graph LR
    T101[T-101<br/>Setup] --> T116[T-116<br/>Ingestion]
    T102[T-102<br/>Research] --> T103[T-103<br/>EDA]
    T116 --> T103
    T103 --> T104[T-104<br/>Preprocessing]
    T104 --> T105[T-105<br/>Features]
    T105 --> T106[T-106<br/>Baselines]
    T105 --> T107[T-107<br/>GBDT]
    T105 --> T108[T-108<br/>MLP]
    T106 --> T109[T-109<br/>Selection]
    T107 --> T109
    T108 --> T109
    T109 --> T110[T-110<br/>API]
    T109 --> T111[T-111<br/>Dashboard]
    T110 --> T112[T-112<br/>Tests]
    T110 --> T113[T-113<br/>Latency]
    T110 --> T114[T-114<br/>Docker]
    T111 --> T114
    T114 --> T119[T-119<br/>Peer review]
    T112 --> T115[T-115<br/>Docs]
    T113 --> T115
    T119 --> T115
    T105 -.-> T117[T-117<br/>Weather]
    T104 -.-> T118[T-118<br/>Demand]
```

Edge list (transitively reduced):

```
T-101 → T-116
T-102 → T-103
T-116 → T-103
T-103 → T-104
T-104 → T-105
T-105 → T-106,  T-105 → T-107,  T-105 → T-108     (parallel)
T-106 → T-109,  T-107 → T-109,  T-108 → T-109
T-109 → T-110,  T-109 → T-111
T-110 → T-112,  T-110 → T-113,  T-110 → T-114
T-111 → T-114
T-114 → T-119
T-112 → T-115,  T-113 → T-115,  T-119 → T-115

optional:  T-105 ⇢ T-117    T-104 ⇢ T-118
```

**Changes from the original DAG**

- `T-106 → T-108` and `T-107 → T-108` removed. The MLP does not depend on gradient
  boosting; the three model families run in parallel off T-105.
- `T-112` was orphaned — it appeared in the directory with no edges at all. Now
  `T-110 → T-112 → T-115`.
- `T-116` inserted before T-103. Nothing previously put data on disk.
- `T-115` now depends on T-112 and (via T-119) on T-111. Final docs need passing
  tests and a demo that has been shown to someone.

---

## Tickets and Acceptance Criteria

### T-101 · `SETUP` · Repository setup and modular structure

**Depends on:** —

- [x] `src/` and `api/` trees exist with the module layout described in CLAUDE.md.
- [x] Root `requirements.txt` (training side) and `requirements-dev.txt` (with
      `pytest`) exist. `make test` runs without a separate manual install.
- [x] `.env` is created from `.env.original`; `make build` and `make down` succeed
      on a clean checkout.
- [x] `dataset/` and `models/` exist with `.gitkeep` and are gitignored.
- [x] Random seed constant defined once in `src/config.py` and imported everywhere.

### T-102 · `RESEARCH` · Literature review summary and schemas

**Depends on:** —
**Artifact:** `docs/Literature_Review.md`

- [x] Short written summary (≤2 pages) of the reference papers and articles.
- [x] Documented list of features other authors found predictive, mapped onto the
      columns actually present in the 2022 schema.
- [x] Architecture diagram of the end-to-end system (training → artifact → API → UI).

### T-116 · `DATA` · Reproducible dataset ingestion

**Depends on:** T-101 — **Artifact:** `src/data_utils.py`, `dataset/taxi_zone_centroids.csv`

- [x] Script (`src/data_utils.py`) downloads, into `dataset/`:
      the May 2022 Yellow Taxi parquet, the Taxi Zone Lookup CSV, and the Taxi Zone
      Shapefile.
- [x] Downloads are idempotent — re-running skips files already present.
- [x] File sizes / row counts are logged and asserted so a truncated download fails
      loudly instead of silently.
- [x] Zone centroids are derived from the shapefile and cached as a lookup table
      keyed by `LocationID`.
- [x] Documented in the README: one command, from empty checkout to populated
      `dataset/`.

### T-103 · `EDA` · Exploratory data analysis notebook

**Depends on:** T-102, T-116 — **Artifact:** `notebooks/01_EDA.ipynb`

- [x] Row count, column dtypes, missing values and duplicate rate reported.
- [x] Distributions of both targets (`fare_amount`, duration) with skew quantified;
      explicit recommendation on whether to log-transform.
- [x] Outlier analysis: negative/zero fares, zero-distance trips, durations of 0 or
      >6 h, `passenger_count = 0`, trips whose timestamps fall outside May 2022.
- [x] Top pickup/dropoff zones and the fare/duration profile of airport rate codes
      (`RatecodeID` 2 and 3).
- [x] Temporal profile: fare and duration by hour-of-day and day-of-week — the
      evidence base for the temporal split in T-104.
- [x] Correlation analysis restricted to the allowed-feature list above.

### T-104 · `DATA` · Preprocessing and data cleaning module

**Depends on:** T-103 — **Artifact:** `src/preprocessing.py`, `notebooks/02_preprocessing.ipynb`

- [x] Cleaning rules from T-103 implemented as pure, individually testable functions.
- [x] All banned columns dropped in a single explicit step; a test asserts none of
      them survive into the feature frame.
- [x] **Temporal split**, not random: train on the first ~3 weeks of May 2022, test on
      the last week. Split boundary is a constant in `src/config.py`.
- [x] Rows dropped are counted and logged by rule, so the cleaning cost is visible.
- [x] Processed dataset written to `dataset/` in parquet.

### T-105 · `DATA` · Feature engineering pipeline

**Depends on:** T-104 — **Artifact:** `src/features.py`, `notebooks/03_feature_engineering.ipynb`, `models/feature_pipeline.pkl`

- [x] Temporal features: hour, day-of-week, month-day, weekend flag, rush-hour flag,
      US holiday flag; cyclical encoding for hour and weekday.
- [x] Zone features built from `PULocationID` / `DOLocationID` — **not** from raw
      coordinates. Includes borough and service-zone joins from the lookup table.
- [x] Haversine distance between zone centroids, plus its ratio to `trip_distance`.
- [x] Categorical encoding strategy chosen and justified (265 zones — one-hot is
      likely wrong; target/ordinal encoding fitted on train only).
- [x] The whole pipeline is a single fitted object (`sklearn` `Pipeline` or
      `ColumnTransformer`) that can be serialized — no loose transformation steps.
- [x] Fitted on train split only; a test asserts no statistic is computed on test data.
- [x] The `trip_distance` assumption (¹ above) documented in a module docstring.

### T-106 · `ML` · Baseline regressors (linear and trees)

**Depends on:** T-105 — **Artifact:** `src/train.py`, `notebooks/04_model_experiments.ipynb`, `models/dt_models.pkl`, `models/dummy_models.pkl`

- [x] Trivial baseline established first: predict the training mean for each target.
      Every later model is reported as improvement over this.
- [x] Decision Tree trained for **both** targets.
- [x] Linear Regression trained for both targets, behind a median imputer — the
      coordinate features carry NaN for the two zones without a shapefile centroid,
      which trees tolerate and linear models do not.
- [x] Decision recorded and justified: two single-output models vs. one multi-output
      model. This choice binds T-107, T-108 and the `MODEL_PATH` contract in T-110.
- [x] Metrics reported per target: MAE, RMSE, MAPE, R².
- [x] Training time and single-row inference time recorded.

### T-107 · `ML` · Gradient-boosting ensembles (LightGBM and XGBoost)

**Depends on:** T-105 — **Runs parallel to T-106, T-108**

- [x] LightGBM and XGBoost trained for both targets on the same splits as T-106.
- [x] Hyperparameter search documented (method, search space, budget) and reproducible.
- [x] Feature importance reported; any feature that looks suspiciously dominant is
      re-checked against the leakage contract.
- [x] Same metric set and timing measurements as T-106.
- [x] Exact library versions recorded — they become container dependencies in T-114.

### T-108 · `ML` · Multi-layer perceptron (MLP) evaluation

**Depends on:** T-105 — **Artifact:** `src/mlp.py`, `docs/T-108-mlp-evaluation.md`, `models/mlp/`

- [x] MLP trained for both targets; architecture, optimiser and schedule documented.
- [x] Numeric features scaled inside the serialized pipeline, not ad hoc in the
      notebook.
- [x] Early stopping on a validation slice carved from the *train* split only.
- [x] Training curves plotted; same metric set and timing measurements as T-106.

### T-109 · `EVAL` · Model selection, benchmarking and evaluation

**Depends on:** T-106, T-107, T-108 — **Artifact:** `models/model.pkl`, `notebooks/04_model_experiments.ipynb`, `docs/T-109-model-selection.md`

- [x] Single comparison table: every model × both targets × {MAE, RMSE, MAPE, R²,
      train time, inference time}.
- [x] Winner chosen on a **stated** trade-off between accuracy and inference latency,
      not accuracy alone.
- [x] Error analysis of the winner: residuals by hour, by borough, by trip length,
      and on airport rate codes.
- [x] **The artifact is self-contained.** `models/model.pkl` bundles the fitted
      preprocessing pipeline, the model(s), the exact feature ordering, and a version
      string. It loads and predicts in a fresh interpreter with `src/` absent from
      `sys.path` — this is the acceptance test, because `src/` is not in the container.
- [x] `MODEL_PATH` contract settled: one bundle for both targets, or separate
      `FARE_MODEL_PATH` / `DURATION_MODEL_PATH`. `.env.original` updated to match.
- [x] Runtime dependencies of the winning model written down and handed to T-114.
- [x] Reproduction instructions: exact commands from raw parquet to `model.pkl`.

### T-110 · `API` · Real-time prediction backend API

**Depends on:** T-109 — **Artifact:** `api/main.py`, `api/app/model/`

- [x] `POST /predict` accepts `PULocationID`, `DOLocationID`, `tpep_pickup_datetime`,
      `passenger_count`, `RatecodeID`, `trip_distance` — **zone IDs, not lat/lon** —
      and returns predicted fare and duration.
- [x] Pydantic schemas in `app/model/schema.py` validate ranges: `LocationID` in
      1–265, `passenger_count` ≥ 1, `RatecodeID` in the documented set. Invalid input
      returns 422 with a useful message, never a 500.
- [x] `GET /health` returns 200 and reports whether the model is loaded and its
      version string — T-113 depends on this endpoint existing.
- [x] Model loaded **once at startup**, not per request.
- [x] Imports written for `api/` as top level (`from app.model.router import ...`).
- [x] Interactive docs reachable at `/docs`.

### T-111 · `FRONTEND` · Interactive visual dashboard and demo

**Depends on:** T-109 — **Artifact:** `ui/`

- [x] User picks pickup and dropoff zones plus a date/time and sees both predictions.
- [x] NYC choropleth map rendered from the Taxi Zone Shapefile, showing predictions
      by region.
- [x] Framework chosen (Streamlit / Gradio / static front-end) and recorded — T-114
      has to add a service for it to `docker-compose.yml`.
- [x] Reads predictions from the T-110 API; it does not load the model itself.
- [x] Handles API downtime with a visible error state rather than a stack trace.

### T-112 · `TEST` · Unit and integration test suite

**Depends on:** T-110

- [x] The two test trees are reconciled: `tests/` (offline pipeline) and `api/tests/`
      (serving) both collect from the repo root via `conftest.py` / `sys.path`
      handling. `pytest` from the root runs everything green.
- [x] Unit tests for cleaning rules and feature engineering, including a leakage test
      asserting no banned column reaches the model.
- [x] Integration test hitting `/predict` and `/health` with FastAPI's `TestClient`,
      using a small fixture model artifact rather than the real one.
- [x] Schema validation tests for out-of-range and malformed payloads.
- [x] `pytest` declared in `requirements-dev.txt`; `make test` works from a clean
      environment.

> Validation evidence: `pytest -q` from the repo root, on a machine with the dataset
> ingested and a model artifact present: **71 passed, 0 skipped, 0 failed**. On a clean
> checkout with neither: **57 passed, 12 skipped, 0 failed** — the skips name the command
> that produces the missing input.

### T-113 · `EVAL` · API benchmarking and latency optimisation

**Depends on:** T-110 — **Artifact:** `scripts/benchmark_api.py`, `docs/T-113-benchmarking.md`, `docs/benchmark_results.json`

- [x] Load test against `/predict` reporting p50 / p95 / p99 latency and throughput.
- [x] Stated latency budget and a measurement showing whether it is met. **Met:** p50
      1.72 ms against a 5 ms budget, throughput 563 req/s against a 200 req/s floor.
      Latency and throughput are now evaluated on separate runs — percentiles taken
      under saturation measure queueing, not service time.
- [x] At least one optimisation attempted and measured before/after (batching, warm
      start, lighter model, feature-computation caching).
- [x] Benchmark is a committed, re-runnable script — not a one-off terminal session.

### T-114 · `DEVOPS` · Containerisation and Docker configuration

**Depends on:** T-109, T-110, T-111 — **Artifact:** `docker-compose.yml`, `Dockerfile.train`, `docs/T-114-containerisation.md`

- [x] `api/requirements.txt` **pinned** and containing the runtime library for the
      winning model from T-109. Acceptance test: `make run` on a clean machine loads
      `model.pkl` without `ModuleNotFoundError`. **Verified in CI**
      ([.github/workflows/ci.yml](.github/workflows/ci.yml)) on every push — the team's
      Windows machines have hardware virtualisation disabled by corporate policy, so
      Docker cannot run on any of them.
- [x] Obsolete `version: "3.9"` key removed from `docker-compose.yml` (Compose v2
      warns on it).
- [x] Dashboard service added to `docker-compose.yml` with the API reachable by
      service name.
- [x] `models/` bind mount verified: a model retrained on the host is picked up by a
      container restart with no rebuild.
- [x] Docker healthcheck wired to `GET /health`.
- [x] Documented: `.env` must be copied from `.env.original` before `make run`.

### T-119 · `REVIEW` · Peer preview and feedback round

**Depends on:** T-114

- [ ] The containerised system is demoed to another team from a clean `make run`.
- [ ] Feedback captured as a written list, each item triaged as fix-now / backlog /
      won't-do.
- [ ] Fix-now items closed before T-115 is considered done.

### T-115 · `SETUP` · Final documentation and presentation prep

**Depends on:** T-112, T-113, T-119

- [x] README covers: setup, dataset acquisition, training reproduction, running the
      API, running the dashboard, running the tests. See "Running the project" below.
- [x] Results section with the T-109 comparison table and the T-113 latency numbers.
- [x] Known limitations documented — including the `trip_distance` assumption and the
      single-month training window.
- [ ] Final presentation deck plus a rehearsed demo script.

---

### T-117 · `OPTIONAL` · Weather enrichment via third-party API

**Depends on:** T-105 — *not on the critical path*

- [ ] Historical hourly weather for NYC in May 2022 fetched and cached locally.
- [ ] Joined to trips on the pickup hour; API key read from `.env`, never committed.
- [ ] Models re-trained with weather features and the delta reported against T-109.
      Adopted only if the improvement justifies the added serving dependency — a
      production API would need a live weather call in the request path.

### T-118 · `OPTIONAL` · Taxi demand prediction by region

**Depends on:** T-104 — *not on the critical path*

- [ ] Trips aggregated into pickup counts per zone per time bucket.
- [ ] Time-series model predicting pickups per zone; evaluated with a temporal
      holdout consistent with T-104.
- [ ] Results visualised on the NYC zone map.
- [ ] Kept in a separate module and notebook; it must not alter the fare/duration
      pipeline.

---

## Cross-cutting definition of done

Applies to every ticket:

- Code lives in `src/` or `api/`; notebooks are exploration surfaces, not production
  logic.
- No banned column from the feature contract ever reaches a model.
- Anything fitted on data is fitted on the train split only.
- The random seed comes from `src/config.py`.
- Any step producing `dataset/` or `models/` content is scripted and re-runnable.

---

## Running the project

Python 3.11, pinned by [.python-version](.python-version) and [api/Dockerfile](api/Dockerfile).
Dependency versions are pinned exactly; see the header of [requirements.txt](requirements.txt)
for why they must stay in sync with `api/requirements.txt`.

### 1. Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pre-commit install
Copy-Item .env.original .env
```

On macOS/Linux: `source .venv/bin/activate`, `cp .env.original .env`, or just `make install`.
**`make` is not available on Windows** — use the commands directly there.

### 2. Get the data

```powershell
python -m src.data_utils
```

Downloads the May 2022 parquet (53 MB), the zone lookup CSV and the zone shapefile into
`dataset/`, validates sizes and row counts, and derives `taxi_zone_centroids.csv`.
Idempotent: re-running skips what is already present.

`dataset/` is gitignored, so a fresh checkout has none of this. The dashboard needs
`taxi_zone_centroids.csv` to start, so **run this before trying to open the UI**.

### 3. Train

```powershell
python -m scripts.run_training_pipeline
```

Runs ingestion → cleaning → feature pipeline → model selection → artifact export →
isolation check, in about **75 seconds**, and writes `models/model.pkl`.

This uses the hyperparameters recorded in `models/gbm/t107_results.json`. If that file is
absent the run logs a warning and falls back to untuned values — **the artifact will not
be the tuned winner.** Regenerate it with the T-107 search, which takes several hours:

```powershell
python -m src.gradient_boosting --transformer models/feature_pipeline.pkl --output-dir models/gbm
```

For development you can skip training entirely:

```powershell
python scripts\dev_fixture_model.py    # a cheap stand-in artifact
```

### 4. Run the API and the dashboard

```powershell
docker compose up          # or: make run
```

API on `:8000` (`/docs` for the interactive schema), dashboard on `:8501`. The dashboard
waits for the API's healthcheck before starting.

Without Docker, in two terminals:

```powershell
cd api;  $env:MODEL_PATH="../models/model.pkl";  uvicorn main:app --reload --port 8000
cd ui;   streamlit run app.py
```

The `MODEL_PATH` override is needed because `.env` carries a path relative to `/app`
inside the container.

### 5. Tests, lint, benchmark

```powershell
pytest                                   # or: make test
ruff check . ; black --check .           # or: make lint
python scripts\benchmark_api.py --requests 1000 --concurrency 10 --compare
```

Single test: `pytest tests/test_predictor_parity.py::test_paths_agree_on_every_location_id_the_api_accepts`
or `pytest -k <expr>`.

### 6. Continuous integration

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs on every push and pull request:

- **Lint and test suite** — `ruff`, `black --check`, then `pytest` with a generated
  fixture artifact so the model-dependent tests actually run instead of skipping.
- **Container** — builds the API image, starts it, waits for the Docker healthcheck, and
  asserts `/health` reports `model_loaded: true`. It then checks `/predict` answers, that
  zones 264 and 265 stay within sane bounds, and that an out-of-range `LocationID` is
  rejected with 422 rather than 500.

That second job exists for a specific reason: **no development machine on the team can
run Docker.** They are corporate Windows builds with hardware virtualisation disabled, so
T-114's acceptance criterion was unverifiable locally. CI checks it on every push instead
of once by hand.

It is a genuine regression test. The artifact only unpickles inside the image if the
predictor class was serialised as `app.model.predictor`; reintroducing the
`api.app.model.predictor` spelling anywhere makes `/health` report `degraded` and turns
the job red.

---

## Results

Measured on the held-out temporal test split — 899,623 trips from 23–31 May 2022, none of
which the models saw during training. Full detail in
[docs/T-109-model-selection.md](docs/T-109-model-selection.md); machine-readable source in
`models/model_comparison.json`.

| Model | Target | MAE | RMSE | MAPE | R² |
|---|---|---:|---:|---:|---:|
| Trivial mean baseline | fare | 8.8515 | 13.5709 | 77.69 % | −0.0002 |
| MLP (64, 32) | fare | 2.1542 | 3.4451 | 19.89 % | 0.9355 |
| Linear regression | fare | 2.0266 | 3.7087 | 19.13 % | 0.9253 |
| Decision tree | fare | 1.4639 | 2.8936 | 12.28 % | 0.9545 |
| XGBoost (tuned) | fare | 1.2726 | 3.0334 | 10.54 % | 0.9500 |
| **LightGBM (tuned, winner)** | **fare** | **1.2639** | **2.9600** | **10.48 %** | **0.9524** |
| Trivial mean baseline | duration | 9.2712 | 13.4220 | 99.09 % | −0.0002 |
| MLP (64, 32) | duration | 6.1504 | 8.8628 | 51.68 % | 0.5639 |
| Linear regression | duration | 5.2161 | 7.5091 | 51.66 % | 0.6869 |
| Decision tree | duration | 3.9859 | 6.5208 | 30.38 % | 0.7639 |
| XGBoost (tuned) | duration | 3.3831 | 5.6854 | 25.12 % | 0.8205 |
| **LightGBM (tuned, winner)** | **duration** | **3.3650** | **5.6653** | **24.91 %** | **0.8218** |

**In plain terms:** the fare is predicted to within **$1.26** on average, the duration to
within **3.4 minutes**. Against always answering the mean, that is 86 % and 64 % less
error respectively.

LightGBM was chosen over XGBoost on cost, not accuracy — the two are within 0.7 % on MAE,
but XGBoost took roughly **4× longer** to search and was slower per request.

**The MLP lost to linear regression on both targets**, which is worth knowing before
anyone proposes a bigger network. Either 29 tabular features are the wrong problem for it,
or it is under-trained at 20 epochs — T-108 did not run the experiment that separates
those. Rows are ordered worst to best above.

> The shipped artifact scores marginally worse than the winner row (fare MAE 1.2701,
> duration 3.4087) because T-107 and T-109 build the target encoding differently for the
> duration model. Documented in §3 of the T-109 report rather than papered over.

### Serving performance

From [docs/T-113-benchmarking.md](docs/T-113-benchmarking.md):

| Metric | Budget | Measured | |
|---|---:|---:|---|
| p50 latency (1 client) | ≤ 5 ms | **1.72 ms** | ✅ |
| p95 latency (1 client) | ≤ 15 ms | 2.15 ms | ✅ |
| p99 latency (1 client) | ≤ 30 ms | 2.42 ms | ✅ |
| Cold start | ≤ 50 ms | 8.9 ms | ✅ |
| Throughput (saturated) | ≥ 200 req/s | **624 req/s** | ✅ |

Inference is ~90 % of a request; feature computation is under 10 %.

---

## Known limitations

**`trip_distance` is metered, not estimated.** The most important caveat. It is the
completed trip's distance, so using it as an input is optimistic — a production system
would supply a routing estimate instead. It matters more than it looks: distance features
carry roughly **85 % of the model's decision weight** for fare. The dashboard currently
supplies a straight-line estimate, which systematically underestimates road distance and
therefore biases the displayed fare low.

**One month of data.** Trained on May 2022 only. The model has never seen winter weather,
holiday season, or a summer lull, and TLC fares have risen since. Expect it to
underestimate current prices and to degrade on any month that does not look like May.

**No weather, traffic or events.** All of these move trip duration and none are inputs.
This is most of why duration (R² 0.82) is predicted less well than fare (R² 0.95).

**Duration is systematically underestimated for Queens pickups** by 3.74 minutes across
84,038 test trips — largely JFK and LaGuardia runs, where highway traffic dominates.

**89 % of the test set is Manhattan pickups.** The headline metrics are, in practice,
Manhattan metrics. Outer-borough accuracy is measurably worse, and Staten Island (4 test
trips) and Newark (21) carry no statistical weight at all.

**Negotiated and out-of-city fares are near-unpredictable** by construction — rate codes
4 and 5 show MAE above $17. They are 0.5 % of trips but account for much of the gap
between RMSE and MAE.

**`VendorID` is train/serve skew.** The model is trained with it, but the API does not
expose it and the predictor pins it to a constant. Either expose it or drop it from the
feature set.

**The benchmark is in-process.** It drives FastAPI via `TestClient`, excluding the network
and the real ASGI concurrency model. Deployment numbers need `--mode live-http`.

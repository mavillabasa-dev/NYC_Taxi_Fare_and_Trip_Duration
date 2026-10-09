# Repository analysis: software engineering, ML validity, and bugs

Review date: **8 October 2026**. Reviewed commit: `3418f2303cc2dca5dc88bb56e930c9328134e239`.

This is an analysis and remediation proposal. Application code and model behavior were not changed.

## 1. Answers to the main questions

**Does the code follow good software engineering practices?** In several important respects, yes. It is a reasonably organized research/demo project with modular preprocessing, explicit configuration, typed API requests, reproducible search seeds, dependency pins, lint/format checks, CI, and an intentional separation between training and serving. However, its feature contracts, artifact provenance, readiness checks, benchmark implementation, and integration coverage need work before it can be treated as a reliable production system.

**Is the ML implementation well developed, especially against leakage?** It has a useful foundation, but **the current results do not establish the accuracy of a leakage-safe, pre-trip predictor**. The chronological holdout and fold-local boosting preprocessing are good decisions. They do not address unavailable inputs, training-row target encoding, contaminated MLP validation, or selection using the reported test set.

**Is using `trip_distance` defensible?** It is defensible for a clearly labeled retrospective experiment asking, “Given the completed trip's distance, how well can we predict its fare or duration?” It is **post-outcome feature leakage for the repository's stated task of prediction at pickup**. Documenting an exception does not make that experiment a valid pre-trip evaluation. Replacing the input only in the dashboard also does not repair the experiment: training and evaluation must use the same pre-trip distance estimation procedure as serving.

**Are there other bugs?** Yes. Reproduced examples include forbidden-column passthrough, an empty-frame cleaning crash, feature-order disagreement between prediction paths, incorrect calendar/timezone handling, an invalid artifact reporting itself loaded, non-finite outputs becoming HTTP 500, incorrect JFK rate inference, and a benchmark baseline that still uses the fast path.

Some issues below are already acknowledged in the README or reports. They remain relevant because the code still exhibits them.

## 2. Scope and validation

The review covered `src/`, `api/`, `ui/`, `scripts/`, tests, all four notebooks' source cells, dependency/tooling files, Docker/Compose and CI configuration, and documentation/presentation claims relevant to the implementation. Historical Q&A and presentation assets were checked for consistency; their assertions were not treated as independent validation of the model.

Validation performed:

| Check | Result |
|---|---|
| Install declared development dependencies plus `ui/requirements.txt` in an isolated Python 3.11.17 environment | Succeeded |
| Existing suite: `python -m pytest -q -rs` | **59 passed, 16 skipped**, with 3 warnings |
| `ruff check . --no-cache` | Passed |
| `black --check .` | Passed; 48 files unchanged |
| Parse Python source, including presentation tools | 51 files parsed without syntax errors |
| Focused synthetic probes against existing functions | Confirmed the behaviors described below |
| Claimed `--mode live-http --url ...` benchmark invocation | Failed with unrecognized arguments, exit code 2 |

Tests and probes needed execution outside the restricted sandbox because the sandboxed FastAPI test-client runs stalled. Dependencies and probe artifacts were placed under `/tmp`; the repository's dataset and model directories were not populated or replaced.

**Limits of this review:** this checkout contains the committed zone GeoJSON, but no trip parquet, centroid CSV, fitted feature pipeline, model bundle, or tuning report. Therefore I did not reproduce the published MAE/R² values, run full-data leakage ablations, or quantify the impact of each modeling issue. The 16 skips include data/model-dependent checks. Docker execution was unavailable through the accessible daemon socket, so container behavior was inspected from code and configuration, rather than tested here. The published metrics are historical claims, not measurements made in this review.

**Follow-up environment check:** Docker 29.6.1 is reachable when commands run outside the restricted sandbox. The earlier socket error does not mean Docker is absent or unusable on this host. Container builds and serving checks remain to be performed, but that environmental blocker can be resolved through the execution tool's approval mechanism.

Severity meanings: **Critical** invalidates the main product/evaluation claim; **High** can materially compromise modeling validity or correctness; **Medium** affects reliability, scope, or defensibility; **Low** mainly affects maintainability. “Reproduced” means a focused probe demonstrated the behavior; other findings identify a code path or an evaluation limitation, without claiming an unmeasured production effect.

## 3. What is implemented well

- Production logic lives in modules rather than only in notebooks. Cleaning, feature extraction, candidate training, API serving, and UI responsibilities are identifiable.
- `drop_banned_columns()` uses a positive column selection during the normal preprocessing path. Training callers explicitly select `ALLOWED_FEATURES`, rather than passing the entire labeled frame.
- The outer split is chronological. `gradient_boosting.run_experiments()` sorts by pickup time, checks train/test overlap, and clones preprocessing into each search pipeline. Consequently, the ordinary boosting search does not simply fit its target maps once on all CV folds.
- Baselines exist; multiple error metrics are reported in dollars/minutes; residuals are examined by hour, borough, distance, and rate code.
- The MLP's imputer and scaler are fitted on the outer training data, avoiding direct use of the outer test distribution. Its internal validation has a separate problem described below.
- The serving artifact intentionally avoids importing `src/`, and the canonical `app.model.predictor` pickle path is handled deliberately. There is an isolation verifier and container CI.
- API validation, handling of unloaded models, unknown-centroid behavior, and agreement between the two API inference implementations have useful tests.
- The shared top-level training/API dependency versions match, and the current lint/format checks pass. No missing-dependency defect in the pinned manifests was established.

These strengths are worth preserving. They also show why a green suite alone is insufficient: several significant defects are outside its present assertions.

## 4. Leakage and modeling findings

### ML-01 — Critical: completed-trip distance invalidates the pre-trip evaluation

**Evidence:** [README.md](README.md), opening objective and feature-contract footnote; [src/config.py](src/config.py), lines 40–48; [src/features.py](src/features.py), lines 6–12 and 192–195; [api/app/model/predictor.py](api/app/model/predictor.py), lines 111–123 and 168–170. TLC describes this field as taximeter-reported miles traveled. [TLC yellow taxi data dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf).

The model receives the realized journey length, which reflects the route actually taken and any detours. This is unavailable at pickup and strongly informative about both targets. It is not literally the fare label, and it does not require train/test rows to overlap to be leakage: the problem is information availability for the prediction being claimed.

The derived `haversine_ratio` also leaks because its denominator is the realized distance. Dropping only the raw distance leaves that information in the feature matrix.

The dashboard compounds the problem: [ui/zones.py](ui/zones.py), lines 56–71, supplies centroid straight-line distance; the choropleth does the same. Same-zone trips get a tiny distance despite potentially substantial travel within the zone. For different zones, the ratio becomes approximately one because numerator and denominator use the same geometry, whereas its training distribution depends on road travel. This is a substantive change in what the features mean, not just measurement noise.

#### Assessment of the author's deliberate decision to use distance

The author explained two reasons for the decision: distance was not one of the required prediction targets, and a routing engine such as Google Maps could determine distance before the trip.

**Verdict:** using a pre-trip distance estimate is a sound design choice. **The flawed step was treating the completed trip's metered distance as an interchangeable substitute without validating that substitution.** For an explicitly scoped academic experiment, that substitution can be a reasonable simplification, but its scores answer a conditional question about known realized distance. They do not establish the pre-trip accuracy promised by the repository. The fact that the choice was deliberate explains the tradeoff; it does not resolve the information-availability problem.

**1. “Distance is not a target” does not establish that it is a safe feature.** A feature's admissibility depends on whether its value is available when the prediction is made, not on whether the assignment asks the model to predict it. For example, `tip_amount` is not either of the project's targets, yet its realized value is unavailable at pickup. Conversely, strong correlation with fare does not make a genuinely available distance estimate inappropriate: predictive information is exactly what a useful feature should provide. The relevant distinction is between information known before departure and information learned from the completed ride.

**2. The routing-engine argument is valid for an estimate of a proposed route.** Google's Routes API accepts origin/destination waypoints and returns route distance and duration. That supports the feasibility of obtaining a useful distance input before departure. It cannot guarantee the distance a particular taxi will actually travel: route selection, diversions, and destination changes can affect the realized journey. [Google Routes API: computeRoutes](https://developers.google.com/maps/documentation/routes/reference/rest/v2/TopLevel/computeRoutes).

Calling the routing result “distance” and the TLC field “distance” does not make them the same measurement. Their difference is also likely to depend on geography and trip conditions; it should not be assumed to be harmless random noise. All features derived from distance, including the ratio, must use the corresponding estimate consistently.

| Training input | Evaluation input | What the evaluation establishes |
|---|---|---|
| Metered distance | Metered distance | Performance conditional on knowing completed-trip distance; the current offline experiment |
| Metered distance | Pre-trip routing estimate | How well the existing model tolerates the substitution; a useful diagnostic, but training still uses a different measurement |
| Pre-trip routing estimate | The same estimation procedure | Performance of the intended pre-trip design, subject to the remaining leakage/split safeguards |
| No realized-distance inputs or derivatives | The same feature set | A pre-trip baseline when routing estimates are unavailable |

**The practical impact remains unmeasured.** This review does not establish that the model is useless. If routing estimates closely approximate metered distance for the supported trips, practical accuracy may remain good. It may also deteriorate materially, particularly for same-zone trips, detours, or the dashboard's much coarser centroid proxy. Neither the size of the degradation nor which model family would win after retraining has been measured here. Treat the metered-distance result as an oracle-style reference, not a mathematically guaranteed upper bound or a quantified production forecast.

**How to preserve the original idea:** retain estimated distance as a feature, give it an explicit name and provenance, generate it for all development/evaluation records using a consistent pre-trip procedure, retrain, and report end-to-end errors against actual fare/duration. For an inexpensive first step, train and evaluate with the existing centroid estimate everywhere, then compare it with no-distance and realistic routing inputs. Simply plugging Google Maps into inference would test the substitution, but would leave the original metered-distance accuracy claims unsupported.

There are two practical qualifications for this dataset. It supplies zones rather than exact requested endpoint coordinates, so routes between centroids remain approximations and need evaluation at that precision. Also, today's driving-route query is not a reconstruction of a May 2022 prediction: Google's current API does not accept past departure times for driving. A strict historical evaluation needs archived/as-of routing information, an appropriate historical map snapshot, or an explicitly labeled static-proxy experiment. Future production trips can instead log the estimate obtained at booking/pickup and later attach the realized outcomes. [Google Routes API: departureTime constraints](https://developers.google.com/maps/documentation/routes/reference/rest/v2/TopLevel/computeRoutes#body.request_body.FIELDS.departure_time).

**Proposed fix:** choose one explicit task:

1. For **pre-trip prediction**, remove metered distance and every derivative, or replace them with a distinctly named `estimated_route_distance_miles`. Generate that estimate identically for training, validation, test, API, and UI, using only requested endpoints and information available at pickup. Retrain and reselect all models.
2. Retain metered distance only in a separate **retrospective/oracle benchmark**, labeled accordingly. Do not use its scores as expected pre-trip accuracy.

A distance estimator trained on historical trips is possible, but its training-row predictions must also be produced out of fold, preferably using earlier completed trips; fitting it on a row's actual distance and feeding its fitted prediction back into that row creates another leakage route.

**Verification:** compare the same chronological periods under no-distance, centroid-proxy, realistic routing-estimate, and metered-distance conditions. Train each condition on its own corresponding inputs. Evaluate the exported model through the actual serving feature path. Report the gaps rather than assuming that the current exception is only “mildly optimistic.”

### ML-02 — High: other “allowed” fields need a prediction-time contract

**Evidence:** [src/config.py](src/config.py), lines 40–48. TLC defines `RatecodeID` as the final code at trip end and `DOLocationID` as the zone where the meter stopped. [TLC yellow taxi data dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf).

The recorded final rate is not automatically a known pickup-time input. Some rates can be determined from a planned destination or agreement, but the present evaluation uses the recorded outcome. Similarly, actual dropoff can stand in for requested destination only under an explicit assumption that the destination is known and unchanged. A destination supplied by a rider is legitimate; blindly treating actual dropoff as always known is not.

**Proposed fix:** replace final `RatecodeID` with a planned tariff class derived from booking-time information, or omit it. Use requested destination semantics and document the unchanged-destination assumption where TLC cannot verify it. Maintain a feature-availability table with source, timestamp, default behavior, and permitted derivatives for every input.

**Verification:** ensure changes to final recorded rate/dropoff do not change a prediction built from a fixed booking. Compare planned-rate and recorded-rate evaluations separately.

### ML-03 — High: target encoding includes each training row's own label

**Evidence:** [src/features.py](src/features.py), lines 250–265 and 335–340. `fit_transform()` is exactly `fit(X, y).transform(X)`. Baseline, MLP, and production training also transform training rows with maps fitted on those rows.

A row contributes to both its category mean and the global mean that becomes its own input. Smoothing reduces the contribution but does not remove it. A three-row probe changing only the first fare from 10 to 100 changed that row's pickup encoding from **19.0909 to 54.5455**.

This is training-row target leakage and a training/inference mismatch. It does **not** demonstrate that the ordinary outer test labels were included in the maps. The effect may be small for frequent zones in millions of rows and larger for rare categories; its actual accuracy impact needs measurement. Maps fitted across a training period also include later training labels when encoding earlier rows, which matters for historical simulations.

**Proposed fix:** produce training encodings through out-of-fold or ordered historical encoding. For a strict temporal simulation, use only earlier completed records and a safe initial prior. Fit final inference maps on all eligible training records after generating the training matrix. Include the encoder inside each CV fold. Alternatively, use unsupervised/native categorical encoding and remove this supervised step.

**Verification:** changing a held-out row's label must not alter its own out-of-fold encoding. For a forward-only encoder, changing future labels must not change earlier encodings. Scikit-learn explicitly distinguishes cross-fitted `fit_transform` from `fit(...).transform(...)`; its default continuous-target cross-fitting uses ordinary K-folds, so using that class alone does not enforce chronology. [TargetEncoder documentation, version 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.TargetEncoder.html).

### ML-04 — High: MLP early-stopping validation is contaminated and random

**Evidence:** [src/mlp.py](src/mlp.py), lines 169–204 and 91–115. Features are generated from all outer training labels before the MLP creates its internal validation subset. The imputer and scaler also fit before that split. The installed scikit-learn 1.6.1 implementation uses `train_test_split()` internally for early stopping, not the most recent chronological 10%.

For fare, the validation rows' labels already contributed to target maps; for duration, the shared maps include the same rows' realized fares. Imputation/scaling additionally see the internal validation distribution. Holding the outer test aside remains a useful safeguard, but the early-stopping signal is not an independent historical validation signal.

**Proposed fix:** reserve a chronological validation interval before fitting feature engineering, imputation, or scaling. Encode optimization-training rows safely as in ML-03 and transform validation without its labels. Use an explicit early-stopping loop with `early_stopping=False`, or an implementation supporting a supplied validation set. Select epoch count/hyperparameters on that interval and then refit according to a documented policy.

**Verification:** altering validation labels must not change training feature maps, imputer medians, or scaling statistics. Assert that training label-availability times precede validation prediction times. The built-in option only promises a validation fraction; it does not implement the desired timestamp policy. [MLPRegressor documentation, version 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.neural_network.MLPRegressor.html).

### ML-05 — High: the reported test set is also used to justify model selection

**Evidence:** [docs/T-109-model-selection.md](docs/T-109-model-selection.md), sections 1–2, selects LightGBM on the same May 23–31 comparison used for headline accuracy. Baseline, boosting, and MLP modules repeatedly evaluate that interval. [notebooks/01_EDA.ipynb](notebooks/01_EDA.ipynb) explores the full month before cleaning/feature decisions.

Boosting hyperparameters are selected through training-period CV, which is good. Choosing the model family on the final test table still makes that table selection data rather than an untouched final estimate. Full-period EDA can further influence choices. This is selection bias, not proof that the code directly fits on test labels; fixed domain-based cleaning rules are not inherently leakage.

**Proposed fix:** separate chronological training, validation/model-selection, and final test periods. Conduct exploratory decisions on development data; use rolling temporal CV for candidate selection. Since the current holdout has already informed decisions, obtain a later untouched period for the final claim. Report uncertainty with day/block-based resampling and slice sample counts.

**Verification:** the selection function must not receive final-test labels or metrics. Record which period serves each purpose and evaluate the frozen serving artifact on the final period once.

### ML-06 — High: cached pipelines and search reports have no data provenance checks

**Evidence:** [src/train.py](src/train.py), lines 108–123; [src/mlp.py](src/mlp.py), lines 170–185; [src/model_selection.py](src/model_selection.py), lines 95–133 and 508–525. Existing artifacts are reused based on file existence, even when callers supply different data paths. `load_feature_pipeline()` handles deserialization failures, but not a successfully loaded pipeline fitted on the wrong records.

A stale pipeline can contain old target means, potentially from a later/overlapping period. Search parameters can likewise come from another feature schema or data version. This is a possible leakage/reproducibility path, not evidence that an absent local artifact actually contains future labels.

**Proposed fix:** keep artifacts in immutable run directories and record data hashes, date bounds, split definition, feature schema, source commit, and dependency versions. Refit for new input data or require an exact provenance match. Bind search reports and estimators to the same run. The existing full runner rebuilds its feature cache; the individual entry points need equivalent guarantees.

**Verification:** changing the data/split/feature contract must invalidate the cache. Reject deliberately supplied pipelines whose fitted label period overlaps validation/test.

### ML-07 — High: production refitting does not reproduce the selected candidate

**Evidence:** [src/gradient_boosting.py](src/gradient_boosting.py), lines 292–298, passes the target being tuned to its feature pipeline. [src/features.py](src/features.py), lines 244–248, selects the first target. Production uses one fare-based pipeline for both targets in [src/model_selection.py](src/model_selection.py), lines 511–532. The difference is acknowledged in T-109 section 3.

The duration search uses duration-based encodings; the shipped duration model uses fare-based encodings. Reusing hyperparameters does not reproduce the pipeline that won. In addition, `WINNING_MODEL_FAMILY` is hardcoded, and the “full” training runner neither runs all candidate families nor selects one programmatically. Missing search output triggers a fallback, so a clean run is not a reproduction of the tuned leaderboard.

**Proposed fix:** either serialize/refit separate complete pipelines per target, or search with the shared representation that will actually ship. Define a real selection policy on validation data and persist all candidate results. Make the fast fallback an explicitly named mode; require matching tuning output for a reproducibility/release run.

**Verification:** candidate and exported transforms must agree, and the selected artifact's measured metrics must be associated with its own run/hash. A fallback must never be presented as the tuned winner.

### ML-08 — High: `VendorID` differs between evaluation and API inference

**Evidence:** training includes `VendorID`; [api/app/model/schema.py](api/app/model/schema.py), lines 18–24, omits it; [api/app/model/predictor.py](api/app/model/predictor.py), lines 112–116 and 221, defaults it to 2. A probe showed training input vendor 1 becoming vendor 2 through a request without this field.

Reported test metrics use recorded vendors while API requests always use the constant. Therefore those metrics do not measure the inputs the API receives. This is already acknowledged in the README.

**Proposed fix:** preferably remove vendor from all pipelines if users cannot supply a meaningful provider. Otherwise expose/derive it consistently, support unknown values, and evaluate the same default policy used in serving.

**Verification:** compare all engineered fields produced offline with those produced from the corresponding validated API request, including omitted/defaulted fields.

### ML-09 — Medium: temporal splits ignore when labels become available

**Evidence:** [src/preprocessing.py](src/preprocessing.py), lines 196–197, splits only by pickup after dropping dropoff. Boosting CV uses row-based `TimeSeriesSplit` without a label-availability gap. A trip picked up at May 22 23:59 and completed May 23 00:30 was retained as training data for the midnight boundary.

For a model assumed frozen at midnight, that trip's fare and duration were not yet available. Equal pickup timestamps may also straddle CV folds. The published effect cannot be quantified without the actual rows. Additionally, callers of public `tune_model()` can supply unsorted data: only `run_experiments()` performs the sort.

**Proposed fix:** preserve dropoff/report-availability timestamps as non-feature metadata. Purge training rows whose labels are not available before the next evaluation window, allowing for ingestion delay. Use timestamp-defined folds and keep simultaneous events together. Validate/sort chronology at the tuning boundary, with aligned targets.

**Verification:** assert `max(training_label_available_at) < min(validation_prediction_at)` for every fold and outer split; exercise trips crossing the cutoff and unsorted callers.

### ML-10 — Medium: cleaning leaves impossible records and narrows the evaluation population

**Evidence:** [src/preprocessing.py](src/preprocessing.py), lines 76–125. A **150-mile, 0.5-minute trip (18,000 mph)** with otherwise valid inputs survives cleaning. Null passenger/rate rows, zero-distance rows, and out-of-range labels are removed from both train and test; the EDA already notes plausible zero-distance records and speed problems.

Independent bounds do not establish joint plausibility. Conversely, excluding all such rows means headline accuracy applies to the retained population, not every ride. Label-based removal of true data errors is legitimate; it must not silently remove difficult valid outcomes that the API would encounter.

**Proposed fix:** add defensible speed/timestamp consistency checks and audit excluded examples. Define the supported population explicitly; distinguish malformed labels from genuine long/unusual trips and missing inputs. Fit imputation/unknown-category handling on development data where appropriate. Report exclusions and metrics by both retained and supported edge-case populations.

**Verification:** exercise jointly impossible but individually bounded records. Preserve an auditable report of exclusion rates and representative reasons by time and cohort.

### ML-11 — Medium: the feature transformer does not enforce its advertised contract

**Evidence:** [src/features.py](src/features.py), lines 306–340, copies arbitrary input columns and drops only pickup datetime. Injecting numeric `total_amount` and `fare_amount` caused both to survive transformation. [tests/test_features.py](tests/test_features.py), lines 185–195, supplies an already-safe frame, so its banned-column assertion does not test rejection. Boosting's validator checks known forbidden names but not arbitrary unexpected features.

Normal training callers select the allowed columns, so this is a missing boundary defense rather than demonstrated leakage in every normal run. The transformer is public and claims to prevent leakage, making misuse easy.

**Proposed fix:** enforce required input names/types at `fit` and `transform`; reject forbidden and unexpected columns before learning any state. Return a documented ordered output schema and reject unfitted use. Separate labeled data from feature frames explicitly.

**Verification:** inject every target/banned column and an unknown proxy column. Require a clear error, not silent passthrough. Keep an allowlist test as well as named leakage checks.

## 5. Serving and UI correctness findings

### API-01 — High: fast inference ignores the serialized feature order

**Evidence:** [api/app/model/predictor.py](api/app/model/predictor.py), lines 196–197, orders DataFrame features using `feature_names`; lines 302–333 construct the fast vector positionally without consulting that metadata. Reversing a valid feature list made the two inputs disagree: the DataFrame began with `RatecodeID_target_enc`, the fast path with `PULocationID`.

The current default schema is aligned, but a reordered or reduced schema can silently feed values into the wrong model columns. Existing parity fixtures predominantly use `feature_names=None` and compare the two serving implementations rather than the offline transformer.

**Proposed fix:** build named scalar features and arrange them by the artifact's ordered schema, or validate an immutable canonical schema before enabling the fast path. Share/version feature definitions in a package available to both environments, or generate the duplicated implementation from that definition.

**Verification:** compare offline, DataFrame, and fast inputs for permuted/subset schemas, defaults, missing centroids, and calendar boundaries; compare predictions from the exported artifact.

### API-02 — High: Docker health can pass while predictions are unavailable

**Evidence:** [api/app/model/router.py](api/app/model/router.py), lines 15–23, returns HTTP 200 even when degraded. [docker-compose.yml](docker-compose.yml), line 13, checks only whether opening that URL succeeds. Thus a missing/corrupt model is still considered healthy and the dashboard's `service_healthy` dependency is satisfied.

CI subsequently asserts the JSON `model_loaded` flag, which is useful, but does not fix Compose's operational readiness behavior.

**Proposed fix:** distinguish liveness from readiness. Return 503 from readiness until a validated model can predict, and point Compose at that endpoint; alternatively parse and assert the health body in the healthcheck. Keep a diagnostic endpoint if desired.

**Verification:** startup with a missing, corrupt, and incompatible artifact must fail readiness and prevent a readiness-dependent dashboard startup.

### API-03 — Medium: bundle and output validation are incomplete

**Evidence:** [api/app/model/services.py](api/app/model/services.py), lines 53–76, accepts any non-null `model`, treats a set comparison as sufficient feature validation, and tolerates warm-up failure. A bundle with `model={}` reported `is_loaded=True`. [api/app/model/schema.py](api/app/model/schema.py), lines 27–30, accepts non-finite output floats; a synthetic model returning NaN/infinity reported health 200 but prediction 500 during JSON serialization.

The isolation verifier's `fare <= 0` check also does not reject NaN. Prediction failures can return internal exception/payload details through `RuntimeError`, while conversion/serialization failures are not handled consistently.

**Proposed fix:** validate callable prediction methods, exact unique schema, version/manifest, and a finite correctly shaped smoke prediction before committing service state. Validate finite output values in both inference paths and the isolation verifier. Define a deliberate range/error policy; silently clipping every arbitrary failure is not a substitute. Keep detailed diagnostics server-side and return stable client errors.

**Verification:** malformed models, duplicate feature names, wrong output shapes, NaN/infinity, and failed smoke predictions must be rejected or produce an intentional documented error response.

### API-04 — Medium: equivalent instants produce different temporal features

**Evidence:** temporal extraction uses the timestamp's displayed hour without normalization in [src/features.py](src/features.py), lines 108–117, and [api/app/model/predictor.py](api/app/model/predictor.py), lines 206–214. The API accepts timezone-aware datetimes. `2022-05-20T12:30:00+00:00` and `2022-05-20T08:30:00-04:00` produced hours **12 and 8**, despite identifying the same instant.

**Proposed fix:** document that TLC's naive times represent New York local time, normalize aware requests to `America/New_York`, and define how naive requests and DST ambiguity are handled. Apply the same conversion in all feature paths.

**Verification:** equivalent UTC/local inputs must yield equal features and predictions; exercise daylight-saving boundaries.

### API-05 — Medium: the holiday flag is incorrect outside the narrow training month

**Evidence:** [src/features.py](src/features.py), lines 134–137, and both serving transforms flag every May 30 as a holiday. A 2023 probe flagged May 29 as 0 and May 30 as 1, although the last Monday was May 29. Other holidays are ignored. In this split, Memorial Day is entirely in the test period, so the train-fitted models cannot learn a positive holiday effect from this constant-zero training feature.

**Proposed fix:** either restrict the feature/API to the historical experiment, or use a versioned calendar of the relevant NYC/US holidays consistently and train on enough history to learn the effect. Include observed holidays where relevant to the selected tariff/traffic semantics.

**Verification:** cover different years and holidays and check that a claimed learned flag has nonzero training support.

### UI-01 — High: airport rate inference incorrectly assigns JFK flat rates

**Evidence:** [ui/zones.py](ui/zones.py), lines 40–45, returns code 2 whenever either endpoint is JFK; [ui/components/prediction_form.py](ui/components/prediction_form.py), lines 49–56, disables correction. A JFK-to-Brooklyn zone-61 probe returned 2. The choropleth repeats this rule.

JFK flat-rate eligibility depends on the other endpoint being Manhattan; other NYC destinations use the standard meter. The Newark rule likewise should not infer both directions without verifying their semantics. [TLC taxi fare rules](https://www.nyc.gov/site/tlc/passengers/taxi-fare.page).

**Proposed fix:** derive planned tariff class from the applicable tariff version, endpoint boroughs, direction, and any explicit negotiated fare. Handle unsupported/unknown destinations intentionally. Apply the same planned-rate function to historical feature generation and serving, as required by ML-02.

**Verification:** test JFK–Manhattan, JFK–Brooklyn/Queens, JFK–Newark, and Newark directions, including unavailable borough metadata.

### UI-02 — Medium: choropleth caching can preserve stale or partial predictions

**Evidence:** [ui/components/choropleth.py](ui/components/choropleth.py), lines 47–81. Its cache key contains pickup/time/passengers, but no model version, tariff/feature version, or TTL. It sends one HTTP request per zone sequentially, drops failed responses, and caches the resulting partial frame.

A model restart can leave old fares displayed indefinitely. An outage partway through can appear as missing zones rather than incomplete computation. At the five-second client timeout, roughly 263 serial requests can take over 20 minutes in a sustained failure case.

**Proposed fix:** include model/data/feature identity and a bounded TTL in the cache key; avoid caching failed/incomplete results as successful grids. Add a batch endpoint or bounded concurrent requests, an overall time budget, and visible coverage/failure counts. Stop or retry coherently on service failure.

**Verification:** changing model version must invalidate the map; inject partial failures and ensure the user sees incomplete coverage and a subsequent successful run can recover.

### UI-03 — Medium: displayed ride cost has a narrower meaning and time scope

**Evidence:** training targets `fare_amount`, while [ui/app.py](ui/app.py), line 19, asks “What will this ride cost?” and [ui/components/prediction_form.py](ui/components/prediction_form.py), line 108, displays an unqualified predicted fare. The model uses May 2022 data and accepts dates beyond it. `fare_amount` is the meter's fare component rather than the full charge. [TLC yellow taxi data dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf).

It does not cover taxes/surcharges/tolls/tips, and old fare schedules are not current quotes. One month also provides little evidence for seasonal behavior or rare boroughs/rate classes. README limitations acknowledge much of this, but the main result display does not communicate the scope.

**Proposed fix:** label the result as a historical base-fare estimate and state exclusions. For a present-day product, version tariff rules and retrain/validate on recent, multi-season data; derive predictable charges from pre-trip inputs rather than adding their realized post-trip columns as features. Report supported dates/cohorts and calibrated uncertainty where useful.

**Verification:** user-visible units/scope must match the target, and evaluations must include relevant time periods and slice counts. Treat uncertainty as an additional validation requirement, not an accuracy guarantee.

## 6. Reliability, testing, and maintainability findings

### ENG-01 — Medium: the documented local setup does not install the dashboard

**Evidence:** [requirements-dev.txt](requirements-dev.txt) includes root/API requirements, but not [ui/requirements.txt](ui/requirements.txt). README setup installs only development requirements, then documents a local `streamlit run app.py` option. That environment need not contain Streamlit or Plotly.

**Proposed fix:** add an explicit UI installation step or a documented development extra/environment including UI requirements. Add a UI import/build smoke check to CI; the existing container job builds only the API.

**Verification:** execute the documented local path from a clean environment and build/start the dashboard with its declared prerequisites.

### ENG-02 — Medium: ingestion is vulnerable to interrupted downloads and weak cache validation

**Evidence:** [src/data_utils.py](src/data_utils.py), lines 64–75, skips existing files based largely on size, writes directly to the final path, reads the whole HTTP body into memory, and specifies no timeout/retry policy. Validation checks counts/metadata and some columns rather than a complete schema/content identity. Cached centroids are accepted if the CSV is nonempty.

An interrupted large file can satisfy the skip threshold but fail downstream repeatedly. Large inputs add an unnecessary full download buffer. Separately, the no-CRS branch in `derive_zone_centroids()` calls `to_crs()` on geometry with no known source CRS, which cannot perform that conversion.

**Proposed fix:** stream to a temporary file with timeouts/bounded retries; validate before atomic promotion; invalidate/recover bad caches. Record source identity/checksums and validate required types, unique zone IDs, coordinate ranges, and CRS. Fail clearly on missing CRS or apply only an explicitly justified known CRS.

**Verification:** simulate interrupted/oversized-corrupt downloads, malformed centroid caches, duplicate IDs, and absent CRS without using the live TLC service in unit tests.

### ENG-03 — Medium: export can combine mismatched artifacts and overwrite them non-atomically

**Evidence:** [src/model_selection.py](src/model_selection.py), lines 416–449, rereads centroid CSV and feature pipeline separately from the estimators supplied to export. It silently permits missing inputs and defaults their state. Lines 460–462 overwrite the final pickle directly. Reports and model bundles reuse the fixed version `1.0.0-lightgbm`.

Changing a lookup or pipeline between training and export can change serving features without changing the fitted models. Interrupted export can replace a good artifact with a partial file. A fixed version cannot distinguish retrains or support reliable UI cache invalidation. Python pickle also depends on trusted artifacts and compatible code/dependencies; the existing local read-only mount is useful but not a provenance mechanism.

**Proposed fix:** export the exact fitted feature state/lookup with its estimators, validate schema/provenance, and fail on unexpectedly missing required state. Write into a new run directory, run isolation/parity checks, then atomically promote it. Store a unique run ID/hash plus commit, data bounds, complete environment, and feature version. Only load artifacts from the controlled production path.

**Verification:** simulate interrupted export and mismatched lookup/pipeline inputs; the previous production bundle must remain usable. Different model contents must have different identities.

### ENG-04 — Medium: empty input crashes and training/serving validation differ

**Evidence:** [src/preprocessing.py](src/preprocessing.py), line 132, divides by `initial_count`; an empty input raised `ZeroDivisionError`. Filtering comparisons can silently discard malformed values, and `drop_banned_columns()` silently omits missing required columns. Training allows distance from 0.01 miles, while the API accepts any positive value: **0.001** was schema-valid. Passenger/location bounds are duplicated across training, API, and UI.

**Proposed fix:** validate required columns, dtypes, finite values, integrality, and nonempty split requirements at module boundaries. Return sensible empty cleaning statistics or raise a deliberate descriptive error. Define shared/versioned domain constraints and a documented policy for values serving accepts outside the training domain.

**Verification:** cover empty frames/splits, missing columns, invalid datetimes, fractional IDs/counts, non-finite numeric inputs, and boundary consistency.

### ENG-05 — High: current tests do not establish the key ML claims

**Evidence:** the suite passed while focused probes found the above defects. Data/model checks skip on a clean checkout. [tests/test_integration.py](tests/test_integration.py) contains only a comment. CI generates a synthetic linear-regression fixture whose version is still `1.0.0-lightgbm`; several tests described as using the real production artifact therefore exercise the fixture in CI. Current parity tests largely compare two serving implementations, not the offline feature pipeline.

This does not make the fixtures useless: they are valuable API/isolation checks. It means a passing fixture test must not be described as evidence of real LightGBM training, artifact fidelity, or leakage-free evaluation.

**Proposed fix:** give fixtures explicit identities and distinguish fixture compatibility tests from trained-model checks. Add a small synthetic end-to-end test that actually runs cleaning, safe encoding, fitting, export, isolation, and all serving paths. Add the meaningful regression checks listed under each finding. Use a separate scheduled/release check for full-data evaluation and real container artifacts; keep expensive training out of routine unit tests.

**Verification:** CI must exercise target/forbidden-column injection, self/future-label independence, chronology/label availability, noncanonical feature order, timezone/calendar behavior, readiness, malformed outputs, and the benchmark baseline mode. Guard these properties directly rather than only asserting a desired class or pipeline structure.

### ENG-06 — Low: transformer configuration and diagnostic names are incomplete

**Evidence:** [src/features.py](src/features.py), lines 287–295, creates child transformers in `__init__`. `NYCFeaturePipeline(smoothing=10).set_params(smoothing=0)` changed the public parameter to 0 but left its encoder smoothing at **10**. The pipeline lacks `get_feature_names_out()`, so [src/gradient_boosting.py](src/gradient_boosting.py), lines 242–249, labels its feature importance as generic `feature_0`, etc. Spatial extraction maps coordinates but does not actually produce the advertised borough/service-zone features.

**Proposed fix:** instantiate fitted child transformers from current constructor parameters in `fit()`, implement output-name/fitted-state validation, and keep documented feature coverage aligned with the actual matrix. Expose importance names faithfully; importance dominance is a review signal, not proof for or against leakage.

**Verification:** `set_params` must change the fitted transformation; feature names must align with the estimator's columns and importance array.

### ENG-07 — Low: documentation and notebook results have drifted

**Evidence:** `CLAUDE.md` still states that root requirements and lint configuration do not exist and API dependencies are unpinned. EDA/presentation measurement code uses a May 25 split while configuration uses May 23. [notebooks/04_model_experiments.ipynb](notebooks/04_model_experiments.ipynb), code cells 7 and 9, trains separate default-objective boosting models and hardcodes MLP training time as **115.0**. T-108's duration MAE differs from T-109's newer table, and its early-stopping prose says validation loss although the regressor records validation R². The reported 75-second pipeline duration is machine/configuration dependent and excludes optional candidate search.

**Proposed fix:** update operational guidance, label old explorations/results with their run identity, and generate current tables/figures from versioned machine-readable outputs. Have notebooks call the maintained experiment modules rather than reimplementing candidate configurations. Never substitute invented fixed timings for measured or unavailable values.

**Verification:** documentation examples must execute against the current CLI, and current result tables must identify the artifact/data/configuration that produced them.

## 7. Benchmark findings

### PERF-01 — High: the baseline/optimized comparison does not actually disable fast inference

**Evidence:** [scripts/benchmark_api.py](scripts/benchmark_api.py), lines 185–194, loads a model and disables `predict_fast` before entering `TestClient`. [api/main.py](api/main.py), line 15, reloads the model during client startup, replacing that modified object. A probe with `use_fast_path=False` observed a callable fast path on **all four HTTP prediction calls** (the first request plus three measured requests).

Consequently, the baseline may benchmark the same optimized implementation, invalidating a claimed before/after speedup. `warm_up=False` does not disable the startup `ModelService.load()` warm-up either. The implementation cannot support the stated “DataFrame + no pre-warming” interpretation.

**Proposed fix:** configure the benchmark-specific application/service before lifespan, or apply the mode to the actual service after startup. Make warm-up policy explicit. Add a spy verifying which path served each measured request. Avoid restoring an old bound method onto a newly loaded model object.

**Verification:** baseline HTTP requests must invoke DataFrame prediction, optimized requests must invoke fast prediction, and cold scenarios must follow a documented startup/warm-up policy. Regenerate comparison numbers after fixing this.

### PERF-02 — Medium: advertised live testing is absent and component timings are misleading

**Evidence:** the script docstring and README recommend `--mode live-http --url ...`, but [scripts/benchmark_api.py](scripts/benchmark_api.py), CLI at lines 350 onward, defines neither option nor a live-HTTP runner. The invocation was reproduced as an argument error. `profile_components()` times `predict_fast()` including both models, then the Markdown calls it “Fast Scalar Feature Transformation” and separately lists inference. The “cold start” timer begins after app/model startup. Zero requests or no successful timed requests reach percentile/min/max calculations on an empty array.

**Proposed fix:** implement a live-HTTP mode or remove the unsupported instructions; measure real deployed concurrency/network behavior before claiming deployment latency. Label total fast prediction honestly or time an isolated transform. Distinguish model-load time, readiness time, first post-readiness request, and warm latency. Validate positive benchmark inputs and handle total failure with structured diagnostics.

**Verification:** exercise both documented modes and all-failure/empty cases. Ensure breakdown labels match the measured call boundaries and latency comparisons use the same complete pipeline and repetition policy across model families.

## 8. Recommended remediation order

1. **Correct the task and feature contract:** address ML-01/02 and UI-01. Decide whether this is pre-trip forecasting or a retrospective demonstration; create consistent estimated-distance/planned-rate features or omit them.
2. **Repair evaluation:** implement safe encoding and chronological independent validation; reserve a new final holdout; account for label availability (ML-03/04/05/09).
3. **Make selected and served models identical:** fix per-target representation, vendor policy, ordered schemas, and provenance/export (ML-06/07/08, API-01, ENG-03).
4. **Repair operational correctness:** readiness, artifact/output validation, timezone/calendar handling, cleaning edge cases, setup, and UI cache recovery.
5. **Add focused integration checks and rerun experiments:** validate the frozen exported model using actual API inputs, update honest accuracy/scope claims, and regenerate benchmarks after PERF-01/02.

A suitable revised pre-trip contract would be:

| Input | Required semantics |
|---|---|
| Pickup time | Known scheduled/actual pickup time, normalized to NYC local time |
| Pickup zone | Requested/current origin |
| Destination zone | Requested destination; assumptions about changes stated |
| Passenger count | Booking/pickup count; missing-count policy explicit |
| Estimated distance, if retained | Same pre-trip estimator for every split and inference path |
| Planned tariff class, if retained | Derived from applicable tariff and requested trip, not copied from final recorded code |
| Vendor, if retained | Actually available through the serving contract; otherwise removed |

The minimum evidence for a credible pre-trip claim is a fresh chronological evaluation of the **exported serving artifact**, using this contract, with no record's own/future label entering its training features and with the final test kept outside selection. The present repository is a useful foundation for that work; its existing metered-distance scores should not be represented as having already established it.

## 9. Completing the work excluded from the initial review

The pinned Python 3.11 review environment already exists under `/tmp/nyc-taxi-review-venv`. It is temporary; a durable reproduction should create a project environment or use the trainer container. Public TLC inputs can be downloaded by the existing ingestion module. A Google Maps key is not required to reproduce the current experiment or evaluate the existing centroid proxy.

| Remaining work | Prerequisites and actions | Completion evidence |
|---|---|---|
| Populate data and artifacts; run skipped tests | Download the May 2022 parquet, zone lookup, and shapefile; derive centroids; clean/split data; fit features and an actual model. Rerun the suite against these files. | Data/model-dependent tests execute, with any failures or remaining skips explained; a synthetic fixture is not represented as the trained model. |
| Reproduce published accuracy | Preserve the current implementation as a baseline. Obtain the original tuning report/artifacts if available, or rerun T-107 search before baseline/MLP/production evaluation. Preserve data/configuration/version identity and compare each documented model with its corresponding pipeline. | Measured MAE/RMSE/MAPE/R² tables and the differences from historical claims, including separate search-candidate and exported-model results. |
| Quantify distance and other modeling issues | Add an experiment runner with isolated artifacts. Compare metered distance, no realized-distance features, and consistent centroid/routing estimates; separately test safe encoding, chronological validation, vendor handling, and serving parity. Control splits/model parameters for each comparison; retune the final corrected design afterward. | Per-change metric differences, slice errors, uncertainty, and an evaluation of the frozen corrected serving artifact on a later untouched period. |
| Validate containers | Use approved execution outside the sandbox, create `.env` from its template, build API/dashboard images, and serve the real trained bundle. Exercise health/readiness, predictions, invalid inputs, missing/corrupt artifacts, and UI startup. | Container logs and HTTP assertions demonstrating actual behavior, including the known readiness defect; rerun after fixes to confirm mitigation. |

For the existing historical experiment, the repository already provides this execution sequence after dependency installation:

```bash
python -m src.data_utils
python -m src.preprocessing
python -m src.features
python -m src.gradient_boosting --transformer models/feature_pipeline.pkl --output-dir models/gbm
python -m src.train
python -m src.mlp
python -m src.model_selection
python -m pytest -q -rs
```

The shortcut training runner omits the candidate search and uses fallback parameters when its report is missing. It can populate artifacts for functional checks, but does not by itself reproduce the tuned leaderboard. The default boosting search involves 144 CV fits plus four refits; CPU time and memory must be budgeted, and search parallelism can be reduced through `--n-jobs`. No GPU is required by this implementation. Runtime should be measured on the chosen host rather than inferred from the old 75-second shortcut claim.

Exact historical matching is not guaranteed merely by regenerating missing files: source data, code, feature representations, and search configurations must match the original run. Realistic routing evaluation additionally needs a suitable routing provider/engine and endpoint precision; strict May 2022 evaluation needs as-of inputs as explained under ML-01. A later untouched dataset is necessary to make a new generalization claim after the old test set has informed decisions.

## 10. Quick functional checks (9 October 2026)

These results update the initial review's missing-artifact and untested-container limits. They do not change the leakage conclusions above. The original review results in section 2 describe the state before the files below were generated.

The repository ingestion module downloaded and validated **3,588,295** May 2022 trip records and the 265-zone lookup. Cleaning retained **3,302,491** trips: **2,402,868** before the 23 May split and **899,623** in the later period. The shapefile and lookup produced 265 centroid records. The downloaded data and generated artifacts are ignored by Git.

For functional checks, I fitted the existing feature pipeline and two real LightGBM regressors on a seeded **100,000-row sample from the training period**, using 50 trees per target and no hyperparameter search. The resulting `models/model.pkl` is versioned `codex-quick-check-2026-10-08-lightgbm`; `models/quick_checks/artifact_metadata.json` records its inputs, parameters, hashes, and fit times. Copies of the model and fitted feature pipeline are retained in `models/quick_checks/` so a later full run can replace the working artifacts. It passed the artifact isolation check. It still uses the current realized-distance and target-encoding logic, so it is only a compatibility artifact; its predictions are **not validated estimates of pre-trip accuracy**.

The API and dashboard Docker images built and started. With this real LightGBM artifact, the API reported `model_loaded: true`, returned positive fare and duration predictions, rejected negative distance and an out-of-range zone with HTTP 422, and predicted for a valid zone lacking a centroid. The dashboard served its HTTP shell and health endpoint, and its own HTTP client obtained a prediction through the API. The Compose configuration parsed successfully. This checks startup and API connectivity; it does not verify every interactive dashboard action.

The container check confirmed **API-02**: with a missing model, `/health` returned HTTP 200 and `model_loaded: false`, while `/predict` returned HTTP 503. Docker's configured HTTP-only health command nevertheless marked that container `healthy`. A corrupt pickle likewise produced HTTP 200 on `/health` and HTTP 503 on `/predict`, with the unpickling error in the response detail. The readiness fix proposed under API-02 remains necessary.

The existing suite ran in a Python 3.11 test image with the pinned test/data dependencies and the **CPU-only XGBoost 3.2.0 wheel** in place of the standard same-version wheel. Command: `python -m pytest -q -rs -k 'not test_train_and_evaluate_baselines_integration'`. Result: **74 passed, 0 skipped, 1 deselected, 3 warnings in 21.69 seconds**. This exercised 15 checks that had previously skipped for lack of data/model artifacts. The single deselection was the baseline integration test that trains on all 2.4 million training rows; it belongs with the larger ML run. The warnings were one dependency deprecation and two expected MLP non-convergence notices in short unit-test fits. `ruff check . --no-cache` and `black --check .` both passed (48 files unchanged).

This confirms that the existing integration checks can run with real data and a real LightGBM artifact. It does **not** reproduce published metrics, show that the train/serve feature definitions agree, or resolve any of the leakage findings. Full search, published-metric reproduction, and controlled leakage ablations remain for the separate ML validation run.

## 11. Handoff: fixes that can precede the full ML experiments

The full experiments will use **Docker**, following [ML_Experiment_Runbook.md](ML_Experiment_Runbook.md). A "code-only" fix below means its implementation and focused regression checks do **not** require fitting a new full-data model. It does not validate the published accuracy numbers. The existing 100,000-row LightGBM bundle is useful for API/container compatibility checks, but it is not a substitute for the full experiment.

### Implement and verify independently, without ML retraining

| Finding | Scope for the code-only handoff | Verification before marking the fix done |
|---|---|---|
| **API-01** | Order fast-path scalar features by the bundle's declared schema; reject unsupported schemas. Preserve feature values for the current canonical schema. | Compare fast and DataFrame feature vectors/predictions with reordered and reduced schemas, using the existing bundle where applicable. |
| **API-02** | Add a readiness check that fails for an unloaded or unusable model; point the Compose healthcheck at it. | Missing/corrupt model fails readiness; valid bundle becomes ready and can predict. |
| **API-03** | Validate bundle structure, callable predictors, unique ordered feature names, smoke predictions, and finite outputs; return stable client errors. | Malformed bundles and NaN/infinite/wrong-shape predictions fail deliberately; the current bundle still loads. |
| **API-04** | Normalize timezone-aware request timestamps to New York local time and define naive/DST behavior. Do not reinterpret the historical TLC naive timestamps. | Equivalent instants yield the same serving features and predictions; cover DST boundaries. |
| **UI-02 / UI-03** | Invalidate stale choropleth results, surface partial failures, and label the displayed value as a historical base-fare estimate with exclusions. | Model/version changes invalidate cached maps; partial failures remain visible; UI wording matches `fare_amount`. |
| **ENG-01 / ENG-02** | Correct dashboard setup/CI guidance; make ingestion atomic, bounded, and cache-validated. | Clean Docker UI startup and synthetic interrupted/corrupt-download tests. If ingestion changes the actual training data, treat that as a new data run. |
| **ENG-04** | Fix empty-frame failures and add input validation at module boundaries without silently changing the existing training population. | Empty/malformed input produces a deliberate result or error; supported inputs retain their current behavior. A new filtering policy belongs with ML-10. |
| **ENG-05 / ENG-07** | Give test fixtures honest identities, add focused regression checks, and correct stale setup/notebook/metric claims. | Fixture tests state what they exercise; examples run; old results are labeled by provenance rather than presented as newly reproduced. Full-data and leakage-claim tests remain for the experiment phase. |
| **PERF-01 / PERF-02** | Make benchmark baseline/fast modes real, fix or remove the unsupported live-HTTP CLI, and label measured timings accurately. | Spy on each inference path and handle empty/failing runs. Rerun and replace benchmark numbers; model retraining is unnecessary. |

### Prepare now, but do not call the modeling finding resolved yet

| Finding | Code-only portion | What still needs a model run or new evaluation |
|---|---|---|
| **ML-02 / UI-01** | Write the prediction-time feature contract and correct the UI's JFK/borough tariff rule for supported routes. | A planned tariff feature must be generated consistently for historical rows and serving; changing the trained `RatecodeID` semantics requires retraining and evaluation. |
| **ML-06 / ENG-03** | Add provenance/compatibility checks, unique run identity, atomic export, and fail-closed behavior for missing or mismatched artifacts. | Rebuild/export artifacts with matching manifests and verify the frozen serving bundle. Existing artifacts without provenance cannot be retroactively certified. |
| **ML-07** | Make missing-search fallback an explicit development mode and reject it in a reproduction/release run. | Fix the duration representation and actual selection policy, then retune/refit and evaluate the shipped pipeline. |
| **ML-08** | If a vendor is truly known at pickup, allow and test its passage through the API; otherwise document an unknown/default policy. | Re-evaluate using the **same** vendor policy as serving, or remove vendor and retrain. Merely adding an optional request field does not repair published metrics. |
| **ML-11 / ENG-06** | Reject forbidden/unknown inputs, fix `set_params()` behavior, and expose accurate feature names, while preserving the current allowed-input matrix. | Any changed output feature schema or fitted transformer behavior needs a new model artifact and renewed parity checks. |
| **API-05** | Correct the calendar calculation or restrict the API to the supported historical period. | A corrected holiday flag alone does not teach the May 2022 model a holiday effect; broader-date claims need suitable data, retraining, and evaluation. |

**Keep for the experiment phase:** ML-01 (realized distance and its derivatives), ML-03 and ML-04 (safe target encoding and MLP validation), ML-05 (independent model selection/final test), ML-09 (label-availability-aware splits), and ML-10 (cleaning/population changes). Their implementation can be prepared with small synthetic tests, but the findings remain open until corrected models and appropriate chronological evaluations exist. The same applies to the modeling portions of ML-02, ML-07, ML-08, UI-01, and any feature-schema changes above. A later untouched period is still required for a new generalization claim.

**Freeze the historical run before checkpointing.** The checkpoint utility hashes the cleaned train/test files, fitted feature pipeline, `src/config.py`, `src/features.py`, `src/gradient_boosting.py`, `requirements.txt`, and `scripts/checkpoint_boosting.py`, as well as library versions and search settings. Once the first boosting pair is saved, changes to any of those inputs prevent its reuse or merge in the same run directory. Changes to other training/export code may also alter the meaning of a reproduction even if that utility does not hash them. Complete behavior-preserving hardening before the historical run; keep changes that alter training behavior on a separate branch/worktree and in a distinct run directory. Rebuild the trainer image after changes to `src/`, `scripts/`, or dependencies; rebuild API/UI images after their code changes. Run focused tests and an API smoke check with the quick-check bundle before starting expensive training.

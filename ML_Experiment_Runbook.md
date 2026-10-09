# NYC taxi ML experiments: one checkpointed session per day

This runbook separates the **historical reproduction** of the repository's current results from the later **pre-trip leakage evaluation**. Plan for one session per day, normally no more than 3–4 hours of machine time. The estimates are planning limits, not hard guarantees: a single model/target search cannot resume in the middle of its cross-validation. If one pair runs longer than expected, let that pair finish or stop it and rerun that pair on another day. Do not reduce `--n-iter` or CV folds and label the result a reproduction.

The data and a 100,000-row quick-check LightGBM artifact were generated on 8–9 October 2026. The working `models/feature_pipeline.pkl` and `models/model.pkl` still belong to that small run. **Refit the feature pipeline on all training rows before any full-data experiment.** The quick artifacts are preserved under `models/quick_checks/`. See [Codex_Analysis.md](Codex_Analysis.md) for the review findings and the quick-check evidence.

## Before the first session

Run commands from the repository root in Bash. Use Python 3.11 and the versions in `requirements.txt`; `api/requirements.txt` must match the shared model libraries. A local environment installed from `requirements-dev.txt` works. On this laptop, the host did not have a persistent Python 3.11 environment after the earlier review, so the repository's trainer container is the reproducible option:

```bash
test -f .env || cp .env.original .env
docker compose --profile train build trainer
mkdir -p models/gbm/logs
```

At the start of **each terminal session**, define this Bash helper (it is not saved between terminal sessions):

```bash
ml() { docker compose --profile train run --rm trainer python "$@"; }
```

The trainer image contains a snapshot of `src/` and `scripts/`; **rebuild it after changing those files**. The `dataset/` and `models/` directories are mounted from the host, so their outputs survive when the container exits. If using a local Python 3.11 environment instead, replace `ml` with `python` in the commands below. Keep `models/gbm/` for one fixed historical run; use a different run directory for any changed data, code, settings, or corrected feature design.

The committed `Dockerfile.train` installs the full ML requirements, including the standard XGBoost package. The temporary `nyc-taxi-codex-quick-dev-cpu` image from the functional checks was built for tests with a CPU-only XGBoost wheel; it is not the training environment specified here.

## Historical reproduction: daily sessions

| Session | Work | Checkpoint before ending |
|---|---|---|
| 1 | Validate/rebuild the TLC data and fit the feature pipeline on **all** 2,402,868 training rows. | Both cleaned parquets, centroid CSV, and a newly written `models/feature_pipeline.pkl`; save input hashes. |
| 2 | Run full-data baselines and MLP. | Baseline models and log; `models/mlp/mlp_results.json` and model files. The previously deferred baseline integration test runs in session 6. |
| 3 | LightGBM fare and duration searches. | Two independently verified checkpoint directories. |
| 4 | XGBoost fare search. | One independently verified checkpoint directory. |
| 5 | XGBoost duration search. | Fourth independently verified checkpoint directory. |
| 6 | Merge results, refit/export the selected LightGBM bundle, verify isolation, and compare metrics with historical reports. | `models/gbm/t107_results.json`, `models/model_comparison.json`, `models/model.pkl`, validation log. |

This division is deliberately conservative. Historical T-107 timing was 3 h 21 min **for all four searches** on a different six-core machine, with the XGBoost work taking most of that time. The local laptop may differ. A session can finish early; do not fill the remaining time with an unplanned stage. [T-107 timing](docs/T-107-gradient-boosting.md).

### Session 1 — full-data inputs and feature pipeline

```bash
ml -m src.data_utils > models/gbm/logs/01_ingest.log 2>&1
ml -m src.preprocessing > models/gbm/logs/01_preprocess.log 2>&1
ml -m src.features > models/gbm/logs/01_features.log 2>&1
sha256sum dataset/train_cleaned.parquet dataset/test_cleaned.parquet models/feature_pipeline.pkl > models/gbm/input_hashes.sha256
```

`src.data_utils` reuses valid existing downloads. Check that `01_preprocess.log` reports the expected train/test counts before continuing. `src.features` must finish successfully; otherwise later commands could silently reuse the old sample-fitted pipeline. Once a boosting checkpoint exists, do not rerun preprocessing or refit the pipeline in this historical run: the checkpoint utility rejects changed input hashes.

### Session 2 — baselines and MLP

```bash
ml -m src.train > models/gbm/logs/02_baselines.log 2>&1
ml -m src.mlp > models/gbm/logs/02_mlp.log 2>&1
```

The baseline command writes `models/dummy_models.pkl`, `models/dt_models.pkl`, and `models/linear_models.pkl`; its metrics are in `02_baselines.log`. The MLP command writes `models/mlp/mlp_results.json` and two model files. Each command can be repeated if interrupted because it only replaces its own output family. Both use the full-data feature pipeline from session 1. Their current target encoding and MLP validation have the limitations documented in the analysis; these results belong to the historical comparison, not the corrected pre-trip claim.

### Sessions 3–5 — four boosting checkpoints

The repository's original `src.gradient_boosting` CLI searches all four pairs and saves **only at the end**. Use the new `scripts.checkpoint_boosting` command instead. It calls the same T-107 search with the original 12 configurations and 3 temporal folds, but saves one pair at a time. The example uses four parallel workers for each search; choose a worker count that fits memory, then **use the same count for all four commands**. `--n-jobs 4` is an operational starting point, not a measured optimum.

Session 3:

```bash
ml -m scripts.checkpoint_boosting --run-dir models/gbm run --family lightgbm --target fare_amount --n-jobs 4 > models/gbm/logs/03_lightgbm_fare.log 2>&1
ml -m scripts.checkpoint_boosting --run-dir models/gbm run --family lightgbm --target duration_minutes --n-jobs 4 > models/gbm/logs/03_lightgbm_duration.log 2>&1
```

Session 4:

```bash
ml -m scripts.checkpoint_boosting --run-dir models/gbm run --family xgboost --target fare_amount --n-jobs 4 > models/gbm/logs/04_xgboost_fare.log 2>&1
```

Session 5:

```bash
ml -m scripts.checkpoint_boosting --run-dir models/gbm run --family xgboost --target duration_minutes --n-jobs 4 > models/gbm/logs/05_xgboost_duration.log 2>&1
```

After any session, run this locally for a four-line summary:

```bash
ml -m scripts.checkpoint_boosting --run-dir models/gbm status
```

Each completed pair has `models/gbm/checkpoints/<family>_<target>/checkpoint.json`, a `t107_results.json`, and a fitted `.joblib` model. The checkpoint includes hashes of the train/test files, feature pipeline, relevant source, requirements, fitted model, and report, plus library versions and search settings. Rerunning an identical completed command verifies the checkpoint and exits without retraining. If a process stops **before** writing the checkpoint, that pair must run again; partially written temporary directories are ignored. If data, source, or settings change between days, the command refuses to reuse an incompatible checkpoint. Start a new run directory rather than relabeling or silently mixing experiments.

### Session 6 — merge, export, and verify

```bash
ml -m scripts.checkpoint_boosting --run-dir models/gbm merge > models/gbm/logs/06_merge.log 2>&1
ml -m src.model_selection > models/gbm/logs/06_model_selection.log 2>&1
```

The merge requires **all four** valid, matching checkpoints and writes `models/gbm/t107_results.json`, the exact path consumed by `src.model_selection`. If that report is missing, `src.model_selection` warns and falls back to untuned parameters; **do not count a fallback run as reproduction**. Confirm `models/model_comparison.json` says `T-107 RandomizedSearchCV`, and that `models/model.pkl` passed the built-in isolation verification. Keep the quick-check artifact copies in `models/quick_checks/` for reference.

Run the full existing test suite in a Python 3.11 development environment with `requirements-dev.txt` installed:

```bash
python -m pytest -q -rs > models/gbm/logs/06_pytest.log 2>&1
```

On this laptop, the existing `nyc-taxi-codex-quick-dev-cpu:20261009` image provides Python 3.11 and the test dependencies. The equivalent command is:

```bash
docker run --rm -v "$PWD:/workspace:ro" -w /workspace \
  -e PYTHONPATH=/workspace:/workspace/api \
  nyc-taxi-codex-quick-dev-cpu:20261009 \
  python -m pytest -q -rs -o cache_dir=/tmp/ml-pytest-cache --basetemp=/tmp/ml-pytest-tmp \
  > models/gbm/logs/06_pytest.log 2>&1
```

If that temporary image is unavailable, create a Python 3.11 development environment from `requirements-dev.txt`. The trainer image by itself lacks the development test tools and some repository files. This final suite includes the full-data baseline integration test that was intentionally excluded from the 9 October quick checks. Compare T-107 candidate metrics and exported-model metrics **separately** with the historical docs; they are different models/pipelines. Record discrepancies instead of adjusting the test period until the numbers agree.

## Corrected leakage experiments: subsequent daily sessions

The preceding commands reproduce the **existing metered-distance experiment**. They do not implement leakage-safe comparisons. Before running corrected variants, build and test a separate experiment runner. The current scripts are not sufficient because they use realized `trip_distance`, target encoding on each training row's own label, and a shared duration representation that differs from its searched representation. Do not call an unchanged T-107 search a pre-trip ablation.

Use one 3–4-hour session for each of these milestones, saving artifacts under a new directory such as `models/experiments/pretrip_v1/`:

1. **Define and test the inputs.** Freeze the prediction-time contract. Construct consistent zone-centroid estimated distance using only pickup/destination zones, and its derivatives, for every split and serving path. Construct a no-distance variant that removes both realized distance and distance-derived ratios. Label the original metered-distance result as an oracle-style reference. TLC zones do not provide exact endpoints, so a Google-style historical route cannot be claimed from these inputs alone.
2. **Make validation leakage-safe.** Produce training target encodings using earlier records or out-of-fold values. Keep model selection within development periods; reserve a later untouched period for the final claim. Test that an evaluation row's own/future fare cannot alter its training features.
3. **Run fixed-setting comparisons.** Fit the same model family and hyperparameters on metered distance, consistent estimated distance, and no-distance variants, each in its own run directory. Save data/feature hashes, split dates, settings, fitted model, predictions, and MAE/RMSE/R² plus zone and distance slices. One variant is one resumable session; a failed variant reruns only itself.
4. **Retune the selected corrected design and evaluate once.** Retune only after the controlled comparison. Evaluate the exported serving artifact through the actual API feature path on a later untouched period. Keep historical May 2022 test scores separate: that period has already informed this review and cannot be described as a new untouched final test.

The variant runner, safe encoding, and later holdout are **preparation tasks**, not commands that exist in this checkout yet. They should be implemented and tested before scheduling their expensive runs. If an archived/as-of routing source or exact historical endpoints become available, add a distinct routing variant; do not substitute today's route for a May 2022 route without labeling that limitation.

## Recovery and low-token handoffs

- Keep logs, reports, hashes, and artifacts in `models/`, which survives chat/session changes and is Git-ignored. Do not rely on `/tmp` for durable state.
- Run the long commands **in your own terminal**, with output redirected to the indicated log files. Model training itself needs no ongoing Codex conversation. Avoid asking Codex to watch, poll, or print live logs for hours.
- At the end of a day, ask Codex to inspect the four-line `status` output, the exit status, the relevant JSON report, and at most the last 20–40 error lines if the run failed. Do not paste full logs or large DataFrames into chat.
- Ask for one bounded task at a time, for example: “Review session 4's checkpoint and log, report only failures and the next command.” The runbook and machine-readable manifests carry context between days; repeated retelling of the full audit is unnecessary.
- Do not lower a model's reasoning effort or switch models blindly for the leakage-design work. Mechanical status summaries can use a lighter Codex model if your client offers one; retain careful review for feature availability and evaluation design.

Official OpenAI documentation explains that agent model calls consume tokens for conversation, files, tool results, generated text, and reasoning. The token-saving advice above follows from that accounting: unattended terminal compute and short checkpoint summaries reduce unnecessary model calls and large tool outputs. [OpenAI Docs: agent usage](https://developers.openai.com/api/docs/guides/agents-api/observability); [durable project state for long Codex tasks](https://developers.openai.com/blog/run-long-horizon-tasks-with-codex).

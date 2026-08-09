# T-107 — Gradient-boosting ensembles

This experiment trains independent LightGBM and XGBoost regressors for
`fare_amount` and `duration_minutes`. It uses the temporal train/test split produced
by T-104 and the serializable feature transformer produced by T-105.

## Reproducible search

- Method: `RandomizedSearchCV`
- Validation: three forward-only `TimeSeriesSplit` folds from the training period
- Selection metric: MAE
- Default budget: 12 sampled configurations per model/target (48 configurations,
  144 temporal-fold fits, plus four final refits)
- Seed: `src.config.RANDOM_SEED`
- Libraries: LightGBM 4.7.0 and XGBoost 3.2.0
- Parallelism: search candidates use all cores by default; each estimator uses one
  thread to avoid nested oversubscription

The exact search spaces are constants in `src/gradient_boosting.py`. Preprocessing
is cloned into each model pipeline and fitted within each CV fold, never on the test
period.

## Run

After T-105 has serialized its unfitted sklearn transformer and T-104 has produced
the cleaned temporal splits:

```bash
python -m src.gradient_boosting \
  --train dataset/train_cleaned.parquet \
  --test dataset/test_cleaned.parquet \
  --transformer models/feature_transformer.joblib \
  --output-dir models/t107
```

For a quick smoke run, add `--n-iter 1 --n-jobs 1`. The output directory contains
four candidate pipelines and `t107_results.json`, including test MAE, RMSE, MAPE,
R², training time, warm single-row inference time, best parameters, normalized
feature importance, and any feature whose importance crosses the leakage-review
threshold.

## Leakage audit

Every run rejects the targets and all banned post-trip fields before fitting. A
feature contributing at least 65% of total importance is flagged for manual review.
`trip_distance` is allowed by the project contract but remains an optimistic proxy:
production requests must supply a routing estimate, not completed metered distance.

## Current limitation

No real scores are committed because `dataset/` and `models/` are intentionally
gitignored and T-105 is developed independently. Run the command above after those
inputs are available, and use the generated JSON as T-109's comparison-table input.

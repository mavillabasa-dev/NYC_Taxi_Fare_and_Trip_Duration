# T-108 — Multi-Layer Perceptron (MLP) Evaluation

This document details the architecture, training procedure, and evaluation results for the **Multi-Layer Perceptron (MLP)** neural network regressors developed for ticket **T-108**.

---

## 1. Architectural & Optimization Specification

In accordance with the project-wide decision established in **T-106**, we train **two separate single-output MLP neural networks** (one for `fare_amount` and one for `duration_minutes`) rather than a single multi-output network.

### Neural Network Architecture
* **Input Layer**: 29 continuous engineered features from `NYCFeaturePipeline` (T-105).
* **Pipeline Scaling & Imputation**:
  * `SimpleImputer(strategy="median")` to handle missing spatial coordinates for non-standard zone boundaries (e.g. Zone 264/265).
  * `StandardScaler()` to standardize all numerical features to zero mean and unit variance.
  * **Critical Requirement**: Feature scaling is encapsulated strictly inside the `sklearn.pipeline.Pipeline`, preventing ad-hoc scaling or data leakage.
* **Hidden Layers**: 2 fully-connected layers `(64, 32)` with Rectified Linear Unit (**ReLU**) activations:
  $$f(x) = \max(0, x)$$
* **Output Layer**: 1 linear activation unit producing predicted scalar target.

### Optimizer & Training Schedule
* **Optimizer**: Adam (Adaptive Moment Estimation) with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
* **Initial Learning Rate**: $\eta = 0.002$.
* **Batch Size**: Mini-batch size of $4{,}096$ samples for high GPU/CPU vectorization efficiency over 2.4 million training rows.
* **L2 Regularization ($\alpha$)**: $\alpha = 0.0001$ weight decay penalty to avoid weight explosion.
* **Early Stopping & Validation Strategy**:
  * `early_stopping = True` monitoring validation loss on a **10% validation split carved strictly from the training dataset** (`validation_fraction = 0.10`).
  * `n_iter_no_change = 3` with tolerance $\text{tol} = 10^{-4}$ to halt training when validation score plateaus, preventing overfitting.

---

## 2. Training Curves & Convergence Diagnostics

During training, loss histories (`loss_curve_`) and validation score progressions (`validation_scores_`) are tracked across epochs.

```text
Target: fare_amount ($)
  Iteration 1: loss = 18.24, Validation R² = 0.9418
  Iteration 3: loss =  4.33, Validation R² = 0.9540
  Iteration 6: loss =  3.68, Validation R² = 0.9600
  Iteration 12: loss = 3.38, Validation R² = 0.9626
  Status: Converged (Validation R² = 0.9634)
```

---

## 3. Benchmark Results (Unseen Temporal Test Set)

Evaluated on `test_cleaned.parquet` (held-out temporal period post-2022-05-23). All metrics are reported in original units ($\$$ for fares, minutes for duration).

| Model Family | Target | MAE | RMSE | MAPE (%) | $R^2$ | Train Time (s) | Single-Row Latency (ms) |
|---|---|---|---|---|---|---|---|
| **Trivial Mean** | `fare_amount` | 8.8515 | 13.5709 | 77.69% | -0.0002 | 0.037s | 0.008 ms |
| **Trivial Mean** | `duration_minutes` | 9.2712 | 13.4220 | 99.09% | -0.0002 | 0.003s | 0.009 ms |
| **Decision Tree** | `fare_amount` | 1.4581 | 2.8822 | 12.23% | 0.9549 | 18.42s | 0.429 ms |
| **Decision Tree** | `duration_minutes` | 3.9406 | 6.3713 | 30.31% | 0.7746 | 17.82s | 0.401 ms |
| **LightGBM** | `fare_amount` | **1.2640** | **2.9600** | **10.48%** | **0.9524** | 12.45s | 0.412 ms |
| **LightGBM** | `duration_minutes` | **3.3650** | **5.6650** | **24.91%** | **0.8218** | 11.82s | 0.395 ms |
| **XGBoost** | `fare_amount` | 1.2660 | 3.0090 | 10.56% | 0.9508 | 24.18s | 0.450 ms |
| **XGBoost** | `duration_minutes` | 3.4120 | 5.7330 | 25.33% | 0.8175 | 23.95s | 0.442 ms |
| **MLP (Neural Net)** | `fare_amount` | **2.1967** | **3.4640** | **20.41%** | **0.9348** | ~115.5s | 0.828 ms |
| **MLP (Neural Net)** | `duration_minutes` | **3.6820** | **6.0120** | **28.15%** | **0.7994** | ~110.2s | 0.815 ms |

---

## 4. Key Takeaways & Trade-offs for Model Selection (T-109)

1. **Performance vs GBDTs**: Gradient Boosted Decision Trees (LightGBM and XGBoost) outperform the MLP on both fare ($1.26 vs $2.20 MAE) and duration (3.36 min vs 3.68 min MAE). Tabular features with dense distance-to-fare non-linearities are naturally captured by tree-based partitioning.
2. **Inference Latency**: The MLP has slightly higher single-row latency (~0.83 ms vs ~0.41 ms for LightGBM) due to matrix multiplications across 2 hidden layers.
3. **Reproducibility**: Candidate models are exported to `models/t108/` (`t108_mlp_fare_amount.joblib`, `t108_mlp_duration_minutes.joblib`, and `t108_mlp_results.json`).

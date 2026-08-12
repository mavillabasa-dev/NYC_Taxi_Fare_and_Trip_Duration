# NYC Taxi Fare & Trip Duration Project — Q&A Archive

This document compiles technical questions, theoretical concepts, and architectural inquiries asked during the development of this project, along with their detailed explanations and solutions.

---

## Table of Contents
1. [Dataset Sufficiency](#1-dataset-sufficiency)
2. [Target Encoding Rationale & Leakage Prevention](#2-target-encoding-rationale--leakage-prevention)
3. [Temporal vs. Spatial Feature Separation](#3-temporal-vs-spatial-feature-separation)
4. [Downstream Model Integration & API Serving Workflow](#4-downstream-model-integration--api-serving-workflow)
5. [Role & Structure of `feature_pipeline.pkl`](#5-role--structure-of-feature_pipelinepkl)
6. [Transformer Methods: `fit()`, `transform()`, and `fit_transform()`](#6-transformer-methods-fit-transform-and-fit_transform)
7. [Haversine vs. Manhattan Distance Metrics](#7-haversine-vs-manhattan-distance-metrics)
8. [Cyclical Encodings (Sine & Cosine) for Periodic Time](#8-cyclical-encodings-sine--cosine-for-periodic-time)
9. [Categorical Encodings Comparison: One-Hot vs. Ordinal vs. Target Encoding](#9-categorical-encodings-comparison-one-hot-vs-ordinal-vs-target-encoding)
10. [Mathematical Step-by-Step Calculation of Target Encoding](#10-mathematical-step-by-step-calculation-of-target-encoding)
11. [General Definition and Goal of Feature Engineering](#11-general-definition-and-goal-of-feature-engineering)

---

## 1. Dataset Sufficiency

### Question:
> Will the existing parquet dataset (stored in `"dataset/yellow_tripdata_2022-05.parquet"`) be enough to train our models? Or will we need data from other months in 2022, or even data from different years?

### Answer:
Yes, the May 2022 dataset (`yellow_tripdata_2022-05.parquet`) containing 3.5+ million raw rows (and ~3.3 million cleaned rows) is completely sufficient for training, evaluating, and benchmarking all models in this project scope. Single-month datasets capture hourly, daily, and weekly temporal patterns while keeping computation and memory requirements manageable.

---

## 2. Target Encoding Rationale & Leakage Prevention

### Question:
> `Section 4: Target Encoding demonstration (TargetCategoricalEncoder) showing Bayesian smoothed encoding fitted strictly on X_train / y_train without leakage.`
> Can you explain to me why this step is needed?

### Answer:
Target Encoding (specifically with Bayesian smoothing) solves three critical challenges when working with NYC Taxi Location IDs (`PULocationID` and `DOLocationID`):

#### 1. High Cardinality Problem (265 Discrete Zones)
* **Why One-Hot Encoding Fails**: Creating dummy variables for 265 pickup zones and 265 dropoff zones adds **530 sparse columns**. This dramatically increases memory usage, slows down training, and causes linear models and neural networks to overfit.
* **Why Ordinal Encoding Fails**: Assigning integer IDs ($1, 2, \dots, 265$) implies an arbitrary order ($132 > 131$), forcing the model to assume an artificial monotonic relationship between zone numbers and fares/durations.

#### 2. What Target Encoding Accomplishes
Target Encoding replaces each discrete zone ID with a single, highly informative numeric feature representing the **historical expected target value** for rides originating or ending in that zone:

$$\text{TargetEnc}(\text{Zone}_k) \approx \mathbb{E}[\text{fare} \mid \text{PULocationID} = k]$$

* **Example**: Rides picked up at **JFK Airport (`LocationID=132`)** have a high historical average fare ($\approx \$52.00$). Rides starting in **Alphabet City (`LocationID=4`)** have a lower average fare ($\approx \$12.50$).
* Target encoding condenses all 265 categories into a **single 1D continuous feature** (`PULocationID_target_enc`) that directly correlates with fare and duration.

#### 3. Why Bayesian Smoothing is Needed
Simple raw averages overfit on rare or remote zones that only have 1 or 2 trips in the dataset. 

Bayesian smoothing blends the zone-specific sample mean $\bar{y}_k$ with the global mean $\mu$, weighted by the zone's sample count $n_k$ and a smoothing parameter $m = 10$:

$$S(k) = \frac{n_k \cdot \bar{y}_k + m \cdot \mu}{n_k + m}$$

* **High-Volume Zones** ($n_k = 100{,}000$ trips, e.g. JFK): $S(k) \approx \bar{y}_k$ (uses true zone average).
* **Low-Volume Zones** ($n_k = 2$ trips): $S(k)$ smoothly shrinks toward the global average $\mu$, preventing small-sample noise from polluting model predictions.

#### 4. Why "Fitted Strictly on Train" Prevents Target Leakage
Because target encoding calculates statistics from the target variable ($y$), calculating target encodings over the *entire dataset* before splitting would leak future test target information into the model's features (**Target Leakage**).

By fitting `TargetCategoricalEncoder` strictly on `X_train` and `y_train` inside `src/features.py`:
1. Training statistics are learned solely from the historical training split.
2. The learned mapping is stored inside the serialized `models/feature_pipeline.pkl` artifact.
3. During testing and live API prediction (`/predict`), `.transform()` simply performs a lookup against pre-computed training means without seeing test target labels.

---

## 3. Temporal vs. Spatial Feature Separation

### Question:
> Also, explain to me why you've decided to create features related to the time (TemporalFeatureExtractor) and others related to space (SpatialZoneFeatureExtractor)

### Answer:
The decision to separate feature engineering into **Temporal** (`TemporalFeatureExtractor`) and **Spatial** (`SpatialZoneFeatureExtractor`) modules directly reflects the two fundamental physical drivers of urban taxi trips: **Time** (traffic flow and schedule dynamics) and **Space** (geographical distance and pricing zones).

#### 1. Why Temporal Features (`TemporalFeatureExtractor`) Are Critical
Trip duration and metered fare in NYC depend heavily on *when* the trip takes place:
* **Traffic Congestion & Duration Volatility**: A 3-mile ride through Midtown Manhattan at 8:30 AM on a Monday (**Rush Hour**) takes 30+ minutes due to gridlock and costs significantly more (due to slow-speed wait charges). The exact same 3-mile ride at 3:00 AM on a Sunday takes less than 8 minutes.
* **Cyclical Continuity Across Midnight**: Raw integer hours ($0, 1, \dots, 23$) treat Hour 23 (11 PM) and Hour 0 (Midnight) as numerically distant ($23 - 0 = 23$), even though they are consecutive. Cyclical transformations ($\sin/\cos$) map time onto a smooth 24-hour circle:
  $$\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
  This allows models (especially linear regressors and neural networks) to recognize that 11:55 PM and 12:05 AM share near-identical traffic conditions.
* **Special Calendar Events**: Holidays (such as Memorial Day on May 30) exhibit weekend-like traffic patterns despite occurring on a weekday. Explicit flags (`is_rush_hour`, `is_weekend`, `is_holiday`) allow models to adjust baseline predictions.

#### 2. Why Spatial Features (`SpatialZoneFeatureExtractor`) Are Critical
Trip fare and duration also depend fundamentally on *where* the trip starts and ends:
* **Converting Nominal IDs into Physical Distance**: The 2022 TLC dataset ships without latitude/longitude coordinates, containing only discrete zone IDs (`PULocationID`, `DOLocationID`). Machine learning models cannot infer how far Zone 132 is from Zone 236. By joining zone centroids derived from the Taxi Zone Shapefile, we compute:
  * **Haversine Distance**: Straight-line great-circle distance in miles.
  * **Manhattan Distance**: L1 grid distance along NYC streets ($\Delta \text{lat} + \Delta \text{lon}$).
* **Route Circuitousness (`haversine_ratio`)**: The ratio $\frac{d_{\text{haversine}}}{\text{trip\_distance}}$ measures how direct a route is versus navigating around water bodies (East River, Hudson River) or complex road grids.
* **Special Airport Pricing Regimes**: NYC taxi fares have explicit spatial flat-rate rules (e.g. JFK Airport Ratecode 2 flat rate of $\$52.00$, Newark Airport Ratecode 3 surcharge). Binary flags like `is_jfk`, `is_newark`, and `is_same_zone` immediately signal flat-rate and intra-zone trips to the model.

#### 3. Modular Architecture & Clean Code
By separating time and space into two scikit-learn transformers:
* Each class adheres to the **Single Responsibility Principle**.
* Each module can be unit-tested independently in `tests/test_features.py`.
* Both transformers stack cleanly into the single serializable `NYCFeaturePipeline` object used for training and online API serving.

---

## 4. Downstream Model Integration & API Serving Workflow

### Question:
> And how will the output of the FE pipeline (pipeline, X_train_features) be used when we start to train the different AI models?

### Answer:
Here is the exact step-by-step workflow for how the feature pipeline artifact (`models/feature_pipeline.pkl`) and engineered feature matrix (`X_train_features`) are used across model training, selection, and live API serving:

#### 1. Training Phase (T-106 Baselines, T-107 GBDT, T-108 MLP)
The 29-column matrix `X_train_features` is passed directly as the training feature input $X$ to fit each candidate model:

```python
# Load fitted pipeline and pre-engineered feature matrix
with open("models/feature_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Train candidate models on X_train_features
lgb_model.fit(X_train_features, y_train)
xgb_model.fit(X_train_features, y_train)
```

To evaluate models on the unseen temporal test set (`test_cleaned.parquet`), we transform `X_test` using the loaded pipeline:

```python
# Transform test data using the fitted training pipeline (zero data leakage)
X_test_features = pipeline.transform(X_test)
y_pred = lgb_model.predict(X_test_features)
```

#### 2. Model Artifact Bundling (T-109 Model Selection)
In **T-109**, once the winning model is selected (e.g. LightGBM), the fitted `pipeline` object is bundled together with the trained model into a single self-contained artifact at `models/model.pkl`:

```python
model_bundle = {
    "pipeline": pipeline,
    "model": winning_lgbm_model,
    "feature_order": list(X_train_features.columns),
    "version": "1.0.0"
}
with open("models/model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)
```

#### 3. Real-Time Online Serving (T-110 FastAPI Backend)
In **T-110**, when the FastAPI backend runs, `ModelService` in `api/app/model/services.py` loads `models/model.pkl` once at startup.

When a user hits the `POST /predict` API with raw pickup details:

```json
{
  "tpep_pickup_datetime": "2022-05-15 14:30:00",
  "PULocationID": 132,
  "DOLocationID": 236,
  "passenger_count": 1,
  "RatecodeID": 2,
  "trip_distance": 15.2,
  "VendorID": 1
}
```

The API processes the request in two steps:
1. **Feature Engineering**: `df_features = pipeline.transform(df_raw)` (converts raw pickup details into the 29 engineered features).
2. **Model Prediction**: `predictions = model.predict(df_features)` (returns predicted fare amount and duration minutes).

---

## 5. Role & Structure of `feature_pipeline.pkl`

### Question:
> What does the file `feature_pipeline.pkl` represents? Is it akin to model training artifacts, in which we store the parameters of the model?

### Answer:
**Yes, exactly!** `feature_pipeline.pkl` is a fitted artifact that stores the **learned parameters of the feature transformation steps**, just like a trained model artifact stores model weights.

#### What Parameters Are Stored Inside `feature_pipeline.pkl`?
When you call `pipeline.fit(X_train, y_train)`, the pipeline learns and freezes several internal state parameters:
1. **Target Encoding Mappings (`target_maps_` & `global_means_`)**: The smoothed historical mean fare/duration for each of the 265 pickup and dropoff zones learned from the training split, plus the global fallback target mean ($\mu \approx \$15.20$).
2. **Spatial Centroid Lookups (`centroids_df`)**: The spatial latitude and longitude coordinates for each `LocationID` (1–265) derived from the Taxi Zone Shapefile.
3. **Feature Contract & Column Ordering (`feature_names_`)**: The exact sequence and names of all 29 output features expected by downstream ML models.

#### Why Serializing `feature_pipeline.pkl` Is Essential
Without serializing the fitted feature pipeline:
* **Data Leakage / Inconsistency**: You would have to recompute target encodings or spatial lookups at prediction time, risking subtle discrepancies or data leakage.
* **Production Serving**: Online requests (`POST /predict`) only send raw user inputs. Unpickling `feature_pipeline.pkl` allows the API to call `.transform()` and instantaneously convert those raw inputs into the exact 29 features required by the trained model.

---

## 6. Transformer Methods: `fit()`, `transform()`, and `fit_transform()`

### Question:
> Another question: for the feature engineering classes we crested, I see sometimes we use the method `fit()`, but also `transform()` and `fit_transform()`. What are the differences between those methods?

### Answer:
In `scikit-learn` (and our `src/features.py` pipeline), these three methods define the lifecycle of a data transformer:

#### Summary Comparison Table
| Method | What It Does | Input Data | Mutates Transformer State? | Output |
|---|---|---|---|---|
| **`fit(X, y)`** | **Learns parameters** from data (e.g. mean fare per zone, feature names). | **Training Set Only** (`X_train`, `y_train`) | **Yes** (saves learned parameters) | Returns `self` (no data returned) |
| **`transform(X)`** | **Applies learned parameters** to transform data into new features. | **Test Set & Production API Requests** | **No** (uses frozen parameters) | Returns transformed DataFrame |
| **`fit_transform(X, y)`** | **Learns parameters AND transforms** training data in one step. | **Training Set Only** | **Yes** | Returns transformed DataFrame |

#### Detailed Breakdowns & Code Examples
1. **`fit(X, y=None)`**: The "learning" phase. It calculates statistics from the training data and saves them as internal attributes (like `self.target_maps_` or `self.centroids_df`).
2. **`transform(X)`**: The "execution" phase. It uses the parameters already saved in `self` to transform new data. It **never** recalculates parameters or looks at target labels ($y$), ensuring zero data leakage.
3. **`fit_transform(X, y=None)`**: A convenient shortcut that calls `.fit(X, y)` followed immediately by `.transform(X)`.

#### Golden Rule for Machine Learning Pipelines
* **Training Data** (`X_train`, `y_train`): Use **`fit_transform()`** (or `.fit()` then `.transform()`) to learn parameters and generate training features.
* **Test & Production API Data** (`X_test` / live JSON payloads): Use **`transform()`** ONLY (**NEVER** call `.fit()` or `.fit_transform()` on test or production data, as that would cause data leakage).

---

## 7. Haversine vs. Manhattan Distance Metrics

### Question:
> Another question: Explain to me what are Haversine and Manhattan distances in simple terms, and how do they compare.

### Answer:
Here is a simple, intuitive comparison of **Haversine** and **Manhattan** distances:

#### 1. Haversine Distance ("As the Crow Flies")
* **Analogy**: Imagine a bird flying in a direct, straight line through the air from Point A to Point B.
* **What It Measures**: The shortest **great-circle distance** between two points on the curved surface of the Earth.
```text
Point A  ---------------------------->  Point B  (Straight Line)
```

#### 2. Manhattan Distance ("Taxicab Grid Distance")
* **Analogy**: Imagine a taxi driving through the street grid of Manhattan. A car cannot drive diagonally through buildings — it must drive North/South along an avenue, turn 90°, and then drive East/West along a street.
* **What It Measures**: The sum of the vertical and horizontal grid distances: $|\Delta \text{Latitude}| + |\Delta \text{Longitude}|$.
```text
Point A  --------------------+
                             |
                             |
                             v
                          Point B  (Right-Angle City Blocks)
```

#### Key Differences & Why We Use Both in NYC
| Feature | Haversine Distance | Manhattan Distance |
|---|---|---|
| **Real-world Trajectory** | Direct straight line | Following street grids (Avenues & Streets) |
| **Formula Type** | Great-circle trigonometry | $L_1$ norm distance ($|\Delta x| + |\Delta y|$) |
| **Typical Value** | **Shorter** (Theoretical minimum) | **Longer** ($\ge$ Haversine) |
| **Best Used For** | Long open highway trips (e.g. JFK to Manhattan) | Dense city grid navigation (e.g. Midtown Manhattan) |

#### Why Combining Both Helps Our Machine Learning Models
* **City Grid Realism**: In New York City, cars travel along grid blocks. Manhattan distance is usually a much better approximation of physical driving distance than a straight line.
* **Detecting Detours**: The ratio of straight-line Haversine distance to actual metered distance ($\frac{\text{Haversine}}{\text{trip\_distance}}$) helps the model identify when a driver had to take a long detour around geographical barriers like the East River or Central Park.

---

## 8. Cyclical Encodings (Sine & Cosine) for Periodic Time

### Question:
> Another question: Explain to me what are cyclical encodings, and what do they accomplish in our FE pipeline.

### Answer:
**Cyclical Encodings** transform repeating time cycles (like hours of the day or days of the week) into continuous 2D coordinates on a circle so that machine learning models understand that the end of a cycle connects back to the beginning.

#### 1. The Problem with Raw Numbers for Time
Consider the hours of the day ($0, 1, 2, \dots, 23$):
* In real life, **Hour 23 (11 PM)** and **Hour 0 (Midnight)** occur right next to each other. Traffic at 11:55 PM is almost identical to traffic at 12:05 AM.
* However, a machine learning model looking at raw integers sees $23$ and $0$ as maximum distance apart ($23 - 0 = 23$). The model assumes 11 PM and Midnight are completely opposite extremes!

#### 2. The Solution: Mapping Time onto a 2D Clock Face
Cyclical encoding maps each time component onto a 2D circle using trigonometric **Sine** ($\sin$) and **Cosine** ($\cos$) functions:

$$\text{sin\_hour} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{cos\_hour} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

Think of $(\sin, \cos)$ as $(x, y)$ coordinates on a circular clock face:
```text
                     Hour 0 (Midnight)
                      (sin=0, cos=1)
                            |
         Hour 18 (6 PM)    -+-    Hour 6 (6 AM)
         (sin=-1, cos=0)    |     (sin=1, cos=0)
                            |
                     Hour 12 (Noon)
                      (sin=0, cos=-1)
```

Coordinates of 11 PM and Midnight:
* **Hour 0 (Midnight)**: $(\sin = 0.0, \, \cos = 1.0)$
* **Hour 23 (11 PM)**: $(\sin = -0.26, \, \cos = 0.97)$

In $(\sin, \cos)$ space, the distance between 11 PM and Midnight is tiny ($\approx 0.26$), correctly reflecting that they are only 1 hour apart!

#### 3. Why We Need BOTH Sine and Cosine
If you only used Sine ($\sin$), **6 AM** ($\sin=1.0$) and **6 PM** ($\sin=1.0$) would have identical values. Combining $(\sin, \cos)$ creates a **unique $(x, y)$ coordinate pair** for every hour of the day:
* **6 AM**: $(\sin = 1.0, \, \cos = 0.0)$
* **6 PM**: $(\sin = -1.0, \, \cos = 0.0)$

#### 4. What This Accomplishes in Our Pipeline (`src/features.py`)
In `TemporalFeatureExtractor`, we compute `sin_hour` & `cos_hour` (24-hour cycle) and `sin_dayofweek` & `cos_dayofweek` (7-day week cycle). This enables models (especially Linear Regression, MLPs, and SVMs) to learn smooth predictions across midnight and across week boundaries (Sunday $\to$ Monday).

---

## 9. Categorical Encodings Comparison: One-Hot vs. Ordinal vs. Target Encoding

### Question:
> Another question: Explain to me what are One-Hot Encoding, Ordinal Encoding and Target Categorical Encoding, what are their strengths and weaknesses, and why we decided to use the latter in our FE pipeline.

### Answer:
Here is a breakdown of **One-Hot Encoding**, **Ordinal Encoding**, and **Target Categorical Encoding**, including their strengths, weaknesses, and why **Target Encoding** was chosen for our pipeline:

#### 1. Summary Comparison Table
| Encoding Method | How It Works | Output Dimension for 265 Zones | Primary Strength | Primary Weakness |
|---|---|---|---|---|
| **One-Hot Encoding** | Creates a separate binary ($0/1$) column for each category level. | **530 sparse columns** ($265 \text{ PU} + 265 \text{ DO}$) | Assumes no artificial order between categories. | Explodes feature dimensionality (sparse matrix, high RAM, overfitting). |
| **Ordinal Encoding** | Maps each category to an arbitrary integer ($1, 2, \dots, 265$). | **2 dense columns** | Extremely compact (1 column per feature). | Assumes false mathematical ordering ($200 > 2$). |
| **Target Encoding (Our Choice)** | Replaces category ID with historical average target ($\bar{y}$) learned from train set. | **2 dense columns** | **Compact 1D signal with direct monetary correlation**. | Risk of target leakage if not fitted strictly on train data. |

#### 2. Deep Dive: Strengths & Weaknesses
##### A. One-Hot Encoding
* **Strengths**: Excellent for low-cardinality features (`VendorID`, `passenger_count`). Avoids imposing mathematical order.
* **Weaknesses**: For 265 pickup and dropoff zones, produces **530 sparse columns**, inflating RAM, slowing training, and causing high variance in linear models and tree splits.

##### B. Ordinal Encoding
* **Strengths**: Highly compact (1 column per feature). Excellent when categories have a natural physical order (`Small < Medium < Large`).
* **Weaknesses**: For nominal IDs (like Taxi Zone IDs), forces a **false linear assumption** ($200 > 2$).

##### C. Target Categorical Encoding (Our Choice)
* **Strengths**:
  1. **Compact 1D Representation**: Encodes location IDs into a single continuous column.
  2. **Direct Price & Time Signal**: Immediately tells models that **JFK Airport (`LocationID=132`)** corresponds to an average fare of $\approx \$52.00$, while **Alphabet City (`LocationID=4`)** corresponds to $\approx \$12.50$.
  3. **Linear Correlation**: Converts arbitrary categorical IDs into continuous numeric scores that linearly correlate with target fares and durations.
* **Weaknesses**: Risk of overfitting on rare categories (handled via **Bayesian smoothing**) and target leakage (handled by fitting **strictly on training data**).

#### 3. Rationale for Our Choice in `src/features.py`
In NYC taxi prediction, `PULocationID` and `DOLocationID` are high-cardinality nominal features (265 zones each). Target Encoding is the optimal strategy because it provides a compact, continuous 1D representation of neighborhood pricing and trip duration profiles without inflating feature count.

---

## 10. Mathematical Step-by-Step Calculation of Target Encoding

### Question:
> For Target Encoding, what do you mean by this?:
> `Replaces category ID with historical average target ($\bar{y}$) learned from train set`
> Does this mean each category ID is replaced by a number? how is that number calculated?

### Answer:
**Yes, exactly!** Each category ID (e.g. `PULocationID = 132`) is replaced by a single floating-point number representing the historical expected fare learned from the training set.

#### Step 1: Calculate Historical Zone Averages from Training Data
During `target_encoder.fit(X_train, y_train)`, the code calculates $n_k$ (trip count) and $\bar{y}_k$ (average fare) for each zone $k$, as well as global average fare $\mu \approx \$15.20$:
* **JFK Airport (`PULocationID = 132`)**: $n_{\text{JFK}} = 100{,}000$, $\bar{y}_{\text{JFK}} = \$52.00$
* **Alphabet City (`PULocationID = 4`)**: $n_{\text{Alphabet}} = 15{,}000$, $\bar{y}_{\text{Alphabet}} = \$12.50$
* **Rare Remote Zone (`PULocationID = 2`)**: $n_{\text{Rare}} = 2$, $\bar{y}_{\text{Rare}} = \$90.00$

#### Step 2: Apply Bayesian Smoothing Formula
To prevent rare zones from outputting extreme noisy numbers, we blend the zone average $\bar{y}_k$ with global average $\mu$ ($m = 10$):

$$\text{Encoded Number} = \frac{n_k \cdot \bar{y}_k + m \cdot \mu}{n_k + m}$$

* **JFK (`LocationID = 132`)**: $\frac{100{,}000 \cdot 52.00 + 10 \cdot 15.20}{100{,}000 + 10} \approx \mathbf{52.00}$
* **Rare Zone (`LocationID = 2`)**: $\frac{2 \cdot 90.00 + 10 \cdot 15.20}{2 + 10} = \frac{180 + 152}{12} = \mathbf{27.67}$ (smooths noisy $\$90.00$ down towards global $\$15.20$).

#### Step 3: Replace Category IDs in Data Matrix
When `target_encoder.transform(X)` is called, every row's `PULocationID` is looked up in the dictionary and replaced:
| Raw `PULocationID` | Transformed `PULocationID_target_enc` | Interpretation for ML Model |
|---|---|---|
| `132` (JFK) | **`52.00`** | Expect high fare ($\approx \$52.00$) |
| `4` (Alphabet City) | **`12.50`** | Expect low fare ($\approx \$12.50$) |
| `132` (JFK) | **`52.00`** | Expect high fare ($\approx \$52.00$) |
| `2` (Rare Zone) | **`27.67`** | Expect smoothed moderate fare |

---

## 11. General Definition and Goal of Feature Engineering

### Question:
> So, correct me if I'm wrong, but based on all of this, feature engineering is about finding what combinations of our preprocessed data (which may include applying mathematical functions to our preprocessed data) are the "best", so it makes our machine learning models as precise as possible?

### Answer:
**Spot on! You hit the nail right on the head.** 

That is the exact core definition and purpose of **Feature Engineering**.

#### Why Feature Engineering Is So Essential
Cleaned, preprocessed data removes errors (like negative fares or missing values), but raw columns alone rarely reveal the true underlying patterns to a machine learning algorithm.

Machine learning models are mathematical function approximators. They cannot "intuit" physical domain concepts on their own. Feature Engineering is the art and science of **translating domain knowledge into mathematical representations** that make patterns obvious to the model.

#### The 3 Core Pillars of Feature Engineering (Illustrated by Our Pipeline)
1. **Applying Mathematical Transformations to Expose Hidden Geometry**: Applying $\sin$ and $\cos$ cyclical functions so the model recognizes that 11:59 PM and 12:01 AM are 2 minutes apart, and extracting `is_rush_hour = 1`.
2. **Re-encoding Raw ID Cards into Continuous Predictors**: Target Encoding converts `132` into `52.00` (historical average fare in dollars), providing a direct monetary signal.
3. **Combining Multiple Features into Interaction Ratios**: Computing Haversine centroid distance ($13.5$ miles) and taking the ratio $\frac{13.5}{15.2} = 0.88$ to measure route directness vs. traffic detours.

#### The Famous Rule in Machine Learning
As AI pioneer **Andrew Ng** famously stated:
> *"Applied machine learning is basically feature engineering. Coming up with features is complicated, messy, human-intensive, requiring domain knowledge..."*

By creating these 29 engineered features in `src/features.py`, we make it significantly easier for all downstream models (Linear Regression, Decision Trees, LightGBM, XGBoost, and MLPs) to achieve higher prediction accuracy ($\text{R}^2$), lower error ($\text{MAE}/\text{RMSE}$), and faster training convergence.

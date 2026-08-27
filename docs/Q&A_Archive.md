# NYC Taxi Fare & Trip Duration Project — Q&A Archive

This document compiles technical questions, theoretical concepts, and architectural inquiries asked during the development of this project, along with their detailed explanations and solutions.

---

## Table of Contents

### [Part I: Dataset Scope, Domain Assumptions & Leakage Prevention](#part-i-dataset-scope-domain-assumptions--leakage-prevention)
1. [Dataset Sufficiency](#1-dataset-sufficiency)
2. [Target Encoding Rationale & Leakage Prevention](#2-target-encoding-rationale--leakage-prevention)
3. [Why `trip_distance` is an Input Feature Rather than a Prediction Target (Deterministic vs. Stochastic Modeling)](#3-why-trip_distance-is-an-input-feature-rather-than-a-prediction-target-deterministic-vs-stochastic-modeling)
4. [`TimeSeriesSplit` & Forward-Chaining Cross-Validation (Preventing Temporal Leakage)](#4-timeseriessplit--forward-chaining-cross-validation-preventing-temporal-leakage)

### [Part II: Geospatial & Coordinate Engineering](#part-ii-geospatial--coordinate-engineering)
5. [Haversine vs. Manhattan Distance Metrics](#5-haversine-vs-manhattan-distance-metrics)
6. [Trip Distance vs. Haversine Straight-Line Distance in Taxi Modeling](#6-trip-distance-vs-haversine-straight-line-distance-in-taxi-modeling)
7. [Zone Centroids & Geographic Coordinate Extraction from TLC Shapefiles](#7-zone-centroids--geographic-coordinate-extraction-from-tlc-shapefiles)
8. [Geodesic vs. Geodetic Distance in Spatial Modeling](#8-geodesic-vs-geodetic-distance-in-spatial-modeling)

### [Part III: Temporal & Categorical Feature Transformations](#part-iii-temporal--categorical-feature-transformations)
9. [Temporal vs. Spatial Feature Separation](#9-temporal-vs-spatial-feature-separation)
10. [Cyclical Encodings (Sine & Cosine) for Periodic Time](#10-cyclical-encodings-sine--cosine-for-periodic-time)
11. [Categorical Encodings Comparison: One-Hot vs. Ordinal vs. Target Encoding](#11-categorical-encodings-comparison-one-hot-vs-ordinal-vs-target-encoding)
12. [Mathematical Step-by-Step Calculation of Target Encoding](#12-mathematical-step-by-step-calculation-of-target-encoding)
13. [General Definition and Goal of Feature Engineering](#13-general-definition-and-goal-of-feature-engineering)

### [Part IV: Pipeline Architecture & Scikit-Learn Design Patterns](#part-iv-pipeline-architecture--scikit-learn-design-patterns)
14. [Role & Structure of `feature_pipeline.pkl`](#14-role--structure-of-feature_pipelinepkl)
15. [Transformer Methods: `fit()`, `transform()`, and `fit_transform()`](#15-transformer-methods-fit-transform-and-fit_transform)

### [Part V: Model Architectures & Training Strategies](#part-v-model-architectures--training-strategies)
16. [Multi-Output vs. Two Single-Output Regressors for Fare and Duration](#16-multi-output-vs-two-single-output-regressors-for-fare-and-duration)
17. [Role of SimpleImputer and StandardScaler in Neural Network Pipelines](#17-role-of-simpleimputer-and-standardscaler-in-neural-network-pipelines)
18. [Hyperparameter Tuning for MLP Models on Tabular Data](#18-hyperparameter-tuning-for-mlp-models-on-tabular-data)
19. [The Adam Optimizer: Adaptive Moment Estimation Mechanics & Role in MLP Training](#19-the-adam-optimizer-adaptive-moment-estimation-mechanics--role-in-mlp-training)
20. [Differences Between LightGBM and XGBoost Models](#20-differences-between-lightgbm-and-xgboost-models)
21. [Leaf-Wise Tree Growth, Histogram Binning, and L1 Objective Optimization in GBDT Regressors](#21-leaf-wise-tree-growth-histogram-binning-and-l1-objective-optimization-in-gbdt-regressors)

### [Part VI: Model Evaluation, Advanced Metrics & Interpretability](#part-vi-model-evaluation-advanced-metrics--interpretability)
22. [Regression Evaluation Metrics: MAE, RMSE, MAPE, and R²](#22-regression-evaluation-metrics-mae-rmse-mape-and-r2)
23. [Advanced Regression Evaluation Metrics & The Role of ROC / REC Curves in Continuous Modeling](#23-advanced-regression-evaluation-metrics--the-role-of-roc--rec-curves-in-continuous-modeling)
24. [Feature Importance in Tree Ensembles: Split Gain (Loss Reduction) vs. Split Count (Frequency)](#24-feature-importance-in-tree-ensembles-split-gain-loss-reduction-vs-split-count-frequency)

### [Part VII: Production Serving, API Design & System SLAs](#part-vii-production-serving-api-design--system-slas)
25. [Downstream Model Integration & API Serving Workflow](#25-downstream-model-integration--api-serving-workflow)
26. [Pickup Datetime Input Handling & ISO Format in Production UIs](#26-pickup-datetime-input-handling--iso-format-in-production-uis)
27. [Service Level Agreements (SLA) and In-Process Latency Benchmarking in Machine Learning APIs](#27-service-level-agreements-sla-and-in-process-latency-benchmarking-in-machine-learning-apis)

---

# Part I: Dataset Scope, Domain Assumptions & Leakage Prevention

## 1. Dataset Sufficiency

### Question:
> Will the existing parquet dataset (stored in `"dataset/yellow_tripdata_2022-05.parquet"`) be enough to train our models? Or will we need data from other months in 2022, or even data from different years?

### Answer:
**Yes, for the scope of this project and prototype demonstration, the May 2022 dataset is completely sufficient.** 

With **3.59 million raw records** (and ~3.30 million high-quality rows after outlier filtering), a single month provides ample statistical sample size across all 265 taxi zones, allowing models to learn robust hourly, day-of-week, and rush-hour patterns with low computational overhead.

However, relying on a **single-month window (May 2022)** introduces three documented operational caveats that must be considered for real-world production deployment:

#### 1. Unobserved Annual Macro-Seasonality
* **The Caveat**: Training strictly on May data captures late-spring weather, active school-year commuting, and specific holidays (e.g., Memorial Day weekend). It cannot observe annual seasonal dynamics, such as:
  * **Summer Vacation Dips (July–August)**: Significant drops in weekday morning commuting and business travel.
  * **Autumn Congestion Peaks (September–November)**: Increased traffic from UN General Assembly meetings, holiday shopping, and the NYC Marathon.
  * **Winter Blizzard Slowdowns (January–February)**: Severe snowstorms and sub-zero temperatures that dramatically reduce traffic speeds across the entire city.
* **Production Recommendation**: Expanding to a 12-month rolling dataset (or continuous monthly retraining pipelines using Apache Airflow) would enable the models to capture full annual seasonality.

#### 2. Regulatory & Tariff Rate Snapshot
* **The Caveat**: Yellow taxi fares are strictly regulated by the NYC Taxi and Limousine Commission (TLC). The May 2022 data reflects the tariff structure and fuel surcharges active at that specific time (e.g., $\$2.50$ initial base fare, $\$52.00$ JFK flat-rate).
* **Policy Drift**: In late 2022, the TLC implemented significant fare revisions (raising the base fare to $\$3.00$ and the JFK flat rate to $\$70.00$). A model trained strictly on May 2022 data would suffer from concept drift without updated retraining or dynamic rate-table integration.

#### 3. Unobserved Meteorology (Dynamic Weather Telemetry)
* **The Caveat**: While calendar features capture predictable rush-hour cycles, sudden unseasonal weather events (thunderstorms, heavy downpours) introduce high travel duration variance that cannot be captured by timestamp features alone without integrating live NOAA weather feeds.

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

## 3. Why `trip_distance` is an Input Feature Rather than a Prediction Target (Deterministic vs. Stochastic Modeling)

### Question:
> How do we know `trip_distance` is the actual traveled distance? Why is it included as an allowed input feature rather than a prediction target like fare and duration, given that it is recorded after the ride?

### Answer:

#### 1. How `trip_distance` is Measured
According to the official NYC TLC Data Dictionary, `trip_distance` is the **odometer mileage recorded by the in-vehicle taximeter (TPEP system)**. In historical training data, this value reflects the actual road distance traveled from meter activation to completion.

#### 2. Deterministic vs. Stochastic Modeling
* **Distance is Deterministic**: Given a fixed Origin $(A)$ and Destination $(B)$, physical driving distance is a deterministic property of geography and the road network. It can be computed directly using coordinates (Haversine/Manhattan formulas) or calculated via deterministic graph shortest-path algorithms (OSRM, Dijkstra, $A^*$). A machine learning model is **not needed to predict physical distance**.
* **Fare & Duration are Stochastic**: Trip duration and fare cannot be computed simply from a map. They depend on dynamic real-world complexities: NYC rush-hour traffic gridlock, time-of-day bottlenecks, TLC regulated surcharges, airport flat-rate policies, and speed variance. That is why Fare ($) and Duration (mins) are our **two prediction targets ($Y_1, Y_2$)**, while Distance serves as a foundational **known input feature ($X$)**.

#### 3. Prototype UI vs. Production Architecture
* **In the Prototype UI**: The user does not manually guess distance. Streamlit automatically computes straight-line **Haversine distance between zone centroids** as an instant proxy, sending `{"trip_distance": estimated_distance, ...}` in the JSON payload to FastAPI.
* **In Full Production**: An external routing engine API (such as OSRM or Google Maps) would supply the exact road-network driving distance upfront prior to vehicle departure.
* **In the Model**: `trip_distance` is the **single highest-importance feature** in the entire LightGBM model ($>40\%$ split gain), making its inclusion critical for accurate fare and duration quotes.

---

## 4. `TimeSeriesSplit` & Forward-Chaining Cross-Validation (Preventing Temporal Leakage)

### Question:
> What is the `TimeSeriesSplit` class, and what does it do?

### Answer:
`sklearn.model_selection.TimeSeriesSplit` is a cross-validation splitter designed specifically for **time-dependent, chronological data**. It implements **walk-forward validation (also known as rolling or expanding-window evaluation)** to ensure models are trained only on historical data and tested on future data.

#### 1. Why Standard K-Fold Cross-Validation Fails on Time Series
In standard $K$-Fold Cross-Validation (`KFold(shuffle=True)`):
* Observations are randomly shuffled and partitioned across folds.
* In Fold 1, a model might train on rides from **May 28** to predict rides on **May 5**.
* **The Fatal Flaw (Lookahead Bias & Future Leakage)**: Training on future data leaks macro-trends, holiday patterns (e.g. Memorial Day traffic), fuel price fluctuations, and seasonal shifts backwards in time. This creates **unrealistically optimistic evaluation scores** that collapse when deployed into live production where future data does not exist.

#### 2. How `TimeSeriesSplit` Works (Walk-Forward Mechanics)
`TimeSeriesSplit` respects the arrow of time ($t_{\text{train}} < t_{\text{val}}$). For $k = 3$ splits (`n_splits=3`), the dataset is divided chronologically into expanding training windows and forward test slices:

```text
Split 1:  [ Train: Week 1 ]  ------------------------>  [ Validate: Week 2 ]
Split 2:  [ Train: Week 1 + Week 2 ]  --------------->  [ Validate: Week 3 ]
Split 3:  [ Train: Week 1 + Week 2 + Week 3 ]  ------>  [ Validate: Week 4 ]
```

* **Iteration 1**: Trains on the earliest chunk $[0 \dots t_1]$ and tests on the immediate next period $[t_1 \dots t_2]$.
* **Iteration 2**: Expands the training window to $[0 \dots t_2]$ and tests on $[t_2 \dots t_3]$.
* **Iteration 3**: Expands the training window to $[0 \dots t_3]$ and tests on $[t_3 \dots t_4]$.

In every single iteration, the model **never sees a single data point from the future**.

#### 3. Key Parameters & Configuration Options

| Parameter | Type | Default | What It Controls |
|---|---|---|---|
| **`n_splits`** | `int` | `5` | The number of sequential forward splits (in our project, `n_splits=3`). |
| **`max_train_size`** | `int` / `None` | `None` | Caps the maximum size of the training window, converting the expanding window into a **sliding / rolling window** (e.g. training only on the most recent 14 days). |
| **`gap`** | `int` | `0` | Number of samples to exclude between the end of the train set and start of the test set, preventing short-term autocorrelation leakage. |
| **`test_size`** | `int` / `None` | `None` | Explicit size for each validation fold. |

#### 4. Role in the NYC Taxi Modeling Pipeline
In our project:
1. **Primary Evaluation Split**: The entire month is split chronologically at **May 23, 2022** (3-week train set of 2.40M records vs. 1-week test set of 900k records).
2. **Internal Cross-Validation**: During offline hyperparameter tuning and model exploration (e.g. testing `num_leaves` in LightGBM and `max_depth` in XGBoost), `TimeSeriesSplit(n_splits=3)` is applied across the 2.40M training records to evaluate generalization performance without future contamination.

---

# Part II: Geospatial & Coordinate Engineering

## 5. Haversine vs. Manhattan Distance Metrics

### Question:
> Explain to me what are Haversine and Manhattan distances in simple terms, and how do they compare.

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

## 6. Trip Distance vs. Haversine Straight-Line Distance in Taxi Modeling

### Question:
> What does the `Trip distance (miles)` measure? The straight line between the Pickup and dropoff locations?

### Answer:
No. `trip_distance` in the NYC Taxi dataset is **not** the straight-line distance.

#### 1. What `trip_distance` Measures
* In the raw TLC dataset, `trip_distance` is the **metered odometer road distance** (in statute miles) recorded by the taxi's meter as the vehicle navigates the physical NYC street network, one-way avenues, bridges, and highways.
* In a live production taxi-hailing app (like Uber or Lyft), this corresponds to the **routing engine estimate** (e.g. Google Maps or OSRM route distance) computed between the pickup and dropoff addresses before dispatching the driver.

#### 2. How it Differs from Haversine Distance
* **Haversine Distance**: The theoretical **"as-the-crow-flies" great-circle distance** between the geometric centroids of the pickup and dropoff taxi zones.
* **Why Both Are Used in Feature Engineering**:
  * Real taxi trips are almost never straight lines because of Manhattan's grid layout, rivers, and bridge choke points.
  * Our feature engineering pipeline computes `haversine_ratio = haversine_distance / (trip_distance + ε)`.
  * A ratio close to $1.0$ indicates a direct highway route (e.g. JFK expressway corridors), while a low ratio ($< 0.6$) signals circuitous routes, street detours, or cross-river bridge traffic, which heavily impacts trip duration and fare.

---

## 7. Zone Centroids & Geographic Coordinate Extraction from TLC Shapefiles

### Question:
> What does *"Zone Centroid Coordinates: Joins TLC shapefile latitude and longitude for pickup & dropoff zones"* mean? Do we end up with the physical coordinates of the zone centroids?

### Answer:
**Yes, exactly.** In the raw NYC Taxi dataset, pickups and dropoffs are recorded only as **discrete zone IDs from 1 to 265** (e.g. `132` = JFK Airport, `236` = Upper East Side North). The raw Parquet file contains no GPS latitude or longitude coordinates.

#### 1. The Role of the TLC Shapefile
The NYC Taxi & Limousine Commission (TLC) provides an official GIS shapefile containing the multi-polygon geographical boundaries for all 265 taxi zones across the five boroughs.

#### 2. Centroid Extraction & Lookup
To convert discrete zone integers into continuous spatial coordinates:
* During data preprocessing, the geometric center point (**centroid**) `(latitude, longitude)` of every zone polygon is precomputed.
* In the feature engineering pipeline (`src/features.py` and `predictor.py`), the model takes the integer IDs (`PULocationID` and `DOLocationID`) and joins their corresponding coordinates:
  * `pickup_latitude` and `pickup_longitude`
  * `dropoff_latitude` and `dropoff_longitude`

#### 3. Why Continuous Coordinates are Essential
These 4 numeric coordinate features allow the machine learning pipeline to compute:
* **Geodesic Haversine Distance**: Straight-line great-circle distance.
* **Manhattan Grid Distance**: Grid-aligned street network travel distance.
* **Proximity to Key Hubs**: Geographic distance to major transport hubs (JFK, LaGuardia, Newark airports).

---

## 8. Geodesic vs. Geodetic Distance in Spatial Modeling

### Question:
> Should spatial distance across the Earth's surface be called "geodesic" or "geodetic" distance?

### Answer:
**"Geodesic distance"** is the accurate and standard mathematical term when referring to shortest paths across the surface of the Earth.

* **Geodesic Distance (or Great-Circle Distance)**: The shortest path between two points on the curved surface of a sphere or ellipsoid. In our pipeline, the **Haversine formula** calculates the geodesic distance between pickup and dropoff coordinates assuming a spherical Earth ($R \approx 3958.8\text{ miles}$).
* **Geodetic**: Refers to the broader scientific discipline of *geodesy* (measuring the Earth's geometric shape, orientation in space, and gravitational field) and geodetic reference systems/datums (such as WGS84 coordinates).

Therefore, when describing the spatial feature calculated between two coordinate pairs, **"geodesic distance"** is the mathematically precise designation.

---

# Part III: Temporal & Categorical Feature Transformations

## 9. Temporal vs. Spatial Feature Separation

### Question:
> Explain to me why do we have features related to the time (TemporalFeatureExtractor) and others related to space (SpatialZoneFeatureExtractor)

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

## 10. Cyclical Encodings (Sine & Cosine) for Periodic Time

### Question:
> Explain to me what are cyclical encodings, and what do they accomplish in our FE pipeline.

### Answer:
**Cyclical Encodings** transform repeating time cycles (like hours of the day or days of the week) into continuous 2D coordinates on a circle so that machine learning models understand that the end of a cycle connects back to the beginning.

#### 1. The Problem with Raw Numbers for Time
Consider the hours of the day ($0, 1, 2, \dots, 23$):
* In real life, **Hour 23 (11 PM)** and **Hour 0 (Midnight)** occur right next to each other. Traffic at 11:55 PM is almost identical to traffic at 12:05 AM.
* However, a machine learning model looking at raw integers sees $23$ and $0$ as maximum distance apart ($23 - 0 = 23$). The model assumes 11 PM and Midnight are completely opposite extremes!

#### 2. The Solution: Mapping Time onto a 2D Clock Face
Cyclical encoding maps each time component onto a 2D circle using trigonometric **Sine** ($\sin$) and **Cosine** ($\cos$) functions:

$$\text{sin\_hour} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{cos\_hour} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

Think of $(\sin, \ cos)$ as $(x, y)$ coordinates on a circular clock face:
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

## 11. Categorical Encodings Comparison: One-Hot vs. Ordinal vs. Target Encoding

### Question:
> Explain to me what are One-Hot Encoding, Ordinal Encoding and Target Categorical Encoding, what are their strengths and weaknesses, and why we decided to use the latter in our FE pipeline.

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

## 12. Mathematical Step-by-Step Calculation of Target Encoding

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

## 13. General Definition and Goal of Feature Engineering

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

---

# Part IV: Pipeline Architecture & Scikit-Learn Design Patterns

## 14. Role & Structure of `feature_pipeline.pkl`

### Question:
> What does the file `feature_pipeline.pkl` represents? Is it similar to model training artifacts, in which we store the parameters of the model?

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

## 15. Transformer Methods: `fit()`, `transform()`, and `fit_transform()`

### Question:
> For the feature engineering classes we created, I see sometimes we use the method `fit()`, but also `transform()` and `fit_transform()`. What are the differences between those methods?

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

# Part V: Model Architectures & Training Strategies

## 16. Multi-Output vs. Two Single-Output Regressors for Fare and Duration

### Question:
> Do you think it's necessary to train another MLP that's able to predict both the fare and duration?

### Answer:
**No, it is not necessary**, and continuing with two independent single-output models is the recommended approach.

#### 1. Project Consistency & Architecture Contract (T-106 Binding)
In **T-106**, our team established and recorded an explicit architectural decision:
> *Train two single-output models (one for `fare_amount`, one for `duration_minutes`) rather than a single multi-output model.*

All other candidate model families in the project (**Decision Trees**, **LightGBM**, and **XGBoost**) use two independent single-output regressors. Keeping the MLP aligned ensures fair, 1-to-1 comparison across the entire benchmark leaderboard for model selection in **T-109**.

#### 2. Loss Conflict & Gradient Interference
In a multi-output neural network, the loss function must optimize both targets simultaneously:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fare}} + \lambda \cdot \mathcal{L}_{\text{duration}}$$

* **Different Variance Scales**: Fares and trip durations operate on different physical and financial distributions.
* **Gradient Tug-of-War**: Shared hidden layers can suffer from *negative transfer*, where gradient updates from the noisy duration loss degrade the weights needed for precise fare estimation.

#### 3. Independent Hyperparameter Tuning & Early Stopping
* **Fare Amount**: Fares are strongly governed by metered mileage and flat-rate airport codes ($R^2 \approx 0.95$).
* **Trip Duration**: Durations are heavily non-linear, influenced by traffic congestion, hour of the day, and grid bottlenecks ($R^2 \approx 0.80$).

Single-output networks allow each target to reach early stopping independently and use dedicated network capacities without compromising the other.

#### Summary Recommendation
Two separate single-output MLPs (as implemented in `src/mlp.py`) provide:
1. **Higher accuracy per target** (no multi-task gradient conflict).
2. **Clean modularity** matching LightGBM, XGBoost, and Decision Trees.
3. **Decoupled latency & inference diagnostics**.

---

## 17. Role of SimpleImputer and StandardScaler in Neural Network Pipelines

### Question:
> What do the `SimpleImputer` and `StandardScaler` classes do?

### Answer:
Here is what **`SimpleImputer`** and **`StandardScaler`** do and why they are essential for training the Multi-Layer Perceptron (MLP) in `src/mlp.py`:

#### 1. `SimpleImputer(strategy="median")` — Handling Missing Values
* **What It Does**: Replaces any missing values (`NaN` or `None`) in each numeric feature with the **median value** of that feature computed from the training dataset.
* **Why It Is Essential for Neural Networks**:
  * **The Math Problem**: Neural networks compute layer activations via matrix multiplications:
    $$z = W \cdot x + b$$
    If even a single feature $x_i$ is `NaN`, the entire matrix product becomes `NaN`, corrupting gradient descent and causing training to crash.
  * **In Our Dataset**: Special zones (like `LocationID=264` and `265` for Unknown/Outside NYC) have no shapefile geometry, leaving `pu_lat`, `pu_lon`, `do_lat`, and `do_lon` missing. `SimpleImputer` safely fills those coordinates with median NYC values.

#### 2. `StandardScaler()` — Normalizing Feature Scales
* **What It Does**: Transforms every numeric feature so that its **Mean is 0** ($\mu = 0$) and its **Standard Deviation is 1** ($\sigma = 1$) using the standard z-score formula:
  $$z = \frac{x - \mu}{\sigma}$$
* **Why It Is Essential for Neural Networks**:
  * **Scale Disparity**: In our engineered features, some variables have large numeric ranges (e.g. `trip_distance` spanning $0.5$ to $30+$ miles), while others are small (e.g. `sin_hour` spanning $-1.0$ to $+1.0$).
  * **Gradient Stability**: Without scaling, features with larger numbers generate disproportionately huge gradients, causing optimization oscillations or exploding gradients. Standardizing all 29 features onto the same scale ensures fast, stable convergence for the Adam optimizer.

#### 3. Why Put Both Inside the `sklearn.pipeline.Pipeline`?
In `src/mlp.py`:
```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(...))
])
```
1. **Zero Data Leakage**: The medians ($\text{median}$) and scaling parameters ($\mu, \sigma$) are computed **strictly on `X_train`** during `.fit()`.
2. **Seamless Production Serving**: When a new request arrives at the API, calling `.predict()` automatically imputes and scales the new row using the saved training parameters without any manual preprocessing steps.

---

## 18. Hyperparameter Tuning for MLP Models on Tabular Data

### Question:
> Do we need to tune the MLP model?

### Answer:
**No, extensive additional tuning is not needed.**

#### 1. Requirements of Ticket T-108 are Already Satisfied
The goal of **T-108** was to provide an **evaluative benchmark** representing deep learning / neural network architectures against decision trees and gradient boosting.

We have already configured and tuned the critical hyperparameters:
* **Architecture**: 2 hidden layers `(64, 32)` with ReLU activations.
* **Internal Scaling**: `SimpleImputer` + `StandardScaler` inside the pipeline.
* **Optimization & Regularization**: Adam optimizer ($\eta = 0.002$), $L_2$ penalty ($\alpha = 0.0001$), and mini-batch size $4{,}096$.
* **Early Stopping**: Dynamically halts training based on a 10% validation slice from the training split.

#### 2. The "Tabular Data" Reality in Machine Learning
In empirical machine learning research (e.g. *Grinsztajn et al., 2022 — Why tree-based models still outperform deep learning on tabular data*), neural networks consistently struggle to match Gradient Boosted Decision Trees on tabular data.

Comparing our current test set results:

| Model Family | Fare MAE | Duration MAE | Training Time | Inference Latency |
|---|---|---|---|---|
| **LightGBM (T-107)** | **$1.26** | **3.36 min** | **~12s** | **0.41 ms** |
| **XGBoost (T-107)** | **$1.27** | **3.41 min** | **~24s** | **0.45 ms** |
| **MLP (T-108)** | **$2.21** | **6.04 min** | **~115s** | **0.77 ms** |

* NYC Taxi data features sharp geometric boundaries (e.g. Manhattan street blocks, flat-rate airport pricing for JFK).
* Decision Trees and GBDTs partition these exact step-function thresholds with high precision in seconds, whereas neural networks attempt to approximate them using smooth continuous hyperplanes, leading to higher error and longer training times.

#### 3. Impact on Model Selection (T-109)
In **T-109** (Model Selection), **LightGBM** is already the definitive winner:
1. **Lowest Error**: Lowest MAE ($1.26$) and highest $R^2$ ($0.952$).
2. **Fastest Inference**: $\approx 0.41$ ms single-row latency.
3. **Lowest Training Overhead**: Trains in 12 seconds vs. 115+ seconds for MLP.

Further exhaustive grid-searching on the MLP would yield negligible gains while consuming significant compute time without changing the model selection outcome.

---

## 19. The Adam Optimizer: Adaptive Moment Estimation Mechanics & Role in MLP Training

### Question:
> What is the `Adam` class, and what does it do?

### Answer:
**Adam (short for Adaptive Moment Estimation)**, introduced by Diederik Kingma and Jimmy Ba in 2015, is an advanced first-order gradient-based optimization algorithm. It dynamically computes individual, adaptive learning rates for every weight and bias parameter in a neural network by maintaining exponentially decaying moving averages of both past gradients (**first moment / momentum**) and past squared gradients (**second uncentered moment / variance**).

---

#### 1. Why Standard Gradient Descent (SGD) Struggles on Tabular Neural Networks
In standard Stochastic Gradient Descent (SGD):
$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$
* **Fixed Global Learning Rate ($\eta$)**: A single step size is applied to all parameters uniformly. If $\eta$ is set too small, convergence takes days; if set too large, the optimizer overshoots the minimum and diverges.
* **Ravines & Pathological Curvature**: Tabular feature spaces feature disparate scales (e.g. `trip_distance` spanning $0.5$ to $30+$ miles vs. `sin_hour` spanning $-1$ to $+1$). This creates an oblong loss surface where SGD oscillates violently along steep canyon walls instead of descending along the gentle gradient toward the minimum.

---

#### 2. The Two Core Engines That Combine to Form Adam

Adam combines the best advantages of two prior optimization algorithms: **Momentum** and **RMSProp**.

```text
       ┌────────────────────────┐      ┌────────────────────────┐
       │   Momentum (1st Moment)│      │   RMSProp (2nd Moment) │
       │ Tracks Direction/Velocity│     │ Scales Inversely by Vol│
       └───────────┬────────────┘      └───────────┬────────────┘
                   │                               │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │         Adam Optimizer        │
                   │ Adaptive Direction & Step Size│
                   └───────────────────────────────┘
```

##### A. Momentum (First Moment — Exponential Moving Average of Gradients)
Tracks the persistent *direction* and velocity of the gradient, accelerating progress along consistent trajectories and smoothing out noisy oscillations:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
*(where $\beta_1 \approx 0.9$ is the first moment decay factor)*.

##### B. RMSProp (Second Moment — Exponential Moving Average of Squared Gradients)
Tracks the *magnitude* of recent updates to automatically scale individual parameter learning rates:
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
*(where $\beta_2 \approx 0.999$ is the second moment decay factor)*.
* Parameters with **frequently occurring, massive gradients** receive **smaller step sizes** to prevent destabilizing weight explosions.
* Parameters with **infrequent, subtle gradients** receive **larger step sizes** to ensure active learning.

---

#### 3. Bias Correction & Mathematical Update Formula

Because $m_0$ and $v_0$ are initialized to zero vectors, they are initially biased toward zero during the first few training steps ($t \approx 1, 2$). Adam applies an exact analytical **bias correction**:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The final parameter update formula for each parameter $\theta$ at step $t$ is:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

*(where $\eta$ is the base learning rate, and $\epsilon \approx 10^{-8}$ prevents division by zero)*.

---

#### 4. Summary of Key Hyperparameters in Scikit-Learn (`solver="adam"`)

| Hyperparameter | Typical Default | Description in MLP Training |
|---|---|---|
| **`learning_rate_init` ($\eta$)** | `0.001` (or `0.002`) | Base initial learning rate before per-parameter adaptive scaling. |
| **`beta_1`** | `0.9` | Exponential decay rate for first moment (gradient direction/momentum). |
| **`beta_2`** | `0.999` | Exponential decay rate for second moment (gradient variance/scale). |
| **`epsilon`** | `1e-8` | Small numerical constant for floating-point stability. |
| **`early_stopping`** | `True` | Monitors validation loss to halt the optimizer when improvement stalls. |

---

#### 5. Role in Our NYC Taxi MLP Architecture (`src/mlp.py`)
In `MLPRegressor(hidden_layer_sizes=(64, 32), solver="adam", learning_rate_init=0.002)`:
1. **Handles 29 Tabular Feature Distributions**: Automatically balances step sizes across spatial coordinates, cyclical trigonometric angles, and binary rush-hour indicators.
2. **Stable Mini-Batch Convergence**: Works with mini-batches ($4{,}096$ rows) across 2.40 million training records, converging in ~115 seconds with smooth loss reduction.

---

## 20. Differences Between LightGBM and XGBoost Models

### Question:
> Explain to me the differences between LightGBM and XGBost models.

### Answer:
Both **LightGBM** (Microsoft, 2016) and **XGBoost** (Tianqi Chen / DMLC, 2014) are high-performance **Gradient Boosted Decision Tree (GBDT)** frameworks. They sequentially train decision trees to predict residual errors from previous trees.

However, they differ fundamentally in their **tree growth strategy**, **data sampling algorithms**, **categorical feature handling**, and **memory/speed efficiency**.

#### 1. Summary Comparison Table

| Feature / Dimension | **LightGBM** (Light Gradient Boosting) | **XGBoost** (eXtreme Gradient Boosting) |
|---|---|---|
| **Tree Growth Strategy** | **Leaf-wise (Best-First)**: Splits the single leaf yielding the highest loss reduction. | **Level-wise (Depth-First)**: Splits all nodes across a depth level uniformly. |
| **Speed & Sampling Mechanism** | **GOSS** (Gradient-based One-Side Sampling) + **EFB** (Exclusive Feature Bundling). | **Weighted Quantile Sketch** + Histogram binning (`tree_method="hist"`). |
| **Categorical Support** | **Native**: Partitions categories directly by sorting target statistics ($O(k \log k)$). | Historically required manual encoding; native support added in recent versions. |
| **Training Speed & Memory** | **$2\times - 5\times$ faster**; lower RAM footprint. | Highly optimized, but slightly slower on large tabular datasets. |
| **Tree Symmetry** | Asymmetrical, deep, high-efficiency branches. | Symmetrical, balanced trees. |
| **Our Benchmark (Fare MAE / Time)** | **$\$1.2640$** / **$12.45$s** | **$\$1.2660$** / **$24.18$s** |

#### 2. Core Differences Deep-Dive

##### A. Tree Growth: Leaf-Wise vs. Level-Wise
```text
XGBoost (Level-wise):               LightGBM (Leaf-wise):
        [ Root ]                            [ Root ]
       /        \                          /        \
   [Node]      [Node]                  [Node]      [Deep Split]
   /    \      /    \                             /          \
 [Leaf][Leaf][Leaf][Leaf]                     [Node]        [Deep Split]
                                                           /          \
                                                       [Leaf]        [Leaf]
```
* **XGBoost (Level-wise)**: Grows trees level-by-level horizontally. This creates balanced trees and prevents overfitting on small datasets, but spends compute splitting nodes that provide minimal loss reduction.
* **LightGBM (Leaf-wise)**: Chooses only the leaf with the maximum delta loss reduction across the entire tree. For the same number of splits, leaf-wise achieves **lower training loss faster**.

##### B. Algorithmic Innovations in LightGBM (GOSS & EFB)
LightGBM introduced two major techniques that make it significantly faster on millions of rows:
1. **GOSS (Gradient-based One-Side Sampling)**:
   * Data points with *large gradients* are under-fitted (need more learning), while points with *small gradients* are well-trained.
   * GOSS retains 100% of large-gradient rows and randomly samples a subset (e.g. 10–20%) of small-gradient rows, drastically reducing data size while preserving gradient estimation accuracy.
2. **EFB (Exclusive Feature Bundling)**:
   * High-dimensional sparse feature spaces rarely take non-zero values simultaneously. EFB bundles mutually exclusive features into single dense bins, reducing the effective feature dimension.

##### C. Handling of High-Cardinality Categorical Features
* **LightGBM**: Natively sorts categorical levels based on the cumulative target sum and finds the optimal subset split in $O(k \log k)$ without creating sparse one-hot encoded columns.
* **XGBoost**: Historically required pre-encoding (Target Encoding or One-Hot Encoding) before building trees.

#### 3. Which One Should You Choose?

* **Use LightGBM** when:
  * You are working with large tabular datasets ($>100\text{k}$ to millions of rows).
  * You need fast training cycles, low RAM usage, and rapid hyperparameter experimentation.
* **Use XGBoost** when:
  * You have smaller, dense datasets where level-wise regularization prevents overfitting.
  * You require advanced GPU acceleration features or specialized custom objective functions.

In our NYC Taxi project (~2.4M rows), **LightGBM** achieved equivalent accuracy to XGBoost ($R^2 \approx 0.952$) while training in **half the time (12s vs. 24s)**.

---

## 21. Leaf-Wise Tree Growth, Histogram Binning, and L1 Objective Optimization in GBDT Regressors

### Question:
> What do "non-linear interactions", "leaf-wise tree growth", "histogram binning", and "L1 objective optimization (`regression_l1`)" mean in LightGBM and XGBoost models?

### Answer:

#### 1. Non-Linear Spatial & Temporal Interactions
In metropolitan transportation, feature relationships are rarely additive or linear ($y \ne w_1 X_1 + w_2 X_2$):
* At **3:00 AM on a highway**, a 5-mile trip takes **6 minutes** ($50\text{ mph}$).
* At **5:30 PM across Midtown / Queensboro Bridge**, a 5-mile trip takes **45 minutes** ($6.6\text{ mph}$).
* The effect of `trip_distance` on duration varies drastically depending on the non-linear interaction with `pickup_hour`, `PULocationID`, and `is_rush_hour`. Tree ensembles capture these complex conditional interactions naturally through hierarchical branching without requiring manual polynomial feature expansion.

#### 2. Leaf-Wise (Best-First) vs. Level-Wise (Depth-First) Tree Growth
* **Level-Wise Tree Growth (Traditional / XGBoost default)**:
  Splits all nodes at depth $d$ before moving to depth $d+1$, resulting in balanced, symmetrical trees. This can waste computational resources expanding shallow nodes that contribute little error reduction.
* **Leaf-Wise Tree Growth (LightGBM)**:
  Scans all current leaves across the tree and splits **only the single leaf that achieves the maximum reduction in loss (maximum split gain)**, regardless of tree depth.
* **Benefit**: Leaf-wise growth constructs deeper, asymmetric sub-trees specifically in high-complexity regions of the feature space (e.g., congested cross-borough bridge corridors), achieving lower loss with fewer total leaves.

#### 3. Histogram Binning
* Instead of sorting continuous floating-point features (such as continuous GPS coordinates or Haversine distances) across millions of rows to find exact split boundaries, LightGBM quantizes continuous values into **256 discrete integer buckets (histograms)**.
* **Benefits**:
  1. Reduces memory consumption by $80\%$.
  2. Accelerates node split evaluation by over $10\times$.
  3. Acts as an implicit regularizer, preventing trees from overfitting to microscopic GPS jitter.

#### 4. L1 Objective Optimization (`regression_l1` vs. `regression_l2` / MSE)
* **The Pitfall of L2 / MSE Loss**:
  Standard regression minimizes Mean Squared Error ($\text{Loss} = \sum (y_i - \hat{y}_i)^2$). Because errors are squared, extreme anomalies (e.g., a parked taxi with a running meter causing a \$30 error) exert $30^2 = 900\times$ more penalty than a \$1 error, forcing the model to bias predictions away from the normal 95% of trips.
* **What `regression_l1` Does**:
  Minimizes Mean Absolute Error directly ($\text{Loss} = \sum |y_i - \hat{y}_i|$), predicting the conditional **median** rather than the mean.
* **Business Alignment**:
  Passengers and operations teams evaluate quotes in actual dollars and minutes (MAE). By setting `objective="regression_l1"`, the algorithm minimizes the **exact metric we evaluate and ship in production**, rather than optimizing squared error and hoping it correlates with absolute error.

---

# Part VI: Model Evaluation, Advanced Metrics & Interpretability

## 22. Regression Evaluation Metrics: MAE, RMSE, MAPE, and R²

### Question:
> What do the metrics MAE, RMSE, MAPE (%), and R2 measure?

### Answer:
Here is a breakdown of what **MAE**, **RMSE**, **MAPE (%)**, and **$R^2$** measure, how they are calculated, and how to interpret them in the context of our taxi fare and duration models:

#### Summary Comparison Table
| Metric | Full Name | Formula | Units | What It Tells You in Simple Terms |
|---|---|---|---|---|
| **MAE** | Mean Absolute Error | $\frac{1}{N}\sum |y_i - \hat{y}_i|$ | Dollars ($\$$) / Mins | **Average magnitude of error** across all trips. |
| **RMSE** | Root Mean Squared Error | $\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$ | Dollars ($\$$) / Mins | Error metric that **heavily penalizes large mistakes**. |
| **MAPE** | Mean Absolute Percentage Error | $\frac{100\%}{N}\sum |\frac{y_i - \hat{y}_i}{y_i}|$ | Percentage ($\%$) | **Relative error** proportional to the trip size. |
| **$R^2$** | Coefficient of Determination | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Unitless ($-\infty$ to $1.0$) | **% of target variance explained** by the model. |

#### 1. MAE (Mean Absolute Error)
* **What it measures**: The average absolute dollar (or minute) difference between predicted and actual values.
* **Intuition**: *"On average, how many dollars is our prediction off by?"*
* **Example in Our Project**: LightGBM Fare MAE = **$\$1.26$**, meaning our fare prediction is off by $\$1.26$ on average.
* **Key Strength**: Intuitive to explain to stakeholders and robust to extreme outliers.

#### 2. RMSE (Root Mean Squared Error)
* **What it measures**: The square root of the average squared errors.
* **Intuition**: Squaring errors means that **a $\$10$ error hurts 100 times more than a $\$1$ error**.
* **Example in Our Project**: LightGBM Fare RMSE = **$\$2.96$**.
* **Key Insight**: If $\text{RMSE} \gg \text{MAE}$, it signals that the model makes occasional very large errors (e.g. on extreme airport trips or unusual traffic gridlocks).

#### 3. MAPE (Mean Absolute Percentage Error)
* **What it measures**: The error expressed as a percentage of the actual trip value.
* **Intuition**: *"What percentage of the true fare is our error?"*
  * A $\$2.00$ error on a $\$10.00$ short ride is a **$20\%$ error**.
  * The exact same $\$2.00$ error on a $\$100.00$ airport ride is only a **$2\%$ error**.
* **Example in Our Project**: LightGBM Fare MAPE = **$10.48\%$**, meaning on average our predictions are within $\approx 10\%$ of the true fare.

#### 4. $R^2$ (Coefficient of Determination)
* **What it measures**: The proportion of target variance explained by the model compared to a naive model that always predicts the dataset average ($\bar{y}$).
* **Intuition**: 
  * **$R^2 = 1.0$**: Perfect prediction ($0$ error).
  * **$R^2 = 0.0$**: Performs no better than predicting the training mean (Trivial Baseline).
  * **$R^2 < 0.0$**: Worse than predicting the mean.
* **Example in Our Project**:
  * LightGBM Fare $R^2 = \mathbf{0.9524}$ ($95.2\%$ of all variation in NYC taxi fares is successfully explained by our engineered features).
  * LightGBM Duration $R^2 = \mathbf{0.8218}$ ($82.2\%$ of variation in duration is explained).

---

## 23. Advanced Regression Evaluation Metrics & The Role of ROC / REC Curves in Continuous Modeling

### Question:
> What other metrics are commonly used to evaluate regression model families beyond MAE, RMSE, MAPE, and $R^2$? Can the ROC (Receiver Operating Characteristic) curve be used for regression?

### Answer:

#### 1. Additional Standard Regression Metrics
Beyond our primary leaderboard metrics (MAE, RMSE, MAPE, $R^2$), several complementary metrics are widely applied in industrial regression benchmarks:

1. **WAPE (Weighted Absolute Percentage Error)**:
   $$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i}$$
   Unlike MAPE (which divides each individual error by $y_i$, causing division-by-zero or extreme spikes on small \$2.50 minimum-fare rides), WAPE divides the sum of absolute errors by the total volume. It provides a robust, volume-weighted percentage error for revenue modeling.

2. **Median Absolute Error (MedAE / MedianAE)**:
   $$\text{MedAE} = \text{median}(|y_1 - \hat{y}_1|, |y_2 - \hat{y}_2|, \dots, |y_n - \hat{y}_n|)$$
   Completely immune to extreme outliers in the dataset. While our LightGBM Fare MAE is \$1.42, its MedianAE is ~\$0.85, showing that half of all rides are predicted within 85 cents.

3. **MSLE (Mean Squared Logarithmic Error)**:
   $$\text{MSLE} = \frac{1}{n} \sum (\log(1 + y_i) - \log(1 + \hat{y}_i))^2$$
   Penalizes under-predictions more heavily than over-predictions and scales well when targets span multiple orders of magnitude.

4. **Max Error**:
   $$\text{Max Error} = \max |y_i - \hat{y}_i|$$
   Identifies the absolute worst-case failure mode in the evaluation holdout (useful for auditing edge cases, such as outlier out-of-state flat rates).

5. **Explained Variance Score (EVS)**:
   $$\text{EV} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$$
   Measures the proportion of variance explained while ignoring systematic mean offsets (unlike $R^2$, which penalizes biased means).

---

#### 2. Can ROC Curves be Used for Regression?
* **Why Traditional ROC Does Not Apply**:
  The **ROC (Receiver Operating Characteristic)** curve and **AUC (Area Under the Curve)** are fundamentally designed for **binary classification tasks** ($y \in \{0, 1\}$). ROC plots the **True Positive Rate (Sensitivity)** against the **False Positive Rate (1 - Specificity)** across varying classification decision thresholds ($p \in [0, 1]$). Because continuous regression targets (such as \$17.50 fare or 24.3 minutes) do not have binary "positives" or "negatives", traditional ROC curves cannot be plotted directly.

* **How the Concept is Adapted to Regression**:
  1. **REC Curves (Regression Error Characteristic)**:
     The formal regression equivalent of the ROC curve. The $x$-axis represents an **error tolerance threshold ($\epsilon$)**, and the $y$-axis represents the **percentage of predictions within that tolerance** ($|y_i - \hat{y}_i| \le \epsilon$).
     * For example: at $\epsilon = 3\text{ minutes}$, LightGBM achieves $86\%$ cumulative accuracy; at $\epsilon = 5\text{ minutes}$, it reaches $96\%$.
     * The Area Over the Curve (AOC) in an REC plot is mathematically proportional to the Mean Absolute Error.
  2. **Binned Binary Classification**:
     Continuous residuals can be thresholded into operational binary flags (e.g., $1 = \text{Severe Delay (>15 min error)}$, $0 = \text{On-Time Arrival}$), allowing standard classification ROC curves to be constructed for delay risk detection.

---

## 24. Feature Importance in Tree Ensembles: Split Gain (Loss Reduction) vs. Split Count (Frequency)

### Question:
> In feature importance analysis, what is the meaning of "Split Gain" (`% Split Gain`), and how does it differ from "Split Count"?

### Answer:

#### 1. Split Count vs. Split Gain
Tree-based models evaluate feature importance through two distinct criteria:

| Metric | Calculation | Strengths & Weaknesses |
|---|---|---|
| **Split Count (Frequency / `importance_type="split"`)** | The raw number of times a feature was chosen to split a decision node across all trees in the ensemble. | Biased toward continuous features with many distinct values, which may split frequently while contributing very little to actual prediction accuracy. |
| **Split Gain (Loss Reduction / `importance_type="gain"`)** | The total mathematical reduction in training loss achieved whenever a feature is chosen as a split node. | **Industry Gold Standard**: Directly measures how much predictive power and error reduction a feature contributes to the final model. |

#### 2. Mathematical Definition of Split Gain
For every split node $s$ using feature $j$, the Information Gain ($\Delta \text{Loss}$) is computed as:
$$\Delta \text{Loss}(s) = \text{Loss}_{\text{parent}} - (\text{Loss}_{\text{left child}} + \text{Loss}_{\text{right child}})$$

The total gain for feature $j$ is summed across all trees:
$$\text{Total Gain}(X_j) = \sum_{s \in \text{Splits}(X_j)} \Delta \text{Loss}(s)$$

To represent this intuitively, the gains are normalized across all $K$ input features to produce the **Percentage Split Gain**:
$$\% \text{ Split Gain}(X_j) = \frac{\text{Total Gain}(X_j)}{\sum_{k=1}^K \text{Total Gain}(X_k)} \times 100\%$$

#### 3. Interpretation in our NYC Taxi Pipeline
* **`trip_distance` (43.0% Gain on Duration / 26.0% on Fare)**:
  $43\%$ of the entire duration model's predictive ability is driven directly by road distance splits.
* **`manhattan_distance` (32.1% Gain on Fare / 2.1% on Duration)**:
  Because Manhattan taxi fares are strictly metered on the rectangular street grid, grid distance accounts for almost a third of total fare explanation.
* **`pickup_hour / cos_hour` (8.0% Gain on Duration vs. 1.5% on Fare)**:
  Time-of-day features strongly influence duration (traffic congestion shifts travel speed), but have minimal impact on fare since yellow cab base rates are fixed per mile regardless of speed.

---

# Part VII: Production Serving, API Design & System SLAs

## 25. Downstream Model Integration & API Serving Workflow

### Question:
> How will the output of the FE pipeline (pipeline, X_train_features) be used when we start to train the different AI models?

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

## 26. Pickup Datetime Input Handling & ISO Format in Production UIs

### Question:
> Do we really need the `Pickup datetime (ISO)` checkbox/input, alongside its default value `2022-05-20T14:30:00` in the UI?

### Answer:
The backend API requires an ISO-8601 string (`YYYY-MM-DDTHH:MM:SS`) in the request JSON (`tpep_pickup_datetime`) because timestamp parsing extracts multiple cyclical and calendar features:
1. **Hour of Day (`pickup_hour`)** & **Rush Hour Flag (`is_rush_hour`)**: Captures peak commuting vs. nighttime traffic speeds.
2. **Cyclical Features (`sin_hour`, `cos_hour`, `sin_dayofweek`, `cos_dayofweek`)**: Preserves smooth periodic continuity across midnight and Sunday–Monday boundaries.
3. **Calendar Slices (`is_weekend`, `is_holiday`)**: Accounts for Memorial Day weekend traffic variations.

However, asking end users to manually type a raw ISO string like `2022-05-20T14:30:00` is error-prone. In modern web interfaces (e.g. Streamlit), the standard UX pattern is:
* Provide dedicated, native **Date (`st.date_input`)** and **Time (`st.time_input`)** pickers.
* Combine them on the client side into the required ISO format (`datetime.combine(date, time).isoformat()`) before dispatching the payload to `/predict`.

---

## 27. Service Level Agreements (SLA) and In-Process Latency Benchmarking in Machine Learning APIs

### Question:
> What is a Service Level Agreement (SLA) in machine learning engineering, and how does in-process local benchmarking compare to end-to-end network API latency?

### Answer:

#### 1. Definition of Service Level Agreement (SLA)
In software engineering, cloud computing, and MLOps, a **Service Level Agreement (SLA)** is the formal operational contract defining the minimum acceptable quality and performance standards for a production service.

* **In Machine Learning Serving**: Latency SLAs dictate the maximum allowable time (e.g., $P_{99} < 100\text{ ms}$) within which an inference endpoint must return a prediction quote.
* **User Experience Objective**: In ride-hailing and e-commerce applications, quotes must return in $<100\text{ ms}$ to prevent noticeable interface lag or checkout friction.

#### 2. In-Process Benchmarking vs. End-to-End Network Latency
When evaluating machine learning inference speed, there are two distinct measurement layers:

1. **In-Process Model Latency (1.40 ms in our project)**:
   * Measures pure internal Python / C++ execution time (`model.predict_single(features)`).
   * Evaluated by timing 1,000 in-memory scoring cycles on local developer hardware without network overhead.
   * Proves that the binary tree evaluation paths in LightGBM execute in near-instantaneous sub-2ms time.

2. **End-to-End Network Request Latency (3.0–5.0 ms in Docker)**:
   * Includes the full HTTP lifecycle:
     $$\text{Client HTTP Request} \longrightarrow \text{FastAPI Network I/O} \longrightarrow \text{Pydantic Schema Validation} \longrightarrow \text{Model In-Process Scoring (1.40ms)} \longrightarrow \text{JSON Serialization} \longrightarrow \text{Client Response}$$
   * Even with full HTTP serialization and Pydantic data validation, the entire request round-trip operates at ~3–5 ms—comfortably below any commercial $100\text{ ms}$ latency budget.

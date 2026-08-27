# NYC Taxi Fare & Trip Duration Prediction — Presentation Script (English)

**Demo Day Presentation · 18 Slides · 5 Presenters · Total Time: 20:00 (including 2:30 Live Demo)**

---

## 1. Presenter Summary & Time Allocations

| Presenter | Allocated Slides | Section Responsibilities | Total Time |
|---|---|---|---:|
| **Keyneth Lara** | Slides 1–4 | Project Overview & Requirements | **3:45** |
| **William Vera** | Slides 5–8 | Architecture, Data Cleaning & Feature Engineering | **4:20** |
| **Marcos Villabasa** | Slides 9–13 | Model Exploration, Leaderboard & Interpretability | **6:00** |
| **Mauricio Mora** | Slide 14 | Live Docker Demonstration (FastAPI & Streamlit UI) | **2:30** |
| **Néstor Mamani** | Slides 15–18 | Assumptions, Roadmap, Key Takeaways & Q&A | **3:15** |
| **Total Deck** | **Slides 1–18** | **Full Project Storyline & Live Demonstration** | **20:00** |

---

## 2. Full Running Order

| # | Slide Title | Section | Presenter | Time |
|---|---|---|---|---:|
| **01** | How much? How long? | 1. Project Overview | Keyneth Lara | 0:40 |
| **02** | The meter answers too late | 1. Project Overview | Keyneth Lara | 0:50 |
| **03** | Dataset Scale & EDA | 1. Project Overview | Keyneth Lara | 1:10 |
| **04** | Engineering Requirements | 2. Requirements | Keyneth Lara | 1:05 |
| **05** | System Architecture | 3. Scope & Solutions | William Vera | 1:15 |
| **06** | Data Cleaning & Split Rationale | 3. Scope & Solutions | William Vera | 1:15 |
| **07** | Feature Engineering Pipeline | 3. Scope & Solutions | William Vera | 1:15 |
| **08** | Feature Contract & Leakage Prevention | 3. Scope & Solutions | William Vera | 0:35 |
| **09** | Model Exploration Strategy | 4. Metrics & Results | Marcos Villabasa | 1:10 |
| **10** | Comprehensive Model Comparison | 4. Metrics & Results | Marcos Villabasa | 1:20 |
| **11** | Visual Metric Comparisons | 4. Metrics & Results | Marcos Villabasa | 1:15 |
| **12** | Why LightGBM Won | 4. Metrics & Results | Marcos Villabasa | 1:05 |
| **13** | Feature Importance & Gain Analysis | 4. Metrics & Results | Marcos Villabasa | 1:10 |
| **14** | **Live Demonstration Walkthrough** *(Demo)* | 5. Project Demo | Mauricio Mora | **2:30** |
| **15** | Assumptions & Tradeoffs | 6. Conclusions & Next Steps | Néstor Mamani | 1:00 |
| **16** | Next Steps & Roadmap | 6. Conclusions & Next Steps | Néstor Mamani | 1:00 |
| **17** | Key Takeaways | 6. Conclusions & Next Steps | Néstor Mamani | 1:00 |
| **18** | Thank You & Q&A | 6. Conclusions & Next Steps | Néstor Mamani | 0:15 |

---

## 3. Slide-by-Slide Speaking Scripts

### Slide 01: How much? How long?
* **Presenter**: Keyneth Lara
* **Section**: 1. Project Overview
* **Time**: 0:40

> "Good afternoon, everyone. We are the NYC Taxi ML Engineering team. Two fundamental questions define urban mobility: *How much will this ride cost?* and *How long will it take?* For decades, yellow taxi passengers had to wait until the end of the trip to find out. Today, we are presenting an end-to-end machine learning system that answers both questions simultaneously, with sub-2 millisecond latency, before the passenger even steps into the vehicle."

---

### Slide 02: The meter answers too late
* **Presenter**: Keyneth Lara
* **Section**: 1. Project Overview
* **Time**: 0:50

> "The traditional taximeter operates at drop-off ($t_1$). You only learn the final fare after the trip is completed. However, ride-hailing apps transformed customer expectations—passengers demand transparent, upfront price certainty at pickup ($t_0$). The NYC Taxi and Limousine Commission now permits upfront pricing, but implementing it requires an accurate, real-time predictive model. Our entire architecture is constrained to this exact second: we predict using only what is known at pickup, strictly eliminating any in-transit telemetry."

---

### Slide 03: Dataset Scale & EDA
* **Presenter**: Keyneth Lara
* **Section**: 1. Project Overview
* **Time**: 1:15

> "To train our models, we ingested the official NYC TLC Yellow Taxi dataset for May 2022, representing 3.59 million trip records across 265 taxi zones. Our exploratory data analysis revealed three critical characteristics: first, strong right-skewed target distributions with mean fares of $15.15 and durations of 16.5 minutes; second, distinct temporal pulses corresponding to morning and evening weekday rush hours; and third, spatial concentration where airport hubs and Midtown Manhattan account for nearly half of all trips. EDA also uncovered negative fares, zero-passenger anomalies, and post-trip leakage columns that required strict filtering."

---

### Slide 04: Engineering Requirements
* **Presenter**: Keyneth Lara
* **Section**: 2. Requirements
* **Time**: 1:10

> "Before writing code, we established strict engineering requirements. Functionally, the prototype must deliver simultaneous dual-target predictions for fare and duration, significantly outperforming naive mean heuristics and establishing a tight error margin for real-time quotation. Crucially, it must enforce zero data leakage. Non-functionally, the API prototype must meet a responsive sub-100 millisecond latency target, be fully containerized with Docker Compose, and serve an interactive user interface alongside a clean REST endpoint. William will now present our system architecture."

---

### Slide 05: System Architecture
* **Presenter**: William Vera
* **Section**: 3. Scope & Solutions
* **Time**: 1:20

> "Thank you, Keyneth. Here is our end-to-end system architecture. On the left is our offline training pipeline divided into three clear stages: Stage 1 for Data Ingestion and Cleaning, Stage 2 for Feature Engineering, and Stage 3 for Model Training and Evaluation. The crucial design decision is the dashed boundary down the center: only one single self-contained artifact—`model.pkl` (4.2 MB)—crosses into our Docker container runtime on the right. There, our FastAPI microservice serves dual predictions via REST, while our Streamlit dashboard acts as an interactive client communicating over HTTP. This guarantees zero training bloat in production."

---

### Slide 06: Data Cleaning & Split Rationale
* **Presenter**: William Vera
* **Section**: 3. Scope & Solutions
* **Time**: 1:20

> "Our data cleaning pipeline filters 3.59 million raw records down to 3.30 million high-quality rows, achieving a 92.04% retention rate. We enforce physical bounds: passenger counts between 1 and 9, durations between 1 and 300 minutes, distances between 0.1 and 150 miles, and fares between $2.50 and $500. Most importantly, we implemented a strict chronological temporal split at May 23, 2022. Random splitting would allow future pricing trends to contaminate past training. Our 3-week train (2.40M) and 1-week test (900k) split simulates authentic production deployment."

---

### Slide 07: Feature Engineering Pipeline
* **Presenter**: William Vera
* **Section**: 3. Scope & Solutions
* **Time**: 1:20

> "Moving to our feature transformations: from 6 raw inputs, our pipeline generates 29 high-signal engineered features across four modular transformer stages. First, temporal: cyclical sine and cosine encodings for hour-of-day and day-of-week, plus rush-hour and Memorial Day holiday flags. Second, spatial: we extract zone centroid coordinates from the TLC shapefile to compute Haversine straight-line distance, Manhattan grid distance, and airport indicators. Third, categorical: Bayesian smoothed target encodings for pickup, dropoff, and rate codes, fitted strictly on training data. And fourth, raw numerical metadata."

---

### Slide 08: Feature Contract & Leakage Prevention
* **Presenter**: William Vera
* **Section**: 3. Scope & Solutions
* **Time**: 0:35

> "To guarantee zero data leakage, we enforce a strict feature contract. Only 6 pre-ride inputs are allowed at inference time, while 8 post-trip columns—including dropoff time, tolls, tips, and fare totals—are strictly banned and dropped. Our only post-trip assumption is `trip_distance`, which is estimated upfront using GPS routing engines like OSRM. Marcos will now present our model exploration results."

---

### Slide 09: Model Exploration Strategy
* **Presenter**: Marcos Villabasa
* **Section**: 4. Metrics & Results
* **Time**: 1:15

> "Thank you, William. We benchmarked three distinct model families. Family 1 established baselines: a Trivial Mean regressor and a depth-constrained Decision Tree. Family 2 explored gradient boosted decision trees: LightGBM and XGBoost with histogram-based binning. Family 3 implemented deep learning: a multi-layer perceptron with standard scaling, batch normalization, and early stopping. Every model was trained on the identical 2.40 million training records and evaluated on the unseen 900k temporal holdout set."

---

### Slide 10: Comprehensive Model Comparison
* **Presenter**: Marcos Villabasa
* **Section**: 4. Metrics & Results
* **Time**: 1:30

> "Here is our master leaderboard evaluated on the 900,000 unseen test rides. The Trivial Mean baseline had an MAE of $8.85 for fare and 9.27 minutes for duration, with zero predictive power. Decision Trees improved significantly to $1.46 and 3.94 minutes. LightGBM emerged as our top performer, achieving a Fare MAE of $1.42 and Duration MAE of 3.76 minutes, with an R² of 0.952 on fares and 0.797 on duration. XGBoost performed similarly at $1.45, while the MLP neural network trailed at $2.21 MAE. LightGBM achieved the best accuracy across every single metric."

---

### Slide 11: Visual Metric Comparisons
* **Presenter**: Marcos Villabasa
* **Section**: 4. Metrics & Results
* **Time**: 1:20

> "This slide provides a direct graphical comparison of our two core evaluation metrics across all model families. On the left, the MAE benchmark chart compares fare and duration error side-by-side, showing how LightGBM achieves the lowest prediction error on both targets: just $1.42 for fare and 3.76 minutes for duration. On the right, the R² goodness-of-fit chart highlights variance explained: gradient boosted trees capture nearly 80% of trip duration variance, far outperforming the neural network's 56.6%, while maintaining over 95% variance explained on fares."

---

### Slide 12: Why LightGBM Won
* **Presenter**: Marcos Villabasa
* **Section**: 4. Metrics & Results
* **Time**: 1:10

> "Three concrete factors made LightGBM our production choice. First, accuracy: its leaf-wise tree growth and histogram binning captured non-linear cross-borough traffic interactions better than linear or neural models. Second, speed: 1.40 milliseconds in-process execution on a standard laptop means predictions execute virtually instantaneously, allowing our API to easily handle concurrent traffic with low CPU overhead. And third, footprint: the serialized artifact is only 4.2 MB, loads in 120 milliseconds on container startup, and requires zero GPU infrastructure."

---

### Slide 13: Feature Importance & Gain Analysis
* **Presenter**: Marcos Villabasa
* **Section**: 4. Metrics & Results
* **Time**: 1:15

> "Looking at feature importance: distance features dominate the model, accounting for roughly 85% of total split gain for fare and 67% for duration. Specifically, Manhattan distance is the strongest predictor for fares at 32.1%, while metered trip distance drives 43% of duration. Time-of-day features contribute an additional 8% to duration to capture rush-hour congestion, but under 2% to fares. Crucially, our automated leakage audit passed: no single feature exceeds 65% dominance, confirming healthy multi-signal diversity. Mauricio will now lead our live system demonstration."

---

### Slide 14: Live Demonstration Walkthrough (Demo)
* **Presenter**: Mauricio Mora
* **Section**: 5. Project Demo
* **Time**: 2:30

> **[Transition to Live Demo]**: "Thank you, Marcos. I am now switching screen share to our live Dockerized deployment to demonstrate the end-to-end user experience across both our FastAPI backend and our Streamlit dashboard."
> 
> **[1. FastAPI Documentation — Swagger UI]**: "First, here is our live FastAPI service at `http://localhost:8000/docs`. I'll hit the `/health` endpoint to verify the service status and confirm our 4.2 MB serialized model artifact is loaded in memory. Next, let's test the `/predict` endpoint with a real-world trip payload: pickup at Times Square (Zone 230), dropoff at JFK Airport (Zone 132), pickup datetime at 6:30 PM on a Wednesday, 1 passenger, and Ratecode 2. When I click 'Execute', you can see the response returns in just 3 milliseconds: exactly predicting the $52.00 flat-rate tariff and estimating a 48-minute trip duration during evening rush hour. If we pass invalid inputs, like a negative distance or an invalid zone ID, Pydantic's strict data validation instantly rejects the request with a structured 422 error."
> 
> **[2. Streamlit Geospatial Dashboard]**: "Now, switching to our interactive Streamlit application at `http://localhost:8501`. On the Single Trip Estimator tab, dispatchers and passengers can select any of NYC's 265 taxi zones from dropdowns. As I choose pickup and dropoff points, the UI dynamically renders the geodesic route on the map, calculates Manhattan and Haversine distances, and updates dual gauges for fare and duration in real time. On the Citywide Heatmap tab, selecting a single pickup zone triggers real-time batch inference across all 265 destination zones simultaneously, rendering an interactive choropleth map that visually demonstrates cost and travel time gradients across the entire metropolitan area."
> 
> **[Conclusion of Demo]**: "As you can see, the trained LightGBM models deliver sub-millisecond predictions, robust input validation, and intuitive geospatial visualization in a completely isolated production container. Néstor will now present our assumptions and roadmap."

---

### Slide 15: Assumptions & Tradeoffs
* **Presenter**: Néstor Mamani
* **Section**: 6. Conclusions & Next Steps
* **Time**: 1:00

> "Thank you, Mauricio. We document three operational assumptions and engineering tradeoffs. First, temporal scope: our models are trained exclusively on 3.5 million records from May 2022, capturing late spring patterns but unobserved to annual seasonality shifts like summer vacation drops or winter blizzards. Second, dynamic weather: sudden thunderstorms cause speed reductions that our calendar indicators cannot observe without live meteorological telemetry. Third, real-time demand and traffic shocks: while yellow cab fares follow regulated formulas, unscheduled road closures introduce travel duration variance."

---

### Slide 16: Next Steps & Roadmap
* **Presenter**: Néstor Mamani
* **Section**: 6. Conclusions & Next Steps
* **Time**: 1:00

> "Our engineering roadmap outlines three potential production enhancements. First, embedding an open-source OSRM routing container into our Docker Compose stack to compute live routing geometry and road distances before pickup. Second, implementing MLOps telemetry with Prometheus and Evidently AI to proactively detect data drift, seasonal shifts, and latency percentiles. And third, establishing automated monthly TLC data ingestion and champion-challenger retraining pipelines using Apache Airflow."

---

### Slide 17: Key Takeaways
* **Presenter**: Néstor Mamani
* **Section**: 6. Conclusions & Next Steps
* **Time**: 1:00

> "To conclude, our project delivered three core achievements. First, rigorous engineering: by enforcing strict temporal splits and automated feature contracts, we guaranteed zero data leakage and authentic evaluation metrics. Second, high-performance modeling: LightGBM delivered $1.42 Fare MAE and 3.76 min Duration MAE with sub-2 millisecond latency. And third, production-ready architecture: our decoupled microservices and self-contained artifact ensure seamless, isolated deployment."

---

### Slide 18: Thank You & Q&A
* **Presenter**: Néstor Mamani
* **Section**: 6. Conclusions & Next Steps
* **Time**: 0:15

> "How much? How long? Answered before the ride starts. On behalf of our entire team—thank you! We are now open for questions and feedback."

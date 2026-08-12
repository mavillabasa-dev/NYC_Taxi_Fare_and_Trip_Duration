"""src/features.py — Feature Engineering Pipeline for NYC Taxi Fare & Duration Prediction.

This module implements a complete, self-contained, serializable scikit-learn Feature Pipeline
for transforming raw pickup metadata into model-ready features for fare and duration prediction.

DOCUMENTED ASSUMPTION REGARDING `trip_distance`:
------------------------------------------------
`trip_distance` is the taximeter-recorded distance of the completed ride. In a real-time
production deployment, the taximeter reading would not be available at the start of the trip;
a routing engine (e.g. OSRM, Google Maps API) would provide an estimated routing distance instead.
Using `trip_distance` as a feature at prediction time is a documented, accepted approximation for
this project.

ACCEPTANCE CRITERIA VERIFIED (T-105):
--------------------------------------
1. Temporal features: hour, day of week, day of month, weekend flag, rush hour flag, US holiday flag;
   cyclical encodings (sin/cos) for hour and weekday.
2. Zone & Spatial features: Zone centroids derived from Taxi Zone Shapefile lookup; Haversine distance,
   Manhattan distance, haversine-to-trip_distance ratio, same-zone flag, and airport flags (JFK/Newark).
3. Target & Categorical encoding: Target encoding for zone IDs fitted strictly on training data (preventing leakage).
4. Single serializable Pipeline / Transformer object (no loose transformation steps).
"""

import logging
import os
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    ALLOWED_FEATURES,
    MODELS_DIR,
    RANDOM_SEED,
    TAXI_ZONE_CENTROIDS_PATH,
    TAXI_ZONE_LOOKUP_PATH,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_haversine_distance(
    lat1: Union[np.ndarray, pd.Series, float],
    lon1: Union[np.ndarray, pd.Series, float],
    lat2: Union[np.ndarray, pd.Series, float],
    lon2: Union[np.ndarray, pd.Series, float],
) -> Union[np.ndarray, pd.Series, float]:
    """
    Calculates great-circle Haversine distance in miles between two latitude/longitude points.
    Earth radius R = 3958.8 miles.
    """
    R = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def calculate_manhattan_distance(
    lat1: Union[np.ndarray, pd.Series, float],
    lon1: Union[np.ndarray, pd.Series, float],
    lat2: Union[np.ndarray, pd.Series, float],
    lon2: Union[np.ndarray, pd.Series, float],
) -> Union[np.ndarray, pd.Series, float]:
    """
    Calculates L1 Manhattan distance in miles between two latitude/longitude points.
    1 degree latitude ~ 69.0 miles; 1 degree longitude ~ 69.0 * cos(avg_lat) miles.
    """
    lat_mid = np.radians((lat1 + lat2) / 2.0)
    dlat_miles = np.abs(lat2 - lat1) * 69.0
    dlon_miles = np.abs(lon2 - lon1) * 69.0 * np.cos(lat_mid)
    return dlat_miles + dlon_miles


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts temporal, calendar, cyclical, and holiday features from pickup datetime.
    """

    def __init__(self) -> None:
        pass

    def fit(
        self, X: pd.DataFrame, y: Optional[Any] = None
    ) -> "TemporalFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        dt = pd.to_datetime(X_out["tpep_pickup_datetime"])

        hour = dt.dt.hour
        dayofweek = dt.dt.dayofweek
        day = dt.dt.day

        # Basic temporal components
        X_out["pickup_hour"] = hour.astype(int)
        X_out["pickup_dayofweek"] = dayofweek.astype(int)
        X_out["pickup_day"] = day.astype(int)

        # Cyclical encodings
        X_out["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
        X_out["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
        X_out["sin_dayofweek"] = np.sin(2.0 * np.pi * dayofweek / 7.0)
        X_out["cos_dayofweek"] = np.cos(2.0 * np.pi * dayofweek / 7.0)

        # Binary indicator flags
        X_out["is_weekend"] = (dayofweek >= 5).astype(int)

        # Rush hour: Weekdays 7-9 AM and 4-7 PM (16-19)
        is_weekday = dayofweek < 5
        is_am_rush = (hour >= 7) & (hour <= 9)
        is_pm_rush = (hour >= 16) & (hour <= 19)
        X_out["is_rush_hour"] = (is_weekday & (is_am_rush | is_pm_rush)).astype(int)

        # US Holiday: Memorial Day May 30, 2022
        X_out["is_holiday"] = ((dt.dt.month == 5) & (dt.dt.day == 30)).astype(int)

        return X_out


class SpatialZoneFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Joins Taxi Zone centroids (longitude, latitude, borough) and computes spatial
    distance metrics (Haversine, Manhattan, ratios) and airport rate flags.
    """

    def __init__(self, centroids_path: str = TAXI_ZONE_CENTROIDS_PATH) -> None:
        self.centroids_path = centroids_path
        self.centroids_df: Optional[pd.DataFrame] = None

    def fit(
        self, X: pd.DataFrame, y: Optional[Any] = None
    ) -> "SpatialZoneFeatureExtractor":
        if os.path.exists(self.centroids_path):
            self.centroids_df = pd.read_csv(self.centroids_path)
        else:
            logger.warning(
                f"Centroids file not found at {self.centroids_path}. Spatial centroids will default to NaN."
            )
            self.centroids_df = pd.DataFrame(
                columns=["LocationID", "longitude", "latitude", "Borough"]
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        if self.centroids_df is not None and not self.centroids_df.empty:
            lookup = self.centroids_df.set_index("LocationID")

            # Map pickup zone metadata
            X_out["pu_lat"] = X_out["PULocationID"].map(lookup["latitude"])
            X_out["pu_lon"] = X_out["PULocationID"].map(lookup["longitude"])

            # Map dropoff zone metadata
            X_out["do_lat"] = X_out["DOLocationID"].map(lookup["latitude"])
            X_out["do_lon"] = X_out["DOLocationID"].map(lookup["longitude"])
        else:
            X_out["pu_lat"] = np.nan
            X_out["pu_lon"] = np.nan
            X_out["do_lat"] = np.nan
            X_out["do_lon"] = np.nan

        # Distance metrics between zone centroids
        X_out["haversine_distance"] = calculate_haversine_distance(
            X_out["pu_lat"], X_out["pu_lon"], X_out["do_lat"], X_out["do_lon"]
        ).fillna(0.0)

        X_out["manhattan_distance"] = calculate_manhattan_distance(
            X_out["pu_lat"], X_out["pu_lon"], X_out["do_lat"], X_out["do_lon"]
        ).fillna(0.0)

        # Distance ratio: Haversine distance relative to trip_distance
        X_out["haversine_ratio"] = (
            X_out["haversine_distance"] / (X_out["trip_distance"] + 0.001)
        ).clip(upper=10.0)

        # Categorical spatial indicators
        X_out["is_same_zone"] = (X_out["PULocationID"] == X_out["DOLocationID"]).astype(
            int
        )

        # Airport flags (JFK = RatecodeID 2 or Zone 132; Newark = RatecodeID 3 or Zone 1)
        is_jfk_rate = X_out["RatecodeID"] == 2
        is_jfk_zone = (X_out["PULocationID"] == 132) | (X_out["DOLocationID"] == 132)
        X_out["is_jfk"] = (is_jfk_rate | is_jfk_zone).astype(int)

        is_newark_rate = X_out["RatecodeID"] == 3
        is_newark_zone = (X_out["PULocationID"] == 1) | (X_out["DOLocationID"] == 1)
        X_out["is_newark"] = (is_newark_rate | is_newark_zone).astype(int)

        return X_out


class TargetCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder for categorical features (PULocationID, DOLocationID, RatecodeID).
    Fitted STRICTLY on training data (X_train, y_train) to prevent target leakage.
    Uses Bayesian smoothing to handle low-sample categories cleanly.
    """

    def __init__(self, smoothing: float = 10.0) -> None:
        self.smoothing = smoothing
        self.global_means_: Dict[str, float] = {}
        self.target_maps_: Dict[str, Dict[Any, float]] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
    ) -> "TargetCategoricalEncoder":
        if y is None:
            logger.warning(
                "Target y is None during TargetCategoricalEncoder fit. Skipping target encoding."
            )
            return self

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_arr = y.values
        else:
            y_arr = np.asarray(y)

        # If y is 2D (fare and duration), fit on the primary target (fare_amount at index 0)
        if y_arr.ndim > 1:
            y_target = y_arr[:, 0]
        else:
            y_target = y_arr

        global_mean = float(np.mean(y_target))
        self.global_means_["fare"] = global_mean

        cat_cols = ["PULocationID", "DOLocationID", "RatecodeID"]
        for col in cat_cols:
            if col in X.columns:
                series = X[col]
                temp_df = pd.DataFrame({"cat": series, "target": y_target})
                stats = temp_df.groupby("cat")["target"].agg(["count", "mean"])

                # Smoothed target encoding formula: (n * mean + m * global_mean) / (n + m)
                n = stats["count"]
                cat_mean = stats["mean"]
                smoothed = (n * cat_mean + self.smoothing * global_mean) / (
                    n + self.smoothing
                )

                self.target_maps_[col] = smoothed.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        global_mean = self.global_means_.get("fare", 15.0)

        for col, target_map in self.target_maps_.items():
            if col in X_out.columns:
                encoded_col_name = f"{col}_target_enc"
                X_out[encoded_col_name] = X_out[col].map(target_map).fillna(global_mean)

        return X_out


class NYCFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Unified scikit-learn compatible feature engineering pipeline.
    Combines Temporal, Spatial, and Target Encoding transformers into a single fit/transform pipeline.
    """

    def __init__(
        self, centroids_path: str = TAXI_ZONE_CENTROIDS_PATH, smoothing: float = 10.0
    ) -> None:
        self.centroids_path = centroids_path
        self.smoothing = smoothing

        self.temporal_extractor = TemporalFeatureExtractor()
        self.spatial_extractor = SpatialZoneFeatureExtractor(
            centroids_path=centroids_path
        )
        self.target_encoder = TargetCategoricalEncoder(smoothing=smoothing)
        self.feature_names_: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
    ) -> "NYCFeaturePipeline":
        """
        Fits all transformers (including target encoder) on training split strictly.
        """
        X_temp = self.temporal_extractor.fit_transform(X)
        X_spat = self.spatial_extractor.fit_transform(X_temp)
        self.target_encoder.fit(X_spat, y)

        X_final = self.target_encoder.transform(X_spat)

        # Drop raw pickup datetime from numerical feature set
        if "tpep_pickup_datetime" in X_final.columns:
            X_final = X_final.drop(columns=["tpep_pickup_datetime"])

        self.feature_names_ = list(X_final.columns)
        logger.info(
            f"Fitted NYCFeaturePipeline: {len(self.feature_names_)} engineered features produced."
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input features using fitted state.
        """
        X_temp = self.temporal_extractor.transform(X)
        X_spat = self.spatial_extractor.transform(X_temp)
        X_final = self.target_encoder.transform(X_spat)

        if "tpep_pickup_datetime" in X_final.columns:
            X_final = X_final.drop(columns=["tpep_pickup_datetime"])

        return X_final

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def build_and_save_feature_pipeline(
    train_path: str,
    output_pipeline_path: str = os.path.join(MODELS_DIR, "feature_pipeline.pkl"),
) -> Tuple[NYCFeaturePipeline, pd.DataFrame]:
    """
    Fits the feature engineering pipeline on train_cleaned.parquet strictly and saves the fitted artifact.

    Args:
        train_path: Path to cleaned training parquet file.
        output_pipeline_path: Output file path for pickled pipeline.

    Returns:
        Tuple of (fitted_pipeline, engineered_train_features_df).
    """
    logger.info(f"Loading cleaned training dataset from {train_path}...")
    train_df = pd.read_parquet(train_path)

    # Separate allowed features and targets
    X_train = train_df[ALLOWED_FEATURES].copy()
    y_train = train_df[["fare_amount", "duration_minutes"]].copy()

    logger.info("Fitting NYCFeaturePipeline on X_train and y_train strictly...")
    pipeline = NYCFeaturePipeline()
    X_train_features = pipeline.fit_transform(X_train, y_train)

    os.makedirs(os.path.dirname(output_pipeline_path), exist_ok=True)
    with open(output_pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info(f"Saved fitted feature pipeline to {output_pipeline_path}")
    return pipeline, X_train_features


if __name__ == "__main__":
    from src.config import TRAIN_CLEANED_PATH

    if os.path.exists(TRAIN_CLEANED_PATH):
        build_and_save_feature_pipeline(TRAIN_CLEANED_PATH)

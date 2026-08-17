"""api/app/model/predictor.py — Self-contained production model predictor for NYC Taxi API.

This module provides the standalone predictor class that bundles all feature transformations
and trained LightGBM estimators for both fare_amount and duration_minutes.

CRITICAL ARCHITECTURAL GUARANTEE:
---------------------------------
This class has ZERO dependencies on `src/`. It can be instantiated, pickled, unpickled,
and executed entirely inside the containerized API environment using only numpy, pandas,
and lightgbm.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _haversine_np(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculates great-circle Haversine distance in miles between coordinate arrays."""
    r_miles = 3958.8
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r_miles * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _manhattan_np(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculates L1 Manhattan distance in miles between coordinate arrays."""
    lat_mid = np.radians((lat1 + lat2) / 2.0)
    dlat_miles = np.abs(lat2 - lat1) * 69.0
    dlon_miles = np.abs(lon2 - lon1) * 69.0 * np.cos(lat_mid)
    return dlat_miles + dlon_miles


class SelfContainedTaxiModel:
    """
    Self-contained production model bundling feature transformations and LightGBM regressors.

    Takes raw input DataFrames containing:
    ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'passenger_count', 'RatecodeID', 'trip_distance']
    and returns a 2D numpy array of shape (N, 2) with columns [predicted_fare, predicted_duration_minutes].
    """

    def __init__(
        self,
        fare_model: Any,
        duration_model: Any,
        centroid_lookup: Dict[int, Tuple[float, float]],
        target_encodings: Dict[str, Dict[Any, float]],
        global_fare_mean: float = 15.15,
        feature_names: Optional[List[str]] = None,
        version: str = "1.0.0-lightgbm",
    ) -> None:
        self.fare_model = fare_model
        self.duration_model = duration_model
        self.centroid_lookup = centroid_lookup  # LocationID -> (latitude, longitude)
        self.target_encodings = target_encodings  # col_name -> {cat: smoothed_mean}
        self.global_fare_mean = global_fare_mean
        self.feature_names = feature_names
        self.version = version

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw request columns into the 29 engineered features required by LightGBM."""
        df_out = pd.DataFrame(index=df.index)

        # 1. Base input features
        pu_id = df["PULocationID"].values
        do_id = df["DOLocationID"].values
        ratecode = df["RatecodeID"].values
        passengers = df["passenger_count"].values
        trip_dist = df["trip_distance"].values.astype(float)
        vendor_id = (
            df["VendorID"].values
            if "VendorID" in df.columns
            else np.full(len(df), 2, dtype=int)
        )

        df_out["PULocationID"] = pu_id
        df_out["DOLocationID"] = do_id
        df_out["passenger_count"] = passengers
        df_out["RatecodeID"] = ratecode
        df_out["trip_distance"] = trip_dist
        df_out["VendorID"] = vendor_id

        # 2. Temporal & Cyclical features
        dt = pd.to_datetime(df["tpep_pickup_datetime"])
        hour = dt.dt.hour.values
        dayofweek = dt.dt.dayofweek.values
        day = dt.dt.day.values

        df_out["pickup_hour"] = hour
        df_out["pickup_dayofweek"] = dayofweek
        df_out["pickup_day"] = day

        df_out["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
        df_out["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
        df_out["sin_dayofweek"] = np.sin(2.0 * np.pi * dayofweek / 7.0)
        df_out["cos_dayofweek"] = np.cos(2.0 * np.pi * dayofweek / 7.0)

        df_out["is_weekend"] = (dayofweek >= 5).astype(int)

        is_weekday = dayofweek < 5
        is_am_rush = (hour >= 7) & (hour <= 9)
        is_pm_rush = (hour >= 16) & (hour <= 19)
        df_out["is_rush_hour"] = (is_weekday & (is_am_rush | is_pm_rush)).astype(int)
        df_out["is_holiday"] = ((dt.dt.month == 5) & (dt.dt.day == 30)).astype(int)

        # 3. Spatial Zone Centroids
        pu_lat = np.array(
            [self.centroid_lookup.get(int(z), (np.nan, np.nan))[0] for z in pu_id]
        )
        pu_lon = np.array(
            [self.centroid_lookup.get(int(z), (np.nan, np.nan))[1] for z in pu_id]
        )
        do_lat = np.array(
            [self.centroid_lookup.get(int(z), (np.nan, np.nan))[0] for z in do_id]
        )
        do_lon = np.array(
            [self.centroid_lookup.get(int(z), (np.nan, np.nan))[1] for z in do_id]
        )

        df_out["pu_lat"] = pu_lat
        df_out["pu_lon"] = pu_lon
        df_out["do_lat"] = do_lat
        df_out["do_lon"] = do_lon

        haversine = _haversine_np(pu_lat, pu_lon, do_lat, do_lon)
        haversine = np.nan_to_num(haversine, nan=0.0)
        manhattan = _manhattan_np(pu_lat, pu_lon, do_lat, do_lon)
        manhattan = np.nan_to_num(manhattan, nan=0.0)

        df_out["haversine_distance"] = haversine
        df_out["manhattan_distance"] = manhattan
        df_out["haversine_ratio"] = np.clip(haversine / (trip_dist + 0.001), 0.0, 10.0)
        df_out["is_same_zone"] = (pu_id == do_id).astype(int)

        is_jfk_rate = ratecode == 2
        is_jfk_zone = (pu_id == 132) | (do_id == 132)
        df_out["is_jfk"] = (is_jfk_rate | is_jfk_zone).astype(int)

        is_newark_rate = ratecode == 3
        is_newark_zone = (pu_id == 1) | (do_id == 1)
        df_out["is_newark"] = (is_newark_rate | is_newark_zone).astype(int)

        # 4. Smoothed Target Encodings
        pu_map = self.target_encodings.get("PULocationID", {})
        do_map = self.target_encodings.get("DOLocationID", {})
        rate_map = self.target_encodings.get("RatecodeID", {})

        df_out["PULocationID_target_enc"] = [
            pu_map.get(int(z), self.global_fare_mean) for z in pu_id
        ]
        df_out["DOLocationID_target_enc"] = [
            do_map.get(int(z), self.global_fare_mean) for z in do_id
        ]
        df_out["RatecodeID_target_enc"] = [
            rate_map.get(int(r), self.global_fare_mean) for r in ratecode
        ]

        if self.feature_names:
            df_out = df_out[self.feature_names]

        return df_out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts fare_amount and duration_minutes for given raw input DataFrame."""
        X_feat = self.transform_features(X)
        pred_fare = self.fare_model.predict(X_feat)
        pred_duration = self.duration_model.predict(X_feat)

        # Ensure predictions are non-negative
        pred_fare = np.clip(pred_fare, a_min=2.50, a_max=None)
        pred_duration = np.clip(pred_duration, a_min=0.5, a_max=None)

        return np.column_stack((pred_fare, pred_duration))

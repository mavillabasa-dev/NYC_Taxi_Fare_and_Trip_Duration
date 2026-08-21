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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Geodetic & Spatial Constants
EARTH_RADIUS_MILES = 3958.8
MILES_PER_DEGREE_LATITUDE = 69.0
MAX_HAVERSINE_RATIO = 10.0
EPSILON_DISTANCE = 0.001

# Airport Zone & Ratecode IDs
JFK_AIRPORT_ZONE_ID = 132
NEWARK_AIRPORT_ZONE_ID = 1
JFK_RATECODE_ID = 2
NEWARK_RATECODE_ID = 3

# Temporal & Rush Hour Constants
AM_RUSH_START_HOUR = 7
AM_RUSH_END_HOUR = 9
PM_RUSH_START_HOUR = 16
PM_RUSH_END_HOUR = 19
WEEKEND_START_DAY = 5
MEMORIAL_DAY_MONTH = 5
MEMORIAL_DAY_DAY = 30
HOURS_PER_DAY = 24.0
DAYS_PER_WEEK = 7.0

# Target Bounds & Defaults
DEFAULT_GLOBAL_FARE_MEAN = 15.15
DEFAULT_VENDOR_ID = 2.0
MIN_PREDICTED_FARE = 2.50
MIN_PREDICTED_DURATION = 0.5


def _haversine_np(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculates great-circle Haversine distance in miles between coordinate arrays."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _manhattan_np(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculates L1 Manhattan distance in miles between coordinate arrays."""
    lat_mid = np.radians((lat1 + lat2) / 2.0)
    dlat_miles = np.abs(lat2 - lat1) * MILES_PER_DEGREE_LATITUDE
    dlon_miles = np.abs(lon2 - lon1) * MILES_PER_DEGREE_LATITUDE * np.cos(lat_mid)
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
        global_fare_mean: float = DEFAULT_GLOBAL_FARE_MEAN,
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
            else np.full(len(df), int(DEFAULT_VENDOR_ID), dtype=int)
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

        df_out["sin_hour"] = np.sin(2.0 * np.pi * hour / HOURS_PER_DAY)
        df_out["cos_hour"] = np.cos(2.0 * np.pi * hour / HOURS_PER_DAY)
        df_out["sin_dayofweek"] = np.sin(2.0 * np.pi * dayofweek / DAYS_PER_WEEK)
        df_out["cos_dayofweek"] = np.cos(2.0 * np.pi * dayofweek / DAYS_PER_WEEK)

        df_out["is_weekend"] = (dayofweek >= WEEKEND_START_DAY).astype(int)

        is_weekday = dayofweek < WEEKEND_START_DAY
        is_am_rush = (hour >= AM_RUSH_START_HOUR) & (hour <= AM_RUSH_END_HOUR)
        is_pm_rush = (hour >= PM_RUSH_START_HOUR) & (hour <= PM_RUSH_END_HOUR)
        df_out["is_rush_hour"] = (is_weekday & (is_am_rush | is_pm_rush)).astype(int)
        df_out["is_holiday"] = (
            (dt.dt.month == MEMORIAL_DAY_MONTH) & (dt.dt.day == MEMORIAL_DAY_DAY)
        ).astype(int)

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
        df_out["haversine_ratio"] = np.clip(
            haversine / (trip_dist + EPSILON_DISTANCE), 0.0, MAX_HAVERSINE_RATIO
        )
        df_out["is_same_zone"] = (pu_id == do_id).astype(int)

        is_jfk_rate = ratecode == JFK_RATECODE_ID
        is_jfk_zone = (pu_id == JFK_AIRPORT_ZONE_ID) | (do_id == JFK_AIRPORT_ZONE_ID)
        df_out["is_jfk"] = (is_jfk_rate | is_jfk_zone).astype(int)

        is_newark_rate = ratecode == NEWARK_RATECODE_ID
        is_newark_zone = (pu_id == NEWARK_AIRPORT_ZONE_ID) | (
            do_id == NEWARK_AIRPORT_ZONE_ID
        )
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

    def predict_fast(self, payload: dict) -> Tuple[float, float]:
        """
        High-performance single-row prediction path (T-113 optimization).
        Bypasses pandas DataFrame allocation, using fast scalar math and direct booster inference.
        """
        dt_raw = payload["tpep_pickup_datetime"]
        if isinstance(dt_raw, str):
            dt = datetime.fromisoformat(dt_raw)
        else:
            dt = dt_raw

        hour = dt.hour
        dayofweek = dt.weekday()
        day = dt.day

        pu_id = int(payload["PULocationID"])
        do_id = int(payload["DOLocationID"])
        ratecode = int(payload["RatecodeID"])
        passengers = float(payload.get("passenger_count", 1))
        trip_dist = float(payload.get("trip_distance", 1.0))
        vendor_id = float(payload.get("VendorID", DEFAULT_VENDOR_ID))

        sin_hour = math.sin(2.0 * math.pi * hour / HOURS_PER_DAY)
        cos_hour = math.cos(2.0 * math.pi * hour / HOURS_PER_DAY)
        sin_dow = math.sin(2.0 * math.pi * dayofweek / DAYS_PER_WEEK)
        cos_dow = math.cos(2.0 * math.pi * dayofweek / DAYS_PER_WEEK)

        is_weekend = 1.0 if dayofweek >= WEEKEND_START_DAY else 0.0
        is_weekday = dayofweek < WEEKEND_START_DAY
        is_am_rush = 1.0 if (AM_RUSH_START_HOUR <= hour <= AM_RUSH_END_HOUR) else 0.0
        is_pm_rush = 1.0 if (PM_RUSH_START_HOUR <= hour <= PM_RUSH_END_HOUR) else 0.0
        is_rush_hour = 1.0 if (is_weekday and (is_am_rush or is_pm_rush)) else 0.0
        is_holiday = (
            1.0
            if (dt.month == MEMORIAL_DAY_MONTH and dt.day == MEMORIAL_DAY_DAY)
            else 0.0
        )

        pu_coords = self.centroid_lookup.get(pu_id, (0.0, 0.0))
        do_coords = self.centroid_lookup.get(do_id, (0.0, 0.0))
        pu_lat = pu_coords[0] if not math.isnan(pu_coords[0]) else 0.0
        pu_lon = pu_coords[1] if not math.isnan(pu_coords[1]) else 0.0
        do_lat = do_coords[0] if not math.isnan(do_coords[0]) else 0.0
        do_lon = do_coords[1] if not math.isnan(do_coords[1]) else 0.0

        # Fast Haversine scalar
        phi1 = math.radians(pu_lat)
        phi2 = math.radians(do_lat)
        dphi = math.radians(do_lat - pu_lat)
        dlambda = math.radians(do_lon - pu_lon)
        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        )
        haversine = (
            2.0 * EARTH_RADIUS_MILES * math.asin(min(1.0, math.sqrt(max(0.0, a))))
        )

        # Fast Manhattan scalar
        lat_mid = math.radians((pu_lat + do_lat) / 2.0)
        dlat_miles = abs(do_lat - pu_lat) * MILES_PER_DEGREE_LATITUDE
        dlon_miles = abs(do_lon - pu_lon) * MILES_PER_DEGREE_LATITUDE * math.cos(lat_mid)
        manhattan = dlat_miles + dlon_miles

        haversine_ratio = min(
            max(haversine / (trip_dist + EPSILON_DISTANCE), 0.0), MAX_HAVERSINE_RATIO
        )
        is_same_zone = 1.0 if pu_id == do_id else 0.0
        is_jfk = (
            1.0
            if (
                ratecode == JFK_RATECODE_ID
                or pu_id == JFK_AIRPORT_ZONE_ID
                or do_id == JFK_AIRPORT_ZONE_ID
            )
            else 0.0
        )
        is_newark = (
            1.0
            if (
                ratecode == NEWARK_RATECODE_ID
                or pu_id == NEWARK_AIRPORT_ZONE_ID
                or do_id == NEWARK_AIRPORT_ZONE_ID
            )
            else 0.0
        )

        pu_te = float(
            self.target_encodings.get("PULocationID", {}).get(
                pu_id, self.global_fare_mean
            )
        )
        do_te = float(
            self.target_encodings.get("DOLocationID", {}).get(
                do_id, self.global_fare_mean
            )
        )
        rate_te = float(
            self.target_encodings.get("RatecodeID", {}).get(
                ratecode, self.global_fare_mean
            )
        )

        feat_vec = [
            float(pu_id),
            float(do_id),
            passengers,
            float(ratecode),
            trip_dist,
            vendor_id,
            float(hour),
            float(dayofweek),
            float(day),
            sin_hour,
            cos_hour,
            sin_dow,
            cos_dow,
            is_weekend,
            is_rush_hour,
            is_holiday,
            pu_lat,
            pu_lon,
            do_lat,
            do_lon,
            haversine,
            manhattan,
            haversine_ratio,
            is_same_zone,
            is_jfk,
            is_newark,
            pu_te,
            do_te,
            rate_te,
        ]
        feat_arr = np.array([feat_vec], dtype=np.float64)

        fare_booster = getattr(self.fare_model, "booster_", self.fare_model)
        dur_booster = getattr(self.duration_model, "booster_", self.duration_model)

        pred_fare = float(fare_booster.predict(feat_arr)[0])
        pred_dur = float(dur_booster.predict(feat_arr)[0])

        return max(pred_fare, MIN_PREDICTED_FARE), max(pred_dur, MIN_PREDICTED_DURATION)

    def warm_up(self) -> None:
        """Warms up booster threads and inference buffers during application startup."""
        dummy_payload = {
            "PULocationID": JFK_AIRPORT_ZONE_ID,
            "DOLocationID": 236,
            "tpep_pickup_datetime": "2022-05-20T14:30:00",
            "passenger_count": 2,
            "RatecodeID": 1,
            "trip_distance": 6.2,
        }
        self.predict_fast(dummy_payload)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts fare_amount and duration_minutes for given raw input DataFrame."""
        X_feat = self.transform_features(X)
        pred_fare = self.fare_model.predict(X_feat)
        pred_duration = self.duration_model.predict(X_feat)

        # Ensure predictions are non-negative
        pred_fare = np.clip(pred_fare, a_min=MIN_PREDICTED_FARE, a_max=None)
        pred_duration = np.clip(pred_duration, a_min=MIN_PREDICTED_DURATION, a_max=None)

        return np.column_stack((pred_fare, pred_duration))

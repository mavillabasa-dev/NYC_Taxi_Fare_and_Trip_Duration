# tests/test_predictor_parity.py â€” The two inference paths must stay numerically identical.
#
# SelfContainedTaxiModel computes the same 29 features twice:
#
#   transform_features()  vectorised pandas path, used by predict() and by training
#   predict_fast()        scalar path (T-113), used by ModelService for every request
#
# predict_fast builds its feature vector as a hand-written positional list and never
# consults self.feature_names, so nothing but a test keeps the two in step. These
# tests need no dataset and no models/model.pkl: a stub estimator captures whatever
# array it is asked to predict on, which is the feature vector itself.

import math

import numpy as np
import pandas as pd
import pytest

from app.model.predictor import SelfContainedTaxiModel
from app.model.schema import MAX_LOCATION_ID, MIN_LOCATION_ID

# Zones with shapefile geometry. The Taxi Zone Shapefile carries 263 polygons, so
# LocationID 264 ("Unknown") and 265 ("N/A") have no centroid. derive_zone_centroids
# left-joins the lookup against those polygons, which leaves both rows with NaN
# coordinates, and export_production_model copies them into the bundle as (nan, nan) â€”
# so they are present in the lookup, not missing from it. Cover both shapes anyway.
LAST_ZONE_WITH_GEOMETRY = 263


class _SpyEstimator:
    """Stands in for a fitted LGBMRegressor and records its input.

    predict_fast reaches for `booster_` and falls back to the estimator itself, so an
    object with a bare `predict` serves both inference paths.
    """

    def __init__(self) -> None:
        self.last_input = None

    def predict(self, X):
        self.last_input = X
        return np.full(len(X), 10.0)


def _build_model() -> tuple[SelfContainedTaxiModel, _SpyEstimator]:
    centroids: dict[int, tuple[float, float]] = {
        zone: (40.70 + (zone % 20) * 0.02, -74.00 + (zone % 15) * 0.03)
        for zone in range(MIN_LOCATION_ID, LAST_ZONE_WITH_GEOMETRY + 1)
    }
    # Present in the lookup but without geometry, exactly as the real artifact has them.
    centroids[264] = (math.nan, math.nan)
    # 265 deliberately left out, to also cover a LocationID missing from the lookup.

    target_encodings = {
        "PULocationID": {zone: 14.0 + zone % 11 for zone in range(1, 264)},
        "DOLocationID": {zone: 15.0 + zone % 9 for zone in range(1, 264)},
        "RatecodeID": {code: 13.0 + code for code in range(1, 7)},
    }

    fare_spy = _SpyEstimator()
    model = SelfContainedTaxiModel(
        fare_model=fare_spy,
        duration_model=_SpyEstimator(),
        centroid_lookup=centroids,
        target_encodings=target_encodings,
        global_fare_mean=15.15,
        feature_names=None,
        version="parity-test",
    )
    return model, fare_spy


def _payload(**overrides) -> dict:
    base = {
        "PULocationID": 132,
        "DOLocationID": 236,
        "tpep_pickup_datetime": "2022-05-20T14:30:00",
        "passenger_count": 2,
        "RatecodeID": 1,
        "trip_distance": 6.2,
    }
    base.update(overrides)
    return base


def _both_paths(model: SelfContainedTaxiModel, spy: _SpyEstimator, payload: dict):
    """Returns (feature_names, dataframe_values, fast_values) for one payload."""
    slow_row = model.transform_features(pd.DataFrame([payload])).iloc[0]
    model.predict_fast(payload)
    fast_values = np.asarray(spy.last_input)[0]
    return list(slow_row.index), [float(v) for v in slow_row.tolist()], list(fast_values)


def _equal(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def test_fast_path_emits_one_value_per_dataframe_column():
    """Guards the positional list in predict_fast against a feature being added
    to transform_features without it."""
    model, spy = _build_model()
    names, slow, fast = _both_paths(model, spy, _payload())

    assert len(fast) == len(slow), (
        f"transform_features produces {len(slow)} features but predict_fast emits "
        f"{len(fast)}. The hand-written feat_vec in predict_fast is out of date."
    )
    assert len(names) == 29


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("airport to midtown", _payload()),
        ("same zone", _payload(PULocationID=236, DOLocationID=236)),
        ("jfk ratecode", _payload(RatecodeID=2, PULocationID=132)),
        ("newark ratecode", _payload(RatecodeID=3, DOLocationID=1)),
        ("memorial day", _payload(tpep_pickup_datetime="2022-05-30T12:00:00")),
        ("weekend night", _payload(tpep_pickup_datetime="2022-05-21T22:15:00")),
        ("am rush", _payload(tpep_pickup_datetime="2022-05-16T08:30:00")),
        ("minimum distance", _payload(trip_distance=0.01)),
        ("long trip", _payload(trip_distance=140.0)),
    ],
)
def test_paths_agree_on_representative_trips(label, payload):
    model, spy = _build_model()
    names, slow, fast = _both_paths(model, spy, payload)

    for name, slow_value, fast_value in zip(names, slow, fast, strict=True):
        assert _equal(slow_value, fast_value), (
            f"[{label}] feature '{name}' diverges: "
            f"transform_features={slow_value!r} predict_fast={fast_value!r}"
        )


def test_paths_agree_on_every_location_id_the_api_accepts():
    """The schema admits LocationID 1..265, so every one of them must round-trip
    through both paths identically â€” including the zones with no centroid."""
    model, spy = _build_model()
    mismatches = []

    for zone in range(MIN_LOCATION_ID, MAX_LOCATION_ID + 1):
        for role in ("PULocationID", "DOLocationID"):
            payload = _payload(**{role: zone})
            names, slow, fast = _both_paths(model, spy, payload)
            for name, slow_value, fast_value in zip(names, slow, fast, strict=True):
                if not _equal(slow_value, fast_value):
                    mismatches.append(
                        f"{role}={zone} '{name}': " f"dataframe={slow_value!r} fast={fast_value!r}"
                    )

    assert not mismatches, "Inference paths diverge:\n  " + "\n  ".join(mismatches[:20])


@pytest.mark.parametrize("zone", [264, 265])
def test_zone_without_centroid_does_not_fabricate_a_distance(zone):
    """Regression: predict_fast defaulted a missing centroid to (0.0, 0.0) and then
    measured the real distance from the Gulf of Guinea to New York, feeding the model
    a ~5,400-mile trip while the training path had produced 0.0."""
    model, spy = _build_model()
    names, _, fast = _both_paths(model, spy, _payload(PULocationID=zone))
    values = dict(zip(names, fast, strict=True))

    assert math.isnan(values["pu_lat"])
    assert math.isnan(values["pu_lon"])
    assert values["haversine_distance"] == 0.0
    assert values["manhattan_distance"] == 0.0
    assert values["haversine_ratio"] == 0.0


def test_predictions_stay_finite_for_zones_without_centroid():
    """End of the same path: a NaN coordinate must not leak into the response."""
    model, _ = _build_model()
    for zone in (264, 265):
        fare, duration = model.predict_fast(_payload(PULocationID=zone))
        assert math.isfinite(fare) and math.isfinite(duration)

        preds = model.predict(pd.DataFrame([_payload(DOLocationID=zone)]))
        assert np.isfinite(preds).all()

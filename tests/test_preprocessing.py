from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor

from composites.preprocessing import (
    OutlierFilteredRegressor,
    PatchAngleBinaryEncoder,
    build_regression_pipeline,
)
from composites.schema import PATCH_ANGLE


class RecordingRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RecordingRegressor":
        self.X_seen_ = X.copy()
        self.y_seen_ = np.asarray(y).copy()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"feature": [0.0, 1.0, 2.0, 3.0, 100.0], PATCH_ANGLE: [0, 90, 0, 90, 0]},
        index=[10, 11, 12, 13, 14],
    )


def test_iqr_filter_keeps_x_and_y_synchronized() -> None:
    X = _training_frame()
    y = np.array([0.0, 10.0, 20.0, 30.0, 1000.0])
    wrapper = OutlierFilteredRegressor(RecordingRegressor()).fit(X, y)

    assert wrapper.removed_indices_ == [14]
    assert wrapper.n_fit_samples_ == 4
    assert wrapper.estimator_.X_seen_["feature"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert wrapper.estimator_.y_seen_.tolist() == [0.0, 10.0, 20.0, 30.0]


def test_predict_never_filters_rows_or_refits_thresholds() -> None:
    wrapper = OutlierFilteredRegressor(DummyRegressor()).fit(
        _training_frame(), np.arange(5, dtype=float)
    )
    thresholds_before = {name: values.copy() for name, values in wrapper.thresholds_.items()}
    held_out = pd.DataFrame(
        {"feature": [-1_000_000.0, 1_000_000.0], PATCH_ANGLE: [0, 90]},
        index=[100, 101],
    )

    predictions = wrapper.predict(held_out)

    assert len(predictions) == len(held_out)
    assert wrapper.thresholds_ == thresholds_before


def test_scaler_is_fitted_only_on_local_training_rows() -> None:
    X = pd.DataFrame(
        {"feature": [1.0, 2.0, 3.0], PATCH_ANGLE: [0, 90, 0]}, index=[1, 2, 3]
    )
    pipeline = build_regression_pipeline(
        DummyRegressor(), model_family="ridge", filter_outliers=False
    ).fit(X, [1.0, 2.0, 3.0])
    mean_before = pipeline.named_steps["scaler"].mean_.copy()

    pipeline.predict(
        pd.DataFrame({"feature": [10_000.0], PATCH_ANGLE: [90]}, index=[999])
    )

    assert mean_before.tolist() == pytest.approx([2.0, 1.0 / 3.0])
    assert pipeline.named_steps["scaler"].mean_.tolist() == pytest.approx(mean_before)


def test_angle_encoder_maps_only_confirmed_values() -> None:
    X = pd.DataFrame({"feature": [1.0, 2.0], PATCH_ANGLE: [0, 90]})
    encoder = PatchAngleBinaryEncoder().fit(X)

    assert encoder.transform(X)[PATCH_ANGLE].tolist() == [0.0, 1.0]
    with pytest.raises(ValueError, match="Unexpected patch angle"):
        encoder.transform(pd.DataFrame({"feature": [3.0], PATCH_ANGLE: [45]}))


def test_inference_rejects_missing_values() -> None:
    pipeline = build_regression_pipeline(
        DummyRegressor(), model_family="random_forest"
    ).fit(_training_frame(), np.arange(5, dtype=float))

    invalid = pd.DataFrame({"feature": [np.nan], PATCH_ANGLE: [0]})
    with pytest.raises(ValueError, match="Missing values"):
        pipeline.predict(invalid)

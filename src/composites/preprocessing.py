"""Leakage-safe preprocessing components for later cross-validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from composites.schema import PATCH_ANGLE, SOURCE_INDEX

SCALED_MODEL_FAMILIES = frozenset({"ridge", "svr", "mlp"})
UNSCALED_MODEL_FAMILIES = frozenset(
    {"baseline", "random_forest", "gradient_boosting"}
)


def _validated_frame(X: Any, expected_columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Preprocessing requires a pandas DataFrame with named columns")
    if expected_columns is not None and tuple(X.columns) != expected_columns:
        raise ValueError("Input columns or their order differ from fitted data")
    if SOURCE_INDEX in X.columns:
        raise ValueError("source_index must not be used as a model feature")
    if X.isna().any().any():
        raise ValueError("Missing values are not allowed by the strict data contract")
    try:
        values = X.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("All model features must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError("Non-finite model features are not allowed")
    return X


class FiniteNumericValidator(TransformerMixin, BaseEstimator):
    """Reject schema drift, missing values and non-finite inference inputs."""

    def fit(self, X: Any, y: Any = None) -> "FiniteNumericValidator":
        frame = _validated_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        expected = tuple(self.feature_names_in_.tolist())
        return _validated_frame(X, expected).copy()


class PatchAngleBinaryEncoder(TransformerMixin, BaseEstimator):
    """Map the confirmed 0/90 degree angle values to the equivalent 0/1 scale."""

    def fit(self, X: Any, y: Any = None) -> "PatchAngleBinaryEncoder":
        frame = _validated_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        if PATCH_ANGLE in frame.columns:
            observed = sorted(float(value) for value in frame[PATCH_ANGLE].unique())
            if not set(observed).issubset({0.0, 90.0}):
                raise ValueError(f"Unexpected patch angle values: {observed}")
            self.observed_angles_ = observed
        else:
            self.observed_angles_ = []
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        expected = tuple(self.feature_names_in_.tolist())
        transformed = _validated_frame(X, expected).copy()
        if PATCH_ANGLE in transformed.columns:
            observed = set(transformed[PATCH_ANGLE].astype(float).unique())
            if not observed.issubset({0.0, 90.0}):
                raise ValueError(f"Unexpected patch angle values: {sorted(observed)}")
            transformed[PATCH_ANGLE] = transformed[PATCH_ANGLE].astype(float) / 90.0
        return transformed


class OutlierFilteredRegressor(RegressorMixin, BaseEstimator):
    """Optional IQR ablation that filters X rows during fit and never during predict."""

    def __init__(
        self,
        estimator: Any,
        *,
        iqr_multiplier: float = 1.5,
        excluded_features: tuple[str, ...] = (PATCH_ANGLE,),
    ) -> None:
        self.estimator = estimator
        self.iqr_multiplier = iqr_multiplier
        self.excluded_features = excluded_features

    def fit(self, X: Any, y: Any) -> "OutlierFilteredRegressor":
        frame = _validated_frame(X)
        y_values = np.asarray(y)
        if y_values.ndim != 1 or y_values.shape[0] != frame.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        if not np.isfinite(y_values.astype(float)).all():
            raise ValueError("Target must contain finite numeric values")
        if self.iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier must be positive")

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        filter_features = [
            column
            for column in frame.columns
            if column not in self.excluded_features and column != SOURCE_INDEX
        ]
        self.filter_features_ = filter_features
        self.thresholds_: dict[str, dict[str, float]] = {}
        keep_mask = np.ones(len(frame), dtype=bool)
        for column in filter_features:
            q1 = float(frame[column].quantile(0.25))
            q3 = float(frame[column].quantile(0.75))
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            self.thresholds_[column] = {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower": lower,
                "upper": upper,
            }
            keep_mask &= frame[column].between(lower, upper, inclusive="both").to_numpy()

        self.n_input_samples_ = int(len(frame))
        self.n_removed_ = int((~keep_mask).sum())
        self.n_fit_samples_ = int(keep_mask.sum())
        self.removed_indices_ = frame.index[~keep_mask].tolist()
        if self.n_fit_samples_ == 0:
            raise ValueError("IQR filtering removed every training row")

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(frame.loc[keep_mask], y_values[keep_mask])
        return self

    def predict(self, X: Any) -> np.ndarray:
        expected = tuple(self.feature_names_in_.tolist())
        frame = _validated_frame(X, expected)
        return np.asarray(self.estimator_.predict(frame))


def build_regression_pipeline(
    estimator: Any,
    *,
    model_family: str,
    filter_outliers: bool = False,
    iqr_multiplier: float = 1.5,
) -> Any:
    """Build a cloneable pipeline; all learned state is fitted inside each CV fold."""
    if model_family in SCALED_MODEL_FAMILIES:
        scaler: Any = StandardScaler()
    elif model_family in UNSCALED_MODEL_FAMILIES:
        scaler = "passthrough"
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    pipeline = Pipeline(
        steps=[
            ("validate", FiniteNumericValidator()),
            ("angle", PatchAngleBinaryEncoder()),
            ("scaler", scaler),
            ("model", estimator),
        ]
    )
    if not filter_outliers:
        return pipeline
    return OutlierFilteredRegressor(
        pipeline,
        iqr_multiplier=iqr_multiplier,
        excluded_features=(PATCH_ANGLE,),
    )

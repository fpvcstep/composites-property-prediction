from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR

from composites.preprocessing import build_regression_pipeline
from composites.schema import DIRECT_FEATURES, RATIO_FEATURES, SOURCE_INDEX
from composites.training import (
    TASKS,
    _target_scaled,
    native_unit_rmse,
    persisted_cv_positions,
    task_xy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_persisted_cv_maps_source_ids_and_excludes_test() -> None:
    manifest = json.loads((PROJECT_ROOT / "data" / "splits.json").read_text("utf-8"))
    train_index = pd.Index(manifest["train_indices"], name=SOURCE_INDEX)
    splits = persisted_cv_positions(manifest, train_index)

    assert len(splits) == 10
    validation_positions = np.concatenate([validation for _, validation in splits])
    assert sorted(validation_positions.tolist()) == list(range(716))
    assert set(train_index).isdisjoint(manifest["test_indices"])


def test_task_feature_contracts_use_exact_columns_and_original_index() -> None:
    columns = set(DIRECT_FEATURES) | set(RATIO_FEATURES) | {
        TASKS["direct_tensile_modulus"].target,
        TASKS["direct_tensile_strength"].target,
        TASKS["matrix_filler_ratio"].target,
    }
    frame = pd.DataFrame(
        {column: [1.0, 2.0] for column in columns}, index=pd.Index([101, 202])
    )
    for task in TASKS.values():
        X, y = task_xy(frame, task)
        assert tuple(X.columns) == task.features
        assert task.target not in X.columns
        assert X.index.tolist() == [101, 202]
        assert y.index.tolist() == [101, 202]


def test_transformed_target_predictions_and_rmse_are_in_native_units() -> None:
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 20)})
    y = pd.Series(1_000.0 + 500.0 * X["x"])
    estimator = build_regression_pipeline(
        _target_scaled(SVR(C=10.0, epsilon=0.01)), model_family="svr"
    ).fit(X, y)

    predictions = estimator.predict(X)
    assert predictions.mean() > 1_000.0
    assert native_unit_rmse(estimator, X, y) == pytest.approx(
        np.sqrt(np.mean((y.to_numpy() - predictions) ** 2))
    )


def test_joblib_reload_preserves_predictions(tmp_path: Path) -> None:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    model = build_regression_pipeline(
        DummyRegressor(strategy="mean"), model_family="baseline"
    ).fit(X, [10.0, 20.0, 30.0])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path, compress=3)

    reloaded = joblib.load(path)
    assert reloaded.predict(X).tolist() == model.predict(X).tolist()

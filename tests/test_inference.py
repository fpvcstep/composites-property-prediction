from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from composites.inference import (
    DIRECT_TASKS,
    InferenceError,
    ModelUnavailableError,
    load_direct_predictor,
)
from composites.schema import DIRECT_FEATURES, PATCH_ANGLE


def _valid_frame(rows: int = 3) -> pd.DataFrame:
    data = {feature: np.arange(1, rows + 1, dtype=float) for feature in DIRECT_FEATURES}
    data[PATCH_ANGLE] = np.resize([0.0, 90.0], rows)
    return pd.DataFrame(data)


def _write_models(models_dir: Path) -> dict[str, object]:
    models_dir.mkdir()
    frame = _valid_frame(4)
    ranges = {
        feature: {
            "min": float(frame[feature].min()),
            "median": float(frame[feature].median()),
            "max": float(frame[feature].max()),
        }
        for feature in DIRECT_FEATURES
    }
    manifest: dict[str, object] = {"schema_version": 1, "tasks": {}}
    for index, (task_name, (target, unit)) in enumerate(DIRECT_TASKS.items(), start=1):
        artifact = f"{task_name}.joblib"
        model = Pipeline(
            [("regressor", DummyRegressor(strategy="constant", constant=10.0 * index))]
        ).fit(frame, np.arange(len(frame), dtype=float))
        joblib.dump(model, models_dir / artifact)
        manifest["tasks"][task_name] = {
            "target": target,
            "unit": unit,
            "features": list(DIRECT_FEATURES),
            "model_family": "DummyRegressor baseline",
            "artifact": artifact,
            "cv_rmse": 2.0,
            "baseline_cv_rmse": 2.0,
            "test_metrics": {"rmse": 2.1, "mae": 1.5, "r2": 0.0},
            "train_feature_ranges": ranges,
        }
    (models_dir / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def test_batch_predictions_equal_loaded_pipeline_predictions(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    manifest = _write_models(models_dir)
    predictor = load_direct_predictor(models_dir)
    frame = _valid_frame(3)

    actual = predictor.predict(frame)

    for task_name, (target, _) in DIRECT_TASKS.items():
        artifact = manifest["tasks"][task_name]["artifact"]
        model = joblib.load(models_dir / artifact)
        expected = model.predict(frame.loc[:, list(DIRECT_FEATURES)])
        assert actual[target].to_numpy() == pytest.approx(expected)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda frame: frame.drop(columns=[DIRECT_FEATURES[0]]), "Неверные колонки"),
        (lambda frame: frame.assign(extra=1.0), "Неверные колонки"),
        (lambda frame: frame.assign(**{DIRECT_FEATURES[0]: "не число"}), "числовым"),
        (lambda frame: frame.assign(**{DIRECT_FEATURES[0]: np.nan}), "NaN"),
        (lambda frame: frame.assign(**{DIRECT_FEATURES[0]: np.inf}), "NaN"),
        (lambda frame: frame.assign(**{PATCH_ANGLE: 45.0}), "0 или 90"),
    ],
)
def test_invalid_inference_inputs_are_rejected(
    tmp_path: Path, mutate, match: str
) -> None:
    models_dir = tmp_path / "models"
    _write_models(models_dir)
    predictor = load_direct_predictor(models_dir)

    with pytest.raises(InferenceError, match=match):
        predictor.predict(mutate(_valid_frame(1)))


def test_missing_manifest_and_artifact_are_reported(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with pytest.raises(ModelUnavailableError, match="model_manifest.json"):
        load_direct_predictor(models_dir)

    manifest = _write_models(tmp_path / "complete_models")
    broken_dir = tmp_path / "broken_models"
    broken_dir.mkdir()
    (broken_dir / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ModelUnavailableError, match="Не найден артефакт"):
        load_direct_predictor(broken_dir)

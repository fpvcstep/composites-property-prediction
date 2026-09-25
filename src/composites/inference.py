"""Validated loading and inference for the two direct-property models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from composites.data import sha256_file
from composites.schema import (
    DIRECT_FEATURES,
    PATCH_ANGLE,
    TENSILE_MODULUS,
    TENSILE_STRENGTH,
)


DIRECT_TASKS = {
    "direct_tensile_modulus": (TENSILE_MODULUS, "ГПа"),
    "direct_tensile_strength": (TENSILE_STRENGTH, "МПа"),
}


class InferenceError(ValueError):
    """Base error for an invalid inference contract or input."""


class ModelUnavailableError(InferenceError):
    """Raised when trained model files are not available."""


@dataclass(frozen=True)
class FeatureRange:
    minimum: float
    median: float
    maximum: float


@dataclass(frozen=True)
class LoadedTask:
    name: str
    target: str
    unit: str
    features: tuple[str, ...]
    model_family: str
    artifact_path: Path
    cv_rmse: float
    baseline_cv_rmse: float
    test_metrics: dict[str, float | None]
    train_feature_ranges: dict[str, FeatureRange]
    model: Any

    @property
    def is_baseline(self) -> bool:
        normalized = self.model_family.casefold()
        return "baseline" in normalized or "dummy" in normalized


@dataclass(frozen=True)
class DirectPredictor:
    tasks: dict[str, LoadedTask]

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(DIRECT_FEATURES)

    def defaults(self) -> dict[str, float]:
        first_task = self.tasks[next(iter(DIRECT_TASKS))]
        return {
            feature: feature_range.median
            for feature, feature_range in first_task.train_feature_ranges.items()
        }

    def range_warnings(self, data: Mapping[str, object] | pd.DataFrame) -> list[str]:
        frame = _validated_frame(data)
        messages: list[str] = []
        seen: set[tuple[str, float, float, float]] = set()
        for task in self.tasks.values():
            for feature, feature_range in task.train_feature_ranges.items():
                for value in frame[feature].tolist():
                    key = (feature, value, feature_range.minimum, feature_range.maximum)
                    if (
                        value < feature_range.minimum or value > feature_range.maximum
                    ) and key not in seen:
                        messages.append(
                            f"{feature}: {value:g} вне обучающего диапазона "
                            f"[{feature_range.minimum:g}; {feature_range.maximum:g}]."
                        )
                        seen.add(key)
        return messages

    def predict(self, data: Mapping[str, object] | pd.DataFrame) -> pd.DataFrame:
        frame = _validated_frame(data)
        predictions: dict[str, np.ndarray] = {}
        for task in self.tasks.values():
            try:
                values = np.asarray(
                    task.model.predict(frame.loc[:, list(task.features)]), dtype=float
                )
            except Exception as exc:
                raise InferenceError(
                    f"Модель задачи {task.name!r} не смогла выполнить predict."
                ) from exc
            if values.ndim != 1 or len(values) != len(frame):
                raise InferenceError(
                    f"Модель задачи {task.name!r} вернула результат неверной формы."
                )
            if not np.isfinite(values).all():
                raise InferenceError(
                    f"Модель задачи {task.name!r} вернула NaN или бесконечность."
                )
            predictions[task.target] = values
        return pd.DataFrame(predictions, index=frame.index)


def load_direct_predictor(models_dir: str | Path) -> DirectPredictor:
    """Load and validate the manifest and both direct-property artifacts."""
    directory = Path(models_dir)
    manifest_path = directory / "model_manifest.json"
    if not manifest_path.is_file():
        raise ModelUnavailableError(
            "Не найден models/model_manifest.json. Сначала выполните этап обучения "
            "и сохраните manifest вместе с joblib-артефактами."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelUnavailableError(f"Не удалось прочитать {manifest_path}.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise InferenceError("Manifest должен иметь schema_version=1.")
    task_items = manifest.get("tasks")
    if not isinstance(task_items, dict):
        raise InferenceError("Manifest должен содержать словарь tasks.")

    loaded: dict[str, LoadedTask] = {}
    for task_name, (expected_target, expected_unit) in DIRECT_TASKS.items():
        raw = task_items.get(task_name)
        if not isinstance(raw, dict):
            raise InferenceError(f"В manifest отсутствует задача {task_name!r}.")
        target = _required_text(raw, "target", task_name)
        unit = _required_text(raw, "unit", task_name)
        if target != expected_target or unit != expected_unit:
            raise InferenceError(
                f"Задача {task_name!r} имеет неверные target или unit."
            )
        features = raw.get("features")
        if features != list(DIRECT_FEATURES):
            raise InferenceError(
                f"Задача {task_name!r} должна содержать DIRECT_FEATURES "
                "в каноническом порядке."
            )
        model_family = _required_text(raw, "model_family", task_name)
        artifact_text = _required_text(raw, "artifact", task_name)
        artifact_sha256 = raw.get("artifact_sha256")
        if not isinstance(artifact_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", artifact_sha256
        ) is None:
            raise InferenceError(
                f"Задача {task_name!r}: artifact_sha256 должен быть "
                "SHA-256 из 64 строчных hex-символов."
            )
        artifact_relative = Path(artifact_text)
        if artifact_relative.is_absolute():
            raise InferenceError(f"Artifact задачи {task_name!r} должен быть относительным.")
        directory_resolved = directory.resolve()
        artifact_path = (directory / artifact_relative).resolve()
        if not artifact_path.is_relative_to(directory_resolved):
            raise InferenceError(f"Artifact задачи {task_name!r} выходит за каталог models.")
        if not artifact_path.is_file():
            raise ModelUnavailableError(
                f"Не найден артефакт обученной модели: {artifact_relative.as_posix()}."
            )
        try:
            actual_sha256 = sha256_file(artifact_path)
        except OSError as exc:
            raise ModelUnavailableError(
                f"Не удалось прочитать артефакт {artifact_relative.as_posix()} "
                "для проверки SHA-256."
            ) from exc
        if actual_sha256 != artifact_sha256:
            raise InferenceError(
                f"Задача {task_name!r}: SHA-256 артефакта "
                f"{artifact_relative.as_posix()} не совпадает с manifest."
            )

        ranges = _feature_ranges(raw.get("train_feature_ranges"), task_name)
        cv_rmse = _metric(raw.get("cv_rmse"), "cv_rmse", task_name, nonnegative=True)
        baseline_cv_rmse = _metric(
            raw.get("baseline_cv_rmse"),
            "baseline_cv_rmse",
            task_name,
            nonnegative=True,
        )
        test_metrics_raw = raw.get("test_metrics")
        if not isinstance(test_metrics_raw, dict):
            raise InferenceError(f"Задача {task_name!r}: test_metrics должен быть словарём.")
        required_test_metrics = {"rmse", "mae", "r2"}
        if not required_test_metrics <= set(test_metrics_raw):
            raise InferenceError(
                f"Задача {task_name!r}: test_metrics должен содержать rmse, mae и r2."
            )
        test_metrics = {
            key: _optional_metric(
                test_metrics_raw.get(key),
                f"test_metrics.{key}",
                task_name,
                nonnegative=key in {"rmse", "mae"},
            )
            for key in ("rmse", "mae", "r2")
        }
        try:
            model = joblib.load(artifact_path)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Не удалось загрузить артефакт {artifact_relative.as_posix()}."
            ) from exc
        if not callable(getattr(model, "predict", None)):
            raise InferenceError(f"Артефакт задачи {task_name!r} не имеет метода predict.")
        model_feature_names = getattr(model, "feature_names_in_", None)
        if model_feature_names is not None and tuple(model_feature_names) != tuple(features):
            raise InferenceError(
                f"Артефакт задачи {task_name!r} обучен на признаках, которые "
                "не совпадают с manifest по составу или порядку."
            )

        loaded[task_name] = LoadedTask(
            name=task_name,
            target=target,
            unit=unit,
            features=tuple(features),
            model_family=model_family,
            artifact_path=artifact_path,
            cv_rmse=cv_rmse,
            baseline_cv_rmse=baseline_cv_rmse,
            test_metrics=test_metrics,
            train_feature_ranges=ranges,
            model=model,
        )
    return DirectPredictor(loaded)


def _validated_frame(data: Mapping[str, object] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, Mapping):
        frame = pd.DataFrame([dict(data)])
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        raise InferenceError("Вход должен быть словарём признаков или pandas.DataFrame.")
    if frame.columns.has_duplicates:
        raise InferenceError("Вход содержит повторяющиеся колонки.")
    expected = set(DIRECT_FEATURES)
    actual = set(frame.columns)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise InferenceError(f"Неверные колонки: missing={missing}, extra={extra}.")
    if frame.empty:
        raise InferenceError("Вход не должен быть пустым.")
    frame = frame.loc[:, list(DIRECT_FEATURES)]
    for column in frame.columns:
        if not is_numeric_dtype(frame[column].dtype):
            raise InferenceError(f"Признак {column!r} должен быть числовым.")
        if frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise InferenceError(f"Признак {column!r} не принимает логические значения.")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise InferenceError("Вход содержит пропуски, NaN или бесконечность.")
    angles = set(frame[PATCH_ANGLE].astype(float).unique())
    if not angles <= {0.0, 90.0}:
        raise InferenceError("Угол нашивки должен быть равен 0 или 90 градусам.")
    return frame.astype(float)


def _required_text(raw: dict[str, object], key: str, task_name: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InferenceError(f"Задача {task_name!r}: поле {key!r} должно быть строкой.")
    return value


def _feature_ranges(raw: object, task_name: str) -> dict[str, FeatureRange]:
    if not isinstance(raw, dict) or set(raw) != set(DIRECT_FEATURES):
        raise InferenceError(
            f"Задача {task_name!r}: train_feature_ranges должен описывать все признаки."
        )
    parsed: dict[str, FeatureRange] = {}
    for feature in DIRECT_FEATURES:
        item = raw[feature]
        if not isinstance(item, dict) or not {"min", "median", "max"} <= set(item):
            raise InferenceError(
                f"Задача {task_name!r}: диапазон {feature!r} неполон."
            )
        minimum = _finite_number(item["min"], f"{feature}.min", task_name)
        median = _finite_number(item["median"], f"{feature}.median", task_name)
        maximum = _finite_number(item["max"], f"{feature}.max", task_name)
        if not minimum <= median <= maximum:
            raise InferenceError(
                f"Задача {task_name!r}: для {feature!r} требуется min <= median <= max."
            )
        parsed[feature] = FeatureRange(minimum, median, maximum)
    return parsed


def _finite_number(value: object, field: str, task_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceError(f"Задача {task_name!r}: {field} должен быть числом.")
    number = float(value)
    if not math.isfinite(number):
        raise InferenceError(f"Задача {task_name!r}: {field} должен быть конечным.")
    return number


def _metric(
    value: object, field: str, task_name: str, *, nonnegative: bool
) -> float:
    number = _finite_number(value, field, task_name)
    if nonnegative and number < 0:
        raise InferenceError(f"Задача {task_name!r}: {field} не может быть отрицательным.")
    return number


def _optional_metric(
    value: object, field: str, task_name: str, *, nonnegative: bool
) -> float | None:
    if value is None:
        return None
    return _metric(value, field, task_name, nonnegative=nonnegative)

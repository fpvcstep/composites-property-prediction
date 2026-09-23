"""Train-only model selection using the persisted cross-validation protocol."""

from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from composites.data import load_and_merge, sha256_file
from composites.preprocessing import OutlierFilteredRegressor, build_regression_pipeline
from composites.schema import (
    DIRECT_FEATURES,
    HARDENER_AMOUNT,
    MATRIX_FILLER_RATIO,
    RATIO_FEATURES,
    SOURCE_INDEX,
    TENSILE_MODULUS,
    TENSILE_STRENGTH,
)
from composites.split import select_partition, validate_split_manifest

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}
FAMILY_ORDER = {
    "baseline": 0,
    "ridge": 1,
    "random_forest": 2,
    "gradient_boosting": 3,
    "svr": 4,
    "mlp": 5,
}
VARIANT_ORDER = {"raw": 0, "iqr_filtered": 1}


@dataclass(frozen=True)
class TaskSpec:
    key: str
    target: str
    features: tuple[str, ...]
    unit: str


TASKS = {
    "direct_tensile_modulus": TaskSpec(
        "direct_tensile_modulus", TENSILE_MODULUS, DIRECT_FEATURES, "ГПа"
    ),
    "direct_tensile_strength": TaskSpec(
        "direct_tensile_strength", TENSILE_STRENGTH, DIRECT_FEATURES, "МПа"
    ),
    "matrix_filler_ratio": TaskSpec(
        "matrix_filler_ratio", MATRIX_FILLER_RATIO, RATIO_FEATURES, "не указана"
    ),
}


def read_split_manifest(path: Path, data_dir: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    actual = {
        name: sha256_file(data_dir / name)
        for name in manifest["raw_sources_sha256"]
    }
    if actual != manifest["raw_sources_sha256"]:
        raise ValueError("Raw source hashes differ from the split manifest")
    return manifest


def prepare_training_frame(data_dir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    merged, _ = load_and_merge(data_dir)
    train = select_partition(merged, manifest["train_indices"])
    train = train.set_index(SOURCE_INDEX, drop=True)
    if train.index.tolist() != manifest["train_indices"]:
        raise ValueError("Training frame order differs from the persisted source indices")
    return train


def persisted_cv_positions(
    manifest: dict[str, Any], train_index: pd.Index
) -> list[tuple[np.ndarray, np.ndarray]]:
    position = {int(source_id): offset for offset, source_id in enumerate(train_index)}
    test_ids = set(manifest["test_indices"])
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in manifest["cv_folds"]:
        fold_train_ids = fold["train_indices"]
        fold_validation_ids = fold["validation_indices"]
        if set(fold_train_ids) & test_ids or set(fold_validation_ids) & test_ids:
            raise ValueError("Persisted CV contains a held-out test source index")
        try:
            train_positions = np.asarray([position[value] for value in fold_train_ids])
            validation_positions = np.asarray(
                [position[value] for value in fold_validation_ids]
            )
        except KeyError as error:
            raise ValueError("Persisted CV references a row outside train") from error
        if np.intersect1d(train_positions, validation_positions).size:
            raise ValueError("CV train and validation positions overlap")
        splits.append((train_positions, validation_positions))
    return splits


def task_xy(train: pd.DataFrame, task: TaskSpec) -> tuple[pd.DataFrame, pd.Series]:
    X = train.loc[:, task.features].copy()
    y = train.loc[:, task.target].copy()
    if SOURCE_INDEX in X.columns or task.target in X.columns:
        raise ValueError(f"Invalid feature schema for {task.key}")
    if tuple(X.columns) != task.features:
        raise ValueError(f"Feature order differs for {task.key}")
    return X, y


def _target_scaled(regressor: Any) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
        check_inverse=True,
    )


def build_candidate(
    family: str, variant: str, *, random_state: int = 42
) -> Any:
    if family == "baseline":
        estimator = DummyRegressor(strategy="mean")
        model_family = "baseline"
    elif family == "ridge":
        estimator = Ridge()
        model_family = "ridge"
    elif family == "random_forest":
        estimator = RandomForestRegressor(random_state=random_state, n_jobs=1)
        model_family = "random_forest"
    elif family == "gradient_boosting":
        estimator = GradientBoostingRegressor(random_state=random_state)
        model_family = "gradient_boosting"
    elif family == "svr":
        estimator = _target_scaled(SVR(kernel="rbf"))
        model_family = "svr"
    elif family == "mlp":
        estimator = _target_scaled(
            MLPRegressor(
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=40,
                max_iter=1200,
                random_state=random_state,
            )
        )
        model_family = "mlp"
    else:
        raise ValueError(f"Unknown family: {family}")
    return build_regression_pipeline(
        estimator,
        model_family=model_family,
        filter_outliers=variant == "iqr_filtered",
    )


def _parameter_prefix(variant: str) -> str:
    return "estimator__model__" if variant == "iqr_filtered" else "model__"


def parameter_grid(family: str, variant: str) -> list[dict[str, list[Any]]]:
    prefix = _parameter_prefix(variant)
    if family == "ridge":
        return [{f"{prefix}alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}]
    if family == "random_forest":
        values = [
            (200, None, 1),
            (150, None, 3),
            (200, 10, 1),
            (150, 10, 3),
        ]
        return [
            {
                f"{prefix}n_estimators": [trees],
                f"{prefix}max_depth": [depth],
                f"{prefix}min_samples_leaf": [leaf],
            }
            for trees, depth, leaf in values
        ]
    if family == "gradient_boosting":
        values = [(0.03, 2), (0.03, 3), (0.1, 2), (0.1, 3)]
        return [
            {
                f"{prefix}n_estimators": [150],
                f"{prefix}learning_rate": [rate],
                f"{prefix}max_depth": [depth],
            }
            for rate, depth in values
        ]
    if family == "svr":
        return [
            {
                f"{prefix}regressor__C": [0.3, 1.0, 3.0],
                f"{prefix}regressor__epsilon": [0.05, 0.1],
                f"{prefix}regressor__gamma": ["scale"],
            }
        ]
    if family == "mlp":
        return [
            {
                f"{prefix}regressor__hidden_layer_sizes": [(16,), (32, 16)],
                f"{prefix}regressor__alpha": [0.0001, 0.01],
            }
        ]
    raise ValueError(f"No grid for family: {family}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _candidate_id(task: str, family: str, variant: str, params: dict[str, Any]) -> str:
    payload = json.dumps(_jsonable(params), ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return f"{task}:{family}:{variant}:{digest}"


def _baseline_row(
    task: str,
    scores: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task": task,
        "family": "baseline",
        "variant": "raw",
        "params_json": "{}",
        "mean_fit_time": float(np.mean(scores["fit_time"])),
        "std_fit_time": float(np.std(scores["fit_time"])),
        "mean_test_rmse": float(-np.mean(scores["test_rmse"])),
        "std_test_rmse": float(np.std(-scores["test_rmse"])),
        "mean_train_rmse": float(-np.mean(scores["train_rmse"])),
        "mean_test_mae": float(-np.mean(scores["test_mae"])),
        "std_test_mae": float(np.std(-scores["test_mae"])),
        "mean_train_mae": float(-np.mean(scores["train_mae"])),
        "mean_test_r2": float(np.mean(scores["test_r2"])),
        "std_test_r2": float(np.std(scores["test_r2"])),
        "mean_train_r2": float(np.mean(scores["train_r2"])),
    }
    row["candidate_id"] = _candidate_id(task, "baseline", "raw", {})
    for fold in range(len(scores["test_rmse"])):
        row[f"split{fold}_test_rmse"] = float(-scores["test_rmse"][fold])
        row[f"split{fold}_train_rmse"] = float(-scores["train_rmse"][fold])
    return row


def _grid_rows(
    task: str,
    family: str,
    variant: str,
    search: GridSearchCV,
) -> list[dict[str, Any]]:
    results = search.cv_results_
    rows: list[dict[str, Any]] = []
    for index, params in enumerate(results["params"]):
        row: dict[str, Any] = {
            "task": task,
            "family": family,
            "variant": variant,
            "params_json": json.dumps(
                _jsonable(params), ensure_ascii=False, sort_keys=True
            ),
            "mean_fit_time": float(results["mean_fit_time"][index]),
            "std_fit_time": float(results["std_fit_time"][index]),
            "mean_test_rmse": float(-results["mean_test_rmse"][index]),
            "std_test_rmse": float(results["std_test_rmse"][index]),
            "mean_train_rmse": float(-results["mean_train_rmse"][index]),
            "mean_test_mae": float(-results["mean_test_mae"][index]),
            "std_test_mae": float(results["std_test_mae"][index]),
            "mean_train_mae": float(-results["mean_train_mae"][index]),
            "mean_test_r2": float(results["mean_test_r2"][index]),
            "std_test_r2": float(results["std_test_r2"][index]),
            "mean_train_r2": float(results["mean_train_r2"][index]),
            "rank_test_rmse": int(results["rank_test_rmse"][index]),
        }
        row["candidate_id"] = _candidate_id(task, family, variant, params)
        for fold in range(search.n_splits_):
            row[f"split{fold}_test_rmse"] = float(
                -results[f"split{fold}_test_rmse"][index]
            )
            row[f"split{fold}_train_rmse"] = float(
                -results[f"split{fold}_train_rmse"][index]
            )
        rows.append(row)
    return rows


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["mean_test_rmse"],
        FAMILY_ORDER[row["family"]],
        VARIANT_ORDER[row["variant"]],
        row["params_json"],
    )


def _best_by_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winners = []
    for family in sorted({row["family"] for row in rows}, key=FAMILY_ORDER.get):
        winners.append(min((row for row in rows if row["family"] == family), key=_sort_key))
    return winners


def save_model(model: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return {
        "path": f"models/{path.name}",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _extract_mlp(estimator: Any) -> MLPRegressor:
    fitted = estimator
    if isinstance(fitted, OutlierFilteredRegressor):
        fitted = fitted.estimator_
    transformed = fitted.named_steps["model"]
    return transformed.regressor_


def _outlier_audit(estimator: Any) -> dict[str, Any]:
    if not isinstance(estimator, OutlierFilteredRegressor):
        raise TypeError("Expected a fitted OutlierFilteredRegressor")
    return {
        "input_rows": estimator.n_input_samples_,
        "fit_rows": estimator.n_fit_samples_,
        "removed_count": estimator.n_removed_,
        "removed_source_indices": [int(value) for value in estimator.removed_indices_],
        "filter_features": estimator.filter_features_,
        "thresholds": _jsonable(estimator.thresholds_),
    }


def _set_mlp_seed(estimator: Any, variant: str, seed: int) -> Any:
    prefix = _parameter_prefix(variant)
    return estimator.set_params(**{f"{prefix}regressor__random_state": seed})


def _fit_with_warnings(estimator: Any, X: pd.DataFrame, y: pd.Series) -> tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = clone(estimator).fit(X, y)
    messages = sorted({f"{item.category.__name__}: {item.message}" for item in caught})
    return fitted, messages


def run_training(
    project_root: Path,
    *,
    data_dir: Path,
    split_path: Path,
    models_dir: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = read_split_manifest(split_path, data_dir)
    train = prepare_training_frame(data_dir, manifest)
    cv = persisted_cv_positions(manifest, train.index)
    split_sha = sha256_file(split_path)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    registry: dict[str, dict[str, Any]] = {}
    preprocessing_audit: dict[str, Any] = {}
    fit_count = 0
    selection: dict[str, Any] = {}

    for task_key in ("direct_tensile_modulus", "direct_tensile_strength"):
        task = TASKS[task_key]
        X, y = task_xy(train, task)
        baseline = build_candidate("baseline", "raw")
        baseline_scores = cross_validate(
            baseline,
            X,
            y,
            cv=cv,
            scoring=SCORING,
            return_train_score=True,
            n_jobs=2,
        )
        fit_count += len(cv)
        baseline_row = _baseline_row(task_key, baseline_scores)
        all_rows.append(baseline_row)
        registry[baseline_row["candidate_id"]] = {
            "estimator": baseline,
            "params": {},
            "row": baseline_row,
        }

        task_rows = [baseline_row]
        preprocessing_audit[task_key] = {}
        for family in ("ridge", "random_forest", "gradient_boosting", "svr"):
            for variant in ("raw", "iqr_filtered"):
                estimator = build_candidate(family, variant)
                grid = parameter_grid(family, variant)
                search = GridSearchCV(
                    estimator,
                    grid,
                    scoring=SCORING,
                    refit="rmse",
                    cv=cv,
                    n_jobs=2,
                    return_train_score=True,
                    error_score="raise",
                )
                search.fit(X, y)
                candidate_count = len(search.cv_results_["params"])
                fit_count += candidate_count * len(cv) + 1
                rows = _grid_rows(task_key, family, variant, search)
                all_rows.extend(rows)
                task_rows.extend(rows)
                for row, params in zip(rows, search.cv_results_["params"], strict=True):
                    registry[row["candidate_id"]] = {
                        "estimator": estimator,
                        "params": params,
                        "row": row,
                    }
                if variant == "iqr_filtered":
                    preprocessing_audit[task_key][family] = _outlier_audit(
                        search.best_estimator_
                    )

        family_winners = _best_by_family(task_rows)
        overall = min(family_winners, key=_sort_key)
        classical = sorted(
            (row for row in family_winners if row["family"] != "baseline"),
            key=_sort_key,
        )
        artifacts = []
        for rank, row in enumerate(classical, start=1):
            entry = registry[row["candidate_id"]]
            fitted = clone(entry["estimator"]).set_params(**entry["params"]).fit(X, y)
            fit_count += 1
            artifact_path = models_dir / f"{task_key}_rank{rank}.joblib"
            artifact = save_model(fitted, artifact_path)
            artifact.update(
                {
                    "rank": rank,
                    "family": row["family"],
                    "variant": row["variant"],
                    "candidate_id": row["candidate_id"],
                    "cv_rmse": row["mean_test_rmse"],
                }
            )
            artifacts.append(artifact)
        baseline_fitted = clone(baseline).fit(X, y)
        fit_count += 1
        baseline_artifact = save_model(
            baseline_fitted, models_dir / f"{task_key}_baseline.joblib"
        )
        selection[task_key] = {
            "target": task.target,
            "unit": task.unit,
            "features": list(task.features),
            "selection_scope": "baseline_and_family_winners",
            "selected_candidate_id": overall["candidate_id"],
            "selected_family": overall["family"],
            "selected_variant": overall["variant"],
            "selected_params": json.loads(overall["params_json"]),
            "selected_cv_rmse": overall["mean_test_rmse"],
            "baseline_cv_rmse": baseline_row["mean_test_rmse"],
            "classical_family_artifacts": artifacts,
            "baseline_artifact": baseline_artifact,
        }

    ratio_task = TASKS["matrix_filler_ratio"]
    X_ratio, y_ratio = task_xy(train, ratio_task)
    ratio_baseline = build_candidate("baseline", "raw")
    ratio_baseline_scores = cross_validate(
        ratio_baseline,
        X_ratio,
        y_ratio,
        cv=cv,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=2,
    )
    fit_count += len(cv)
    ratio_baseline_row = _baseline_row(ratio_task.key, ratio_baseline_scores)
    all_rows.append(ratio_baseline_row)
    ratio_rows: list[dict[str, Any]] = []
    preprocessing_audit[ratio_task.key] = {}
    mlp_warning_messages: list[str] = []
    for variant in ("raw", "iqr_filtered"):
        estimator = build_candidate("mlp", variant, random_state=42)
        search = GridSearchCV(
            estimator,
            parameter_grid("mlp", variant),
            scoring=SCORING,
            refit="rmse",
            cv=cv,
            n_jobs=2,
            return_train_score=True,
            error_score="raise",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search.fit(X_ratio, y_ratio)
        mlp_warning_messages.extend(
            f"{item.category.__name__}: {item.message}" for item in caught
        )
        candidate_count = len(search.cv_results_["params"])
        fit_count += candidate_count * len(cv) + 1
        rows = _grid_rows(ratio_task.key, "mlp", variant, search)
        all_rows.extend(rows)
        ratio_rows.extend(rows)
        for row, params in zip(rows, search.cv_results_["params"], strict=True):
            registry[row["candidate_id"]] = {
                "estimator": estimator,
                "params": params,
                "row": row,
            }
        if variant == "iqr_filtered":
            preprocessing_audit[ratio_task.key]["mlp"] = _outlier_audit(
                search.best_estimator_
            )

    chosen_mlp = min(ratio_rows, key=_sort_key)
    chosen_entry = registry[chosen_mlp["candidate_id"]]
    chosen_estimator = clone(chosen_entry["estimator"]).set_params(
        **chosen_entry["params"]
    )
    chosen_fitted, final_warnings = _fit_with_warnings(
        chosen_estimator, X_ratio, y_ratio
    )
    fit_count += 1
    mlp_warning_messages.extend(final_warnings)
    mlp_artifact = save_model(
        chosen_fitted, models_dir / "matrix_filler_ratio_mlp.joblib"
    )
    ratio_baseline_fitted = clone(ratio_baseline).fit(X_ratio, y_ratio)
    fit_count += 1
    ratio_baseline_artifact = save_model(
        ratio_baseline_fitted, models_dir / "matrix_filler_ratio_baseline.joblib"
    )

    mlp = _extract_mlp(chosen_fitted)
    validation_scores = list(getattr(mlp, "validation_scores_", []))
    history_rows = []
    for iteration, loss in enumerate(mlp.loss_curve_, start=1):
        history_rows.append(
            {
                "iteration": iteration,
                "loss": float(loss),
                "validation_score": (
                    float(validation_scores[iteration - 1])
                    if iteration <= len(validation_scores)
                    else np.nan
                ),
            }
        )
    pd.DataFrame(history_rows).to_csv(
        reports_dir / "nn_history.csv", index=False, float_format="%.12g"
    )

    stability_rows = []
    for seed in (42, 43, 44):
        seeded = _set_mlp_seed(clone(chosen_estimator), chosen_mlp["variant"], seed)
        scores = cross_validate(
            seeded,
            X_ratio,
            y_ratio,
            cv=cv,
            scoring=SCORING,
            return_train_score=True,
            return_estimator=True,
            n_jobs=2,
        )
        fit_count += len(cv)
        for fold, fitted in enumerate(scores["estimator"], start=1):
            fold_mlp = _extract_mlp(fitted)
            stability_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "validation_rmse": float(-scores["test_rmse"][fold - 1]),
                    "validation_mae": float(-scores["test_mae"][fold - 1]),
                    "validation_r2": float(scores["test_r2"][fold - 1]),
                    "train_rmse": float(-scores["train_rmse"][fold - 1]),
                    "n_iter": int(fold_mlp.n_iter_),
                    "reached_max_iter": bool(fold_mlp.n_iter_ >= fold_mlp.max_iter),
                }
            )
    pd.DataFrame(stability_rows).to_csv(
        reports_dir / "nn_stability.csv", index=False, float_format="%.12g"
    )

    selection[ratio_task.key] = {
        "target": ratio_task.target,
        "unit": ratio_task.unit,
        "features": list(ratio_task.features),
        "selection_scope": "mlp_candidates",
        "selected_candidate_id": chosen_mlp["candidate_id"],
        "selected_family": "mlp",
        "selected_variant": chosen_mlp["variant"],
        "selected_params": json.loads(chosen_mlp["params_json"]),
        "selected_cv_rmse": chosen_mlp["mean_test_rmse"],
        "baseline_cv_rmse": ratio_baseline_row["mean_test_rmse"],
        "mlp_artifact": mlp_artifact,
        "baseline_artifact": ratio_baseline_artifact,
        "n_iter_full_train": int(mlp.n_iter_),
        "reached_max_iter_full_train": bool(mlp.n_iter_ >= mlp.max_iter),
        "warnings": sorted(set(mlp_warning_messages)),
    }

    results_frame = pd.DataFrame(all_rows)
    results_frame = results_frame.sort_values(
        ["task", "mean_test_rmse", "family", "variant", "params_json"]
    )
    results_frame.to_csv(
        reports_dir / "cv_results.csv", index=False, float_format="%.12g"
    )
    summary_rows = []
    for task_key in TASKS:
        winners = _best_by_family(
            [row for row in all_rows if row["task"] == task_key]
        )
        selected_id = selection[task_key]["selected_candidate_id"]
        for row in winners:
            summary_rows.append(
                {
                    "task": task_key,
                    "family": row["family"],
                    "variant": row["variant"],
                    "candidate_id": row["candidate_id"],
                    "cv_rmse_mean": row["mean_test_rmse"],
                    "cv_rmse_std": row["std_test_rmse"],
                    "cv_mae_mean": row["mean_test_mae"],
                    "cv_r2_mean": row["mean_test_r2"],
                    "train_cv_rmse_mean": row["mean_train_rmse"],
                    "selected": row["candidate_id"] == selected_id,
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        reports_dir / "cv_summary.csv", index=False, float_format="%.12g"
    )

    selection_document = {
        "schema_version": 1,
        "frozen_before_test": True,
        "test_rows_used": 0,
        "split_manifest_sha256": split_sha,
        "selection_rule": "minimum mean 10-fold validation RMSE with fixed tie order",
        "tasks": selection,
    }
    (reports_dir / "model_selection.json").write_text(
        json.dumps(selection_document, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (reports_dir / "preprocessing_audit.json").write_text(
        json.dumps(preprocessing_audit, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    hardener = train[HARDENER_AMOUNT]
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "train_rows": len(train),
        "test_rows_used": 0,
        "cv_folds": len(cv),
        "grid_search_n_jobs": 2,
        "estimated_fit_calls_including_refits": fit_count,
        "elapsed_seconds": time.perf_counter() - started,
        "split_manifest_sha256": split_sha,
        "raw_sources_sha256": manifest["raw_sources_sha256"],
        "hardener_train_observation": {
            "field": HARDENER_AMOUNT,
            "count_above_100": int((hardener > 100).sum()),
            "minimum": float(hardener.min()),
            "maximum": float(hardener.max()),
            "interpretation": (
                "Values above 100 would be impossible for mass percent of a finished "
                "mixture, but the percentage basis is undocumented; no rows were removed."
            ),
            "synthetic_origin_claimed": False,
        },
    }
    (reports_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"selection": selection_document, "metadata": metadata}


def assert_prediction_reload_equal(model: Any, path: Path, X: pd.DataFrame) -> None:
    before = np.asarray(model.predict(X))
    loaded = joblib.load(path)
    after = np.asarray(loaded.predict(X))
    if not np.allclose(before, after, rtol=0.0, atol=0.0):
        raise AssertionError("Reloaded model predictions differ from in-memory predictions")


def native_unit_rmse(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    """Explicit helper used to verify inverse target transformation semantics."""
    return float(root_mean_squared_error(y, estimator.predict(X)))

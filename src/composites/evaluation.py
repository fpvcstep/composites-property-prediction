"""Final evaluation of frozen train-only model selections."""

from __future__ import annotations

import json
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from composites.data import load_and_merge, sha256_file
from composites.schema import HARDENER_AMOUNT
from composites.split import select_partition, validate_split_manifest
from composites.training import TASKS, TaskSpec, task_xy


@dataclass(frozen=True)
class ArtifactSpec:
    task_key: str
    family: str
    artifact: str
    sha256: str
    selected: bool


def load_frozen_selection(
    selection_path: Path, split_path: Path
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("frozen_before_test") is not True:
        raise ValueError("Model selection was not frozen before test evaluation")
    if selection.get("test_rows_used") != 0:
        raise ValueError("Selection report already records test usage")
    if selection.get("split_manifest_sha256") != sha256_file(split_path):
        raise ValueError("Selection is bound to a different split manifest")
    return selection


def prepare_partitions(
    data_dir: Path, split_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    merged, _ = load_and_merge(data_dir)
    train = select_partition(merged, manifest["train_indices"]).set_index(
        "source_index", drop=True
    )
    test = select_partition(merged, manifest["test_indices"]).set_index(
        "source_index", drop=True
    )
    if len(train) != 716 or len(test) != 307:
        raise ValueError("Unexpected persisted partition sizes")
    if not train.index.is_unique or not test.index.is_unique:
        raise ValueError("Partition source indices must be unique")
    if set(train.index) & set(test.index):
        raise ValueError("Train and test source indices overlap")
    if set(train.index) | set(test.index) != set(manifest["all_indices"]):
        raise ValueError("Train and test do not cover every persisted source index")
    return train, test, manifest


def artifact_specs(selection: dict[str, Any]) -> list[ArtifactSpec]:
    specs: list[ArtifactSpec] = []
    for task_key in ("direct_tensile_modulus", "direct_tensile_strength"):
        task = selection["tasks"][task_key]
        specs.append(
            ArtifactSpec(
                task_key,
                "baseline",
                task["baseline_artifact"]["path"],
                task["baseline_artifact"]["sha256"],
                task["selected_family"] == "baseline",
            )
        )
        for item in task["classical_family_artifacts"]:
            specs.append(
                ArtifactSpec(
                    task_key,
                    item["family"],
                    item["path"],
                    item["sha256"],
                    item["candidate_id"] == task["selected_candidate_id"],
                )
            )
    ratio = selection["tasks"]["matrix_filler_ratio"]
    specs.extend(
        [
            ArtifactSpec(
                "matrix_filler_ratio",
                "baseline",
                ratio["baseline_artifact"]["path"],
                ratio["baseline_artifact"]["sha256"],
                False,
            ),
            ArtifactSpec(
                "matrix_filler_ratio",
                "mlp",
                ratio["mlp_artifact"]["path"],
                ratio["mlp_artifact"]["sha256"],
                True,
            ),
        ]
    )
    if len(specs) != 12:
        raise ValueError(f"Expected 12 frozen artifacts, found {len(specs)}")
    return specs


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if true.shape != predicted.shape or true.ndim != 1:
        raise ValueError("Prediction and target shapes differ")
    if not np.isfinite(predicted).all():
        raise ValueError("Predictions contain non-finite values")
    return {
        "rmse": float(root_mean_squared_error(true, predicted)),
        "mae": float(mean_absolute_error(true, predicted)),
        "r2": float(r2_score(true, predicted)),
    }


def _selected_artifact(
    selection: dict[str, Any], task_key: str
) -> tuple[str, str]:
    task = selection["tasks"][task_key]
    if task_key == "matrix_filler_ratio":
        return task["selected_family"], task["mlp_artifact"]["path"]
    if task["selected_family"] == "baseline":
        return "baseline", task["baseline_artifact"]["path"]
    for item in task["classical_family_artifacts"]:
        if item["candidate_id"] == task["selected_candidate_id"]:
            return item["family"], item["path"]
    raise ValueError(f"Selected artifact not found for {task_key}")


def _resolve_artifact(project_root: Path, artifact: str) -> Path:
    relative = Path(artifact)
    if relative.is_absolute() or relative.parts[:1] != ("models",):
        raise ValueError(f"Unexpected artifact path: {artifact}")
    resolved = (project_root / relative).resolve()
    models_root = (project_root / "models").resolve()
    if not resolved.is_relative_to(models_root) or not resolved.is_file():
        raise ValueError(f"Artifact is absent or outside models: {artifact}")
    return resolved


def load_verified_artifact(project_root: Path, spec: ArtifactSpec) -> Any:
    """Verify the frozen SHA-256 before deserializing an artifact."""
    path = _resolve_artifact(project_root, spec.artifact)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != spec.sha256:
        raise ValueError(
            f"Frozen artifact hash mismatch for {spec.artifact}: "
            f"expected {spec.sha256}, got {actual_sha256}"
        )
    return joblib.load(path)


def _feature_ranges(frame: pd.DataFrame, task: TaskSpec) -> dict[str, Any]:
    return {
        feature: {
            "min": float(frame[feature].min()),
            "median": float(frame[feature].median()),
            "max": float(frame[feature].max()),
        }
        for feature in task.features
    }


def _plot_predictions(predictions: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for axis, (task_key, task) in zip(axes, TASKS.items(), strict=True):
        subset = predictions.loc[
            (predictions["task"] == task_key) & predictions["selected"]
        ]
        for partition, color in (("train", "#4c78a8"), ("test", "#f58518")):
            part = subset.loc[subset["partition"] == partition]
            axis.scatter(
                part["y_true"],
                part["y_pred"],
                s=18,
                alpha=0.55,
                label=f"{partition}, n={len(part)}",
                color=color,
                edgecolors="none",
            )
        low = float(min(subset["y_true"].min(), subset["y_pred"].min()))
        high = float(max(subset["y_true"].max(), subset["y_pred"].max()))
        axis.plot([low, high], [low, high], "--", color="black", linewidth=1)
        axis.set_title(task.target)
        axis.set_xlabel(f"Фактическое значение, {task.unit}")
        axis.set_ylabel(f"Прогноз, {task.unit}")
        axis.legend()
    fig.suptitle("Факт и прогноз выбранных моделей: полные train и test")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals(predictions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for axis, (task_key, task) in zip(axes, TASKS.items(), strict=True):
        subset = predictions.loc[
            (predictions["task"] == task_key) & predictions["selected"]
        ].copy()
        subset["residual"] = subset["y_true"] - subset["y_pred"]
        for partition, color in (("train", "#4c78a8"), ("test", "#f58518")):
            part = subset.loc[subset["partition"] == partition]
            axis.scatter(
                part["y_pred"],
                part["residual"],
                s=18,
                alpha=0.55,
                label=partition,
                color=color,
                edgecolors="none",
            )
        axis.axhline(0, linestyle="--", color="black", linewidth=1)
        axis.set_title(task.target)
        axis.set_xlabel(f"Прогноз, {task.unit}")
        axis.set_ylabel(f"Остаток факт − прогноз, {task.unit}")
        axis.legend()
    fig.suptitle("Остатки выбранных моделей: полные train и test")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_nn_diagnostics(modeling_dir: Path, output: Path) -> None:
    history = pd.read_csv(modeling_dir / "nn_history.csv")
    stability = pd.read_csv(modeling_dir / "nn_stability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    loss_line = axes[0].plot(
        history["iteration"],
        history["loss"],
        label="Функция потерь train",
        color="#4c78a8",
    )
    validation_axis = axes[0].twinx()
    validation_line = validation_axis.plot(
        history["iteration"],
        history["validation_score"],
        label="Внутренний validation R²",
        color="#f58518",
    )
    axes[0].set_xlabel("Итерация")
    axes[0].set_ylabel("Потери MLP: ½MSE + L2")
    validation_axis.set_ylabel("R² внутренней validation-части")
    axes[0].set_title("MLP: кривая обучения стандартизованной y, полный train n=716")
    axes[0].legend(loss_line + validation_line, [line.get_label() for line in loss_line + validation_line])
    sns.boxplot(data=stability, x="seed", y="validation_rmse", ax=axes[1])
    sns.stripplot(
        data=stability,
        x="seed",
        y="validation_rmse",
        color="black",
        alpha=0.6,
        size=4,
        ax=axes[1],
    )
    axes[1].set_xlabel("Seed инициализации")
    axes[1].set_ylabel("CV RMSE отношения, исходная единица не указана")
    axes[1].set_title("MLP: устойчивость на 10 сохранённых фолдах")
    fig.suptitle(
        "Диагностика MLP: early stopping на внутренних 10% train; test не использован"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_importance(importance: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    for axis, (task_key, task) in zip(axes, TASKS.items(), strict=True):
        subset = importance.loc[importance["task"] == task_key].sort_values(
            "importance_mean_rmse"
        )
        axis.barh(
            subset["feature"],
            subset["importance_mean_rmse"],
            xerr=subset["importance_std_rmse"],
            color="#4c78a8",
            alpha=0.85,
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(task.target)
        axis.set_xlabel(f"Рост RMSE после перестановки, {task.unit}")
        axis.tick_params(axis="y", labelsize=7)
        if subset["importance_mean_rmse"].eq(0.0).all():
            axis.text(
                0.5,
                0.5,
                "Baseline не использует признаки:\nвсе важности равны нулю",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "gray"},
            )
    fig.suptitle("Пермутационная важность на отложенном test, 10 повторов, seed 42")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_single_importance(
    importance: pd.DataFrame, task_key: str, output: Path
) -> None:
    task = TASKS[task_key]
    subset = importance.loc[importance["task"] == task_key].sort_values(
        "importance_mean_rmse"
    )
    labels = [textwrap.fill(value, width=32) for value in subset["feature"]]
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        labels,
        subset["importance_mean_rmse"],
        xerr=subset["importance_std_rmse"],
        color="#4c78a8",
        alpha=0.85,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_title(
        f"Пермутационная важность: {task.target}\n"
        "отложенный test, 10 повторов, seed 42"
    )
    axis.set_xlabel(f"Рост RMSE после перестановки, {task.unit}")
    axis.tick_params(axis="y", labelsize=9)
    if subset["importance_mean_rmse"].eq(0.0).all():
        axis.text(
            0.5,
            0.5,
            "Выбранный baseline не использует признаки:\n"
            "нулевые важности не означают физическую незначимость",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "gray"},
        )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_results(
    output: Path,
    selected_metrics: pd.DataFrame,
    selection: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    test = selected_metrics.loc[selected_metrics["partition"] == "test"].set_index(
        "task"
    )
    train = selected_metrics.loc[selected_metrics["partition"] == "train"].set_index(
        "task"
    )
    lines = [
        "# Результаты моделирования",
        "",
        "Модели и гиперпараметры были заморожены по десятифолдовой кросс-валидации "
        "до открытия test. Итоговая оценка использует все 716 строк train и все 307 "
        "строк test; повторного выбора по test не выполнялось.",
        "",
        "## Метрики выбранных моделей",
        "",
        "| Задача | Модель | Train RMSE | Test RMSE | Test MAE | Test R² |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, task in TASKS.items():
        model_family = selection["tasks"][key]["selected_family"]
        lines.append(
            f"| {task.target} | {model_family} | {train.loc[key, 'rmse']:.3f} | "
            f"{test.loc[key, 'rmse']:.3f} | {test.loc[key, 'mae']:.3f} | "
            f"{test.loc[key, 'r2']:.3f} |"
        )
    modulus_r2 = float(test.loc["direct_tensile_modulus", "r2"])
    strength_r2 = float(test.loc["direct_tensile_strength", "r2"])
    ratio_r2 = float(test.loc["matrix_filler_ratio", "r2"])
    lines.extend(
        [
            "",
            "Для модуля упругости CV выбрала средний baseline. Его прогноз постоянный, "
            f"поэтому пермутационные важности равны нулю; test R² = {modulus_r2:.3f}. "
            "Нулевая важность здесь означает отсутствие использования признаков baseline, "
            "а не доказанную физическую незначимость факторов.",
            "",
            "Для прочности выбрана Gradient Boosting, но её CV-преимущество над baseline "
            f"было минимальным; test R² = {strength_r2:.3f}. Разрыв train/test следует "
            "трактовать как ограничение обобщения, а не как промышленную точность.",
            "",
            "Обязательная MLP отношения уступила baseline ещё на CV и сохранялась по "
            f"условию задания; test R² = {ratio_r2:.3f}. Она оценивает отношение по полному "
            "профилю, но не оптимизирует состав и не устанавливает причинность.",
            "",
            "## Интерпретация и ограничения",
            "",
            "- Пермутационная важность рассчитана на test с 10 повторами и seed 42. "
            "Отрицательное значение допустимо и означает, что перестановка случайно не "
            "ухудшила ошибку на этой ограниченной выборке.",
            "- В поле количества отвердителя 468 из 716 train-значений выше 100. Для "
            "массовой доли готовой смеси это невозможно, но база процентов неизвестна; "
            "наблюдения сохранены и синтетическое происхождение не заявляется.",
            "- Происхождение наблюдений и группы опытов неизвестны. Случайный holdout "
            "характеризует только эту выборку и не доказывает перенос на новые материалы.",
            "- Результаты не заменяют лабораторные испытания.",
            "",
            "## Рисунки",
            "",
            "1. `predicted_vs_actual.png` — факт и прогноз выбранных моделей для полных "
            "train/test, с исходными единицами целей и линией идеального прогноза.",
            "2. `residuals.png` — остатки факт минус прогноз для полных train/test; "
            "систематические структуры указывают на ограничения моделей.",
            "3. `nn_diagnostics.png` — loss/validation score полного train-fit и CV RMSE "
            "по seeds 42–44; seed не выбирался по test.",
            "4. `permutation_importance.png` — рост test RMSE после перестановки признака; "
            "для baseline-модуля все значения нулевые по определению.",
            "   Для печати на A4 те же значения сохранены отдельными читаемыми файлами "
            "`permutation_importance_<task>.png` для каждой из трёх задач.",
            "",
            f"Воспроизводимая оценка заняла {elapsed_seconds:.1f} с. Полные метрики и "
            "прогнозы находятся в `reports/evaluation/`.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(
    project_root: Path,
    *,
    data_dir: Path,
    split_path: Path,
    selection_path: Path,
    models_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
    results_doc: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    selection = load_frozen_selection(selection_path, split_path)
    train, test, _ = prepare_partitions(data_dir, split_path)
    partitions = {"train": train, "test": test}

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    loaded: dict[str, Any] = {}
    for spec in artifact_specs(selection):
        model = load_verified_artifact(project_root, spec)
        loaded[spec.artifact] = model
        task = TASKS[spec.task_key]
        for partition_name, frame in partitions.items():
            X, y = task_xy(frame, task)
            predictions = np.asarray(model.predict(X), dtype=float)
            metrics = regression_metrics(y, predictions)
            metric_rows.append(
                {
                    "task": spec.task_key,
                    "family": spec.family,
                    "partition": partition_name,
                    "n": len(frame),
                    **metrics,
                    "selected": spec.selected,
                    "artifact": spec.artifact,
                }
            )
            prediction_rows.extend(
                {
                    "task": spec.task_key,
                    "family": spec.family,
                    "partition": partition_name,
                    "source_index": int(source_index),
                    "y_true": float(actual),
                    "y_pred": float(predicted),
                    "selected": spec.selected,
                    "artifact": spec.artifact,
                }
                for source_index, actual, predicted in zip(
                    frame.index, y.to_numpy(), predictions, strict=True
                )
            )

    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_doc.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(
        reports_dir / "metrics.csv", index=False, float_format="%.12g"
    )
    predictions_frame.loc[
        :, ["task", "family", "partition", "source_index", "y_true", "y_pred"]
    ].to_csv(
        reports_dir / "predictions.csv", index=False, float_format="%.12g"
    )

    importance_rows: list[dict[str, Any]] = []
    for task_key, task in TASKS.items():
        family, artifact = _selected_artifact(selection, task_key)
        if family == "baseline":
            importance_rows.extend(
                {
                    "task": task_key,
                    "family": family,
                    "feature": feature,
                    "importance_mean_rmse": 0.0,
                    "importance_std_rmse": 0.0,
                }
                for feature in task.features
            )
            continue
        model = loaded[artifact]
        X_test, y_test = task_xy(test, task)
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="neg_root_mean_squared_error",
            n_repeats=10,
            random_state=42,
            n_jobs=2,
        )
        importance_rows.extend(
            {
                "task": task_key,
                "family": family,
                "feature": feature,
                "importance_mean_rmse": float(mean),
                "importance_std_rmse": float(std),
            }
            for feature, mean, std in zip(
                task.features, result.importances_mean, result.importances_std, strict=True
            )
        )
    importance_frame = pd.DataFrame(importance_rows)
    importance_frame.to_csv(
        reports_dir / "permutation_importance.csv",
        index=False,
        float_format="%.12g",
    )

    selected_metrics = metrics_frame.loc[metrics_frame["selected"]].copy()
    manifest_tasks: dict[str, Any] = {}
    for task_key, task in TASKS.items():
        family, artifact = _selected_artifact(selection, task_key)
        task_test = selected_metrics.loc[
            (selected_metrics["task"] == task_key)
            & (selected_metrics["partition"] == "test")
        ].iloc[0]
        selection_task = selection["tasks"][task_key]
        manifest_tasks[task_key] = {
            "target": task.target,
            "unit": task.unit,
            "features": list(task.features),
            "model_family": family,
            "artifact": Path(artifact).name,
            "cv_rmse": float(selection_task["selected_cv_rmse"]),
            "baseline_cv_rmse": float(selection_task["baseline_cv_rmse"]),
            "test_metrics": {
                "rmse": float(task_test["rmse"]),
                "mae": float(task_test["mae"]),
                "r2": float(task_test["r2"]),
            },
            "train_feature_ranges": _feature_ranges(train, task),
        }
    manifest = {"schema_version": 1, "tasks": manifest_tasks}
    (models_dir / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _plot_predictions(predictions_frame, figures_dir / "predicted_vs_actual.png")
    _plot_residuals(predictions_frame, figures_dir / "residuals.png")
    _plot_nn_diagnostics(
        project_root / "reports" / "modeling", figures_dir / "nn_diagnostics.png"
    )
    _plot_importance(importance_frame, figures_dir / "permutation_importance.png")
    for task_key in TASKS:
        _plot_single_importance(
            importance_frame,
            task_key,
            figures_dir / f"permutation_importance_{task_key}.png",
        )
    elapsed = time.perf_counter() - started
    _write_results(results_doc, selected_metrics, selection, elapsed)
    return {
        "elapsed_seconds": elapsed,
        "metrics": selected_metrics.to_dict("records"),
        "manifest": manifest,
    }

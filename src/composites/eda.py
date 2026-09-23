"""Training-only exploratory analysis and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from composites.data import load_and_merge, sha256_file
from composites.schema import (
    BP_SCHEMA,
    MERGED_COLUMNS,
    NUP_SCHEMA,
    PATCH_ANGLE,
    SOURCE_INDEX,
)
from composites.split import select_partition, validate_split_manifest

UNITS = {
    "Соотношение матрица-наполнитель": "единица не указана",
    "Плотность, кг/м3": "кг/м³",
    "модуль упругости, ГПа": "ГПа",
    "Количество отвердителя, м.%": "м.%",
    "Содержание эпоксидных групп,%_2": "% (смысл _2 неизвестен)",
    "Температура вспышки, С_2": "как указано в источнике: С_2",
    "Поверхностная плотность, г/м2": "г/м²",
    "Модуль упругости при растяжении, ГПа": "ГПа",
    "Прочность при растяжении, МПа": "МПа",
    "Потребление смолы, г/м2": "г/м²",
    "Угол нашивки, град": "градусы",
    "Шаг нашивки": "единица неизвестна",
    "Плотность нашивки": "единица неизвестна",
}


def _read_manifest(path: Path, data_dir: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    actual_hashes = {
        BP_SCHEMA.filename: sha256_file(data_dir / BP_SCHEMA.filename),
        NUP_SCHEMA.filename: sha256_file(data_dir / NUP_SCHEMA.filename),
    }
    if manifest["raw_sources_sha256"] != actual_hashes:
        raise ValueError("Raw files differ from those bound to the split manifest")
    return manifest


def _save_histograms(train: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    for axis, column in zip(axes.flat, MERGED_COLUMNS, strict=False):
        sns.histplot(train[column], bins=30, kde=True, ax=axis)
        axis.set_title(column, fontsize=9)
        axis.set_xlabel(UNITS[column], fontsize=8)
        axis.set_ylabel("Число наблюдений")
    for axis in axes.flat[len(MERGED_COLUMNS) :]:
        axis.set_visible(False)
    fig.suptitle("Распределения полей: train, n=716", fontsize=16)
    fig.tight_layout()
    fig.savefig(figures_dir / "histograms_train.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_boxplots(train: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(20, 14))
    for axis, column in zip(axes.flat, MERGED_COLUMNS, strict=False):
        sns.boxplot(x=train[column], ax=axis, orient="h")
        axis.set_title(column, fontsize=9)
        axis.set_xlabel(UNITS[column], fontsize=8)
    for axis in axes.flat[len(MERGED_COLUMNS) :]:
        axis.set_visible(False)
    fig.suptitle("Boxplot и статистически редкие диапазоны: train, n=716", fontsize=16)
    fig.tight_layout()
    fig.savefig(figures_dir / "boxplots_train.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_scatter_blocks(train: pd.DataFrame, figures_dir: Path) -> list[dict[str, Any]]:
    groups = [
        list(MERGED_COLUMNS[0:5]),
        list(MERGED_COLUMNS[5:9]),
        list(MERGED_COLUMNS[9:13]),
    ]
    registry: list[dict[str, Any]] = []
    for left_number, left in enumerate(groups, start=1):
        for right_number in range(left_number, len(groups) + 1):
            right = groups[right_number - 1]
            fig, axes = plt.subplots(
                len(right), len(left), figsize=(3.2 * len(left), 3.0 * len(right)), squeeze=False
            )
            for row, y_name in enumerate(right):
                for column_index, x_name in enumerate(left):
                    axis = axes[row, column_index]
                    if x_name == y_name:
                        axis.hist(train[x_name], bins=25, color="#4c78a8", alpha=0.85)
                    else:
                        axis.scatter(
                            train[x_name], train[y_name], s=8, alpha=0.35, edgecolors="none"
                        )
                    axis.set_xlabel(f"{x_name}\n[{UNITS[x_name]}]", fontsize=6)
                    axis.set_ylabel(f"{y_name}\n[{UNITS[y_name]}]", fontsize=6)
                    axis.tick_params(labelsize=6)
            filename = f"scatter_block_{left_number}_{right_number}_train.png"
            fig.suptitle(
                f"Попарные диаграммы: блоки {left_number} и {right_number}, train n=716",
                fontsize=14,
            )
            fig.tight_layout()
            fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            registry.append(
                {
                    "file": filename,
                    "left": left,
                    "right": right,
                }
            )
    return registry


def _save_correlations(
    pearson: pd.DataFrame, spearman: pd.DataFrame, figures_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    for axis, matrix, title in (
        (axes[0], pearson, "Pearson"),
        (axes[1], spearman, "Spearman"),
    ):
        sns.heatmap(matrix, cmap="vlag", center=0, vmin=-1, vmax=1, ax=axis)
        axis.set_title(f"Корреляция {title}: train, n=716")
        axis.tick_params(axis="both", labelsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "correlations_train.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_normalization(
    train: pd.DataFrame, normalized: pd.DataFrame, figures_dir: Path
) -> None:
    fig, axes = plt.subplots(len(MERGED_COLUMNS), 2, figsize=(14, 34))
    for row, column in enumerate(MERGED_COLUMNS):
        axes[row, 0].hist(train[column], bins=25, color="#4c78a8", alpha=0.85)
        axes[row, 1].hist(normalized[column], bins=25, color="#f58518", alpha=0.85)
        axes[row, 0].set_ylabel(column, fontsize=7)
        axes[row, 0].set_xlabel(UNITS[column], fontsize=7)
        axes[row, 1].set_xlabel("масштаб [0, 1]", fontsize=7)
        axes[row, 0].tick_params(labelsize=6)
        axes[row, 1].tick_params(labelsize=6)
    axes[0, 0].set_title("До MinMaxScaler")
    axes[0, 1].set_title("После MinMaxScaler, fit только на train")
    fig.suptitle(
        "Распределения до и после иллюстрационной нормализации: train, n=716",
        y=0.997,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.988), h_pad=1.4)
    fig.savefig(
        figures_dir / "normalization_before_after_train.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def _correlations_long(pearson: pd.DataFrame, spearman: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, matrix in (("pearson", pearson), ("spearman", spearman)):
        for row, left in enumerate(matrix.columns):
            for right in matrix.columns[row + 1 :]:
                rows.append(
                    {
                        "method": method,
                        "variable_1": left,
                        "variable_2": right,
                        "correlation": float(matrix.loc[left, right]),
                    }
                )
    return pd.DataFrame(rows)


def _top_pairs(correlations: pd.DataFrame, method: str, limit: int = 5) -> list[dict[str, Any]]:
    subset = correlations.loc[correlations["method"] == method].copy()
    subset["absolute"] = subset["correlation"].abs()
    return subset.nlargest(limit, "absolute").drop(columns="absolute").to_dict("records")


def _write_eda_markdown(
    output: Path,
    summary: pd.DataFrame,
    outliers: pd.DataFrame,
    angle_counts: dict[str, int],
    top_pearson: list[dict[str, Any]],
    top_spearman: list[dict[str, Any]],
    scatter_registry: list[dict[str, Any]],
) -> None:
    top_outliers = outliers.sort_values("candidate_count", ascending=False).head(5)
    lines = [
        "# Разведочный анализ обучающей выборки",
        "",
        "Анализ выполнен только на 716 строках train из зафиксированного разбиения. "
        "Отложенные 307 строк не использовались для расчётов, графиков или решений.",
        "",
        "## Краткие результаты",
        "",
        f"- Пропуски во всех 13 полях: {int(summary['null_count'].sum())}.",
        f"- Значения угла нашивки на train: {angle_counts}.",
        "- Значения 0° и 90° подтверждают бинарную числовую кодировку 0/1; другие "
        "категории не вводятся.",
        "- Границы 1,5 IQR рассматриваются только как статистические кандидаты. "
        "Они не доказывают ошибку измерения, строки на этапе EDA не удалялись.",
        "- Происхождение наблюдений и наличие скрытых групп остаются неизвестными; "
        "выводы ограничены интерполяцией в наблюдавшихся диапазонах.",
        "",
        "Пять полей с наибольшим числом IQR-кандидатов:",
        "",
        "| Поле | Кандидаты | Доля train |",
        "|---|---:|---:|",
    ]
    for row in top_outliers.itertuples(index=False):
        lines.append(f"| {row.variable} | {row.candidate_count} | {row.candidate_share:.3f} |")
    lines.extend(["", "Сильнейшие по модулю связи не интерпретируются как причинные:", ""])
    for method, pairs in (("Pearson", top_pearson), ("Spearman", top_spearman)):
        lines.append(f"**{method}:**")
        lines.append("")
        for pair in pairs:
            lines.append(
                f"- {pair['variable_1']} ↔ {pair['variable_2']}: "
                f"{pair['correlation']:.3f}."
            )
        lines.append("")
    lines.extend(
        [
            "## Рисунки",
            "",
            "**Рисунок 1 — Гистограммы всех 13 полей.** Файл `histograms_train.png`; "
            "выборка train, n=716; единицы указаны на осях. Графики показывают форму "
            "распределений и редкие хвосты, но сами по себе не задают очистку.",
            "",
            "**Рисунок 2 — Boxplot всех 13 полей.** Файл `boxplots_train.png`; выборка "
            "train, n=716; единицы указаны на осях. Точки за усами являются кандидатами "
            "по правилу 1,5 IQR, а не подтверждёнными ошибками.",
            "",
            "**Рисунок 3 — Матрицы Pearson и Spearman.** Файл `correlations_train.png`; "
            "выборка train, n=716; коэффициенты безразмерны. Различие коэффициентов "
            "помогает видеть линейные и монотонные связи без причинной трактовки.",
            "",
            "**Рисунок 4 — Распределения до и после MinMaxScaler.** Файл "
            "`normalization_before_after_train.png`; выборка train, n=716; слева исходные "
            "единицы, справа шкала [0, 1]. Scaler обучен только на train и используется "
            "только для иллюстрации, а не переносится в модельный CV.",
            "",
            "### Полное покрытие попарных диаграмм",
            "",
            "Три блока образуют шесть файлов. Декартовы произведения блоков покрывают "
            "каждую пару 13 полей, включая все пары с тремя целевыми переменными; "
            "диагонали повторяют одномерные распределения.",
            "",
        ]
    )
    for number, item in enumerate(scatter_registry, start=5):
        left = "; ".join(f"{name} [{UNITS[name]}]" for name in item["left"])
        right = "; ".join(f"{name} [{UNITS[name]}]" for name in item["right"])
        lines.extend(
            [
                f"**Рисунок {number} — Попарный блок.** Файл `{item['file']}`; train, "
                f"n=716. Оси X: {left}. Оси Y: {right}. Рисунок служит для поиска "
                "формы связей, кластеров и разреженных областей; наблюдаемая связь не "
                "считается причинной.",
                "",
            ]
        )
    lines.extend(
        [
            "## Ограничения и решения для следующего этапа",
            "",
            "- `Температура вспышки, С_2` не переименовывается в температуру отверждения.",
            "- `модуль упругости, ГПа` остаётся отдельным входом и не смешивается с "
            "целевым модулем при растяжении.",
            "- Производные доли и физически не подтверждённые признаки не создаются.",
            "- В модельных pipeline пропуски запрещены; imputer не нужен по фактическому аудиту.",
            "- Raw и IQR-filtered варианты будут сравниваться внутри CV; очистка заранее "
            "не выбирается.",
            "",
            "Полные числовые результаты находятся в `reports/eda/`, рисунки — в "
            "`figures/eda/`.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training_eda(
    data_dir: Path,
    split_path: Path,
    reports_dir: Path,
    figures_dir: Path,
    documentation_path: Path,
) -> dict[str, Any]:
    sns.set_theme(style="whitegrid")
    manifest = _read_manifest(split_path, data_dir)
    merged, _ = load_and_merge(data_dir)
    train = select_partition(merged, manifest["train_indices"])
    values = train.loc[:, MERGED_COLUMNS]

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    documentation_path.parent.mkdir(parents=True, exist_ok=True)

    summary = values.agg(["count", "mean", "median", "min", "max", "std"]).T
    summary["null_count"] = values.isna().sum()
    summary["unique_count"] = values.nunique()
    summary.index.name = "variable"
    summary.reset_index().to_csv(
        reports_dir / "summary.csv", index=False, float_format="%.10g"
    )

    pearson = values.corr(method="pearson")
    spearman = values.corr(method="spearman")
    correlations = _correlations_long(pearson, spearman)
    correlations.to_csv(
        reports_dir / "correlations.csv", index=False, float_format="%.10g"
    )

    outlier_rows = []
    for column in MERGED_COLUMNS:
        if column == PATCH_ANGLE:
            continue
        q1 = float(values[column].quantile(0.25))
        q3 = float(values[column].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        candidate = ~values[column].between(lower, upper, inclusive="both")
        outlier_rows.append(
            {
                "variable": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "candidate_count": int(candidate.sum()),
                "candidate_share": float(candidate.mean()),
            }
        )
    outliers = pd.DataFrame(outlier_rows)
    outliers.to_csv(reports_dir / "outliers.csv", index=False, float_format="%.10g")

    scaler = MinMaxScaler().fit(values)
    normalized = pd.DataFrame(scaler.transform(values), columns=MERGED_COLUMNS)
    normalization = pd.DataFrame(
        {
            "variable": MERGED_COLUMNS,
            "minimum_before": values.min().to_numpy(),
            "maximum_before": values.max().to_numpy(),
            "minimum_after": normalized.min().to_numpy(),
            "maximum_after": normalized.max().to_numpy(),
        }
    )
    normalization.to_csv(
        reports_dir / "normalization.csv", index=False, float_format="%.10g"
    )

    _save_histograms(values, figures_dir)
    _save_boxplots(values, figures_dir)
    scatter_registry = _save_scatter_blocks(values, figures_dir)
    _save_correlations(pearson, spearman, figures_dir)
    _save_normalization(values, normalized, figures_dir)

    angle_counts = {
        str(int(value)): int(count)
        for value, count in values[PATCH_ANGLE].value_counts().sort_index().items()
    }
    rare_ranges = {
        column: {
            "p01": float(values[column].quantile(0.01)),
            "p99": float(values[column].quantile(0.99)),
            "below_p01": int((values[column] < values[column].quantile(0.01)).sum()),
            "above_p99": int((values[column] > values[column].quantile(0.99)).sum()),
        }
        for column in MERGED_COLUMNS
        if column != PATCH_ANGLE
    }
    summary_json: dict[str, Any] = {
        "scope": "training_only",
        "train_rows": int(len(train)),
        "test_rows_used": 0,
        "train_source_indices_sha256": hashlib_indices(manifest["train_indices"]),
        "raw_sources_sha256": manifest["raw_sources_sha256"],
        "angle_counts": angle_counts,
        "rare_ranges_1_99_percentiles": rare_ranges,
        "iqr_rule": {
            "multiplier": 1.5,
            "excluded_fields": [PATCH_ANGLE],
            "interpretation": "statistical candidates, not confirmed errors",
            "rows_removed": 0,
        },
        "top_absolute_pearson_pairs": _top_pairs(correlations, "pearson"),
        "top_absolute_spearman_pairs": _top_pairs(correlations, "spearman"),
        "scatter_coverage": scatter_registry,
        "normalization": "MinMaxScaler fitted on train for illustration only",
    }
    (reports_dir / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_eda_markdown(
        documentation_path,
        summary,
        outliers,
        angle_counts,
        summary_json["top_absolute_pearson_pairs"],
        summary_json["top_absolute_spearman_pairs"],
        scatter_registry,
    )
    return summary_json


def hashlib_indices(indices: list[int]) -> str:
    import hashlib

    payload = ",".join(str(value) for value in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()

"""Streamlit interface for the two direct-property predictions."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from composites.inference import InferenceError, ModelUnavailableError, load_direct_predictor
from composites.schema import (
    DENSITY,
    DIRECT_FEATURES,
    ELASTIC_MODULUS,
    EPOXY_GROUPS,
    FLASH_POINT,
    HARDENER_AMOUNT,
    MATRIX_FILLER_RATIO,
    PATCH_ANGLE,
    PATCH_DENSITY,
    PATCH_STEP,
    RESIN_CONSUMPTION,
    SURFACE_DENSITY,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("COMPOSITES_MODELS_DIR", PROJECT_ROOT / "models"))

FIELD_LABELS = {
    MATRIX_FILLER_RATIO: "Соотношение матрица–наполнитель (единица в исходном наборе не указана)",
    DENSITY: "Плотность, кг/м3",
    ELASTIC_MODULUS: "Модуль упругости, ГПа",
    HARDENER_AMOUNT: "Количество отвердителя, м.%",
    EPOXY_GROUPS: "Содержание эпоксидных групп, %",
    FLASH_POINT: "Температура вспышки, С",
    SURFACE_DENSITY: "Поверхностная плотность, г/м2",
    RESIN_CONSUMPTION: "Потребление смолы, г/м2",
    PATCH_STEP: "Шаг нашивки (единица в исходном наборе не указана)",
    PATCH_DENSITY: "Плотность нашивки (единица в исходном наборе не указана)",
}


st.set_page_config(page_title="Свойства композиционных материалов", layout="centered")
st.title("Прогноз свойств композиционного материала")
st.caption(
    "Учебная оценка по ограниченному набору наблюдений. Результат не заменяет "
    "испытания материала и не подтверждает качество для новых составов или технологий."
)

try:
    predictor = load_direct_predictor(MODELS_DIR)
except ModelUnavailableError as exc:
    st.info(str(exc))
    st.stop()
except InferenceError as exc:
    st.error(f"Manifest обученных моделей некорректен: {exc}")
    st.stop()

defaults = predictor.defaults()
with st.form("prediction_form"):
    values: dict[str, float] = {}
    for feature in DIRECT_FEATURES:
        if feature == PATCH_ANGLE:
            preferred = 90 if defaults[feature] == 90 else 0
            values[feature] = float(
                st.selectbox(
                    "Угол нашивки, град",
                    options=[0, 90],
                    index=1 if preferred == 90 else 0,
                )
            )
        else:
            values[feature] = st.number_input(
                FIELD_LABELS[feature],
                value=float(defaults[feature]),
                format="%.6f",
            )
    submitted = st.form_submit_button("Прогноз", type="primary")

if submitted:
    try:
        for warning in predictor.range_warnings(values):
            st.warning(warning)
        result = predictor.predict(values).iloc[0]
    except InferenceError as exc:
        st.error(f"Проверьте входные данные: {exc}")
    else:
        columns = st.columns(2)
        for column, task in zip(columns, predictor.tasks.values(), strict=True):
            with column:
                st.metric(
                    label=task.target,
                    value=f"{result[task.target]:.6g} {task.unit}",
                )
                metrics = task.test_metrics
                if all(metrics[name] is not None for name in ("rmse", "mae", "r2")):
                    st.caption(
                        "Отложенный тест: "
                        f"RMSE {metrics['rmse']:.4g} {task.unit}; "
                        f"MAE {metrics['mae']:.4g} {task.unit}; "
                        f"R² {metrics['r2']:.4g}."
                    )
                else:
                    st.caption("Метрики отложенного теста ещё не сохранены.")
                if task.is_baseline:
                    st.warning(
                        "Выбрана baseline-модель: прогноз постоянный. Данные не "
                        "подтвердили преимущество более сложной модели на кросс-валидации."
                    )

st.caption(
    "Предупреждение о выходе за обучающий диапазон означает экстраполяцию, "
    "а не нарушение установленной физической границы."
)

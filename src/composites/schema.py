"""Canonical source and modelling schemas."""

from __future__ import annotations

from dataclasses import dataclass

SOURCE_INDEX = "source_index"

MATRIX_FILLER_RATIO = "Соотношение матрица-наполнитель"
DENSITY = "Плотность, кг/м3"
ELASTIC_MODULUS = "модуль упругости, ГПа"
HARDENER_AMOUNT = "Количество отвердителя, м.%"
EPOXY_GROUPS = "Содержание эпоксидных групп,%_2"
FLASH_POINT = "Температура вспышки, С_2"
SURFACE_DENSITY = "Поверхностная плотность, г/м2"
TENSILE_MODULUS = "Модуль упругости при растяжении, ГПа"
TENSILE_STRENGTH = "Прочность при растяжении, МПа"
RESIN_CONSUMPTION = "Потребление смолы, г/м2"
PATCH_ANGLE = "Угол нашивки, град"
PATCH_STEP = "Шаг нашивки"
PATCH_DENSITY = "Плотность нашивки"

BP_COLUMNS = (
    MATRIX_FILLER_RATIO,
    DENSITY,
    ELASTIC_MODULUS,
    HARDENER_AMOUNT,
    EPOXY_GROUPS,
    FLASH_POINT,
    SURFACE_DENSITY,
    TENSILE_MODULUS,
    TENSILE_STRENGTH,
    RESIN_CONSUMPTION,
)

NUP_COLUMNS = (PATCH_ANGLE, PATCH_STEP, PATCH_DENSITY)
MERGED_COLUMNS = BP_COLUMNS + NUP_COLUMNS

DIRECT_TARGETS = (TENSILE_MODULUS, TENSILE_STRENGTH)
DIRECT_FEATURES = tuple(column for column in MERGED_COLUMNS if column not in DIRECT_TARGETS)
RATIO_FEATURES = tuple(column for column in MERGED_COLUMNS if column != MATRIX_FILLER_RATIO)


@dataclass(frozen=True)
class SourceSchema:
    filename: str
    sheet_name: str
    columns: tuple[str, ...]


BP_SCHEMA = SourceSchema("X_bp.xlsx", "X_bp.csv", BP_COLUMNS)
NUP_SCHEMA = SourceSchema("X_nup.xlsx", "X_nup.csv", NUP_COLUMNS)


def modelling_schemas() -> dict[str, dict[str, object]]:
    """Return the three task schemas without a source index."""
    return {
        "direct_tensile_modulus": {
            "target": TENSILE_MODULUS,
            "features": list(DIRECT_FEATURES),
            "feature_count": len(DIRECT_FEATURES),
        },
        "direct_tensile_strength": {
            "target": TENSILE_STRENGTH,
            "features": list(DIRECT_FEATURES),
            "feature_count": len(DIRECT_FEATURES),
        },
        "matrix_filler_ratio": {
            "target": MATRIX_FILLER_RATIO,
            "features": list(RATIO_FEATURES),
            "feature_count": len(RATIO_FEATURES),
        },
    }

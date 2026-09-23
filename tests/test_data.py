from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from composites.data import DataValidationError, audit_dataset, normalize_source_frame
from composites.schema import (
    BP_SCHEMA,
    DIRECT_FEATURES,
    DIRECT_TARGETS,
    MATRIX_FILLER_RATIO,
    RATIO_FEATURES,
    SOURCE_INDEX,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _valid_bp_frame() -> pd.DataFrame:
    values = {"raw_index": [0.0, 1.0]}
    values.update({column: [1.0, 2.0] for column in BP_SCHEMA.columns})
    return pd.DataFrame(values)


def test_real_sources_have_expected_inner_join_and_loss() -> None:
    report = audit_dataset(DATA_DIR)

    assert report["inner_join"]["rows"] == 1023
    assert report["inner_join"]["content_columns"] == 13
    assert report["inner_join"]["missing_indices"] == {
        "only_in_x_bp": [],
        "only_in_x_nup": list(range(1023, 1040)),
    }
    assert report["sources"]["X_bp.xlsx"]["sha256"] == (
        "14b6a82bdc7511c15a8e58397cf42855e9ffa51baa289b8253cec288dca812c8"
    )
    assert report["sources"]["X_nup.xlsx"]["sha256"] == (
        "1c7aa1be64596020f5621ca72f0d0de43d2dc689ca48ff2229bb393f1e08cb52"
    )


def test_integral_float_index_is_converted_without_loss() -> None:
    normalized = normalize_source_frame(_valid_bp_frame(), BP_SCHEMA)

    assert normalized[SOURCE_INDEX].dtype == np.dtype("int64")
    assert normalized[SOURCE_INDEX].tolist() == [0, 1]


@pytest.mark.parametrize("invalid_index", [["0", "1"], [False, True]])
def test_non_numeric_or_boolean_source_index_fails(invalid_index: list[object]) -> None:
    frame = _valid_bp_frame()
    frame["raw_index"] = invalid_index

    with pytest.raises(DataValidationError, match="numeric, non-boolean dtype"):
        normalize_source_frame(frame, BP_SCHEMA)


def test_duplicate_source_index_fails() -> None:
    frame = _valid_bp_frame()
    frame["raw_index"] = [1.0, 1.0]

    with pytest.raises(DataValidationError, match="duplicate source indices"):
        normalize_source_frame(frame, BP_SCHEMA)


def test_missing_required_column_fails() -> None:
    frame = _valid_bp_frame().drop(columns=[BP_SCHEMA.columns[-1]])

    with pytest.raises(DataValidationError, match="missing columns"):
        normalize_source_frame(frame, BP_SCHEMA)


def test_non_finite_value_fails() -> None:
    frame = _valid_bp_frame()
    frame.loc[0, BP_SCHEMA.columns[0]] = np.inf

    with pytest.raises(DataValidationError, match="non-finite"):
        normalize_source_frame(frame, BP_SCHEMA)


def test_model_schemas_exclude_index_and_targets() -> None:
    assert SOURCE_INDEX not in DIRECT_FEATURES
    assert SOURCE_INDEX not in RATIO_FEATURES
    assert len(DIRECT_FEATURES) == 11
    assert not set(DIRECT_TARGETS) & set(DIRECT_FEATURES)
    assert len(RATIO_FEATURES) == 12
    assert MATRIX_FILLER_RATIO not in RATIO_FEATURES

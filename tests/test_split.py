from __future__ import annotations

import copy
from pathlib import Path

import pytest

from composites.split import (
    SplitValidationError,
    create_split_manifest,
    validate_split_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def test_fixed_holdout_and_ten_fold_partition() -> None:
    first = create_split_manifest(DATA_DIR)
    second = create_split_manifest(DATA_DIR)

    assert first == second
    assert first["counts"] == {"all": 1023, "train": 716, "test": 307}
    assert len(first["cv_folds"]) == 10
    assert first["protocol"]["random_state"] == 42
    validate_split_manifest(first)


def test_validation_rejects_train_test_overlap() -> None:
    manifest = create_split_manifest(DATA_DIR)
    broken = copy.deepcopy(manifest)
    broken["test_indices"][0] = broken["train_indices"][0]

    with pytest.raises(SplitValidationError, match="overlap"):
        validate_split_manifest(broken)


def test_validation_rejects_test_row_inside_cv() -> None:
    manifest = create_split_manifest(DATA_DIR)
    broken = copy.deepcopy(manifest)
    broken["cv_folds"][0]["validation_indices"][0] = broken["test_indices"][0]

    with pytest.raises(SplitValidationError):
        validate_split_manifest(broken)

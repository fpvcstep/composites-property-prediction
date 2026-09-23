"""Reproducible holdout and cross-validation split protocol."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from composites.data import load_and_merge, sha256_file
from composites.schema import BP_SCHEMA, NUP_SCHEMA, SOURCE_INDEX

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 10


class SplitValidationError(ValueError):
    """Raised when a persisted split violates the evaluation contract."""


def create_split_manifest(data_dir: Path) -> dict[str, Any]:
    merged, _ = load_and_merge(data_dir)
    all_indices = np.sort(merged[SOURCE_INDEX].to_numpy(dtype=int))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)

    folds: list[dict[str, Any]] = []
    splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold_number, (fold_train, fold_validation) in enumerate(
        splitter.split(train_indices), start=1
    ):
        folds.append(
            {
                "fold": fold_number,
                "train_indices": train_indices[fold_train].tolist(),
                "validation_indices": train_indices[fold_validation].tolist(),
            }
        )

    manifest: dict[str, Any] = {
        "protocol": {
            "holdout_method": "train_test_split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cross_validation": {
                "method": "KFold",
                "n_splits": CV_FOLDS,
                "shuffle": True,
                "random_state": RANDOM_STATE,
            },
        },
        "raw_sources_sha256": {
            BP_SCHEMA.filename: sha256_file(data_dir / BP_SCHEMA.filename),
            NUP_SCHEMA.filename: sha256_file(data_dir / NUP_SCHEMA.filename),
        },
        "all_indices": all_indices.tolist(),
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "counts": {
            "all": int(all_indices.size),
            "train": int(train_indices.size),
            "test": int(test_indices.size),
        },
        "cv_folds": folds,
        "scope": {
            "shared_rows_for_all_three_tasks": True,
            "known_groups": False,
            "duplicate_observations": False,
            "test_is_untouched_until_final_evaluation": True,
        },
    }
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: dict[str, Any]) -> None:
    all_indices = set(manifest["all_indices"])
    train_indices = set(manifest["train_indices"])
    test_indices = set(manifest["test_indices"])

    if train_indices & test_indices:
        raise SplitValidationError("Train and test indices overlap")
    if train_indices | test_indices != all_indices:
        raise SplitValidationError("Train and test do not cover all merged rows")
    if len(train_indices) != len(manifest["train_indices"]):
        raise SplitValidationError("Train indices are not unique")
    if len(test_indices) != len(manifest["test_indices"]):
        raise SplitValidationError("Test indices are not unique")

    counts = manifest["counts"]
    if counts != {
        "all": len(all_indices),
        "train": len(train_indices),
        "test": len(test_indices),
    }:
        raise SplitValidationError("Persisted split counts are inconsistent")

    folds = manifest["cv_folds"]
    expected_folds = manifest["protocol"]["cross_validation"]["n_splits"]
    if len(folds) != expected_folds:
        raise SplitValidationError("Unexpected number of cross-validation folds")

    validation_counter: Counter[int] = Counter()
    for fold in folds:
        fold_train = set(fold["train_indices"])
        fold_validation = set(fold["validation_indices"])
        if fold_train & fold_validation:
            raise SplitValidationError(f"CV fold {fold['fold']} overlaps")
        if fold_train | fold_validation != train_indices:
            raise SplitValidationError(f"CV fold {fold['fold']} does not cover train")
        if not fold_validation.isdisjoint(test_indices):
            raise SplitValidationError(f"CV fold {fold['fold']} contains test rows")
        validation_counter.update(fold_validation)

    if set(validation_counter) != train_indices:
        raise SplitValidationError("CV validation folds do not cover all train rows")
    if set(validation_counter.values()) != {1}:
        raise SplitValidationError("A train row occurs in multiple validation folds")


def select_partition(frame: Any, indices: list[int]) -> Any:
    """Select rows by persisted source index and preserve manifest order."""
    indexed = frame.set_index(SOURCE_INDEX, drop=False)
    missing = sorted(set(indices) - set(indexed.index.tolist()))
    if missing:
        raise SplitValidationError(f"Partition indices absent from data: {missing}")
    return indexed.loc[indices].reset_index(drop=True)

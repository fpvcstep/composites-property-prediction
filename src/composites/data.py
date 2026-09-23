"""Strict loading, validation and audit of the source workbooks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from composites.schema import (
    BP_SCHEMA,
    MERGED_COLUMNS,
    NUP_SCHEMA,
    SOURCE_INDEX,
    SourceSchema,
    modelling_schemas,
)


class DataValidationError(ValueError):
    """Raised when a source file cannot satisfy the fixed data contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_source_frame(frame: pd.DataFrame, schema: SourceSchema) -> pd.DataFrame:
    """Validate one raw worksheet and normalize its integral source index."""
    if frame.shape[1] == 0:
        raise DataValidationError(f"{schema.filename}: worksheet is empty")

    index_column = frame.columns[0]
    content_columns = list(frame.columns[1:])
    missing = sorted(set(schema.columns) - set(content_columns))
    unexpected = sorted(set(content_columns) - set(schema.columns))
    if missing:
        raise DataValidationError(f"{schema.filename}: missing columns: {missing}")
    if unexpected:
        raise DataValidationError(f"{schema.filename}: unexpected columns: {unexpected}")
    if len(content_columns) != len(schema.columns):
        raise DataValidationError(f"{schema.filename}: duplicate or malformed headers")

    raw_index = frame.iloc[:, 0]
    if not pd.api.types.is_numeric_dtype(raw_index.dtype) or pd.api.types.is_bool_dtype(
        raw_index.dtype
    ):
        raise DataValidationError(
            f"{schema.filename}: source index must have a numeric, non-boolean dtype"
        )

    normalized = frame.rename(columns={index_column: SOURCE_INDEX}).copy()
    normalized = normalized.loc[:, [SOURCE_INDEX, *schema.columns]]
    if normalized.isna().any().any():
        columns = normalized.columns[normalized.isna().any()].tolist()
        raise DataValidationError(f"{schema.filename}: null values in columns: {columns}")

    try:
        normalized = normalized.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise DataValidationError(f"{schema.filename}: non-numeric value") from error

    values = normalized.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataValidationError(f"{schema.filename}: non-finite numeric values")

    index_values = normalized[SOURCE_INDEX].to_numpy(dtype=float)
    if not np.equal(index_values, np.floor(index_values)).all():
        raise DataValidationError(f"{schema.filename}: source index is not integral")
    normalized[SOURCE_INDEX] = index_values.astype("int64")
    if normalized[SOURCE_INDEX].duplicated().any():
        duplicates = normalized.loc[
            normalized[SOURCE_INDEX].duplicated(keep=False), SOURCE_INDEX
        ].unique()
        raise DataValidationError(
            f"{schema.filename}: duplicate source indices: {duplicates.tolist()}"
        )
    return normalized


def load_source(path: Path, schema: SourceSchema) -> pd.DataFrame:
    if not path.is_file():
        raise DataValidationError(f"Source file not found: {path}")
    try:
        frame = pd.read_excel(path, sheet_name=schema.sheet_name, engine="openpyxl")
    except ValueError as error:
        raise DataValidationError(
            f"{schema.filename}: missing worksheet {schema.sheet_name!r}"
        ) from error
    return normalize_source_frame(frame, schema)


def load_and_merge(data_dir: Path) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Load both sources and perform the required one-to-one INNER JOIN."""
    bp = load_source(data_dir / BP_SCHEMA.filename, BP_SCHEMA)
    nup = load_source(data_dir / NUP_SCHEMA.filename, NUP_SCHEMA)

    bp_indices = set(bp[SOURCE_INDEX].tolist())
    nup_indices = set(nup[SOURCE_INDEX].tolist())
    missing_indices = {
        "only_in_x_bp": sorted(bp_indices - nup_indices),
        "only_in_x_nup": sorted(nup_indices - bp_indices),
    }
    try:
        merged = bp.merge(
            nup,
            on=SOURCE_INDEX,
            how="inner",
            validate="one_to_one",
            sort=True,
        )
    except pd.errors.MergeError as error:
        raise DataValidationError("Sources do not form a one-to-one join") from error

    expected = [SOURCE_INDEX, *MERGED_COLUMNS]
    if merged.columns.tolist() != expected:
        raise DataValidationError("Merged columns do not match the canonical schema")
    return merged, missing_indices


def _frame_audit(frame: pd.DataFrame) -> dict[str, Any]:
    feature_frame = frame.drop(columns=[SOURCE_INDEX])
    return {
        "rows": int(len(frame)),
        "content_columns": int(feature_frame.shape[1]),
        "columns": feature_frame.columns.tolist(),
        "dtypes": {name: str(dtype) for name, dtype in feature_frame.dtypes.items()},
        "source_index": {
            "dtype": str(frame[SOURCE_INDEX].dtype),
            "minimum": int(frame[SOURCE_INDEX].min()),
            "maximum": int(frame[SOURCE_INDEX].max()),
            "unique": int(frame[SOURCE_INDEX].nunique()),
            "null": int(frame[SOURCE_INDEX].isna().sum()),
        },
        "missing_values": {
            name: int(count) for name, count in feature_frame.isna().sum().items()
        },
        "non_finite_values": {
            name: int(count)
            for name, count in zip(
                feature_frame.columns,
                (~np.isfinite(feature_frame.to_numpy(dtype=float))).sum(axis=0),
                strict=True,
            )
        },
        "duplicate_observations": int(feature_frame.duplicated().sum()),
    }


def audit_dataset(data_dir: Path) -> dict[str, Any]:
    """Return a deterministic, JSON-serializable audit report."""
    bp_path = data_dir / BP_SCHEMA.filename
    nup_path = data_dir / NUP_SCHEMA.filename
    bp = load_source(bp_path, BP_SCHEMA)
    nup = load_source(nup_path, NUP_SCHEMA)
    merged, missing_indices = load_and_merge(data_dir)

    report = {
        "sources": {
            BP_SCHEMA.filename: {
                "sheet": BP_SCHEMA.sheet_name,
                "sha256": sha256_file(bp_path),
                **_frame_audit(bp),
            },
            NUP_SCHEMA.filename: {
                "sheet": NUP_SCHEMA.sheet_name,
                "sha256": sha256_file(nup_path),
                **_frame_audit(nup),
            },
        },
        "inner_join": {
            **_frame_audit(merged),
            "validate": "one_to_one",
            "missing_indices": missing_indices,
        },
        "modelling_schemas": modelling_schemas(),
        "known_unknowns": {
            "observation_origin": "unknown",
            "experimental_groups": "unknown",
            "patch_step_unit": "unknown",
            "patch_density_unit": "unknown",
            "matrix_filler_ratio_basis": "unknown",
            "suffix_2_meaning": "unknown",
        },
    }
    return report

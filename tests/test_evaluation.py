from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from composites.evaluation import (
    ArtifactSpec,
    artifact_specs,
    load_verified_artifact,
    load_frozen_selection,
    prepare_partitions,
    regression_metrics,
)
from composites.inference import load_direct_predictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_PATH = PROJECT_ROOT / "data" / "splits.json"
SELECTION_PATH = PROJECT_ROOT / "reports" / "modeling" / "model_selection.json"


def test_frozen_selection_contains_all_twelve_artifacts() -> None:
    selection = load_frozen_selection(SELECTION_PATH, SPLIT_PATH)
    specs = artifact_specs(selection)

    assert len(specs) == 12
    assert sum(spec.selected for spec in specs) == 3
    assert all(spec.artifact.startswith("models/") for spec in specs)


def test_full_partitions_match_every_persisted_source_index() -> None:
    train, test, manifest = prepare_partitions(DATA_DIR, SPLIT_PATH)

    assert len(train) == 716
    assert len(test) == 307
    assert set(test.index) == set(manifest["test_indices"])
    assert set(train.index).isdisjoint(test.index)
    assert set(train.index) | set(test.index) == set(manifest["all_indices"])


def test_regression_metrics_stay_in_native_units() -> None:
    metrics = regression_metrics([1_000.0, 2_000.0], [900.0, 2_100.0])

    assert metrics["rmse"] == pytest.approx(100.0)
    assert metrics["mae"] == pytest.approx(100.0)
    assert np.isfinite(metrics["r2"])


def test_modified_frozen_artifact_is_rejected_before_deserialization(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    artifact = models / "model.joblib"
    original = b"frozen artifact bytes"
    artifact.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    artifact.write_bytes(original + b" altered")
    spec = ArtifactSpec("task", "family", "models/model.joblib", expected, True)

    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_artifact(tmp_path, spec)


def test_generated_predictions_cover_full_test_for_every_artifact() -> None:
    predictions = pd.read_csv(PROJECT_ROOT / "reports" / "evaluation" / "predictions.csv")
    test = predictions.loc[predictions["partition"] == "test"]
    groups = test.groupby(["task", "family"])

    assert groups.ngroups == 12
    assert set(groups.size()) == {307}
    assert groups["source_index"].nunique().eq(307).all()


def test_manifest_loads_real_direct_models_and_predicts() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "models" / "model_manifest.json").read_text("utf-8")
    )
    assert manifest["schema_version"] == 1
    assert set(manifest["tasks"]) == {
        "direct_tensile_modulus",
        "direct_tensile_strength",
        "matrix_filler_ratio",
    }
    assert all(
        not Path(item["artifact"]).is_absolute()
        and len(Path(item["artifact"]).parts) == 1
        for item in manifest["tasks"].values()
    )
    assert all(
        item["artifact_sha256"]
        == hashlib.sha256(
            (PROJECT_ROOT / "models" / item["artifact"]).read_bytes()
        ).hexdigest()
        for item in manifest["tasks"].values()
    )

    predictor = load_direct_predictor(PROJECT_ROOT / "models")
    frame = pd.DataFrame([predictor.defaults()])
    result = predictor.predict(frame)
    assert result.shape == (1, 2)
    assert np.isfinite(result.to_numpy()).all()

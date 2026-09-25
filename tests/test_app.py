from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from composites.inference import DIRECT_TASKS
from composites.schema import DIRECT_FEATURES, PATCH_ANGLE


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _fixture_models(models_dir: Path) -> None:
    models_dir.mkdir()
    frame = pd.DataFrame(
        {
            feature: ([0.0, 90.0] if feature == PATCH_ANGLE else [1.0, 3.0])
            for feature in DIRECT_FEATURES
        }
    )
    ranges = {
        feature: {"min": float(frame[feature].min()), "median": float(frame[feature].median()), "max": float(frame[feature].max())}
        for feature in DIRECT_FEATURES
    }
    tasks = {}
    for index, (task_name, (target, unit)) in enumerate(DIRECT_TASKS.items(), start=1):
        artifact = f"{task_name}.joblib"
        pipeline = Pipeline(
            [("regressor", DummyRegressor(strategy="constant", constant=index * 10.0))]
        ).fit(frame, np.array([1.0, 2.0]))
        joblib.dump(pipeline, models_dir / artifact)
        tasks[task_name] = {
            "target": target,
            "unit": unit,
            "features": list(DIRECT_FEATURES),
            "model_family": "baseline DummyRegressor",
            "artifact": artifact,
            "artifact_sha256": hashlib.sha256(
                (models_dir / artifact).read_bytes()
            ).hexdigest(),
            "cv_rmse": 1.0,
            "baseline_cv_rmse": 1.0,
            "test_metrics": {"rmse": 1.1, "mae": 0.9, "r2": 0.0},
            "train_feature_ranges": ranges,
        }
    (models_dir / "model_manifest.json").write_text(
        json.dumps({"schema_version": 1, "tasks": tasks}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_streamlit_app_predicts_with_tmp_fixture(tmp_path: Path, monkeypatch) -> None:
    models_dir = tmp_path / "models"
    _fixture_models(models_dir)
    monkeypatch.setenv("COMPOSITES_MODELS_DIR", str(models_dir))

    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not app.exception
    assert len(app.number_input) == 10
    assert len(app.selectbox) == 1

    app.button[0].click().run()

    assert not app.exception
    assert len(app.metric) == 2
    assert "ГПа" in app.metric[0].value
    assert "МПа" in app.metric[1].value
    assert sum("прогноз постоянный" in warning.value for warning in app.warning) == 2

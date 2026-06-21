"""
Shared pytest fixtures and setup.

The API tests (`test_api.py`) need trained model artifacts to exist.
Rather than silently failing or requiring manual setup, this conftest
trains the model once per test session if artifacts are missing — so
`pytest tests/ -v` works correctly on a fresh clone with zero manual steps.
"""

from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(scope="session", autouse=True)
def ensure_model_artifacts_exist():
    """Train the model once before the test session if artifacts are missing."""
    model_path = MODEL_DIR / "logistic_regression.joblib"
    pipeline_path = MODEL_DIR / "feature_pipeline.joblib"

    if not model_path.exists() or not pipeline_path.exists():
        from src.models.train import run_training_pipeline

        run_training_pipeline()

    yield

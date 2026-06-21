"""
Integration tests for the FastAPI serving layer.

Requires model artifacts to exist (run `python -m src.models.train` first)
— these are integration tests against the real trained model, not mocks,
so a passing suite means the actual served predictions are sane.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

SAMPLE_TRANSACTION = {
    "Time": 50000,
    "Amount": 149.62,
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
    "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
    "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
    "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
    "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
    "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
    "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_valid_probability():
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud_prediction"], bool)
    assert "explanation_text" in body
    assert len(body["top_features"]) <= 5


def test_predict_missing_required_field_returns_422():
    bad_input = {k: v for k, v in SAMPLE_TRANSACTION.items() if k != "Amount"}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_negative_amount_rejected():
    bad_input = {**SAMPLE_TRANSACTION, "Amount": -50.0}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_predict_batch():
    response = client.post("/predict/batch", json=[SAMPLE_TRANSACTION, SAMPLE_TRANSACTION])
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2


def test_predict_batch_size_limit():
    too_many = [SAMPLE_TRANSACTION] * 501
    response = client.post("/predict/batch", json=too_many)
    assert response.status_code == 400


def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "LogisticRegression"
    assert body["feature_count"] > 0


def test_monitoring_drift_endpoint():
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    body = response.json()
    assert body["overall_severity"] in ("none", "moderate", "significant")
    assert "score_drift" in body

"""
FastAPI serving layer for the fraud detection model.

Endpoints:
  GET  /health           -> liveness check
  GET  /model/info        -> which model/version is loaded, threshold, metrics
  POST /predict            -> score a single transaction, with SHAP explanation
  POST /predict/batch      -> score multiple transactions
  GET  /monitoring/drift   -> run drift detection vs the training baseline

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger UI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.explain import FraudExplainer, format_explanation_text
from src.monitoring.drift import detect_feature_drift, detect_score_drift

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Interpretable fraud detection with per-prediction explanations "
        "and drift monitoring. Built as a full MLOps demonstration project."
    ),
    version="1.0.0",
)

# --- Globals populated at startup ---
_model = None
_pipeline = None
_explainer = None
_train_baseline: Optional[pd.DataFrame] = None
_decision_threshold = 0.5


class TransactionInput(BaseModel):
    """
    Matches the Kaggle creditcard.csv schema: Time, V1..V28, Amount.
    All V-features default to 0.0 so the demo works even with a
    partially-filled request from the live demo frontend.
    """

    Time: float = Field(..., description="Seconds since the first transaction in the dataset")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 50000,
                "Amount": 149.62,
                "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
            }
        }


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud_prediction: bool
    decision_threshold: float
    explanation_text: str
    top_features: list[dict]


class ModelInfo(BaseModel):
    model_type: str
    decision_threshold: float
    feature_count: int
    training_data_rows: Optional[int] = None


@app.on_event("startup")
def load_artifacts():
    """Load model, feature pipeline, and explainer once at startup."""
    global _model, _pipeline, _explainer, _train_baseline, _decision_threshold

    model_path = MODEL_DIR / "logistic_regression.joblib"
    pipeline_path = MODEL_DIR / "feature_pipeline.joblib"

    if not model_path.exists() or not pipeline_path.exists():
        logger.error(
            "Model artifacts not found at %s. Run `python -m src.models.train` first.",
            MODEL_DIR,
        )
        return

    _model = joblib.load(model_path)
    _pipeline = joblib.load(pipeline_path)

    baseline_path = DATA_DIR / "train_baseline.csv"
    if baseline_path.exists():
        _train_baseline = pd.read_csv(baseline_path)
        background = _pipeline.transform(_train_baseline)
        _explainer = FraudExplainer(_model, _pipeline.feature_names(), background)

    logger.info("Model artifacts loaded successfully.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    return ModelInfo(
        model_type=type(_model).__name__,
        decision_threshold=_decision_threshold,
        feature_count=len(_pipeline.feature_names()),
        training_data_rows=len(_train_baseline) if _train_baseline is not None else None,
    )


def _predict_one(txn: TransactionInput) -> PredictionResponse:
    if _model is None or _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    row_df = pd.DataFrame([txn.dict()])
    X = _pipeline.transform(row_df)

    proba = float(_model.predict_proba(X)[0, 1])
    is_fraud = proba >= _decision_threshold

    if _explainer is not None:
        exp = _explainer.explain(X, index=0)
        explanation_text = format_explanation_text(exp)
        top_features = [{"feature": f, "contribution": round(v, 4)} for f, v in exp.top_features]
    else:
        explanation_text = "Explainer not available (training baseline missing)."
        top_features = []

    return PredictionResponse(
        fraud_probability=round(proba, 6),
        is_fraud_prediction=is_fraud,
        decision_threshold=_decision_threshold,
        explanation_text=explanation_text,
        top_features=top_features,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    return _predict_one(transaction)


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(transactions: list[TransactionInput]):
    if len(transactions) > 500:
        raise HTTPException(status_code=400, detail="Batch size limited to 500 per request.")
    return [_predict_one(t) for t in transactions]


@app.get("/monitoring/drift")
def monitoring_drift():
    """
    Compare the held-out test stream (simulating recent production traffic)
    against the training baseline, and report drift severity.
    """
    if _train_baseline is None:
        raise HTTPException(status_code=503, detail="Training baseline not available.")

    test_stream_path = MODEL_DIR / "test_stream.csv"
    if not test_stream_path.exists():
        raise HTTPException(status_code=503, detail="No production stream data available yet.")

    current_df = pd.read_csv(test_stream_path)
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    report = detect_feature_drift(_train_baseline, current_df, feature_names)

    X_baseline = _pipeline.transform(_train_baseline)
    X_current = _pipeline.transform(current_df)
    baseline_scores = _model.predict_proba(X_baseline)[:, 1]
    current_scores = _model.predict_proba(X_current)[:, 1]
    score_drift = detect_score_drift(baseline_scores, current_scores)

    return {
        "overall_severity": report.overall_severity,
        "n_drifted_features": report.n_drifted_features,
        "n_significant_features": report.n_significant_features,
        "score_drift": {
            "psi": round(score_drift.psi, 4),
            "ks_pvalue": round(score_drift.ks_pvalue, 6),
            "severity": score_drift.severity,
        },
        "top_drifted_features": report.to_dataframe().head(5).to_dict(orient="records"),
        "note": (
            "Population-level drift can miss shifts confined to the rare "
            "fraud class. See /monitoring/drift/conditional for class-aware "
            "drift, which requires confirmed labels."
        ),
    }

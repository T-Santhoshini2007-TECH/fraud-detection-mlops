"""
Model training for fraud detection.

Model choice: Logistic Regression (class_weight='balanced') as the
primary model. This is a deliberate choice, not a limitation:

  - Coefficients are directly interpretable (sign + magnitude per feature).
  - SHAP values on a linear model are exact and cheap to compute, and map
    cleanly onto "why was this transaction flagged" explanations a fraud
    analyst could actually use.
  - With ~0.17% fraud, a complex model's extra capacity buys little without
    far more data; the bottleneck here is class imbalance handling and
    threshold selection, not model capacity.

A small gradient-boosted tree (sklearn's HistGradientBoostingClassifier)
is trained alongside as a performance comparison point — included to
demonstrate the interpretability/performance tradeoff explicitly, not to
quietly replace the primary model.

Imbalance handling: class_weight='balanced' rather than naive resampling,
so probability calibration is preserved (important since we report
fraud probability, not just a label, in the API).

Metric choice: accuracy is reported but explicitly flagged as meaningless
here (predicting "not fraud" always scores ~99.8%). Precision, recall,
F1, PR-AUC, and a cost-weighted business metric are the real metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.load_data import load_dataset, time_ordered_split
from src.features.pipeline import FeaturePipeline, split_X_y

logger = logging.getLogger(__name__)

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when mlflow isn't installed
    MLFLOW_AVAILABLE = False
    logger.warning(
        "mlflow is not installed — experiment tracking will be skipped. "
        "Install it with `pip install mlflow` to enable run tracking "
        "(see requirements.txt)."
    )

    class _NoOpMlflow:
        """Drop-in no-op so the pipeline still runs without mlflow installed."""

        class _NoOpRun:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def set_experiment(self, *a, **k):
            pass

        def start_run(self, *a, **k):
            return self._NoOpRun()

        def log_param(self, *a, **k):
            pass

        def log_metric(self, *a, **k):
            pass

        def log_artifact(self, *a, **k):
            pass

    mlflow = _NoOpMlflow()

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Business cost assumptions, used for cost-weighted evaluation.
# These are illustrative defaults — documented and easy to override —
# not a claim about real-world fraud economics.
COST_FALSE_NEGATIVE = 100.0  # missed fraud: avg fraud loss assumed
COST_FALSE_POSITIVE = 5.0    # false alarm: cost of manual review / friction


@dataclass
class EvalResult:
    model_name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    accuracy_is_misleading: float  # reported explicitly to make the point
    business_cost: float
    confusion: np.ndarray
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        tn, fp, fn, tp = self.confusion.ravel()
        return (
            f"[{self.model_name}] thr={self.threshold:.3f} "
            f"precision={self.precision:.3f} recall={self.recall:.3f} f1={self.f1:.3f} "
            f"roc_auc={self.roc_auc:.3f} pr_auc={self.pr_auc:.3f} "
            f"| naive_accuracy={self.accuracy_is_misleading:.4f} (misleading, see README) "
            f"| business_cost=${self.business_cost:,.0f} "
            f"| TP={tp} FP={fp} FN={fn} TN={tn}"
        )


def _select_threshold_by_cost(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Pick the decision threshold that minimizes total business cost,
    rather than defaulting to 0.5 (which is close to meaningless when
    the positive class is 0.17% of the data).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    best_thr, best_cost = 0.5, float("inf")

    for thr in np.linspace(0.01, 0.99, 99):
        preds = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
        if cost < best_cost:
            best_cost, best_thr = cost, thr

    return best_thr


def evaluate(
    model_name: str, y_true: np.ndarray, y_proba: np.ndarray, threshold: float | None = None
) -> EvalResult:
    if threshold is None:
        threshold = _select_threshold_by_cost(y_true, y_proba)

    preds = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE

    return EvalResult(
        model_name=model_name,
        threshold=threshold,
        precision=precision_score(y_true, preds, zero_division=0),
        recall=recall_score(y_true, preds, zero_division=0),
        f1=f1_score(y_true, preds, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_proba),
        pr_auc=average_precision_score(y_true, y_proba),
        accuracy_is_misleading=(preds == y_true).mean(),
        business_cost=cost,
        confusion=confusion_matrix(y_true, preds, labels=[0, 1]),
    )


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_gbt_comparison(X_train, y_train) -> HistGradientBoostingClassifier:
    """Trained purely as a comparison point against the interpretable model."""
    model = HistGradientBoostingClassifier(
        max_iter=200,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def run_training_pipeline(force_synthetic: bool = False) -> dict:
    """
    Full training pipeline: load -> split -> featurize -> train both
    models -> evaluate -> log everything to MLflow -> persist artifacts.

    Returns a dict of results for both models, keyed by model name.
    """
    mlflow.set_experiment("fraud-detection")

    logger.info("Loading and splitting dataset...")
    df = load_dataset(force_synthetic=force_synthetic)
    train_df, val_df, test_df = time_ordered_split(df)

    pipeline = FeaturePipeline()
    X_train = pipeline.fit_transform(train_df)
    y_train = train_df["Class"].values

    X_val = pipeline.transform(val_df)
    y_val = val_df["Class"].values

    X_test = pipeline.transform(test_df)
    y_test = test_df["Class"].values

    results = {}

    with mlflow.start_run(run_name="logistic_regression"):
        logger.info("Training logistic regression (primary, interpretable model)...")
        lr_model = train_logistic_regression(X_train, y_train)
        lr_val_proba = lr_model.predict_proba(X_val)[:, 1]
        threshold = _select_threshold_by_cost(y_val, lr_val_proba)

        lr_test_proba = lr_model.predict_proba(X_test)[:, 1]
        lr_eval = evaluate("logistic_regression", y_test, lr_test_proba, threshold)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("decision_threshold", lr_eval.threshold)
        mlflow.log_metric("precision", lr_eval.precision)
        mlflow.log_metric("recall", lr_eval.recall)
        mlflow.log_metric("f1", lr_eval.f1)
        mlflow.log_metric("roc_auc", lr_eval.roc_auc)
        mlflow.log_metric("pr_auc", lr_eval.pr_auc)
        mlflow.log_metric("business_cost", lr_eval.business_cost)

        joblib.dump(lr_model, MODEL_DIR / "logistic_regression.joblib")
        joblib.dump(pipeline, MODEL_DIR / "feature_pipeline.joblib")
        mlflow.log_artifact(str(MODEL_DIR / "logistic_regression.joblib"))

        logger.info(lr_eval.summary())
        results["logistic_regression"] = lr_eval

    with mlflow.start_run(run_name="gbt_comparison"):
        logger.info("Training gradient-boosted trees (comparison model)...")
        gbt_model = train_gbt_comparison(X_train, y_train)
        gbt_val_proba = gbt_model.predict_proba(X_val)[:, 1]
        threshold = _select_threshold_by_cost(y_val, gbt_val_proba)

        gbt_test_proba = gbt_model.predict_proba(X_test)[:, 1]
        gbt_eval = evaluate("gbt_comparison", y_test, gbt_test_proba, threshold)

        mlflow.log_param("model_type", "HistGradientBoostingClassifier")
        mlflow.log_param("decision_threshold", gbt_eval.threshold)
        mlflow.log_metric("precision", gbt_eval.precision)
        mlflow.log_metric("recall", gbt_eval.recall)
        mlflow.log_metric("f1", gbt_eval.f1)
        mlflow.log_metric("roc_auc", gbt_eval.roc_auc)
        mlflow.log_metric("pr_auc", gbt_eval.pr_auc)
        mlflow.log_metric("business_cost", gbt_eval.business_cost)

        joblib.dump(gbt_model, MODEL_DIR / "gbt_comparison.joblib")

        logger.info(gbt_eval.summary())
        results["gbt_comparison"] = gbt_eval

    # Save the held-out test set for the monitoring/drift module to use
    # as a "fresh production stream" simulation.
    test_df.to_csv(MODEL_DIR / "test_stream.csv", index=False)
    train_df.to_csv(MODEL_DIR.parent / "data" / "processed" / "train_baseline.csv", index=False)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_training_pipeline()
    print("\n=== Final Comparison ===")
    for name, res in results.items():
        print(res.summary())

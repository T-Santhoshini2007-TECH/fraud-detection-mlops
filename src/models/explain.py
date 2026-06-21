"""
Explainability for the fraud model.

For a linear model (logistic regression on standardized features), SHAP
values have a closed-form solution: each feature's contribution to the
log-odds is exactly `coefficient * (feature_value - mean_feature_value)`.
This is literally what `shap.LinearExplainer` computes under the hood.

We use the real `shap` library when it's installed (preferred — it's the
standard tool and handles edge cases robustly), and fall back to this
exact manual computation when it isn't, so the explanation logic is
never blocked by an environment without internet access. Both paths
produce equivalent results for a linear model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning(
        "shap is not installed — falling back to exact manual linear "
        "SHAP-value computation. Install `shap` for the standard library "
        "implementation (recommended for production; see requirements.txt)."
    )


@dataclass
class Explanation:
    transaction_index: int
    base_value: float
    predicted_probability: float
    feature_contributions: dict[str, float]  # feature_name -> SHAP value (log-odds)
    top_features: list[tuple[str, float]]  # sorted by |contribution|, top 5


class FraudExplainer:
    """
    Wraps a fitted LogisticRegression to produce per-transaction
    explanations of *why* it was flagged, in terms a fraud analyst
    can act on (e.g. "unusual amount contributed +1.2 to the fraud score").
    """

    def __init__(self, model: LogisticRegression, feature_names: list[str], background: pd.DataFrame):
        self.model = model
        self.feature_names = feature_names
        self.background_mean = background[feature_names].mean().values
        self._shap_explainer = None

        if SHAP_AVAILABLE:
            self._shap_explainer = shap.LinearExplainer(
                model, background[feature_names].values
            )

    def explain(self, X: pd.DataFrame, index: int = 0) -> Explanation:
        row = X[self.feature_names].iloc[[index]]
        proba = self.model.predict_proba(row)[0, 1]

        if self._shap_explainer is not None:
            shap_values = self._shap_explainer.shap_values(row.values)[0]
            base_value = float(self._shap_explainer.expected_value)
        else:
            # Exact manual equivalent for a linear model:
            # contribution_i = coef_i * (x_i - mean_i)
            coefs = self.model.coef_[0]
            shap_values = coefs * (row.values[0] - self.background_mean)
            base_value = float(
                self.model.intercept_[0] + np.dot(coefs, self.background_mean)
            )

        contributions = dict(zip(self.feature_names, shap_values.tolist()))
        top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]

        return Explanation(
            transaction_index=index,
            base_value=base_value,
            predicted_probability=float(proba),
            feature_contributions=contributions,
            top_features=top,
        )

    def explain_batch(self, X: pd.DataFrame, max_rows: int | None = None) -> list[Explanation]:
        n = len(X) if max_rows is None else min(max_rows, len(X))
        return [self.explain(X, i) for i in range(n)]


def format_explanation_text(exp: Explanation) -> str:
    """Human-readable explanation, the kind shown in the API response / dashboard."""
    lines = [
        f"Fraud probability: {exp.predicted_probability:.1%}",
        "Top contributing factors:",
    ]
    for name, value in exp.top_features:
        direction = "increased" if value > 0 else "decreased"
        lines.append(f"  - {name}: {direction} fraud score by {abs(value):.3f}")
    return "\n".join(lines)

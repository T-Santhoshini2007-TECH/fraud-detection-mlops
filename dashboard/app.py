"""
Monitoring dashboard for the fraud detection system.

Run locally:
    streamlit run dashboard/app.py

Shows:
  - Model performance summary (precision/recall/F1/cost, NOT misleading accuracy)
  - Live transaction scoring with SHAP-style explanation
  - Drift monitoring: population-level, score-level, and class-conditional
    (with an explicit explanation of why these can disagree on a
    severely imbalanced problem — this is the project's key insight)
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.explain import FraudExplainer, format_explanation_text
from src.monitoring.drift import (
    detect_conditional_drift,
    detect_feature_drift,
    detect_score_drift,
    should_trigger_retrain,
)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

st.set_page_config(page_title="Fraud Detection Monitor", layout="wide", page_icon="🔍")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "logistic_regression.joblib")
    pipeline = joblib.load(MODEL_DIR / "feature_pipeline.joblib")
    train_baseline = pd.read_csv(DATA_DIR / "train_baseline.csv")
    test_stream = pd.read_csv(MODEL_DIR / "test_stream.csv")
    background = pipeline.transform(train_baseline)
    explainer = FraudExplainer(model, pipeline.feature_names(), background)
    return model, pipeline, explainer, train_baseline, test_stream


st.title("🔍 Fraud Detection — Monitoring Dashboard")
st.caption(
    "Interpretable fraud detection with full MLOps monitoring. "
    "Built to demonstrate the complete lifecycle: training → serving → drift detection."
)

try:
    model, pipeline, explainer, train_baseline, test_stream = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run `python -m src.models.train` first "
        "to generate the model, then reload this dashboard."
    )
    st.stop()

tab_overview, tab_live, tab_drift = st.tabs(
    ["📊 Model Overview", "🧾 Score a Transaction", "📈 Drift Monitoring"]
)

# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Why accuracy is the wrong metric here")
    fraud_rate = train_baseline["Class"].mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Training set size", f"{len(train_baseline):,}")
    col2.metric("Fraud rate", f"{fraud_rate:.3%}")
    col3.metric(
        "Accuracy from predicting 'never fraud'",
        f"{(1 - fraud_rate):.3%}",
        help="This is why precision/recall/F1/business-cost are reported instead of accuracy.",
    )

    st.markdown(
        "A model that **always predicts 'not fraud'** would score "
        f"**{(1 - fraud_rate):.2%} accuracy** while catching zero fraud. "
        "This dashboard and the underlying training pipeline report "
        "precision, recall, F1, ROC-AUC, PR-AUC, and a cost-weighted "
        "business metric instead."
    )

    X_test = pipeline.transform(test_stream)
    y_test = test_stream["Class"].values
    proba_test = model.predict_proba(X_test)[:, 1]

    fig = px.histogram(
        x=proba_test, color=y_test.astype(str),
        nbins=50, log_y=True,
        labels={"x": "Predicted fraud probability", "color": "Actual class"},
        title="Predicted fraud probability distribution (test set, log scale)",
        color_discrete_map={"0": "#4C78A8", "1": "#E45756"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_live:
    st.subheader("Score a transaction live")
    st.caption("Pick a real example from the held-out test set, or enter custom values.")

    mode = st.radio("Input mode", ["Sample from test set", "Custom input"], horizontal=True)

    if mode == "Sample from test set":
        sample_type = st.selectbox("Sample type", ["Random", "Known fraud", "Known legitimate"])
        if sample_type == "Known fraud":
            pool = test_stream[test_stream["Class"] == 1]
        elif sample_type == "Known legitimate":
            pool = test_stream[test_stream["Class"] == 0]
        else:
            pool = test_stream

        if st.button("Draw sample"):
            row = pool.sample(1)
            st.session_state["selected_row"] = row

        row = st.session_state.get("selected_row", pool.sample(1))
        st.dataframe(row, use_container_width=True)
        input_df = row.drop(columns=["Class"], errors="ignore")
        actual_label = row["Class"].iloc[0] if "Class" in row.columns else None
    else:
        amount = st.number_input("Amount", min_value=0.0, value=149.62)
        time_val = st.number_input("Time (seconds since start)", min_value=0.0, value=50000.0)
        cols = st.columns(4)
        v_values = {}
        for i in range(1, 29):
            with cols[(i - 1) % 4]:
                v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v_{i}")
        input_df = pd.DataFrame([{**{"Time": time_val, "Amount": amount}, **v_values}])
        actual_label = None

    if st.button("🔍 Score this transaction", type="primary"):
        X = pipeline.transform(input_df)
        proba = float(model.predict_proba(X)[0, 1])
        exp = explainer.explain(X, index=0)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Fraud probability", f"{proba:.2%}")
            if actual_label is not None:
                st.metric("Actual label", "Fraud" if actual_label == 1 else "Legitimate")
            verdict = "🚨 FLAG AS FRAUD" if proba >= 0.33 else "✅ Looks legitimate"
            st.markdown(f"### {verdict}")

        with col2:
            contrib_df = pd.DataFrame(exp.top_features, columns=["feature", "contribution"])
            fig = go.Figure(
                go.Bar(
                    x=contrib_df["contribution"],
                    y=contrib_df["feature"],
                    orientation="h",
                    marker_color=["#E45756" if v > 0 else "#4C78A8" for v in contrib_df["contribution"]],
                )
            )
            fig.update_layout(title="Top feature contributions (SHAP-style, log-odds)", height=300)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_drift:
    st.subheader("Drift monitoring: three complementary views")
    st.markdown(
        "This is the **core insight of this project**: on a severely "
        "imbalanced problem (~0.2% fraud), different drift detection "
        "approaches can disagree — and that disagreement is informative, "
        "not a bug."
    )

    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    with st.spinner("Computing drift reports..."):
        pop_report = detect_feature_drift(train_baseline, test_stream, feature_names)
        cond_report = detect_conditional_drift(train_baseline, test_stream, feature_names)

        X_train = pipeline.transform(train_baseline)
        baseline_scores = model.predict_proba(X_train)[:, 1]
        score_report = detect_score_drift(baseline_scores, proba_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("Population-level drift", pop_report.overall_severity.upper())
    col2.metric("Model score drift", score_report.severity.upper())
    col3.metric("Fraud-conditional drift", cond_report.overall_severity.upper())

    if cond_report.overall_severity == "significant" and pop_report.overall_severity == "none":
        st.warning(
            "⚠️ **Population-level drift shows nothing, but fraud-conditional "
            "drift is significant.** This means fraud PATTERNS have shifted "
            "even though the overall transaction population looks stable — "
            "exactly the blind spot that makes rare-event monitoring hard. "
            "A monitoring system that only checks population-level drift "
            "would miss this entirely."
        )

    st.markdown("#### Fraud-conditional feature drift (requires confirmed labels)")
    st.dataframe(cond_report.to_dataframe(), use_container_width=True)

    st.markdown(f"**Retrain recommended:** {'✅ YES' if should_trigger_retrain(cond_report) else '❌ No'}")

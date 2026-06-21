import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    detect_feature_drift,
    detect_score_drift,
    population_stability_index,
    should_trigger_retrain,
)


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 5000)
    psi = population_stability_index(data, data.copy())
    assert psi < 1e-6


def test_psi_high_for_shifted_distributions():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 5000)
    shifted = rng.normal(5, 1, 5000)  # large shift
    psi = population_stability_index(baseline, shifted)
    assert psi > 0.25  # should clearly register as "significant"


def test_psi_moderate_for_small_shift():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 5000)
    slightly_shifted = rng.normal(0.3, 1, 5000)
    psi = population_stability_index(baseline, slightly_shifted)
    assert 0.0 < psi < 0.25


def test_detect_feature_drift_no_drift():
    rng = np.random.default_rng(0)
    baseline_df = pd.DataFrame({"f1": rng.normal(0, 1, 2000), "f2": rng.normal(0, 1, 2000)})
    current_df = pd.DataFrame({"f1": rng.normal(0, 1, 2000), "f2": rng.normal(0, 1, 2000)})

    report = detect_feature_drift(baseline_df, current_df, ["f1", "f2"])
    assert report.overall_severity == "none"
    assert not should_trigger_retrain(report)


def test_detect_feature_drift_with_drift():
    rng = np.random.default_rng(0)
    baseline_df = pd.DataFrame({
        "f1": rng.normal(0, 1, 2000), "f2": rng.normal(0, 1, 2000), "f3": rng.normal(0, 1, 2000)
    })
    current_df = pd.DataFrame({
        "f1": rng.normal(5, 1, 2000), "f2": rng.normal(5, 1, 2000), "f3": rng.normal(5, 1, 2000)
    })

    report = detect_feature_drift(baseline_df, current_df, ["f1", "f2", "f3"])
    assert report.overall_severity == "significant"
    assert should_trigger_retrain(report)


def test_detect_score_drift():
    rng = np.random.default_rng(0)
    baseline_scores = rng.beta(1, 20, 5000)  # mostly low probabilities
    current_scores_same = rng.beta(1, 20, 5000)
    current_scores_shifted = rng.beta(5, 5, 5000)  # much higher probabilities on average

    stable = detect_score_drift(baseline_scores, current_scores_same)
    drifted = detect_score_drift(baseline_scores, current_scores_shifted)

    assert stable.psi < drifted.psi


def test_missing_features_are_skipped_gracefully():
    df1 = pd.DataFrame({"f1": [1, 2, 3]})
    df2 = pd.DataFrame({"f1": [1, 2, 3]})
    # "f2" doesn't exist in either — should not raise
    report = detect_feature_drift(df1, df2, ["f1", "f2"])
    assert len(report.feature_results) == 1

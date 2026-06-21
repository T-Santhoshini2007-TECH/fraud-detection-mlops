"""
Drift detection for the fraud model.

Two complementary, industry-standard techniques:

1. Population Stability Index (PSI) — measures how much a feature's
   distribution has shifted between a baseline window and a current
   window, by comparing binned histograms. Rule-of-thumb thresholds
   (used widely in credit risk / fraud industry):
       PSI < 0.1            -> no significant drift
       0.1 <= PSI < 0.25     -> moderate drift, worth watching
       PSI >= 0.25           -> significant drift, action needed

2. Kolmogorov-Smirnov (KS) test — a nonparametric statistical test for
   whether two samples come from the same distribution. Gives a p-value:
   p < 0.05 suggests the distributions are meaningfully different.

We also track PERFORMANCE drift directly: if ground-truth labels become
available with a delay (the realistic case — fraud is usually confirmed
days/weeks later via chargebacks), recall/precision on recent windows
can be compared against the training-time baseline.

This module is what makes the "monitoring dashboard" in this project
real rather than decorative — it's measuring the exact phenomenon the
synthetic data generator injects (fraud patterns shifting over time).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.25
KS_PVALUE_THRESHOLD = 0.05


def population_stability_index(
    baseline: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """
    Compute PSI between a baseline distribution and a current distribution
    of the same feature. Bins are defined by baseline quantiles, so the
    baseline is uniform-by-construction across bins (standard approach).
    """
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(baseline, quantiles))
    if len(bin_edges) < 3:
        # Degenerate feature (near-constant) — treat as stable.
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    baseline_pct = np.maximum(baseline_counts / max(len(baseline), 1), 1e-6)
    current_pct = np.maximum(current_counts / max(len(current), 1), 1e-6)

    psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))
    return psi


@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    severity: str  # "none" | "moderate" | "significant"


@dataclass
class DriftReport:
    feature_results: list[FeatureDriftResult]
    n_drifted_features: int
    n_significant_features: int
    overall_severity: str
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "feature": r.feature,
                    "psi": r.psi,
                    "ks_statistic": r.ks_statistic,
                    "ks_pvalue": r.ks_pvalue,
                    "severity": r.severity,
                }
                for r in self.feature_results
            ]
        ).sort_values("psi", ascending=False)


def detect_feature_drift(
    baseline_df: pd.DataFrame, current_df: pd.DataFrame, feature_names: list[str]
) -> DriftReport:
    """
    Compare a baseline window (e.g. the training set) against a current
    window (e.g. the most recent N transactions in production) across
    all given features, and return a structured drift report.
    """
    results = []
    for feature in feature_names:
        if feature not in baseline_df.columns or feature not in current_df.columns:
            continue

        baseline_vals = baseline_df[feature].values.astype(float)
        current_vals = current_df[feature].values.astype(float)

        psi = population_stability_index(baseline_vals, current_vals)
        ks_stat, ks_pvalue = ks_2samp(baseline_vals, current_vals)

        if psi >= PSI_SIGNIFICANT_THRESHOLD:
            severity = "significant"
        elif psi >= PSI_MODERATE_THRESHOLD:
            severity = "moderate"
        else:
            severity = "none"

        results.append(
            FeatureDriftResult(
                feature=feature,
                psi=psi,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_pvalue),
                severity=severity,
            )
        )

    n_drifted = sum(1 for r in results if r.severity != "none")
    n_significant = sum(1 for r in results if r.severity == "significant")

    if n_significant >= 3:
        overall = "significant"
    elif n_drifted >= 3:
        overall = "moderate"
    else:
        overall = "none"

    logger.info(
        "Drift report: %d/%d features drifted (%d significant) -> overall=%s",
        n_drifted, len(results), n_significant, overall,
    )

    return DriftReport(
        feature_results=results,
        n_drifted_features=n_drifted,
        n_significant_features=n_significant,
        overall_severity=overall,
    )


def should_trigger_retrain(report: DriftReport) -> bool:
    """
    Decision rule used by the CI/CD retraining pipeline: retrain if
    overall drift severity is significant. Kept as an explicit, named
    function (rather than inlined) so the retraining policy is a single
    visible, testable decision point.
    """
    return report.overall_severity == "significant"


def detect_score_drift(
    baseline_scores: np.ndarray, current_scores: np.ndarray
) -> FeatureDriftResult:
    """
    Drift on the MODEL'S PREDICTED PROBABILITIES, rather than raw input
    features.

    This matters for exactly the failure mode population-level feature
    drift misses: in a severely imbalanced problem like fraud (~0.2% positive),
    a real shift confined to the rare class can be statistically invisible in
    whole-population feature distributions (99.8% of rows are unaffected legit
    transactions, which swamps the PSI/KS signal). The model's output score
    distribution is far more sensitive to exactly the cases that matter,
    because it's a learned projection that concentrates on the
    decision-relevant signal.

    In production this is usually the FIRST line of defense, with
    per-feature drift used to diagnose *why* once score drift fires.
    """
    psi = population_stability_index(baseline_scores, current_scores)
    ks_stat, ks_pvalue = ks_2samp(baseline_scores, current_scores)

    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        severity = "significant"
    elif psi >= PSI_MODERATE_THRESHOLD:
        severity = "moderate"
    else:
        severity = "none"

    return FeatureDriftResult(
        feature="model_score",
        psi=psi,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pvalue),
        severity=severity,
    )


def detect_conditional_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_names: list[str],
    label_col: str = "Class",
    target_class: int = 1,
) -> DriftReport:
    """
    Drift detection conditioned on class label — i.e. "how has fraud
    itself changed", not "how has the overall population changed".

    Requires ground-truth labels, which in real fraud systems arrive on
    a delay (confirmed via chargeback/dispute, typically days to weeks
    later). This is the metric that actually catches "fraud patterns
    evolved" once labels catch up — faster signals (score drift above)
    are what you monitor in the meantime.
    """
    baseline_class = baseline_df[baseline_df[label_col] == target_class]
    current_class = current_df[current_df[label_col] == target_class]

    if len(baseline_class) < 10 or len(current_class) < 10:
        logger.warning(
            "Too few class=%d examples for reliable conditional drift "
            "(baseline=%d, current=%d) — results may be noisy.",
            target_class, len(baseline_class), len(current_class),
        )

    return detect_feature_drift(baseline_class, current_class, feature_names)

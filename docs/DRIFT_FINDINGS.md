# Drift Detection Findings: Why Population-Level Monitoring Fails on Rare Events

## The setup

The synthetic data generator (`src/data/load_data.py::_generate_synthetic_dataset`)
injects a deliberate concept drift: fraud transactions in the second half of
the time window are generated from a shifted distribution on a different set
of features than fraud in the first half — simulating fraudsters changing
their behavior, which is exactly what happens in real fraud over time.

This document reports what each of the three drift detection methods in
`src/monitoring/drift.py` actually found when run against that injected
drift, using the time-ordered train/test split (train = early period,
test = late period, where the drift was injected).

## Method 1: Population-level feature drift

Compares the full feature distributions (`detect_feature_drift`) between
the training baseline and the test stream — i.e., every transaction,
fraud and legitimate together.

**Result: no significant drift detected.** Every individual feature's PSI
stayed below 0.01 (the "no drift" threshold is 0.1).

**Why:** fraud is ~0.25% of the data in this setup. A shift confined to
125 out of 50,000 rows is statistically swamped by the 99.75% of rows
(legitimate transactions) that didn't change at all. PSI and KS-test are
population-level statistics — they cannot see a signal that small.

## Method 2: Model score drift

Compares the distribution of the model's predicted fraud probabilities
(`detect_score_drift`) between baseline and current data.

**Result: also no significant drift** (PSI ≈ 0.006). Same root cause —
the model's score distribution is dominated by the 99.75% of transactions
whose scores didn't move.

**Why this is still useful in production despite missing this case:** score
drift is usually the *fastest* signal available (no labels needed, just
new transactions scored by the existing model), so it's still worth running
continuously. It's good at catching things like "the overall transaction
mix changed" (e.g., a new merchant category goes live). It's just not
sensitive enough alone for rare-class concept drift.

## Method 3: Fraud-conditional drift

Compares feature distributions **only among confirmed-fraud rows**
(`detect_conditional_drift`) between baseline and current data.

**Result: drift detected clearly and consistently.**

| Feature | PSI | KS statistic | p-value | Severity |
|---|---|---|---|---|
| V6 | 10.87 | 0.835 | 2.5e-14 | significant |
| V1 | 10.63 | 0.861 | 2.2e-15 | significant |
| V17 | 10.63 | 0.846 | 1.3e-14 | significant |
| V7 | 10.50 | 0.823 | 7.9e-14 | significant |
| V3 | 9.25 | 0.770 | 8.4e-12 | significant |

(PSI > 0.25 is the "significant" threshold — these are 30-40x over it.)

**Why this works:** by conditioning on the label, we remove the 99.75% of
unaffected data that was drowning out the signal in the first two methods.
What's left is a clean comparison of "fraud now" vs. "fraud before" — and
that comparison shows the shift unambiguously.

**The catch:** this requires confirmed labels, which in real fraud systems
arrive on a delay (chargebacks/disputes are typically confirmed days to
weeks after the transaction). So this is a *lagging* but *high-confidence*
signal — used to confirm and quantify drift that score-drift monitoring
might have only weakly hinted at.

## The practical takeaway

A production fraud monitoring system should run **all three** in layers:

1. **Score drift** (real-time, no labels needed) — early warning, low
   confidence, catches population-level shifts.
2. **Population feature drift** (real-time, no labels needed) — diagnostic,
   helps explain *what* changed when score drift fires.
3. **Conditional drift** (delayed, requires labels) — the ground-truth
   confirmation that fraud patterns specifically have shifted, used to
   justify retraining decisions with high confidence.

This is also why `should_trigger_retrain()` in this repo is applied to the
conditional drift report in the CI workflow
(`.github/workflows/drift-monitor.yml`) rather than the population-level
one — using the population-level result would have produced a false
"everything is fine" signal despite the model's fraud-detection logic
having genuinely gone stale.

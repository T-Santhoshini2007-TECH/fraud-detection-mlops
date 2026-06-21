# Model Card: Fraud Detection Logistic Regression

## Model details
- **Type:** Logistic Regression, `class_weight='balanced'`
- **Inputs:** 30 features — `hour_of_day_scaled`, `amount_scaled`, and 28
  anonymized PCA components (`V1`-`V28`) from the source dataset
- **Output:** Fraud probability (0-1), thresholded at a cost-optimized
  decision point (selected on the validation set, see `src/models/train.py`)
- **Comparison model:** `HistGradientBoostingClassifier`, included to show
  the interpretability/performance tradeoff explicitly — not the deployed model

## Intended use
A portfolio/demonstration project showing a complete fraud-detection MLOps
pipeline: training, tracking, serving, explainability, and drift monitoring.

**Not intended for:** real financial decision-making. The training data is
either a 2013 academic dataset (European cardholders, two days of
transactions) or a synthetic fallback — neither represents current,
broad-population transaction patterns.

## Why logistic regression, not a more powerful model

This project explicitly prioritizes **interpretability** over raw predictive
performance:
- Every prediction comes with an exact, mathematically-grounded explanation
  (SHAP values for a linear model are an exact closed-form computation, not
  an approximation) — meaningful for a fraud analyst deciding whether to
  act on a flag.
- The cost-weighted evaluation in this repo shows the simpler model
  achieving a *better* business outcome (lower total cost) than the more
  complex comparison model on this dataset — see the README for numbers —
  which is a useful, real reminder that "more complex" isn't automatically
  "better" for a given business objective.

## Training data
- **Real option:** [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/ulb-machine-learning-group/creditcardfraud) —
  284,807 transactions by European cardholders over two days in September
  2013, 492 fraudulent (0.172%). Features V1-V28 are the result of PCA, so
  their real-world meaning is not disclosed (this is also why interpretability
  here is in terms of "feature V6 contributed X", not a named real-world
  concept — a genuine limitation of this specific dataset).
- **Synthetic fallback:** generated when the real file isn't present;
  matches the schema and severe imbalance, with deliberately injected
  concept drift for demonstrating the monitoring system. See
  `src/data/load_data.py` for the exact generation logic.

## Evaluation
See the README's "Model performance" section and `docs/DRIFT_FINDINGS.md`
for the drift-specific evaluation. Headline metrics use precision, recall,
F1, ROC-AUC, PR-AUC, and a cost-weighted business metric — explicitly NOT
accuracy, which is structurally misleading at this class imbalance (see
README for the exact math).

## Known limitations
- PCA-anonymized features limit real-world interpretability of explanations
  beyond "feature V6" — a fraud analyst using a real system would want
  named, business-meaningful features.
- The two-day data collection window (real dataset) cannot capture
  longer-term seasonal fraud patterns (e.g. holiday shopping spikes).
- Cost assumptions used for threshold selection (`COST_FALSE_NEGATIVE`,
  `COST_FALSE_POSITIVE`) are illustrative defaults, not empirically derived.
- No fairness/bias audit was performed — the anonymized features make this
  difficult to do meaningfully on this specific dataset, but it would be
  a required step before any real deployment.

## Ethical considerations
Automated fraud flagging carries real consequences for legitimate
customers (false positives → blocked transactions, friction, potential
account closure). This project's cost-weighted evaluation approach is one
way to make that tradeoff explicit and tunable rather than hidden inside
a single "accuracy" number — but the specific cost values used here are
demonstration defaults and should be replaced with your organization's
actual, carefully-considered cost estimates before any real use.

"""
Data loading for the fraud detection system.

Primary dataset: Kaggle "Credit Card Fraud Detection"
https://www.kaggle.com/datasets/mlg-ulb/ulb-machine-learning-group/creditcardfraud

That dataset contains 284,807 European card transactions over two days,
with 28 PCA-anonymized features (V1-V28), plus `Time` (seconds since first
transaction) and `Amount`. Only 492 (~0.17%) are fraud.

If `data/raw/creditcard.csv` is not present, this module generates a
synthetic dataset with the SAME schema and similar statistical properties
(severe class imbalance, time-ordered, drifting fraud patterns) so the
rest of the pipeline is runnable without the real file. Swap in the real
CSV at any time — nothing downstream needs to change.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "creditcard.csv"
N_FEATURES = 28  # V1..V28, matches the real Kaggle schema


def _generate_synthetic_dataset(
    n_rows: int = 50_000,
    fraud_rate: float = 0.0025,
    n_days: float = 2.0,
    drift_strength: float = 1.4,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset matching the real schema:
    Time, V1..V28, Amount, Class.

    Critically, this injects CONCEPT DRIFT: fraud patterns in the second
    half of the time window are generated from a shifted distribution,
    simulating fraudsters adapting their behavior. This is what makes
    the monitoring/drift-detection part of the system meaningful to
    demo, rather than something with nothing to detect.
    """
    rng = np.random.default_rng(random_state)

    n_fraud = max(int(n_rows * fraud_rate), 20)
    n_legit = n_rows - n_fraud

    total_seconds = n_days * 24 * 3600

    # --- Legit transactions: stable distribution across the whole window ---
    legit_time = rng.uniform(0, total_seconds, size=n_legit)
    legit_features = rng.normal(loc=0.0, scale=1.0, size=(n_legit, N_FEATURES))
    legit_amount = np.round(rng.gamma(shape=1.5, scale=40.0, size=n_legit), 2)

    # --- Fraud transactions: split into "early" and "late" pattern regimes ---
    fraud_time = np.sort(rng.uniform(0, total_seconds, size=n_fraud))
    half = total_seconds / 2
    is_late = fraud_time > half

    fraud_features = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, N_FEATURES))
    # Early fraud: shifted on a few key components (a classic, "old" pattern)
    fraud_features[~is_late, 0:3] += rng.normal(3.0, 0.5, size=(np.sum(~is_late), 3))
    fraud_features[~is_late, 10:12] -= rng.normal(2.0, 0.5, size=(np.sum(~is_late), 2))

    # Late fraud: DIFFERENT shifted components, scaled by drift_strength —
    # this is the "fraud pattern evolved" signal the drift monitor should catch
    n_late = int(np.sum(is_late))
    fraud_features[is_late, 5:8] += rng.normal(
        3.0 * drift_strength, 0.5, size=(n_late, 3)
    )
    fraud_features[is_late, 15:17] -= rng.normal(
        2.0 * drift_strength, 0.5, size=(n_late, 2)
    )

    fraud_amount = np.round(rng.gamma(shape=1.1, scale=120.0, size=n_fraud), 2)

    # --- Assemble ---
    cols = ["Time"] + [f"V{i}" for i in range(1, N_FEATURES + 1)] + ["Amount", "Class"]

    legit_df = pd.DataFrame(
        np.column_stack(
            [legit_time, legit_features, legit_amount, np.zeros(n_legit)]
        ),
        columns=cols,
    )
    fraud_df = pd.DataFrame(
        np.column_stack([fraud_time, fraud_features, fraud_amount, np.ones(n_fraud)]),
        columns=cols,
    )

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sort_values("Time").reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)

    logger.info(
        "Generated synthetic dataset: %d rows, %d fraud (%.3f%%)",
        len(df),
        df["Class"].sum(),
        100 * df["Class"].mean(),
    )
    return df


def load_dataset(path: Path | None = None, force_synthetic: bool = False) -> pd.DataFrame:
    """
    Load the transaction dataset.

    Tries `data/raw/creditcard.csv` first (the real Kaggle file you should
    place there). Falls back to a synthetic dataset with the same schema
    if the file isn't found, so the pipeline always runs.
    """
    target = path or RAW_DATA_PATH

    if not force_synthetic and target.exists():
        logger.info("Loading real dataset from %s", target)
        df = pd.read_csv(target)
        return df

    logger.warning(
        "Real dataset not found at %s — generating synthetic data with the "
        "same schema. Download the real Kaggle dataset and place it there "
        "for real results: "
        "https://www.kaggle.com/datasets/mlg-ulb/ulb-machine-learning-group/creditcardfraud",
        target,
    )
    return _generate_synthetic_dataset()


def time_ordered_split(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by TIME, not randomly.

    This matters: fraud detection systems are deployed once and score
    transactions that happen AFTER training. A random split leaks future
    distribution into training and makes drift invisible. Splitting by
    time simulates the real deployment scenario:

        train  -> what the model learns from
        val    -> used for threshold tuning / early stopping
        test   -> held-out "future" data the model has never seen,
                  also used as the baseline window for drift detection
    """
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)

    logger.info(
        "Time-ordered split: train=%d val=%d test=%d (fraud rates: %.4f / %.4f / %.4f)",
        len(train), len(val), len(test),
        train["Class"].mean(), val["Class"].mean(), test["Class"].mean(),
    )
    return train, val, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_dataset()
    train, val, test = time_ordered_split(df)
    print(df.describe())

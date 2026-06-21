"""
Feature pipeline for fraud detection.

Design choice: we keep features close to the raw, interpretable signal
(scaled amount, scaled time-of-day, the anonymized V-features as-is)
rather than engineering exotic features. This is deliberate — the project
prioritizes INTERPRETABILITY (SHAP explanations a fraud analyst can act
on) over squeezing out marginal AUC with opaque feature crosses.

Leakage discipline:
- The scaler is fit ONLY on the training set, then applied to val/test.
  Fitting on the full dataset before splitting is a common, serious bug
  (it leaks test-set statistics into training) — this pipeline avoids it
  by construction (`fit` vs `transform` are separate, explicit steps).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS_V = [f"V{i}" for i in range(1, 29)]
TARGET_COLUMN = "Class"


@dataclass
class FeaturePipeline:
    """Fit on train, transform train/val/test/inference consistently."""

    amount_scaler: StandardScaler = None
    time_scaler: StandardScaler = None
    fitted: bool = False

    def __post_init__(self):
        self.amount_scaler = StandardScaler()
        self.time_scaler = StandardScaler()

    def _seconds_to_hour_of_day(self, time_seconds: pd.Series) -> pd.Series:
        """Convert raw seconds-since-start into hour-of-day (0-23).

        Fraud and legitimate spending both have strong daily cycles;
        this is a more useful signal than raw elapsed seconds, and is
        still trivially interpretable in a SHAP plot ("transaction at 3am").
        """
        return (time_seconds % 86400) // 3600

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        hour = self._seconds_to_hour_of_day(train_df["Time"]).values.reshape(-1, 1)
        amount = train_df["Amount"].values.reshape(-1, 1)

        self.time_scaler.fit(hour)
        self.amount_scaler.fit(amount)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeaturePipeline.fit() must be called before transform().")

        out = pd.DataFrame(index=df.index)

        hour = self._seconds_to_hour_of_day(df["Time"]).values.reshape(-1, 1)
        out["hour_of_day_scaled"] = self.time_scaler.transform(hour).ravel()

        amount = df["Amount"].values.reshape(-1, 1)
        out["amount_scaled"] = self.amount_scaler.transform(amount).ravel()

        # Anonymized PCA features pass through unchanged — they're already
        # roughly standardized in the source dataset.
        for col in FEATURE_COLUMNS_V:
            if col in df.columns:
                out[col] = df[col].values

        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    def feature_names(self) -> list[str]:
        return ["hour_of_day_scaled", "amount_scaled"] + FEATURE_COLUMNS_V


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a raw dataframe into features-only and target array."""
    y = df[TARGET_COLUMN].values if TARGET_COLUMN in df.columns else None
    X = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    return X, y

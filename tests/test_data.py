import numpy as np
import pandas as pd
import pytest

from src.data.load_data import _generate_synthetic_dataset, load_dataset, time_ordered_split


def test_synthetic_dataset_schema():
    df = _generate_synthetic_dataset(n_rows=5000, random_state=1)
    expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    assert list(df.columns) == expected_cols


def test_synthetic_dataset_class_imbalance():
    """The whole point of this dataset is severe imbalance — verify it."""
    df = _generate_synthetic_dataset(n_rows=10_000, fraud_rate=0.0025, random_state=1)
    fraud_rate = df["Class"].mean()
    assert 0.0 < fraud_rate < 0.02, "Fraud rate should be a small minority class"


def test_synthetic_dataset_no_nulls():
    df = _generate_synthetic_dataset(n_rows=2000, random_state=1)
    assert df.isnull().sum().sum() == 0


def test_load_dataset_falls_back_to_synthetic(tmp_path):
    """If no real CSV is present, loading should not crash — it should
    fall back to synthetic data with the same schema."""
    fake_path = tmp_path / "does_not_exist.csv"
    df = load_dataset(path=fake_path)
    assert len(df) > 0
    assert "Class" in df.columns


def test_time_ordered_split_is_chronological():
    df = _generate_synthetic_dataset(n_rows=5000, random_state=1)
    train, val, test = time_ordered_split(df)

    # No row in train should have a later Time than any row in test
    assert train["Time"].max() <= val["Time"].min()
    assert val["Time"].max() <= test["Time"].min()


def test_time_ordered_split_no_row_loss():
    df = _generate_synthetic_dataset(n_rows=5000, random_state=1)
    train, val, test = time_ordered_split(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_time_ordered_split_proportions():
    df = _generate_synthetic_dataset(n_rows=10_000, random_state=1)
    train, val, test = time_ordered_split(df, train_frac=0.6, val_frac=0.2)
    assert abs(len(train) / len(df) - 0.6) < 0.01
    assert abs(len(val) / len(df) - 0.2) < 0.01

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import _generate_synthetic_dataset, time_ordered_split
from src.features.pipeline import FeaturePipeline


@pytest.fixture
def split_data():
    df = _generate_synthetic_dataset(n_rows=5000, random_state=1)
    return time_ordered_split(df)


def test_pipeline_fit_transform_shapes(split_data):
    train, val, test = split_data
    pipeline = FeaturePipeline()
    X_train = pipeline.fit_transform(train)

    assert len(X_train) == len(train)
    assert "amount_scaled" in X_train.columns
    assert "hour_of_day_scaled" in X_train.columns
    assert "V1" in X_train.columns


def test_pipeline_raises_if_not_fitted(split_data):
    train, val, test = split_data
    pipeline = FeaturePipeline()
    with pytest.raises(RuntimeError):
        pipeline.transform(val)


def test_no_leakage_scaler_fit_only_on_train(split_data):
    """
    Critical correctness test: the scaler's learned mean/scale must come
    ONLY from the training set, not from val/test or the full dataset.
    This directly tests the leakage-prevention design decision.
    """
    train, val, test = split_data
    pipeline = FeaturePipeline()
    pipeline.fit(train)

    # The scaler's mean_ should match the training set's amount mean,
    # not the combined train+val+test mean.
    train_amount_mean = train["Amount"].mean()
    learned_mean = pipeline.amount_scaler.mean_[0]

    assert abs(train_amount_mean - learned_mean) < 1e-6

    combined_mean = pd.concat([train, val, test])["Amount"].mean()
    # These should differ (unless by coincidence) — sanity check that
    # we're not accidentally fitting on everything.
    if abs(train_amount_mean - combined_mean) > 1e-3:
        assert abs(learned_mean - combined_mean) > 1e-6


def test_transform_is_deterministic(split_data):
    train, val, test = split_data
    pipeline = FeaturePipeline()
    pipeline.fit(train)

    out1 = pipeline.transform(val)
    out2 = pipeline.transform(val)
    pd.testing.assert_frame_equal(out1, out2)


def test_hour_of_day_wraps_correctly():
    pipeline = FeaturePipeline()
    df = pd.DataFrame({
        "Time": [0, 3600, 86400, 90000],  # 0h, 1h, next-day 0h, next-day 1h
        "Amount": [10, 10, 10, 10],
        "Class": [0, 0, 0, 0],
    })
    pipeline.fit(df)
    hour = pipeline._seconds_to_hour_of_day(df["Time"])
    assert list(hour) == [0, 1, 0, 1]

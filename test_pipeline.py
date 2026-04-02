"""
StockSense AI — Unit Tests
Run: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.features import engineer_features, extract_feature_row, FEATURE_NAMES
from model.train import generate_ohlcv, THRESHOLD_PCT, FORWARD_DAYS


# ─── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def sample_ohlcv():
    return generate_ohlcv(n_days=300, seed=99)


@pytest.fixture(scope="module")
def featured_df(sample_ohlcv):
    return engineer_features(sample_ohlcv.copy())


# ─── Data Generation ──────────────────────────────────────────────────────────
class TestDataGeneration:
    def test_shape(self, sample_ohlcv):
        assert sample_ohlcv.shape[0] == 300

    def test_columns(self, sample_ohlcv):
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in sample_ohlcv.columns

    def test_no_negative_prices(self, sample_ohlcv):
        for col in ["open", "high", "low", "close"]:
            assert (sample_ohlcv[col] > 0).all()

    def test_high_gte_low(self, sample_ohlcv):
        assert (sample_ohlcv["high"] >= sample_ohlcv["low"]).all()

    def test_reproducibility(self):
        df1 = generate_ohlcv(seed=42)
        df2 = generate_ohlcv(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        df1 = generate_ohlcv(seed=1)
        df2 = generate_ohlcv(seed=2)
        assert not df1["close"].equals(df2["close"])


# ─── Feature Engineering ──────────────────────────────────────────────────────
class TestFeatureEngineering:
    def test_all_features_present(self, featured_df):
        for feat in FEATURE_NAMES:
            assert feat in featured_df.columns, f"Missing feature: {feat}"

    def test_rsi_range(self, featured_df):
        rsi = featured_df["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_bb_pct_mostly_in_range(self, featured_df):
        # Bollinger %B can go outside [0,1] during extreme moves, but mostly bounded
        bb = featured_df["bb_pct"].dropna()
        within = ((bb >= -0.5) & (bb <= 1.5)).mean()
        assert within > 0.95

    def test_no_all_nan_features(self, featured_df):
        for feat in FEATURE_NAMES:
            non_null = featured_df[feat].notna().sum()
            assert non_null > 100, f"{feat} has too many NaNs"

    def test_extract_feature_row_shape(self, sample_ohlcv):
        row = extract_feature_row(sample_ohlcv)
        assert row.shape == (1, len(FEATURE_NAMES))

    def test_extract_feature_row_no_nan(self, sample_ohlcv):
        row = extract_feature_row(sample_ohlcv)
        assert not row.isnull().any().any()

    def test_volume_features_positive(self, featured_df):
        vol_ma = featured_df["vol_ma_ratio"].dropna()
        assert (vol_ma >= 0).all()


# ─── Target Label ─────────────────────────────────────────────────────────────
class TestTargetLabel:
    def test_target_binary(self, featured_df):
        target = featured_df["target"].dropna()
        assert set(target.unique()).issubset({0, 1})

    def test_target_not_all_same(self, featured_df):
        target = featured_df["target"].dropna()
        assert target.nunique() == 2

    def test_target_rate_reasonable(self, featured_df):
        """Signal rate should be between 10% and 90%."""
        rate = featured_df["target"].dropna().mean()
        assert 0.10 <= rate <= 0.90, f"Unrealistic signal rate: {rate:.1%}"


# ─── Constants ────────────────────────────────────────────────────────────────
class TestConstants:
    def test_forward_days_positive(self):
        assert FORWARD_DAYS > 0

    def test_threshold_pct_positive(self):
        assert THRESHOLD_PCT > 0

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 27

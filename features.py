"""
StockSense AI — Feature Engineering Utilities
Shared between training pipeline and API inference layer.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast  = series.ewm(span=fast,   adjust=False).mean()
    ema_slow  = series.ewm(span=slow,   adjust=False).mean()
    macd      = ema_fast - ema_slow
    macd_sig  = macd.ewm(span=signal,   adjust=False).mean()
    macd_hist = macd - macd_sig
    return macd, macd_sig, macd_hist


def compute_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    sma      = series.rolling(period).mean()
    std      = series.rolling(period).std()
    upper    = sma + num_std * std
    lower    = sma - num_std * std
    bb_pct   = (series - lower) / (upper - lower + 1e-9)
    bb_width = (upper - lower) / (sma + 1e-9)
    return bb_pct, bb_width


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame with columns [open, high, low, close, volume]
    Output : Same DataFrame with all feature columns added.
    The last row's features are used for live prediction.
    """
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # Trend
    df["ema_9"]          = c.ewm(span=9,  adjust=False).mean()
    df["ema_21"]         = c.ewm(span=21, adjust=False).mean()
    df["ema_cross"]      = df["ema_9"] - df["ema_21"]
    df["sma_50"]         = c.rolling(50).mean()
    df["price_vs_sma50"] = c / (df["sma_50"] + 1e-9)

    # Momentum
    df["rsi_14"]         = compute_rsi(c, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(c)
    df["momentum_10"]    = c / c.shift(10) - 1
    df["momentum_20"]    = c / c.shift(20) - 1
    df["roc_5"]          = c.pct_change(5)
    df["roc_10"]         = c.pct_change(10)

    # Volatility
    df["bb_pct"], df["bb_width"] = compute_bollinger(c)
    df["atr_14"]         = compute_atr(h, l, c)
    df["volatility_10"]  = c.pct_change().rolling(10).std()
    df["volatility_20"]  = c.pct_change().rolling(20).std()
    df["high_low_pct"]   = (h - l) / (c + 1e-9)
    df["close_open_pct"] = (c - df["open"]) / (df["open"] + 1e-9)

    # Volume
    df["vol_change"]     = v.pct_change()
    df["vol_ma_ratio"]   = v / (v.rolling(20).mean() + 1e-9)

    # Lag returns
    ret = c.pct_change()
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_ret_{lag}"] = ret.shift(lag)

    return df


FEATURE_NAMES = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width", "atr_14",
    "ema_9", "ema_21", "ema_cross", "sma_50", "price_vs_sma50",
    "momentum_10", "momentum_20", "roc_5", "roc_10",
    "vol_change", "vol_ma_ratio",
    "volatility_10", "volatility_20",
    "high_low_pct", "close_open_pct",
    "lag_ret_1", "lag_ret_2", "lag_ret_3", "lag_ret_5", "lag_ret_10",
]


def extract_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    """Return a single-row DataFrame of features from the last row of OHLCV data."""
    df = engineer_features(df.copy())
    row = df[FEATURE_NAMES].iloc[[-1]].reset_index(drop=True)
    return row

"""
Feature engineering — 11 deterministic technical features.
Does NOT drop NaN rows — the pipeline handles alignment.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


FEATURE_COLS = [
    "ret_1d", "ret_2d", "ret_3d",
    "vol_5d", "vol_20d", "atr_14",
    "hl_range", "co", "upper_wick", "lower_wick",
    "vol_z20",
]


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range over `window` days."""
    close_prev = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - close_prev).abs()
    tr3 = (df["low"] - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def build_features(
    df: pd.DataFrame,
    return_lags: tuple = (1, 2, 3),
    vol_windows: tuple = (5, 20),
    atr_window: int = 14,
    vol_z_window: int = 20,
) -> pd.DataFrame:
    """
    Build 11 features from OHLCV DataFrame.
    Returns DataFrame with feature columns added (NaN in warmup rows).
    Does NOT drop NaN — caller handles alignment.
    """
    out = df.copy()

    # 1. Lagged returns
    for lag in return_lags:
        out[f"ret_{lag}d"] = out["close"].pct_change(lag)

    # 2. Realized volatility
    for w in vol_windows:
        out[f"vol_{w}d"] = out["ret_1d"].rolling(window=w).std()

    # 3. ATR
    out["atr_14"] = compute_atr(out, window=atr_window)

    # 4. OHLC ratios
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    out["co"] = (out["close"] - out["open"]) / out["open"]
    body_high = pd.concat([out["open"], out["close"]], axis=1).max(axis=1)
    body_low = pd.concat([out["open"], out["close"]], axis=1).min(axis=1)
    out["upper_wick"] = (out["high"] - body_high) / out["close"]
    out["lower_wick"] = (body_low - out["low"]) / out["close"]

    # 5. Volume z-score
    log_vol = np.log(out["volume"] + 1)
    vol_mean = log_vol.rolling(window=vol_z_window).mean()
    vol_std = log_vol.rolling(window=vol_z_window).std()
    out["vol_z20"] = (log_vol - vol_mean) / (vol_std + 1e-8)

    return out


def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Extract feature columns from a DataFrame that already has features built."""
    return df[FEATURE_COLS], FEATURE_COLS

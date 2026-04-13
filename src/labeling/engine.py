"""
Labeling engine — binary directional and ternary volatility-adaptive labels.

Forward return: r_t(H) = Close(t+H) / Open(t+1) - 1
  Entry at next-day open, exit at horizon close.

Binary: y = 1 if r > 0 else 0
Ternary: y = +1 if r > +θσ, -1 if r < -θσ, 0 otherwise
  Remapped to {0, 1, 2} for classification: bearish=-1→0, neutral=0→1, bullish=+1→2
"""

import numpy as np
import pandas as pd


def compute_forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    r_t(H) = Close(t+H) / Open(t+1) - 1
    NaN for the last H+1 rows (no future data available).
    """
    close_future = df["close"].shift(-horizon)
    open_next = df["open"].shift(-1)
    fwd_ret = close_future / open_next - 1.0
    return fwd_ret


def make_labels(
    df: pd.DataFrame,
    horizon: int,
    mode: str = "multiclass_volatility",
    threshold_mult: float = 0.5,
    vol_window: int = 20,
) -> pd.Series:
    """
    Generate labels for the given DataFrame.

    Args:
        df: OHLCV DataFrame (must have 'close' and 'open' columns).
        horizon: Forward-looking horizon in trading days.
        mode: 'directional_binary' or 'multiclass_volatility'.
        threshold_mult: Multiplier for volatility threshold (ternary only).
        vol_window: Window for rolling volatility (ternary only).

    Returns:
        pd.Series of integer labels. NaN where labels cannot be computed.
        Binary: {0, 1}
        Ternary: {0=bearish, 1=neutral, 2=bullish}
    """
    fwd_ret = compute_forward_return(df, horizon)

    if mode == "directional_binary":
        labels = (fwd_ret > 0).astype(float)
        labels[fwd_ret.isna()] = np.nan
        return labels

    elif mode == "multiclass_volatility":
        # Rolling volatility of daily log returns
        daily_ret = df["close"].pct_change()
        roll_vol = daily_ret.rolling(window=vol_window).std()
        threshold = threshold_mult * roll_vol

        labels = pd.Series(np.nan, index=df.index)
        labels[fwd_ret > threshold] = 2.0   # bullish
        labels[fwd_ret < -threshold] = 0.0  # bearish
        labels[(fwd_ret >= -threshold) & (fwd_ret <= threshold)] = 1.0  # neutral
        labels[fwd_ret.isna() | roll_vol.isna()] = np.nan
        return labels

    else:
        raise ValueError(f"Unknown label mode: {mode}")


def label_distribution(labels: pd.Series) -> dict:
    """Return distribution of labels as {label: count}."""
    clean = labels.dropna()
    counts = clean.value_counts().sort_index()
    total = len(clean)
    return {int(k): {"count": int(v), "pct": round(v / total * 100, 1)} for k, v in counts.items()}

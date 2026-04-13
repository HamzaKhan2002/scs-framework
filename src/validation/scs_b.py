"""
Signal Credibility Score — Phase B (SCS-B).

5 components for walk-forward validation:
  S_time:   Temporal robustness across walk-forward windows
  S_asset:  Cross-asset robustness
  S_cost:   Cost tolerance (baseline vs stress)
  S_struct: Structural stability across windows
  S_eco:    Economic coherence (turnover, trade frequency)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

EPSILON = 1e-8


def compute_s_time_b(results: pd.DataFrame) -> float:
    """S_time: Robustness across walk-forward windows."""
    sharpe_by_window = results.groupby("window")["sharpe"].mean().values
    if len(sharpe_by_window) <= 1:
        return 0.0

    sign_ratio = (sharpe_by_window > 0).mean()
    mean_val = sharpe_by_window.mean()
    std_val = sharpe_by_window.std(ddof=1)
    cv = std_val / (abs(mean_val) + EPSILON)

    score = 0.60 * sign_ratio + 0.40 * np.exp(-cv)

    if mean_val < 0:
        score = min(score, 0.30)

    return float(np.clip(score, 0, 1))


def compute_s_asset_b(results: pd.DataFrame) -> float:
    """S_asset: Robustness across assets."""
    sharpe_by_asset = results.groupby("ticker")["sharpe"].mean().values
    if len(sharpe_by_asset) <= 1:
        return 0.0

    sign_ratio = (sharpe_by_asset > 0).mean()
    mean_val = sharpe_by_asset.mean()
    std_val = sharpe_by_asset.std(ddof=1)
    cv = std_val / (abs(mean_val) + EPSILON)

    score = 0.60 * sign_ratio + 0.40 * np.exp(-cv)

    if mean_val < 0:
        score = min(score, 0.30)

    return float(np.clip(score, 0, 1))


def compute_s_cost(results_base: pd.DataFrame, results_stress: pd.DataFrame) -> float:
    """
    S_cost: Cost tolerance.
    Measures fraction of Sharpe that survives when costs are doubled.
    """
    sharpe_base = results_base["sharpe"].mean()
    sharpe_stress = results_stress["sharpe"].mean()

    if sharpe_base <= 0:
        return 0.0

    ratio = sharpe_stress / (sharpe_base + EPSILON)
    return float(np.clip(ratio, 0.0, 1.0))


def compute_s_struct(results: pd.DataFrame) -> float:
    """
    S_struct: Structural stability.
    Measures consistency of per-asset Sharpe vectors across walk-forward windows.
    """
    windows = results["window"].unique()
    if len(windows) < 2:
        return 0.5

    # Build matrix: window × asset Sharpe
    pivot = results.pivot_table(
        index="window", columns="ticker", values="sharpe", aggfunc="mean"
    ).dropna(axis=1, how="any")

    if pivot.shape[1] < 2:
        return 0.5

    # Pairwise correlation between windows
    corr_matrix = pivot.T.corr()
    n = len(corr_matrix)
    if n < 2:
        return 0.5

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(corr_matrix.iloc[i, j])

    if not pairs:
        return 0.5

    mean_corr = np.mean(pairs)
    # Normalize [-1, 1] → [0, 1]
    return float(np.clip((mean_corr + 1) / 2, 0, 1))


def compute_s_eco(results: pd.DataFrame, horizon: int) -> float:
    """
    S_eco: Economic coherence.
    Checks trade frequency is economically plausible for the horizon.
    """
    mean_trades = results.groupby(["window", "ticker"])["n_trades"].mean().mean()

    # Normalize: 50 trades per year optimal for 10-day horizon
    trade_score = min(1.0, mean_trades / 50) if mean_trades > 0 else 0.0

    # Turnover: trades per year should be in [5, 100] for 10d horizon
    # (252 / 10 = 25.2 max theoretical)
    annual_trades = mean_trades
    if 5 <= annual_trades <= 50:
        turnover_score = 1.0
    elif annual_trades > 50:
        turnover_score = max(0.5, 1.0 - (annual_trades - 50) / 100)
    else:
        turnover_score = max(0.0, annual_trades / 5)

    return float(np.clip(0.50 * trade_score + 0.50 * turnover_score, 0, 1))


def compute_scs_b(
    results_base: pd.DataFrame,
    results_stress: pd.DataFrame,
    horizon: int = 10,
    weights: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Compute full SCS-B score.

    Args:
        results_base: DataFrame with columns:
            window, ticker, sharpe, n_trades, total_return
            (under baseline costs)
        results_stress: Same structure under stress costs.
        horizon: Holding horizon.
        weights: Component weights.

    Returns:
        Dict with SCS_B, all components, and verdict.
    """
    if weights is None:
        weights = {"S_time": 0.25, "S_asset": 0.25, "S_cost": 0.20,
                    "S_struct": 0.15, "S_eco": 0.15}

    s_time = compute_s_time_b(results_base)
    s_asset = compute_s_asset_b(results_base)
    s_cost = compute_s_cost(results_base, results_stress)
    s_struct = compute_s_struct(results_base)
    s_eco = compute_s_eco(results_base, horizon)

    scs_b = (
        weights["S_time"] * s_time
        + weights["S_asset"] * s_asset
        + weights["S_cost"] * s_cost
        + weights["S_struct"] * s_struct
        + weights["S_eco"] * s_eco
    )

    if scs_b >= 0.60:
        verdict = "VALID_FOR_PHASE_C"
    elif scs_b >= 0.45:
        verdict = "BORDERLINE"
    else:
        verdict = "REJECTED"

    return {
        "SCS_B": round(scs_b, 4),
        "S_time": round(s_time, 4),
        "S_asset": round(s_asset, 4),
        "S_cost": round(s_cost, 4),
        "S_struct": round(s_struct, 4),
        "S_eco": round(s_eco, 4),
        "verdict": verdict,
    }

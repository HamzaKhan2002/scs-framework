"""
Signal Credibility Score — Phase A (SCS-A).

5 components, all working:
  S_time:  Temporal robustness across sub-periods
  S_asset: Cross-asset robustness
  S_model: Inter-model consensus (Spearman correlation)
  S_seed:  Stochastic seed stability
  S_dist:  Trade-level distributional quality

Signal groups are (horizon, label_mode) containing ALL model types.
This activates S_model properly.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import rankdata, skew, kurtosis
from typing import Dict, Any, List

EPSILON = 1e-8


def _robustness_score(values: np.ndarray) -> float:
    """
    Generic robustness scorer used by S_time, S_asset, S_seed.
    Input: array of mean Sharpe ratios per group (period/asset/seed).

    Components:
      sign_ratio (30%): fraction of groups with positive Sharpe
      stability  (20%): exp(-CV) — lower CV = higher score
      magnitude  (25%): normalized mean absolute Sharpe
      worst_case (25%): penalty based on minimum Sharpe

    Hard gates:
      mean < 0       → capped at 0.30
      min < -1.0     → capped at 0.40
      sign_ratio < 0.5 → capped at 0.50
    """
    if len(values) <= 1:
        return 0.0

    sign_ratio = (values > 0).mean()
    mean_val = values.mean()
    std_val = values.std(ddof=1) if len(values) > 1 else 0.0
    cv = std_val / (abs(mean_val) + EPSILON)
    min_val = values.min()

    # Components
    c_sign = sign_ratio
    c_stability = np.exp(-cv)
    c_magnitude = min(1.0, abs(mean_val) / 0.5)

    if min_val > -0.5:
        c_worst = 1.0
    elif min_val > -1.0:
        c_worst = max(0.0, 1.0 + (min_val + 0.5) / 0.5)
    else:
        c_worst = 0.0

    score = 0.30 * c_sign + 0.20 * c_stability + 0.25 * c_magnitude + 0.25 * c_worst

    # Hard gates
    if mean_val < 0:
        score = min(score, 0.30)
    if min_val < -1.0:
        score = min(score, 0.40)
    if sign_ratio < 0.5:
        score = min(score, 0.50)

    return float(np.clip(score, 0.0, 1.0))


def compute_s_time(results: pd.DataFrame) -> float:
    """S_time: Robustness across time periods."""
    sharpe_by_period = results.groupby("period")["sharpe"].mean().values
    return _robustness_score(sharpe_by_period)


def compute_s_asset(results: pd.DataFrame) -> float:
    """S_asset: Robustness across assets."""
    sharpe_by_asset = results.groupby("ticker")["sharpe"].mean().values
    return _robustness_score(sharpe_by_asset)


def _kish_effective_n(weights: np.ndarray) -> float:
    """Kish effective sample size: (Σw)² / Σ(w²)."""
    s = weights.sum()
    if s == 0:
        return 0.0
    return float(s ** 2 / (weights ** 2).sum())


def _weighted_spearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation on ranks."""
    rx = rankdata(x)
    ry = rankdata(y)
    sw = w.sum()
    if sw == 0:
        return 0.0
    mx = np.average(rx, weights=w)
    my = np.average(ry, weights=w)
    dx = rx - mx
    dy = ry - my
    cov = np.sum(w * dx * dy) / sw
    vx = np.sum(w * dx ** 2) / sw
    vy = np.sum(w * dy ** 2) / sw
    denom = np.sqrt(vx * vy)
    if denom < 1e-12:
        return 0.0
    return float(cov / denom)


def compute_s_model(results: pd.DataFrame, min_effective_n: int = 5) -> float:
    """
    S_model v2: Performance-weighted inter-model rank consensus.

    Weights each (ticker, period) condition by max(max_model_SR, 0),
    focusing consensus measurement on conditions where at least one
    model is profitable. Uses Kish effective N guard to ensure
    sufficient diversity of informative conditions.

    Returns max(mean_pairwise_weighted_spearman, 0) ∈ [0, 1].
    """
    pivot = results.pivot_table(
        index=["ticker", "period"],
        columns="model_type",
        values="sharpe",
        aggfunc="mean",
    ).dropna(axis=0, how="any")

    models = pivot.columns.tolist()
    if len(models) < 2:
        return 0.0

    if len(pivot) < 3:
        return 0.0

    # Weights: max(best_model_SR, 0) per condition
    weights = pivot.clip(lower=0).max(axis=1).values

    # Kish effective N guard
    n_eff = _kish_effective_n(weights)
    if n_eff < min_effective_n:
        return 0.0

    # Pairwise weighted Spearman correlations
    rhos = []
    for i, j in combinations(models, 2):
        rho = _weighted_spearman(pivot[i].values, pivot[j].values, weights)
        rhos.append(rho)

    mean_rho = float(np.mean(rhos))
    return float(np.clip(max(mean_rho, 0.0), 0.0, 1.0))


def compute_s_seed(results: pd.DataFrame) -> float:
    """S_seed: Stability across random seeds."""
    sharpe_by_seed = results.groupby("seed")["sharpe"].mean().values

    if len(sharpe_by_seed) <= 1:
        return 0.0

    sign_ratio = (sharpe_by_seed > 0).mean()
    mean_val = sharpe_by_seed.mean()
    std_val = sharpe_by_seed.std(ddof=1)
    cv = std_val / (abs(mean_val) + EPSILON)
    stability = np.exp(-cv)

    # Ranking stability across conditions
    ranking_vars = []
    for ticker in results["ticker"].unique():
        for period in results["period"].unique():
            sub = results[(results["ticker"] == ticker) & (results["period"] == period)]
            seed_sharpes = sub.groupby("seed")["sharpe"].mean()
            if len(seed_sharpes) > 1:
                ranks = seed_sharpes.rank(ascending=False).values
                ranking_vars.append(np.std(ranks))

    if ranking_vars:
        ranking_stability = np.exp(-np.mean(ranking_vars) / len(sharpe_by_seed))
    else:
        ranking_stability = 0.5

    s_seed = 0.40 * sign_ratio + 0.30 * stability + 0.30 * ranking_stability

    # Hard gate
    if sign_ratio < 0.4:
        s_seed = min(s_seed, 0.30)

    return float(np.clip(s_seed, 0.0, 1.0))


def compute_s_dist(all_trade_returns: List[float]) -> float:
    """
    S_dist: Trade-level distributional quality.

    Operates on ACTUAL TRADE RETURNS (not aggregated config-level returns).

    Components:
      skewness (25%): Acceptable if |skew| <= 5
      kurtosis (20%): Acceptable if kurt <= 8
      win_ratio (35%): Most important — fraction of positive trades
      trade_size_stability (15%): CV of |returns|
      concentration (5%): Top 3 trades / total P&L
    """
    returns = np.array(all_trade_returns)
    n = len(returns)

    if n < 20:
        return 0.30  # Too few trades

    # Skewness
    skew_val = abs(skew(returns))
    if skew_val <= 5:
        c_skew = 1.0
    elif skew_val <= 8:
        c_skew = 0.7 + 0.3 * ((8 - skew_val) / 3)
    else:
        c_skew = 0.5

    # Kurtosis
    kurt_val = kurtosis(returns, fisher=True)
    if kurt_val <= 8:
        c_kurt = 1.0
    elif kurt_val <= 15:
        c_kurt = 0.8 + 0.2 * ((15 - kurt_val) / 7)
    else:
        c_kurt = 0.4

    # Win ratio
    win_ratio = (returns > 0).mean()
    if win_ratio >= 0.45:
        c_win = 1.0
    elif win_ratio >= 0.35:
        c_win = 0.7 + 0.3 * ((win_ratio - 0.35) / 0.10)
    elif win_ratio >= 0.25:
        c_win = 0.3
    else:
        c_win = 0.1

    # Trade size stability
    abs_ret = np.abs(returns)
    trade_cv = abs_ret.std() / (abs_ret.mean() + EPSILON)
    if trade_cv <= 2.0:
        c_stab = 1.0
    elif trade_cv <= 3.5:
        c_stab = 0.9 - 0.3 * ((trade_cv - 2.0) / 1.5)
    else:
        c_stab = 0.6

    # Concentration
    sorted_abs = np.sort(abs_ret)[::-1]
    top3 = sorted_abs[:min(3, len(sorted_abs))].sum()
    total = abs_ret.sum()
    conc = top3 / total if total > 0 else 0
    if conc <= 0.30:
        c_conc = 1.0
    elif conc <= 0.50:
        c_conc = 0.8
    else:
        c_conc = 0.4

    s_dist = 0.25 * c_skew + 0.20 * c_kurt + 0.35 * c_win + 0.15 * c_stab + 0.05 * c_conc

    # Hard caps for limited data
    if n < 50:
        s_dist = min(s_dist, 0.50)
    elif n < 100:
        s_dist = min(s_dist, 0.70)

    return float(np.clip(s_dist, 0.0, 1.0))


def compute_scs_a(
    results: pd.DataFrame,
    all_trade_returns: List[float],
    weights: Dict[str, float] = None,
    hard_gates: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Compute full SCS-A score.

    Args:
        results: DataFrame with columns:
            ticker, period, seed, model_type, sharpe, n_trades, total_return
        all_trade_returns: Flat list of all individual trade returns.
        weights: Dict of component weights (default from config).
        hard_gates: Dict of gate thresholds.

    Returns:
        Dict with SCS_A score, all component scores, and verdict.
    """
    if weights is None:
        weights = {"S_time": 0.25, "S_asset": 0.25, "S_model": 0.25,
                    "S_seed": 0.15, "S_dist": 0.10}
    if hard_gates is None:
        hard_gates = {"min_global_sharpe": 0.0, "min_positive_ratio": 0.50,
                      "min_trades_per_bucket": 8}

    # Pre-gates
    mean_sharpe = results["sharpe"].mean()
    if mean_sharpe < hard_gates.get("min_global_sharpe", 0.0):
        return _rejected("PRE_GATE: mean Sharpe < {:.2f}".format(
            hard_gates["min_global_sharpe"]))

    # Positive ratio check
    grouped = results.groupby(["ticker", "period"])["sharpe"].mean()
    pos_ratio = (grouped > 0).mean()
    if pos_ratio < hard_gates.get("min_positive_ratio", 0.50):
        return _rejected(f"PRE_GATE: positive ratio {pos_ratio:.2f} < threshold")

    # Compute components
    s_time = compute_s_time(results)
    s_asset = compute_s_asset(results)
    s_model = compute_s_model(results)
    s_seed = compute_s_seed(results)
    s_dist = compute_s_dist(all_trade_returns)

    scs_a = (
        weights["S_time"] * s_time
        + weights["S_asset"] * s_asset
        + weights["S_model"] * s_model
        + weights["S_seed"] * s_seed
        + weights["S_dist"] * s_dist
    )

    if scs_a >= 0.70:
        verdict = "PHASE_B_APPROVED"
    elif scs_a >= 0.50:
        verdict = "BORDERLINE"
    else:
        verdict = "REJECTED"

    return {
        "SCS_A": round(scs_a, 4),
        "S_time": round(s_time, 4),
        "S_asset": round(s_asset, 4),
        "S_model": round(s_model, 4),
        "S_seed": round(s_seed, 4),
        "S_dist": round(s_dist, 4),
        "verdict": verdict,
        "mean_sharpe": round(mean_sharpe, 4),
        "positive_ratio": round(pos_ratio, 4),
        "n_results": len(results),
        "n_trades_total": len(all_trade_returns),
    }


def _rejected(reason: str) -> Dict[str, Any]:
    return {
        "SCS_A": 0.0, "S_time": 0.0, "S_asset": 0.0, "S_model": 0.0,
        "S_seed": 0.0, "S_dist": 0.0, "verdict": "REJECTED",
        "rejection_reason": reason,
    }

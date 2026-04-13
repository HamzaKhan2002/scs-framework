"""
Bootstrap confidence intervals for portfolio metrics.
Uses block bootstrap to respect temporal structure of trades.
"""

import numpy as np
from typing import Dict, List, Callable
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    distribution: np.ndarray


def block_bootstrap_trades(
    trade_returns: List[float],
    metric_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 10000,
    block_size: int = 10,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """
    Block bootstrap on trade returns.

    Block size ≈ average holding period (10 days) to preserve
    any serial correlation between adjacent trades.
    """
    rng = np.random.RandomState(seed)
    returns = np.array(trade_returns)
    n = len(returns)

    if n < 5:
        point = metric_fn(returns)
        return BootstrapResult(point, point, point, 0.0, np.array([point]))

    point_estimate = metric_fn(returns)
    boot_samples = []

    for _ in range(n_bootstrap):
        # Block bootstrap: sample blocks of consecutive trades
        n_blocks = max(1, n // block_size + 1)
        indices = []
        for _ in range(n_blocks):
            start = rng.randint(0, max(1, n - block_size))
            indices.extend(range(start, min(start + block_size, n)))
        indices = indices[:n]  # Trim to original size
        sample = returns[indices]
        boot_samples.append(metric_fn(sample))

    boot_arr = np.array(boot_samples)
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_arr, 100 * alpha / 2)
    ci_upper = np.percentile(boot_arr, 100 * (1 - alpha / 2))

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=boot_arr.std(),
        distribution=boot_arr,
    )


def sharpe_from_trades(returns: np.ndarray, horizon: int = 10) -> float:
    """Compute annualized Sharpe from trade returns."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252 / horizon)


def bootstrap_sharpe(
    trade_returns: List[float],
    horizon: int = 10,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap CI specifically for Sharpe ratio."""
    return block_bootstrap_trades(
        trade_returns,
        metric_fn=lambda r: sharpe_from_trades(r, horizon),
        n_bootstrap=n_bootstrap,
        block_size=horizon,
        confidence=confidence,
        seed=seed,
    )


def bootstrap_all_metrics(
    trade_returns: List[float],
    daily_returns: np.ndarray,
    horizon: int = 10,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, BootstrapResult]:
    """Bootstrap CI for all key metrics."""
    results = {}

    # Sharpe from daily equity returns
    results["sharpe_daily"] = block_bootstrap_trades(
        daily_returns.tolist(),
        metric_fn=lambda r: r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
        n_bootstrap=n_bootstrap,
        block_size=21,  # ~1 month blocks for daily returns
        seed=seed,
    )

    # Total return
    results["total_return"] = block_bootstrap_trades(
        trade_returns,
        metric_fn=lambda r: r.sum() * 100,
        n_bootstrap=n_bootstrap,
        block_size=horizon,
        seed=seed,
    )

    # Win rate
    results["win_rate"] = block_bootstrap_trades(
        trade_returns,
        metric_fn=lambda r: (r > 0).mean() * 100,
        n_bootstrap=n_bootstrap,
        block_size=horizon,
        seed=seed,
    )

    return results

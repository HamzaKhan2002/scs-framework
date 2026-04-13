"""
Block bootstrap test for Sharpe ratio difference.

More robust than Ledoit-Wolf when returns exhibit non-stationarity
or non-normality beyond what HAC correction handles.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BlockBootstrapResult:
    strategy_sharpe: float
    benchmark_sharpe: float
    sharpe_difference: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    block_size: int
    is_significant: bool


def block_bootstrap_sharpe_test(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    n_bootstrap: int = 10000,
    block_size: Optional[int] = None,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BlockBootstrapResult:
    """
    Block bootstrap test for H0: SR_strategy <= SR_benchmark.

    Uses circular block bootstrap to preserve serial dependence.

    Args:
        strategy_returns: Daily returns of strategy
        benchmark_returns: Daily returns of benchmark
        n_bootstrap: Number of bootstrap samples
        block_size: Block size for bootstrap (default: sqrt(n))
        confidence_level: For CI computation
        seed: RNG seed

    Returns:
        BlockBootstrapResult with p-value and CI for SR difference
    """
    n = min(len(strategy_returns), len(benchmark_returns))
    strat = strategy_returns[:n]
    bench = benchmark_returns[:n]

    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))

    # Observed Sharpe difference
    sr_strat = np.mean(strat) / (np.std(strat, ddof=1) + 1e-10) * np.sqrt(252)
    sr_bench = np.mean(bench) / (np.std(bench, ddof=1) + 1e-10) * np.sqrt(252)
    observed_diff = sr_strat - sr_bench

    rng = np.random.RandomState(seed)
    boot_diffs = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Circular block bootstrap
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.randint(0, n, size=n_blocks)

        indices = []
        for start in starts:
            block = [(start + j) % n for j in range(block_size)]
            indices.extend(block)
        indices = indices[:n]

        boot_strat = strat[indices]
        boot_bench = bench[indices]

        sr_s = np.mean(boot_strat) / (np.std(boot_strat, ddof=1) + 1e-10) * np.sqrt(252)
        sr_b = np.mean(boot_bench) / (np.std(boot_bench, ddof=1) + 1e-10) * np.sqrt(252)
        boot_diffs[b] = sr_s - sr_b

    # One-sided p-value: P(strategy > benchmark)
    # Under null, center bootstrap distribution at 0
    centered_diffs = boot_diffs - np.mean(boot_diffs)
    p_value = float(np.mean(centered_diffs >= observed_diff))

    # CI for the difference
    alpha = 1 - confidence_level
    ci_lo = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))

    return BlockBootstrapResult(
        strategy_sharpe=round(sr_strat, 4),
        benchmark_sharpe=round(sr_bench, 4),
        sharpe_difference=round(observed_diff, 4),
        p_value=round(p_value, 4),
        ci_lower=round(ci_lo, 4),
        ci_upper=round(ci_hi, 4),
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        is_significant=p_value < 0.05,
    )

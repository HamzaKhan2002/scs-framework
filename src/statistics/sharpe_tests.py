"""
Sharpe ratio comparison tests.

Implements Ledoit & Wolf (2008) HAC-robust test for equality of Sharpe ratios.
"""

import numpy as np
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class SharpeTestResult:
    strategy_sharpe: float
    benchmark_sharpe: float
    sharpe_difference: float
    t_statistic: float
    p_value: float
    is_significant: bool


def _newey_west_variance(x: np.ndarray, max_lag: int = None) -> float:
    """Newey-West HAC variance estimator."""
    n = len(x)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))

    x_centered = x - x.mean()
    gamma_0 = np.dot(x_centered, x_centered) / n

    gamma_sum = 0.0
    for j in range(1, max_lag + 1):
        weight = 1 - j / (max_lag + 1)  # Bartlett kernel
        gamma_j = np.dot(x_centered[j:], x_centered[:-j]) / n
        gamma_sum += 2 * weight * gamma_j

    return (gamma_0 + gamma_sum) / n


def ledoit_wolf_sharpe_test(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> SharpeTestResult:
    """
    Test H0: SR_strategy = SR_benchmark
    vs   H1: SR_strategy > SR_benchmark (one-sided)

    Uses HAC standard errors via Newey-West.

    Based on Ledoit & Wolf (2008), "Robust performance hypothesis testing
    with the Sharpe ratio", Journal of Empirical Finance.
    """
    r1 = np.asarray(strategy_returns).flatten()
    r2 = np.asarray(benchmark_returns).flatten()

    n = min(len(r1), len(r2))
    r1 = r1[:n]
    r2 = r2[:n]

    mu1, mu2 = r1.mean(), r2.mean()
    s1, s2 = r1.std(ddof=1), r2.std(ddof=1)

    if s1 == 0 or s2 == 0:
        return SharpeTestResult(0, 0, 0, 0, 1.0, False)

    sr1 = mu1 / s1 * np.sqrt(252)
    sr2 = mu2 / s2 * np.sqrt(252)

    # Difference of Sharpe ratios - HAC inference
    # The key quantity is the difference in risk-adjusted returns
    d = r1 / s1 - r2 / s2

    hac_var = _newey_west_variance(d)
    se = np.sqrt(hac_var) * np.sqrt(252)

    if se == 0:
        return SharpeTestResult(round(sr1, 4), round(sr2, 4), round(sr1 - sr2, 4),
                                0.0, 1.0, False)

    t_stat = (sr1 - sr2) / se
    p_value = 1 - norm.cdf(t_stat)  # One-sided

    return SharpeTestResult(
        strategy_sharpe=round(sr1, 4),
        benchmark_sharpe=round(sr2, 4),
        sharpe_difference=round(sr1 - sr2, 4),
        t_statistic=round(t_stat, 4),
        p_value=round(p_value, 4),
        is_significant=p_value < 0.05,
    )

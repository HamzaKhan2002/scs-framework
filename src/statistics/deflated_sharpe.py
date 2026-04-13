"""
Deflated Sharpe Ratio — Bailey & López de Prado (2014).

Corrects for the number of trials (signal groups tested in Phase A).
Tests: P(SR* > 0 | N trials).
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


@dataclass
class DSRResult:
    observed_sharpe: float
    deflated_sharpe: float
    expected_max_sharpe: float
    p_value: float
    is_significant: bool
    n_trials: int


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    sharpe_std: float = None,
) -> DSRResult:
    """
    Compute the Deflated Sharpe Ratio.

    Args:
        observed_sharpe: The Sharpe ratio of the selected strategy.
        n_trials: Number of strategy variants tested (e.g., 4 signal groups).
        n_obs: Number of return observations (e.g., daily returns).
        skewness: Skewness of strategy returns.
        excess_kurtosis: Excess kurtosis of strategy returns.
        sharpe_std: Std of Sharpe ratio estimates across trials.
            If None, estimated as 1/sqrt(n_obs).

    Returns:
        DSRResult with deflated Sharpe, p-value, and significance.
    """
    if sharpe_std is None:
        # Standard error of Sharpe under normality
        sharpe_std = np.sqrt(
            (1 + 0.5 * observed_sharpe**2
             - skewness * observed_sharpe
             + (excess_kurtosis / 4) * observed_sharpe**2) / n_obs
        )

    if sharpe_std <= 0 or n_trials <= 0:
        return DSRResult(observed_sharpe, observed_sharpe, 0.0, 0.5, False, n_trials)

    # Expected maximum Sharpe under null (all trials are noise)
    # E[max(Z_1,...,Z_N)] ≈ σ * ((1-γ) * Φ^{-1}(1-1/N) + γ * Φ^{-1}(1-1/(Ne)))
    gamma = 0.5772156649  # Euler-Mascheroni constant
    if n_trials > 1:
        z1 = norm.ppf(1 - 1 / n_trials)
        z2 = norm.ppf(1 - 1 / (n_trials * np.e))
        e_max = sharpe_std * ((1 - gamma) * z1 + gamma * z2)
    else:
        e_max = 0.0

    # Deflated Sharpe Ratio
    dsr = (observed_sharpe - e_max) / sharpe_std
    p_value = 1 - norm.cdf(dsr)

    return DSRResult(
        observed_sharpe=observed_sharpe,
        deflated_sharpe=round(dsr, 4),
        expected_max_sharpe=round(e_max, 4),
        p_value=round(p_value, 4),
        is_significant=p_value < 0.05,
        n_trials=n_trials,
    )

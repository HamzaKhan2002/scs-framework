"""
Permutation test — null distribution of Sharpe ratios.

Shuffles labels N times, runs the full pipeline each time,
and builds a distribution of Sharpe ratios under the null hypothesis
that labels are unrelated to features.
"""

import numpy as np
from typing import Callable, List
from dataclasses import dataclass


@dataclass
class PermutationResult:
    observed_sharpe: float
    null_mean: float
    null_std: float
    p_value: float
    is_significant: bool
    null_distribution: np.ndarray
    n_permutations: int


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    pipeline_fn: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 1000,
    seed: int = 42,
) -> PermutationResult:
    """
    Label-shuffling permutation test.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Label vector (n_samples,).
        pipeline_fn: Function that takes (X, y) and returns a Sharpe ratio.
            This should encapsulate: split → train → predict → backtest → Sharpe.
        n_permutations: Number of permutations.
        seed: Random seed.

    Returns:
        PermutationResult with observed, null distribution, and p-value.
    """
    rng = np.random.RandomState(seed)

    # Observed Sharpe with real labels
    observed = pipeline_fn(X, y)

    # Null distribution with shuffled labels
    null_sharpes = []
    for i in range(n_permutations):
        y_shuffled = rng.permutation(y)
        try:
            s = pipeline_fn(X, y_shuffled)
        except Exception:
            s = 0.0
        null_sharpes.append(s)

    null_arr = np.array(null_sharpes)
    null_mean = null_arr.mean()
    null_std = null_arr.std()

    # One-sided p-value: fraction of null >= observed
    p_value = (null_arr >= observed).mean()

    return PermutationResult(
        observed_sharpe=round(observed, 4),
        null_mean=round(null_mean, 4),
        null_std=round(null_std, 4),
        p_value=round(p_value, 4),
        is_significant=p_value < 0.05,
        null_distribution=null_arr,
        n_permutations=n_permutations,
    )

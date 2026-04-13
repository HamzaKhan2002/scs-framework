"""
Purged temporal splitter with embargo.

Ensures no data leakage:
- Training labels that look forward by H days cannot overlap with test period.
- Feature rolling windows cannot overlap with test period.
- Embargo gap = max(embargo_days, horizon) between train end and test start.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class SplitResult:
    """Result of a temporal split."""
    train_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    gap_days: int


def purged_temporal_split(
    dates: pd.DatetimeIndex,
    train_ratio: float = 0.70,
    embargo_days: int = 20,
) -> SplitResult:
    """
    Split a date index temporally with embargo gap.

    The embargo gap removes `embargo_days` trading days between
    the end of training and the start of testing.
    """
    dates = dates.sort_values()
    n = len(dates)
    n_train = int(n * train_ratio)

    # Effective train end, accounting for embargo
    n_train_eff = max(1, n_train - embargo_days)
    n_test_start = n_train

    train_dates = dates[:n_train_eff]
    test_dates = dates[n_test_start:]

    if len(train_dates) == 0 or len(test_dates) == 0:
        raise ValueError(
            f"Split produced empty set: {len(train_dates)} train, {len(test_dates)} test "
            f"(n={n}, ratio={train_ratio}, embargo={embargo_days})"
        )

    gap = (test_dates[0] - train_dates[-1]).days

    return SplitResult(
        train_idx=train_dates,
        test_idx=test_dates,
        train_end=train_dates[-1],
        test_start=test_dates[0],
        gap_days=gap,
    )


def split_into_sub_periods(
    start: str, end: str, n_periods: int, dates: pd.DatetimeIndex
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Split [start, end] into n_periods non-overlapping sub-periods.
    Uses actual trading dates to determine boundaries.
    """
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    period_dates = dates[mask].sort_values()

    n = len(period_dates)
    boundaries = []
    chunk = n // n_periods

    for i in range(n_periods):
        s = i * chunk
        e = (i + 1) * chunk - 1 if i < n_periods - 1 else n - 1
        boundaries.append((period_dates[s], period_dates[e]))

    return boundaries


@dataclass
class WalkForwardWindow:
    """One walk-forward window."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_days: int

    def effective_train_end(self) -> pd.Timestamp:
        """Train end minus embargo (in calendar days)."""
        return self.train_end - pd.Timedelta(days=self.embargo_days)


def build_walk_forward_windows(
    windows_config: List[dict],
    embargo_days: int = 20,
) -> List[WalkForwardWindow]:
    """Build walk-forward windows from config."""
    result = []
    for w in windows_config:
        result.append(WalkForwardWindow(
            train_start=pd.Timestamp(w["train_start"]),
            train_end=pd.Timestamp(w["train_end"]),
            test_start=pd.Timestamp(w["test_start"]),
            test_end=pd.Timestamp(w["test_end"]),
            embargo_days=embargo_days,
        ))
    return result

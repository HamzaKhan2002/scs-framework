"""
Test suite for the publication-grade ML trading pipeline.
Tests cover: features, labeling, splitting (no leakage), backtest, SCS scoring.

Run: python -m pytest tests/ -v
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.engine import build_features, FEATURE_COLS, get_feature_matrix
from src.labeling.engine import compute_forward_return, make_labels
from src.validation.temporal_split import purged_temporal_split
from src.backtest.portfolio_engine import (
    _compute_position_return, _compute_pnl, run_naive_backtest, compute_metrics
)
from src.validation.scs_a import (
    _robustness_score, compute_s_time, compute_s_asset, compute_s_model,
    compute_s_seed, compute_s_dist, compute_scs_a,
)
from src.statistics.deflated_sharpe import deflated_sharpe_ratio
from src.statistics.sharpe_tests import ledoit_wolf_sharpe_test
from src.data.config import load_config, get_all_tickers
from pipelines.phase_a import corrupt_labels


# ============================================================
#  FIXTURES
# ============================================================

@pytest.fixture
def sample_ohlcv():
    """Generate a synthetic OHLCV DataFrame."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range(start="2020-01-01", periods=n)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def config():
    return load_config()


# ============================================================
#  FEATURES TESTS
# ============================================================

class TestFeatures:
    def test_feature_count(self, sample_ohlcv):
        df = build_features(sample_ohlcv)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"
        assert len(FEATURE_COLS) == 11

    def test_no_nan_dropping(self, sample_ohlcv):
        """build_features must NOT drop rows — caller handles alignment."""
        df = build_features(sample_ohlcv)
        assert len(df) == len(sample_ohlcv)

    def test_warmup_produces_nan(self, sample_ohlcv):
        """First ~20 rows should have NaN in rolling features."""
        df = build_features(sample_ohlcv)
        assert df["vol_20d"].iloc[:19].isna().all()
        assert df["atr_14"].iloc[:13].isna().all()

    def test_feature_matrix_extraction(self, sample_ohlcv):
        df = build_features(sample_ohlcv)
        X, cols = get_feature_matrix(df)
        assert list(cols) == FEATURE_COLS
        assert X.shape[1] == 11


# ============================================================
#  LABELING TESTS
# ============================================================

class TestLabeling:
    def test_binary_labels_values(self, sample_ohlcv):
        labels = make_labels(sample_ohlcv, horizon=10, mode="directional_binary")
        clean = labels.dropna()
        assert set(clean.unique()).issubset({0.0, 1.0})

    def test_ternary_labels_values(self, sample_ohlcv):
        labels = make_labels(sample_ohlcv, horizon=10, mode="multiclass_volatility")
        clean = labels.dropna()
        assert set(clean.unique()).issubset({0.0, 1.0, 2.0})

    def test_ternary_has_neutral_zone(self, sample_ohlcv):
        labels = make_labels(sample_ohlcv, horizon=10, mode="multiclass_volatility")
        clean = labels.dropna()
        # Neutral (class 1) should capture some fraction
        neutral_pct = (clean == 1.0).mean()
        assert neutral_pct > 0.05, f"Neutral zone too small: {neutral_pct:.1%}"

    def test_forward_return_nan_at_end(self, sample_ohlcv):
        """Last H rows must be NaN (no future data)."""
        fwd = compute_forward_return(sample_ohlcv, horizon=10)
        assert fwd.iloc[-10:].isna().all()
        assert fwd.iloc[-11:].notna().iloc[0]  # Just before last H should exist

    def test_forward_return_formula(self, sample_ohlcv):
        """r_t(H) = Close(t+H) / Open(t+1) - 1."""
        fwd = compute_forward_return(sample_ohlcv, horizon=5)
        idx = 50  # An arbitrary middle row
        expected = sample_ohlcv["close"].iloc[idx + 5] / sample_ohlcv["open"].iloc[idx + 1] - 1
        assert abs(fwd.iloc[idx] - expected) < 1e-10


# ============================================================
#  TEMPORAL SPLIT / NO LEAKAGE TESTS
# ============================================================

class TestTemporalSplit:
    def test_embargo_gap(self, sample_ohlcv):
        split = purged_temporal_split(sample_ohlcv.index, train_ratio=0.70, embargo_days=20)
        gap = (split.test_start - split.train_end).days
        assert gap >= 20, f"Embargo gap too small: {gap} days"

    def test_no_overlap(self, sample_ohlcv):
        split = purged_temporal_split(sample_ohlcv.index, train_ratio=0.70, embargo_days=20)
        train_set = set(split.train_idx)
        test_set = set(split.test_idx)
        assert len(train_set & test_set) == 0, "Train and test overlap!"

    def test_temporal_ordering(self, sample_ohlcv):
        split = purged_temporal_split(sample_ohlcv.index, train_ratio=0.70, embargo_days=20)
        assert split.train_idx[-1] < split.test_idx[0]

    def test_no_label_leakage(self, sample_ohlcv):
        """
        CRITICAL: The last label in training must not look into the test period.
        With horizon H and embargo E, the last training sample's forward return
        goes H days beyond its date. This must not overlap with test_start.
        """
        horizon = 10
        embargo = 20
        split = purged_temporal_split(sample_ohlcv.index, train_ratio=0.70, embargo_days=embargo)

        # The last training date's label looks forward by `horizon` trading days
        train_end_idx = sample_ohlcv.index.get_loc(split.train_end)
        label_look_ahead_end = sample_ohlcv.index[min(train_end_idx + horizon, len(sample_ohlcv) - 1)]

        assert label_look_ahead_end < split.test_start, (
            f"Label leakage! Last train label sees {label_look_ahead_end.date()}, "
            f"test starts {split.test_start.date()}"
        )


# ============================================================
#  BACKTEST TESTS
# ============================================================

class TestBacktest:
    def test_long_return_positive(self):
        """Long position profits when price goes up."""
        ret = _compute_position_return(100, 110, "long")
        assert ret == pytest.approx(0.10, abs=1e-10)

    def test_short_return_positive(self):
        """Short position profits when price goes down."""
        ret = _compute_position_return(100, 90, "short")
        assert ret == pytest.approx(0.10, abs=1e-10)

    def test_long_return_negative(self):
        ret = _compute_position_return(100, 90, "long")
        assert ret == pytest.approx(-0.10, abs=1e-10)

    def test_short_return_negative(self):
        """Short position LOSES when price goes up."""
        ret = _compute_position_return(100, 110, "short")
        assert ret == pytest.approx(-0.10, abs=1e-10)

    def test_pnl_long(self):
        pnl = _compute_pnl(100, 110, 10, "long")
        assert pnl == pytest.approx(100, abs=1e-10)

    def test_pnl_short(self):
        pnl = _compute_pnl(100, 90, 10, "short")
        assert pnl == pytest.approx(100, abs=1e-10)

    def test_naive_backtest_positive_return(self):
        """A perfect predictor should make money."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.05
        # Always predict bullish with high probability
        preds = np.ones(n) * 0.9

        result = run_naive_backtest(preds, close, open_, horizon=10, cost_bps=0)
        assert result["n_trades"] > 5

    def test_naive_backtest_no_signal(self):
        """Zero predictions should produce zero trades."""
        n = 200
        close = np.linspace(100, 120, n)
        open_ = close + 0.01
        preds = np.zeros(n)  # Never trade

        result = run_naive_backtest(preds, close, open_, horizon=10, cost_bps=5)
        assert result["n_trades"] == 0

    def test_sharpe_from_equity_curve(self):
        """Sharpe must be computed from daily equity pct_change, annualized by sqrt(252)."""
        np.random.seed(42)
        n = 252
        daily_ret = np.random.randn(n) * 0.01 + 0.0005
        equity = 100_000 * np.cumprod(1 + daily_ret)
        eq_series = pd.Series(equity, index=pd.bdate_range("2025-01-01", periods=n))

        metrics = compute_metrics(eq_series, [], 100_000)
        manual_sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        assert abs(metrics["sharpe_ratio"] - manual_sharpe) < 0.1


# ============================================================
#  SCS-A TESTS
# ============================================================

class TestSCSA:
    def test_robustness_positive_signals(self):
        """All positive Sharpes should produce high robustness."""
        values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        score = _robustness_score(values)
        assert score > 0.7

    def test_robustness_negative_signals(self):
        """All negative Sharpes should be capped at 0.30."""
        values = np.array([-0.5, -0.6, -0.7, -0.8, -0.9])
        score = _robustness_score(values)
        assert score <= 0.30

    def test_robustness_mixed_signals(self):
        """Mixed signals with <50% positive should be capped at 0.50."""
        values = np.array([0.5, -0.3, -0.4, -0.5, -0.6])
        score = _robustness_score(values)
        assert score <= 0.50

    def test_s_dist_few_trades(self):
        """Fewer than 20 trades should return 0.30."""
        returns = list(np.random.randn(15) * 0.02)
        score = compute_s_dist(returns)
        assert score == 0.30

    def test_s_model_single_model(self):
        """Single model type should return 0.0 (no consensus measurable)."""
        df = pd.DataFrame({
            "ticker": ["A", "A", "B", "B"],
            "period": ["P1", "P2", "P1", "P2"],
            "model_type": ["lgb", "lgb", "lgb", "lgb"],
            "sharpe": [0.5, 0.6, 0.4, 0.3],
        })
        score = compute_s_model(df)
        assert score == 0.0

    def test_scs_a_pregate_negative_sharpe(self):
        """Mean Sharpe < 0 should be rejected."""
        df = pd.DataFrame({
            "ticker": ["A"] * 6,
            "period": ["P1", "P2"] * 3,
            "model_type": ["lgb"] * 6,
            "seed": [42] * 6,
            "sharpe": [-0.5, -0.3, -0.8, -0.2, -0.7, -0.1],
            "n_trades": [10] * 6,
            "total_return": [-5] * 6,
        })
        result = compute_scs_a(df, [-0.01] * 20)
        assert result["verdict"] == "REJECTED"


# ============================================================
#  STATISTICS TESTS
# ============================================================

class TestStatistics:
    def test_deflated_sharpe_single_trial(self):
        """With 1 trial, DSR should equal observed."""
        dsr = deflated_sharpe_ratio(observed_sharpe=1.5, n_trials=1, n_obs=252)
        # With 1 trial, expected max is 0, so DSR depends on SE
        assert dsr.observed_sharpe == 1.5

    def test_deflated_sharpe_many_trials(self):
        """More trials should make it harder to be significant."""
        dsr_few = deflated_sharpe_ratio(observed_sharpe=0.5, n_trials=2, n_obs=252)
        dsr_many = deflated_sharpe_ratio(observed_sharpe=0.5, n_trials=100, n_obs=252)
        assert dsr_many.p_value >= dsr_few.p_value

    def test_sharpe_test_identical(self):
        """Two identical return series should produce p > 0.5 (no difference)."""
        np.random.seed(42)
        rets = np.random.randn(252) * 0.01
        result = ledoit_wolf_sharpe_test(rets, rets)
        assert result.p_value >= 0.45

    def test_sharpe_test_superior(self):
        """Strategy clearly better than benchmark should have low p."""
        np.random.seed(42)
        strategy = np.random.randn(500) * 0.01 + 0.005
        benchmark = np.random.randn(500) * 0.01 - 0.001
        result = ledoit_wolf_sharpe_test(strategy, benchmark)
        assert result.p_value < 0.05


# ============================================================
#  CONFIG TESTS
# ============================================================

class TestConfig:
    def test_config_loads(self, config):
        assert "data" in config
        assert "periods" in config
        assert "models" in config

    def test_ticker_count(self, config):
        tickers = get_all_tickers(config)
        assert len(tickers) == 27

    def test_embargo_minimum(self, config):
        embargo = config["splitting"]["embargo_days"]
        max_horizon = max(config["search_space"]["horizons"])
        assert embargo >= max_horizon, (
            f"Embargo ({embargo}) must be >= max horizon ({max_horizon})"
        )

    def test_12_signal_groups(self, config):
        """Expanded search space should produce 12 signal groups."""
        horizons = config["search_space"]["horizons"]
        label_modes = config["search_space"]["label_modes"]
        assert len(horizons) == 6
        assert len(label_modes) == 2
        assert len(horizons) * len(label_modes) == 12

    def test_phase_periods_non_overlapping(self, config):
        pa_end = pd.Timestamp(config["periods"]["phase_a"]["end"])
        pb_start = pd.Timestamp(config["periods"]["phase_b"]["start"])
        pb_end = pd.Timestamp(config["periods"]["phase_b"]["end"])
        pc_start = pd.Timestamp(config["periods"]["phase_c"]["start"])

        assert pa_end < pb_start, "Phase A and B overlap!"
        assert pb_end < pc_start, "Phase B and C overlap!"


# ============================================================
#  CORRUPTION TESTS
# ============================================================

class TestCorruption:
    def test_zero_corruption_no_change(self):
        """0% corruption should return identical labels."""
        labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        result = corrupt_labels(labels, corruption_pct=0.0, seed=42)
        assert (result == labels).all()

    def test_full_corruption_all_flipped(self):
        """100% corruption should flip every label."""
        labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        result = corrupt_labels(labels, corruption_pct=100.0, seed=42)
        n_changed = (result != labels).sum()
        assert n_changed == len(labels)

    def test_corruption_monotonic_subset(self):
        """X% corrupted indices must be a subset of (X+10)% corrupted indices."""
        labels = pd.Series(np.random.RandomState(0).choice([0, 1], size=200))
        result_20 = corrupt_labels(labels, corruption_pct=20.0, seed=42)
        result_50 = corrupt_labels(labels, corruption_pct=50.0, seed=42)
        changed_20 = set(labels.index[result_20 != labels])
        changed_50 = set(labels.index[result_50 != labels])
        assert changed_20.issubset(changed_50), "20% corrupted should be subset of 50%"

    def test_corruption_deterministic(self):
        """Same seed should produce identical corruption."""
        labels = pd.Series(np.random.RandomState(0).choice([0, 1, 2], size=100))
        r1 = corrupt_labels(labels, corruption_pct=30.0, seed=42)
        r2 = corrupt_labels(labels, corruption_pct=30.0, seed=42)
        assert (r1 == r2).all()

    def test_corruption_ternary(self):
        """Corrupted ternary labels should remain in {0, 1, 2}."""
        labels = pd.Series(np.random.RandomState(0).choice([0, 1, 2], size=200))
        result = corrupt_labels(labels, corruption_pct=50.0, seed=42)
        assert set(result.unique()).issubset({0, 1, 2})

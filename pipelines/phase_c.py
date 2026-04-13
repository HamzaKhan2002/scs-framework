"""
Phase C — Out-of-Sample Exploitation Pipeline.

Uses FROZEN models from Phase B on true OOS data.
Runs portfolio backtest + all statistical tests.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config import (
    load_config, get_all_tickers, get_feature_config,
    get_portfolio_config, ROOT_DIR,
)
from src.data.loader import load_universe
from src.features.engine import build_features, get_feature_matrix, FEATURE_COLS
from src.labeling.engine import make_labels
from src.models.classifiers import predict_proba
from src.backtest.portfolio_engine import run_portfolio_backtest, BacktestResult
from src.statistics.bootstrap import bootstrap_all_metrics
from src.statistics.deflated_sharpe import deflated_sharpe_ratio
from src.statistics.sharpe_tests import ledoit_wolf_sharpe_test


def generate_predictions(
    model: Any,
    data_oos: Dict[str, pd.DataFrame],
    horizon: int,
    label_mode: str,
    cfg: Dict,
) -> Dict[str, pd.DataFrame]:
    """
    Generate daily predictions for all tickers.

    Returns {ticker: DataFrame[proba_long, proba_short, direction]}.
    """
    feat_cfg = get_feature_config(cfg)
    lab_params = {}
    for ls in cfg["labeling"]["strategies"]:
        if ls["name"] == label_mode:
            lab_params = ls
            break

    predictions = {}

    for ticker, df in data_oos.items():
        try:
            df_feat = build_features(
                df,
                return_lags=feat_cfg.return_lags,
                vol_windows=feat_cfg.volatility_windows,
                atr_window=feat_cfg.atr_window,
                vol_z_window=feat_cfg.volume_z_window,
            )

            # We need labels for alignment only (they use future data,
            # but we don't use them for decisions — model is frozen)
            labels = make_labels(
                df_feat, horizon=horizon, mode=label_mode,
                threshold_mult=lab_params.get("threshold_mult", 0.5),
                vol_window=lab_params.get("vol_window", 20),
            )

            # Drop NaN from features (NOT from labels — we predict on all rows with features)
            df_feat_clean = df_feat.dropna(subset=FEATURE_COLS)
            if len(df_feat_clean) < 10:
                continue

            X, _ = get_feature_matrix(df_feat_clean)
            proba = predict_proba(model, X.values)
            n_classes = proba.shape[1]

            if n_classes == 2:
                # Binary: proba_long = P(class 1), no short
                pred_df = pd.DataFrame({
                    "proba_long": proba[:, 1],
                    "proba_short": 0.0,
                    "direction": "long",
                }, index=df_feat_clean.index)
            elif n_classes == 3:
                # Ternary: proba_long = P(class 2 = bullish), proba_short = P(class 0 = bearish)
                pred_df = pd.DataFrame({
                    "proba_long": proba[:, 2],
                    "proba_short": proba[:, 0],
                    "direction": "long",   # Default; backtest engine decides
                }, index=df_feat_clean.index)
            else:
                pred_df = pd.DataFrame({
                    "proba_long": proba[:, -1],
                    "proba_short": 0.0,
                    "direction": "long",
                }, index=df_feat_clean.index)

            predictions[ticker] = pred_df

        except Exception as e:
            print(f"    [WARN] {ticker}: {e}")
            continue

    return predictions


def compute_benchmark_returns(
    data_oos: Dict[str, pd.DataFrame],
) -> Dict[str, pd.Series]:
    """
    Compute benchmark daily returns.
    Equal-weight B&H: daily return = mean(daily return across all tickers).
    """
    daily_returns = {}
    for ticker, df in data_oos.items():
        daily_returns[ticker] = df["close"].pct_change()

    ret_df = pd.DataFrame(daily_returns).dropna()
    eq_weight = ret_df.mean(axis=1)  # Equal-weight portfolio
    spy_ret = daily_returns.get("SPY", eq_weight)

    return {
        "equal_weight": eq_weight,
        "SPY": spy_ret.loc[ret_df.index] if isinstance(spy_ret, pd.Series) else eq_weight,
    }


def run_phase_c(
    cfg: Dict = None,
    trained_models: Dict = None,
    approved_groups: Dict = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Phase C: Out-of-Sample Exploitation.

    For each approved signal:
        1. Generate predictions on OOS data
        2. Run portfolio backtest (SINGLE config)
        3. Run all statistical tests
        4. Generate comparison with benchmarks
    """
    if cfg is None:
        cfg = load_config()

    tickers = get_all_tickers(cfg)
    port_cfg = get_portfolio_config(cfg)

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE C — OUT-OF-SAMPLE EXPLOITATION")
        print("=" * 70)

    # Load OOS data
    data_oos = load_universe(
        tickers,
        cfg["periods"]["phase_c"]["start"],
        cfg["periods"]["phase_c"]["end"],
        cache_dir=Path(cfg["data"]["cache_dir"]),
    )
    if verbose:
        sample = list(data_oos.keys())[0]
        print(f"  OOS Period: {cfg['periods']['phase_c']['start']} → {cfg['periods']['phase_c']['end']}")
        print(f"  Tickers: {len(data_oos)}")
        print(f"  Trading days: {len(data_oos[sample])}")

    # Benchmarks
    benchmarks = compute_benchmark_returns(data_oos)

    all_results = {}

    groups_to_test = approved_groups or trained_models or {}

    for group_key, info in groups_to_test.items():
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Signal: {group_key}")

        model = info.get("model") if trained_models else None
        if model is None and trained_models:
            model = trained_models.get(group_key, {}).get("model")
        if model is None:
            print(f"    [SKIP] No model for {group_key}")
            continue

        horizon = info.get("horizon", int(group_key.split("d_")[0]))
        label_mode = info.get("label_mode", group_key.split("d_", 1)[1])

        # Generate predictions
        predictions = generate_predictions(model, data_oos, horizon, label_mode, cfg)
        if verbose:
            total_preds = sum(len(p) for p in predictions.values())
            print(f"    Predictions generated: {total_preds} across {len(predictions)} tickers")

        # Run portfolio backtest — use the signal group's horizon, not a fixed config value
        bt_result = run_portfolio_backtest(
            data=data_oos,
            predictions=predictions,
            horizon=horizon,
            max_positions=port_cfg.max_positions,
            capital_per_trade_pct=port_cfg.capital_per_trade_pct,
            max_exposure_per_ticker_pct=port_cfg.max_exposure_per_ticker_pct,
            transaction_cost_bps=port_cfg.transaction_cost_bps,
            take_profit_pct=port_cfg.take_profit_pct,
            stop_loss_pct=port_cfg.stop_loss_pct,
            allow_short=port_cfg.allow_short,
            initial_capital=port_cfg.initial_capital,
            min_proba_long=port_cfg.min_proba_long,
            min_proba_short=port_cfg.min_proba_short,
        )

        metrics = bt_result.metrics
        if verbose:
            print(f"    Return: {metrics['total_return_pct']:.2f}%")
            print(f"    Sharpe: {metrics['sharpe_ratio']:.4f}")
            print(f"    Max DD: {metrics['max_drawdown_pct']:.2f}%")
            print(f"    Trades: {metrics['n_trades']} (L:{metrics['n_long']} S:{metrics['n_short']})")
            print(f"    Win rate: {metrics['win_rate']:.1f}%")

        # ---- STATISTICAL TESTS ----
        if verbose:
            print(f"\n    Statistical Tests:")

        stats_results = {}
        daily_returns = bt_result.daily_equity.pct_change().dropna().values
        trade_rets = metrics.get("trade_returns", [])

        # 1. Bootstrap CI
        if len(trade_rets) > 10 and len(daily_returns) > 20:
            boot = bootstrap_all_metrics(
                trade_rets, daily_returns, horizon=horizon,
                n_bootstrap=cfg["statistics"]["bootstrap"]["n_samples"],
                seed=42,
            )
            stats_results["bootstrap"] = {}
            for k, v in boot.items():
                stats_results["bootstrap"][k] = {
                    "point": round(v.point_estimate, 4),
                    "ci_lower": round(v.ci_lower, 4),
                    "ci_upper": round(v.ci_upper, 4),
                    "std_error": round(v.std_error, 4),
                }
                if verbose:
                    print(f"      Bootstrap {k}: {v.point_estimate:.4f} "
                          f"[{v.ci_lower:.4f}, {v.ci_upper:.4f}]")

        # 2. Deflated Sharpe Ratio
        if len(daily_returns) > 20:
            from scipy.stats import skew, kurtosis
            dsr = deflated_sharpe_ratio(
                observed_sharpe=metrics["sharpe_ratio"],
                n_trials=cfg["statistics"]["deflated_sharpe"]["n_trials"],
                n_obs=len(daily_returns),
                skewness=skew(daily_returns),
                excess_kurtosis=kurtosis(daily_returns, fisher=True),
            )
            stats_results["deflated_sharpe"] = {
                "observed": dsr.observed_sharpe,
                "deflated": dsr.deflated_sharpe,
                "expected_max": dsr.expected_max_sharpe,
                "p_value": dsr.p_value,
                "significant": dsr.is_significant,
            }
            if verbose:
                sig = "YES" if dsr.is_significant else "NO"
                print(f"      DSR: {dsr.deflated_sharpe:.4f} (p={dsr.p_value:.4f}) Sig={sig}")

        # 3. Sharpe difference test vs benchmark
        bm_key = "equal_weight"
        if bm_key in benchmarks and len(daily_returns) > 20:
            bm_rets = benchmarks[bm_key].values
            min_len = min(len(daily_returns), len(bm_rets))
            st = ledoit_wolf_sharpe_test(
                daily_returns[:min_len],
                bm_rets[:min_len],
            )
            stats_results["sharpe_test_vs_bh"] = {
                "strategy_sharpe": st.strategy_sharpe,
                "benchmark_sharpe": st.benchmark_sharpe,
                "difference": st.sharpe_difference,
                "t_stat": st.t_statistic,
                "p_value": st.p_value,
                "significant": st.is_significant,
            }
            if verbose:
                sig = "YES" if st.is_significant else "NO"
                print(f"      Sharpe test vs B&H: Δ={st.sharpe_difference:.4f} "
                      f"(t={st.t_statistic:.4f}, p={st.p_value:.4f}) Sig={sig}")

        # 4. Cost sensitivity
        if verbose:
            print(f"    Cost sensitivity:")
        cost_sens = {}
        for bps in cfg["statistics"]["cost_sensitivity"]["bps_range"]:
            bt_cost = run_portfolio_backtest(
                data=data_oos, predictions=predictions,
                horizon=horizon,
                max_positions=port_cfg.max_positions,
                capital_per_trade_pct=port_cfg.capital_per_trade_pct,
                max_exposure_per_ticker_pct=port_cfg.max_exposure_per_ticker_pct,
                transaction_cost_bps=float(bps),
                take_profit_pct=port_cfg.take_profit_pct,
                stop_loss_pct=port_cfg.stop_loss_pct,
                allow_short=port_cfg.allow_short,
                initial_capital=port_cfg.initial_capital,
                min_proba_long=port_cfg.min_proba_long,
                min_proba_short=port_cfg.min_proba_short,
            )
            cost_sens[bps] = {
                "sharpe": bt_cost.metrics["sharpe_ratio"],
                "return": bt_cost.metrics["total_return_pct"],
            }
            if verbose:
                print(f"      {bps:3d} bps → Sharpe={bt_cost.metrics['sharpe_ratio']:.4f} "
                      f"Return={bt_cost.metrics['total_return_pct']:.2f}%")

        stats_results["cost_sensitivity"] = cost_sens

        all_results[group_key] = {
            "metrics": {k: v for k, v in metrics.items() if k != "trade_returns"},
            "statistics": stats_results,
            "n_predictions": sum(len(p) for p in predictions.values()),
        }

    # Save results
    results_dir = ROOT_DIR / "results" / "phase_c"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"phase_c_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    if verbose:
        print(f"\n  Results saved: {out_path}")
        print("\n  PHASE C SUMMARY")
        print("  " + "─" * 50)
        for gk, res in all_results.items():
            m = res["metrics"]
            print(f"  {gk:35s} Sharpe={m['sharpe_ratio']:.4f} "
                  f"Return={m['total_return_pct']:.2f}% DD={m['max_drawdown_pct']:.2f}%")

    return {
        "results": all_results,
        "timestamp": ts,
    }


if __name__ == "__main__":
    run_phase_c(verbose=True)

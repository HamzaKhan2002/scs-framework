"""
Phase A — Signal Discovery Pipeline.

Evaluates signal configurations across multiple models, seeds, periods, and assets.
Signal groups are (horizon, label_mode) containing ALL model types.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config import load_config, get_all_tickers, get_feature_config, ROOT_DIR
from src.data.loader import load_universe
from src.features.engine import build_features, get_feature_matrix, FEATURE_COLS
from src.labeling.engine import make_labels
from src.models.classifiers import train_model, predict_proba
from src.validation.temporal_split import purged_temporal_split, split_into_sub_periods
from src.validation.scs_a import compute_scs_a
from src.backtest.portfolio_engine import run_naive_backtest


def corrupt_labels(labels: pd.Series, corruption_pct: float, seed: int = 0) -> pd.Series:
    """
    Flip corruption_pct% of labels to a random alternative class.
    Uses deterministic seed so X% corrupted is a subset of (X+10)% corrupted.
    """
    if corruption_pct <= 0:
        return labels
    rng = np.random.RandomState(seed)
    clean = labels.dropna()
    n_corrupt = int(len(clean) * corruption_pct / 100)
    if n_corrupt == 0:
        return labels
    all_indices = clean.index.tolist()
    rng.shuffle(all_indices)
    corrupt_indices = all_indices[:n_corrupt]
    unique_classes = sorted(clean.unique())
    result = labels.copy()
    for idx in corrupt_indices:
        current = result.loc[idx]
        alternatives = [c for c in unique_classes if c != current]
        if alternatives:
            result.loc[idx] = rng.choice(alternatives)
    return result


def run_phase_a(cfg: Dict = None, verbose: bool = True, label_corruption_pct: float = 0.0, corruption_seed: int = 42) -> Dict[str, Any]:
    """
    Run Phase A: Signal Discovery.

    For each signal group (horizon × label_mode):
        For each model_type:
            For each seed:
                For each sub_period:
                    For each ticker:
                        → train, predict, backtest, record

    Returns dict with SCS-A for each group and all raw results.
    """
    if cfg is None:
        cfg = load_config()

    tickers = get_all_tickers(cfg)
    feat_cfg = get_feature_config(cfg)
    search = cfg["search_space"]
    split_cfg = cfg["splitting"]
    scs_cfg = cfg["scs_a"]

    # Load Phase A data
    if verbose:
        print("=" * 70)
        print("PHASE A — SIGNAL DISCOVERY")
        print("=" * 70)
        print(f"  Period: {cfg['periods']['phase_a']['start']} → {cfg['periods']['phase_a']['end']}")
        print(f"  Tickers: {len(tickers)}")
        if label_corruption_pct > 0:
            print(f"  Label corruption: {label_corruption_pct}%")

    data = load_universe(
        tickers,
        cfg["periods"]["phase_a"]["start"],
        cfg["periods"]["phase_a"]["end"],
        cache_dir=Path(cfg["data"]["cache_dir"]),
    )
    if verbose:
        sample_ticker = list(data.keys())[0]
        print(f"  Trading days: {len(data[sample_ticker])}")

    # Define sub-periods
    sample_dates = data[list(data.keys())[0]].index
    sub_periods = split_into_sub_periods(
        cfg["periods"]["phase_a"]["start"],
        cfg["periods"]["phase_a"]["end"],
        split_cfg["phase_a"]["n_sub_periods"],
        sample_dates,
    )
    if verbose:
        for i, (s, e) in enumerate(sub_periods):
            print(f"  Sub-period P{i+1}: {s.date()} → {e.date()}")

    all_group_results = {}

    # Signal groups: (horizon, label_mode) — contains ALL model types
    for horizon in search["horizons"]:
        for label_mode in search["label_modes"]:
            group_key = f"{horizon}d_{label_mode}"
            if verbose:
                print(f"\n{'─'*60}")
                print(f"  Signal Group: {group_key}")
                print(f"{'─'*60}")

            group_records = []
            group_trades = []

            for model_type in search["model_types"]:
                model_cfg = cfg["models"].get(model_type, {})

                for seed in search["seeds"]:
                    for pi, (p_start, p_end) in enumerate(sub_periods):
                        period_name = f"P{pi+1}"

                        for ticker in data:
                            df = data[ticker].copy()
                            # Filter to sub-period
                            mask = (df.index >= p_start) & (df.index <= p_end)
                            df_period = df[mask]

                            if len(df_period) < 60:
                                continue

                            # Build features
                            df_feat = build_features(
                                df_period,
                                return_lags=feat_cfg.return_lags,
                                vol_windows=feat_cfg.volatility_windows,
                                atr_window=feat_cfg.atr_window,
                                vol_z_window=feat_cfg.volume_z_window,
                            )

                            # Get label config from labeling strategies
                            lab_params = {}
                            for ls in cfg["labeling"]["strategies"]:
                                if ls["name"] == label_mode:
                                    lab_params = ls
                                    break

                            labels = make_labels(
                                df_feat, horizon=horizon, mode=label_mode,
                                threshold_mult=lab_params.get("threshold_mult", 0.5),
                                vol_window=lab_params.get("vol_window", 20),
                            )

                            # Apply label corruption if specified
                            if label_corruption_pct > 0:
                                labels = corrupt_labels(labels, label_corruption_pct, seed=corruption_seed)

                            # Align: drop NaN from both
                            df_feat["_label"] = labels
                            df_clean = df_feat.dropna(subset=FEATURE_COLS + ["_label"])

                            if len(df_clean) < 40:
                                continue

                            X, _ = get_feature_matrix(df_clean)
                            y = df_clean["_label"].values.astype(int)

                            # Purged temporal split
                            try:
                                split = purged_temporal_split(
                                    df_clean.index,
                                    train_ratio=split_cfg["phase_a"]["train_ratio"],
                                    embargo_days=split_cfg["embargo_days"],
                                )
                            except ValueError:
                                continue

                            X_train = X.loc[split.train_idx].values
                            y_train = y[df_clean.index.isin(split.train_idx)]
                            X_test = X.loc[split.test_idx].values
                            y_test = y[df_clean.index.isin(split.test_idx)]

                            if len(X_train) < 20 or len(X_test) < 10:
                                continue

                            # Train model
                            try:
                                model = train_model(
                                    X_train, y_train,
                                    model_type=model_type,
                                    config=model_cfg,
                                    seed=seed,
                                    feature_names=FEATURE_COLS,
                                )
                            except Exception:
                                continue

                            # Predict using predict_proba (NOT predict!)
                            try:
                                proba = predict_proba(model, X_test)
                            except Exception:
                                continue

                            n_classes = proba.shape[1]
                            if n_classes == 2:
                                preds = proba[:, 1]
                            elif n_classes == 3:
                                preds = proba[:, 2]  # bullish class
                            else:
                                preds = proba[:, -1]

                            # Get price arrays for test period
                            test_df = df_clean.loc[split.test_idx]
                            close_arr = test_df["close"].values
                            open_arr = test_df["open"].values

                            # Naive backtest
                            bt = run_naive_backtest(
                                preds, close_arr, open_arr,
                                horizon=horizon,
                                cost_bps=scs_cfg.get("cost_bps", 5.0),
                            )

                            group_records.append({
                                "ticker": ticker,
                                "period": period_name,
                                "seed": seed,
                                "model_type": model_type,
                                "sharpe": bt["sharpe"],
                                "total_return": bt["total_return"],
                                "n_trades": bt["n_trades"],
                                "win_rate": bt["win_rate"],
                            })
                            group_trades.extend(bt["trades"])

            if verbose:
                print(f"    Runs: {len(group_records)} | Trades: {len(group_trades)}")

            if len(group_records) == 0:
                scs_result = {"SCS_A": 0.0, "verdict": "NO_DATA"}
            else:
                df_results = pd.DataFrame(group_records)
                scs_result = compute_scs_a(
                    df_results,
                    group_trades,
                    weights=scs_cfg["weights"],
                    hard_gates=scs_cfg["hard_gates"],
                )

            scs_result["group_key"] = group_key
            scs_result["horizon"] = horizon
            scs_result["label_mode"] = label_mode
            scs_result["n_runs"] = len(group_records)
            scs_result["n_trades_total"] = len(group_trades)
            all_group_results[group_key] = scs_result

            if verbose:
                print(f"    SCS-A = {scs_result['SCS_A']:.4f} → {scs_result['verdict']}")
                for k in ["S_time", "S_asset", "S_model", "S_seed", "S_dist"]:
                    if k in scs_result:
                        print(f"      {k} = {scs_result[k]:.4f}")

    # Save results
    results_dir = ROOT_DIR / "results" / "phase_a"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"phase_a_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_group_results, f, indent=2, default=str)

    if verbose:
        print(f"\n  Results saved: {out_path}")
        print("\n  PHASE A SUMMARY")
        print("  " + "─" * 50)
        for gk, res in all_group_results.items():
            print(f"  {gk:35s} SCS-A={res['SCS_A']:.4f}  {res['verdict']}")

    return {
        "group_results": all_group_results,
        "approved": {k: v for k, v in all_group_results.items()
                     if v.get("verdict") == "PHASE_B_APPROVED"},
        "timestamp": ts,
    }


if __name__ == "__main__":
    result = run_phase_a(verbose=True)

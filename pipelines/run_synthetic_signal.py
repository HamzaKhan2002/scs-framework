"""
Synthetic Signal Experiment — Injects a known-profitable signal and tests SCS response.

Uses the formula: y_synthetic = sign(actual_5d_return + k * std(actual_5d_return) * randn)
where k controls noise level. Three levels tested:
  k=1.0 (strong signal, ~75% accuracy)
  k=2.0 (medium signal, ~60-65% accuracy)
  k=4.0 (weak signal, ~53-55% accuracy)

Runs Phase A only (SCS-A scoring) for each noise level on the 5d binary horizon.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config import load_config, get_all_tickers, get_feature_config, ROOT_DIR
from src.data.loader import load_universe
from src.features.engine import build_features, get_feature_matrix, FEATURE_COLS
from src.models.classifiers import train_model, predict_proba
from src.validation.temporal_split import purged_temporal_split, split_into_sub_periods
from src.validation.scs_a import compute_scs_a
from src.backtest.portfolio_engine import run_naive_backtest


def make_synthetic_labels(df: pd.DataFrame, horizon: int, k: float, seed: int = 42) -> tuple:
    """
    Generate synthetic binary labels with controlled noise.

    y = sign(actual_forward_return + k * std(actual_forward_return) * noise)

    k=0: perfect signal (100% accuracy vs true label)
    k→∞: pure noise (~50% accuracy)
    """
    # Compute actual forward returns
    forward_ret = df["close"].shift(-horizon) / df["open"].shift(-1) - 1

    rng = np.random.RandomState(seed)
    noise = rng.randn(len(forward_ret))

    std_ret = forward_ret.std()
    noisy_ret = forward_ret + k * std_ret * noise

    # Binary label: 1 if positive, 0 if negative
    labels = (noisy_ret > 0).astype(float)
    labels[forward_ret.isna()] = np.nan

    return labels, forward_ret


def run_synthetic_phase_a(cfg, data, k_value, horizon=5, verbose=True):
    """Run Phase A with synthetic labels at noise level k."""
    tickers = list(data.keys())
    feat_cfg = get_feature_config(cfg)
    split_cfg = cfg["splitting"]
    scs_a_cfg = cfg["scs_a"]

    model_types = cfg["search_space"]["model_types"]
    seeds = cfg["search_space"]["seeds"]

    group_key = f"5d_synthetic_k{k_value}"
    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Synthetic signal: k={k_value} (5d binary)")

    # Prepare sub-periods
    sample_df = data[tickers[0]]
    sub_periods = split_into_sub_periods(
        cfg["periods"]["phase_a"]["start"],
        cfg["periods"]["phase_a"]["end"],
        split_cfg["phase_a"]["n_sub_periods"],
        sample_df.index,
    )

    all_results = []
    all_trades = []
    accuracy_samples = []

    for pi, (p_start, p_end) in enumerate(sub_periods):
        period_name = f"P{pi+1}"

        for ticker in tickers:
            df = data[ticker]
            mask = (df.index >= p_start) & (df.index <= p_end)
            df_period = df[mask]
            if len(df_period) < 60:
                continue

            df_feat = build_features(
                df_period,
                return_lags=feat_cfg.return_lags,
                vol_windows=feat_cfg.volatility_windows,
                atr_window=feat_cfg.atr_window,
                vol_z_window=feat_cfg.volume_z_window,
            )

            # Generate synthetic labels
            synth_labels, actual_ret = make_synthetic_labels(df_feat, horizon, k_value, seed=42)
            true_labels = (actual_ret > 0).astype(float)
            true_labels[actual_ret.isna()] = np.nan

            df_feat["_label"] = synth_labels
            df_clean = df_feat.dropna(subset=FEATURE_COLS + ["_label"])
            if len(df_clean) < 40:
                continue

            # Measure synthetic label accuracy vs true
            valid_mask = true_labels.loc[df_clean.index].notna()
            if valid_mask.sum() > 0:
                acc = (synth_labels.loc[df_clean.index][valid_mask] ==
                       true_labels.loc[df_clean.index][valid_mask]).mean()
                accuracy_samples.append(acc)

            X, _ = get_feature_matrix(df_clean)
            y = df_clean["_label"].values.astype(int)

            split = purged_temporal_split(
                df_clean.index,
                train_ratio=split_cfg["phase_a"]["train_ratio"],
                embargo_days=split_cfg["embargo_days"],
            )
            if len(split.train_idx) < 20 or len(split.test_idx) < 10:
                continue

            X_train = X.loc[split.train_idx].values
            y_train = y[df_clean.index.isin(split.train_idx)]
            X_test = X.loc[split.test_idx].values

            for model_type in model_types:
                model_cfg = cfg["models"][model_type]
                for seed in seeds:
                    try:
                        model = train_model(
                            X_train, y_train,
                            model_type=model_type, config=model_cfg,
                            seed=seed, feature_names=FEATURE_COLS,
                        )
                    except Exception:
                        continue

                    try:
                        proba = predict_proba(model, X_test)
                    except Exception:
                        continue

                    preds = proba[:, 1] if proba.shape[1] == 2 else proba[:, -1]
                    test_df = df_clean.loc[split.test_idx]
                    close_arr = test_df["close"].values
                    open_arr = test_df["open"].values

                    bt = run_naive_backtest(preds, close_arr, open_arr, horizon=horizon)
                    all_trades.extend(bt["trades"])
                    all_results.append({
                        "period": period_name,
                        "ticker": ticker,
                        "model_type": model_type,
                        "seed": seed,
                        "sharpe": bt["sharpe"],
                        "n_trades": bt["n_trades"],
                        "total_return": bt["total_return"],
                        "win_rate": bt["win_rate"],
                    })
                    all_trades.extend(bt["trades"])

    if not all_results:
        return {"SCS_A": 0.0, "verdict": "NO_DATA", "n_runs": 0}

    df_results = pd.DataFrame(all_results)

    scs_result = compute_scs_a(
        df_results, all_trades,
        weights=scs_a_cfg["weights"],
        hard_gates=scs_a_cfg["hard_gates"],
    )

    mean_accuracy = np.mean(accuracy_samples) if accuracy_samples else 0
    scs_result["mean_label_accuracy"] = round(mean_accuracy, 4)
    scs_result["n_runs"] = len(all_results)
    scs_result["k_value"] = k_value
    scs_result["group_key"] = group_key

    if verbose:
        print(f"    Mean synthetic label accuracy: {mean_accuracy:.1%}")
        print(f"    N runs: {len(all_results)}")
        print(f"    SCS-A = {scs_result['SCS_A']:.4f} → {scs_result['verdict']}")
        for comp in ["S_time", "S_asset", "S_model", "S_seed", "S_dist"]:
            if comp in scs_result:
                print(f"      {comp} = {scs_result[comp]:.4f}")

    return scs_result


def run_synthetic_experiment():
    cfg = load_config()
    tickers = get_all_tickers(cfg)

    print("=" * 70)
    print("SYNTHETIC SIGNAL EXPERIMENT")
    print("Injecting known-profitable signal at three noise levels")
    print("=" * 70)

    # Load Phase A data
    data = load_universe(
        tickers,
        cfg["periods"]["phase_a"]["start"],
        cfg["periods"]["phase_a"]["end"],
        cache_dir=Path(cfg["data"]["cache_dir"]),
    )
    print(f"  Tickers loaded: {len(data)}")
    print(f"  Period: {cfg['periods']['phase_a']['start']} → {cfg['periods']['phase_a']['end']}")

    k_values = [1.0, 2.0, 4.0]
    all_results = {}

    for k in k_values:
        t0 = time.time()
        result = run_synthetic_phase_a(cfg, data, k, horizon=5, verbose=True)
        elapsed = time.time() - t0
        result["time_seconds"] = round(elapsed, 1)
        all_results[str(k)] = result

        # Save incrementally
        out_path = ROOT_DIR / "results" / "experiments" / "experiment_5_synthetic_signal.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  k={k} completed in {elapsed:.1f}s — saved to {out_path}")

    # Also run baseline (no synthetic, real labels) for comparison
    print(f"\n{'─'*60}")
    print("  Baseline: Real 5d binary labels (k=N/A)")
    # Load from experiment_1
    pa_path = ROOT_DIR / "results" / "experiments" / "experiment_1_expanded.json"
    with open(pa_path) as f:
        exp1 = json.load(f)
    baseline = exp1["phase_a"]["5d_directional_binary"]
    all_results["baseline"] = baseline
    print(f"    SCS-A = {baseline['SCS_A']:.4f} → {baseline['verdict']}")

    # Save final
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 70)
    print("SYNTHETIC SIGNAL SUMMARY")
    print("=" * 70)
    print(f"{'k':>6} {'Accuracy':>10} {'SCS-A':>8} {'S_time':>8} {'S_asset':>8} "
          f"{'S_model':>8} {'S_seed':>8} {'Verdict':>15}")
    print("-" * 85)
    for k_str, res in all_results.items():
        acc = res.get("mean_label_accuracy", "-")
        acc_str = f"{acc:.1%}" if isinstance(acc, float) else "-"
        print(f"{k_str:>6} {acc_str:>10} {res['SCS_A']:>8.4f} "
              f"{res.get('S_time', 0):>8.4f} {res.get('S_asset', 0):>8.4f} "
              f"{res.get('S_model', 0):>8.4f} {res.get('S_seed', 0):>8.4f} "
              f"{res['verdict']:>15}")


if __name__ == "__main__":
    run_synthetic_experiment()

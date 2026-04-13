"""
Oracle Power Analysis — computes detection power at each threshold.

For each noise level k, runs the oracle injection experiment N times
with different RNG seeds and computes P(SCS-A >= tau | k).

Combined with FDR results, this gives a full calibration (ROC-like).

Usage:
    # Local test
    PYTHONIOENCODING=utf-8 python pipelines/run_power_analysis.py --n-seeds 3 --noise-levels 0.5,1.5,4.0

    # VM run
    PYTHONIOENCODING=utf-8 python pipelines/run_power_analysis.py --n-seeds 100 --workers 16
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import load_config, get_all_tickers, get_feature_config
from src.data.loader import load_universe
from src.features.engine import build_features, get_feature_matrix, FEATURE_COLS
from src.labeling.engine import make_labels, compute_forward_return
from src.models.classifiers import train_model, predict_proba
from src.validation.temporal_split import purged_temporal_split, split_into_sub_periods
from src.validation.scs_a import compute_scs_a
from src.backtest.portfolio_engine import run_naive_backtest

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DEFAULT_NOISE_LEVELS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]


def inject_oracle_feature(df, horizon, k, seed):
    """Inject oracle feature: x_oracle = r_t(horizon) + k * sigma(r) * epsilon."""
    rng = np.random.RandomState(seed)
    fwd_ret = compute_forward_return(df, horizon=horizon)
    sigma = fwd_ret.std()
    noise = rng.normal(0, 1, size=len(fwd_ret))
    oracle = fwd_ret + k * sigma * noise
    return oracle


def run_oracle_seed(args):
    """Run Phase A with oracle feature for one (k, seed) combination."""
    k, seed, cfg = args

    try:
        tickers = get_all_tickers(cfg)
        feat_cfg = get_feature_config(cfg)
        search = cfg["search_space"]
        split_cfg = cfg["splitting"]
        scs_cfg = cfg["scs_a"]

        data = load_universe(
            tickers,
            cfg["periods"]["phase_a"]["start"],
            cfg["periods"]["phase_a"]["end"],
            cache_dir=Path(cfg["data"]["cache_dir"]),
        )

        sample_dates = data[list(data.keys())[0]].index
        sub_periods = split_into_sub_periods(
            cfg["periods"]["phase_a"]["start"],
            cfg["periods"]["phase_a"]["end"],
            split_cfg["phase_a"]["n_sub_periods"],
            sample_dates,
        )

        # Only run for 5d binary (the target for oracle injection)
        horizon = 5
        label_mode = "directional_binary"
        group_key = f"{horizon}d_{label_mode}"

        lab_params = {}
        for ls in cfg["labeling"]["strategies"]:
            if ls["name"] == label_mode:
                lab_params = ls
                break

        group_records = []
        group_trades = []

        for model_type in search["model_types"]:
            model_cfg = cfg["models"].get(model_type, {})
            for s in search["seeds"]:
                for pi, (p_start, p_end) in enumerate(sub_periods):
                    for ticker in data:
                        df = data[ticker].copy()
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

                        # Inject oracle feature
                        oracle = inject_oracle_feature(df_feat, horizon, k, seed + hash(ticker) % 10000)
                        df_feat["oracle_feature"] = oracle
                        feature_cols = FEATURE_COLS + ["oracle_feature"]

                        labels = make_labels(
                            df_feat, horizon=horizon, mode=label_mode,
                            threshold_mult=lab_params.get("threshold_mult", 0.5),
                            vol_window=lab_params.get("vol_window", 20),
                        )

                        df_feat["_label"] = labels
                        df_clean = df_feat.dropna(subset=feature_cols + ["_label"])
                        if len(df_clean) < 40:
                            continue

                        X = df_clean[feature_cols]
                        y = df_clean["_label"].values.astype(int)

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

                        if len(X_train) < 20 or len(X_test) < 10:
                            continue

                        try:
                            model = train_model(
                                X_train, y_train,
                                model_type=model_type,
                                config=model_cfg,
                                seed=s,
                                feature_names=feature_cols,
                            )
                            proba = predict_proba(model, X_test)
                        except Exception:
                            continue

                        preds = proba[:, 1] if proba.shape[1] == 2 else proba[:, -1]
                        test_df = df_clean.loc[split.test_idx]

                        bt = run_naive_backtest(
                            preds, test_df["close"].values, test_df["open"].values,
                            horizon=horizon, cost_bps=5.0,
                        )

                        group_records.append({
                            "ticker": ticker, "period": f"P{pi+1}",
                            "seed": s, "model_type": model_type,
                            "sharpe": bt["sharpe"], "total_return": bt["total_return"],
                            "n_trades": bt["n_trades"], "win_rate": bt["win_rate"],
                        })
                        group_trades.extend(bt["trades"])

        if not group_records:
            return {"k": k, "seed": seed, "status": "no_data", "SCS_A": 0.0}

        import pandas as pd
        df_results = pd.DataFrame(group_records)
        scs_result = compute_scs_a(
            df_results, group_trades,
            weights=scs_cfg["weights"],
            hard_gates=scs_cfg["hard_gates"],
        )

        return {
            "k": k, "seed": seed, "status": "ok",
            "SCS_A": scs_result.get("SCS_A", 0.0),
            "verdict": scs_result.get("verdict", "UNKNOWN"),
            "mean_sharpe": scs_result.get("mean_sharpe", 0.0),
        }

    except Exception as e:
        return {"k": k, "seed": seed, "status": "error", "error": str(e)}


def compute_power(results, noise_levels):
    """Compute power at each (k, tau) combination."""
    summary = {}
    for k in noise_levels:
        k_results = [r for r in results if r["k"] == k and r["status"] == "ok"]
        n_valid = len(k_results)
        if n_valid == 0:
            continue

        scs_scores = [r["SCS_A"] for r in k_results]
        k_power = {}
        for tau in THRESHOLDS:
            n_pass = sum(1 for s in scs_scores if s >= tau)
            k_power[f"tau_{tau:.2f}"] = round(n_pass / n_valid, 4)

        summary[f"k_{k}"] = {
            "k": k,
            "n_seeds": n_valid,
            "mean_scs_a": round(float(np.mean(scs_scores)), 4),
            "std_scs_a": round(float(np.std(scs_scores)), 4),
            "mean_sharpe": round(float(np.mean([r["mean_sharpe"] for r in k_results])), 4),
            "power": k_power,
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Oracle Power Analysis")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--noise-levels", type=str, default=None,
                        help="Comma-separated noise levels (default: 0.1,0.3,...,8.0)")
    args = parser.parse_args()

    noise_levels = DEFAULT_NOISE_LEVELS
    if args.noise_levels:
        noise_levels = [float(x) for x in args.noise_levels.split(",")]

    cfg = load_config()
    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "power_analysis_raw.json"
    summary_path = out_dir / "power_analysis_summary.json"

    # Build task list
    tasks = [(k, seed, cfg) for k in noise_levels for seed in range(args.n_seeds)]

    print("=" * 70)
    print("ORACLE POWER ANALYSIS")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Seeds per level: {args.n_seeds}")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Workers: {args.workers}")
    print("=" * 70)

    # Load existing results for resumption
    existing = []
    if raw_path.exists():
        with open(raw_path) as f:
            existing = json.load(f)
        completed = {(r["k"], r["seed"]) for r in existing if r["status"] == "ok"}
        tasks = [(k, s, c) for k, s, c in tasks if (k, s) not in completed]
        print(f"  Loaded {len(completed)} completed, {len(tasks)} remaining")

    all_results = list(existing)
    t0 = time.time()

    if args.workers <= 1:
        for i, task in enumerate(tasks):
            k, seed, _ = task
            print(f"\n  [{i+1}/{len(tasks)}] k={k}, seed={seed}...")
            result = run_oracle_seed(task)
            all_results.append(result)
            print(f"    SCS-A = {result.get('SCS_A', 'N/A')}")

            if (i + 1) % 5 == 0:
                with open(raw_path, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)
    else:
        n_workers = min(args.workers, cpu_count(), len(tasks))
        with Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(run_oracle_seed, tasks)):
                all_results.append(result)
                print(f"  [{i+1}/{len(tasks)}] k={result['k']}, seed={result['seed']}: SCS-A={result.get('SCS_A', 'N/A')}")
                if (i + 1) % 10 == 0:
                    with open(raw_path, "w") as f:
                        json.dump(all_results, f, indent=2, default=str)

    # Save raw
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Compute and save summary
    ok_results = [r for r in all_results if r["status"] == "ok"]
    summary = compute_power(ok_results, noise_levels)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print power table
    print(f"\n{'='*70}")
    print("POWER ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'k':>5s} {'n':>4s} {'Mean SCS-A':>11s} {'P@0.50':>7s} {'P@0.70':>7s} {'P@0.85':>7s} {'P@0.95':>7s}")
    for kk in sorted(summary.keys()):
        s = summary[kk]
        p = s["power"]
        print(f"  {s['k']:>5.1f} {s['n_seeds']:>4d} {s['mean_scs_a']:>11.4f} "
              f"{p.get('tau_0.50', 0):>7.4f} {p.get('tau_0.70', 0):>7.4f} "
              f"{p.get('tau_0.85', 0):>7.4f} {p.get('tau_0.95', 0):>7.4f}")

    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

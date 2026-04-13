"""
Oracle Feature Injection Experiment.

Proves the SCS framework can approve a signal the model can actually learn
and reject one it cannot, by injecting a synthetic feature (not label) into
the feature matrix.

Oracle feature:
    oracle = actual_5d_forward_return + k * std(actual_5d_forward_return) * noise

The oracle is directly correlated with the real label by construction.
Labels are unchanged (real directional_binary 5d).
Only the feature matrix gains one extra column.

Noise levels: k in {0.5, 1.5, 4.0}
  k=0.5 → oracle almost perfectly predicts the label
  k=1.5 → realistic noise level
  k=4.0 → oracle is mostly noise

WARNING: The oracle feature uses actual forward returns — this is intentional
lookahead for a controlled validation experiment, NOT a deployable strategy.
This tests the scoring and gating machinery of the SCS framework, not the
predictive power of the feature set.

Usage: PYTHONIOENCODING=utf-8 python pipelines/synthetic_signal_experiment.py
"""

import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config import load_config, get_all_tickers, get_feature_config, ROOT_DIR
from src.data.loader import load_universe
from src.features.engine import build_features, get_feature_matrix, FEATURE_COLS
from src.labeling.engine import make_labels
from src.models.classifiers import train_model, predict_proba
from src.validation.temporal_split import purged_temporal_split, split_into_sub_periods
from src.validation.scs_a import compute_scs_a, _robustness_score
from src.backtest.portfolio_engine import run_naive_backtest

# Extended feature list: 11 standard + oracle
ORACLE_FEATURE_COLS = FEATURE_COLS + ["oracle"]


# ── Oracle feature generation ──────────────────────────────────

def generate_oracle_feature(df, horizon=5, k=0.5, seed=42):
    """
    Generate an oracle feature correlated with the real forward return.

    oracle = actual_5d_fwd_return + k * std(actual_5d_fwd_return) * noise

    WARNING: Uses actual forward returns (lookahead). For controlled
    validation experiments only — not a deployable feature.

    Returns a Series aligned with df.index. NaN where forward return
    is unavailable.
    """
    rng = np.random.RandomState(seed)
    close = df["close"]
    fwd_ret = close.shift(-horizon) / close - 1.0
    std_ret = fwd_ret.std()
    noise = pd.Series(rng.randn(len(df)), index=df.index)
    oracle = fwd_ret + k * std_ret * noise
    return oracle


def calibrate_oracle_accuracy(data, horizon=5):
    """
    For each k, report: correlation between oracle and actual forward return,
    and accuracy of sign(oracle) vs sign(actual forward return).
    """
    k_values = [0.5, 1.5, 4.0]

    print("\n  Oracle feature calibration (mean across all tickers):")
    print(f"  {'k':>5s} {'Corr(oracle,ret)':>18s} {'sign accuracy':>15s}")
    print(f"  {'-'*42}")

    results = {}
    for k in k_values:
        corrs, accs = [], []
        for ticker, df in data.items():
            oracle = generate_oracle_feature(df, horizon=horizon, k=k)
            fwd_ret = df["close"].shift(-horizon) / df["close"] - 1.0
            valid = oracle.notna() & fwd_ret.notna()
            if valid.sum() < 20:
                continue
            corrs.append(oracle[valid].corr(fwd_ret[valid]))
            sign_match = ((oracle[valid] > 0) == (fwd_ret[valid] > 0)).mean()
            accs.append(sign_match)
        mean_corr = np.mean(corrs)
        mean_acc = np.mean(accs)
        print(f"  k={k:.1f} {mean_corr:18.4f} {mean_acc:15.1%}")
        results[k] = {"corr": mean_corr, "accuracy": mean_acc}

    return results


# ── Phase A with oracle feature ────────────────────────────────

def run_phase_a_oracle(cfg, data, sub_periods, k_value, verbose=True):
    """
    Run Phase A with an extra oracle feature injected into the feature matrix.

    Labels are real directional_binary 5d (unchanged).
    The oracle feature is generated with a fixed seed=42 (independent of model seed).
    """
    feat_cfg = get_feature_config(cfg)
    search = cfg["search_space"]
    split_cfg = cfg["splitting"]
    scs_cfg = cfg["scs_a"]

    horizon = 5
    group_key = f"oracle_k{k_value}"

    # Get labeling params for directional_binary
    lab_params = {}
    for ls in cfg["labeling"]["strategies"]:
        if ls["name"] == "directional_binary":
            lab_params = ls
            break

    if verbose:
        print(f"\n  Signal: {group_key}")

    group_records = []
    group_trades = []

    for model_type in search["model_types"]:
        model_cfg = cfg["models"].get(model_type, {})

        for seed in search["seeds"]:
            for pi, (p_start, p_end) in enumerate(sub_periods):
                period_name = f"P{pi+1}"

                for ticker in data:
                    df = data[ticker].copy()
                    mask = (df.index >= p_start) & (df.index <= p_end)
                    df_period = df[mask]

                    if len(df_period) < 60:
                        continue

                    # Build standard features
                    df_feat = build_features(
                        df_period,
                        return_lags=feat_cfg.return_lags,
                        vol_windows=feat_cfg.volatility_windows,
                        atr_window=feat_cfg.atr_window,
                        vol_z_window=feat_cfg.volume_z_window,
                    )

                    # Inject oracle feature (fixed seed=42, independent of model seed)
                    df_feat["oracle"] = generate_oracle_feature(
                        df_period, horizon=horizon, k=k_value, seed=42,
                    )

                    # Real labels (unchanged)
                    labels = make_labels(
                        df_period, horizon=horizon, mode="directional_binary",
                        threshold_mult=lab_params.get("threshold_mult", 0.5),
                        vol_window=lab_params.get("vol_window", 20),
                    )

                    df_feat["_label"] = labels
                    df_clean = df_feat.dropna(
                        subset=ORACLE_FEATURE_COLS + ["_label"],
                    )

                    if len(df_clean) < 40:
                        continue

                    # Extended feature matrix: 11 standard + oracle
                    X = df_clean[ORACLE_FEATURE_COLS]
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

                    try:
                        model = train_model(
                            X_train, y_train,
                            model_type=model_type,
                            config=model_cfg,
                            seed=seed,
                            feature_names=ORACLE_FEATURE_COLS,
                        )
                    except Exception:
                        continue

                    try:
                        proba = predict_proba(model, X_test)
                    except Exception:
                        continue

                    n_classes = proba.shape[1]
                    preds = proba[:, 1] if n_classes == 2 else proba[:, -1]

                    test_df = df_clean.loc[split.test_idx]
                    bt = run_naive_backtest(
                        preds, test_df["close"].values, test_df["open"].values,
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
    scs_result["k_value"] = k_value
    scs_result["n_runs"] = len(group_records)

    if verbose:
        print(f"    SCS-A = {scs_result.get('SCS_A', 0.0):.4f} -> {scs_result.get('verdict', 'N/A')}")

    return scs_result


def print_component_detail(cfg, data, feat_cfg, split_cfg, scs_cfg, sub_periods, k):
    """Re-run and print per-period, per-ticker, per-seed Sharpe detail."""
    search = cfg["search_space"]

    lab_params = {}
    for ls in cfg["labeling"]["strategies"]:
        if ls["name"] == "directional_binary":
            lab_params = ls
            break

    records = []
    for model_type in search["model_types"]:
        model_cfg = cfg["models"].get(model_type, {})
        for seed in search["seeds"]:
            for pi, (p_start, p_end) in enumerate(sub_periods):
                period_name = f"P{pi+1}"
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
                    df_feat["oracle"] = generate_oracle_feature(
                        df_period, horizon=5, k=k, seed=42,
                    )
                    labels = make_labels(
                        df_period, horizon=5, mode="directional_binary",
                        threshold_mult=lab_params.get("threshold_mult", 0.5),
                        vol_window=lab_params.get("vol_window", 20),
                    )
                    df_feat["_label"] = labels
                    df_clean = df_feat.dropna(
                        subset=ORACLE_FEATURE_COLS + ["_label"],
                    )
                    if len(df_clean) < 40:
                        continue
                    X = df_clean[ORACLE_FEATURE_COLS]
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
                            X_train, y_train, model_type=model_type,
                            config=model_cfg, seed=seed,
                            feature_names=ORACLE_FEATURE_COLS,
                        )
                        proba = predict_proba(model, X_test)
                    except Exception:
                        continue
                    n_classes = proba.shape[1]
                    preds = proba[:, 1] if n_classes == 2 else proba[:, -1]
                    test_df = df_clean.loc[split.test_idx]
                    bt = run_naive_backtest(
                        preds, test_df["close"].values, test_df["open"].values,
                        horizon=5, cost_bps=scs_cfg.get("cost_bps", 5.0),
                    )
                    records.append({
                        "ticker": ticker, "period": period_name,
                        "seed": seed, "model_type": model_type,
                        "sharpe": bt["sharpe"],
                    })

    df_rec = pd.DataFrame(records)

    # S_time
    sharpe_by_period = df_rec.groupby("period")["sharpe"].mean()
    vals = sharpe_by_period.values
    sign_ratio = (vals > 0).mean()
    mean_v = vals.mean()
    std_v = vals.std(ddof=1) if len(vals) > 1 else 0.0
    cv = std_v / (abs(mean_v) + 1e-8)
    min_v = vals.min()
    c_sign = sign_ratio
    c_stab = np.exp(-cv)
    c_mag = min(1.0, abs(mean_v) / 0.5)
    if min_v > -0.5:
        c_worst = 1.0
    elif min_v > -1.0:
        c_worst = max(0.0, 1.0 + (min_v + 0.5) / 0.5)
    else:
        c_worst = 0.0
    R_time = _robustness_score(vals)

    print(f"\n  S_time detail (R(v) = {R_time:.4f}):")
    for name, val in sorted(sharpe_by_period.items(), key=lambda x: -x[1]):
        marker = "+" if val > 0 else "-"
        print(f"    {name:6s} mean_sharpe = {val:+.4f}  [{marker}]")
    print(f"    sign_ratio={sign_ratio:.4f}  CV={cv:.4f}  min={min_v:+.4f}")
    print(f"    c_sign={c_sign:.3f}  c_stab={c_stab:.3f}  c_mag={c_mag:.3f}  c_worst={c_worst:.3f}")
    gates = []
    if mean_v < 0:
        gates.append("mean<0->cap0.30")
    if min_v < -1.0:
        gates.append("min<-1->cap0.40")
    if sign_ratio < 0.5:
        gates.append("sign<0.5->cap0.50")
    print(f"    Hard gates: {' | '.join(gates) if gates else 'none'}")

    # S_asset (top 5, bottom 5)
    sharpe_by_asset = df_rec.groupby("ticker")["sharpe"].mean().sort_values(
        ascending=False,
    )
    vals_a = sharpe_by_asset.values
    R_asset = _robustness_score(vals_a)
    n_pos = (vals_a > 0).sum()
    print(f"\n  S_asset detail (R(v) = {R_asset:.4f}, {n_pos}/{len(vals_a)} positive):")
    print("    Top 5:")
    for name, val in list(sharpe_by_asset.items())[:5]:
        print(f"      {name:8s} {val:+.4f}")
    print("    Bottom 5:")
    for name, val in list(sharpe_by_asset.items())[-5:]:
        print(f"      {name:8s} {val:+.4f}")
    cv_a = vals_a.std(ddof=1) / (abs(vals_a.mean()) + 1e-8)
    print(f"    mean={vals_a.mean():+.4f}  min={vals_a.min():+.4f}  CV={cv_a:.4f}")

    # S_seed
    sharpe_by_seed = df_rec.groupby("seed")["sharpe"].mean()
    vals_s = sharpe_by_seed.values
    cv_s = vals_s.std(ddof=1) / (abs(vals_s.mean()) + 1e-8) if len(vals_s) > 1 else 0.0
    print(f"\n  S_seed detail:")
    for name, val in sorted(sharpe_by_seed.items(), key=lambda x: -x[1]):
        print(f"    seed={name:5d}  mean_sharpe = {val:+.4f}")
    print(f"    CV = {cv_s:.4f}  range = [{vals_s.min():+.4f}, {vals_s.max():+.4f}]")


# ── Main experiment ─────────────────────────────────────────────

def run_oracle_experiment(verbose=True):
    """Run the oracle feature injection experiment."""
    cfg = load_config()
    tickers = get_all_tickers(cfg)
    feat_cfg = get_feature_config(cfg)
    split_cfg = cfg["splitting"]
    scs_cfg = cfg["scs_a"]

    print("=" * 70)
    print("ORACLE FEATURE INJECTION EXPERIMENT")
    print("=" * 70)
    print("  Design: inject oracle = fwd_ret + k*std*noise into feature matrix")
    print("  Labels: real directional_binary 5d (unchanged)")
    print("  WARNING: oracle uses lookahead — controlled validation only")

    data = load_universe(
        tickers,
        cfg["periods"]["phase_a"]["start"],
        cfg["periods"]["phase_a"]["end"],
        cache_dir=Path(cfg["data"]["cache_dir"]),
    )
    print(f"\n  Tickers: {len(data)}")
    print(f"  Period: {cfg['periods']['phase_a']['start']} -> {cfg['periods']['phase_a']['end']}")

    sample_dates = data[list(data.keys())[0]].index
    sub_periods = split_into_sub_periods(
        cfg["periods"]["phase_a"]["start"],
        cfg["periods"]["phase_a"]["end"],
        split_cfg["phase_a"]["n_sub_periods"],
        sample_dates,
    )

    # Calibrate
    calib = calibrate_oracle_accuracy(data, horizon=5)

    # Run dose-response
    k_values = [0.5, 1.5, 4.0]
    dose_response = {}

    for k in k_values:
        print(f"\n{'=' * 60}")
        print(f"  NOISE LEVEL k = {k}")
        print(f"{'=' * 60}")
        t0 = time.time()
        result = run_phase_a_oracle(cfg, data, sub_periods, k_value=k, verbose=verbose)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")
        dose_response[f"k={k}"] = result

        # Print component detail for non-pre-gate-rejected
        rejected_pregate = (
            result.get("verdict") == "REJECTED"
            and result.get("rejection_reason", "").startswith("PRE_GATE")
        )
        if not rejected_pregate:
            print_component_detail(
                cfg, data, feat_cfg, split_cfg, scs_cfg, sub_periods, k,
            )
        else:
            print(f"\n  REJECTED at pre-gate: {result.get('rejection_reason', '')}")

    # Save results
    all_results = {
        "experiment": "oracle_feature_injection",
        "description": (
            "Injects oracle=fwd_ret+k*std*noise into feature matrix. "
            "Labels are real directional_binary 5d. "
            "Tests SCS scoring/gating machinery, not feature predictive power. "
            "Oracle uses lookahead (controlled validation only)."
        ),
        "calibration": {
            f"k={k}": {"corr": v["corr"], "accuracy": v["accuracy"]}
            for k, v in calib.items()
        },
        "dose_response": {
            k_label: clean_result(v) for k_label, v in dose_response.items()
        },
    }

    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment_oracle_feature.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    header = (
        f"  {'k':>5s} {'Corr':>7s} {'Acc':>7s} {'SCS-A':>7s} "
        f"{'S_time':>7s} {'S_asset':>7s} {'S_model':>7s} {'S_seed':>7s} "
        f"{'S_dist':>7s} {'Verdict':>18s}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for k in k_values:
        res = dose_response[f"k={k}"]
        cal = calib[k]
        scs = res.get("SCS_A", 0.0)
        st = res.get("S_time", 0.0)
        sa = res.get("S_asset", 0.0)
        sm = res.get("S_model", 0.0)
        ss = res.get("S_seed", 0.0)
        sd = res.get("S_dist", 0.0)
        v = res.get("verdict", "N/A")
        print(
            f"  k={k:.1f} {cal['corr']:7.4f} {cal['accuracy']:6.1%} {scs:7.4f} "
            f"{st:7.4f} {sa:7.4f} {sm:7.4f} {ss:7.4f} {sd:7.4f} {v:>18s}"
        )

    # Stop condition
    print(f"\n{'=' * 70}")
    print("STOP CONDITION CHECK")
    print(f"{'=' * 70}")
    k05_scs = dose_response["k=0.5"].get("SCS_A", 0)
    k40_verdict = dose_response["k=4.0"].get("verdict", "")
    k05_pass = k05_scs >= 0.70
    k40_fail = k40_verdict == "REJECTED"
    print(f"  k=0.5 SCS-A >= 0.70: {'YES' if k05_pass else 'NO'} (SCS-A = {k05_scs:.4f})")
    print(f"  k=4.0 rejected:      {'YES' if k40_fail else 'NO'} ({k40_verdict})")
    if k05_pass and k40_fail:
        print("  -> EXPERIMENT WORKS AS DESIGNED. Update paper.")
    elif k05_pass:
        print("  -> k=0.5 passes but k=4.0 not rejected. Partial success.")
    else:
        print("  -> k=0.5 still fails. Further diagnosis needed.")

    return all_results


def clean_result(obj):
    """Clean a result dict for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_result(v) for k, v in obj.items()
                if k not in ("trade_returns",)}
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, list):
        return [clean_result(i) for i in obj]
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


if __name__ == "__main__":
    run_oracle_experiment(verbose=True)

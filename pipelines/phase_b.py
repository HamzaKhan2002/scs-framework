"""
Phase B — Walk-Forward Validation Pipeline.

Validates Phase A approved signals using expanding walk-forward windows.
Trains on pooled cross-sectional data with embargo.
Tests under baseline and stress costs.
"""

import sys
import json
import joblib
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
from src.validation.temporal_split import build_walk_forward_windows
from src.validation.scs_b import compute_scs_b
from src.backtest.portfolio_engine import run_naive_backtest


def run_phase_b(
    cfg: Dict = None,
    approved_groups: Dict = None,
    verbose: bool = True,
    model_types: List[str] = None,
    seeds: List[int] = None,
) -> Dict[str, Any]:
    """
    Run Phase B: Walk-Forward Validation.

    For each approved signal group:
        For each walk-forward window:
            For each model type:
                For each seed:
                    1. Pool all tickers' training data
                    2. Apply embargo
                    3. Train model on pooled data
                    4. Test on each ticker individually
                    5. Record results under baseline + stress costs

        Compute SCS-B, apply verdict.

    Args:
        model_types: List of model types to train (default: all 3 from config).
        seeds: List of seeds for training (default: [42, 123, 789]).
    """
    if cfg is None:
        cfg = load_config()

    if model_types is None:
        model_types = cfg["search_space"]["model_types"]
    if seeds is None:
        seeds = [42, 123, 789]

    tickers = get_all_tickers(cfg)
    feat_cfg = get_feature_config(cfg)
    split_cfg = cfg["splitting"]
    scs_b_cfg = cfg["scs_b"]

    if verbose:
        print(f"\n" + "=" * 70)
        print("PHASE B — WALK-FORWARD VALIDATION")
        print("=" * 70)
        print(f"  Model types: {model_types}")
        print(f"  Seeds: {seeds}")

    # Load Phase B data
    data = load_universe(
        tickers,
        cfg["periods"]["phase_b"]["start"],
        cfg["periods"]["phase_b"]["end"],
        cache_dir=Path(cfg["data"]["cache_dir"]),
    )
    if verbose:
        print(f"  Tickers loaded: {len(data)}")
        print(f"  Period: {cfg['periods']['phase_b']['start']} → {cfg['periods']['phase_b']['end']}")

    wf_windows = build_walk_forward_windows(
        split_cfg["phase_b"]["windows"],
        embargo_days=split_cfg["embargo_days"],
    )
    if verbose:
        for i, w in enumerate(wf_windows):
            print(f"  Window W{i+1}: train→{w.train_end.date()} | "
                  f"test {w.test_start.date()}→{w.test_end.date()}")

    if approved_groups is None:
        # Default: test all 4 groups
        approved_groups = {}
        for h in cfg["search_space"]["horizons"]:
            for lm in cfg["search_space"]["label_modes"]:
                approved_groups[f"{h}d_{lm}"] = {"horizon": h, "label_mode": lm}

    all_group_results = {}
    trained_models = {}  # Store last window's model for Phase C

    for group_key, group_info in approved_groups.items():
        horizon = group_info.get("horizon", int(group_key.split("d_")[0]))
        label_mode = group_info.get("label_mode", group_key.split("d_")[1])

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Signal Group: {group_key}")

        lab_params = {}
        for ls in cfg["labeling"]["strategies"]:
            if ls["name"] == label_mode:
                lab_params = ls
                break

        records_base = []
        records_stress = []
        last_models = {}  # {model_type: model} — store last window's models

        for wi, window in enumerate(wf_windows):
            window_name = f"W{wi+1}"
            if verbose:
                print(f"    Window {window_name}...")

            # Effective train end with embargo
            eff_train_end = window.effective_train_end()

            # Pool training data across all tickers
            X_train_pool = []
            y_train_pool = []

            for ticker in data:
                df = data[ticker]
                mask_train = (df.index >= window.train_start) & (df.index <= eff_train_end)
                df_train = df[mask_train]

                if len(df_train) < 60:
                    continue

                df_feat = build_features(
                    df_train,
                    return_lags=feat_cfg.return_lags,
                    vol_windows=feat_cfg.volatility_windows,
                    atr_window=feat_cfg.atr_window,
                    vol_z_window=feat_cfg.volume_z_window,
                )
                labels = make_labels(
                    df_feat, horizon=horizon, mode=label_mode,
                    threshold_mult=lab_params.get("threshold_mult", 0.5),
                    vol_window=lab_params.get("vol_window", 20),
                )
                df_feat["_label"] = labels
                df_clean = df_feat.dropna(subset=FEATURE_COLS + ["_label"])
                if len(df_clean) < 20:
                    continue

                X, _ = get_feature_matrix(df_clean)
                y = df_clean["_label"].values.astype(int)
                X_train_pool.append(X.values)
                y_train_pool.append(y)

            if not X_train_pool:
                continue

            X_train_all = np.vstack(X_train_pool)
            y_train_all = np.concatenate(y_train_pool)

            if verbose:
                print(f"      Training samples: {len(X_train_all)}")

            # Train all model types × seeds, average predictions
            for model_type in model_types:
                model_cfg = cfg["models"][model_type]

                for seed in seeds:
                    model = train_model(
                        X_train_all, y_train_all,
                        model_type=model_type,
                        config=model_cfg,
                        seed=seed,
                        feature_names=FEATURE_COLS,
                    )
                    last_models[f"{model_type}_s{seed}"] = model

                    # Test on each ticker individually
                    for ticker in data:
                        df = data[ticker]
                        mask_test = (df.index >= window.test_start) & (df.index <= window.test_end)
                        df_test = df[mask_test]

                        if len(df_test) < 20:
                            continue

                        df_feat = build_features(
                            df_test,
                            return_lags=feat_cfg.return_lags,
                            vol_windows=feat_cfg.volatility_windows,
                            atr_window=feat_cfg.atr_window,
                            vol_z_window=feat_cfg.volume_z_window,
                        )
                        labels = make_labels(
                            df_feat, horizon=horizon, mode=label_mode,
                            threshold_mult=lab_params.get("threshold_mult", 0.5),
                            vol_window=lab_params.get("vol_window", 20),
                        )
                        df_feat["_label"] = labels
                        df_clean = df_feat.dropna(subset=FEATURE_COLS + ["_label"])
                        if len(df_clean) < 10:
                            continue

                        X_test, _ = get_feature_matrix(df_clean)

                        try:
                            proba = predict_proba(model, X_test.values)
                        except Exception:
                            continue

                        n_classes = proba.shape[1]
                        if n_classes == 2:
                            preds = proba[:, 1]
                        elif n_classes == 3:
                            preds = proba[:, 2]
                        else:
                            preds = proba[:, -1]

                        close_arr = df_clean["close"].values
                        open_arr = df_clean["open"].values

                        # Baseline costs
                        bt_base = run_naive_backtest(
                            preds, close_arr, open_arr,
                            horizon=horizon,
                            cost_bps=scs_b_cfg["cost_scenarios"]["baseline_bps"],
                        )
                        records_base.append({
                            "window": window_name, "ticker": ticker,
                            "model_type": model_type, "seed": seed,
                            "sharpe": bt_base["sharpe"],
                            "total_return": bt_base["total_return"],
                            "n_trades": bt_base["n_trades"],
                            "win_rate": bt_base["win_rate"],
                        })

                        # Stress costs
                        bt_stress = run_naive_backtest(
                            preds, close_arr, open_arr,
                            horizon=horizon,
                            cost_bps=scs_b_cfg["cost_scenarios"]["stress_bps"],
                        )
                        records_stress.append({
                            "window": window_name, "ticker": ticker,
                            "model_type": model_type, "seed": seed,
                            "sharpe": bt_stress["sharpe"],
                            "total_return": bt_stress["total_return"],
                            "n_trades": bt_stress["n_trades"],
                            "win_rate": bt_stress["win_rate"],
                        })

        if not records_base:
            scs_result = {"SCS_B": 0.0, "verdict": "NO_DATA"}
        else:
            df_base = pd.DataFrame(records_base)
            df_stress = pd.DataFrame(records_stress)
            scs_result = compute_scs_b(
                df_base, df_stress,
                horizon=horizon,
                weights=scs_b_cfg["weights"],
            )

        scs_result["group_key"] = group_key
        scs_result["horizon"] = horizon
        scs_result["label_mode"] = label_mode
        scs_result["n_runs_base"] = len(records_base)
        all_group_results[group_key] = scs_result

        # Store last window's models for Phase C (use lightgbm seed=42 as primary)
        primary_key = "lightgbm_s42"
        if primary_key in last_models:
            trained_models[group_key] = {
                "model": last_models[primary_key],
                "horizon": horizon,
                "label_mode": label_mode,
                "model_type": "lightgbm",
                "all_models": {k: m for k, m in last_models.items()},
            }
        elif last_models:
            # Fallback: use whatever model is available
            first_key = list(last_models.keys())[0]
            mt = first_key.split("_s")[0]
            trained_models[group_key] = {
                "model": last_models[first_key],
                "horizon": horizon,
                "label_mode": label_mode,
                "model_type": mt,
                "all_models": {k: m for k, m in last_models.items()},
            }

        if verbose:
            print(f"    SCS-B = {scs_result['SCS_B']:.4f} → {scs_result['verdict']}")
            for k in ["S_time", "S_asset", "S_cost", "S_struct", "S_eco"]:
                if k in scs_result:
                    print(f"      {k} = {scs_result[k]:.4f}")

    # Save results
    results_dir = ROOT_DIR / "results" / "phase_b"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"phase_b_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_group_results, f, indent=2, default=str)

    # Save trained models
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for gk, minfo in trained_models.items():
        joblib.dump(minfo["model"], models_dir / f"{gk}_model.joblib")

    if verbose:
        print(f"\n  Results saved: {out_path}")
        print("\n  PHASE B SUMMARY")
        print("  " + "─" * 50)
        for gk, res in all_group_results.items():
            print(f"  {gk:35s} SCS-B={res['SCS_B']:.4f}  {res['verdict']}")

    return {
        "group_results": all_group_results,
        "approved": {k: v for k, v in all_group_results.items()
                     if v.get("verdict") == "VALID_FOR_PHASE_C"},
        "trained_models": trained_models,
        "timestamp": ts,
    }


if __name__ == "__main__":
    result = run_phase_b(verbose=True)

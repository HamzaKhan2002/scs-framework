"""
Probability of Backtest Overfitting (PBO) — Bailey et al. (2017).

Computes PBO for each signal group using Phase B walk-forward fold results.
Provides a head-to-head comparison between CPCV-based validation and SCS.

Usage:
    PYTHONIOENCODING=utf-8 python pipelines/compute_pbo.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Any

from scipy.stats import spearmanr

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import load_config, get_all_tickers, get_feature_config
from src.data.loader import load_universe
from src.features.engine import build_features, get_feature_matrix, FEATURE_COLS
from src.labeling.engine import make_labels
from src.models.classifiers import train_model, predict_proba
from src.validation.temporal_split import build_walk_forward_windows
from src.backtest.portfolio_engine import run_naive_backtest


def compute_fold_sharpes(cfg: Dict, verbose: bool = True) -> pd.DataFrame:
    """
    Compute mean Sharpe ratio for each (signal_group, fold) combination.

    Returns DataFrame with columns: group_key, fold, fold_idx, mean_sharpe, n_tickers
    """
    tickers = get_all_tickers(cfg)
    feat_cfg = get_feature_config(cfg)
    split_cfg = cfg["splitting"]

    data = load_universe(
        tickers,
        cfg["periods"]["phase_b"]["start"],
        cfg["periods"]["phase_b"]["end"],
        cache_dir=Path(cfg["data"]["cache_dir"]),
    )

    wf_windows = build_walk_forward_windows(
        split_cfg["phase_b"]["windows"],
        embargo_days=split_cfg["embargo_days"],
    )

    records: List[Dict[str, Any]] = []

    for horizon in cfg["search_space"]["horizons"]:
        for label_mode in cfg["search_space"]["label_modes"]:
            group_key = f"{horizon}d_{label_mode}"

            lab_params: Dict[str, Any] = {}
            for ls in cfg["labeling"]["strategies"]:
                if ls["name"] == label_mode:
                    lab_params = ls
                    break

            for wi, window in enumerate(wf_windows):
                fold_name = f"F{wi+1}"
                eff_train_end = window.effective_train_end()

                # Pool training data
                X_train_pool: List[np.ndarray] = []
                y_train_pool: List[np.ndarray] = []

                for ticker in data:
                    df = data[ticker]
                    mask_train = (df.index >= window.train_start) & (
                        df.index <= eff_train_end
                    )
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
                        df_feat,
                        horizon=horizon,
                        mode=label_mode,
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

                # Train LightGBM model
                model = train_model(
                    X_train_all,
                    y_train_all,
                    model_type="lightgbm",
                    config=cfg["models"]["lightgbm"],
                    seed=42,
                    feature_names=FEATURE_COLS,
                )

                # Test on each ticker for this fold
                fold_sharpes: List[float] = []
                for ticker in data:
                    df = data[ticker]
                    mask_test = (df.index >= window.test_start) & (
                        df.index <= window.test_end
                    )
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
                        df_feat,
                        horizon=horizon,
                        mode=label_mode,
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

                    bt = run_naive_backtest(
                        preds,
                        df_clean["close"].values,
                        df_clean["open"].values,
                        horizon=horizon,
                        cost_bps=5.0,
                    )
                    fold_sharpes.append(bt["sharpe"])

                if fold_sharpes:
                    records.append(
                        {
                            "group_key": group_key,
                            "fold": fold_name,
                            "fold_idx": wi,
                            "mean_sharpe": float(np.mean(fold_sharpes)),
                            "n_tickers": len(fold_sharpes),
                        }
                    )

                    if verbose:
                        print(
                            f"  {group_key} {fold_name}: "
                            f"mean SR = {np.mean(fold_sharpes):.4f} "
                            f"({len(fold_sharpes)} tickers)"
                        )

    return pd.DataFrame(records)


def compute_pbo(
    fold_sharpes_df: pd.DataFrame, n_test_folds: int = 2
) -> Dict[str, Any]:
    """
    Compute PBO using combinatorial purged cross-validation.

    Args:
        fold_sharpes_df: DataFrame with columns [group_key, fold_idx, mean_sharpe]
        n_test_folds: Number of folds to hold out as test in each combination (S in CPCV)

    Returns:
        Dict with PBO per group, overall PBO, and IS-OOS rank correlations
    """
    groups = sorted(fold_sharpes_df["group_key"].unique())
    fold_indices = sorted(fold_sharpes_df["fold_idx"].unique())
    n_folds = len(fold_indices)

    # Build performance matrix: groups x folds
    perf_matrix: Dict[str, Dict[int, float]] = {}
    for g in groups:
        g_data = fold_sharpes_df[fold_sharpes_df["group_key"] == g]
        perf_matrix[g] = {
            row["fold_idx"]: row["mean_sharpe"] for _, row in g_data.iterrows()
        }

    # Generate all C(n_folds, n_test_folds) combinations
    combos = list(combinations(fold_indices, n_test_folds))
    n_combos = len(combos)

    print(
        f"\n  CPCV: {n_folds} folds, S={n_test_folds} "
        f"-> C({n_folds},{n_test_folds}) = {n_combos} paths"
    )

    # For each combination, compute IS and OOS performance per group
    is_best_underperforms = 0
    rank_correlations: List[float] = []
    per_group_logits: Dict[str, List[float]] = {g: [] for g in groups}
    n_groups = len(groups)

    for combo in combos:
        test_folds = set(combo)
        train_folds = set(fold_indices) - test_folds

        # Compute IS (train) and OOS (test) mean Sharpe per group
        is_sharpes: Dict[str, float] = {}
        oos_sharpes: Dict[str, float] = {}

        for g in groups:
            is_vals = [perf_matrix[g].get(f, np.nan) for f in train_folds]
            oos_vals = [perf_matrix[g].get(f, np.nan) for f in test_folds]
            is_sharpes[g] = float(np.nanmean(is_vals))
            oos_sharpes[g] = float(np.nanmean(oos_vals))

        # Rank groups by IS Sharpe (descending)
        is_ranking = sorted(groups, key=lambda g: is_sharpes[g], reverse=True)
        oos_ranking = sorted(groups, key=lambda g: oos_sharpes[g], reverse=True)

        # PBO check: does IS-best underperform the median OOS?
        is_best = is_ranking[0]
        oos_median = float(np.median(list(oos_sharpes.values())))
        if oos_sharpes[is_best] < oos_median:
            is_best_underperforms += 1

        # Rank correlation between IS and OOS rankings
        is_ranks = [is_ranking.index(g) for g in groups]
        oos_ranks = [oos_ranking.index(g) for g in groups]
        rho, _ = spearmanr(is_ranks, oos_ranks)
        rank_correlations.append(float(rho))

        # Per-group: compute logit of relative OOS rank
        for g in groups:
            oos_rank = oos_ranking.index(g) + 1  # 1-indexed
            w = oos_rank / (n_groups + 1)
            logit_w = np.log(w / (1 - w))
            per_group_logits[g].append(float(logit_w))

    pbo = is_best_underperforms / n_combos
    mean_rank_corr = float(np.mean(rank_correlations))

    # Per-group PBO: fraction of paths where group's OOS rank is below median
    per_group_pbo: Dict[str, float] = {}
    for g in groups:
        # logit > 0 means rank > median (i.e. below-median performance)
        n_below_median = sum(1 for l in per_group_logits[g] if l > 0)
        per_group_pbo[g] = round(n_below_median / n_combos, 4)

    return {
        "pbo": round(pbo, 4),
        "n_combos": n_combos,
        "mean_is_oos_rank_correlation": round(mean_rank_corr, 4),
        "rank_correlations": [round(r, 4) for r in rank_correlations],
        "per_group_pbo": per_group_pbo,
        "per_group_mean_logit": {
            g: round(float(np.mean(v)), 4) for g, v in per_group_logits.items()
        },
    }


def main():
    print("=" * 70)
    print("CPCV / PBO COMPUTATION")
    print("=" * 70)

    cfg = load_config()
    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Compute per-fold Sharpe for each group
    cache_path = out_dir / "pbo_fold_sharpes.json"

    if cache_path.exists():
        print("  Loading cached fold Sharpes...")
        with open(cache_path) as f:
            cached = json.load(f)
        fold_df = pd.DataFrame(cached)
    else:
        print("  Computing fold Sharpes (this runs Phase B-style evaluation)...")
        fold_df = compute_fold_sharpes(cfg, verbose=True)
        # Save cache
        with open(cache_path, "w") as f:
            json.dump(fold_df.to_dict(orient="records"), f, indent=2)
        print(f"  Cached to {cache_path}")

    # Step 2: Compute PBO with S=2 test folds
    print("\n  Computing PBO (S=2)...")
    pbo_results = compute_pbo(fold_df, n_test_folds=2)

    print(f"\n  {'='*50}")
    print(f"  PBO = {pbo_results['pbo']:.4f} ({pbo_results['pbo']*100:.1f}%)")
    print(
        f"  Mean IS-OOS rank correlation: "
        f"{pbo_results['mean_is_oos_rank_correlation']:.4f}"
    )
    print(f"  Combinatorial paths: {pbo_results['n_combos']}")
    print(f"\n  Per-group PBO:")
    for g, p in sorted(pbo_results["per_group_pbo"].items()):
        print(f"    {g:35s} PBO = {p:.4f}")

    # Save results
    result_path = out_dir / "pbo_results.json"
    with open(result_path, "w") as f:
        json.dump(pbo_results, f, indent=2)
    print(f"\n  Results saved to {result_path}")


if __name__ == "__main__":
    main()

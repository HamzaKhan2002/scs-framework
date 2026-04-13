"""
SHAP Feature Importance Stability — analyzes whether the same features
are important across Phase A sub-periods and model types.

Usage:
    PYTHONIOENCODING=utf-8 python pipelines/run_shap_stability.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import load_config, get_all_tickers, get_feature_config
from src.data.loader import load_universe
from src.features.engine import build_features, get_feature_matrix, FEATURE_COLS
from src.labeling.engine import make_labels
from src.models.classifiers import train_model
from src.validation.temporal_split import purged_temporal_split, split_into_sub_periods


def compute_shap_importance(model, X_test, model_type, feature_names):
    """Compute SHAP feature importance for a trained model."""
    try:
        import shap
    except ImportError:
        print("  [WARN] shap not installed. Using model-native importance for tree models.")
        if model_type in ("lightgbm", "xgboost"):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance / (importance.sum() + 1e-10)))
        return None

    if model_type in ("lightgbm", "xgboost"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            # Multi-class: average across classes
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)
            # Handle 3D arrays (n_samples, n_features, n_classes)
            if shap_values.ndim == 3:
                shap_values = shap_values.mean(axis=2)
        mean_shap = shap_values.mean(axis=0)
        mean_shap = np.asarray(mean_shap).flatten()
        return dict(zip(feature_names, mean_shap / (mean_shap.sum() + 1e-10)))
    elif model_type == "logistic_regression":
        # Use coefficient magnitudes as proxy
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        coefs = np.abs(clf.coef_)
        if coefs.ndim > 1:
            coefs = coefs.mean(axis=0)
        coefs = np.asarray(coefs).flatten()
        return dict(zip(feature_names, coefs / (coefs.sum() + 1e-10)))

    return None


def main():
    print("=" * 70)
    print("SHAP FEATURE IMPORTANCE STABILITY")
    print("=" * 70)

    cfg = load_config()
    tickers = get_all_tickers(cfg)
    feat_cfg = get_feature_config(cfg)
    split_cfg = cfg["splitting"]

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

    # Focus on approved signal groups with strongest OOS performance
    target_groups = [
        (10, "directional_binary"),
        (20, "directional_binary"),
        (20, "multiclass_volatility"),
    ]

    all_importance = []

    for horizon, label_mode in target_groups:
        group_key = f"{horizon}d_{label_mode}"
        lab_params = {}
        for ls in cfg["labeling"]["strategies"]:
            if ls["name"] == label_mode:
                lab_params = ls
                break

        print(f"\n  Group: {group_key}")

        for model_type in cfg["search_space"]["model_types"]:
            for pi, (p_start, p_end) in enumerate(sub_periods):
                period_name = f"P{pi+1}"

                # Pool all tickers for this period
                X_train_all, y_train_all, X_test_all = [], [], []

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
                    labels = make_labels(
                        df_feat, horizon=horizon, mode=label_mode,
                        threshold_mult=lab_params.get("threshold_mult", 0.5),
                        vol_window=lab_params.get("vol_window", 20),
                    )
                    df_feat["_label"] = labels
                    df_clean = df_feat.dropna(subset=FEATURE_COLS + ["_label"])
                    if len(df_clean) < 40:
                        continue

                    X, _ = get_feature_matrix(df_clean)
                    y = df_clean["_label"].values.astype(int)

                    try:
                        split = purged_temporal_split(
                            df_clean.index,
                            train_ratio=split_cfg["phase_a"]["train_ratio"],
                            embargo_days=split_cfg["embargo_days"],
                        )
                    except ValueError:
                        continue

                    X_train_all.append(X.loc[split.train_idx].values)
                    y_train_all.append(y[df_clean.index.isin(split.train_idx)])
                    X_test_all.append(X.loc[split.test_idx].values)

                if not X_train_all:
                    continue

                X_tr = np.vstack(X_train_all)
                y_tr = np.concatenate(y_train_all)
                X_te = np.vstack(X_test_all)

                model_cfg = cfg["models"].get(model_type, {})
                try:
                    model = train_model(
                        X_tr, y_tr,
                        model_type=model_type,
                        config=model_cfg,
                        seed=42,
                        feature_names=FEATURE_COLS,
                    )
                except Exception:
                    continue

                importance = compute_shap_importance(model, X_te, model_type, FEATURE_COLS)
                if importance:
                    record = {
                        "group": group_key,
                        "model_type": model_type,
                        "period": period_name,
                    }
                    record.update(importance)
                    all_importance.append(record)

                    top3 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                    top3_str = ", ".join(f"{k}={v:.3f}" for k, v in top3)
                    print(f"    {model_type:20s} {period_name}: {top3_str}")

    # Analyze stability
    stability_results = {}
    if all_importance:
        df_imp = pd.DataFrame(all_importance)

        # Compute rank correlation of feature importance across periods
        from scipy.stats import spearmanr

        for group_key in df_imp["group"].unique():
            g_data = df_imp[df_imp["group"] == group_key]

            for model_type in g_data["model_type"].unique():
                m_data = g_data[g_data["model_type"] == model_type]
                if len(m_data) < 2:
                    continue

                # Pairwise Spearman correlations of feature importance vectors
                periods = m_data["period"].unique()
                corrs = []
                for i in range(len(periods)):
                    for j in range(i+1, len(periods)):
                        v1 = m_data[m_data["period"] == periods[i]][FEATURE_COLS].values.flatten()
                        v2 = m_data[m_data["period"] == periods[j]][FEATURE_COLS].values.flatten()
                        rho, _ = spearmanr(v1, v2)
                        corrs.append(rho)

                key = f"{group_key}__{model_type}"
                stability_results[key] = {
                    "group": group_key,
                    "model_type": model_type,
                    "mean_spearman": round(float(np.mean(corrs)), 4),
                    "n_pairs": len(corrs),
                }

        print(f"\n  Feature Importance Stability (Spearman across periods):")
        for k, v in sorted(stability_results.items()):
            print(f"    {k:40s} rho = {v['mean_spearman']:.4f} ({v['n_pairs']} pairs)")

    # Save everything
    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "feature_importance": all_importance,
        "stability": stability_results if all_importance else {},
    }

    result_path = out_dir / "shap_stability.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to {result_path}")


if __name__ == "__main__":
    main()

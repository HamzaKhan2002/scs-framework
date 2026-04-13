"""
Hyperparameter Sensitivity Analysis — tests whether results are stable
across 3 hyperparameter configurations.

This replaces Optuna optimization: we demonstrate that results are
insensitive to hyperparameter choice, making the fixed-parameter
approach defensible.

Usage:
    PYTHONIOENCODING=utf-8 python pipelines/run_hyperparam_sensitivity.py
"""

import sys
import json
import copy
import time
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import load_config
from pipelines.phase_a import run_phase_a

# Three hyperparameter configurations: conservative, moderate, aggressive
CONFIGS = {
    "conservative": {
        "lightgbm": {
            "learning_rate": 0.03, "n_estimators": 300, "max_depth": 2,
            "num_leaves": 4, "min_child_samples": 300, "subsample": 0.6,
            "colsample_bytree": 0.6, "reg_alpha": 2.0, "reg_lambda": 10.0,
        },
        "xgboost": {
            "learning_rate": 0.03, "n_estimators": 300, "max_depth": 3,
            "min_child_weight": 10, "subsample": 0.6, "colsample_bytree": 0.6,
            "reg_alpha": 2.0, "reg_lambda": 10.0,
        },
        "logistic_regression": {"penalty": "l2", "C": 0.1, "solver": "lbfgs", "max_iter": 1000},
    },
    "baseline": {
        "lightgbm": {
            "learning_rate": 0.05, "n_estimators": 400, "max_depth": 3,
            "num_leaves": 8, "min_child_samples": 200, "subsample": 0.7,
            "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 5.0,
        },
        "xgboost": {
            "learning_rate": 0.05, "n_estimators": 400, "max_depth": 6,
            "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 1.0, "reg_lambda": 5.0,
        },
        "logistic_regression": {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000},
    },
    "aggressive": {
        "lightgbm": {
            "learning_rate": 0.10, "n_estimators": 600, "max_depth": 5,
            "num_leaves": 31, "min_child_samples": 50, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
        },
        "xgboost": {
            "learning_rate": 0.10, "n_estimators": 600, "max_depth": 8,
            "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        },
        "logistic_regression": {"penalty": "l2", "C": 10.0, "solver": "lbfgs", "max_iter": 1000},
    },
}


def main():
    print("=" * 70)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)

    base_cfg = load_config()
    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "hyperparam_sensitivity.json"

    all_results = {}

    for config_name, model_overrides in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Configuration: {config_name}")
        print(f"{'='*60}")

        cfg = copy.deepcopy(base_cfg)
        cfg["models"] = model_overrides

        t0 = time.time()
        pa = run_phase_a(cfg=cfg, verbose=True)
        elapsed = time.time() - t0

        config_result = {
            "config_name": config_name,
            "model_params": model_overrides,
            "elapsed_seconds": round(elapsed, 1),
            "groups": {},
        }

        for gk, res in pa["group_results"].items():
            config_result["groups"][gk] = {
                "SCS_A": res.get("SCS_A", 0.0),
                "verdict": res.get("verdict", "UNKNOWN"),
                "S_time": res.get("S_time", 0.0),
                "S_asset": res.get("S_asset", 0.0),
                "S_model": res.get("S_model", 0.0),
                "S_seed": res.get("S_seed", 0.0),
                "S_dist": res.get("S_dist", 0.0),
                "mean_sharpe": res.get("mean_sharpe", 0.0),
            }

        n_pass = sum(1 for r in pa["group_results"].values()
                     if r.get("verdict") == "PHASE_B_APPROVED")
        scs_scores = [r["SCS_A"] for r in pa["group_results"].values()
                      if r.get("SCS_A", 0) > 0]

        config_result["summary"] = {
            "n_pass_070": n_pass,
            "mean_scs_a": round(float(np.mean(scs_scores)), 4) if scs_scores else 0.0,
        }

        all_results[config_name] = config_result

        # Save incrementally
        with open(result_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n  {config_name}: {n_pass}/12 pass, mean SCS-A = {config_result['summary']['mean_scs_a']:.4f}")

    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Config':<15s} {'N_pass':>7s} {'Mean SCS-A':>11s}")
    for name, r in all_results.items():
        s = r["summary"]
        print(f"  {name:<15s} {s['n_pass_070']:>5d}/12 {s['mean_scs_a']:>11.4f}")

    # Check if pass/fail decisions are identical
    baseline_verdicts = {gk: r["verdict"] for gk, r in all_results["baseline"]["groups"].items()}
    all_same = True
    for name, r in all_results.items():
        if name == "baseline":
            continue
        for gk, gr in r["groups"].items():
            if gr["verdict"] != baseline_verdicts.get(gk):
                all_same = False
                print(f"  DIFFERENCE: {gk} is {baseline_verdicts.get(gk)} in baseline but {gr['verdict']} in {name}")

    if all_same:
        print(f"\n  RESULT: All configurations produce IDENTICAL pass/fail decisions.")

    print(f"\n  Results saved to {result_path}")


if __name__ == "__main__":
    main()

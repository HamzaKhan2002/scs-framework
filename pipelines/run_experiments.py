"""
Experiment Runner — Comprehensive validation experiments for the SCS framework.

Runs three experiments:
1. Expanded pipeline (12 signal groups) with forced Phase C on all groups
2. Progressive label corruption (dose-response curve)
3. Threshold sensitivity analysis

Usage: PYTHONIOENCODING=utf-8 python pipelines/run_experiments.py
"""

import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config import load_config, ROOT_DIR
from pipelines.phase_a import run_phase_a
from pipelines.phase_b import run_phase_b
from pipelines.phase_c import run_phase_c


def experiment_1_expanded_pipeline(cfg, verbose=True):
    """
    Experiment 1: Run expanded 12-group pipeline.
    Phase A on all 12 groups → Phase B on ALL (forced) → Phase C on ALL (forced).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: EXPANDED PIPELINE (12 SIGNAL GROUPS)")
    print("=" * 70)

    t0 = time.time()

    # Phase A
    ta = time.time()
    pa = run_phase_a(cfg=cfg, verbose=verbose)
    print(f"\n  Phase A completed in {time.time()-ta:.1f}s")

    # Phase B — force ALL groups through (not just approved)
    all_groups = {}
    for gk, res in pa["group_results"].items():
        parts = gk.split("d_", 1)
        all_groups[gk] = {
            "horizon": int(parts[0]),
            "label_mode": parts[1],
        }

    tb = time.time()
    pb = run_phase_b(cfg=cfg, approved_groups=all_groups, verbose=verbose)
    print(f"\n  Phase B completed in {time.time()-tb:.1f}s")

    # Phase C — force ALL groups through
    tc = time.time()
    trained_models = pb.get("trained_models", {})
    pc = run_phase_c(
        cfg=cfg,
        trained_models=trained_models,
        approved_groups=trained_models,
        verbose=verbose,
    )
    print(f"\n  Phase C completed in {time.time()-tc:.1f}s")

    # Build combined results
    results = {
        "phase_a": pa["group_results"],
        "phase_b": pb["group_results"],
        "phase_c": pc["results"],
        "total_time": round(time.time() - t0, 1),
    }

    # Save
    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()
                    if k not in ("trained_models", "trade_returns")}
        elif isinstance(obj, list):
            return [clean_for_json(i) for i in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    with open(out_dir / "experiment_1_expanded.json", "w") as f:
        json.dump(clean_for_json(results), f, indent=2, default=str)

    return results


def experiment_2_corruption(cfg, verbose=True):
    """
    Experiment 2: Progressive label corruption.
    Run Phase A at corruption levels {0, 10, 20, 30, 50, 75, 100}%.
    Record SCS-A for each group at each level.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: PROGRESSIVE LABEL CORRUPTION")
    print("=" * 70)

    corruption_levels = [0, 10, 20, 30, 50, 75, 100]
    all_results = {}

    for pct in corruption_levels:
        print(f"\n  --- Corruption level: {pct}% ---")
        t0 = time.time()
        pa = run_phase_a(cfg=cfg, verbose=False, label_corruption_pct=float(pct))
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        level_results = {}
        for gk, res in pa["group_results"].items():
            level_results[gk] = {
                "SCS_A": res.get("SCS_A", 0.0),
                "S_time": res.get("S_time", 0.0),
                "S_asset": res.get("S_asset", 0.0),
                "S_model": res.get("S_model", 0.0),
                "S_seed": res.get("S_seed", 0.0),
                "S_dist": res.get("S_dist", 0.0),
                "verdict": res.get("verdict", "UNKNOWN"),
                "mean_sharpe": res.get("mean_sharpe", 0.0),
                "n_runs": res.get("n_runs", 0),
            }
        all_results[str(pct)] = level_results

    # Save
    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "experiment_2_corruption.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 70)
    print("CORRUPTION DOSE-RESPONSE SUMMARY")
    print("=" * 70)
    groups = sorted(all_results["0"].keys())
    header = f"  {'Group':35s}" + "".join(f" {p:>5s}%" for p in [str(x) for x in corruption_levels])
    print(header)
    print("  " + "-" * (35 + 7 * len(corruption_levels)))
    for gk in groups:
        row = f"  {gk:35s}"
        for pct in corruption_levels:
            scs = all_results[str(pct)][gk]["SCS_A"]
            row += f"  {scs:.3f}"
        print(row)

    return all_results


def experiment_3_threshold_sensitivity(exp1_results):
    """
    Experiment 3: Threshold sensitivity analysis.
    Using results from experiment 1, compute what happens at each threshold.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)

    pa = exp1_results["phase_a"]
    pb = exp1_results.get("phase_b", {})
    pc = exp1_results.get("phase_c", {})

    # Collect all SCS-A scores and OOS results
    groups = []
    for gk, res in pa.items():
        scs_a = res.get("SCS_A", 0.0)
        scs_b = pb.get(gk, {}).get("SCS_B", 0.0) if pb else 0.0
        pc_res = pc.get(gk, {}) if pc else {}
        metrics = pc_res.get("metrics", {})

        groups.append({
            "group": gk,
            "scs_a": scs_a,
            "scs_b": scs_b,
            "oos_sharpe": metrics.get("sharpe_ratio", None),
            "oos_return": metrics.get("total_return_pct", None),
            "oos_max_dd": metrics.get("max_drawdown_pct", None),
            "n_trades": metrics.get("n_trades", 0),
            "verdict_a": res.get("verdict", ""),
        })

    # Sort by SCS-A
    groups.sort(key=lambda x: x["scs_a"], reverse=True)

    # Threshold analysis
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    table = []

    print(f"\n  {'Thresh':>7s} {'N_pass':>7s} {'Avg SR':>8s} {'Avg Ret':>8s} {'Groups':>40s}")
    print("  " + "-" * 72)

    for t in thresholds:
        passing = [g for g in groups if g["scs_a"] >= t and g["oos_sharpe"] is not None]
        n_pass = len(passing)
        avg_sharpe = np.mean([g["oos_sharpe"] for g in passing]) if passing else None
        avg_return = np.mean([g["oos_return"] for g in passing]) if passing else None
        names = ", ".join(g["group"][:15] for g in passing) if passing else "none"

        row = {
            "threshold": t,
            "n_passing": n_pass,
            "avg_oos_sharpe": round(avg_sharpe, 4) if avg_sharpe is not None else None,
            "avg_oos_return": round(avg_return, 2) if avg_return is not None else None,
            "groups": [g["group"] for g in passing],
        }
        table.append(row)

        sr_str = f"{avg_sharpe:.4f}" if avg_sharpe is not None else "  N/A"
        ret_str = f"{avg_return:.2f}%" if avg_return is not None else "  N/A"
        print(f"  {t:7.2f} {n_pass:7d} {sr_str:>8s} {ret_str:>8s} {names:>40s}")

    # All groups ranked
    print(f"\n  ALL GROUPS RANKED BY SCS-A:")
    print(f"  {'Group':35s} {'SCS-A':>7s} {'SCS-B':>7s} {'OOS SR':>8s} {'OOS Ret':>8s}")
    print("  " + "-" * 67)
    for g in groups:
        sr = f"{g['oos_sharpe']:.4f}" if g['oos_sharpe'] is not None else "  N/A"
        ret = f"{g['oos_return']:.2f}%" if g['oos_return'] is not None else "  N/A"
        print(f"  {g['group']:35s} {g['scs_a']:7.4f} {g['scs_b']:7.4f} {sr:>8s} {ret:>8s}")

    # Save
    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "experiment_3_threshold.json", "w") as f:
        json.dump({"threshold_table": table, "all_groups": groups}, f, indent=2, default=str)

    return {"threshold_table": table, "all_groups": groups}


if __name__ == "__main__":
    cfg = load_config()
    t_total = time.time()

    # Experiment 1: Expanded pipeline
    exp1 = experiment_1_expanded_pipeline(cfg, verbose=True)

    # Experiment 3: Threshold sensitivity (uses exp1 data, no extra computation)
    exp3 = experiment_3_threshold_sensitivity(exp1)

    # Experiment 2: Progressive corruption (computationally expensive)
    exp2 = experiment_2_corruption(cfg, verbose=True)

    print(f"\n\nALL EXPERIMENTS COMPLETED in {time.time()-t_total:.1f}s")

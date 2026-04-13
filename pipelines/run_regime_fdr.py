"""
Regime-Conditional FDR — runs FDR simulation with an alternative discovery window.

Tests whether the false positive rate differs between favorable (2010-2013)
and noisy (2015-2018) discovery windows.

Usage:
    PYTHONIOENCODING=utf-8 python pipelines/run_regime_fdr.py --n-seeds 100 --workers 16
"""

import sys
import json
import copy
import time
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import load_config
from pipelines.phase_a import run_phase_a

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

ALTERNATIVE_WINDOW = {
    "start": "2015-01-01",
    "end": "2018-12-31",
}


def run_single_seed(args):
    """Run Phase A with corruption=100% for one seed under alt window."""
    seed, cfg = args
    try:
        pa = run_phase_a(
            cfg=cfg, verbose=False,
            label_corruption_pct=100.0,
            corruption_seed=seed,
        )
        seed_results = {}
        for gk, res in pa["group_results"].items():
            seed_results[gk] = {
                "SCS_A": res.get("SCS_A", 0.0),
                "verdict": res.get("verdict", "UNKNOWN"),
                "mean_sharpe": res.get("mean_sharpe", 0.0),
            }
        return {"seed": seed, "status": "ok", "groups": seed_results}
    except Exception as e:
        return {"seed": seed, "status": "error", "error": str(e)}


def compute_fdr_summary(all_seed_results):
    """Compute FPR at each threshold."""
    if not all_seed_results:
        return {}

    sample = all_seed_results[0]["groups"]
    groups = sorted(sample.keys())
    binary_groups = [g for g in groups if "binary" in g or "directional" in g]
    ternary_groups = [g for g in groups if "multiclass" in g or "volatility" in g]

    summary = {}
    for tau in THRESHOLDS:
        tau_key = f"tau_{tau:.2f}"
        group_passes = {g: 0 for g in groups}
        n_valid = 0
        n_pass_per_seed = []

        for sr in all_seed_results:
            if sr["status"] != "ok":
                continue
            n_valid += 1
            n_pass = 0
            for g in groups:
                if sr["groups"][g]["SCS_A"] >= tau:
                    group_passes[g] += 1
                    n_pass += 1
            n_pass_per_seed.append(n_pass)

        if n_valid == 0:
            continue

        fpr_per_group = {g: group_passes[g] / n_valid for g in groups}
        binary_fprs = [fpr_per_group[g] for g in binary_groups]
        ternary_fprs = [fpr_per_group[g] for g in ternary_groups]

        ci_lo, ci_hi = np.percentile(n_pass_per_seed, [2.5, 97.5]) if n_pass_per_seed else (0, 0)

        summary[tau_key] = {
            "tau": tau,
            "mean_n_pass": round(float(np.mean(n_pass_per_seed)), 2),
            "ci_95_lo": round(float(ci_lo), 2),
            "ci_95_hi": round(float(ci_hi), 2),
            "mean_fpr_binary": round(float(np.mean(binary_fprs)), 4) if binary_fprs else None,
            "mean_fpr_ternary": round(float(np.mean(ternary_fprs)), 4) if ternary_fprs else None,
            "mean_fpr_all": round(float(np.mean(list(fpr_per_group.values()))), 4),
            "n_valid_seeds": n_valid,
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Regime-Conditional FDR")
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config()
    # Modify config for alternative discovery window
    cfg_alt = copy.deepcopy(cfg)
    cfg_alt["periods"]["phase_a"] = ALTERNATIVE_WINDOW

    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "regime_fdr_raw.json"
    summary_path = out_dir / "regime_fdr_summary.json"

    print("=" * 70)
    print("REGIME-CONDITIONAL FDR (2015-2018 window)")
    print(f"  Seeds: {args.n_seeds}")
    print(f"  Workers: {args.workers}")
    print("=" * 70)

    # Load existing
    existing = []
    completed_seeds = set()
    if raw_path.exists():
        with open(raw_path) as f:
            existing = json.load(f)
        completed_seeds = {r["seed"] for r in existing if r["status"] == "ok"}

    seeds_to_run = [s for s in range(args.n_seeds) if s not in completed_seeds]
    all_results = list(existing)

    t0 = time.time()
    if args.workers <= 1:
        for i, seed in enumerate(seeds_to_run):
            print(f"  [{i+1}/{len(seeds_to_run)}] Seed {seed}...")
            result = run_single_seed((seed, cfg_alt))
            all_results.append(result)
            with open(raw_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
    else:
        task_args = [(seed, cfg_alt) for seed in seeds_to_run]
        n_workers = min(args.workers, cpu_count(), len(seeds_to_run))
        with Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(run_single_seed, task_args)):
                all_results.append(result)
                if (i + 1) % 10 == 0:
                    with open(raw_path, "w") as f:
                        json.dump(all_results, f, indent=2, default=str)

    # Save final
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    ok_results = [r for r in all_results if r["status"] == "ok"]
    summary = compute_fdr_summary(ok_results)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Completed in {time.time() - t0:.1f}s")
    print(f"  Results: {summary_path}")


if __name__ == "__main__":
    main()

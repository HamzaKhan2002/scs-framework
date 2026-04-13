"""
Monte Carlo FDR Calibration — Run Phase A with corruption=100% over N seeds.

For each seed, runs the full Phase A pipeline with all labels randomized.
Records which groups pass at each threshold tau, enabling computation of
empirical false positive rates and false discovery rates.

Sequential version (safe for Windows/macOS). For VM parallelization,
use --workers N to spawn multiple processes via multiprocessing.Pool.

Usage:
    # Local test with 5 seeds, sequential
    PYTHONIOENCODING=utf-8 python pipelines/run_fdr_simulation.py --n-seeds 5

    # VM run with 1000 seeds, 32 workers
    PYTHONIOENCODING=utf-8 python pipelines/run_fdr_simulation.py --n-seeds 1000 --workers 32
"""

import argparse
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import load_config
from pipelines.phase_a import run_phase_a

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def run_single_seed(args):
    """Run Phase A with corruption=100% for a single seed. Returns per-group results."""
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
                "S_time": res.get("S_time", 0.0),
                "S_asset": res.get("S_asset", 0.0),
                "S_model": res.get("S_model", 0.0),
                "S_seed": res.get("S_seed", 0.0),
                "S_dist": res.get("S_dist", 0.0),
                "verdict": res.get("verdict", "UNKNOWN"),
                "mean_sharpe": res.get("mean_sharpe", 0.0),
            }
        return {"seed": seed, "status": "ok", "groups": seed_results}
    except Exception as e:
        return {"seed": seed, "status": "error", "error": str(e)}


def compute_fdr_summary(all_seed_results):
    """Compute FPR at each threshold, broken down by binary/ternary."""
    if not all_seed_results:
        return {}

    # Identify binary vs ternary groups
    sample = all_seed_results[0]["groups"]
    groups = sorted(sample.keys())
    binary_groups = [g for g in groups if "binary" in g or "directional" in g]
    ternary_groups = [g for g in groups if "multiclass" in g or "volatility" in g]

    n_seeds = len(all_seed_results)
    summary = {}

    for tau in THRESHOLDS:
        tau_key = f"tau_{tau:.2f}"

        # Count passes per group
        group_passes = {g: 0 for g in groups}
        for sr in all_seed_results:
            if sr["status"] != "ok":
                continue
            for g in groups:
                scs_a = sr["groups"][g]["SCS_A"]
                if scs_a >= tau:
                    group_passes[g] += 1

        # Compute rates
        n_valid = sum(1 for sr in all_seed_results if sr["status"] == "ok")
        fpr_per_group = {g: group_passes[g] / n_valid for g in groups}

        binary_fprs = [fpr_per_group[g] for g in binary_groups]
        ternary_fprs = [fpr_per_group[g] for g in ternary_groups]

        mean_n_pass = np.mean([
            sum(1 for g in groups if sr["groups"][g]["SCS_A"] >= tau)
            for sr in all_seed_results if sr["status"] == "ok"
        ])

        # 95% CI via bootstrap (percentile method)
        n_pass_per_seed = [
            sum(1 for g in groups if sr["groups"][g]["SCS_A"] >= tau)
            for sr in all_seed_results if sr["status"] == "ok"
        ]
        ci_lo, ci_hi = np.percentile(n_pass_per_seed, [2.5, 97.5]) if n_pass_per_seed else (0, 0)

        summary[tau_key] = {
            "tau": tau,
            "mean_n_pass": round(float(mean_n_pass), 2),
            "ci_95_lo": round(float(ci_lo), 2),
            "ci_95_hi": round(float(ci_hi), 2),
            "fpr_per_group": {g: round(v, 4) for g, v in fpr_per_group.items()},
            "mean_fpr_binary": round(float(np.mean(binary_fprs)), 4) if binary_fprs else None,
            "mean_fpr_ternary": round(float(np.mean(ternary_fprs)), 4) if ternary_fprs else None,
            "mean_fpr_all": round(float(np.mean(list(fpr_per_group.values()))), 4),
            "n_valid_seeds": n_valid,
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo FDR Calibration")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (1=sequential)")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed (for resumption)")
    args = parser.parse_args()

    cfg = load_config()
    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "fdr_simulation_raw.json"
    summary_path = out_dir / "fdr_summary.json"

    # Load existing results for resumption
    existing = []
    completed_seeds = set()
    if raw_path.exists():
        with open(raw_path) as f:
            existing = json.load(f)
        completed_seeds = {r["seed"] for r in existing if r["status"] == "ok"}
        print(f"  Loaded {len(completed_seeds)} completed seeds from previous run")

    seeds_to_run = [s for s in range(args.start_seed, args.start_seed + args.n_seeds)
                    if s not in completed_seeds]

    print("=" * 70)
    print("MONTE CARLO FDR CALIBRATION")
    print(f"  Total seeds requested: {args.n_seeds}")
    print(f"  Seeds remaining: {len(seeds_to_run)}")
    print(f"  Workers: {args.workers}")
    print(f"  Corruption: 100%")
    print("=" * 70)

    if not seeds_to_run:
        print("  All seeds already completed. Computing summary...")
        summary = compute_fdr_summary(existing)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print_summary(summary)
        return

    t0 = time.time()
    all_results = list(existing)

    if args.workers <= 1:
        # Sequential mode
        for i, seed in enumerate(seeds_to_run):
            print(f"\n  [{i+1}/{len(seeds_to_run)}] Running seed {seed}...")
            t1 = time.time()
            result = run_single_seed((seed, cfg))
            elapsed = time.time() - t1
            all_results.append(result)
            status = result["status"]
            print(f"  [{status.upper()}] Seed {seed} completed in {elapsed:.1f}s")

            # Save incrementally
            with open(raw_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
    else:
        # Parallel mode
        task_args = [(seed, cfg) for seed in seeds_to_run]
        n_workers = min(args.workers, cpu_count(), len(seeds_to_run))
        print(f"  Launching pool with {n_workers} workers...")

        with Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(run_single_seed, task_args)):
                all_results.append(result)
                status = result["status"]
                seed = result["seed"]
                print(f"  [{i+1}/{len(seeds_to_run)}] Seed {seed}: {status.upper()}")

                # Save incrementally every 10 seeds
                if (i + 1) % 10 == 0 or (i + 1) == len(seeds_to_run):
                    with open(raw_path, "w") as f:
                        json.dump(all_results, f, indent=2, default=str)
                    print(f"  Saved {len(all_results)} total results")

    total = time.time() - t0
    print(f"\n  All seeds completed in {total:.1f}s (wall clock)")

    # Compute and save summary
    ok_results = [r for r in all_results if r["status"] == "ok"]
    print(f"\n  Computing FDR summary from {len(ok_results)} valid seeds...")
    summary = compute_fdr_summary(ok_results)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save final raw results
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print_summary(summary)
    print(f"\n  Raw results: {raw_path}")
    print(f"  Summary: {summary_path}")


def print_summary(summary):
    """Print a human-readable FDR summary table."""
    print("\n" + "=" * 70)
    print("FDR CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"  {'Tau':>6s}  {'Mean N_pass':>11s}  {'95% CI':>12s}  {'FPR(bin)':>9s}  {'FPR(tern)':>10s}  {'FPR(all)':>9s}")
    print("  " + "-" * 62)

    for tau_key in sorted(summary.keys()):
        s = summary[tau_key]
        ci = f"[{s['ci_95_lo']:.1f}, {s['ci_95_hi']:.1f}]"
        fpr_b = f"{s['mean_fpr_binary']:.4f}" if s['mean_fpr_binary'] is not None else "---"
        fpr_t = f"{s['mean_fpr_ternary']:.4f}" if s['mean_fpr_ternary'] is not None else "---"
        print(f"  {s['tau']:>6.2f}  {s['mean_n_pass']:>11.2f}  {ci:>12s}  {fpr_b:>9s}  {fpr_t:>10s}  {s['mean_fpr_all']:>9.4f}")

    print(f"\n  N valid seeds: {summary[list(summary.keys())[0]]['n_valid_seeds']}")


if __name__ == "__main__":
    main()

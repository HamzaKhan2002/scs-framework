"""
Run ONLY Experiment 2: Progressive Label Corruption — SEQUENTIAL version.
Avoids ProcessPoolExecutor conflicts with joblib on Windows.

Each level takes ~75 min. Total ~9 hours for 7 levels.

Usage: PYTHONIOENCODING=utf-8 python pipelines/run_corruption_only.py
"""

import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import load_config
from pipelines.phase_a import run_phase_a


if __name__ == "__main__":
    cfg = load_config()
    corruption_levels = [0, 10, 20, 30, 50, 75, 100]

    print("=" * 70)
    print("EXPERIMENT 2: PROGRESSIVE LABEL CORRUPTION (SEQUENTIAL)")
    print(f"  Corruption levels: {corruption_levels}")
    print("=" * 70)

    t0 = time.time()
    all_results = {}
    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pct in corruption_levels:
        print(f"\n  --- Corruption level: {pct}% ---")
        t1 = time.time()
        pa = run_phase_a(cfg=cfg, verbose=False, label_corruption_pct=float(pct))
        elapsed = time.time() - t1

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
        print(f"  [DONE] Corruption {pct:>3d}% completed in {elapsed:.1f}s")

        # Save incrementally after each level (crash-safe)
        with open(out_dir / "experiment_2_corruption.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Saved {len(all_results)} levels so far")

    total = time.time() - t0
    print(f"\nAll levels completed in {total:.1f}s (wall clock)")

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

    print(f"\nResults saved to: {out_dir / 'experiment_2_corruption.json'}")

"""
Multi-Discovery-Window Analysis — tests SCS-A stability across different training periods.

Runs the full Phase A pipeline for 4 different discovery windows and compares
SCS-A distributions, pass rates, and component score stability.

Usage:
    # Local test (Phase A only, fast)
    PYTHONIOENCODING=utf-8 python pipelines/run_multi_discovery.py --phase-a-only

    # Full pipeline (A + B + C) on VM
    PYTHONIOENCODING=utf-8 python pipelines/run_multi_discovery.py --full
"""

import sys
import json
import copy
import time
import argparse
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import load_config
from pipelines.phase_a import run_phase_a
from pipelines.phase_b import run_phase_b
from pipelines.phase_c import run_phase_c


DISCOVERY_WINDOWS = [
    {
        "name": "2010-2013",
        "phase_a": {"start": "2010-01-01", "end": "2013-12-31"},
        "phase_b": {"start": "2014-01-01", "end": "2022-12-31"},
        "phase_c": {"start": "2023-01-01", "end": "2023-12-31"},
        "wf_windows": [
            {"train_start": "2014-01-01", "train_end": "2016-12-31", "test_start": "2017-01-01", "test_end": "2017-12-31"},
            {"train_start": "2014-01-01", "train_end": "2017-12-31", "test_start": "2018-01-01", "test_end": "2018-12-31"},
            {"train_start": "2014-01-01", "train_end": "2018-12-31", "test_start": "2019-01-01", "test_end": "2019-12-31"},
            {"train_start": "2014-01-01", "train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2020-12-31"},
            {"train_start": "2014-01-01", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
            {"train_start": "2014-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
        ],
    },
    {
        "name": "2011-2014",
        "phase_a": {"start": "2011-01-01", "end": "2014-12-31"},
        "phase_b": {"start": "2015-01-01", "end": "2022-12-31"},
        "phase_c": {"start": "2023-01-01", "end": "2023-12-31"},
        "wf_windows": [
            {"train_start": "2015-01-01", "train_end": "2017-12-31", "test_start": "2018-01-01", "test_end": "2018-12-31"},
            {"train_start": "2015-01-01", "train_end": "2018-12-31", "test_start": "2019-01-01", "test_end": "2019-12-31"},
            {"train_start": "2015-01-01", "train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2020-12-31"},
            {"train_start": "2015-01-01", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
            {"train_start": "2015-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
        ],
    },
    {
        "name": "2012-2015",
        "phase_a": {"start": "2012-01-01", "end": "2015-12-31"},
        "phase_b": {"start": "2016-01-01", "end": "2022-12-31"},
        "phase_c": {"start": "2023-01-01", "end": "2023-12-31"},
        "wf_windows": [
            {"train_start": "2016-01-01", "train_end": "2018-12-31", "test_start": "2019-01-01", "test_end": "2019-12-31"},
            {"train_start": "2016-01-01", "train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2020-12-31"},
            {"train_start": "2016-01-01", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
            {"train_start": "2016-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
        ],
    },
    {
        "name": "2013-2016",
        "phase_a": {"start": "2013-01-01", "end": "2016-12-31"},
        "phase_b": {"start": "2017-01-01", "end": "2022-12-31"},
        "phase_c": {"start": "2023-01-01", "end": "2023-12-31"},
        "wf_windows": [
            {"train_start": "2017-01-01", "train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2020-12-31"},
            {"train_start": "2017-01-01", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
            {"train_start": "2017-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
        ],
    },
]


def build_config_for_window(base_cfg: dict, window: dict) -> dict:
    """Create a modified config for a specific discovery window."""
    cfg = copy.deepcopy(base_cfg)
    cfg["periods"]["phase_a"] = window["phase_a"]
    cfg["periods"]["phase_b"] = window["phase_b"]
    cfg["periods"]["phase_c"] = window["phase_c"]
    cfg["splitting"]["phase_b"]["windows"] = window["wf_windows"]
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Multi-Discovery-Window Analysis")
    parser.add_argument("--phase-a-only", action="store_true", help="Run Phase A only (faster)")
    parser.add_argument("--full", action="store_true", help="Run full pipeline (A + B + C)")
    parser.add_argument("--windows", type=str, default=None,
                        help="Comma-separated window names to run (e.g., '2010-2013,2012-2015')")
    args = parser.parse_args()

    if not args.phase_a_only and not args.full:
        args.phase_a_only = True

    base_cfg = load_config()
    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "multi_discovery_window.json"

    # Filter windows if specified
    windows = DISCOVERY_WINDOWS
    if args.windows:
        names = [n.strip() for n in args.windows.split(",")]
        windows = [w for w in windows if w["name"] in names]

    print("=" * 70)
    print("MULTI-DISCOVERY-WINDOW ANALYSIS")
    print(f"  Windows: {[w['name'] for w in windows]}")
    print(f"  Mode: {'Full pipeline' if args.full else 'Phase A only'}")
    print("=" * 70)

    all_results = {}

    for window in windows:
        name = window["name"]
        print(f"\n{'='*70}")
        print(f"  DISCOVERY WINDOW: {name}")
        print(f"{'='*70}")

        cfg = build_config_for_window(base_cfg, window)
        t0 = time.time()

        # Phase A
        pa = run_phase_a(cfg=cfg, verbose=True)

        window_result = {
            "window": name,
            "phase_a": window["phase_a"],
            "phase_b": window["phase_b"],
            "phase_c": window["phase_c"],
            "n_wf_folds": len(window["wf_windows"]),
            "phase_a_results": {},
        }

        for gk, res in pa["group_results"].items():
            window_result["phase_a_results"][gk] = {
                "SCS_A": res.get("SCS_A", 0.0),
                "S_time": res.get("S_time", 0.0),
                "S_asset": res.get("S_asset", 0.0),
                "S_model": res.get("S_model", 0.0),
                "S_seed": res.get("S_seed", 0.0),
                "S_dist": res.get("S_dist", 0.0),
                "verdict": res.get("verdict", "UNKNOWN"),
                "mean_sharpe": res.get("mean_sharpe", 0.0),
            }

        n_pass = sum(1 for r in pa["group_results"].values()
                     if r.get("verdict") == "PHASE_B_APPROVED")
        scs_scores = [r["SCS_A"] for r in pa["group_results"].values()
                      if r.get("SCS_A", 0) > 0]

        window_result["summary"] = {
            "n_pass_070": n_pass,
            "mean_scs_a": round(float(np.mean(scs_scores)), 4) if scs_scores else 0.0,
            "min_scs_a": round(float(np.min(scs_scores)), 4) if scs_scores else 0.0,
            "max_scs_a": round(float(np.max(scs_scores)), 4) if scs_scores else 0.0,
        }

        # Phase B + C if full mode
        if args.full:
            pb = run_phase_b(cfg=cfg, verbose=True)
            window_result["phase_b_results"] = {}
            for gk, res in pb["group_results"].items():
                window_result["phase_b_results"][gk] = {
                    "SCS_B": res.get("SCS_B", 0.0),
                    "verdict": res.get("verdict", "UNKNOWN"),
                }

            pc = run_phase_c(
                cfg=cfg,
                trained_models=pb.get("trained_models"),
                approved_groups=pb.get("approved"),
                verbose=True,
            )
            window_result["phase_c_results"] = {}
            for gk, res in pc.get("results", {}).items():
                m = res.get("metrics", {})
                window_result["phase_c_results"][gk] = {
                    "sharpe": m.get("sharpe_ratio", 0.0),
                    "total_return_pct": m.get("total_return_pct", 0.0),
                }

        elapsed = time.time() - t0
        window_result["elapsed_seconds"] = round(elapsed, 1)
        all_results[name] = window_result

        print(f"\n  Window {name}: {n_pass}/12 pass, mean SCS-A = {window_result['summary']['mean_scs_a']:.4f}, {elapsed:.1f}s")

        # Save incrementally
        with open(result_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Window':<15s} {'N_pass':>7s} {'Mean SCS-A':>11s} {'Min SCS-A':>10s} {'Time':>8s}")
    print("  " + "-" * 55)
    for name, wr in all_results.items():
        s = wr["summary"]
        print(f"  {name:<15s} {s['n_pass_070']:>5d}/12 {s['mean_scs_a']:>11.4f} {s['min_scs_a']:>10.4f} {wr['elapsed_seconds']:>7.1f}s")

    print(f"\n  Results saved to {result_path}")


if __name__ == "__main__":
    main()

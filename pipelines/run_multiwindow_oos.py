"""
Multi-Window OOS Runner — Phase B + C for 2023, 2024, 2025.

Design:
  - Phase A is IDENTICAL for all three windows (2010-2013 discovery)
  - Phase B re-runs with expanding walk-forward:
      Window 1: train 2019-2022, test 2023
      Window 2: train 2019-2023, test 2024
      Window 3: train 2019-2024, test 2025 (= original pipeline)
  - Phase C tests on the respective OOS year

  Phase A results are loaded from the existing experiment_1 JSON.
  Only Phase B and C need to run for each window.
"""

import sys
import json
import copy
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config import load_config, ROOT_DIR
from pipelines.phase_b import run_phase_b
from pipelines.phase_c import run_phase_c


# ── OOS Window Definitions ──────────────────────────────────────────
OOS_WINDOWS = {
    "2023": {
        "phase_b": {
            "start": "2019-01-01",
            "end": "2022-12-31",
            "windows": [
                {
                    "train_start": "2019-01-01",
                    "train_end": "2020-12-31",
                    "test_start": "2021-01-01",
                    "test_end": "2021-12-31",
                },
                {
                    "train_start": "2019-01-01",
                    "train_end": "2021-12-31",
                    "test_start": "2022-01-01",
                    "test_end": "2022-12-31",
                },
            ],
        },
        "phase_c": {
            "start": "2023-01-01",
            "end": "2023-12-31",
        },
    },
    "2024": {
        "phase_b": {
            "start": "2019-01-01",
            "end": "2023-12-31",
            "windows": [
                {
                    "train_start": "2019-01-01",
                    "train_end": "2020-12-31",
                    "test_start": "2021-01-01",
                    "test_end": "2021-12-31",
                },
                {
                    "train_start": "2019-01-01",
                    "train_end": "2021-12-31",
                    "test_start": "2022-01-01",
                    "test_end": "2022-12-31",
                },
                {
                    "train_start": "2019-01-01",
                    "train_end": "2022-12-31",
                    "test_start": "2023-01-01",
                    "test_end": "2023-12-31",
                },
            ],
        },
        "phase_c": {
            "start": "2024-01-01",
            "end": "2024-12-31",
        },
    },
    "2025": {
        "phase_b": {
            "start": "2019-01-01",
            "end": "2024-12-31",
            "windows": [
                {
                    "train_start": "2019-01-01",
                    "train_end": "2021-12-31",
                    "test_start": "2022-01-01",
                    "test_end": "2022-12-31",
                },
                {
                    "train_start": "2019-01-01",
                    "train_end": "2022-12-31",
                    "test_start": "2023-01-01",
                    "test_end": "2023-12-31",
                },
                {
                    "train_start": "2019-01-01",
                    "train_end": "2023-12-31",
                    "test_start": "2024-01-01",
                    "test_end": "2024-12-31",
                },
            ],
        },
        "phase_c": {
            "start": "2025-01-01",
            "end": "2025-12-31",
        },
    },
}


def build_all_groups(cfg):
    """Build dict of ALL 12 signal groups for forced-through analysis."""
    groups = {}
    for h in cfg["search_space"]["horizons"]:
        for lm in cfg["search_space"]["label_modes"]:
            key = f"{h}d_{lm}"
            groups[key] = {"horizon": h, "label_mode": lm}
    return groups


def run_multiwindow():
    base_cfg = load_config()
    all_groups = build_all_groups(base_cfg)

    # Load existing Phase A results (identical for all windows)
    pa_path = ROOT_DIR / "results" / "experiments" / "experiment_1_expanded.json"
    with open(pa_path) as f:
        exp1 = json.load(f)
    phase_a_results = exp1["phase_a"]

    print("=" * 70)
    print("MULTI-WINDOW OOS EXPERIMENT")
    print("Phase A: 2010-2013 (identical, loaded from experiment_1)")
    print("=" * 70)

    all_window_results = {}

    for oos_year, window_def in OOS_WINDOWS.items():
        print(f"\n{'#' * 70}")
        print(f"  OOS WINDOW: {oos_year}")
        print(f"  Phase B: {window_def['phase_b']['start']} → {window_def['phase_b']['end']}")
        print(f"  Phase C: {window_def['phase_c']['start']} → {window_def['phase_c']['end']}")
        print(f"{'#' * 70}")

        # Build config override for this window
        cfg = copy.deepcopy(base_cfg)
        cfg["periods"]["phase_b"] = {
            "start": window_def["phase_b"]["start"],
            "end": window_def["phase_b"]["end"],
        }
        cfg["periods"]["phase_c"] = {
            "start": window_def["phase_c"]["start"],
            "end": window_def["phase_c"]["end"],
        }
        cfg["splitting"]["phase_b"]["windows"] = window_def["phase_b"]["windows"]

        t0 = time.time()

        # ── Phase B ──
        pb_result = run_phase_b(
            cfg=cfg,
            approved_groups=all_groups,  # Force all 12 through
            verbose=True,
        )

        # ── Phase C ──
        pc_result = run_phase_c(
            cfg=cfg,
            trained_models=pb_result.get("trained_models", {}),
            approved_groups=pb_result.get("trained_models", {}),
            verbose=True,
        )

        elapsed = time.time() - t0
        print(f"\n  Window {oos_year} completed in {elapsed:.1f}s")

        all_window_results[oos_year] = {
            "phase_a": phase_a_results,
            "phase_b": pb_result["group_results"],
            "phase_c": pc_result["results"],
            "time_seconds": round(elapsed, 1),
        }

        # Save incrementally
        out_path = ROOT_DIR / "results" / "experiments" / "multiwindow_oos.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_window_results, f, indent=2, default=str)
        print(f"  Saved incrementally to {out_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("MULTI-WINDOW OOS SUMMARY")
    print("=" * 70)

    # Approved groups from Phase A (same for all windows)
    scs_a_threshold = base_cfg["scs_a"]["threshold"]
    approved_keys = [k for k, v in phase_a_results.items() if v["SCS_A"] >= scs_a_threshold]
    print(f"\nPhase A approved (SCS-A >= {scs_a_threshold}): {approved_keys}")

    for oos_year in OOS_WINDOWS:
        res = all_window_results[oos_year]
        print(f"\n  OOS {oos_year}:")
        for gk in sorted(res["phase_c"].keys()):
            pc = res["phase_c"][gk]
            pb = res["phase_b"].get(gk, {})
            m = pc["metrics"]
            scs_b = pb.get("SCS_B", "N/A")
            scs_b_verdict = pb.get("verdict", "N/A")
            dsr_p = pc.get("statistics", {}).get("deflated_sharpe", {}).get("p_value", "N/A")
            lw_p = pc.get("statistics", {}).get("sharpe_test_vs_bh", {}).get("p_value", "N/A")
            tag = " *" if gk in approved_keys else ""
            print(f"    {gk:30s} SCS-B={scs_b:.4f} [{scs_b_verdict:18s}] "
                  f"SR={m['sharpe_ratio']:.4f} Ret={m['total_return_pct']:+.2f}% "
                  f"DSR_p={dsr_p} LW_p={lw_p}{tag}")


if __name__ == "__main__":
    run_multiwindow()

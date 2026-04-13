"""
Multi-Window Out-of-Sample Pipeline (v2).

Runs Phase B + Phase C for three expanding walk-forward windows,
reusing existing Phase A results (unchanged across all windows).

Windows (v2 — 2010-2013 discovery):
  W1: Phase B 2014-2020 (4 folds) → Phase C 2021
  W2: Phase B 2014-2021 (5 folds) → Phase C 2022
  W3: Phase B 2014-2022 (6 folds) → Phase C 2023

Phase A (2010-2013) is identical for all three windows.
SCS-B >= 0.60 gate applied uniformly.

Usage: PYTHONIOENCODING=utf-8 python pipelines/phase_c_multiwindow.py
"""

import sys
import copy
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config import load_config, ROOT_DIR
from pipelines.phase_b import run_phase_b
from pipelines.phase_c import run_phase_c


# ── Window definitions ──────────────────────────────────────────

WINDOWS = {
    "W1_2021": {
        "phase_b": {"start": "2014-01-01", "end": "2020-12-31"},
        "phase_c": {"start": "2021-01-01", "end": "2021-12-31"},
        "walk_forward": [
            {
                "train_start": "2014-01-01",
                "train_end": "2016-12-31",
                "test_start": "2017-01-01",
                "test_end": "2017-12-31",
            },
            {
                "train_start": "2014-01-01",
                "train_end": "2017-12-31",
                "test_start": "2018-01-01",
                "test_end": "2018-12-31",
            },
            {
                "train_start": "2014-01-01",
                "train_end": "2018-12-31",
                "test_start": "2019-01-01",
                "test_end": "2019-12-31",
            },
            {
                "train_start": "2014-01-01",
                "train_end": "2019-12-31",
                "test_start": "2020-01-01",
                "test_end": "2020-12-31",
            },
        ],
        "label": "2021",
    },
    "W2_2022": {
        "phase_b": {"start": "2014-01-01", "end": "2021-12-31"},
        "phase_c": {"start": "2022-01-01", "end": "2022-12-31"},
        "walk_forward": [
            {
                "train_start": "2014-01-01",
                "train_end": "2016-12-31",
                "test_start": "2017-01-01",
                "test_end": "2017-12-31",
            },
            {
                "train_start": "2014-01-01",
                "train_end": "2017-12-31",
                "test_start": "2018-01-01",
                "test_end": "2018-12-31",
            },
            {
                "train_start": "2014-01-01",
                "train_end": "2018-12-31",
                "test_start": "2019-01-01",
                "test_end": "2019-12-31",
            },
            {
                "train_start": "2014-01-01",
                "train_end": "2019-12-31",
                "test_start": "2020-01-01",
                "test_end": "2020-12-31",
            },
            {
                "train_start": "2014-01-01",
                "train_end": "2020-12-31",
                "test_start": "2021-01-01",
                "test_end": "2021-12-31",
            },
        ],
        "label": "2022",
    },
    "W3_2023": {
        "phase_b": {"start": "2014-01-01", "end": "2022-12-31"},
        "phase_c": {"start": "2023-01-01", "end": "2023-12-31"},
        "walk_forward": [
            {
                "train_start": "2014-01-01",
                "train_end": "2016-12-31",
                "test_start": "2017-01-01",
                "test_end": "2017-12-31",
            },
            {
                "train_start": "2014-01-01",
                "train_end": "2017-12-31",
                "test_start": "2018-01-01",
                "test_end": "2018-12-31",
            },
            {
                "train_start": "2014-01-01",
                "train_end": "2018-12-31",
                "test_start": "2019-01-01",
                "test_end": "2019-12-31",
            },
            {
                "train_start": "2014-01-01",
                "train_end": "2019-12-31",
                "test_start": "2020-01-01",
                "test_end": "2020-12-31",
            },
            {
                "train_start": "2014-01-01",
                "train_end": "2020-12-31",
                "test_start": "2021-01-01",
                "test_end": "2021-12-31",
            },
            {
                "train_start": "2014-01-01",
                "train_end": "2021-12-31",
                "test_start": "2022-01-01",
                "test_end": "2022-12-31",
            },
        ],
        "label": "2023",
    },
}


def load_phase_a_results():
    """Load existing Phase A results from experiment_1_expanded.json."""
    path = ROOT_DIR / "results" / "experiments" / "experiment_1_expanded.json"
    with open(path) as f:
        data = json.load(f)
    return data["phase_a"]


def make_all_groups(phase_a):
    """Build all 12 signal groups from Phase A results."""
    groups = {}
    for gk, res in phase_a.items():
        parts = gk.split("d_", 1)
        groups[gk] = {
            "horizon": int(parts[0]),
            "label_mode": parts[1],
        }
    return groups


def override_config(base_cfg, window_def):
    """Create a config copy with overridden periods and walk-forward windows."""
    cfg = copy.deepcopy(base_cfg)
    cfg["periods"]["phase_b"] = window_def["phase_b"]
    cfg["periods"]["phase_c"] = window_def["phase_c"]
    cfg["splitting"]["phase_b"]["windows"] = window_def["walk_forward"]
    return cfg


def clean_for_json(obj):
    """Recursively clean objects for JSON serialization."""
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


def run_multiwindow_oos(verbose=True):
    """Run multi-window OOS pipeline."""
    base_cfg = load_config()
    phase_a = load_phase_a_results()
    all_groups = make_all_groups(phase_a)

    all_window_results = {}
    scs_b_gate = 0.60

    print("=" * 70)
    print("MULTI-WINDOW OUT-OF-SAMPLE PIPELINE")
    print("=" * 70)
    print(f"  Phase A: loaded from experiment_1_expanded.json (12 groups)")
    print(f"  SCS-B gate: >= {scs_b_gate}")
    print()

    for window_name, window_def in WINDOWS.items():
        print(f"\n{'='*70}")
        print(f"  WINDOW: {window_name} (OOS year: {window_def['label']})")
        print(f"  Phase B: {window_def['phase_b']['start']} -> {window_def['phase_b']['end']}")
        print(f"  Phase C: {window_def['phase_c']['start']} -> {window_def['phase_c']['end']}")
        print(f"  Walk-forward folds: {len(window_def['walk_forward'])}")
        print(f"{'='*70}")

        cfg = override_config(base_cfg, window_def)

        # ── Phase B ──
        t0 = time.time()
        pb = run_phase_b(cfg=cfg, approved_groups=all_groups, verbose=verbose)
        pb_time = time.time() - t0
        print(f"\n  Phase B completed in {pb_time:.1f}s")

        # Apply SCS-B gate
        approved_b = {k: v for k, v in pb["group_results"].items()
                      if v.get("SCS_B", 0) >= scs_b_gate}
        print(f"  SCS-B approved: {len(approved_b)} / {len(pb['group_results'])}")
        for gk, res in sorted(approved_b.items(), key=lambda x: -x[1]["SCS_B"]):
            print(f"    {gk:35s} SCS-B={res['SCS_B']:.4f}")

        # Filter trained_models to approved groups only
        trained_approved = {k: v for k, v in pb.get("trained_models", {}).items()
                           if k in approved_b}

        if not trained_approved:
            print(f"  [WARN] No groups passed SCS-B gate for {window_name}")
            all_window_results[window_name] = {
                "phase_b": pb["group_results"],
                "phase_c": {},
                "approved_groups": [],
                "oos_year": window_def["label"],
                "n_wf_folds": len(window_def["walk_forward"]),
            }
            continue

        # ── Phase C ──
        t0 = time.time()
        pc = run_phase_c(
            cfg=cfg,
            trained_models=trained_approved,
            approved_groups=trained_approved,
            verbose=verbose,
        )
        pc_time = time.time() - t0
        print(f"\n  Phase C completed in {pc_time:.1f}s")

        all_window_results[window_name] = {
            "phase_b": clean_for_json(pb["group_results"]),
            "phase_c": clean_for_json(pc["results"]),
            "approved_groups": list(approved_b.keys()),
            "oos_year": window_def["label"],
            "n_wf_folds": len(window_def["walk_forward"]),
        }

    # ── Save results ──
    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment_multiwindow_oos.json"
    with open(out_path, "w") as f:
        json.dump(clean_for_json(all_window_results), f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    # ── Save per-window JSONs ──
    pc_dir = ROOT_DIR / "results" / "phase_c"
    pc_dir.mkdir(parents=True, exist_ok=True)
    for wn, wr in all_window_results.items():
        year = wr["oos_year"]
        wpath = pc_dir / f"window_{year}.json"
        with open(wpath, "w") as f:
            json.dump(clean_for_json(wr), f, indent=2, default=str)
        print(f"  Saved: {wpath}")

    # ── Spearman correlation at N=36 ──
    compute_spearman_n36(phase_a, all_window_results)

    # ── Summary table ──
    print_summary(phase_a, all_window_results)

    return all_window_results


def compute_spearman_n36(phase_a, all_window_results):
    """Compute Spearman between SCS-A and OOS Sharpe across all windows (N=36)."""
    print(f"\n{'='*70}")
    print("SPEARMAN CORRELATION: SCS-A vs OOS Sharpe (N=36)")
    print(f"{'='*70}")

    scs_a_vals = []
    oos_sharpe_vals = []

    for wn, wr in all_window_results.items():
        pc = wr.get("phase_c", {})
        for gk, res in phase_a.items():
            scs_a = res.get("SCS_A", 0.0)
            oos_sr = pc.get(gk, {}).get("metrics", {}).get("sharpe_ratio", None)
            if oos_sr is not None:
                scs_a_vals.append(scs_a)
                oos_sharpe_vals.append(oos_sr)

    n = len(scs_a_vals)
    print(f"  Observations: N = {n}")

    if n >= 3:
        rho, pval = spearmanr(scs_a_vals, oos_sharpe_vals)
        print(f"  All observations: rho = {rho:.4f}, p = {pval:.4f}")

        # Non-zero SCS-A only
        nz_scs = []
        nz_sr = []
        for s, r in zip(scs_a_vals, oos_sharpe_vals):
            if s > 0:
                nz_scs.append(s)
                nz_sr.append(r)
        if len(nz_scs) >= 3:
            rho_nz, pval_nz = spearmanr(nz_scs, nz_sr)
            print(f"  Non-zero only (N={len(nz_scs)}): rho = {rho_nz:.4f}, p = {pval_nz:.4f}")
    else:
        print("  [WARN] Too few observations for Spearman")


def print_summary(phase_a, all_window_results):
    """Print a summary table of results across all windows."""
    print(f"\n{'='*70}")
    print("MULTI-WINDOW OOS SUMMARY")
    print(f"{'='*70}")

    # Per-window summary
    for wn, wr in all_window_results.items():
        year = wr["oos_year"]
        folds = wr["n_wf_folds"]
        pc = wr.get("phase_c", {})
        approved = wr.get("approved_groups", [])

        print(f"\n  {wn} (OOS={year}, folds={folds}, approved={len(approved)})")
        print(f"  {'Group':35s} {'SCS-A':>7s} {'SCS-B':>7s} {'OOS SR':>8s} {'Return':>8s} {'MaxDD':>8s} {'DSR p':>7s} {'LW p':>7s}")
        print(f"  {'-'*91}")

        pb = wr.get("phase_b", {})
        for gk in sorted(pc.keys()):
            m = pc[gk].get("metrics", {})
            stats = pc[gk].get("statistics", {})
            scs_a = phase_a.get(gk, {}).get("SCS_A", 0.0)
            scs_b = pb.get(gk, {}).get("SCS_B", 0.0)
            sr = m.get("sharpe_ratio", 0.0)
            ret = m.get("total_return_pct", 0.0)
            dd = m.get("max_drawdown_pct", 0.0)
            dsr_p = stats.get("deflated_sharpe", {}).get("p_value", None)
            lw_p = stats.get("sharpe_test_vs_bh", {}).get("p_value", None)
            dsr_s = f"{dsr_p:.4f}" if dsr_p is not None else "N/A"
            lw_s = f"{lw_p:.4f}" if lw_p is not None else "N/A"
            print(f"  {gk:35s} {scs_a:7.3f} {scs_b:7.3f} {sr:8.4f} {ret:+7.2f}% {dd:7.2f}% {dsr_s:>7s} {lw_s:>7s}")


if __name__ == "__main__":
    run_multiwindow_oos(verbose=True)

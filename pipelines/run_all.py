"""
Master Pipeline — runs Phase A → B → C end-to-end.

Single entry point: python -m pipelines.run_all
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config import load_config, ROOT_DIR
from pipelines.phase_a import run_phase_a
from pipelines.phase_b import run_phase_b
from pipelines.phase_c import run_phase_c


def run_all(verbose: bool = True):
    """Run the complete A → B → C pipeline."""
    cfg = load_config()
    t0 = time.time()

    print("╔" + "═" * 68 + "╗")
    print("║  ML TRADING SIGNAL VALIDATION — PUBLICATION-GRADE PIPELINE       ║")
    print("╚" + "═" * 68 + "╝")
    print()

    # ---- PHASE A ----
    ta = time.time()
    phase_a_result = run_phase_a(cfg=cfg, verbose=verbose)
    print(f"\n  Phase A completed in {time.time()-ta:.1f}s")

    approved_a = phase_a_result["approved"]
    if not approved_a:
        print("\n  !! NO SIGNALS APPROVED IN PHASE A — PIPELINE STOPS")
        print("  This is a valid scientific result: the SCS framework correctly")
        print("  identifies that no robust signal exists in the search space.")
        _save_final(phase_a_result, None, None, time.time()-t0)
        return

    print(f"\n  Approved for Phase B: {list(approved_a.keys())}")

    # ---- PHASE B ----
    tb = time.time()
    phase_b_result = run_phase_b(
        cfg=cfg,
        approved_groups=approved_a,
        verbose=verbose,
    )
    print(f"\n  Phase B completed in {time.time()-tb:.1f}s")

    approved_b = phase_b_result["approved"]
    if not approved_b:
        print("\n  !! NO SIGNALS APPROVED IN PHASE B — PIPELINE STOPS")
        print("  Signals passed Phase A but failed walk-forward validation.")
        _save_final(phase_a_result, phase_b_result, None, time.time()-t0)
        return

    print(f"\n  Approved for Phase C: {list(approved_b.keys())}")

    # ---- PHASE C ----
    tc = time.time()
    phase_c_result = run_phase_c(
        cfg=cfg,
        trained_models=phase_b_result.get("trained_models", {}),
        approved_groups=approved_b,
        verbose=verbose,
    )
    print(f"\n  Phase C completed in {time.time()-tc:.1f}s")

    # ---- FINAL SUMMARY ----
    total_time = time.time() - t0
    _print_final_summary(phase_a_result, phase_b_result, phase_c_result, total_time)
    _save_final(phase_a_result, phase_b_result, phase_c_result, total_time)


def _print_final_summary(pa, pb, pc, total_time):
    print("\n" + "=" * 70)
    print("FINAL PIPELINE SUMMARY")
    print("=" * 70)

    # Phase A
    print("\n  PHASE A — Signal Discovery")
    for gk, res in pa["group_results"].items():
        status = "✓" if res.get("verdict") == "PHASE_B_APPROVED" else "✗"
        print(f"    {status} {gk:30s} SCS-A = {res['SCS_A']:.4f}  [{res['verdict']}]")

    # Phase B
    if pb:
        print("\n  PHASE B — Walk-Forward Validation")
        for gk, res in pb["group_results"].items():
            status = "✓" if res.get("verdict") == "VALID_FOR_PHASE_C" else "✗"
            print(f"    {status} {gk:30s} SCS-B = {res['SCS_B']:.4f}  [{res['verdict']}]")

    # Phase C
    if pc:
        print("\n  PHASE C — Out-of-Sample Results")
        for gk, res in pc["results"].items():
            m = res["metrics"]
            stats = res.get("statistics", {})
            dsr = stats.get("deflated_sharpe", {})
            sig = "SIG" if dsr.get("significant", False) else "N/S"
            print(f"    {gk}:")
            print(f"      Return = {m['total_return_pct']:+.2f}%  |  "
                  f"Sharpe = {m['sharpe_ratio']:.4f}  |  "
                  f"Max DD = {m['max_drawdown_pct']:.2f}%")
            print(f"      Trades = {m['n_trades']} (L:{m['n_long']} S:{m['n_short']})  |  "
                  f"Win Rate = {m['win_rate']:.1f}%  |  "
                  f"DSR = {dsr.get('deflated', 'N/A')} ({sig})")

    print(f"\n  Total pipeline time: {total_time:.1f}s")
    print("=" * 70)


def _save_final(pa, pb, pc, total_time):
    out = {
        "phase_a": pa,
        "phase_b": pb if pb else None,
        "phase_c": pc if pc else None,
        "total_time_seconds": round(total_time, 1),
    }
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Clean non-serializable objects
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()
                    if k != "trained_models"}  # Skip model objects
        elif isinstance(obj, list):
            return [clean(i) for i in obj]
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    with open(results_dir / "pipeline_final.json", "w") as f:
        json.dump(clean(out), f, indent=2, default=str)


if __name__ == "__main__":
    run_all(verbose=True)

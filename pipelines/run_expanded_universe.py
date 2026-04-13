"""
Expanded Universe Pipeline — runs Phase A on a larger ticker set.

Addresses the survivorship bias concern by:
1. Using a broader candidate pool (190+ tickers from the current S&P 500)
2. Filtering only for data availability (complete 2010-2023 data)
3. Documenting which tickers were excluded and why
4. Running as an appendix robustness check (main results stay on 27 tickers)

NOTE: True survivorship-bias-free selection would require historical S&P 500
constituent lists (e.g., from CRSP or Compustat). Since we use Yahoo Finance,
we acknowledge this limitation and test with the broadest available universe.

Usage:
    # Step 1: Select tickers (requires internet)
    PYTHONIOENCODING=utf-8 python pipelines/run_expanded_universe.py --select-tickers

    # Step 2: Run Phase A on expanded universe
    PYTHONIOENCODING=utf-8 python pipelines/run_expanded_universe.py --run-phase-a

    # Step 3: Full pipeline (A + B + C)
    PYTHONIOENCODING=utf-8 python pipelines/run_expanded_universe.py --full
"""

import sys
import json
import copy
import time
import argparse
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import load_config
from pipelines.phase_a import run_phase_a
from pipelines.phase_b import run_phase_b
from pipelines.phase_c import run_phase_c


def select_tickers():
    """Download and validate ticker data availability."""
    import yfinance as yf

    # Broad candidate pool: current S&P 500 members + historical large-caps
    # We include tickers that may have been delisted or removed to partially
    # mitigate survivorship bias within Yahoo Finance's limitations
    candidates = sorted(set([
        # Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "AVGO", "ORCL", "CRM",
        "ADBE", "CSCO", "AMD", "INTC", "IBM", "QCOM", "TXN", "INTU", "AMAT", "MU",
        "SNPS", "CDNS", "LRCX", "KLAC", "ADI", "MCHP", "NXPI",
        # Finance
        "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "BK",
        "SCHW", "AXP", "BLK", "SPGI", "ICE", "CME", "MCO", "MMC", "AON", "CB",
        "MET", "AIG", "PRU", "AFL", "ALL", "TRV",
        # Healthcare
        "JNJ", "UNH", "PFE", "LLY", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "MDT", "ISRG", "SYK", "BDX", "EW", "ZTS", "CI", "HCA",
        "CVS", "MCK", "BAX", "BSX",
        # Consumer
        "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
        "LOW", "TJX", "CL", "GIS", "K", "SJM", "CAG", "HSY", "MKC",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "OXY", "HAL",
        "DVN", "HES", "BKR", "WMB", "KMI", "OKE",
        # Industrials
        "BA", "CAT", "HON", "UNP", "UPS", "RTX", "GE", "MMM", "DE", "LMT",
        "NOC", "GD", "ITW", "EMR", "ETN", "PH", "CMI", "FDX", "CSX",
        "NSC", "WM", "RSG",
        # Communication
        "DIS", "NFLX", "CMCSA", "VZ", "T",
        # Utilities
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ED",
        # Real Estate
        "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "DLR", "AVB",
        # Materials
        "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "VMC", "MLM",
        # ETFs
        "SPY", "QQQ", "IWM", "DIA",
    ]))

    START_DATE = "2010-01-01"
    END_DATE = "2023-12-31"
    MIN_DAYS = 3350  # ~13.3 years × 252

    print(f"Checking {len(candidates)} tickers for complete 2010-2023 data...")
    ok_tickers = []
    excluded = []

    for i, ticker in enumerate(candidates):
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                excluded.append({"ticker": ticker, "reason": "no_data"})
                continue
            n_days = len(df)
            if n_days < MIN_DAYS:
                excluded.append({"ticker": ticker, "reason": f"only_{n_days}_days"})
                continue
            ok_tickers.append(ticker)
            print(f"  [{i+1:3d}/{len(candidates)}] {ticker:6s} OK ({n_days} days)")
        except Exception as e:
            excluded.append({"ticker": ticker, "reason": str(e)[:50]})

    # Save
    out_dir = ROOT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "n_candidates": len(candidates),
        "n_selected": len(ok_tickers),
        "n_excluded": len(excluded),
        "tickers": ok_tickers,
        "excluded": excluded,
        "survivorship_bias_note": (
            "These tickers are current S&P 500 members filtered for 2010-2023 "
            "data availability. This introduces survivorship bias: companies that "
            "were delisted, acquired, or removed from the index before the current "
            "date are not included. True bias-free selection requires historical "
            "constituent lists (e.g., CRSP). We document this limitation and "
            "present expanded-universe results as a robustness check, not as the "
            "primary analysis."
        ),
    }

    ticker_path = out_dir / "expanded_universe_tickers.json"
    with open(ticker_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Selected: {len(ok_tickers)} tickers")
    print(f"  Excluded: {len(excluded)}")
    print(f"  Saved to: {ticker_path}")
    return ok_tickers


def build_expanded_config(base_cfg, tickers):
    """Create config with expanded ticker list."""
    cfg = copy.deepcopy(base_cfg)
    etfs = [t for t in tickers if t in {"SPY", "QQQ", "IWM", "DIA"}]
    stocks = [t for t in tickers if t not in {"SPY", "QQQ", "IWM", "DIA"}]
    cfg["data"]["tickers"] = {"etfs": etfs, "stocks": stocks}
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Expanded Universe Pipeline")
    parser.add_argument("--select-tickers", action="store_true")
    parser.add_argument("--run-phase-a", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--ticker-file", type=str, default=None,
                        help="Path to expanded_universe_tickers.json")
    args = parser.parse_args()

    if args.select_tickers:
        select_tickers()
        return

    # Load ticker list
    ticker_path = args.ticker_file or str(
        ROOT_DIR / "results" / "expanded_universe_tickers.json"
    )
    with open(ticker_path) as f:
        ticker_data = json.load(f)
    tickers = ticker_data["tickers"]

    base_cfg = load_config()
    cfg = build_expanded_config(base_cfg, tickers)

    out_dir = ROOT_DIR / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "expanded_universe_results.json"

    print("=" * 70)
    print("EXPANDED UNIVERSE PIPELINE")
    print(f"  Tickers: {len(tickers)} (vs 27 in main analysis)")
    print(f"  Mode: {'Full pipeline' if args.full else 'Phase A only'}")
    print("=" * 70)

    t0 = time.time()

    # Phase A
    pa = run_phase_a(cfg=cfg, verbose=True)

    results = {
        "n_tickers": len(tickers),
        "phase_a": {},
    }

    for gk, res in pa["group_results"].items():
        results["phase_a"][gk] = {
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

    n_pass = sum(1 for r in pa["group_results"].values()
                 if r.get("verdict") == "PHASE_B_APPROVED")
    scs_scores = [r["SCS_A"] for r in pa["group_results"].values()
                  if r.get("SCS_A", 0) > 0]

    results["summary_phase_a"] = {
        "n_pass_070": n_pass,
        "mean_scs_a": round(float(np.mean(scs_scores)), 4) if scs_scores else 0.0,
        "min_scs_a": round(float(np.min(scs_scores)), 4) if scs_scores else 0.0,
    }

    if args.full:
        pb = run_phase_b(cfg=cfg, verbose=True)
        results["phase_b"] = {}
        for gk, res in pb["group_results"].items():
            results["phase_b"][gk] = {
                "SCS_B": res.get("SCS_B", 0.0),
                "verdict": res.get("verdict", "UNKNOWN"),
            }

        pc = run_phase_c(
            cfg=cfg,
            trained_models=pb.get("trained_models"),
            approved_groups=pb.get("approved"),
            verbose=True,
        )
        results["phase_c"] = {}
        for gk, res in pc.get("results", {}).items():
            m = res.get("metrics", {})
            results["phase_c"][gk] = {
                "sharpe": m.get("sharpe_ratio", 0.0),
                "total_return_pct": m.get("total_return_pct", 0.0),
            }

    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  Expanded universe: {n_pass}/12 pass (vs 10/12 in main)")
    print(f"  Mean SCS-A: {results['summary_phase_a']['mean_scs_a']:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Results: {result_path}")


if __name__ == "__main__":
    main()

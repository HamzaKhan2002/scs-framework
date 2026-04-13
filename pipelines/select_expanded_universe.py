"""
Expanded Universe Ticker Selection — Find S&P 500 stocks with complete
daily data from 2010-01-01 to 2023-12-31 on Yahoo Finance.

Outputs a YAML-compatible ticker list for config.yaml.

Usage:
    PYTHONIOENCODING=utf-8 python pipelines/select_expanded_universe.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import yfinance as yf
import pandas as pd

# S&P 500 tickers — large, representative sample
# Source: Wikipedia S&P 500 list (we hardcode the most common ones)
SP500_CANDIDATES = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AVGO", "ORCL", "CRM",
    "ADBE", "CSCO", "AMD", "INTC", "IBM", "QCOM", "TXN", "INTU", "AMAT", "MU",
    "NOW", "SNPS", "CDNS", "LRCX", "KLAC", "ADI", "MCHP", "NXPI", "FTNT", "PANW",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "BK",
    "SCHW", "AXP", "BLK", "SPGI", "ICE", "CME", "MCO", "MMC", "AON", "CB",
    "MET", "AIG", "PRU", "AFL", "ALL", "TRV",
    # Healthcare
    "JNJ", "UNH", "PFE", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "MDT", "ISRG", "SYK", "BDX", "EW", "ZTS", "CI", "HCA",
    "CVS", "MCK", "HUM", "CNC", "BAX", "BSX",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
    "LOW", "TJX", "EL", "CL", "GIS", "K", "SJM", "CAG", "HSY", "MKC",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "OXY", "HAL",
    "DVN", "HES", "BKR", "FANG", "WMB", "KMI", "OKE",
    # Industrials
    "BA", "CAT", "HON", "UNP", "UPS", "RTX", "GE", "MMM", "DE", "LMT",
    "NOC", "GD", "ITW", "EMR", "ETN", "PH", "ROK", "CMI", "FDX", "CSX",
    "NSC", "WM", "RSG", "IR",
    # Communication
    "DIS", "NFLX", "CMCSA", "TMUS", "VZ", "T", "CHTR", "OMC",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ED",
    # Real Estate (REITs)
    "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "DLR", "WELL", "AVB",
    # Materials
    "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "VMC", "MLM",
    # ETFs (keep current ones)
    "SPY", "QQQ", "IWM", "DIA",
]

START_DATE = "2010-01-01"
END_DATE = "2023-12-31"
MIN_TRADING_DAYS = 3400  # ~13.5 years × 252 days = 3402, with some tolerance


def check_ticker(ticker: str) -> dict:
    """Check if a ticker has complete data from 2010-2023."""
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return {"ticker": ticker, "ok": False, "reason": "no_data", "days": 0}

        n_days = len(df)
        first = df.index[0].strftime("%Y-%m-%d")
        last = df.index[-1].strftime("%Y-%m-%d")

        if n_days < MIN_TRADING_DAYS:
            return {"ticker": ticker, "ok": False, "reason": f"only_{n_days}_days",
                    "days": n_days, "first": first, "last": last}

        # Check for gaps > 10 days (excluding weekends)
        gaps = df.index.to_series().diff().dt.days
        max_gap = gaps.max()

        return {"ticker": ticker, "ok": True, "days": n_days,
                "first": first, "last": last, "max_gap_days": int(max_gap)}
    except Exception as e:
        return {"ticker": ticker, "ok": False, "reason": str(e), "days": 0}


def main():
    candidates = sorted(set(SP500_CANDIDATES))
    print(f"Checking {len(candidates)} tickers for complete 2010-2023 data...")
    print(f"Minimum trading days: {MIN_TRADING_DAYS}")
    print("=" * 70)

    results = []
    for i, ticker in enumerate(candidates):
        result = check_ticker(ticker)
        results.append(result)
        status = "OK" if result["ok"] else "SKIP"
        days = result["days"]
        reason = result.get("reason", "")
        print(f"  [{i+1:3d}/{len(candidates)}] {ticker:6s} {status:4s} ({days} days) {reason}")

    ok_tickers = [r["ticker"] for r in results if r["ok"]]
    skip_tickers = [r for r in results if not r["ok"]]

    print("\n" + "=" * 70)
    print(f"RESULT: {len(ok_tickers)} tickers with complete 2010-2023 data")
    print(f"SKIPPED: {len(skip_tickers)} tickers")
    print("=" * 70)

    # Separate ETFs from stocks
    etfs = [t for t in ok_tickers if t in {"SPY", "QQQ", "IWM", "DIA"}]
    stocks = [t for t in ok_tickers if t not in {"SPY", "QQQ", "IWM", "DIA"}]

    print(f"\nETFs ({len(etfs)}): {etfs}")
    print(f"Stocks ({len(stocks)}): {stocks}")

    # Output YAML format
    print("\n# YAML config format:")
    print("data:")
    print(f"  tickers:")
    print(f"    etfs: [{', '.join(etfs)}]")

    # Format stocks in rows of 8
    print(f"    stocks: [{stocks[0]},")
    for i in range(1, len(stocks), 8):
        chunk = stocks[i:i+8]
        line = ", ".join(chunk)
        if i + 8 < len(stocks):
            print(f"             {line},")
        else:
            print(f"             {line}]")

    # Save to file
    out_path = Path(__file__).resolve().parent.parent / "results" / "expanded_universe_tickers.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# Expanded universe: {len(ok_tickers)} tickers with complete 2010-2023 data\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# ETFs: {', '.join(etfs)}\n")
        f.write(f"# Stocks: {', '.join(stocks)}\n")
        for t in ok_tickers:
            f.write(f"{t}\n")
    print(f"\nTicker list saved to: {out_path}")


if __name__ == "__main__":
    main()

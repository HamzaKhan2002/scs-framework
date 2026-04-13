"""
Data loader — download OHLCV from Yahoo Finance with local parquet caching.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def _cache_path(cache_dir: Path, ticker: str, start: str, end: str) -> Path:
    return cache_dir / f"{ticker}_{start}_{end}.parquet"


def load_single_ticker(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load daily OHLCV for one ticker. Cache to parquet if cache_dir given.
    Returns DataFrame with columns: open, high, low, close, volume.
    Index: DatetimeIndex (date only, no time).
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = _cache_path(cache_dir, ticker, start, end)
        if cp.exists():
            df = pd.read_parquet(cp)
            if len(df) > 0:
                return df

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker} ({start} to {end})")

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()

    # Drop any rows where OHLCV is all NaN
    df = df.dropna(how="all")

    if cache_dir is not None:
        df.to_parquet(_cache_path(cache_dir, ticker, start, end))

    return df


def load_universe(
    tickers: List[str],
    start: str,
    end: str,
    cache_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all tickers. Returns dict {ticker: DataFrame}.
    Aligns to common trading dates (inner join).
    """
    data = {}
    failed = []
    for t in tickers:
        try:
            data[t] = load_single_ticker(t, start, end, cache_dir)
        except Exception as e:
            failed.append((t, str(e)))

    if failed:
        print(f"  [WARN] Failed to load {len(failed)} tickers: {[f[0] for f in failed]}")

    if not data:
        raise ValueError("No data loaded for any ticker")

    # Align to common dates
    common_dates = sorted(set.intersection(*[set(df.index) for df in data.values()]))
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates across tickers")

    aligned = {}
    for t, df in data.items():
        aligned[t] = df.loc[df.index.isin(common_dates)].copy()

    return aligned

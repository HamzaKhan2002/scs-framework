"""
Config loader — single source of truth.
Loads config.yaml and provides typed access to all parameters.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load config.yaml and return raw dict."""
    if path is None:
        path = ROOT_DIR / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class FeatureConfig:
    return_lags: tuple = (1, 2, 3)
    volatility_windows: tuple = (5, 20)
    atr_window: int = 14
    volume_z_window: int = 20


@dataclass(frozen=True)
class LabelConfig:
    horizon: int = 10
    mode: str = "multiclass_volatility"
    threshold_mult: float = 0.5
    vol_window: int = 20


@dataclass(frozen=True)
class SplitConfig:
    embargo_days: int = 20
    train_ratio: float = 0.70


@dataclass(frozen=True)
class PortfolioConfig:
    horizon: int = 10
    max_positions: int = 15
    capital_per_trade_pct: float = 10.0
    max_exposure_per_ticker_pct: float = 12.0
    transaction_cost_bps: float = 5.0
    take_profit_pct: float = 8.0
    stop_loss_pct: float = 5.0
    allow_short: bool = True
    initial_capital: float = 100_000.0
    min_proba_long: float = 0.55
    min_proba_short: float = 0.55


def get_all_tickers(cfg: Dict) -> List[str]:
    """Return flat list of all tickers."""
    return cfg["data"]["tickers"]["etfs"] + cfg["data"]["tickers"]["stocks"]


def get_feature_config(cfg: Dict) -> FeatureConfig:
    f = cfg["features"]
    return FeatureConfig(
        return_lags=tuple(f["return_lags"]),
        volatility_windows=tuple(f["volatility_windows"]),
        atr_window=f["atr_window"],
        volume_z_window=f["volume_z_window"],
    )


def get_portfolio_config(cfg: Dict) -> PortfolioConfig:
    p = cfg["phase_c"]
    return PortfolioConfig(
        horizon=p["horizon"],
        max_positions=p["max_positions"],
        capital_per_trade_pct=p["capital_per_trade_pct"],
        max_exposure_per_ticker_pct=p["max_exposure_per_ticker_pct"],
        transaction_cost_bps=p["transaction_cost_bps"],
        take_profit_pct=p["take_profit_pct"],
        stop_loss_pct=p["stop_loss_pct"],
        allow_short=p["allow_short"],
        initial_capital=p["initial_capital"],
        min_proba_long=p["min_proba_long"],
        min_proba_short=p["min_proba_short"],
    )

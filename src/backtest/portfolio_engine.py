"""
Portfolio backtest engine — THE SINGLE backtest engine for the entire project.

Handles:
- Multi-position portfolio (up to max_positions simultaneous)
- LONG and SHORT positions with correct SL/TP for both directions
- Transaction costs per round trip
- Daily equity curve for proper Sharpe annualization
- Trade-level logging (every entry/exit with full details)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Trade:
    """A completed trade with full details."""
    ticker: str
    direction: str          # 'long' or 'short'
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    capital_used: float
    gross_pnl: float
    cost: float
    net_pnl: float
    return_pct: float
    exit_reason: str        # 'horizon', 'stop_loss', 'take_profit'
    holding_days: int


@dataclass
class OpenPosition:
    """An open position being tracked."""
    ticker: str
    direction: str
    entry_date: pd.Timestamp
    planned_exit_date: pd.Timestamp
    entry_price: float
    shares: int
    capital_used: float
    entry_cost: float


@dataclass
class BacktestResult:
    """Complete backtest output."""
    daily_equity: pd.Series
    trades: List[Trade]
    metrics: Dict
    config_used: Dict


def _compute_position_return(entry_price: float, current_price: float, direction: str) -> float:
    """
    Compute return for a position, correctly handling direction.

    LONG:  profit when price goes UP.   return = (current - entry) / entry
    SHORT: profit when price goes DOWN. return = (entry - current) / entry
    """
    if direction == "long":
        return (current_price - entry_price) / entry_price
    else:  # short
        return (entry_price - current_price) / entry_price


def _compute_pnl(entry_price: float, exit_price: float, shares: int, direction: str) -> float:
    """Compute gross PnL for a completed trade."""
    if direction == "long":
        return (exit_price - entry_price) * shares
    else:  # short
        return (entry_price - exit_price) * shares


def run_portfolio_backtest(
    data: Dict[str, pd.DataFrame],
    predictions: Dict[str, pd.DataFrame],
    horizon: int = 10,
    max_positions: int = 15,
    capital_per_trade_pct: float = 10.0,
    max_exposure_per_ticker_pct: float = 12.0,
    transaction_cost_bps: float = 5.0,
    take_profit_pct: float = 8.0,
    stop_loss_pct: float = 5.0,
    allow_short: bool = True,
    initial_capital: float = 100_000.0,
    min_proba_long: float = 0.55,
    min_proba_short: float = 0.55,
) -> BacktestResult:
    """
    Run a portfolio-level backtest.

    Args:
        data: {ticker: OHLCV DataFrame} — must have 'close' column.
        predictions: {ticker: DataFrame with columns ['proba_long', 'proba_short', 'direction']}.
            For binary models: proba_long = P(class 1), proba_short = 0, direction = 'long'.
            For ternary models: proba_long = P(class 2), proba_short = P(class 0).
        horizon: Holding period in trading days.
        ... other params from PortfolioConfig.

    Returns:
        BacktestResult with daily equity, all trades, and computed metrics.
    """
    equity = initial_capital
    positions: List[OpenPosition] = []
    completed_trades: List[Trade] = []
    daily_equity_records = []
    cost_rate = transaction_cost_bps / 10_000  # Convert bps to fraction

    # All trading dates
    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    # Map date -> index for horizon calculation
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for date in all_dates:
        prices = {}
        for ticker, df in data.items():
            if date in df.index:
                prices[ticker] = df.loc[date, "close"]

        # ---- [1] CHECK SL/TP ON OPEN POSITIONS ----
        to_close_sltp = []
        for pos in positions:
            if pos.ticker not in prices:
                continue
            current_price = prices[pos.ticker]
            ret = _compute_position_return(pos.entry_price, current_price, pos.direction)

            if ret <= -(stop_loss_pct / 100):
                to_close_sltp.append((pos, "stop_loss", current_price))
            elif ret >= (take_profit_pct / 100):
                to_close_sltp.append((pos, "take_profit", current_price))

        for pos, reason, exit_price in to_close_sltp:
            if pos in positions:
                gross_pnl = _compute_pnl(pos.entry_price, exit_price, pos.shares, pos.direction)
                exit_cost = exit_price * pos.shares * cost_rate
                net_pnl = gross_pnl - exit_cost - pos.entry_cost
                ret_pct = _compute_position_return(pos.entry_price, exit_price, pos.direction)
                holding = (date - pos.entry_date).days

                completed_trades.append(Trade(
                    ticker=pos.ticker, direction=pos.direction,
                    entry_date=pos.entry_date, exit_date=date,
                    entry_price=pos.entry_price, exit_price=exit_price,
                    shares=pos.shares, capital_used=pos.capital_used,
                    gross_pnl=gross_pnl, cost=exit_cost + pos.entry_cost,
                    net_pnl=net_pnl, return_pct=ret_pct,
                    exit_reason=reason, holding_days=holding,
                ))
                equity += net_pnl
                positions.remove(pos)

        # ---- [2] CLOSE EXPIRED POSITIONS (HORIZON) ----
        to_close_horizon = [p for p in positions if date >= p.planned_exit_date]
        for pos in to_close_horizon:
            if pos.ticker not in prices:
                continue
            exit_price = prices[pos.ticker]
            gross_pnl = _compute_pnl(pos.entry_price, exit_price, pos.shares, pos.direction)
            exit_cost = exit_price * pos.shares * cost_rate
            net_pnl = gross_pnl - exit_cost - pos.entry_cost
            ret_pct = _compute_position_return(pos.entry_price, exit_price, pos.direction)
            holding = (date - pos.entry_date).days

            completed_trades.append(Trade(
                ticker=pos.ticker, direction=pos.direction,
                entry_date=pos.entry_date, exit_date=date,
                entry_price=pos.entry_price, exit_price=exit_price,
                shares=pos.shares, capital_used=pos.capital_used,
                gross_pnl=gross_pnl, cost=exit_cost + pos.entry_cost,
                net_pnl=net_pnl, return_pct=ret_pct,
                exit_reason="horizon", holding_days=holding,
            ))
            equity += net_pnl
            positions.remove(pos)

        # ---- [3] GENERATE NEW SIGNALS & OPEN POSITIONS ----
        current_tickers = {p.ticker for p in positions}
        slots_available = max_positions - len(positions)

        if slots_available > 0:
            # Collect today's signals
            signals = []
            for ticker in predictions:
                if ticker in current_tickers:
                    continue
                if ticker not in prices:
                    continue
                pred_df = predictions[ticker]
                if date not in pred_df.index:
                    continue

                row = pred_df.loc[date]
                proba_l = row.get("proba_long", 0.0)
                proba_s = row.get("proba_short", 0.0)

                # Determine best direction and signal strength
                if allow_short and proba_s > proba_l and proba_s >= min_proba_short:
                    signals.append((ticker, proba_s, "short"))
                elif proba_l >= min_proba_long:
                    signals.append((ticker, proba_l, "long"))

            # Sort by probability descending, take top available
            signals.sort(key=lambda x: x[1], reverse=True)
            signals = signals[:slots_available]

            # Open positions
            for ticker, proba, direction in signals:
                price = prices[ticker]
                trade_capital = equity * (capital_per_trade_pct / 100)

                # Check exposure limit
                ticker_exposure = sum(
                    p.capital_used for p in positions if p.ticker == ticker
                )
                max_ticker_cap = equity * (max_exposure_per_ticker_pct / 100)
                if ticker_exposure + trade_capital > max_ticker_cap:
                    continue

                shares = int(trade_capital / price)
                if shares <= 0:
                    continue

                actual_capital = shares * price
                entry_cost = actual_capital * cost_rate

                # Compute planned exit date
                idx = date_to_idx.get(date, 0)
                exit_idx = min(idx + horizon, len(all_dates) - 1)
                planned_exit = all_dates[exit_idx]

                positions.append(OpenPosition(
                    ticker=ticker, direction=direction,
                    entry_date=date, planned_exit_date=planned_exit,
                    entry_price=price, shares=shares,
                    capital_used=actual_capital, entry_cost=entry_cost,
                ))
                equity -= entry_cost  # Deduct entry cost immediately

        # ---- [4] MARK-TO-MARKET ----
        mtm = equity
        for pos in positions:
            if pos.ticker in prices:
                current_price = prices[pos.ticker]
                unrealized = _compute_pnl(pos.entry_price, current_price, pos.shares, pos.direction)
                mtm += unrealized

        daily_equity_records.append({"date": date, "equity": mtm})

    # Close any remaining positions at last available price
    last_date = all_dates[-1]
    for pos in list(positions):
        if pos.ticker in prices:
            exit_price = prices[pos.ticker]
            gross_pnl = _compute_pnl(pos.entry_price, exit_price, pos.shares, pos.direction)
            exit_cost = exit_price * pos.shares * cost_rate
            net_pnl = gross_pnl - exit_cost - pos.entry_cost
            ret_pct = _compute_position_return(pos.entry_price, exit_price, pos.direction)
            holding = (last_date - pos.entry_date).days

            completed_trades.append(Trade(
                ticker=pos.ticker, direction=pos.direction,
                entry_date=pos.entry_date, exit_date=last_date,
                entry_price=pos.entry_price, exit_price=exit_price,
                shares=pos.shares, capital_used=pos.capital_used,
                gross_pnl=gross_pnl, cost=exit_cost + pos.entry_cost,
                net_pnl=net_pnl, return_pct=ret_pct,
                exit_reason="end_of_period", holding_days=holding,
            ))

    # ---- BUILD DAILY EQUITY SERIES ----
    eq_df = pd.DataFrame(daily_equity_records).set_index("date")
    daily_equity = eq_df["equity"]

    # ---- COMPUTE METRICS ----
    metrics = compute_metrics(daily_equity, completed_trades, initial_capital)

    return BacktestResult(
        daily_equity=daily_equity,
        trades=completed_trades,
        metrics=metrics,
        config_used={
            "horizon": horizon, "max_positions": max_positions,
            "capital_per_trade_pct": capital_per_trade_pct,
            "transaction_cost_bps": transaction_cost_bps,
            "take_profit_pct": take_profit_pct, "stop_loss_pct": stop_loss_pct,
            "allow_short": allow_short, "initial_capital": initial_capital,
        },
    )


def compute_metrics(
    daily_equity: pd.Series,
    trades: List[Trade],
    initial_capital: float,
) -> Dict:
    """Compute all portfolio metrics from daily equity curve and trade list."""
    # Daily returns from equity curve — the ONLY correct way to annualize Sharpe
    daily_returns = daily_equity.pct_change().dropna()

    total_return = (daily_equity.iloc[-1] / initial_capital) - 1
    n_days = len(daily_returns)

    if n_days > 0 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    neg_returns = daily_returns[daily_returns < 0]
    if len(neg_returns) > 0 and neg_returns.std() > 0:
        sortino = daily_returns.mean() / neg_returns.std() * np.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown from equity curve
    cummax = daily_equity.cummax()
    drawdown = (daily_equity - cummax) / cummax
    max_dd = drawdown.min()

    # Trade-level stats
    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        win_rate = len(wins) / n_trades
        total_pnl = sum(t.net_pnl for t in trades)
        total_cost = sum(t.cost for t in trades)
        avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.net_pnl for t in losses]) if losses else 0
        gross_profit = sum(t.net_pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.net_pnl for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        long_trades = [t for t in trades if t.direction == "long"]
        short_trades = [t for t in trades if t.direction == "short"]
        trade_returns = [t.return_pct for t in trades]
    else:
        win_rate = 0
        total_pnl = 0
        total_cost = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        long_trades = []
        short_trades = []
        trade_returns = []

    return {
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(total_return * 100, 2),  # 1 year OOS
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_trades": n_trades,
        "n_long": len(long_trades),
        "n_short": len(short_trades),
        "win_rate": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 3),
        "total_pnl": round(total_pnl, 2),
        "total_costs": round(total_cost, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "final_equity": round(daily_equity.iloc[-1], 2),
        "n_trading_days": n_days,
        "daily_return_mean": round(daily_returns.mean() * 100, 4) if len(daily_returns) > 0 else 0,
        "daily_return_std": round(daily_returns.std() * 100, 4) if len(daily_returns) > 0 else 0,
        "trade_returns": trade_returns,  # For bootstrap
    }


def run_naive_backtest(
    predictions: np.ndarray,
    prices_close: np.ndarray,
    prices_open: np.ndarray,
    horizon: int,
    cost_bps: float = 5.0,
) -> Dict:
    """
    Simple long-only backtest for Phase A scoring.
    No portfolio constraints, no SL/TP — just signal quality assessment.

    Args:
        predictions: Binary array (1=trade, 0=no trade) or probability array.
        prices_close: Close prices array.
        prices_open: Open prices array (for next-day entry).
        horizon: Holding period.
        cost_bps: Round-trip cost in basis points.

    Returns:
        Dict with sharpe, trades list, etc.
    """
    cost_rate = cost_bps / 10_000
    trades = []
    n = len(predictions)

    i = 0
    while i < n - horizon - 1:
        if predictions[i] >= 0.5:
            entry_price = prices_open[i + 1] if i + 1 < n else prices_close[i]
            exit_price = prices_close[min(i + horizon, n - 1)]
            ret = (exit_price / entry_price) - 1.0 - cost_rate
            trades.append(ret)
            i += horizon  # No overlapping trades
        else:
            i += 1

    trades = np.array(trades)
    n_trades = len(trades)

    if n_trades < 2:
        return {"sharpe": 0.0, "n_trades": n_trades, "trades": trades.tolist(),
                "total_return": 0.0, "win_rate": 0.0}

    # Correct Sharpe annualization for H-day holding period
    mean_ret = trades.mean()
    std_ret = trades.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252 / horizon) if std_ret > 0 else 0.0

    return {
        "sharpe": round(sharpe, 4),
        "n_trades": n_trades,
        "trades": trades.tolist(),
        "total_return": round(trades.sum() * 100, 2),
        "win_rate": round((trades > 0).mean() * 100, 2),
        "mean_return": round(mean_ret * 100, 4),
    }

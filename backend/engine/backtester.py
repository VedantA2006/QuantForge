# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Vectorized Backtest Engine                    ║
# ║  Hybrid: vectorized signals + iterative position tracking   ║
# ╚══════════════════════════════════════════════════════════════╝

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field, asdict

from backend.strategy.tree import Strategy
from backend.config import (
    INITIAL_BALANCE, DEFAULT_FEE, DEFAULT_SLIPPAGE, COOLDOWN_BARS,
)

log = logging.getLogger("quantforge.engine.backtester")


@dataclass
class BacktestResult:
    strategy_id: str = ""
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    expectancy: float = 0.0
    monthly_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    avg_holding_bars: float = 0.0
    wins: int = 0
    losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_monthly_return: float = 0.0
    yearly_returns: List[float] = field(default_factory=list)
    avg_yearly_return: float = 0.0
    trade_history: List[Dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @property
    def is_valid(self) -> bool:
        return self.total_trades >= 10 and self.max_drawdown_pct < 100


def run_backtest(
    strategy: Strategy,
    df: pd.DataFrame,
    initial_balance: float = INITIAL_BALANCE,
    fee: float = DEFAULT_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
) -> BacktestResult:
    """
    Run a full backtest for a strategy on OHLCV data with indicators.

    Phase 1: Vectorized signal generation (fast)
    Phase 2: Iterative position management (realistic)
    """
    result = BacktestResult(strategy_id=strategy.strategy_id)

    if df.empty or len(df) < 50:
        return result

    # ── Phase 1: Vectorized signal generation ─────────────────────────
    try:
        signals = strategy.evaluate_signals(df)
    except Exception as e:
        log.debug(f"Signal eval failed for {strategy.name}: {e}")
        return result

    # ── Phase 2: Iterative position management ────────────────────────
    balance = initial_balance
    position = None
    trades = []
    equity = [initial_balance]
    cooldown = strategy.risk_params.cooldown
    last_entry_bar = -(cooldown + 1)

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["atr_14"].values if "atr_14" in df.columns else np.full(len(df), 0)
    sig_vals = signals.values

    for i in range(1, len(df)):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        # ── Check exit ────────────────────────────────────────────────
        if position is not None:
            side = position["side"]
            sl = position["sl"]
            tp = position["tp"]
            risk = position["risk"]

            exit_reason = None
            exit_price = None

            if side == 1:  # LONG
                if o <= sl:
                    exit_reason, exit_price = "SL_GAP", o
                elif o >= tp:
                    exit_reason, exit_price = "TP_GAP", o
                elif l <= sl:
                    exit_reason, exit_price = "SL", sl
                elif h >= tp:
                    exit_reason, exit_price = "TP", tp
            else:  # SHORT
                if o >= sl:
                    exit_reason, exit_price = "SL_GAP", o
                elif o <= tp:
                    exit_reason, exit_price = "TP_GAP", o
                elif h >= sl:
                    exit_reason, exit_price = "SL", sl
                elif l <= tp:
                    exit_reason, exit_price = "TP", tp

            if exit_reason:
                is_win = "TP" in exit_reason
                rr = strategy.risk_params.rr_ratio
                sl_mult = strategy.risk_params.sl_atr_mult
                pnl_mult = (rr * sl_mult) if is_win else -sl_mult
                pnl = risk * pnl_mult
                pnl -= risk * fee * 2  # entry + exit fees
                balance += pnl
                trades.append({
                    "side": "LONG" if side == 1 else "SHORT",
                    "entry_bar": position["entry_bar"],
                    "exit_bar": i,
                    "entry_price": position["entry"],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "pnl": pnl, 
                    "is_win": is_win,
                    "holding_bars": i - position["entry_bar"],
                })
                equity.append(balance)
                position = None

        # ── Check entry ───────────────────────────────────────────────
        if position is None and i - last_entry_bar >= cooldown:
            sig = sig_vals[i]
            if sig != 0 and not np.isnan(atrs[i]) and atrs[i] > 0:
                side = 1 if sig > 0 else -1
                entry_price = c * (1 + slippage * side)  # slippage
                atr_val = atrs[i]
                sl_dist = atr_val * strategy.risk_params.sl_atr_mult
                tp_dist = sl_dist * strategy.risk_params.rr_ratio

                if side == 1:
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + tp_dist
                else:
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - tp_dist

                risk_usd = balance * strategy.risk_params.risk_pct
                position = {
                    "side": side,
                    "entry": entry_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "risk": risk_usd,
                    "entry_bar": i,
                }
                last_entry_bar = i

    # ── Close any open position at last bar ───────────────────────────
    if position is not None:
        pnl = 0
        balance += pnl
        equity.append(balance)

    # ── Compute metrics ───────────────────────────────────────────────
    result = _compute_metrics(
        trades, equity, initial_balance, strategy.strategy_id, df
    )
    return result


def _compute_metrics(
    trades: List[Dict],
    equity: List[float],
    initial_balance: float,
    strategy_id: str,
    df: pd.DataFrame,
) -> BacktestResult:
    """Compute all performance metrics from trade list."""
    result = BacktestResult(strategy_id=strategy_id)
    n = len(trades)
    result.total_trades = n

    if n == 0:
        result.equity_curve = equity
        return result
        
    result.trade_history = trades

    pnls = np.array([t["pnl"] for t in trades])
    wins_mask = np.array([t["is_win"] for t in trades])
    holding = np.array([t["holding_bars"] for t in trades])

    result.wins = int(wins_mask.sum())
    result.losses = n - result.wins
    result.win_rate = result.wins / n * 100

    # Profit factor & Win/Loss Averages
    wins_pnl = pnls[pnls > 0]
    loss_pnl = pnls[pnls < 0]
    gross_profit = wins_pnl.sum() if len(wins_pnl) > 0 else 0
    gross_loss = abs(loss_pnl.sum()) if len(loss_pnl) > 0 else 0
    
    result.profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else
        (10.0 if gross_profit > 0 else 0.0)
    )
    result.avg_win = float(wins_pnl.mean()) if len(wins_pnl) > 0 else 0.0
    result.avg_loss = float(abs(loss_pnl.mean())) if len(loss_pnl) > 0 else 0.0

    # Returns
    result.total_return_pct = (equity[-1] - initial_balance) / initial_balance * 100
    result.equity_curve = equity

    # Drawdown
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1) * 100
    result.max_drawdown_pct = abs(dd.min())

    # Sharpe ratio (per-trade returns)
    ret_series = np.diff(eq) / eq[:-1]
    if len(ret_series) > 1 and np.std(ret_series) > 0:
        trades_per_year = max(n, 1)
        result.sharpe_ratio = (
            np.mean(ret_series) / np.std(ret_series) * np.sqrt(trades_per_year)
        )
    else:
        result.sharpe_ratio = 0.0

    # Sortino ratio
    downside = ret_series[ret_series < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        result.sortino_ratio = (
            np.mean(ret_series) / np.std(downside) * np.sqrt(max(n, 1))
        )
    else:
        result.sortino_ratio = result.sharpe_ratio

    # Expectancy
    result.expectancy = float(np.mean(pnls))

    # Avg holding bars
    result.avg_holding_bars = float(np.mean(holding))

    # Periodic returns
    if "datetime" in df.columns:
        try:
            result.monthly_returns, result.yearly_returns = _calc_periodic_returns(trades, initial_balance, df)
        except Exception:
            result.monthly_returns = []
            result.yearly_returns = []

    result.avg_monthly_return = round(float(np.mean(result.monthly_returns)), 2) if result.monthly_returns else 0.0
    result.avg_yearly_return = round(float(np.mean(result.yearly_returns)), 2) if result.yearly_returns else 0.0

    return result


def _calc_periodic_returns(trades: List[Dict], initial_balance: float, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    """Calculate true calendar monthly and yearly returns from trades."""
    if "datetime" not in df.columns:
        return [], []
        
    datetimes = pd.to_datetime(df["datetime"])
    daily_dates = datetimes.dt.normalize().unique()
    daily_bal = pd.Series(index=daily_dates, dtype=float)
    
    current_bal = initial_balance
    if len(daily_bal) > 0:
        daily_bal.iloc[0] = current_bal
        
    for t in trades:
        exit_dt = datetimes.iloc[t["exit_bar"]].normalize()
        current_bal += t["pnl"]
        daily_bal.loc[exit_dt] = current_bal
        
    daily_bal = daily_bal.ffill()
    
    resampler_code_m = "ME" if pd.__version__ >= "2.2.0" else "M"
    resampler_code_y = "YE" if pd.__version__ >= "2.2.0" else "Y"
    
    monthly_bal = daily_bal.resample(resampler_code_m).last().dropna()
    yearly_bal = daily_bal.resample(resampler_code_y).last().dropna()
    
    monthly_returns = []
    prev_bal = initial_balance
    for end_bal in monthly_bal:
        ret = (end_bal / prev_bal - 1) * 100
        monthly_returns.append(round(ret, 2))
        prev_bal = end_bal
        
    yearly_returns = []
    prev_bal = initial_balance
    for end_bal in yearly_bal:
        ret = (end_bal / prev_bal - 1) * 100
        yearly_returns.append(round(ret, 2))
        prev_bal = end_bal
        
    return monthly_returns, yearly_returns

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

    # Monthly returns (approximate)
    if "datetime" in df.columns:
        try:
            result.monthly_returns = _calc_monthly_returns(equity, df)
        except Exception:
            result.monthly_returns = []

    result.avg_monthly_return = round(float(np.mean(result.monthly_returns)), 2) if result.monthly_returns else 0.0

    return result


def _calc_monthly_returns(equity: List[float], df: pd.DataFrame) -> List[float]:
    """Approximate monthly returns from equity curve."""
    if len(equity) < 2:
        return []
    # Simple: divide equity curve into ~30-bar chunks
    chunk = max(1, len(equity) // 12)
    monthly = []
    for i in range(0, len(equity) - 1, chunk):
        end = min(i + chunk, len(equity) - 1)
        if equity[i] > 0:
            ret = (equity[end] - equity[i]) / equity[i] * 100
            monthly.append(round(ret, 2))
    return monthly

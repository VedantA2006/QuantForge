# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — FastAPI Route Handlers                        ║
# ╚══════════════════════════════════════════════════════════════╝

from fastapi import APIRouter, Query
from typing import Optional
from backend.config import SYMBOL, INTERVALS

router = APIRouter(prefix="/api")

# These will be set by main.py after DB and state init
_db = None
_state = None


def init_routes(db, state):
    global _db, _state
    _db = db
    _state = state


@router.get("/status")
async def get_status():
    """System status: running, cycle count, uptime."""
    return {
        "status": _state.get("status", "idle"),
        "cycle": _state.get("cycle", 0),
        "total_evaluated": _state.get("total_evaluated", 0),
        "total_passed": _state.get("total_passed", 0),
        "uptime_seconds": _state.get("uptime_seconds", 0),
        "last_cycle_duration": _state.get("last_cycle_duration", 0),
        "data_loaded": _state.get("data_loaded", False),
        "symbol": SYMBOL,
        "intervals": INTERVALS,
    }


@router.get("/best")
async def get_best(limit: int = Query(10, ge=1, le=50)):
    """Top N strategies with full metrics."""
    strategies = _db.get_best(limit)
    # Sanitize for JSON (remove non-serializable fields and heavy arrays)
    for s in strategies:
        ec = s.get("metrics", {}).get("equity_curve", [])
        if len(ec) > 200:
            # Downsample equity curve for frontend
            step = max(1, len(ec) // 200)
            s["metrics"]["equity_curve"] = ec[::step]
        # Remove large trade history from the leaderboard payload
        if "trade_history" in s.get("metrics", {}):
            s["metrics"].pop("trade_history", None)
    return {"strategies": strategies, "count": len(strategies)}


@router.get("/metrics")
async def get_metrics():
    """Aggregate system metrics."""
    summary = _db.get_metrics_summary()
    summary.update({
        "cycle": _state.get("cycle", 0),
        "status": _state.get("status", "idle"),
        "strategies_per_cycle": _state.get("batch_size", 20),
        "ga_cycles": _state.get("ga_cycles", 0),
    })
    return summary


@router.get("/logs")
async def get_logs(limit: int = Query(50, ge=1, le=200)):
    """Recent log entries."""
    logs = _db.get_logs(limit)
    # Convert datetime to ISO string
    for entry in logs:
        if "timestamp" in entry:
            entry["timestamp"] = str(entry["timestamp"])
    return {"logs": logs, "count": len(logs)}


@router.get("/strategy/{strategy_id}")
async def get_strategy(strategy_id: str):
    """Single strategy detail with equity curve."""
    strategies = _db.get_best(50)
    for s in strategies:
        if s.get("strategy_id") == strategy_id:
            return s
    return {"error": "Strategy not found"}


@router.get("/strategy/{strategy_id}/code")
async def get_strategy_code(strategy_id: str):
    """Generate standalone Python backtest code for the strategy."""
    strategies = _db.get_best(50)
    strategy_data = next((s for s in strategies if s.get("strategy_id") == strategy_id), None)
    if not strategy_data:
        return {"error": "Strategy not found"}

    from backend.strategy.tree import strategy_from_dict
    from fastapi.responses import PlainTextResponse
    import os

    strat = strategy_from_dict(strategy_data["strategy"])
    buy_code = strat.buy_rule.to_code() if strat.buy_rule else "False"
    sell_code = strat.sell_rule.to_code() if strat.sell_rule else "False"

    # Read indicators.py to inline its logic
    indicators_path = os.path.join(os.path.dirname(__file__), "..", "data", "indicators.py")
    with open(indicators_path, "r", encoding="utf-8") as f:
        indicators_code = f.read()
    
    # Strip the header and imports from indicators.py
    ind_lines = indicators_code.split("\n")
    ind_start = 0
    for i, line in enumerate(ind_lines):
        if line.startswith("def ema"):
            ind_start = i
            break
    indicators_pure = "\n".join(ind_lines[ind_start:])

    risk_header = (
        f"  RISK: SL={strat.risk_params.sl_atr_mult}\u00d7ATR, "
        f"TP={round(strat.risk_params.sl_atr_mult * strat.risk_params.rr_ratio, 3)}\u00d7ATR "
        f"({strat.risk_params.sl_atr_mult}\u00d7{strat.risk_params.rr_ratio}), "
        f"Win=+{round(strat.risk_params.sl_atr_mult * strat.risk_params.rr_ratio, 3)}R  "
        f"Loss=-{strat.risk_params.sl_atr_mult}R  "
        f"risk={round(strat.risk_params.risk_pct * 100, 2)}% per trade"
    )

    code_template = f'''"""
======================================================================
  QUANTFORGE — LIVE TRADING ENGINE & BACKTESTER
  Strategy: {strat.name} ({strategy_id})
======================================================================
{risk_header}
"""

import sys
import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ======================================================================
#  USER SETTINGS
# ======================================================================
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DURATION = "5y"

INITIAL_BALANCE = 1000.0
FEE = 0.0005
SLIPPAGE = 0.0005

# Risk Parameters
SL_ATR_MULT = {strat.risk_params.sl_atr_mult}
RR_RATIO = {strat.risk_params.rr_ratio}
RISK_PCT = {strat.risk_params.risk_pct}
COOLDOWN = {strat.risk_params.cooldown}

# ======================================================================
#  INDICATORS LIBRARY (Inlined from QuantForge)
# ======================================================================
{indicators_pure}

# ======================================================================
#  STRATEGY SIGNAL ENGINE
# ======================================================================
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate expression trees to generate signals."""
    df = df.copy()
    signals = pd.Series(0.0, index=df.index)
    
    buy_mask = {buy_code}
    signals[buy_mask] = 1.0
    
    sell_mask = {sell_code}
    signals[sell_mask] = -1.0
    
    df["signal"] = signals
    return df

# ======================================================================
#  BYBIT DATA INGESTION
# ======================================================================
def fetch_bybit_data(symbol: str, interval: str, duration: str) -> pd.DataFrame:
    print(f"Fetching {{duration}} of data for {{symbol}} {{interval}}...")
    _INTERVAL_MAP = {{"1h": "60", "4h": "240", "1d": "D"}}
    bv_int = _INTERVAL_MAP.get(interval, "60")
    
    days = int(duration[:-1]) * 365 if duration.endswith('y') else 30
    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    end_ms = int(datetime.utcnow().timestamp() * 1000)
    
    all_rows = []
    while True:
        resp = requests.get("https://api.bybit.com/v5/market/kline", params={{
            "category": "linear", "symbol": symbol, "interval": bv_int,
            "start": start_ms, "end": end_ms, "limit": 1000
        }}).json()
        
        rows = resp.get("result", {{}}).get("list", [])
        if not rows: break
        all_rows.extend(reversed(rows))
        
        if len(rows) < 1000: break
        end_ms = int(rows[-1][0]) - 1
        time.sleep(0.1)

    df = pd.DataFrame(all_rows, columns=["time", "open", "high", "low", "close", "volume", "turnover"]).astype(float)
    df["datetime"] = pd.to_datetime(df["time"], unit="ms")
    return df.sort_values("time").reset_index(drop=True)

# ======================================================================
#  BACKTEST LOOP
# ======================================================================
def run_backtest():
    df = fetch_bybit_data(SYMBOL, INTERVAL, DURATION)
    df = compute_all_indicators(df)
    df = generate_signals(df)

    balance = INITIAL_BALANCE
    position = None
    trades = []
    last_entry = -(COOLDOWN + 1)

    opens, highs, lows, closes = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    atrs, signals = df["atr_14"].values, df["signal"].values
    datetimes = df["datetime"].values

    # Track end-of-day balance for monthly/yearly resampling
    balance_by_date = {{}}

    print(f"Running backtest on {{len(df)}} candles...")

    for i in range(1, len(df)):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        bar_dt = pd.Timestamp(datetimes[i])

        if position is not None:
            side, sl, tp, risk = position["side"], position["sl"], position["tp"], position["risk"]
            exit_reason = None

            if side == 1:
                if o <= sl: exit_reason = "SL_GAP"
                elif o >= tp: exit_reason = "TP_GAP"
                elif l <= sl: exit_reason = "SL"
                elif h >= tp: exit_reason = "TP"
            else:
                if o >= sl: exit_reason = "SL_GAP"
                elif o <= tp: exit_reason = "TP_GAP"
                elif h >= sl: exit_reason = "SL"
                elif l <= tp: exit_reason = "TP"

            if exit_reason:
                is_win = "TP" in exit_reason
                pnl = risk * ((RR_RATIO * SL_ATR_MULT) if is_win else -SL_ATR_MULT) - (risk * FEE * 2)
                balance += pnl

                if "GAP" in exit_reason:
                    exit_price = o
                elif is_win:
                    exit_price = position["tp"]
                else:
                    exit_price = position["sl"]

                trades.append({{
                    "entry_datetime": position["entry_dt"].strftime("%Y-%m-%d %H:%M"),
                    "exit_datetime": bar_dt.strftime("%Y-%m-%d %H:%M"),
                    "side": "LONG" if position["side"] == 1 else "SHORT",
                    "entry_price": round(position["entry"], 2),
                    "exit_price": round(exit_price, 2),
                    "sl": round(position["sl"], 2),
                    "tp": round(position["tp"], 2),
                    "pnl": round(pnl, 2),
                    "result": exit_reason,
                    "is_win": is_win,
                    "balance": round(balance, 2),
                }})
                position = None

        if position is None and i - last_entry >= COOLDOWN:
            sig = signals[i]
            if sig != 0 and atrs[i] > 0:
                side = 1 if sig > 0 else -1
                entry = c * (1 + SLIPPAGE * side)
                sl_dist = atrs[i] * SL_ATR_MULT
                tp_dist = atrs[i] * SL_ATR_MULT * RR_RATIO

                position = {{
                    "side": side,
                    "entry": entry,
                    "entry_dt": bar_dt,
                    "sl": entry - sl_dist if side == 1 else entry + sl_dist,
                    "tp": entry + tp_dist if side == 1 else entry - tp_dist,
                    "risk": balance * RISK_PCT,
                }}
                last_entry = i

        day_key = bar_dt.strftime("%Y-%m-%d")
        balance_by_date[day_key] = balance

    # ── Summary Stats ─────────────────────────────────────────────────────
    wins = sum(1 for t in trades if t["is_win"])
    win_rate = wins / len(trades) * 100 if trades else 0
    profit_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    print("\\n" + "="*50)
    print(f"BACKTEST RESULTS: {{SYMBOL}} {{INTERVAL}}")
    print("="*50)
    print(f"Total Trades : {{len(trades)}}")
    print(f"Win Rate     : {{win_rate:.1f}}%")
    print(f"Net Profit   : {{profit_pct:.1f}}%")
    print(f"Final Balance: ${{balance:.2f}}")
    print("="*50)

    # ── Monthly Returns ───────────────────────────────────────────────────
    date_idx = pd.to_datetime(list(balance_by_date.keys()))
    bal_series = pd.Series(list(balance_by_date.values()), index=date_idx).sort_index()
    monthly_bal = bal_series.resample("M").last().dropna()

    monthly_returns = {{}}
    prev_bal = INITIAL_BALANCE
    for dt, end_bal in monthly_bal.items():
        ret = (end_bal / prev_bal - 1) * 100
        monthly_returns[dt.strftime("%Y-%m")] = ret
        prev_bal = end_bal

    print("\\n[Monthly Returns (%)]")
    print(f"{{' Month':<10}} {{'Return (%)':>10}}")
    for month, ret in monthly_returns.items():
        print(f"{{month:<10}} {{ret:>10.2f}}")

    all_monthly = list(monthly_returns.values())
    avg_monthly = sum(all_monthly) / len(all_monthly) if all_monthly else 0
    print(f"\\nSum of {{len(all_monthly)}} monthly returns : {{sum(all_monthly):.2f}}%")
    print(f"Number of months                  : {{len(all_monthly)}}")
    print(f"Average monthly return            : {{avg_monthly:.2f}}%")

    # ── Yearly Returns ────────────────────────────────────────────────────
    yearly_bal = bal_series.resample("Y").last().dropna()
    yearly_returns = {{}}
    prev_bal = INITIAL_BALANCE
    for dt, end_bal in yearly_bal.items():
        ret = (end_bal / prev_bal - 1) * 100
        yearly_returns[str(dt.year)] = ret
        prev_bal = end_bal

    print("\\n[Yearly Returns (%)]")
    print(f"{{'Year':<6}} {{'Return (%)':>10}}")
    for year, ret in yearly_returns.items():
        print(f"{{year:<6}} {{ret:>10.2f}}")

    # ── CSV Export ────────────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"trades_{strategy_id}.csv")
    trade_df = pd.DataFrame(trades)
    export_cols = [c for c in trade_df.columns if c != "is_win"]
    trade_df[export_cols].to_csv(csv_path, index=False)
    print(f"\\n[Trades exported to]: {{csv_path}}")
    print(f"   Total trades saved: {{len(trades)}}")


if __name__ == "__main__":
    run_backtest()
'''

    return PlainTextResponse(content=code_template, media_type="text/x-python", headers={
        "Content-Disposition": f"attachment; filename=run_{strategy_id}.py"
    })


@router.post("/control/start")
async def start_loop():
    """Signal the discovery loop to start."""
    _state["should_run"] = True
    return {"message": "Discovery loop starting"}


@router.post("/control/stop")
async def stop_loop():
    """Signal the discovery loop to stop."""
    _state["should_run"] = False
    return {"message": "Discovery loop stopping"}

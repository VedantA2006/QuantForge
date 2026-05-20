# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — FastAPI Route Handlers                        ║
# ╚══════════════════════════════════════════════════════════════╝

from fastapi import APIRouter, Query
from typing import Optional, Any
from backend.config import SYMBOL, INTERVALS

router = APIRouter(prefix="/api")

# These will be set by main.py after DB and state init
_db: Any = None
_state: Any = None


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


@router.get("/hof")
async def get_hof(limit: int = Query(50, ge=1, le=100)):
    """Hall of Fame strategies."""
    if not hasattr(_db, 'db') or _db.db is None:
        return {"strategies": [], "count": 0}
        
    strategies = list(_db.db.hof_strategies.find({}, {"_id": 0}).sort("rank_score", -1).limit(limit))
    for s in strategies:
        ec = s.get("metrics", {}).get("equity_curve", [])
        if len(ec) > 200:
            step = max(1, len(ec) // 200)
            s["metrics"]["equity_curve"] = ec[::step]
        if "trade_history" in s.get("metrics", {}):
            s["metrics"].pop("trade_history", None)
            
    return {"strategies": strategies, "count": len(strategies)}


@router.get("/surrogate/stats")
async def get_surrogate_stats():
    """Surrogate model stats."""
    import sys
    # Find surrogate model instance from main module state
    surrogate_stats = {"is_trained": False, "predictions_made": 0, "strategies_filtered": 0, "training_samples": 0, "mae_last_100": 0.0}
    try:
        if "backend.main" in sys.modules:
            main_mod = sys.modules["backend.main"]
            if hasattr(main_mod, "surrogate_model"):
                surrogate = main_mod.surrogate_model
                surrogate_stats = surrogate.stats
                surrogate_stats["is_trained"] = surrogate.is_trained
                if surrogate.stats["predictions_made"] > 0:
                    surrogate_stats["filter_rate"] = round(surrogate.stats["strategies_filtered"] / surrogate.stats["predictions_made"] * 100, 1)
                else:
                    surrogate_stats["filter_rate"] = 0.0
    except Exception:
        pass
    return surrogate_stats


@router.get("/rl/stats")
async def get_rl_stats():
    """RL agent stats."""
    import sys
    rl_stats: dict[str, Any] = {"total_updates": 0, "avg_reward_50": 0.0, "weights_buy": {}, "weights_sell": {}}
    try:
        if "backend.main" in sys.modules:
            main_mod = sys.modules["backend.main"]
            if hasattr(main_mod, "rl_builder"):
                rl = main_mod.rl_builder
                rl_stats["total_updates"] = rl.stats["total_updates"]
                rl_stats["avg_reward_50"] = round(rl.stats["avg_reward_50"], 4)
                
                # Get current favored categories (top 5) for buy and sell based on state * W
                state = rl._get_state_vector()
                buy_logits = state @ rl.W_buy
                sell_logits = state @ rl.W_sell
                
                import numpy as np
                top_buy_idx = np.argsort(buy_logits)[-5:][::-1]
                top_sell_idx = np.argsort(sell_logits)[-5:][::-1]
                
                rl_stats["favored_buy_categories"] = [rl.categories[i] for i in top_buy_idx]
                rl_stats["favored_sell_categories"] = [rl.categories[i] for i in top_sell_idx]
    except Exception:
        pass
    return rl_stats


@router.get("/category/stats")
async def get_category_stats():
    """Category weights and stats."""
    if not hasattr(_db, '_fallback') or _db._fallback:
        return {"categories": []}
        
    stats = list(_db.db.category_stats.find({}, {"_id": 0}))
    from backend.strategy.generator import CATEGORY_WEIGHTS
    
    result = []
    for s in stats:
        cat = s.get("category", "")
        if not cat: continue
        
        s["current_weight"] = round(CATEGORY_WEIGHTS.get(cat, 0.0), 4)
        s["avg_score"] = round(s.get("avg_score", 0.0), 4)
        result.append(s)
        
    result.sort(key=lambda x: x.get("current_weight", 0), reverse=True)
    return {"categories": result}


@router.get("/strategy/{strategy_id}")
async def get_strategy(strategy_id: str):
    """Single strategy detail with equity curve."""
    # Search best_strategies first
    strategies = _db.get_best(50)
    for s in strategies:
        if s.get("strategy_id") == strategy_id:
            return s
    # Fallback: search HoF and fully_passed collections
    if hasattr(_db, 'db') and _db.db is not None:
        for collection_name in ("hof_strategies", "fully_passed_strategies"):
            col = _db.db[collection_name]
            doc = col.find_one({"strategy_id": strategy_id}, {"_id": 0})
            if doc:
                return doc
    return {"error": "Strategy not found"}


@router.get("/strategy/{strategy_id}/code")
async def get_strategy_code(strategy_id: str):
    """Generate standalone Python backtest code for the strategy."""
    # Search best_strategies first
    strategies = _db.get_best(50)
    strategy_data = next((s for s in strategies if s.get("strategy_id") == strategy_id), None)
    # Fallback: search HoF and fully_passed collections
    if not strategy_data and hasattr(_db, 'db') and _db.db is not None:
        for collection_name in ("hof_strategies", "fully_passed_strategies"):
            col = _db.db[collection_name]
            doc = col.find_one({"strategy_id": strategy_id}, {"_id": 0})
            if doc:
                strategy_data = doc
                break
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
SLIPPAGE = 0.0001
SPREAD = 0.0002

# Risk Parameters
SL_ATR_MULT = {strat.risk_params.sl_atr_mult}
RR_RATIO = {strat.risk_params.rr_ratio}
RISK_PCT = {strat.risk_params.risk_pct}
COOLDOWN = {strat.risk_params.cooldown}
TRAIL_MULT = {strat.risk_params.trail_mult}
TP1_RATIO = {strat.risk_params.tp1_ratio}

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
    equity = [INITIAL_BALANCE]
    last_entry = -(COOLDOWN + 1)

    opens, highs, lows, closes = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    atr_col = "tf_1h_atr_14" if "tf_1h_atr_14" in df.columns else "atr_14"
    atrs = df[atr_col].values if atr_col in df.columns else np.full(len(df), 0.0)
    signals = df["signal"].values
    datetimes = df["datetime"].values

    # Regime column for per-regime stats
    regime_col = "tf_1h_regime" if "tf_1h_regime" in df.columns else "regime"
    has_regime = regime_col in df.columns
    regimes = df[regime_col].values if has_regime else np.zeros(len(df))

    print(f"Running backtest on {{len(df)}} candles...")

    for i in range(1, len(df)):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        bar_dt = pd.Timestamp(datetimes[i])

        if position is not None:
            side = position["side"]
            sl = position["sl"]
            tp = position["tp"]
            qty = position["qty"]
            entry_price = position["entry"]

            # ── Trailing stop update ──────────────────────────────────
            if TRAIL_MULT > 0 and not np.isnan(atrs[i]) and atrs[i] > 0:
                atr_now = atrs[i]
                if side == 1:
                    new_sl = c - atr_now * TRAIL_MULT
                    if new_sl > sl:
                        sl = new_sl
                        position["sl"] = sl
                else:
                    new_sl = c + atr_now * TRAIL_MULT
                    if new_sl < sl:
                        sl = new_sl
                        position["sl"] = sl

            # ── Partial TP1 check ─────────────────────────────────────
            if TP1_RATIO > 0 and not position["tp1_hit"]:
                if side == 1:
                    tp1_price = entry_price + (tp - entry_price) * TP1_RATIO
                    if h >= tp1_price:
                        partial_qty = qty / 2
                        position["qty"] -= partial_qty
                        qty = position["qty"]
                        actual_tp1 = tp1_price * (1 - (SLIPPAGE + SPREAD / 2.0))
                        partial_pnl = (actual_tp1 - entry_price) * partial_qty
                        partial_pnl -= partial_qty * actual_tp1 * FEE
                        position["partial_pnl"] += partial_pnl
                        position["tp1_hit"] = True
                else:
                    tp1_price = entry_price - (entry_price - tp) * TP1_RATIO
                    if l <= tp1_price:
                        partial_qty = qty / 2
                        position["qty"] -= partial_qty
                        qty = position["qty"]
                        actual_tp1 = tp1_price * (1 + (SLIPPAGE + SPREAD / 2.0))
                        partial_pnl = (entry_price - actual_tp1) * partial_qty
                        partial_pnl -= partial_qty * actual_tp1 * FEE
                        position["partial_pnl"] += partial_pnl
                        position["tp1_hit"] = True

            # ── 3-layer exit resolver ─────────────────────────────────
            exit_reason = None
            exit_price = None

            if side == 1:  # LONG
                if o <= sl:
                    exit_price, exit_reason = o, "GAP_SL"
                elif o >= tp:
                    exit_price, exit_reason = o, "GAP_TP"
                else:
                    hit_sl = l <= sl
                    hit_tp = h >= tp
                    if hit_sl and hit_tp:
                        exit_price, exit_reason = sl, "SL"
                    elif hit_sl:
                        exit_price, exit_reason = sl, "SL"
                    elif hit_tp:
                        exit_price, exit_reason = tp, "TP"
            else:  # SHORT
                if o >= sl:
                    exit_price, exit_reason = o, "GAP_SL"
                elif o <= tp:
                    exit_price, exit_reason = o, "GAP_TP"
                else:
                    hit_sl = h >= sl
                    hit_tp = l <= tp
                    if hit_sl and hit_tp:
                        exit_price, exit_reason = sl, "SL"
                    elif hit_sl:
                        exit_price, exit_reason = sl, "SL"
                    elif hit_tp:
                        exit_price, exit_reason = tp, "TP"

            # Force close on last candle if no exit has hit
            if exit_reason is None and i == len(df) - 1:
                exit_price = c
                exit_reason = "FORCE_CLOSE"

            if exit_reason:
                exit_price_with_slippage = exit_price * (1 - (SLIPPAGE + SPREAD / 2.0) * side)
                exit_fee = exit_price_with_slippage * qty * FEE

                if side == 1:
                    remaining_pnl = (exit_price_with_slippage - entry_price) * qty - exit_fee
                else:
                    remaining_pnl = (entry_price - exit_price_with_slippage) * qty - exit_fee

                pnl = remaining_pnl + position["partial_pnl"] - position["entry_fee"]
                balance += pnl
                entry_regime = position.get("entry_regime", 0)
                is_win = pnl > 0

                trades.append({{
                    "entry_datetime": position["entry_dt"].strftime("%Y-%m-%d %H:%M"),
                    "exit_datetime": bar_dt.strftime("%Y-%m-%d %H:%M"),
                    "side": "LONG" if position["side"] == 1 else "SHORT",
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price_with_slippage, 2),
                    "sl": round(position["sl"], 2),
                    "tp": round(position["tp"], 2),
                    "pnl": round(pnl, 2),
                    "result": exit_reason,
                    "is_win": is_win,
                    "balance": round(balance, 2),
                    "holding_bars": i - position["entry_bar"],
                    "regime": entry_regime,
                }})
                position = None

        if position is None and i - last_entry >= COOLDOWN:
            sig = signals[i-1]  # Signal from previous bar
            if sig != 0 and not np.isnan(atrs[i]) and atrs[i] > 0:
                side = 1 if sig > 0 else -1
                entry_price = o * (1 + (SLIPPAGE + SPREAD / 2.0) * side)
                atr_val = atrs[i]
                sl_dist = atr_val * SL_ATR_MULT
                tp_dist = sl_dist * RR_RATIO

                if side == 1:
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + tp_dist
                else:
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - tp_dist

                risk_usd = balance * RISK_PCT
                qty = risk_usd / sl_dist
                entry_fee = entry_price * qty * FEE

                position = {{
                    "side": side,
                    "entry": entry_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "qty": qty,
                    "risk_usd": risk_usd,
                    "entry_bar": i,
                    "entry_dt": bar_dt,
                    "tp1_hit": False,
                    "partial_pnl": 0.0,
                    "entry_fee": entry_fee,
                    "entry_regime": int(regimes[i]) if has_regime else 0,
                }}
                last_entry = i

        mark_to_market = balance
        if position is not None:
            unrealized = (c - position["entry"]) * position["qty"] * position["side"]
            mark_to_market = balance + unrealized + position["partial_pnl"] - position["entry_fee"]
        equity.append(mark_to_market)

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

    # ── Monthly & Yearly Returns (Faithful Calendar Resampling) ───────────
    datetimes_pd = pd.to_datetime(df["datetime"])
    equity_series = pd.Series(equity, index=datetimes_pd)
    daily_bal = equity_series.resample("D").last().ffill()

    resampler_code_m = "ME" if pd.__version__ >= "2.2.0" else "M"
    resampler_code_y = "YE" if pd.__version__ >= "2.2.0" else "Y"

    monthly_bal = daily_bal.resample(resampler_code_m).last().dropna()
    yearly_bal = daily_bal.resample(resampler_code_y).last().dropna()

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

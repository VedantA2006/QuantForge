# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Main Application                              ║
# ║  FastAPI server + background discovery loop                  ║
# ╚══════════════════════════════════════════════════════════════╝

import sys
import os
import time
import asyncio
import logging
import traceback
from datetime import datetime, timezone
from contextlib import asynccontextmanager

# ── Ensure backend is importable ──────────────────────────────
# Add quantforge/ (parent of backend/) to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    API_HOST, API_PORT, CORS_ORIGINS, SYMBOL, INTERVALS,
    BATCH_SIZE, LOOP_DELAY_SECONDS, GA_EVOLVE_EVERY_N_CYCLES,
    GA_POPULATION_SIZE, GA_GENERATIONS,
)
from backend.data.cache import DataCache
from backend.data.indicators import compute_all_indicators
from backend.strategy.generator import generate_batch
from backend.strategy.tree import Strategy
from backend.engine.backtester import run_backtest
from backend.engine.validator import validate_strategy
from backend.engine.ranker import rank_strategies
from backend.engine.optimizer import evolve_population
from backend.db.mongo import MongoDB
from backend.api.routes import router, init_routes

# ── Logging ───────────────────────────────────────────────────
# Fix Windows console encoding for emoji characters
_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_stream = logging.StreamHandler(sys.stdout)
_stream.setFormatter(_fmt)
_stream.setLevel(logging.INFO)
# Use errors='replace' to handle emoji on Windows cp1252
try:
    import io
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

_file = logging.FileHandler("quantforge.log", encoding="utf-8")
_file.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_stream, _file])
log = logging.getLogger("quantforge.main")

# ── Global State ──────────────────────────────────────────────
state = {
    "status": "idle",
    "cycle": 0,
    "total_evaluated": 0,
    "total_passed": 0,
    "uptime_seconds": 0,
    "last_cycle_duration": 0,
    "data_loaded": False,
    "should_run": True,
    "batch_size": BATCH_SIZE,
    "ga_cycles": 0,
    "start_time": time.time(),
}

db = MongoDB()
data_cache = DataCache()


# ═══════════════════════════════════════════════════════════════
#  DISCOVERY LOOP
# ═══════════════════════════════════════════════════════════════

async def discovery_loop():
    """
    Main discovery loop:
    GENERATE → BACKTEST → VALIDATE → RANK → STORE → OPTIMIZE → REPEAT
    """
    log.info("=" * 60)
    log.info("  QuantForge — Strategy Discovery Engine Starting")
    log.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────
    try:
        log.info(f"Loading data for {SYMBOL} {INTERVALS}...")
        df_raw = data_cache.get_data(symbol=SYMBOL, interval=INTERVALS[0])
        df = compute_all_indicators(df_raw)
        state["data_loaded"] = True
        log.info(f"[OK] Data ready: {len(df):,} candles with indicators")
        db.insert_log("INFO", "main", f"Data loaded: {len(df)} candles", cycle=0)
    except Exception as e:
        log.error(f"[FAIL] Failed to load data: {e}")
        db.insert_log("ERROR", "main", f"Data load failed: {e}", cycle=0)
        state["status"] = "error"
        return

    ga_offspring = []

    while state["should_run"]:
        cycle_start = time.time()
        state["cycle"] += 1
        cycle = state["cycle"]
        state["status"] = "running"

        try:
            log.info(f"\n{'='*50}")
            log.info(f"  CYCLE {cycle}")
            log.info(f"{'='*50}")

            # ── 1. GENERATE ───────────────────────────────────
            strategies = generate_batch(
                ga_offspring=ga_offspring,
                batch_size=BATCH_SIZE,
            )
            ga_offspring = []  # consumed

            # ── 2. BACKTEST ───────────────────────────────────
            results = []
            for strat in strategies:
                bt = run_backtest(strat, df)
                state["total_evaluated"] += 1

                if bt.total_trades < 10:
                    continue

                # ── 3. VALIDATE ───────────────────────────────
                val = validate_strategy(strat, df)

                results.append((bt, val))

                # Store strategy + result
                db.insert_strategy(strat.to_dict())
                db.insert_result({
                    "strategy_id": strat.strategy_id,
                    "metrics": bt.to_dict(),
                    "validation": val.to_dict(),
                    "passed": val.passed,
                })

            # ── 4. RANK ───────────────────────────────────────
            ranked = rank_strategies(results)

            passed_count = len(ranked)
            state["total_passed"] += passed_count

            # ── 5. STORE best ─────────────────────────────────
            for r in ranked:   # store all ranked strategies
                strat_dict = next(
                    (s.to_dict() for s in strategies
                     if s.strategy_id == r.strategy_id),
                    {}
                )
                
                db.upsert_best({
                    "strategy_id": r.strategy_id,
                    "rank_score": r.rank_score,
                    "metrics": r.backtest.to_dict(),
                    "validation": r.validation.to_dict(),
                    "strategy": strat_dict,
                })
                
                # Check strict conditions for "Fully Passed"
                # 1. Ratio of winning trade > 2.0 * losing trade
                # 2. Minimum avg each month return >= 30%
                # 3. Win rate >= 65%
                bt = r.backtest
                monthly_avg = sum(bt.monthly_returns) / len(bt.monthly_returns) if bt.monthly_returns else 0.0
                
                if bt.avg_win >= (1.5 * bt.avg_loss) and monthly_avg >= 3.0 and bt.win_rate >= 35.0:
                    db.save_fully_passed({
                        "strategy_id": r.strategy_id,
                        "rank_score": r.rank_score,
                        "metrics": bt.to_dict(),
                        "validation": r.validation.to_dict(),
                        "strategy": strat_dict,
                        "monthly_avg": monthly_avg,
                        "win_loss_ratio": bt.avg_win / bt.avg_loss if bt.avg_loss > 0 else 100.0,
                    })
                    log.info(f"💎 SUPER STRATEGY DISCOVERED: {r.strategy_id} (Win Rate: {bt.win_rate:.1f}%, RR: {bt.avg_win/bt.avg_loss if bt.avg_loss>0 else 100:.1f}, Monthly: {monthly_avg:.1f}%)")

            # ── 6. OPTIMIZE (GA) every N cycles ───────────────
            if cycle % GA_EVOLVE_EVERY_N_CYCLES == 0:
                top_stored = db.get_top_strategies_with_fitness(
                    GA_POPULATION_SIZE
                )
                if len(top_stored) >= 4:
                    from backend.strategy.tree import strategy_from_dict
                    pop_with_fitness = []
                    for doc in top_stored:
                        try:
                            s = strategy_from_dict(doc.get("strategy", {}))
                            fitness = doc.get("rank_score", 0)
                            pop_with_fitness.append((s, fitness))
                        except Exception:
                            continue

                    if len(pop_with_fitness) >= 4:
                        ga_offspring = evolve_population(
                            pop_with_fitness,
                        )
                        state["ga_cycles"] += 1
                        log.info(
                            f"[GA] Evolution cycle {state['ga_cycles']}: "
                            f"{len(ga_offspring)} offspring generated"
                        )

            # ── Cycle summary ─────────────────────────────────
            cycle_dur = time.time() - cycle_start
            state["last_cycle_duration"] = round(cycle_dur, 2)
            state["uptime_seconds"] = round(time.time() - state["start_time"])

            log.info(
                f"  Cycle {cycle} complete: "
                f"evaluated={len(strategies)}, "
                f"passed={passed_count}, "
                f"duration={cycle_dur:.1f}s"
            )
            db.insert_log(
                "INFO", "main",
                f"Cycle {cycle}: {len(strategies)} eval, {passed_count} passed",
                cycle=cycle,
            )

        except Exception as e:
            log.error(f"Cycle {cycle} error: {e}\n{traceback.format_exc()}")
            db.insert_log("ERROR", "main", f"Cycle error: {e}", cycle=cycle)

        # ── Delay between cycles ──────────────────────────────
        await asyncio.sleep(LOOP_DELAY_SECONDS)

    state["status"] = "stopped"
    log.info("Discovery loop stopped")


# ═══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start discovery loop on app startup."""
    init_routes(db, state)
    task = asyncio.create_task(discovery_loop())
    yield
    state["should_run"] = False
    task.cancel()


app = FastAPI(
    title="QuantForge",
    description="Autonomous Trading Strategy Discovery Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "QuantForge",
        "version": "1.0.0",
        "status": state.get("status", "idle"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)

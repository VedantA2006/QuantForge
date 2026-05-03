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
from backend.strategy.generator import generate_batch, update_category_weights, record_category_stats
from backend.strategy.tree import Strategy
from backend.engine.backtester import run_backtest
from backend.engine.validator import validate_strategy
from backend.engine.ranker import rank_strategies
from backend.engine.optimizer import evolve_population, BayesianOptimizer
from backend.engine.surrogate import SurrogateModel
from backend.engine.parallel import StrategyPool
from backend.engine.rl_builder import RLBuilder
from backend.engine.hof import check_hof_promotion
from backend.db.mongo import MongoDB
from backend.api.routes import router, init_routes

# ── ML Global Instances ───────────────────────────────────────
surrogate_model = SurrogateModel()
rl_builder = RLBuilder()
strategy_pool = StrategyPool()

# ── Logging ───────────────────────────────────────────────────
_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_stream = logging.StreamHandler(sys.stdout)
_stream.setFormatter(_fmt)
_stream.setLevel(logging.INFO)
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
    "bayes_cycles": 0,
    "start_time": time.time(),
}

db = MongoDB()
data_cache = DataCache()
bayesian_optimizer = BayesianOptimizer()


# ═══════════════════════════════════════════════════════════════
#  DISCOVERY LOOP
# ═══════════════════════════════════════════════════════════════

async def discovery_loop():
    """
    Main discovery loop:
    GENERATE → BACKTEST → VALIDATE → RANK → STORE → OPTIMIZE → REPEAT
    """
    log.info("=" * 60)
    log.info("  QuantForge v2.0 — Strategy Discovery Engine Starting")
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

            # ── Update category weights from DB ───────────────
            update_category_weights(db)

            # ── 1. GENERATE ───────────────────────────────────
            # Get Bayesian suggestions every GA cycle
            bayes_suggestions = None
            if cycle % GA_EVOLVE_EVERY_N_CYCLES == 0:
                bayes_suggestions = bayesian_optimizer.suggest(SYMBOL, n=5)
                if bayes_suggestions:
                    state["bayes_cycles"] += 1

            batch = generate_batch(
                ga_offspring=ga_offspring,
                batch_size=BATCH_SIZE,
                bayesian_suggestions=bayes_suggestions,
                rl_agent=rl_builder,
            )
            ga_offspring = []  # consumed
            
            # Extract strategies and episodes
            strategies = [item[0] for item in batch]
            episodes = {item[0].strategy_id: item[1] for item in batch if item[1]}

            # ── SURROGATE FILTER ──────────────────────────────
            from backend.config import SURROGATE_FILTER_THRESHOLD
            filtered_strategies = []
            
            # Get current best score for thresholding
            best_score = 0.0
            top = db.get_best(1)
            if top:
                best_score = top[0].get("rank_score", 0.0)
                
            threshold = SURROGATE_FILTER_THRESHOLD * best_score
            
            for strat in strategies:
                # Cold start: allow all if not trained
                if not surrogate_model.is_trained:
                    filtered_strategies.append(strat)
                    continue
                    
                pred = surrogate_model.predict(strat)
                if pred >= threshold or strat.origin in ["elite", "bayesian", "crossover"]:
                    filtered_strategies.append(strat)
                else:
                    surrogate_model.stats["strategies_filtered"] += 1
                    
            log.info(f"[SURROGATE] Allowed {len(filtered_strategies)}/{len(strategies)} strategies (threshold={threshold:.2f})")

            # ── 2 & 3. BACKTEST & VALIDATE (Parallel) ─────────
            results = []
            if not strategy_pool.pool:
                strategy_pool.setup(df)
                
            # Run the parallel batch
            evaluated = strategy_pool.evaluate_batch(filtered_strategies, df)
            
            for strat, bt, val in evaluated:
                state["total_evaluated"] += 1
                
                # Surrogate feedback
                if bt.total_trades >= 10:
                    actual_score = bt.sharpe_ratio * 0.3 + bt.total_return_pct * 0.001
                    pred = surrogate_model.predict(strat)
                    if surrogate_model.is_trained:
                        surrogate_model.record_actual(pred, actual_score)
                    
                    # Record for Bayesian optimizer
                    bayesian_optimizer.record(
                        SYMBOL,
                        strat.risk_params.param_vector(),
                        actual_score,
                        db
                    )

                results.append((bt, val))

                # Store strategy + result
                db.insert_strategy(strat.to_dict())
                db.insert_result({
                    "strategy_id": strat.strategy_id,
                    "metrics": bt.to_dict(),
                    "validation": val.to_dict(),
                    "passed": val.passed,
                })

                # Record category stats
                score = bt.sharpe_ratio * 0.3 + bt.total_return_pct * 0.001 if bt.total_trades > 0 else 0.0
                record_category_stats(db, strat, score, is_top20=False)

            # ── 4. RANK ───────────────────────────────────────
            ranked = rank_strategies(results)

            passed_count = len(ranked)
            state["total_passed"] += passed_count

            # ── 5. STORE best ─────────────────────────────────
            for idx, r in enumerate(ranked):
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

                # Check HOF Promotion
                if check_hof_promotion(r):
                    strat_dict["is_hof"] = True
                    db.db.hof_strategies.update_one(
                        {"strategy_id": r.strategy_id},
                        {"$set": {
                            "strategy_id": r.strategy_id,
                            "rank_score": r.rank_score,
                            "metrics": r.backtest.to_dict(),
                            "validation": r.validation.to_dict(),
                            "strategy": strat_dict,
                        }},
                        upsert=True
                    )
                    log.info(f"[HOF] Strategy {r.strategy_id} promoted — Sharpe={r.backtest.sharpe_ratio:.2f}, Return={r.backtest.total_return_pct:.1f}%, DD={r.backtest.max_drawdown_pct:.1f}%")

                # Record top 20 category stats
                if idx < 20:
                    strat_obj = next(
                        (s for s in filtered_strategies if s.strategy_id == r.strategy_id),
                        None,
                    )
                    if strat_obj:
                        record_category_stats(db, strat_obj, r.rank_score, is_top20=True)
                        
                # Update RL Agent
                if r.strategy_id in episodes:
                    rl_builder.update(episodes[r.strategy_id], r.rank_score)

            # Check strict conditions for "Fully Passed"
            for r in ranked:
                bt = r.backtest
                monthly_avg = sum(bt.monthly_returns) / len(bt.monthly_returns) if bt.monthly_returns else 0.0

                if bt.avg_win >= (1.5 * bt.avg_loss) and monthly_avg >= 3.0 and bt.win_rate >= 35.0:
                    strat_dict = next(
                        (s.to_dict() for s in filtered_strategies if s.strategy_id == r.strategy_id),
                        {}
                    )
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

            # ── 6. OPTIMIZE (GA & Surrogate) ──────────────────
            from backend.config import SURROGATE_RETRAIN_EVERY
            if cycle % SURROGATE_RETRAIN_EVERY == 0:
                # Retrain surrogate
                recent = list(db.db.results.find({}, {"_id": 0, "strategy_id": 1, "passed": 1, "metrics.sharpe_ratio": 1, "metrics.total_return_pct": 1}).sort("_id", -1).limit(2000))
                if recent:
                    s_ids = [d["strategy_id"] for d in recent]
                    strat_docs = list(db.db.strategies.find({"strategy_id": {"$in": s_ids}}))
                    doc_map = {d["strategy_id"]: d for d in strat_docs}
                    
                    train_strats = []
                    train_scores = []
                    from backend.strategy.tree import strategy_from_dict
                    for r in recent:
                        if r["strategy_id"] in doc_map:
                            try:
                                s = strategy_from_dict(doc_map[r["strategy_id"]])
                                m = r.get("metrics", {})
                                score = (m.get("sharpe_ratio", 0) * 0.3) + (m.get("total_return_pct", 0) * 0.001)
                                train_strats.append(s)
                                train_scores.append(score)
                            except Exception:
                                pass
                    surrogate_model.train(train_strats, train_scores)

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
    init_routes(db, state)
    bayesian_optimizer.load_history(db)
    task = asyncio.create_task(discovery_loop())
    yield
    state["should_run"] = False
    task.cancel()
    strategy_pool.close()


app = FastAPI(
    title="QuantForge",
    description="Autonomous Trading Strategy Discovery Engine",
    version="2.0.0",
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
        "version": "2.0.0",
        "status": state.get("status", "idle"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)

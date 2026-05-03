# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Multiprocessing Pool                          ║
# ║  Evaluates strategies in parallel to speed up discovery     ║
# ╚══════════════════════════════════════════════════════════════╝

import io
import logging
import multiprocessing as mp
import pandas as pd
from typing import Dict, Tuple, Optional

from backend.strategy.tree import strategy_from_dict
from backend.engine.backtester import run_backtest
from backend.engine.validator import validate_strategy
from backend.config import POOL_ENABLED

log = logging.getLogger("quantforge.engine.parallel")

_global_df = None

def init_worker(df_bytes: bytes):
    """Initialize worker process by deserializing the dataframe once."""
    global _global_df
    try:
        _global_df = pd.read_parquet(io.BytesIO(df_bytes))
    except Exception as e:
        print(f"Worker init failed: {e}")

def evaluate_worker(strategy_dict: dict) -> Optional[Tuple[str, dict, dict, dict]]:
    """Worker function to backtest, prune, and validate a single strategy."""
    global _global_df
    if _global_df is None:
        return None
        
    try:
        strat = strategy_from_dict(strategy_dict)
        bt = run_backtest(strat, _global_df)
        
        if bt.total_trades < 10:
            return (strat.strategy_id, bt.to_dict(), {"passed": False, "rejection_reason": "too_few_trades"}, strat.to_dict())
            
        base_score = bt.sharpe_ratio * 0.3 + bt.total_return_pct * 0.001
        
        # Pruning
        from backend.engine.pruner import prune_strategy
        from backend.config import PRUNER_ENABLED
        if PRUNER_ENABLED:
            strat, _ = prune_strategy(strat, _global_df, base_score)
            # Re-evaluate with pruned strategy if it changed
            # Actually, prune_strategy returns the backtested result implicitly, but we must re-evaluate on full df
            # since pruner uses full df in this scope.
            bt = run_backtest(strat, _global_df)
            
        val = validate_strategy(strat, _global_df)
        return (strat.strategy_id, bt.to_dict(), val.to_dict(), strat.to_dict())
    except Exception as e:
        return None

class StrategyPool:
    def __init__(self):
        self.pool = None
        self._df_bytes = None
        
    def setup(self, df: pd.DataFrame):
        if not POOL_ENABLED:
            return
            
        try:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            self._df_bytes = buf.getvalue()
            
            n_cores = min(mp.cpu_count(), 8)
            self.pool = mp.Pool(
                processes=n_cores,
                initializer=init_worker,
                initargs=(self._df_bytes,)
            )
            log.info(f"Initialized Multiprocessing Pool with {n_cores} workers")
        except Exception as e:
            log.warning(f"Failed to initialize pool: {e}")
            self.pool = None
            
    def evaluate_batch(self, strategies: list, fallback_df: pd.DataFrame) -> list:
        """Evaluate a batch of strategies, returning results."""
        if self.pool is None or not POOL_ENABLED:
            # Sequential fallback
            results = []
            for s in strategies:
                try:
                    bt = run_backtest(s, fallback_df)
                    if bt.total_trades < 10:
                        continue
                    
                    from backend.engine.pruner import prune_strategy
                    from backend.config import PRUNER_ENABLED
                    if PRUNER_ENABLED:
                        base_score = bt.sharpe_ratio * 0.3 + bt.total_return_pct * 0.001
                        s, _ = prune_strategy(s, fallback_df, base_score)
                        bt = run_backtest(s, fallback_df)
                        
                    val = validate_strategy(s, fallback_df)
                    results.append((s, bt, val))
                except Exception as e:
                    log.warning(f"Strategy eval failed: {e}")
            return results
            
        # Parallel execution
        dicts = [s.to_dict() for s in strategies]
        results_raw = []
        
        try:
            for res in self.pool.imap_unordered(evaluate_worker, dicts):
                if res is not None:
                    results_raw.append(res)
        except Exception as e:
            log.error(f"Pool evaluation failed: {e}")
            
        # Reconstruct objects
        from backend.engine.backtester import BacktestResult
        from backend.engine.validator import ValidationResult
        
        final_results = []
        for sid, bt_dict, val_dict, strat_dict in results_raw:
            if val_dict.get("passed", False) or val_dict.get("rejection_reason") != "too_few_trades":
                s = strategy_from_dict(strat_dict)
                bt = BacktestResult(**bt_dict)
                val = ValidationResult(**val_dict)
                final_results.append((s, bt, val))
                
        return final_results
        
    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()

# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Hall of Fame Gate                             ║
# ║  Evaluates if a top strategy belongs in the HOF             ║
# ╚══════════════════════════════════════════════════════════════╝

import logging
from backend.engine.ranker import RankedStrategy
from backend.config import HOF_ENABLED

log = logging.getLogger("quantforge.engine.hof")

def check_hof_promotion(ranked: RankedStrategy) -> bool:
    """
    Evaluates a RankedStrategy against strict Hall of Fame criteria.
    Returns True if it qualifies.
    """
    if not HOF_ENABLED:
        return False
        
    bt = ranked.backtest
    val = ranked.validation
    
    # ── Strict Thresholds ──────────────────────────────────────────────
    if bt.sharpe_ratio < 1.0: return False
    if bt.max_drawdown_pct > 30.0: return False
    if not (30.0 <= bt.win_rate <= 80.0): return False
    if bt.avg_monthly_return < 1.0: return False
    if val.walk_forward_ratio < 0.50: return False
    if bt.mc_drawdown_p95 > 35.0: return False
    if bt.parameter_sensitivity > 0.40: return False
    if bt.avg_trades_per_month < 1.5: return False
    
    # ── Walk-forward fold profitability (3 out of 5) ───────────────────
    profitable_folds = 0
    for fold in val.fold_results:
        if fold.get("return_pct", 0) > 0:
            profitable_folds += 1
            
    if profitable_folds < 3:
        return False
        
    # ── Yearly return profitability (all years > 0) ────────────────────
    yearly = bt.yearly_returns
    if not yearly: return False
    
    for y_ret in yearly.values():
        if y_ret <= 0:
            return False
            
    return True

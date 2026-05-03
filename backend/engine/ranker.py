# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Calmar-Based Composite Ranking Engine        ║
# ╚══════════════════════════════════════════════════════════════╝

import numpy as np
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

from backend.engine.backtester import BacktestResult
from backend.engine.validator import ValidationResult

log = logging.getLogger("quantforge.engine.ranker")


@dataclass
class RankedStrategy:
    strategy_id: str
    rank_score: float
    sharpe_norm: float
    return_norm: float
    dd_norm: float
    consistency_norm: float
    backtest: BacktestResult
    validation: ValidationResult

    def to_dict(self):
        return {
            "strategy_id": self.strategy_id,
            "rank_score": round(self.rank_score, 6),
            "components": {
                "sharpe": round(self.sharpe_norm, 4),
                "return": round(self.return_norm, 4),
                "drawdown": round(self.dd_norm, 4),
                "consistency": round(self.consistency_norm, 4),
            },
            "metrics": self.backtest.to_dict(),
            "validation": self.validation.to_dict(),
        }


def rank_strategies(
    results: List[Tuple[BacktestResult, ValidationResult]],
) -> List[RankedStrategy]:
    """
    Rank strategies using Calmar-based composite score with hard gates.
    Only strategies that passed validation are ranked.
    """
    # Filter to validated strategies with enough trades
    valid = [
        (bt, val) for bt, val in results
        if val.passed and bt.total_trades >= 10
    ]

    if not valid:
        log.info("No strategies passed validation for ranking")
        return []

    # ── Hard gates — disqualify immediately ───────────────────────────────
    gated = []
    for bt, val in valid:
        # Max drawdown > 20% → reject
        if bt.max_drawdown_pct > 20:
            continue
        # Win rate out of bounds
        if bt.win_rate < 30 or bt.win_rate > 85:
            continue
        # Too few trades per month
        if bt.avg_trades_per_month < 1:
            continue
        # Monte Carlo DD too high
        if bt.mc_drawdown_p95 > 25:
            continue
        # Parameter sensitivity too high
        if bt.parameter_sensitivity > 0.4:
            continue
        # Avg monthly return too low
        if bt.avg_monthly_return < 1.0:
            continue
        gated.append((bt, val))

    if not gated:
        log.info("No strategies passed hard gates for ranking")
        return []

    # ── Calmar-based composite scoring ────────────────────────────────────
    ranked = []
    for bt, val in gated:
        # Calmar ratio
        calmar = bt.total_return_pct / bt.max_drawdown_pct if bt.max_drawdown_pct > 0 else 0

        # Regime consistency penalty
        regime_wrs = [bt.regime_bull_wr, bt.regime_bear_wr, bt.regime_sideways_wr]
        active_wrs = [w for w in regime_wrs if w > 0]
        if active_wrs:
            variance_penalty = max(active_wrs) - min(active_wrs)
            regime_score = max(0, 10 - variance_penalty / 5)
        else:
            regime_score = 5.0  # neutral if no regime data

        composite = (
            bt.total_return_pct        * 0.25 +
            calmar                     * 0.20 +
            bt.profit_factor           * 0.15 +
            bt.sharpe_ratio            * 0.15 +
            val.walk_forward_ratio * 10 * 0.10 +
            regime_score               * 0.10 +
            -bt.mc_drawdown_p95        * 0.05
        )

        ranked.append(RankedStrategy(
            strategy_id=bt.strategy_id,
            rank_score=float(composite),
            sharpe_norm=float(bt.sharpe_ratio),
            return_norm=float(bt.total_return_pct),
            dd_norm=float(bt.max_drawdown_pct),
            consistency_norm=float(val.consistency_score),
            backtest=bt,
            validation=val,
        ))

    ranked.sort(key=lambda r: r.rank_score, reverse=True)

    if ranked:
        top = ranked[0]
        log.info(
            f"[BEST] Top strategy: {top.strategy_id} "
            f"score={top.rank_score:.4f} "
            f"sharpe={top.backtest.sharpe_ratio:.2f} "
            f"return={top.backtest.total_return_pct:.1f}% "
            f"dd={top.backtest.max_drawdown_pct:.1f}%"
        )

    return ranked

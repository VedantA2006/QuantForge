# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Ranking Engine                                ║
# ║  Weighted composite scoring for strategy comparison         ║
# ╚══════════════════════════════════════════════════════════════╝

import numpy as np
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

from backend.engine.backtester import BacktestResult
from backend.engine.validator import ValidationResult
from backend.config import (
    RANK_W_SHARPE, RANK_W_RETURN, RANK_W_DRAWDOWN, RANK_W_CONSISTENCY,
)

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


def _normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Handles edge cases."""
    if len(values) == 0:
        return values
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-10:
        return np.ones_like(values) * 0.5
    return (values - vmin) / (vmax - vmin)


def rank_strategies(
    results: List[Tuple[BacktestResult, ValidationResult]],
) -> List[RankedStrategy]:
    """
    Rank strategies using weighted composite score.

    Score = W_sharpe × norm(sharpe)
          + W_return × norm(return)
          + W_dd     × norm(1 / max_dd)
          + W_consist × norm(consistency)

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

    # Extract raw metrics
    sharpes = np.array([bt.sharpe_ratio for bt, _ in valid])
    returns = np.array([bt.total_return_pct for bt, _ in valid])
    # Invert drawdown: lower DD = better score
    dds = np.array([
        1.0 / max(bt.max_drawdown_pct, 0.1) for bt, _ in valid
    ])
    consistencies = np.array([val.consistency_score for _, val in valid])

    # Normalize each dimension
    s_norm = _normalize(sharpes)
    r_norm = _normalize(returns)
    d_norm = _normalize(dds)
    c_norm = _normalize(consistencies)

    # Composite score
    scores = (
        RANK_W_SHARPE * s_norm +
        RANK_W_RETURN * r_norm +
        RANK_W_DRAWDOWN * d_norm +
        RANK_W_CONSISTENCY * c_norm
    )

    # Build ranked list
    ranked = []
    for i, (bt, val) in enumerate(valid):
        ranked.append(RankedStrategy(
            strategy_id=bt.strategy_id,
            rank_score=float(scores[i]),
            sharpe_norm=float(s_norm[i]),
            return_norm=float(r_norm[i]),
            dd_norm=float(d_norm[i]),
            consistency_norm=float(c_norm[i]),
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

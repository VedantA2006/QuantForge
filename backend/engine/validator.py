# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Walk-Forward Validation Engine                ║
# ║  Rejects overfitted strategies via expanding-window OOS     ║
# ╚══════════════════════════════════════════════════════════════╝

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from backend.strategy.tree import Strategy
from backend.engine.backtester import run_backtest, BacktestResult
from backend.config import (
    VALIDATION_FOLDS, TRAIN_RATIO, MIN_TRADES_PER_FOLD,
    OOS_SHARPE_DECAY_LIMIT, OOS_DD_AMPLIFY_LIMIT, WR_VARIANCE_LIMIT,
)

log = logging.getLogger("quantforge.engine.validator")


@dataclass
class ValidationResult:
    strategy_id: str = ""
    passed: bool = False
    rejection_reason: str = ""
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    fold_results: List[Dict] = field(default_factory=list)
    consistency_score: float = 0.0

    def to_dict(self):
        return {
            "strategy_id": self.strategy_id,
            "passed": self.passed,
            "rejection_reason": self.rejection_reason,
            "is_sharpe": round(self.is_sharpe, 4),
            "oos_sharpe": round(self.oos_sharpe, 4),
            "fold_results": self.fold_results,
            "consistency_score": round(self.consistency_score, 4),
        }


def validate_strategy(
    strategy: Strategy,
    df: pd.DataFrame,
    n_folds: int = VALIDATION_FOLDS,
) -> ValidationResult:
    """
    Walk-forward validation with expanding training window.

    Protocol:
    - Split data into n_folds expanding windows
    - Each fold: train on [0:split], test on [split:end_of_fold]
    - Backtest on both in-sample and out-of-sample
    - Reject if OOS performance degrades too much vs IS
    """
    result = ValidationResult(strategy_id=strategy.strategy_id)

    n = len(df)
    if n < 200:
        result.rejection_reason = "insufficient_data"
        return result

    # ── Simple train/test split first ─────────────────────────────────
    split_idx = int(n * TRAIN_RATIO)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    is_result = run_backtest(strategy, train_df)
    oos_result = run_backtest(strategy, test_df)

    result.is_sharpe = is_result.sharpe_ratio
    result.oos_sharpe = oos_result.sharpe_ratio

    # ── Quick rejection checks ────────────────────────────────────────
    if is_result.total_trades < MIN_TRADES_PER_FOLD:
        result.rejection_reason = f"too_few_trades_is ({is_result.total_trades})"
        return result

    if oos_result.total_trades < MIN_TRADES_PER_FOLD:
        result.rejection_reason = f"too_few_trades_oos ({oos_result.total_trades})"
        return result

    if oos_result.profit_factor < 1.0:
        result.rejection_reason = f"oos_pf_below_1 ({oos_result.profit_factor:.2f})"
        return result

    # ── Sharpe decay check ────────────────────────────────────────────
    if is_result.sharpe_ratio > 0:
        sharpe_ratio = oos_result.sharpe_ratio / is_result.sharpe_ratio
        if sharpe_ratio < OOS_SHARPE_DECAY_LIMIT:
            result.rejection_reason = (
                f"sharpe_decay ({oos_result.sharpe_ratio:.2f} vs "
                f"{is_result.sharpe_ratio:.2f} = {sharpe_ratio:.1%})"
            )
            return result

    # ── Drawdown amplification check ──────────────────────────────────
    if is_result.max_drawdown_pct > 0:
        dd_ratio = oos_result.max_drawdown_pct / is_result.max_drawdown_pct
        if dd_ratio > OOS_DD_AMPLIFY_LIMIT:
            result.rejection_reason = (
                f"dd_amplification ({oos_result.max_drawdown_pct:.1f}% vs "
                f"{is_result.max_drawdown_pct:.1f}% = {dd_ratio:.1f}x)"
            )
            return result

    # ── Walk-forward folds ────────────────────────────────────────────
    fold_sharpes = []
    fold_wrs = []
    min_train = max(200, n // (n_folds + 1))
    chunk = (n - min_train) // n_folds

    for fold in range(n_folds):
        tr_end = min_train + chunk * fold
        te_end = min(tr_end + chunk, n)
        if tr_end >= n or te_end <= tr_end:
            break

        fold_test = df.iloc[tr_end:te_end]
        if len(fold_test) < 30:
            continue

        fold_result = run_backtest(strategy, fold_test)
        fold_sharpes.append(fold_result.sharpe_ratio)
        fold_wrs.append(fold_result.win_rate)

        result.fold_results.append({
            "fold": fold + 1,
            "train_size": tr_end,
            "test_size": te_end - tr_end,
            "sharpe": round(fold_result.sharpe_ratio, 4),
            "win_rate": round(fold_result.win_rate, 2),
            "pf": round(fold_result.profit_factor, 3),
            "trades": fold_result.total_trades,
            "return_pct": round(fold_result.total_return_pct, 2),
        })

    # ── Win rate consistency check ────────────────────────────────────
    if len(fold_wrs) >= 3:
        wr_std = np.std(fold_wrs) / 100.0
        if wr_std > WR_VARIANCE_LIMIT:
            result.rejection_reason = (
                f"win_rate_unstable (std={wr_std:.3f} > {WR_VARIANCE_LIMIT})"
            )
            return result

    # ── Consistency score ─────────────────────────────────────────────
    if len(fold_sharpes) >= 2 and np.std(fold_sharpes) > 0:
        result.consistency_score = np.mean(fold_sharpes) / np.std(fold_sharpes)
    elif len(fold_sharpes) >= 1:
        result.consistency_score = np.mean(fold_sharpes)
    else:
        result.consistency_score = 0.0

    # ── Profit factor check on any OOS fold ───────────────────────────
    for fr in result.fold_results:
        if fr["pf"] < 1.0 and fr["trades"] >= 10:
            result.rejection_reason = (
                f"fold_{fr['fold']}_pf_below_1 (pf={fr['pf']:.2f})"
            )
            return result

    result.passed = True
    log.debug(
        f"[PASS] {strategy.name} passed validation: "
        f"IS_sharpe={result.is_sharpe:.2f}, OOS_sharpe={result.oos_sharpe:.2f}, "
        f"consistency={result.consistency_score:.2f}"
    )
    return result

# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Walk-Forward Validation + MC + Sensitivity    ║
# ╚══════════════════════════════════════════════════════════════╝

import random as _random
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from backend.strategy.tree import Strategy, RiskParams
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
    walk_forward_ratio: float = 0.0

    def to_dict(self):
        from backend.utils import clean_for_mongo
        return clean_for_mongo({
            "strategy_id": self.strategy_id,
            "passed": self.passed,
            "rejection_reason": self.rejection_reason,
            "is_sharpe": round(self.is_sharpe, 4),
            "oos_sharpe": round(self.oos_sharpe, 4),
            "fold_results": self.fold_results,
            "consistency_score": round(self.consistency_score, 4),
            "walk_forward_ratio": round(self.walk_forward_ratio, 4),
        })


def _monte_carlo_dd(trades: list, n_sims: int = 200) -> float:
    """Monte Carlo drawdown simulation. Returns 95th percentile max drawdown."""
    pnls = [t["pnl"] for t in trades]
    if not pnls:
        return 0.0
    initial = 10000.0
    dds = []
    for _ in range(n_sims):
        _random.shuffle(pnls)
        bal = initial
        peak = initial
        max_dd = 0.0
        for p in pnls:
            bal += p
            if bal > peak:
                peak = bal
            dd = (peak - bal) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        dds.append(max_dd)
    return float(np.percentile(dds, 95))


def _parameter_sensitivity(strategy: Strategy, df: pd.DataFrame, base_score: float) -> float:
    """Check if small parameter perturbations cause large score drops."""
    import copy
    perturbations = [
        {"sl_atr_mult": strategy.risk_params.sl_atr_mult + 0.2},
        {"sl_atr_mult": max(0.5, strategy.risk_params.sl_atr_mult - 0.2)},
        {"rr_ratio": strategy.risk_params.rr_ratio + 0.3},
        {"rr_ratio": max(1.0, strategy.risk_params.rr_ratio - 0.3)},
    ]

    if base_score <= 0:
        return 0.0

    max_drop = 0.0
    for p in perturbations:
        s_copy = copy.deepcopy(strategy)
        for k, v in p.items():
            setattr(s_copy.risk_params, k, round(v, 1))
        try:
            bt = run_backtest(s_copy, df)
            if bt.total_trades < 5:
                continue
            # Simple composite score for comparison
            perturbed_score = bt.sharpe_ratio * 0.3 + bt.total_return_pct * 0.001
            drop = 1.0 - (perturbed_score / base_score) if base_score > 0 else 0
            if drop > max_drop:
                max_drop = drop
        except Exception:
            continue

    return round(max(0, max_drop), 3)


def validate_strategy(
    strategy: Strategy,
    df: pd.DataFrame,
    n_folds: int = VALIDATION_FOLDS,
) -> ValidationResult:
    """Walk-forward validation with MC drawdown and parameter sensitivity."""
    result = ValidationResult(strategy_id=strategy.strategy_id)

    n = len(df)
    if n < 200:
        result.rejection_reason = "insufficient_data"
        return result

    # ── Simple train/test split ───────────────────────────────────────────
    split_idx = int(n * TRAIN_RATIO)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    is_result = run_backtest(strategy, train_df)
    oos_result = run_backtest(strategy, test_df)

    result.is_sharpe = is_result.sharpe_ratio
    result.oos_sharpe = oos_result.sharpe_ratio

    # ── Quick rejection checks ────────────────────────────────────────────
    if is_result.total_trades < MIN_TRADES_PER_FOLD:
        result.rejection_reason = f"too_few_trades_is ({is_result.total_trades})"
        return result

    if oos_result.total_trades < MIN_TRADES_PER_FOLD:
        result.rejection_reason = f"too_few_trades_oos ({oos_result.total_trades})"
        return result

    bars_per_trade = len(df) / max(is_result.total_trades, 1)
    if bars_per_trade < 20:
        result.rejection_reason = f"overtrading ({is_result.total_trades} trades)"
        return result

    if oos_result.profit_factor < 1.0:
        result.rejection_reason = f"oos_pf_below_1 ({oos_result.profit_factor:.2f})"
        return result

    if oos_result.avg_monthly_return < 0:
        result.rejection_reason = f"oos_monthly_negative ({oos_result.avg_monthly_return:.2f}%)"
        return result

    # ── Sharpe decay check ────────────────────────────────────────────────
    if is_result.sharpe_ratio > 0:
        sharpe_ratio = oos_result.sharpe_ratio / is_result.sharpe_ratio
        result.walk_forward_ratio = sharpe_ratio
        if sharpe_ratio < OOS_SHARPE_DECAY_LIMIT:
            result.rejection_reason = (
                f"sharpe_decay ({oos_result.sharpe_ratio:.2f} vs "
                f"{is_result.sharpe_ratio:.2f} = {sharpe_ratio:.1%})"
            )
            return result

    # ── Drawdown amplification check ──────────────────────────────────────
    if is_result.max_drawdown_pct > 0:
        dd_ratio = oos_result.max_drawdown_pct / is_result.max_drawdown_pct
        if dd_ratio > OOS_DD_AMPLIFY_LIMIT:
            result.rejection_reason = (
                f"dd_amplification ({oos_result.max_drawdown_pct:.1f}% vs "
                f"{is_result.max_drawdown_pct:.1f}% = {dd_ratio:.1f}x)"
            )
            return result

    # ── Walk-forward folds (over training data only, not full df) ─────────
    fold_sharpes = []
    fold_wrs = []
    n_train = len(train_df)
    min_train = max(200, n_train // (n_folds + 1))
    chunk = (n_train - min_train) // n_folds

    for fold in range(n_folds):
        tr_end = min_train + chunk * fold
        te_end = min(tr_end + chunk, n_train)
        if tr_end >= n_train or te_end <= tr_end:
            break

        fold_train = train_df.iloc[:tr_end]
        fold_test = train_df.iloc[tr_end:te_end]
        if len(fold_test) < 30:
            continue

        is_fold = run_backtest(strategy, fold_train)
        oos_fold = run_backtest(strategy, fold_test)

        fold_sharpes.append(oos_fold.sharpe_ratio)
        fold_wrs.append(oos_fold.win_rate)

        result.fold_results.append({
            "fold": fold + 1,
            "train_size": tr_end,
            "test_size": te_end - tr_end,
            "is_sharpe": round(is_fold.sharpe_ratio, 4),
            "oos_sharpe": round(oos_fold.sharpe_ratio, 4),
            "sharpe": round(oos_fold.sharpe_ratio, 4),
            "win_rate": round(oos_fold.win_rate, 2),
            "pf": round(oos_fold.profit_factor, 3),
            "trades": oos_fold.total_trades,
            "return_pct": round(oos_fold.total_return_pct, 2),
        })

    # ── Win rate consistency check ────────────────────────────────────────
    if len(fold_wrs) >= 3:
        wr_std = np.std(fold_wrs) / 100.0
        if wr_std > WR_VARIANCE_LIMIT:
            result.rejection_reason = (
                f"win_rate_unstable (std={wr_std:.3f} > {WR_VARIANCE_LIMIT})"
            )
            return result

    # ── Consistency score ─────────────────────────────────────────────────
    if len(fold_sharpes) >= 2 and np.std(fold_sharpes) > 0:
        result.consistency_score = np.mean(fold_sharpes) / np.std(fold_sharpes)
    elif len(fold_sharpes) >= 1:
        result.consistency_score = np.mean(fold_sharpes)
    else:
        result.consistency_score = 0.0

    # ── Profit factor check on any OOS fold ───────────────────────────────
    for fr in result.fold_results:
        if fr["pf"] < 1.0 and fr["trades"] >= 10:
            result.rejection_reason = (
                f"fold_{fr['fold']}_pf_below_1 (pf={fr['pf']:.2f})"
            )
            return result

    # ── Monte Carlo drawdown ──────────────────────────────────────────────
    full_bt = run_backtest(strategy, df)
    if full_bt.trade_history:
        mc_dd = _monte_carlo_dd(full_bt.trade_history, n_sims=200)
        full_bt.mc_drawdown_p95 = mc_dd
        if mc_dd > 25:
            result.rejection_reason = f"mc_drawdown_p95 ({mc_dd:.1f}% > 25%)"
            return result

    # ── Parameter sensitivity ─────────────────────────────────────────────
    base_score = is_result.sharpe_ratio * 0.3 + is_result.total_return_pct * 0.001
    sensitivity = _parameter_sensitivity(strategy, train_df, base_score)
    full_bt.parameter_sensitivity = sensitivity
    if sensitivity > 0.4:
        result.rejection_reason = f"parameter_sensitive (drop={sensitivity:.1%})"
        return result

    result.passed = True
    log.debug(
        f"[PASS] {strategy.name} passed validation: "
        f"IS_sharpe={result.is_sharpe:.2f}, OOS_sharpe={result.oos_sharpe:.2f}, "
        f"consistency={result.consistency_score:.2f}"
    )
    return result

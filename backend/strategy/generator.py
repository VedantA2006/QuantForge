# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Strategy Generator                            ║
# ║  Random + Template + GA offspring generation                ║
# ╚══════════════════════════════════════════════════════════════╝

import random
import logging
from typing import List
from backend.config import (
    BATCH_SIZE, MAX_TREE_DEPTH, RANDOM_RATIO, TEMPLATE_RATIO, GA_RATIO,
)
from backend.strategy.tree import (
    Strategy, RiskParams, Node, BooleanNode, ComparisonNode,
    IndicatorNode, ConstantNode, ArithmeticNode,
)
from backend.strategy.templates import generate_from_template
from backend.data.indicators import ALL_INDICATOR_COLUMNS

log = logging.getLogger("quantforge.strategy.generator")

# Indicators suitable for comparison with constants
_BOUNDED_INDICATORS = [
    "rsi_7", "rsi_14", "rsi_21", "stoch_k", "stoch_d", "adx_14",
]
_BOUNDED_RANGES = {
    "rsi_7": (20, 80), "rsi_14": (20, 80), "rsi_21": (20, 80),
    "stoch_k": (15, 85), "stoch_d": (15, 85), "adx_14": (15, 50),
}
# Indicators suitable for cross-comparison
_CROSS_INDICATORS = [
    "ema_8", "ema_13", "ema_21", "ema_34", "ema_55", "ema_89", "ema_200",
    "sma_20", "sma_50", "sma_200", "bb_upper", "bb_middle", "bb_lower",
    "close", "open", "high", "low", "macd_line", "macd_signal",
]
_COMPARISON_OPS = [">", "<", ">=", "<=", "crossover", "crossunder"]
_BOOLEAN_OPS = ["AND", "OR"]


def _random_leaf(allow_constant: bool = True) -> Node:
    """Generate a random leaf node."""
    if allow_constant and random.random() < 0.3:
        # Bounded indicator vs constant
        ind = random.choice(_BOUNDED_INDICATORS)
        lo, hi = _BOUNDED_RANGES[ind]
        return ConstantNode(value=round(random.uniform(lo, hi), 1))
    return IndicatorNode(column=random.choice(_CROSS_INDICATORS))


def _random_comparison() -> ComparisonNode:
    """Generate a random comparison node."""
    if random.random() < 0.5:
        # Indicator vs indicator
        left = IndicatorNode(column=random.choice(_CROSS_INDICATORS))
        right = IndicatorNode(column=random.choice(_CROSS_INDICATORS))
        op = random.choice(_COMPARISON_OPS)
    else:
        # Bounded indicator vs constant
        ind = random.choice(_BOUNDED_INDICATORS)
        lo, hi = _BOUNDED_RANGES[ind]
        left = IndicatorNode(column=ind)
        right = ConstantNode(value=round(random.uniform(lo, hi), 1))
        op = random.choice([">", "<", ">=", "<="])
    return ComparisonNode(operator=op, left=left, right=right)


def _random_tree(max_depth: int = MAX_TREE_DEPTH, current_depth: int = 0) -> Node:
    """Recursively generate a random expression tree."""
    if current_depth >= max_depth or (current_depth > 1 and random.random() < 0.4):
        return _random_comparison()

    left = _random_tree(max_depth, current_depth + 1)
    right = _random_tree(max_depth, current_depth + 1)
    return BooleanNode(
        operator=random.choice(_BOOLEAN_OPS),
        left=left, right=right,
    )


def generate_random_strategy() -> Strategy:
    """Generate a completely random strategy."""
    depth = random.randint(2, MAX_TREE_DEPTH)
    buy_rule = _random_tree(depth)
    sell_rule = _random_tree(depth)
    risk = RiskParams(
        sl_atr_mult=round(random.uniform(1.0, 2.0), 1),
        rr_ratio=round(random.uniform(2.0, 4.0), 1),
        risk_pct=round(random.uniform(0.005, 0.015), 3),
        cooldown=random.randint(2, 5),
    )
    return Strategy(origin="random", buy_rule=buy_rule,
                    sell_rule=sell_rule, risk_params=risk)


def generate_batch(
    ga_offspring: List[Strategy] = None,
    batch_size: int = BATCH_SIZE,
) -> List[Strategy]:
    """
    Generate a batch of strategies with a mix of:
    - Random strategies
    - Template-based strategies
    - GA offspring (if provided)
    """
    strategies = []
    n_random = int(batch_size * RANDOM_RATIO)
    n_template = int(batch_size * TEMPLATE_RATIO)
    n_ga = batch_size - n_random - n_template

    # Random
    for _ in range(n_random):
        strategies.append(generate_random_strategy())

    # Template
    for _ in range(n_template):
        strategies.append(generate_from_template())

    # GA offspring
    if ga_offspring:
        strategies.extend(ga_offspring[:n_ga])
    else:
        # Fill with more templates if no GA offspring
        for _ in range(n_ga):
            strategies.append(generate_from_template())

    log.info(f"Generated batch: {n_random} random, {n_template} template, "
             f"{len(strategies) - n_random - n_template} GA/template")
    return strategies

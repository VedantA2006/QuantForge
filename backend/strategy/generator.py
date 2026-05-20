# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Strategy Generator (Category-Based + Tree)   ║
# ╚══════════════════════════════════════════════════════════════╝

import random
import logging
from typing import List, Optional, Tuple, Dict

from backend.strategy.tree import (
    Strategy, RiskParams, Node, BooleanNode, ComparisonNode,
    IndicatorNode, ConstantNode, ArithmeticNode,
)
from backend.config import BATCH_SIZE, MAX_TREE_DEPTH, RANDOM_RATIO, TEMPLATE_RATIO, GA_RATIO

log = logging.getLogger("quantforge.strategy.generator")

# ── Timeframe prefixes with sampling weights ──────────────────────────────
_TF_WEIGHTS = {"tf_1h_": 0.50, "tf_4h_": 0.30, "tf_1d_": 0.20}

def _pick_tf() -> str:
    return random.choices(list(_TF_WEIGHTS.keys()), weights=list(_TF_WEIGHTS.values()), k=1)[0]

# ── Indicator lists ───────────────────────────────────────────────────────
_BOUNDED_INDICATORS = [
    "rsi_7", "rsi_14", "rsi_21", "stoch_k", "stoch_d", "adx_14",
    "bb_width", "roc_10", "willr_14", "cci_20", "mfi_14", "cmf_20",
]
_BOUNDED_RANGES = {
    "rsi_7": (20, 80), "rsi_14": (20, 80), "rsi_21": (20, 80),
    "stoch_k": (15, 85), "stoch_d": (15, 85), "adx_14": (15, 50),
    "bb_width": (0.02, 0.15), "roc_10": (-15.0, 15.0),
    "willr_14": (-90, -10), "cci_20": (-200, 200),
    "mfi_14": (20, 80), "cmf_20": (-0.3, 0.3),
}
_CROSS_INDICATORS = [
    "ema_8", "ema_13", "ema_21", "ema_34", "ema_55", "ema_89", "ema_200",
    "sma_20", "sma_50", "sma_200", "bb_upper", "bb_middle", "bb_lower",
    "close", "open", "high", "low", "macd_line", "macd_signal",
    "supertrend_10_3", "high_10", "high_20", "low_10", "low_20",
]
_COMPARISON_OPS = [">", "<", ">=", "<=", "crossover", "crossunder"]
_BOOLEAN_OPS = ["AND", "OR"]

# ── Category-based condition generation ───────────────────────────────────
CATEGORY_WEIGHTS: Dict[str, float] = {
    "ema_crossover": 10, "rsi_thresh": 10, "macd_thresh": 8, "stoch_thresh": 7,
    "adx_thresh": 9, "bb_crossover": 7, "momentum_roc": 6, "candle_struct": 5,
    "volume_profile": 6, "price_struct": 5, "regime_filter": 8,
    "supertrend": 7, "vwap_dev": 5, "cmf": 5, "williams_r": 5,
    "ema_vs_sma": 6, "sma_crossover": 6, "price_vs_sma": 6,
    "rsi_range": 5, "rsi_momentum": 5, "stoch_cross": 6, "mfi_thresh": 5,
    "cci_thresh": 5, "bb_squeeze": 6, "breakout_nh": 6, "volume_spike": 5,
    "multi_tf_confirm": 8, "cross_tf_rsi": 7, "mean_reversion": 5,
    "consec_candles": 4, "wick_bias": 4, "obv_momentum": 5, "willr_extreme": 5,
}

_CATEGORY_COLUMN_KEYWORDS = {
    "ema_crossover":    ["ema_8", "ema_13", "ema_21", "ema_34", "ema_55", "ema_89"],
    "sma_crossover":    ["sma_20", "sma_50"],
    "ema_vs_sma":       ["ema_", "sma_"],
    "price_vs_sma":     ["sma_20", "sma_50", "sma_200"],
    "rsi_thresh":       ["rsi_14"],
    "rsi_range":        ["rsi_7", "rsi_21"],
    "rsi_momentum":     ["rsi_14", "rsi_7"],
    "macd_thresh":      ["macd_line", "macd_signal"],
    "stoch_thresh":     ["stoch_k"],
    "stoch_cross":      ["stoch_k", "stoch_d"],
    "adx_thresh":       ["adx_14"],
    "bb_crossover":     ["bb_upper", "bb_lower"],
    "bb_squeeze":       ["bb_width"],
    "momentum_roc":     ["roc_10"],
    "candle_struct":    ["is_engulfing"],
    "volume_profile":   ["volume_ratio"],
    "volume_spike":     ["volume_ratio"],
    "price_struct":     ["high_10", "high_20", "low_10", "low_20"],
    "breakout_nh":      ["high_10", "high_20"],
    "regime_filter":    ["regime"],
    "supertrend":       ["supertrend_10_3"],
    "vwap_dev":         ["vwap_dev"],
    "cmf":              ["cmf_20"],
    "williams_r":       ["willr_14"],
    "willr_extreme":    ["willr_14"],
    "mfi_thresh":       ["mfi_14"],
    "cci_thresh":       ["cci_20"],
    "multi_tf_confirm": ["tf_4h_", "tf_1d_"],
    "cross_tf_rsi":     ["tf_4h_rsi"],
    "mean_reversion":   ["bb_lower", "bb_upper", "rsi_14"],
    "consec_candles":   ["consec_bullish", "consec_bearish"],
    "wick_bias":        ["wick"],
    "obv_momentum":     ["obv_slope"],
}

def update_category_weights(db):
    """Update category weights from MongoDB stats. Categories in top strategies get boosted."""
    try:
        if hasattr(db, '_fallback') and db._fallback:
            return
            
        stats = list(db.db.category_stats.find({}, {"_id": 0}))
        
        # Calculate base weights
        base_weights = {}
        for s in stats:
            cat = s.get("category", "")
            if cat not in CATEGORY_WEIGHTS: continue
            
            top20 = s.get("appearances_top20", 0)
            total = max(s.get("appearances_total", 1), 1)
            avg_score = s.get("avg_score", 0.0)
            
            w = (top20 / total) * avg_score
            base_weights[cat] = max(0.001, w)
            
        # Apply co-occurrence boost
        cooc = list(db.db.category_cooccurrences.find({}, {"_id": 0}))
        cooc_map = {c["pair"]: c["count"] for c in cooc}
        
        final_weights = {}
        for cat in base_weights:
            boost_factor = 1.0
            for partner in base_weights:
                if cat == partner: continue
                pair_name = "-".join(sorted([cat, partner]))
                if cooc_map.get(pair_name, 0) > 3:  # Only significant partners
                    boost_factor += 0.10
            final_weights[cat] = base_weights[cat] * boost_factor
            
        # Normalize to sum to 1
        total_w = sum(final_weights.values())
        if total_w > 0:
            for cat in CATEGORY_WEIGHTS:
                CATEGORY_WEIGHTS[cat] = final_weights.get(cat, 0.001) / total_w
                
    except Exception:
        pass


def _generate_category_condition(category: str, direction: str = "buy") -> ComparisonNode:
    """Generate a single condition node from a category. direction = 'buy' or 'sell'."""
    tf = _pick_tf()
    is_buy = direction == "buy"

    if category == "ema_crossover":
        fast = random.choice(["ema_8", "ema_13", "ema_21"])
        slow = random.choice(["ema_34", "ema_55", "ema_89"])
        op = "crossover" if is_buy else "crossunder"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}{fast}"), right=IndicatorNode(column=f"{tf}{slow}"))

    elif category == "rsi_thresh":
        thresh = random.randint(25, 35) if is_buy else random.randint(65, 75)
        op = "<" if is_buy else ">"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}rsi_14"), right=ConstantNode(value=thresh))

    elif category == "macd_thresh":
        op = "crossover" if is_buy else "crossunder"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}macd_line"), right=IndicatorNode(column=f"{tf}macd_signal"))

    elif category == "stoch_thresh":
        thresh = random.randint(15, 25) if is_buy else random.randint(75, 85)
        op = "<" if is_buy else ">"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}stoch_k"), right=ConstantNode(value=thresh))

    elif category == "adx_thresh":
        thresh = random.randint(20, 35)
        return ComparisonNode(operator=">", left=IndicatorNode(column=f"{tf}adx_14"), right=ConstantNode(value=thresh))

    elif category == "bb_crossover":
        band = "bb_lower" if is_buy else "bb_upper"
        op = "<" if is_buy else ">"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}close" if tf == "tf_1h_" else f"{tf}ema_21"), right=IndicatorNode(column=f"{tf}{band}"))

    elif category == "momentum_roc":
        thresh = round(random.uniform(1, 5), 1)
        op = ">" if is_buy else "<"
        val = thresh if is_buy else -thresh
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}roc_10"), right=ConstantNode(value=val))

    elif category == "candle_struct":
        pattern = "is_engulfing_bull" if is_buy else "is_engulfing_bear"
        return ComparisonNode(operator=">", left=IndicatorNode(column=f"{tf}{pattern}"), right=ConstantNode(value=0.5))

    elif category == "volume_profile":
        thresh = round(random.uniform(1.2, 2.0), 1)
        return ComparisonNode(operator=">", left=IndicatorNode(column=f"{tf}volume_ratio"), right=ConstantNode(value=thresh))

    elif category == "price_struct":
        n = random.choice([10, 20])
        col = f"high_{n}" if is_buy else f"low_{n}"
        op = ">=" if is_buy else "<="
        return ComparisonNode(operator=op, left=IndicatorNode(column="close"), right=IndicatorNode(column=f"{tf}{col}"))

    elif category == "regime_filter":
        val = 0.5
        op = ">" if is_buy else "<"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}regime"), right=ConstantNode(value=val))

    elif category == "supertrend":
        op = ">" if is_buy else "<"
        return ComparisonNode(operator=op, left=IndicatorNode(column="close"), right=IndicatorNode(column=f"{tf}supertrend_10_3"))

    elif category == "vwap_dev":
        thresh = round(random.uniform(-2, -0.5), 1) if is_buy else round(random.uniform(0.5, 2), 1)
        op = "<" if is_buy else ">"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}vwap_dev"), right=ConstantNode(value=thresh))

    elif category == "cmf":
        thresh = round(random.uniform(0.05, 0.2), 2)
        op = ">" if is_buy else "<"
        val = thresh if is_buy else -thresh
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}cmf_20"), right=ConstantNode(value=val))

    elif category == "williams_r":
        thresh = random.randint(-90, -70) if is_buy else random.randint(-30, -10)
        op = "<" if is_buy else ">"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}willr_14"), right=ConstantNode(value=thresh))

    elif category == "mfi_thresh":
        thresh = random.randint(20, 35) if is_buy else random.randint(65, 80)
        op = "<" if is_buy else ">"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}mfi_14"), right=ConstantNode(value=thresh))

    elif category == "cci_thresh":
        thresh = random.randint(-150, -80) if is_buy else random.randint(80, 150)
        op = "<" if is_buy else ">"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}cci_20"), right=ConstantNode(value=thresh))

    elif category == "bb_squeeze":
        thresh = round(random.uniform(0.02, 0.06), 3)
        return ComparisonNode(operator="<", left=IndicatorNode(column=f"{tf}bb_width"), right=ConstantNode(value=thresh))

    elif category == "multi_tf_confirm":
        tf2 = "tf_4h_" if tf == "tf_1h_" else "tf_1h_"
        op = ">" if is_buy else "<"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf2}ema_21"), right=IndicatorNode(column=f"{tf2}ema_55"))

    elif category == "cross_tf_rsi":
        tf2 = "tf_4h_"
        thresh = random.randint(40, 55) if is_buy else random.randint(55, 70)
        op = ">" if is_buy else "<"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf2}rsi_14"), right=ConstantNode(value=thresh))

    elif category == "obv_momentum":
        op = ">" if is_buy else "<"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}obv_slope_5"), right=ConstantNode(value=0))

    elif category == "consec_candles":
        col = "consec_bullish_2" if is_buy else "consec_bearish_2"
        op = ">=" if is_buy else ">="
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}{col}"), right=ConstantNode(value=0.5))

    # fallback — simple EMA comparison
    ema_col = random.choice(["ema_8", "ema_13", "ema_21"])
    sma_col = random.choice(["sma_20", "sma_50"])
    op = ">" if is_buy else "<"
    return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}{ema_col}"), right=IndicatorNode(column=f"{tf}{sma_col}"))
def _build_compound_rule(direction: str, n_conditions: Optional[int] = None) -> Node:
    """Build a compound boolean rule from random categories with conflict prevention."""
    if n_conditions is None:
        n_conditions = random.randint(2, 4)  # Cap at 4 to reduce complexity/overfitting

    cats = list(CATEGORY_WEIGHTS.keys())
    weights = list(CATEGORY_WEIGHTS.values())
    chosen = []

    conflict_map = {
        "rsi_thresh": {"rsi_range", "rsi_momentum"},
        "rsi_range": {"rsi_thresh", "rsi_momentum"},
        "rsi_momentum": {"rsi_thresh", "rsi_range"},
        "stoch_thresh": {"stoch_cross"},
        "stoch_cross": {"stoch_thresh"},
        "ema_crossover": {"sma_crossover", "ema_vs_sma"},
        "sma_crossover": {"ema_crossover", "ema_vs_sma"},
        "ema_vs_sma": {"ema_crossover", "sma_crossover"},
        "willr_extreme": {"williams_r"},
        "williams_r": {"willr_extreme"},
        "volume_profile": {"volume_spike"},
        "volume_spike": {"volume_profile"},
    }

    while len(chosen) < n_conditions and cats and sum(weights) > 0:
        c = random.choices(cats, weights=weights, k=1)[0]
        chosen.append(c)

        to_remove = {c}
        if c in conflict_map:
            to_remove.update(conflict_map[c])

        for item in to_remove:
            if item in cats:
                idx = cats.index(item)
                cats.pop(idx)
                weights.pop(idx)

    nodes = [_generate_category_condition(cat, direction) for cat in chosen]

    if not nodes:
        tf = _pick_tf()
        op = ">" if direction == "buy" else "<"
        return ComparisonNode(operator=op, left=IndicatorNode(column=f"{tf}close"), right=IndicatorNode(column=f"{tf}ema_21"))

    # Join with AND (biased 2:1 over OR)
    tree = nodes[0]
    for node in nodes[1:]:
        op = "AND" if random.random() < 0.67 else "OR"
        tree = BooleanNode(operator=op, left=tree, right=node)
    return tree
def _rand_risk():
    trail = round(random.choice([0.0, 0.0, 0.0, random.uniform(1.5, 3.0)]), 1)
    tp1 = round(random.choice([0.0, 0.0, 0.0, random.uniform(0.3, 0.6)]), 2)
    return RiskParams(
        sl_atr_mult=round(random.uniform(1.0, 2.0), 1),
        rr_ratio=round(random.uniform(2.0, 4.0), 1),
        risk_pct=round(random.uniform(0.005, 0.015), 3),
        cooldown=random.randint(4, 8),
        trail_mult=trail,
        tp1_ratio=tp1,
    )


# ── Legacy tree-based generation (kept for GA compatibility) ──────────────

def _random_leaf(allow_constant: bool = True) -> Node:
    tf = _pick_tf()
    if allow_constant and random.random() < 0.3:
        ind = random.choice(_BOUNDED_INDICATORS)
        lo, hi = _BOUNDED_RANGES[ind]
        return ConstantNode(value=round(random.uniform(lo, hi), 1))
    col = random.choice(_CROSS_INDICATORS)
    return IndicatorNode(column=f"{tf}{col}")


def _random_comparison() -> ComparisonNode:
    tf = _pick_tf()
    if random.random() < 0.5:
        left = IndicatorNode(column=f"{tf}{random.choice(_CROSS_INDICATORS)}")
        right = IndicatorNode(column=f"{tf}{random.choice(_CROSS_INDICATORS)}")
        op = random.choice(_COMPARISON_OPS)
    else:
        ind = random.choice(_BOUNDED_INDICATORS)
        lo, hi = _BOUNDED_RANGES[ind]
        left = IndicatorNode(column=f"{tf}{ind}")
        right = ConstantNode(value=round(random.uniform(lo, hi), 1))
        op = random.choice([">", "<", ">=", "<="])
    return ComparisonNode(operator=op, left=left, right=right)


def _random_tree(max_depth: int = MAX_TREE_DEPTH, current_depth: int = 0) -> Node:
    if current_depth >= max_depth or (current_depth > 1 and random.random() < 0.4):
        return _random_comparison()
    left = _random_tree(max_depth, current_depth + 1)
    right = _random_tree(max_depth, current_depth + 1)
    return BooleanNode(operator="AND", left=left, right=right)


def generate_random_strategy() -> Strategy:
    """Generate a strategy using the expanded category-based generator."""
    buy_rule = _build_compound_rule("buy")
    sell_rule = _build_compound_rule("sell")
    return Strategy(origin="random", buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def generate_from_params(params: list) -> Strategy:
    """Generate a strategy from a Bayesian-suggested parameter vector."""
    risk = RiskParams(
        sl_atr_mult=round(params[0], 1),
        rr_ratio=round(params[1], 1),
        risk_pct=round(params[2], 3),
        cooldown=int(round(params[3])),
        trail_mult=round(params[4], 1),
        tp1_ratio=round(params[5], 2),
    )
    buy_rule = _build_compound_rule("buy")
    sell_rule = _build_compound_rule("sell")
    return Strategy(origin="bayesian", buy_rule=buy_rule, sell_rule=sell_rule, risk_params=risk)


# ── Batch generation ──────────────────────────────────────────────────────

from backend.strategy.templates import generate_from_template

def generate_batch(
    ga_offspring: Optional[List[Strategy]] = None,
    batch_size: int = BATCH_SIZE,
    bayesian_suggestions: Optional[List[list]] = None,
    rl_agent = None,
) -> List[Tuple[Strategy, Optional[dict]]]:
    """Generate a batch with mix of random, template, GA, Bayesian, and RL strategies. Returns list of (Strategy, episode_data)."""
    strategies: List[Tuple[Strategy, Optional[dict]]] = []

    # Bayesian suggestions first
    if bayesian_suggestions:
        for params in bayesian_suggestions:
            try:
                strategies.append((generate_from_params(params), None))
            except Exception:
                pass

    # GA offspring
    if ga_offspring:
        for s in ga_offspring[:int(batch_size * GA_RATIO)]:
            strategies.append((s, None))
            
    # RL agent
    from backend.config import RL_ENABLED, RL_BATCH_RATIO
    n_rl = 0
    if rl_agent and RL_ENABLED:
        n_rl = int(batch_size * RL_BATCH_RATIO)
        for _ in range(n_rl):
            strat, episode = rl_agent.generate_strategy()
            if strat:
                strategies.append((strat, episode))

    remaining = batch_size - len(strategies)
    if remaining <= 0:
        log.info(f"Generated batch: {len(strategies)} total")
        return strategies[:batch_size]

    n_random = max(1, int(remaining * (RANDOM_RATIO / (RANDOM_RATIO + TEMPLATE_RATIO))))
    n_template = remaining - n_random

    for _ in range(n_random):
        strategies.append((generate_random_strategy(), None))
    for _ in range(n_template):
        strategies.append((generate_from_template(), None))

    log.info(f"Generated batch: {n_random} random, {n_template} template, {len(ga_offspring or [])} GA, {len(bayesian_suggestions or [])} bayesian, {n_rl} RL")
    return strategies[:batch_size]


def record_category_stats(db, strategy: Strategy, rank_score: float, is_top20: bool = False):
    """Record which categories appear in a strategy's conditions for weight adaptation."""
    try:
        if hasattr(db, '_fallback') and db._fallback:
            return
        nodes = []
        if strategy.buy_rule:
            nodes.extend(strategy.buy_rule.collect_nodes())
        if strategy.sell_rule:
            nodes.extend(strategy.sell_rule.collect_nodes())

        seen = set()
        for node in nodes:
            if isinstance(node, IndicatorNode):
                col = node.column
                for cat, keywords in _CATEGORY_COLUMN_KEYWORDS.items():
                    if any(kw in col for kw in keywords):
                        seen.add(cat)

        for cat in seen:
            # Get current avg
            doc = db.db.category_stats.find_one({"category": cat})
            if not doc:
                doc = {"appearances_total": 0, "avg_score": 0.0}
            
            old_avg = doc.get("avg_score", 0.0)
            old_n = doc.get("appearances_total", 0)
            
            new_avg = (old_avg * old_n + rank_score) / (old_n + 1)
            
            update = {
                "$inc": {"appearances_total": 1},
                "$set": {"avg_score": new_avg}
            }
            if is_top20:
                update["$inc"]["appearances_top20"] = 1
            db.db.category_stats.update_one(
                {"category": cat}, update, upsert=True
            )
            
        # Co-occurrences in top 20
        if is_top20:
            seen_list = list(seen)
            for i in range(len(seen_list)):
                for j in range(i + 1, len(seen_list)):
                    pair = "-".join(sorted([seen_list[i], seen_list[j]]))
                    db.db.category_cooccurrences.update_one(
                        {"pair": pair}, {"$inc": {"count": 1}}, upsert=True
                    )
    except Exception:
        pass

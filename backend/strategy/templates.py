# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Strategy Templates                            ║
# ╚══════════════════════════════════════════════════════════════╝

import random
from backend.strategy.tree import (
    Strategy, RiskParams, BooleanNode, ComparisonNode,
    IndicatorNode, ConstantNode,
)
from backend.strategy.generator import _rand_risk

_TF_WEIGHTS = {"tf_1h_": 0.50, "tf_4h_": 0.30, "tf_1d_": 0.20}

def _pick_tf() -> str:
    return random.choices(list(_TF_WEIGHTS.keys()), weights=list(_TF_WEIGHTS.values()), k=1)[0]

def _ind(col: str, tf: str = None) -> IndicatorNode:
    """Create an IndicatorNode with a timeframe prefix. Price columns are bare."""
    bare_cols = {"open", "high", "low", "close", "volume"}
    if col in bare_cols:
        return IndicatorNode(column=col)
    prefix = tf if tf else _pick_tf()
    return IndicatorNode(column=f"{prefix}{col}")

def ema_crossover_template():
    tf = _pick_tf()
    fast = random.choice([8, 13, 21])
    slow = random.choice([34, 55, 89])
    trend = random.choice([89, 200])
    if fast >= slow: fast, slow = min(fast, slow), max(fast, slow)
    if slow >= trend: trend = slow * 2

    buy_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator=random.choice(["crossover", ">"]),
            left=_ind(f"ema_{fast}", tf),
            right=_ind(f"ema_{slow}", tf)),
        right=ComparisonNode(operator=">",
            left=_ind("close"),
            right=_ind(f"ema_{trend}", tf)))
    sell_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator=random.choice(["crossunder", "<"]),
            left=_ind(f"ema_{fast}", tf),
            right=_ind(f"ema_{slow}", tf)),
        right=ComparisonNode(operator="<",
            left=_ind("close"),
            right=_ind(f"ema_{trend}", tf)))
    return Strategy(name=f"ema_cross_{fast}_{slow}", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def rsi_mean_reversion_template():
    tf = _pick_tf()
    p = random.choice([7, 14, 21])
    bt = random.randint(25, 40)
    st = random.randint(60, 80)
    ag = round(random.uniform(0.001, 0.005), 4)
    buy_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="<",
            left=_ind(f"rsi_{p}", tf), right=ConstantNode(value=bt)),
        right=ComparisonNode(operator=">",
            left=_ind("atr_pct", tf), right=ConstantNode(value=ag)))
    sell_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator=">",
            left=_ind(f"rsi_{p}", tf), right=ConstantNode(value=st)),
        right=ComparisonNode(operator=">",
            left=_ind("atr_pct", tf), right=ConstantNode(value=ag)))
    return Strategy(name=f"rsi_mr_{p}", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def bollinger_breakout_template():
    tf = _pick_tf()
    vt = round(random.uniform(1.2, 2.0), 1)
    buy_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator=">",
            left=_ind("close"), right=_ind("bb_upper", tf)),
        right=ComparisonNode(operator=">",
            left=_ind("volume_ratio", tf), right=ConstantNode(value=vt)))
    sell_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="<",
            left=_ind("close"), right=_ind("bb_lower", tf)),
        right=ComparisonNode(operator=">",
            left=_ind("volume_ratio", tf), right=ConstantNode(value=vt)))
    return Strategy(name="bb_breakout", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def macd_momentum_template():
    tf = _pick_tf()
    at = random.randint(20, 35)
    buy_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="crossover",
            left=_ind("macd_line", tf), right=_ind("macd_signal", tf)),
        right=ComparisonNode(operator=">",
            left=_ind("adx_14", tf), right=ConstantNode(value=at)))
    sell_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="crossunder",
            left=_ind("macd_line", tf), right=_ind("macd_signal", tf)),
        right=ComparisonNode(operator=">",
            left=_ind("adx_14", tf), right=ConstantNode(value=at)))
    return Strategy(name=f"macd_adx_{at}", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def stochastic_template():
    tf = _pick_tf()
    ep = random.choice([34, 55, 89])
    kb = random.randint(15, 30)
    ks = random.randint(70, 85)
    buy_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="crossover",
            left=_ind("stoch_k", tf), right=ConstantNode(value=kb)),
        right=ComparisonNode(operator=">",
            left=_ind("close"), right=_ind(f"ema_{ep}", tf)))
    sell_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="crossunder",
            left=_ind("stoch_k", tf), right=ConstantNode(value=ks)),
        right=ComparisonNode(operator="<",
            left=_ind("close"), right=_ind(f"ema_{ep}", tf)))
    return Strategy(name=f"stoch_{ep}", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def triple_ema_template():
    tf = _pick_tf()
    f = random.choice([8, 13])
    m = random.choice([21, 34])
    s = random.choice([55, 89])
    rb = random.randint(50, 65)
    rs = random.randint(30, 45)
    buy_rule = BooleanNode(operator="AND",
        left=BooleanNode(operator="AND",
            left=ComparisonNode(operator=">",
                left=_ind(f"ema_{f}", tf), right=_ind(f"ema_{m}", tf)),
            right=ComparisonNode(operator=">",
                left=_ind(f"ema_{m}", tf), right=_ind(f"ema_{s}", tf))),
        right=BooleanNode(operator="AND",
            left=ComparisonNode(operator=">",
                left=_ind("rsi_14", tf), right=ConstantNode(value=rb)),
            right=ComparisonNode(operator=">",
                left=_ind("volume_ratio", tf), right=ConstantNode(value=1.0))))
    sell_rule = BooleanNode(operator="AND",
        left=BooleanNode(operator="AND",
            left=ComparisonNode(operator="<",
                left=_ind(f"ema_{f}", tf), right=_ind(f"ema_{m}", tf)),
            right=ComparisonNode(operator="<",
                left=_ind(f"ema_{m}", tf), right=_ind(f"ema_{s}", tf))),
        right=BooleanNode(operator="AND",
            left=ComparisonNode(operator="<",
                left=_ind("rsi_14", tf), right=ConstantNode(value=rs)),
            right=ComparisonNode(operator=">",
                left=_ind("volume_ratio", tf), right=ConstantNode(value=1.0))))
    return Strategy(name=f"triple_ema_{f}_{m}_{s}", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def ema_adx_trend_template():
    tf = _pick_tf()
    fast = random.choice([8, 13, 21])
    slow = random.choice([34, 55])
    adx_thresh = random.randint(25, 35)
    rsi_thresh = random.randint(52, 62)

    buy_rule = BooleanNode(operator="AND",
        left=BooleanNode(operator="AND",
            left=ComparisonNode(operator=">",
                left=_ind(f"ema_{fast}", tf),
                right=_ind(f"ema_{slow}", tf)),
            right=ComparisonNode(operator=">",
                left=_ind("adx_14", tf),
                right=ConstantNode(value=adx_thresh))),
        right=ComparisonNode(operator=">",
            left=_ind("rsi_14", tf),
            right=ConstantNode(value=rsi_thresh)))

    sell_rule = BooleanNode(operator="AND",
        left=BooleanNode(operator="AND",
            left=ComparisonNode(operator="<",
                left=_ind(f"ema_{fast}", tf),
                right=_ind(f"ema_{slow}", tf)),
            right=ComparisonNode(operator=">",
                left=_ind("adx_14", tf),
                right=ConstantNode(value=adx_thresh))),
        right=ComparisonNode(operator="<",
            left=_ind("rsi_14", tf),
            right=ConstantNode(value=100 - rsi_thresh)))

    return Strategy(name=f"ema_adx_{fast}_{slow}", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def macd_ema_trend_template():
    tf = _pick_tf()
    trend_ema = random.choice([89, 200])
    adx_thresh = random.randint(20, 30)

    buy_rule = BooleanNode(operator="AND",
        left=BooleanNode(operator="AND",
            left=ComparisonNode(operator="crossover",
                left=_ind("macd_line", tf),
                right=_ind("macd_signal", tf)),
            right=ComparisonNode(operator=">",
                left=_ind("close"),
                right=_ind(f"ema_{trend_ema}", tf))),
        right=ComparisonNode(operator=">",
            left=_ind("adx_14", tf),
            right=ConstantNode(value=adx_thresh)))

    sell_rule = BooleanNode(operator="AND",
        left=BooleanNode(operator="AND",
            left=ComparisonNode(operator="crossunder",
                left=_ind("macd_line", tf),
                right=_ind("macd_signal", tf)),
            right=ComparisonNode(operator="<",
                left=_ind("close"),
                right=_ind(f"ema_{trend_ema}", tf))),
        right=ComparisonNode(operator=">",
            left=_ind("adx_14", tf),
            right=ConstantNode(value=adx_thresh)))

    return Strategy(name=f"macd_ema_{trend_ema}", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def bb_rsi_reversion_template():
    tf = _pick_tf()
    rsi_ob = random.randint(65, 75)
    rsi_os = random.randint(25, 35)
    vol_thresh = round(random.uniform(0.8, 1.2), 1)

    buy_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="<",
            left=_ind("close"),
            right=_ind("bb_lower", tf)),
        right=ComparisonNode(operator="<",
            left=_ind("rsi_14", tf),
            right=ConstantNode(value=rsi_os)))

    sell_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator=">",
            left=_ind("close"),
            right=_ind("bb_upper", tf)),
        right=ComparisonNode(operator=">",
            left=_ind("rsi_14", tf),
            right=ConstantNode(value=rsi_ob)))

    return Strategy(name="bb_rsi_reversion", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def regime_trend_template():
    tf = _pick_tf()
    strat = random.choice([ema_crossover_template, macd_momentum_template, ema_adx_trend_template, macd_ema_trend_template])()
    
    strat.buy_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator=">", left=_ind("regime", tf), right=ConstantNode(value=0.5)),
        right=strat.buy_rule)
        
    strat.sell_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator=">", left=_ind("regime", tf), right=ConstantNode(value=0.5)),
        right=strat.sell_rule)
        
    strat.name = "regime_" + strat.name
    return strat


def regime_reversion_template():
    tf = _pick_tf()
    strat = random.choice([rsi_mean_reversion_template, bb_rsi_reversion_template, stochastic_template])()
    
    strat.buy_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="<", left=_ind("regime", tf), right=ConstantNode(value=0.5)),
        right=strat.buy_rule)
        
    strat.sell_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="<", left=_ind("regime", tf), right=ConstantNode(value=0.5)),
        right=strat.sell_rule)
        
    strat.name = "regime_" + strat.name
    return strat


TEMPLATES = [
    ema_crossover_template, rsi_mean_reversion_template,
    bollinger_breakout_template, macd_momentum_template,
    stochastic_template, triple_ema_template,
    ema_adx_trend_template, macd_ema_trend_template,
    bb_rsi_reversion_template,
    regime_trend_template, regime_reversion_template,
]


def generate_from_template():
    return random.choice(TEMPLATES)()

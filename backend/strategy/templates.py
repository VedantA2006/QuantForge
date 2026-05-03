# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Strategy Templates                            ║
# ╚══════════════════════════════════════════════════════════════╝

import random
from backend.strategy.tree import (
    Strategy, RiskParams, BooleanNode, ComparisonNode,
    IndicatorNode, ConstantNode,
)


def _rand_risk():
    return RiskParams(
        sl_atr_mult=round(random.uniform(1.0, 3.5), 1),
        rr_ratio=round(random.uniform(1.5, 4.0), 1),
        risk_pct=round(random.uniform(0.005, 0.02), 3),
        cooldown=random.randint(1, 5),
    )


def ema_crossover_template():
    fast = random.choice([8, 13, 21])
    slow = random.choice([34, 55, 89])
    trend = random.choice([89, 200])
    if fast >= slow: fast, slow = min(fast, slow), max(fast, slow)
    if slow >= trend: trend = slow * 2

    buy_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator=random.choice(["crossover", ">"]),
            left=IndicatorNode(column=f"ema_{fast}"),
            right=IndicatorNode(column=f"ema_{slow}")),
        right=ComparisonNode(operator=">",
            left=IndicatorNode(column="close"),
            right=IndicatorNode(column=f"ema_{trend}")))
    sell_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator=random.choice(["crossunder", "<"]),
            left=IndicatorNode(column=f"ema_{fast}"),
            right=IndicatorNode(column=f"ema_{slow}")),
        right=ComparisonNode(operator="<",
            left=IndicatorNode(column="close"),
            right=IndicatorNode(column=f"ema_{trend}")))
    return Strategy(name=f"ema_cross_{fast}_{slow}", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def rsi_mean_reversion_template():
    p = random.choice([7, 14, 21])
    bt = random.randint(25, 40)
    st = random.randint(60, 80)
    ag = round(random.uniform(0.001, 0.005), 4)
    buy_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="<",
            left=IndicatorNode(column=f"rsi_{p}"), right=ConstantNode(value=bt)),
        right=ComparisonNode(operator=">",
            left=IndicatorNode(column="atr_pct"), right=ConstantNode(value=ag)))
    sell_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator=">",
            left=IndicatorNode(column=f"rsi_{p}"), right=ConstantNode(value=st)),
        right=ComparisonNode(operator=">",
            left=IndicatorNode(column="atr_pct"), right=ConstantNode(value=ag)))
    return Strategy(name=f"rsi_mr_{p}", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def bollinger_breakout_template():
    vt = round(random.uniform(1.2, 2.0), 1)
    buy_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator=">",
            left=IndicatorNode(column="close"), right=IndicatorNode(column="bb_upper")),
        right=ComparisonNode(operator=">",
            left=IndicatorNode(column="volume_ratio"), right=ConstantNode(value=vt)))
    sell_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="<",
            left=IndicatorNode(column="close"), right=IndicatorNode(column="bb_lower")),
        right=ComparisonNode(operator=">",
            left=IndicatorNode(column="volume_ratio"), right=ConstantNode(value=vt)))
    return Strategy(name="bb_breakout", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def macd_momentum_template():
    at = random.randint(20, 35)
    buy_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="crossover",
            left=IndicatorNode(column="macd_line"), right=IndicatorNode(column="macd_signal")),
        right=ComparisonNode(operator=">",
            left=IndicatorNode(column="adx_14"), right=ConstantNode(value=at)))
    sell_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="crossunder",
            left=IndicatorNode(column="macd_line"), right=IndicatorNode(column="macd_signal")),
        right=ComparisonNode(operator=">",
            left=IndicatorNode(column="adx_14"), right=ConstantNode(value=at)))
    return Strategy(name=f"macd_adx_{at}", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def stochastic_template():
    ep = random.choice([34, 55, 89])
    kb = random.randint(15, 30)
    ks = random.randint(70, 85)
    buy_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="crossover",
            left=IndicatorNode(column="stoch_k"), right=ConstantNode(value=kb)),
        right=ComparisonNode(operator=">",
            left=IndicatorNode(column="close"), right=IndicatorNode(column=f"ema_{ep}")))
    sell_rule = BooleanNode(operator="AND",
        left=ComparisonNode(operator="crossunder",
            left=IndicatorNode(column="stoch_k"), right=ConstantNode(value=ks)),
        right=ComparisonNode(operator="<",
            left=IndicatorNode(column="close"), right=IndicatorNode(column=f"ema_{ep}")))
    return Strategy(name=f"stoch_{ep}", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


def triple_ema_template():
    f = random.choice([8, 13])
    m = random.choice([21, 34])
    s = random.choice([55, 89])
    rb = random.randint(50, 65)
    rs = random.randint(30, 45)
    buy_rule = BooleanNode(operator="AND",
        left=BooleanNode(operator="AND",
            left=ComparisonNode(operator=">",
                left=IndicatorNode(column=f"ema_{f}"), right=IndicatorNode(column=f"ema_{m}")),
            right=ComparisonNode(operator=">",
                left=IndicatorNode(column=f"ema_{m}"), right=IndicatorNode(column=f"ema_{s}"))),
        right=BooleanNode(operator="AND",
            left=ComparisonNode(operator=">",
                left=IndicatorNode(column="rsi_14"), right=ConstantNode(value=rb)),
            right=ComparisonNode(operator=">",
                left=IndicatorNode(column="volume_ratio"), right=ConstantNode(value=1.0))))
    sell_rule = BooleanNode(operator="AND",
        left=BooleanNode(operator="AND",
            left=ComparisonNode(operator="<",
                left=IndicatorNode(column=f"ema_{f}"), right=IndicatorNode(column=f"ema_{m}")),
            right=ComparisonNode(operator="<",
                left=IndicatorNode(column=f"ema_{m}"), right=IndicatorNode(column=f"ema_{s}"))),
        right=BooleanNode(operator="AND",
            left=ComparisonNode(operator="<",
                left=IndicatorNode(column="rsi_14"), right=ConstantNode(value=rs)),
            right=ComparisonNode(operator=">",
                left=IndicatorNode(column="volume_ratio"), right=ConstantNode(value=1.0))))
    return Strategy(name=f"triple_ema_{f}_{m}_{s}", origin="template",
                    buy_rule=buy_rule, sell_rule=sell_rule, risk_params=_rand_risk())


TEMPLATES = [
    ema_crossover_template, rsi_mean_reversion_template,
    bollinger_breakout_template, macd_momentum_template,
    stochastic_template, triple_ema_template,
]


def generate_from_template():
    return random.choice(TEMPLATES)()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  QuantForge — Expression Tree DSL                                        ║
# ║  Strategies are trees of indicator comparisons + boolean logic           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import uuid
import copy
import json
import random
import numpy as np
import pandas as pd
from typing import Optional, Any, List
from dataclasses import dataclass, field, asdict


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Node:
    """Base class for all expression tree nodes."""
    node_type: str = "base"

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def to_code(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return asdict(self)

    def depth(self) -> int:
        return 1

    def size(self) -> int:
        return 1

    def collect_nodes(self) -> list:
        return [self]

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class IndicatorNode(Node):
    """Leaf node: references a pre-computed indicator column."""
    node_type: str = "indicator"
    column: str = "close"  # column name in DataFrame

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df[self.column]

    def to_code(self) -> str:
        return f"df['{self.column}']"

    def __repr__(self):
        return f"Ind({self.column})"


@dataclass
class ConstantNode(Node):
    """Leaf node: a constant numeric value."""
    node_type: str = "constant"
    value: float = 50.0

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index)

    def to_code(self) -> str:
        return str(self.value)

    def __repr__(self):
        return f"Const({self.value})"


@dataclass
class ComparisonNode(Node):
    """Internal node: compares two sub-expressions."""
    node_type: str = "comparison"
    operator: str = ">"  # >, <, >=, <=, ==
    left: Optional[Node] = None
    right: Optional[Node] = None

    _OPS = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "crossover": lambda a, b: (a > b) & (a.shift(1) <= b.shift(1)),
        "crossunder": lambda a, b: (a < b) & (a.shift(1) >= b.shift(1)),
    }

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        left_val = self.left.evaluate(df) if self.left else pd.Series(0, index=df.index)
        right_val = self.right.evaluate(df) if self.right else pd.Series(0, index=df.index)
        op_fn = self._OPS.get(self.operator, self._OPS[">"])
        return op_fn(left_val, right_val).astype(float).fillna(0)

    def to_code(self) -> str:
        l = self.left.to_code() if self.left else "0"
        r = self.right.to_code() if self.right else "0"
        if self.operator == "crossover":
            return f"(({l} > {r}) & ({l}.shift(1) <= {r}.shift(1)))"
        if self.operator == "crossunder":
            return f"(({l} < {r}) & ({l}.shift(1) >= {r}.shift(1)))"
        if self.operator == "==":
            return f"({l} == {r})"
        return f"({l} {self.operator} {r})"

    def depth(self) -> int:
        ld = self.left.depth() if self.left else 0
        rd = self.right.depth() if self.right else 0
        return 1 + max(ld, rd)

    def size(self) -> int:
        ls = self.left.size() if self.left else 0
        rs = self.right.size() if self.right else 0
        return 1 + ls + rs

    def collect_nodes(self) -> list:
        nodes = [self]
        if self.left:
            nodes.extend(self.left.collect_nodes())
        if self.right:
            nodes.extend(self.right.collect_nodes())
        return nodes

    def __repr__(self):
        return f"({self.left} {self.operator} {self.right})"


@dataclass
class BooleanNode(Node):
    """Internal node: combines boolean sub-expressions."""
    node_type: str = "boolean"
    operator: str = "AND"  # AND, OR
    left: Optional[Node] = None
    right: Optional[Node] = None

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        left_val = self.left.evaluate(df) if self.left else pd.Series(1, index=df.index)
        right_val = self.right.evaluate(df) if self.right else pd.Series(1, index=df.index)
        if self.operator == "AND":
            return ((left_val > 0) & (right_val > 0)).astype(float)
        elif self.operator == "OR":
            return ((left_val > 0) | (right_val > 0)).astype(float)
        return left_val

    def to_code(self) -> str:
        l = self.left.to_code() if self.left else "True"
        r = self.right.to_code() if self.right else "True"
        op = "&" if self.operator == "AND" else "|"
        return f"({l} {op} {r})"

    def depth(self) -> int:
        ld = self.left.depth() if self.left else 0
        rd = self.right.depth() if self.right else 0
        return 1 + max(ld, rd)

    def size(self) -> int:
        ls = self.left.size() if self.left else 0
        rs = self.right.size() if self.right else 0
        return 1 + ls + rs

    def collect_nodes(self) -> list:
        nodes = [self]
        if self.left:
            nodes.extend(self.left.collect_nodes())
        if self.right:
            nodes.extend(self.right.collect_nodes())
        return nodes

    def __repr__(self):
        return f"({self.left} {self.operator} {self.right})"


@dataclass
class ArithmeticNode(Node):
    """Internal node: arithmetic on two sub-expressions (for derived signals)."""
    node_type: str = "arithmetic"
    operator: str = "-"  # +, -, *, /
    left: Optional[Node] = None
    right: Optional[Node] = None

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        left_val = self.left.evaluate(df) if self.left else pd.Series(0, index=df.index)
        right_val = self.right.evaluate(df) if self.right else pd.Series(0, index=df.index)
        if self.operator == "+":
            return left_val + right_val
        elif self.operator == "-":
            return left_val - right_val
        elif self.operator == "*":
            return left_val * right_val
        elif self.operator == "/":
            return left_val / right_val.replace(0, np.nan)
        return left_val

    def to_code(self) -> str:
        l = self.left.to_code() if self.left else "0"
        r = self.right.to_code() if self.right else "0"
        return f"({l} {self.operator} {r})"

    def depth(self) -> int:
        ld = self.left.depth() if self.left else 0
        rd = self.right.depth() if self.right else 0
        return 1 + max(ld, rd)

    def size(self) -> int:
        ls = self.left.size() if self.left else 0
        rs = self.right.size() if self.right else 0
        return 1 + ls + rs

    def collect_nodes(self) -> list:
        nodes = [self]
        if self.left:
            nodes.extend(self.left.collect_nodes())
        if self.right:
            nodes.extend(self.right.collect_nodes())
        return nodes

    def __repr__(self):
        return f"({self.left} {self.operator} {self.right})"


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskParams:
    """Position sizing and risk management parameters."""
    sl_atr_mult: float = 2.0
    rr_ratio: float = 2.0
    risk_pct: float = 0.01
    cooldown: int = 3

    def to_dict(self):
        return asdict(self)


@dataclass
class Strategy:
    """
    Complete strategy definition.
    Contains buy/sell rules as expression trees + risk parameters.
    """
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    generation: int = 0
    origin: str = "random"  # random, template, crossover, mutation
    buy_rule: Optional[Node] = None
    sell_rule: Optional[Node] = None
    risk_params: RiskParams = field(default_factory=RiskParams)

    def __post_init__(self):
        if not self.name:
            self.name = f"strat_{self.strategy_id}"

    def evaluate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate strategy on DataFrame. Returns Series with values:
        1.0 = BUY, -1.0 = SELL, 0.0 = HOLD
        """
        signals = pd.Series(0.0, index=df.index)

        if self.buy_rule:
            buy_mask = self.buy_rule.evaluate(df) > 0
            signals[buy_mask] = 1.0

        if self.sell_rule:
            sell_mask = self.sell_rule.evaluate(df) > 0
            # Sell takes precedence where both fire (conservative)
            signals[sell_mask] = -1.0

        return signals

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "generation": self.generation,
            "origin": self.origin,
            "buy_rule": _node_to_dict(self.buy_rule),
            "sell_rule": _node_to_dict(self.sell_rule),
            "risk_params": self.risk_params.to_dict(),
            "readable": self.readable(),
        }

    def readable(self) -> str:
        """Human-readable strategy description."""
        buy_str = repr(self.buy_rule) if self.buy_rule else "None"
        sell_str = repr(self.sell_rule) if self.sell_rule else "None"
        return (
            f"[{self.name}] origin={self.origin} gen={self.generation}\n"
            f"  BUY:  {buy_str}\n"
            f"  SELL: {sell_str}\n"
            f"  RISK: SL={self.risk_params.sl_atr_mult}×ATR, "
            f"RR={self.risk_params.rr_ratio}, risk={self.risk_params.risk_pct*100}%"
        )

    def copy(self):
        return copy.deepcopy(self)

    def complexity(self) -> int:
        """Total node count across buy + sell rules."""
        b = self.buy_rule.size() if self.buy_rule else 0
        s = self.sell_rule.size() if self.sell_rule else 0
        return b + s


# ═══════════════════════════════════════════════════════════════════════════════
#  SERIALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _node_to_dict(node: Optional[Node]) -> Optional[dict]:
    if node is None:
        return None
    d = {"node_type": node.node_type}
    if isinstance(node, IndicatorNode):
        d["column"] = node.column
    elif isinstance(node, ConstantNode):
        d["value"] = node.value
    elif isinstance(node, (ComparisonNode, BooleanNode, ArithmeticNode)):
        d["operator"] = node.operator
        d["left"] = _node_to_dict(node.left)
        d["right"] = _node_to_dict(node.right)
    return d


def _dict_to_node(d: Optional[dict]) -> Optional[Node]:
    if d is None:
        return None
    nt = d.get("node_type", "")
    if nt == "indicator":
        return IndicatorNode(column=d.get("column", "close"))
    elif nt == "constant":
        return ConstantNode(value=d.get("value", 0.0))
    elif nt == "comparison":
        return ComparisonNode(
            operator=d.get("operator", ">"),
            left=_dict_to_node(d.get("left")),
            right=_dict_to_node(d.get("right")),
        )
    elif nt == "boolean":
        return BooleanNode(
            operator=d.get("operator", "AND"),
            left=_dict_to_node(d.get("left")),
            right=_dict_to_node(d.get("right")),
        )
    elif nt == "arithmetic":
        return ArithmeticNode(
            operator=d.get("operator", "-"),
            left=_dict_to_node(d.get("left")),
            right=_dict_to_node(d.get("right")),
        )
    return None


def strategy_from_dict(d: dict) -> Strategy:
    """Deserialize a strategy from its dict representation."""
    return Strategy(
        strategy_id=d.get("strategy_id", str(uuid.uuid4())[:8]),
        name=d.get("name", ""),
        generation=d.get("generation", 0),
        origin=d.get("origin", "loaded"),
        buy_rule=_dict_to_node(d.get("buy_rule")),
        sell_rule=_dict_to_node(d.get("sell_rule")),
        risk_params=RiskParams(**d["risk_params"]) if "risk_params" in d else RiskParams(),
    )

# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Adaptive Condition Pruner                     ║
# ║  Removes redundant logic to prevent overfitting             ║
# ╚══════════════════════════════════════════════════════════════╝

import copy
import logging
import pandas as pd
from typing import Tuple

from backend.strategy.tree import Strategy, Node, BooleanNode
from backend.engine.backtester import run_backtest
from backend.config import PRUNER_ENABLED, PRUNER_MIN_COMPLEXITY

log = logging.getLogger("quantforge.engine.pruner")

def _get_leaf_paths(node: Node, path: list = None) -> list:
    """Return list of (path, node) for all non-BooleanNode leaves."""
    if path is None:
        path = []
    if node is None:
        return []
    results = []
    if not isinstance(node, BooleanNode):
        results.append((list(path), node))
    if hasattr(node, 'left') and node.left:
        results.extend(_get_leaf_paths(node.left, path + ['left']))
    if hasattr(node, 'right') and node.right:
        results.extend(_get_leaf_paths(node.right, path + ['right']))
    return results


def _remove_by_path(root: Node, path: list) -> Node:
    """Remove the node at `path` from root, replacing parent BooleanNode with its other child."""
    if not path:
        return None
    if len(path) == 1:
        # Direct child of root
        other = 'right' if path[0] == 'left' else 'left'
        return getattr(root, other, None)
    # Navigate to parent
    node = root
    for step in path[:-2]:
        node = getattr(node, step)
    parent = getattr(node, path[-2]) if len(path) >= 2 else root
    attr = path[-1]
    if isinstance(parent, BooleanNode):
        other_attr = 'right' if attr == 'left' else 'left'
        sibling = getattr(parent, other_attr)
        if len(path) == 2:
            setattr(root, path[-2], sibling) if hasattr(root, path[-2]) else None
        grandparent = root
        for step in path[:-2]:
            grandparent = getattr(grandparent, step)
        setattr(grandparent, path[-2], sibling)
    return root


def prune_strategy(strategy: Strategy, train_df: pd.DataFrame, base_score: float) -> Tuple[Strategy, int]:
    if not PRUNER_ENABLED:
        return strategy, 0

    complexity = 0
    if strategy.buy_rule: complexity += strategy.buy_rule.size()
    if strategy.sell_rule: complexity += strategy.sell_rule.size()
    if complexity < PRUNER_MIN_COMPLEXITY:
        return strategy, 0

    nodes_removed = 0
    pruned_strat = copy.deepcopy(strategy)

    import random
    for _ in range(10):
        improved = False
        for rule_attr in ['buy_rule', 'sell_rule']:
            rule = getattr(pruned_strat, rule_attr)
            if rule is None:
                continue
            leaf_paths = _get_leaf_paths(rule)
            random.shuffle(leaf_paths)
            for path, _ in leaf_paths:
                test_strat = copy.deepcopy(pruned_strat)
                test_rule = getattr(test_strat, rule_attr)
                new_rule = _remove_by_path(test_rule, path)
                setattr(test_strat, rule_attr, new_rule)
                try:
                    bt = run_backtest(test_strat, train_df)
                    test_score = bt.sharpe_ratio * 0.3 + bt.total_return_pct * 0.001
                    if test_score >= base_score * 0.98 and bt.total_trades >= 10:
                        pruned_strat = test_strat
                        nodes_removed += 1
                        improved = True
                        break
                except Exception:
                    continue
            if improved:
                break
        if not improved:
            break

    if nodes_removed > 0:
        log.info(f"[PRUNER] Pruned {nodes_removed} nodes from {strategy.strategy_id}")
    return pruned_strat, nodes_removed

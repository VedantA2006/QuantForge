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

def _get_parent_and_attr(node: Node, target_id: str) -> Tuple[Node, str]:
    """Find parent of a specific node id."""
    def _search(curr: Node, parent: Node, attr: str):
        if curr is None: return None
        if id(curr) == target_id: return (parent, attr)
        if hasattr(curr, 'left') and curr.left:
            res = _search(curr.left, curr, 'left')
            if res: return res
        if hasattr(curr, 'right') and curr.right:
            res = _search(curr.right, curr, 'right')
            if res: return res
        return None
        
    return _search(node, None, None)

def _remove_node(root: Node, target_id: str) -> Node:
    """Removes a node and restructures the tree."""
    if root is None: return None
    if id(root) == target_id:
        return None
        
    result = _get_parent_and_attr(root, target_id)
    if not result: return root
    
    parent, attr = result
    
    # If parent is BooleanNode, replace parent with the other child
    if isinstance(parent, BooleanNode):
        other_child = parent.right if attr == 'left' else parent.left
        grand_result = _get_parent_and_attr(root, id(parent))
        if not grand_result:
            return other_child # We are replacing the root
        g_parent, g_attr = grand_result
        setattr(g_parent, g_attr, other_child)
    else:
        setattr(parent, attr, None)
        
    return root

def prune_strategy(strategy: Strategy, train_df: pd.DataFrame, base_score: float) -> Tuple[Strategy, int]:
    """
    Attempts to remove nodes if score doesn't drop > 2%.
    Returns (pruned_strategy, nodes_removed).
    """
    if not PRUNER_ENABLED:
        return strategy, 0
        
    complexity = 0
    if strategy.buy_rule: complexity += strategy.buy_rule.size()
    if strategy.sell_rule: complexity += strategy.sell_rule.size()
    
    if complexity < PRUNER_MIN_COMPLEXITY:
        return strategy, 0
        
    pruned_strat = copy.deepcopy(strategy)
    nodes_removed = 0
    iterations = 0
    
    while iterations < 10:
        iterations += 1
        nodes = []
        if pruned_strat.buy_rule:
            nodes.extend([(n, "buy") for n in pruned_strat.buy_rule.collect_nodes() if not isinstance(n, BooleanNode)])
        if pruned_strat.sell_rule:
            nodes.extend([(n, "sell") for n in pruned_strat.sell_rule.collect_nodes() if not isinstance(n, BooleanNode)])
            
        if not nodes:
            break
            
        removed_this_pass = False
        for node, rule_type in nodes:
            test_strat = copy.deepcopy(pruned_strat)
            target_id = id(node)
            
            # Re-find node id in the copy
            # We can just remove the equivalent path, or simpler:
            # We must map the structure. A safer way is to just use the original structure's paths, but id() changes.
            # Let's rebuild the tree without the node's index.
            pass
            
            # Actually, simpler logic: collect nodes in test_strat.
            test_nodes = []
            if rule_type == "buy":
                test_nodes = test_strat.buy_rule.collect_nodes() if test_strat.buy_rule else []
            else:
                test_nodes = test_strat.sell_rule.collect_nodes() if test_strat.sell_rule else []
                
            # Filter non-boolean
            test_nodes = [n for n in test_nodes if not isinstance(n, BooleanNode)]
            if not test_nodes: continue
            
            # Try removing a random node for simplicity and speed, up to 3 attempts per pass
            import random
            test_node = random.choice(test_nodes)
            
            if rule_type == "buy":
                test_strat.buy_rule = _remove_node(test_strat.buy_rule, id(test_node))
            else:
                test_strat.sell_rule = _remove_node(test_strat.sell_rule, id(test_node))
                
            # Backtest
            try:
                bt = run_backtest(test_strat, train_df)
                test_score = bt.sharpe_ratio * 0.3 + bt.total_return_pct * 0.001
                
                if test_score >= base_score * 0.98 and bt.total_trades >= 10:
                    pruned_strat = test_strat
                    nodes_removed += 1
                    removed_this_pass = True
                    break # Restart while loop with new strat
            except Exception:
                continue
                
        if not removed_this_pass:
            break
            
    if nodes_removed > 0:
        log.info(f"[PRUNER] Pruned {nodes_removed} nodes from {strategy.strategy_id}")
        
    return pruned_strat, nodes_removed

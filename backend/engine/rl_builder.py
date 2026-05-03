# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — RL Condition Builder                          ║
# ║  Reinforcement Learning agent to assemble optimal trees     ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import random
import logging
import numpy as np
from pathlib import Path

from backend.strategy.tree import Strategy, BooleanNode
from backend.strategy.templates import _rand_risk
import uuid
from backend.strategy.generator import CATEGORY_WEIGHTS, _generate_category_condition
from backend.config import RL_ENABLED, RL_LR

log = logging.getLogger("quantforge.engine.rl_builder")

RL_WEIGHTS_PATH = Path(os.path.dirname(__file__)) / ".." / "data" / "rl_weights.npy"


class RLBuilder:
    def __init__(self):
        self.categories = list(CATEGORY_WEIGHTS.keys())
        self.n_cat = len(self.categories)
        self.lr = RL_LR
        
        # W shapes: (n_cat, n_cat) — learned affinity between state and category
        self.W_buy = np.zeros((self.n_cat, self.n_cat))
        self.W_sell = np.zeros((self.n_cat, self.n_cat))
        
        self.updates_count = 0
        self.stats = {
            "total_updates": 0,
            "avg_reward_50": 0.0,
        }
        self._recent_rewards = []
        
        if RL_WEIGHTS_PATH.exists():
            try:
                data = np.load(RL_WEIGHTS_PATH, allow_pickle=True).item()
                self.W_buy = data["W_buy"]
                self.W_sell = data["W_sell"]
                self.stats["total_updates"] = data.get("total_updates", 0)
                log.info("Loaded pre-trained RL weights")
            except Exception as e:
                log.warning(f"Failed to load RL weights: {e}")

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _get_state_vector(self):
        """Current state is the category weights."""
        state = np.zeros(self.n_cat)
        for i, c in enumerate(self.categories):
            state[i] = CATEGORY_WEIGHTS.get(c, 1.0)
        # Normalize
        if state.sum() > 0:
            state = state / state.sum()
        return state

    def generate_strategy(self) -> tuple:
        """Returns (Strategy, episode_data)"""
        if not RL_ENABLED:
            return None, None
            
        state = self._get_state_vector()
        
        # Sample Buy
        logits_buy = state @ self.W_buy
        probs_buy = self._softmax(logits_buy)
        n_buy = random.randint(2, 5)
        buy_indices = np.random.choice(self.n_cat, size=n_buy, replace=False, p=probs_buy)
        
        # Sample Sell
        logits_sell = state @ self.W_sell
        probs_sell = self._softmax(logits_sell)
        n_sell = random.randint(2, 5)
        sell_indices = np.random.choice(self.n_cat, size=n_sell, replace=False, p=probs_sell)
        
        # Build Buy Tree
        buy_nodes = []
        for idx in buy_indices:
            cat = self.categories[idx]
            node = _generate_category_condition(cat, "buy")
            if node: buy_nodes.append(node)
            
        buy_tree = None
        if buy_nodes:
            buy_tree = buy_nodes[0]
            for node in buy_nodes[1:]:
                buy_tree = BooleanNode(operator="AND", left=buy_tree, right=node)
                
        # Build Sell Tree
        sell_nodes = []
        for idx in sell_indices:
            cat = self.categories[idx]
            node = _generate_category_condition(cat, "sell")
            if node: sell_nodes.append(node)
            
        sell_tree = None
        if sell_nodes:
            sell_tree = sell_nodes[0]
            for node in sell_nodes[1:]:
                sell_tree = BooleanNode(operator="AND", left=sell_tree, right=node)
                
        strat_id = uuid.uuid4().hex[:8]
        strat = Strategy(
            strategy_id=strat_id,
            name="rl_" + strat_id,
            origin="rl_agent",
            buy_rule=buy_tree,
            sell_rule=sell_tree,
            risk_params=_rand_risk(),
            generation=0,
        )
        
        episode = {
            "state": state,
            "buy_indices": buy_indices,
            "sell_indices": sell_indices,
            "probs_buy": probs_buy,
            "probs_sell": probs_sell
        }
        
        return strat, episode

    def update(self, episode: dict, reward: float):
        if not RL_ENABLED or not episode:
            return
            
        state = episode["state"]
        
        # Update Buy W
        for idx in episode["buy_indices"]:
            prob = episode["probs_buy"][idx]
            # REINFORCE update: grad of log(prob) w.r.t W is state * (1 - prob)
            grad = np.outer(state, (1 - prob))
            self.W_buy[:, idx] += self.lr * reward * grad[:, 0]
            
        # Update Sell W
        for idx in episode["sell_indices"]:
            prob = episode["probs_sell"][idx]
            grad = np.outer(state, (1 - prob))
            self.W_sell[:, idx] += self.lr * reward * grad[:, 0]
            
        self.stats["total_updates"] += 1
        self.updates_count += 1
        
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > 50:
            self._recent_rewards.pop(0)
        self.stats["avg_reward_50"] = sum(self._recent_rewards) / len(self._recent_rewards)
        
        # Persist every 50 updates
        if self.updates_count >= 50:
            self.updates_count = 0
            try:
                RL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                np.save(RL_WEIGHTS_PATH, {
                    "W_buy": self.W_buy,
                    "W_sell": self.W_sell,
                    "total_updates": self.stats["total_updates"]
                })
            except Exception as e:
                log.warning(f"Failed to save RL weights: {e}")

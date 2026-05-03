# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Genetic Algorithm + Bayesian Optimizer        ║
# ╚══════════════════════════════════════════════════════════════╝

import copy
import random
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict

from backend.strategy.tree import (
    Strategy, RiskParams, Node, BooleanNode, ComparisonNode,
    IndicatorNode, ConstantNode, ArithmeticNode,
)
from backend.strategy.generator import (
    _random_comparison, _random_tree, _BOUNDED_INDICATORS,
    _BOUNDED_RANGES, _CROSS_INDICATORS, _COMPARISON_OPS, _BOOLEAN_OPS,
)
from backend.config import (
    GA_POPULATION_SIZE, GA_GENERATIONS, GA_TOURNAMENT_K,
    GA_CROSSOVER_RATE, GA_MUTATION_RATE, GA_PARAM_MUTATION_RATE,
    GA_ELITISM_COUNT,
)

log = logging.getLogger("quantforge.engine.optimizer")


# ═══════════════════════════════════════════════════════════════
#  GA — SELECTION
# ═══════════════════════════════════════════════════════════════

def tournament_select(population: List[Tuple[Strategy, float]], k: int = GA_TOURNAMENT_K) -> Strategy:
    contestants = random.sample(population, min(k, len(population)))
    winner = max(contestants, key=lambda x: x[1])
    return winner[0].copy()


# ═══════════════════════════════════════════════════════════════
#  GA — CROSSOVER
# ═══════════════════════════════════════════════════════════════

def _get_random_subtree_parent(node: Node) -> Optional[Tuple[Node, str]]:
    candidates = []
    _collect_internal(node, candidates)
    if not candidates:
        return None
    parent, attr = random.choice(candidates)
    return parent, attr


def _collect_internal(node: Node, acc: list, parent=None, attr=None):
    if isinstance(node, (BooleanNode, ComparisonNode, ArithmeticNode)):
        if parent is not None:
            acc.append((parent, attr))
        if hasattr(node, 'left') and node.left:
            _collect_internal(node.left, acc, node, 'left')
        if hasattr(node, 'right') and node.right:
            _collect_internal(node.right, acc, node, 'right')


def _get_random_subtree(node: Node) -> Optional[Node]:
    all_nodes = node.collect_nodes()
    if len(all_nodes) <= 1:
        return node.copy()
    return random.choice(all_nodes[1:]).copy()


def crossover(parent_a: Strategy, parent_b: Strategy) -> Strategy:
    child = parent_a.copy()
    child.origin = "crossover"
    child.generation = max(parent_a.generation, parent_b.generation) + 1

    if random.random() < 0.5 and child.buy_rule and parent_b.buy_rule:
        target_rule = "buy_rule"
        donor_tree = parent_b.buy_rule
    elif child.sell_rule and parent_b.sell_rule:
        target_rule = "sell_rule"
        donor_tree = parent_b.sell_rule
    else:
        return child

    child_tree = getattr(child, target_rule)
    if child_tree is None or donor_tree is None:
        return child

    donor_sub = _get_random_subtree(donor_tree)
    if donor_sub is None:
        return child

    result = _get_random_subtree_parent(child_tree)
    if result is None:
        setattr(child, target_rule, donor_sub)
    else:
        parent_node, attr = result
        setattr(parent_node, attr, donor_sub)

    child.risk_params = _perturb_risk(parent_a.risk_params)
    child.strategy_id = child.strategy_id[:4] + "x" + parent_b.strategy_id[:3]
    child.name = f"cross_{child.strategy_id}"
    return child


# ═══════════════════════════════════════════════════════════════
#  GA — MUTATION
# ═══════════════════════════════════════════════════════════════

def mutate(strategy: Strategy) -> Strategy:
    child = strategy.copy()
    child.origin = "mutation"
    child.generation += 1

    if random.random() < 0.5 and child.buy_rule:
        rule = child.buy_rule
    elif child.sell_rule:
        rule = child.sell_rule
    else:
        return child

    nodes = rule.collect_nodes()
    if not nodes:
        return child

    target = random.choice(nodes)
    mutation_type = random.choices(
        ["indicator", "operator", "constant", "regrow", "param"],
        weights=[0.25, 0.20, 0.25, 0.15, 0.15], k=1,
    )[0]

    if mutation_type == "indicator" and isinstance(target, IndicatorNode):
        target.column = random.choice(_CROSS_INDICATORS)
    elif mutation_type == "operator":
        if isinstance(target, ComparisonNode):
            target.operator = random.choice(_COMPARISON_OPS)
        elif isinstance(target, BooleanNode):
            target.operator = random.choice(_BOOLEAN_OPS)
    elif mutation_type == "constant" and isinstance(target, ConstantNode):
        noise = random.gauss(0, target.value * 0.15) if target.value != 0 else random.gauss(0, 5)
        target.value = round(target.value + noise, 2)
        target.value = max(0, min(target.value, 100))
    elif mutation_type == "regrow":
        result = _get_random_subtree_parent(rule)
        if result:
            parent_node, attr = result
            setattr(parent_node, attr, _random_comparison())
    elif mutation_type == "param":
        child.risk_params = _perturb_risk(child.risk_params)

    child.name = f"mut_{child.strategy_id}"
    return child


def _perturb_risk(params: RiskParams) -> RiskParams:
    return RiskParams(
        sl_atr_mult=round(max(1.0, min(2.0, params.sl_atr_mult + random.gauss(0, 0.15))), 1),
        rr_ratio=round(max(2.0, min(5.0, params.rr_ratio + random.gauss(0, 0.2))), 1),
        risk_pct=round(max(0.005, min(0.015, params.risk_pct + random.gauss(0, 0.001))), 3),
        cooldown=max(4, min(10, params.cooldown + random.choice([-1, 0, 0, 1]))),
        trail_mult=round(max(0, min(4.0, params.trail_mult + random.gauss(0, 0.3))), 1),
        tp1_ratio=round(max(0, min(0.7, params.tp1_ratio + random.gauss(0, 0.05))), 2),
    )


# ═══════════════════════════════════════════════════════════════
#  GA — EVOLUTION (single generation)
# ═══════════════════════════════════════════════════════════════

def evolve_population(
    population_with_fitness: List[Tuple[Strategy, float]],
    pop_size: int = GA_POPULATION_SIZE,
) -> List[Strategy]:
    if len(population_with_fitness) < 4:
        log.warning("Population too small for GA evolution")
        return []

    pop = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
    log.info(f"[GA] Evolution: pop={len(pop)}, best_fitness={pop[0][1]:.4f}")

    next_gen = []
    for i in range(min(GA_ELITISM_COUNT, len(pop))):
        elite = pop[i][0].copy()
        elite.origin = "elite"
        next_gen.append(elite)

    while len(next_gen) < pop_size:
        if random.random() < GA_CROSSOVER_RATE and len(pop) >= 2:
            p1 = tournament_select(pop)
            p2 = tournament_select(pop)
            child = crossover(p1, p2)
        else:
            parent = tournament_select(pop)
            child = parent.copy()

        if random.random() < GA_MUTATION_RATE:
            child = mutate(child)
        elif random.random() < GA_PARAM_MUTATION_RATE:
            child.risk_params = _perturb_risk(child.risk_params)

        child.generation += 1
        next_gen.append(child)

    offspring = next_gen[GA_ELITISM_COUNT:]
    log.info(f"[GA] Produced {len(offspring)} offspring")
    return offspring


# ═══════════════════════════════════════════════════════════════
#  BAYESIAN OPTIMIZER (GP-UCB)
# ═══════════════════════════════════════════════════════════════

class BayesianOptimizer:
    """Gaussian Process Bayesian optimizer for risk parameter tuning."""

    # Parameter bounds: 6 risk + 8 structural (buy_depth, sell_depth, buy_size, sell_size, n_ema, n_rsi, n_macd, origin)
    BOUNDS = np.array([
        [1.0, 2.0],    # sl_atr_mult
        [2.0, 5.0],    # rr_ratio
        [0.005, 0.015],# risk_pct
        [4, 10],       # cooldown
        [0.0, 3.0],    # trail_mult
        [0.0, 0.6],    # tp1_ratio
        [0, 10],       # buy_depth
        [0, 10],       # sell_depth
        [0, 30],       # buy_size
        [0, 30],       # sell_size
        [0, 10],       # n_ema
        [0, 10],       # n_rsi
        [0, 10],       # n_macd
        [0, 5],        # origin
    ])

    def __init__(self):
        self._history = {}
        self._gp_model = {}
        self._fit_count = {}
        self._gp_available = False
        self._beta = 2.0
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel
            self._gp_available = True
        except ImportError:
            log.warning("sklearn not installed — Bayesian optimizer disabled")

    def load_history(self, db):
        """Replay history from MongoDB."""
        try:
            obs = db.db["gp_observations"].find().sort([("_id", -1)]).limit(5000)
            for doc in reversed(list(obs)):
                asset = doc.get("asset", "BTCUSDT")
                if asset not in self._history:
                    self._history[asset] = []
                    self._fit_count[asset] = 0
                self._history[asset].append((np.array(doc["params"]), doc["score"]))
            log.info(f"[BAYES] Loaded {sum(len(v) for v in self._history.values())} history observations")
        except Exception as e:
            log.warning(f"[BAYES] Failed to load history: {e}")

    def record(self, asset: str, params_vector: list, score: float, db=None):
        """Record an observation."""
        if asset not in self._history:
            self._history[asset] = []
            self._fit_count[asset] = 0
            
        # Pad to 14 features if needed
        full_params = list(params_vector)
        while len(full_params) < 14:
            full_params.append(0.0)
            
        self._history[asset].append((np.array(full_params[:14]), score))
        
        if db:
            try:
                db.db["gp_observations"].insert_one({
                    "asset": asset,
                    "params": full_params[:14],
                    "score": score
                })
            except Exception:
                pass

    def suggest(self, asset: str, n: int = 5) -> list:
        """Suggest n parameter vectors using GP-UCB acquisition."""
        self._beta = max(0.1, self._beta - 0.01) # Decay beta
        
        if not self._gp_available:
            return self._random_suggestions(n)

        history = self._history.get(asset, [])
        if len(history) < 30:
            return self._random_suggestions(n)

        # Fit or refit GP
        count_since_fit = len(history) - self._fit_count.get(asset, 0)
        if asset not in self._gp_model or count_since_fit >= 25:
            self._fit_gp(asset)

        gp, y_mean, y_std_dev = self._gp_model.get(asset, (None, 0, 1))
        if gp is None:
            return self._random_suggestions(n)

        # UCB acquisition over random candidates
        candidates = self._random_candidates(2000)
        try:
            mean_norm, std_norm = gp.predict(candidates, return_std=True)
            # Unstandardize (optional, but UCB works on normalized scale too)
            ucb = mean_norm + self._beta * std_norm
            top_idx = np.argsort(ucb)[-n:][::-1]
            suggestions = [candidates[i].tolist() for i in top_idx]
            log.info(f"[BAYES] Suggested {n} candidates (best UCB={ucb[top_idx[0]]:.4f}, beta={self._beta:.2f})")
            return suggestions
        except Exception as e:
            log.warning(f"[BAYES] Prediction failed: {e}")
            return self._random_suggestions(n)

    def _fit_gp(self, asset: str):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel

            history = self._history[asset]
            X = np.array([h[0] for h in history])
            y = np.array([h[1] for h in history])

            # Normalize X
            X_norm = (X - self.BOUNDS[:, 0]) / (self.BOUNDS[:, 1] - self.BOUNDS[:, 0])
            
            # Standardize Y
            y_mean = np.mean(y)
            y_std = np.std(y) if np.std(y) > 0 else 1.0
            y_norm = (y - y_mean) / y_std

            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.01)
            gp.fit(X_norm, y_norm)
            
            self._gp_model[asset] = (gp, y_mean, y_std)
            self._fit_count[asset] = len(history)
            log.info(f"[BAYES] GP fitted on {len(history)} observations for {asset}")
        except Exception as e:
            log.warning(f"[BAYES] GP fit failed: {e}")

    def _random_candidates(self, n: int) -> np.ndarray:
        candidates = np.random.uniform(0, 1, size=(n, len(self.BOUNDS)))
        return candidates

    def _random_suggestions(self, n: int) -> list:
        suggestions = []
        for _ in range(n):
            params = []
            for lo, hi in self.BOUNDS:
                params.append(round(random.uniform(lo, hi), 3))
            suggestions.append(params)
        return suggestions

# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Genetic Algorithm Optimizer                   ║
# ║  Evolves strategies via tournament, crossover, mutation     ║
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
#  SELECTION
# ═══════════════════════════════════════════════════════════════

def tournament_select(
    population: List[Tuple[Strategy, float]],
    k: int = GA_TOURNAMENT_K,
) -> Strategy:
    """Tournament selection: pick k random, return the fittest."""
    contestants = random.sample(population, min(k, len(population)))
    winner = max(contestants, key=lambda x: x[1])
    return winner[0].copy()


# ═══════════════════════════════════════════════════════════════
#  CROSSOVER
# ═══════════════════════════════════════════════════════════════

def _get_random_subtree_parent(node: Node) -> Optional[Tuple[Node, str]]:
    """Find a random internal node and which child to swap."""
    candidates = []
    _collect_internal(node, candidates)
    if not candidates:
        return None
    parent, attr = random.choice(candidates)
    return parent, attr


def _collect_internal(node: Node, acc: list, parent=None, attr=None):
    """Collect all (parent, child_attr) pairs for internal nodes."""
    if isinstance(node, (BooleanNode, ComparisonNode, ArithmeticNode)):
        if parent is not None:
            acc.append((parent, attr))
        if hasattr(node, 'left') and node.left:
            _collect_internal(node.left, acc, node, 'left')
        if hasattr(node, 'right') and node.right:
            _collect_internal(node.right, acc, node, 'right')


def _get_random_subtree(node: Node) -> Optional[Node]:
    """Get a random subtree from the tree."""
    all_nodes = node.collect_nodes()
    if len(all_nodes) <= 1:
        return node.copy()
    return random.choice(all_nodes[1:]).copy()  # skip root


def crossover(parent_a: Strategy, parent_b: Strategy) -> Strategy:
    """
    Subtree crossover: swap a random subtree between two parents.
    Risk params come from the fitter parent (parent_a assumed fitter).
    """
    child = parent_a.copy()
    child.origin = "crossover"
    child.generation = max(parent_a.generation, parent_b.generation) + 1

    # Decide which rule to crossover
    if random.random() < 0.5 and child.buy_rule and parent_b.buy_rule:
        target_rule = "buy_rule"
        donor_tree = parent_b.buy_rule
    elif child.sell_rule and parent_b.sell_rule:
        target_rule = "sell_rule"
        donor_tree = parent_b.sell_rule
    else:
        return child  # can't crossover

    child_tree = getattr(child, target_rule)
    if child_tree is None or donor_tree is None:
        return child

    # Get a subtree from donor
    donor_sub = _get_random_subtree(donor_tree)
    if donor_sub is None:
        return child

    # Find a swap point in child
    result = _get_random_subtree_parent(child_tree)
    if result is None:
        # Replace entire rule
        setattr(child, target_rule, donor_sub)
    else:
        parent_node, attr = result
        setattr(parent_node, attr, donor_sub)

    # Small perturbation to risk params
    child.risk_params = _perturb_risk(parent_a.risk_params)
    child.strategy_id = child.strategy_id[:4] + "x" + parent_b.strategy_id[:3]
    child.name = f"cross_{child.strategy_id}"

    return child


# ═══════════════════════════════════════════════════════════════
#  MUTATION
# ═══════════════════════════════════════════════════════════════

def mutate(strategy: Strategy) -> Strategy:
    """
    Apply one or more mutations to a strategy.
    Mutation types:
    1. Indicator swap
    2. Operator swap
    3. Constant perturbation
    4. Subtree regrow
    5. Parameter mutation
    """
    child = strategy.copy()
    child.origin = "mutation"
    child.generation += 1

    # Pick which rule to mutate
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
        weights=[0.25, 0.20, 0.25, 0.15, 0.15],
        k=1,
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
    """Small gaussian perturbation to risk parameters."""
    return RiskParams(
        sl_atr_mult=round(max(0.5, params.sl_atr_mult + random.gauss(0, 0.2)), 1),
        rr_ratio=round(max(1.0, params.rr_ratio + random.gauss(0, 0.3)), 1),
        risk_pct=round(max(0.003, min(0.03, params.risk_pct + random.gauss(0, 0.002))), 3),
        cooldown=max(1, min(8, params.cooldown + random.choice([-1, 0, 0, 1]))),
    )


# ═══════════════════════════════════════════════════════════════
#  EVOLUTION LOOP
# ═══════════════════════════════════════════════════════════════

def evolve_population(
    population_with_fitness: List[Tuple[Strategy, float]],
    generations: int = GA_GENERATIONS,
    pop_size: int = GA_POPULATION_SIZE,
) -> List[Strategy]:
    """
    Run genetic algorithm evolution.

    Args:
        population_with_fitness: List of (Strategy, fitness_score) tuples
        generations: Number of generations to evolve
        pop_size: Target population size

    Returns:
        List of offspring strategies ready for evaluation
    """
    if len(population_with_fitness) < 4:
        log.warning("Population too small for GA evolution")
        return []

    # Sort by fitness
    pop = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)

    log.info(
        f"[GA] Evolution: pop={len(pop)}, gens={generations}, "
        f"best_fitness={pop[0][1]:.4f}"
    )

    current_pop = pop

    for gen in range(generations):
        next_gen = []

        # Elitism: carry top performers unchanged
        for i in range(min(GA_ELITISM_COUNT, len(current_pop))):
            elite = current_pop[i][0].copy()
            elite.origin = "elite"
            next_gen.append(elite)

        # Fill rest with crossover + mutation
        while len(next_gen) < pop_size:
            if random.random() < GA_CROSSOVER_RATE and len(current_pop) >= 2:
                p1 = tournament_select(current_pop)
                p2 = tournament_select(current_pop)
                child = crossover(p1, p2)
            else:
                parent = tournament_select(current_pop)
                child = parent.copy()

            # Mutation
            if random.random() < GA_MUTATION_RATE:
                child = mutate(child)
            elif random.random() < GA_PARAM_MUTATION_RATE:
                child.risk_params = _perturb_risk(child.risk_params)

            child.generation = gen + 1
            next_gen.append(child)

        # For the evolution loop we don't re-evaluate here
        # (that happens in the main discovery loop)
        # We just track the previous fitness for selection
        current_pop = [
            (s, pop[i % len(pop)][1]) for i, s in enumerate(next_gen)
        ]

    # Return final generation offspring (excluding elites already stored)
    offspring = [s for s, _ in current_pop[GA_ELITISM_COUNT:]]
    log.info(f"[GA] Produced {len(offspring)} offspring")
    return offspring

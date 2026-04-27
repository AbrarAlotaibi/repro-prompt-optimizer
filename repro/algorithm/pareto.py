"""
Pareto Frontier Computation

Multi-objective optimization for attack selection.
"""

from typing import List, Tuple, Dict, Sequence
from repro.algorithm.models import AttackCandidate


def dominates(a: Dict[str, float], b: Dict[str, float],
              maximize: Sequence[str], minimize: Sequence[str]) -> bool:
    """Check if metrics 'a' Pareto-dominate metrics 'b'."""
    better_or_equal = True
    strictly_better = False
    for m in maximize:
        if a[m] < b[m]:
            better_or_equal = False
            break
        if a[m] > b[m]:
            strictly_better = True
    if better_or_equal:
        for m in minimize:
            if a[m] > b[m]:
                better_or_equal = False
                break
            if a[m] < b[m]:
                strictly_better = True
    return better_or_equal and strictly_better


def pareto_frontier(cands: List[Tuple[AttackCandidate, Dict[str, float]]],
                    maximize: Sequence[str], minimize: Sequence[str]):
    """Extract Pareto frontier from candidates."""
    front = []
    for i, (ci, mi) in enumerate(cands):
        dominated = False
        for j, (cj, mj) in enumerate(cands):
            if i == j:
                continue
            if dominates(mj, mi, maximize, minimize):
                dominated = True
                break
        if not dominated:
            front.append((ci, mi))
    return front


def epsilon_pareto_frontier(cands: List[Tuple[AttackCandidate, Dict[str, float]]],
                             maximize: Sequence[str], minimize: Sequence[str],
                             epsilon: float = 0.05, min_frontier_size: int = 3):
    """
    Epsilon-Pareto frontier: includes candidates within epsilon of the true Pareto front.
    This preserves diversity by keeping near-optimal solutions.

    Args:
        epsilon: Tolerance for metric differences (e.g., 0.05 = 5% difference)
        min_frontier_size: Minimum number of candidates to include (default 3 for crossover)
    """
    if not cands:
        return []

    # Start with true Pareto frontier
    front = pareto_frontier(cands, maximize, minimize)

    # If frontier is too small, add epsilon-dominated candidates
    if len(front) < min_frontier_size and len(cands) > len(front):
        remaining = [c for c in cands if c not in front]

        # For each remaining candidate, check if it's epsilon-dominated
        for ci, mi in remaining:
            near_optimal = False
            for cj, mj in front:
                # Check if within epsilon on all metrics
                within_epsilon = True
                for key in maximize:
                    if mi.get(key, 0) < mj.get(key, 0) * (1 - epsilon):
                        within_epsilon = False
                        break
                for key in minimize:
                    if mi.get(key, float('inf')) > mj.get(key, float('inf')) * (1 + epsilon):
                        within_epsilon = False
                        break
                if within_epsilon:
                    near_optimal = True
                    break

            if near_optimal:
                front.append((ci, mi))
                if len(front) >= min_frontier_size * 2:  # Don't make it too large
                    break

    # force-add top candidates by primary metric to ensure diversity for crossover
    if len(front) < min_frontier_size and len(cands) > len(front):
        remaining = [c for c in cands if c not in front]

        # Sort remaining by primary maximize metric (usually ASR or XFER)
        primary_metric = maximize[0] if maximize else None
        if primary_metric:
            remaining.sort(key=lambda x: x[1].get(primary_metric, 0), reverse=True)

        # Add top candidates until we reach min_frontier_size
        for ci, mi in remaining:
            if len(front) >= min_frontier_size:
                break
            front.append((ci, mi))

    return front

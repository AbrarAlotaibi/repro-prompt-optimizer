"""
REPRO Algorithm Components

Core algorithmic components for REPRO attack optimization.
"""

from repro.algorithm.models import (
    AttackCandidate,
    Task,
    Environment,
    Conversation,
    EvalRow,
    LogRow,
)
from repro.algorithm.pareto import (
    dominates,
    pareto_frontier,
    epsilon_pareto_frontier,
)
from repro.algorithm.crossover import ReflectiveCrossover
from repro.algorithm.reflective_editor import ReflectiveLMEditor
from repro.algorithm.evaluator import Evaluator
from repro.algorithm.optimizer import REPROAttackOptimizer
from repro.algorithm.datasets import (
    load_tasks_from_jsonl,
    load_tasks_from_hf,
    load_tasks_from_csv,
)

__all__ = [
    # Data models
    "AttackCandidate",
    "Task",
    "Environment",
    "Conversation",
    "EvalRow",
    "LogRow",
    # Pareto
    "dominates",
    "pareto_frontier",
    "epsilon_pareto_frontier",
    # Crossover
    "ReflectiveCrossover",
    # Core components
    "ReflectiveLMEditor",
    "Evaluator",
    "REPROAttackOptimizer",
    # Datasets
    "load_tasks_from_jsonl",
    "load_tasks_from_hf",
    "load_tasks_from_csv",
]

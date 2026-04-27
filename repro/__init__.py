"""
REPRO: Reflective Evolutionary Prompt Optimization

A modular red-teaming framework for LLM security evaluation using evolutionary
algorithms and Reflective Language Models.

Inspired by DSPy and genetic algorithms.
"""

__version__ = "1.0.0"

from repro.algorithm import (
    REPROAttackOptimizer,
    AttackCandidate,
    Task,
    Environment,
    ReflectiveLMEditor,
    Evaluator,
)
from repro.models.target import TargetModel, OpenAIModel, OllamaModel
from repro.models.judge import Judge, OpenAIJudge, OllamaJudge

__all__ = [
    # Main optimizer
    "REPROAttackOptimizer",
    # Core models
    "AttackCandidate",
    "Task",
    "Environment",
    # Components
    "ReflectiveLMEditor",
    "Evaluator",
    # Target models
    "TargetModel",
    "OpenAIModel",
    "OllamaModel",
    # Judges
    "Judge",
    "OpenAIJudge",
    "OllamaJudge",
]

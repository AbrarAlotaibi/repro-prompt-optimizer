"""Separate tracking for crossover operations."""
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CrossoverStats:
    """Statistics for crossover operations"""
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    parent_count_2: int = 0  # 2-parent crossover attempts
    parent_count_3: int = 0  # 3-parent crossover attempts
    self_crossover: int = 0  # single parent self-crossover
    parent_count_2_successes: int = 0
    parent_count_3_successes: int = 0
    self_crossover_successes: int = 0
    avg_generation_time: float = 0.0
    recent_success_rate: float = 0.0

    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def success_rate_2p(self) -> float:
        if self.parent_count_2 == 0:
            return 0.0
        return self.parent_count_2_successes / self.parent_count_2

    def success_rate_3p(self) -> float:
        if self.parent_count_3 == 0:
            return 0.0
        return self.parent_count_3_successes / self.parent_count_3

    def success_rate_self(self) -> float:
        if self.self_crossover == 0:
            return 0.0
        return self.self_crossover_successes / self.self_crossover

@dataclass
class CrossoverAttempt:
    """Record of a single crossover attempt"""
    timestamp: float
    parent_count: int
    parent_strategies: List[str]
    success: bool
    generation_time: float
    offspring_count: int

class CrossoverTracker:
    """Track crossover operations separately from mutation strategies"""

    def __init__(self):
        self.stats = CrossoverStats()
        self.recent_attempts: List[CrossoverAttempt] = []
        self.max_recent_history = 50

    def record_crossover_attempt(self, parents: List[Any], success: bool,
                                generation_time: float = 0.0, offspring_count: int = 0):
        """Record a crossover operation"""

        parent_count = len(parents)
        parent_strategies = []
        for parent in parents:
            if hasattr(parent, 'strategy'):
                parent_strategies.append(parent.strategy or 'llm_generated')
            elif hasattr(parent, 'meta') and parent.meta:
                parent_strategies.append(parent.meta.get('edit', '') or 'llm_generated')
            elif isinstance(parent, tuple) and len(parent) >= 2:
                parent_strategies.append('llm_generated')
            else:
                parent_strategies.append('llm_generated')

        attempt = CrossoverAttempt(
            timestamp=time.time(),
            parent_count=parent_count,
            parent_strategies=parent_strategies,
            success=success,
            generation_time=generation_time,
            offspring_count=offspring_count
        )

        self.stats.attempts += 1
        if success:
            self.stats.successes += 1
        else:
            self.stats.failures += 1

        if parent_count == 2:
            self.stats.parent_count_2 += 1
            if success:
                self.stats.parent_count_2_successes += 1
        elif parent_count == 3:
            self.stats.parent_count_3 += 1
            if success:
                self.stats.parent_count_3_successes += 1
        elif parent_count == 1:
            self.stats.self_crossover += 1
            if success:
                self.stats.self_crossover_successes += 1

        if generation_time > 0:
            total_time = self.stats.avg_generation_time * (self.stats.attempts - 1) + generation_time
            self.stats.avg_generation_time = total_time / self.stats.attempts

        self.recent_attempts.append(attempt)
        if len(self.recent_attempts) > self.max_recent_history:
            self.recent_attempts.pop(0)

        if len(self.recent_attempts) >= 5:
            recent_successes = sum(1 for a in self.recent_attempts[-20:] if a.success)
            recent_total = min(20, len(self.recent_attempts))
            self.stats.recent_success_rate = recent_successes / recent_total

    def get_crossover_stats(self) -> Dict[str, Any]:
        """Get comprehensive crossover statistics for monitoring"""
        return {
            "total_attempts": self.stats.attempts,
            "total_successes": self.stats.successes,
            "success_rate": self.stats.success_rate(),
            "recent_success_rate": self.stats.recent_success_rate,
            "per_type": {
                "2_parent": {
                    "attempts": self.stats.parent_count_2,
                    "successes": self.stats.parent_count_2_successes,
                    "success_rate": self.stats.success_rate_2p()
                },
                "3_parent": {
                    "attempts": self.stats.parent_count_3,
                    "successes": self.stats.parent_count_3_successes,
                    "success_rate": self.stats.success_rate_3p()
                },
                "self_crossover": {
                    "attempts": self.stats.self_crossover,
                    "successes": self.stats.self_crossover_successes,
                    "success_rate": self.stats.success_rate_self()
                }
            },
            "avg_generation_time_ms": self.stats.avg_generation_time * 1000,
            "recent_attempts": len(self.recent_attempts)
        }

    def get_monitor_summary(self) -> str:
        """Get one-line summary for monitoring display"""
        if self.stats.attempts == 0:
            return "Crossover: No attempts yet"

        # Build per-type breakdown with success rates
        type_parts = []
        if self.stats.parent_count_2 > 0:
            type_parts.append(f"2p: {self.stats.parent_count_2}@{self.stats.success_rate_2p():.0%}")
        if self.stats.parent_count_3 > 0:
            type_parts.append(f"3p: {self.stats.parent_count_3}@{self.stats.success_rate_3p():.0%}")
        if self.stats.self_crossover > 0:
            type_parts.append(f"self: {self.stats.self_crossover}@{self.stats.success_rate_self():.0%}")

        type_str = ", ".join(type_parts) if type_parts else "no types"

        return (f"Crossover: {self.stats.attempts} total, "
                f"{self.stats.success_rate():.1%} overall, "
                f"recent: {self.stats.recent_success_rate:.1%} | "
                f"{type_str}")

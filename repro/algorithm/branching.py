"""Branching with pruning for candidate exploration."""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from repro.algorithm.models import AttackCandidate, Task, Environment

logger = logging.getLogger(__name__)


@dataclass
class BranchingConfig:
    """Configuration for branching and pruning."""
    branch_factor: float = 3.0
    keep_ratio: float = 0.4
    min_keep: int = 2
    max_keep: int = 8
    num_quick_tasks: int = 2
    quick_timeout: int = 30
    enabled: bool = True


class BranchingPruner:
    """Generate more candidates, quick-eval, and keep only the best."""

    def __init__(self, config: Optional[BranchingConfig] = None):
        self.config = config or BranchingConfig()
        self.stats = {
            "total_generated": 0,
            "total_pruned": 0,
            "total_kept": 0,
            "pruning_sessions": 0,
            "avg_top_score": 0.0,
            "avg_bottom_score": 0.0,
        }

    def get_branching_k(self, base_k: int, current_asr: float = 0.0) -> int:
        """
        Calculate how many candidates to generate before pruning.

        Args:
            base_k: Base number of children to generate
            current_asr: Current best ASR (0.0-1.0) to adjust branching factor

        Returns:
            Number of candidates to generate (may be > base_k for pruning)
        """
        if not self.config.enabled:
            return base_k

        if current_asr >= 0.70:
            factor = 1.2
        elif current_asr >= 0.40:
            factor = 1.5
        else:
            factor = self.config.branch_factor

        branching_k = int(base_k * factor)

        if factor != self.config.branch_factor:
            logger.debug(f"Dynamic branching: ASR={current_asr:.1%} → factor={factor:.1f}x ({base_k}→{branching_k})")

        return branching_k

    def get_keep_count(self, num_candidates: int, base_k: int) -> int:
        """Calculate how many candidates to keep after pruning."""
        if not self.config.enabled:
            return num_candidates

        keep = int(num_candidates * self.config.keep_ratio)

        keep = max(self.config.min_keep, keep)
        keep = min(self.config.max_keep, keep)

        keep = min(keep, num_candidates)

        keep = max(keep, min(base_k, num_candidates))

        return keep

    async def prune_candidates(self,
                               candidates: List[AttackCandidate],
                               evaluator,
                               model,
                               env: Environment,
                               tasks: List[Task],
                               model_name: str,
                               base_k: int,
                               max_concurrent: int = 6,
                               rng: Optional["random.Random"] = None) -> List[AttackCandidate]:
        """Prune candidates using quick evaluation, return top performers."""
        if not self.config.enabled or len(candidates) <= base_k:
            return candidates

        self.stats["pruning_sessions"] += 1
        self.stats["total_generated"] += len(candidates)

        logger.info(f"Branching: Generated {len(candidates)} candidates, pruning to top ~{base_k}")

        try:
            scored = await evaluator.quick_eval_async(
                model=model,
                env=env,
                candidates=candidates,
                tasks=tasks,
                model_name=model_name,
                num_tasks=self.config.num_quick_tasks,
                timeout=self.config.quick_timeout,
                max_concurrent=max_concurrent,
                rng=rng,
            )

            if not scored:
                logger.warning("Quick eval returned empty results, keeping all candidates")
                return candidates

            keep_count = self.get_keep_count(len(scored), base_k)

            top_candidates = [c for c, _ in scored[:keep_count]]

            self.stats["total_kept"] += len(top_candidates)
            self.stats["total_pruned"] += len(candidates) - len(top_candidates)

            if scored:
                top_score = scored[0][1]
                bottom_score = scored[-1][1]
                n = self.stats["pruning_sessions"]
                self.stats["avg_top_score"] = (
                    (self.stats["avg_top_score"] * (n - 1) + top_score) / n
                )
                self.stats["avg_bottom_score"] = (
                    (self.stats["avg_bottom_score"] * (n - 1) + bottom_score) / n
                )

            logger.info(f"Pruning complete: Kept {len(top_candidates)}/{len(candidates)} "
                       f"(top score: {scored[0][1]:.2f}, cutoff: {scored[keep_count-1][1]:.2f})")

            return top_candidates

        except Exception as e:
            logger.error(f"Pruning failed: {e}. Falling back to random selection.")
            import random
            keep_count = self.get_keep_count(len(candidates), base_k)
            return random.sample(candidates, min(keep_count, len(candidates)))

    def get_statistics(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        prune_rate = (
            self.stats["total_pruned"] / max(1, self.stats["total_generated"])
        )
        return {
            **self.stats,
            "prune_rate": prune_rate,
            "config": {
                "branch_factor": self.config.branch_factor,
                "keep_ratio": self.config.keep_ratio,
                "enabled": self.config.enabled,
            }
        }

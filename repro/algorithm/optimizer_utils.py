"""REPRO Optimizer - UtilsMixin methods."""

import logging
from typing import List, Dict, Tuple, Optional, Set
from repro.algorithm.task_categories import categorize_task, TASK_CATEGORIES
from repro.algorithm.strategy_families import get_strategy_family
from repro.algorithm.models import AttackCandidate, Task

logger = logging.getLogger(__name__)


class UtilsMixin:
    """Mixin providing utils methods."""

    def _get_hard_task_strategies(self, task_diff_stats: dict) -> List[tuple]:
        """Get strategies that have successfully cracked hard/stubborn tasks.

        Returns list of tuples: [(strategy_name, success_count, [task_indices]), ...]
        """
        if not hasattr(self, 'task_targeting') or not task_diff_stats:
            return []

        hardest_tasks = task_diff_stats.get('hardest_tasks', [])
        if not hardest_tasks:
            return []

        hard_task_indices = {int(t[0]) for t in hardest_tasks[:10]}

        # Check task_targeting for strategy success on hard tasks
        strategy_hard_success = {}

        if hasattr(self.task_targeting, 'task_history'):
            for task_idx, history in self.task_targeting.task_history.items():
                if int(task_idx) in hard_task_indices:
                    for entry in history:
                        if entry.get('success', False):
                            strat = entry.get('strategy', 'unknown')
                            if strat not in strategy_hard_success:
                                strategy_hard_success[strat] = {'count': 0, 'tasks': set()}
                            strategy_hard_success[strat]['count'] += 1
                            strategy_hard_success[strat]['tasks'].add(int(task_idx))

        # Sort by success count and return top 5
        sorted_strats = sorted(
            strategy_hard_success.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:5]

        return [(name, data['count'], list(data['tasks'])) for name, data in sorted_strats]

    def _get_stubborn_task_details(self, task_diff_stats: dict) -> List[dict]:
        """Return top stubborn tasks with success stats and task text."""
        if not task_diff_stats or not hasattr(self, "task_difficulty_tracker"):
            return []

        stubborn_indices = task_diff_stats.get("stubborn_tasks", [])
        if not stubborn_indices:
            return []

        details = []
        for raw_idx in stubborn_indices:
            try:
                task_idx = int(raw_idx)
            except (TypeError, ValueError):
                continue

            attempts = self.task_difficulty_tracker.task_attempts.get(task_idx, 0)
            successes = self.task_difficulty_tracker.task_successes.get(task_idx, 0)
            rate = successes / attempts if attempts > 0 else 0.0

            task_text = ""
            if hasattr(self, "all_tasks") and self.all_tasks and 0 <= task_idx < len(self.all_tasks):
                task = self.all_tasks[task_idx]
                task_text = getattr(task, "user_content", "") or getattr(task, "goal", "") or getattr(task, "text", "")

            details.append(
                {
                    "task_idx": int(task_idx),
                    "task_text": task_text,
                    "successes": successes,
                    "attempts": attempts,
                    "success_rate": rate,
                }
            )

        details.sort(key=lambda x: (x["success_rate"], -x["attempts"]))
        return details[:5]

    def _get_elite_task_stats(self) -> Dict[str, int]:
        """Get task difficulty stats based on elite survivors only (for display)."""
        if not hasattr(self, 'elite_task_tracker'):
            return {}

        current_subset = list(self._curriculum_subset_indices) if self.curriculum_mode and self._curriculum_subset_indices else None
        return self.elite_task_tracker.get_statistics(current_subset=current_subset)

    def _get_task_category_counts(self) -> Dict[str, int]:
        """Get counts of tasks per category."""
        if not hasattr(self, 'all_tasks') or not self.all_tasks:
            return {}

        from collections import Counter
        category_counts = Counter()

        for task in self.all_tasks:
            task_text = getattr(task, 'user_content', '') or getattr(task, 'goal', '') or ''
            if task_text:
                cat_name, _, _ = categorize_task(task_text)
                category_counts[cat_name] += 1

        # Return as dict, excluding 'general' if there are other categories
        result = dict(category_counts)
        if len(result) > 1 and 'general' in result:
            # Keep general but put it last by not including it in main dict
            pass
        return result

    def _get_preferred_strategies_for_tasks(self, tasks: List[Task]) -> List[str]:
        """Derive preferred strategies from dominant task categories."""
        if not tasks:
            return []

        from collections import Counter

        category_counts = Counter()
        for task in tasks:
            task_text = getattr(task, 'user_content', '') or getattr(task, 'goal', '') or ''
            if task_text:
                cat_name, _, _ = categorize_task(task_text)
                category_counts[cat_name] += 1

        if not category_counts:
            return []

        categories = category_counts.most_common()
        non_general = [c for c in categories if c[0] != "general"]
        top_categories = (non_general or categories)[:2]

        preferred: List[str] = []
        for cat_name, _ in top_categories:
            cat_info = TASK_CATEGORIES.get(cat_name)
            if cat_info:
                preferred.extend(cat_info.preferred_strategies[:3])

        seen = set()
        deduped = []
        for strat in preferred:
            if strat not in seen:
                seen.add(strat)
                deduped.append(strat)
            if len(deduped) >= 6:
                break

        return deduped

    def _get_family_exhaustion_counts(self) -> Dict[str, int]:
        """Get counts of how many tasks each strategy family is exhausted for."""
        if not hasattr(self, 'task_family_explorer'):
            return {}

        stats = self.task_family_explorer.get_statistics()
        return stats.get('family_exhaustion_counts', {})

    def _calculate_diversity(self, population: List[AttackCandidate]) -> float:
        """Calculate population diversity using semantic embeddings (0=identical, 1=different)."""
        if len(population) < 2:
            return 0.0

        try:
            # Try to use sentence-transformers for semantic embeddings
            from sentence_transformers import SentenceTransformer
            import numpy as np

            # Use cached model if available, otherwise load
            if not hasattr(self, '_embedding_model'):
                try:
                    # Use lightweight model for speed (MiniLM is ~80MB, very fast)
                    self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Loaded sentence embedding model for semantic diversity")
                except Exception as e:
                    logger.warning(f"Failed to load embedding model: {e}. Falling back to Jaccard distance.")
                    self._embedding_model = None

            if self._embedding_model is not None:
                # Extract texts
                texts = [p.text for p in population]

                # Generate embeddings (batched for efficiency)
                embeddings = self._embedding_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

                # Normalize embeddings for cosine distance
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

                # Calculate pairwise cosine distances (sampled for efficiency)
                num_samples = min(50, len(population) * (len(population) - 1) // 2)
                total_distance = 0.0

                for _ in range(num_samples):
                    i, j = self.rng.sample(range(len(population)), 2)
                    # Cosine distance = 1 - cosine similarity
                    cosine_sim = np.dot(embeddings[i], embeddings[j])
                    cosine_dist = 1.0 - cosine_sim
                    total_distance += cosine_dist

                avg_distance = total_distance / num_samples if num_samples > 0 else 0.0

                logger.debug(f"Semantic diversity (cosine distance): {avg_distance:.3f}")
                return avg_distance

        except ImportError:
            logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
        except Exception as e:
            logger.warning(f"Semantic diversity calculation failed: {e}. Falling back to Jaccard.")

        # Fallback: Jaccard distance (word overlap)
        num_samples = min(50, len(population) * (len(population) - 1) // 2)
        total_distance = 0.0

        for _ in range(num_samples):
            i, j = self.rng.sample(range(len(population)), 2)
            words_i = set(population[i].text.lower().split())
            words_j = set(population[j].text.lower().split())
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            distance = 1.0 - (intersection / union if union > 0 else 0)
            total_distance += distance

        logger.debug(f"Lexical diversity (Jaccard distance): {total_distance / num_samples:.3f}")
        return total_distance / num_samples if num_samples > 0 else 0.0

    def _ensure_original_family(self, candidate: AttackCandidate, parent: Optional[AttackCandidate] = None) -> None:
        """Ensure original_family is set for diversity-aware parent selection."""
        if not hasattr(candidate, "meta") or candidate.meta is None:
            candidate.meta = {}
        if candidate.meta.get("original_family"):
            return
        if parent and getattr(parent, "meta", None) and parent.meta.get("original_family"):
            candidate.meta["original_family"] = parent.meta["original_family"]
            return
        strategy = (
            candidate.meta.get("edit")
            or candidate.meta.get("seed_strategy")
            or candidate.meta.get("optimizer_strategy_hint")
            or candidate.meta.get("strategy")
        )
        if strategy:
            candidate.meta["original_family"] = get_strategy_family(strategy)

    def _get_strategy_monitor_summary(self) -> str:
        """Get strategy selection summary for monitoring"""
        stats = self.strategy_optimizer.get_strategy_summary()

        # Get top 3 strategies by attempts
        top_strategies = sorted(stats.items(),
                              key=lambda x: x[1]["attempts"],
                              reverse=True)[:3]

        summary_parts = []
        for strategy, data in top_strategies:
            success_rate = data["success_rate"]
            summary_parts.append(f"{strategy}({data['attempts']},{success_rate:.1%})")

        return " | ".join(summary_parts)


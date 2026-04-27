"""Diversity preservation utilities for REPRO."""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from repro.algorithm.models import AttackCandidate
from repro.algorithm.utils import extract_task_results

logger = logging.getLogger(__name__)


@dataclass
class DiversityMetrics:
    """Metrics for tracking diversity preservation."""
    population_diversity: float = 0.0
    avg_crowding_distance: float = 0.0
    niche_count: int = 0
    diversity_injections: int = 0
    crowding_replacements: int = 0


class DiversityPreserver:
    """Manage diversity preservation across the population."""

    def __init__(self,
                 sharing_radius: float = 0.3,
                 alpha: float = 1.0,
                 diversity_weight: float = 0.35,
                 min_diversity_threshold: float = 0.55,
                 embedding_model=None,
                 task_coverage_min_ideal: float = 0.1,
                 task_coverage_max_ideal: float = 0.9,
                 task_coverage_scale: float = 10.0):
        """
        Args:
            sharing_radius: Radius for fitness sharing (sigma_share)
            alpha: Shape parameter for sharing function (typically 1.0)
            diversity_weight: Weight for diversity in selection (0-1), increased to 0.35
            min_diversity_threshold: Trigger diversity injection below this (0.55 for aggressive maintenance)
            embedding_model: Pre-loaded sentence transformer model (optional)
            task_coverage_min_ideal: Minimum ideal coverage rate for task diversity (default 0.1)
            task_coverage_max_ideal: Maximum ideal coverage rate for task diversity (default 0.9)
            task_coverage_scale: Scale factor for converting variance to diversity score (default 10.0)
        """
        self.sharing_radius = sharing_radius
        self.alpha = alpha
        self.diversity_weight = diversity_weight
        self.min_diversity_threshold = min_diversity_threshold
        self._embedding_model = embedding_model
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self.metrics = DiversityMetrics()
        self.task_coverage_min_ideal = task_coverage_min_ideal
        self.task_coverage_max_ideal = task_coverage_max_ideal
        self.task_coverage_scale = task_coverage_scale
        self.base_diversity_weight = diversity_weight

    def set_diversity_weight_boost(self, factor: float):
        """Boost diversity weight during stagnation."""
        self.diversity_weight = min(0.9, self.base_diversity_weight * factor)
        logger.info(f"Diversity weight boosted to {self.diversity_weight:.2f} (factor={factor})")

    def reset_diversity_weight(self):
        """Reset diversity weight to base value."""
        if self.diversity_weight != self.base_diversity_weight:
            self.diversity_weight = self.base_diversity_weight
            logger.info(f"Diversity weight reset to {self.diversity_weight:.2f}")

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded embedding model for diversity preservation")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self._embedding_model = False  # Mark as failed
        return self._embedding_model if self._embedding_model else None

    def _get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embeddings for texts, using cache when possible."""
        model = self._get_embedding_model()
        if model is None:
            return None

        uncached_texts = []
        uncached_indices = []
        for i, text in enumerate(texts):
            text_key = text[:500]  # Use first 500 chars as key
            if text_key not in self._embeddings_cache:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            new_embeddings = model.encode(uncached_texts, show_progress_bar=False, convert_to_numpy=True)
            for text, idx, emb in zip(uncached_texts, uncached_indices, new_embeddings):
                self._embeddings_cache[text[:500]] = emb

        embeddings = np.array([self._embeddings_cache[t[:500]] for t in texts])

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms > 0, norms, 1)

        return embeddings

    def _compute_distance_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine distances."""
        similarity = np.dot(embeddings, embeddings.T)
        distance = 1 - similarity
        return distance

    def compute_pairwise_distance(self, cand1: AttackCandidate, cand2: AttackCandidate) -> float:
        """
        Compute cosine distance between two candidates.

        Returns:
            float: Distance in [0, 2], where 0 = identical, 1 = orthogonal
        """
        embeddings = self._get_embeddings([cand1.text, cand2.text])
        if embeddings is None:
            # Fallback: Jaccard distance on word sets
            words1 = set(cand1.text.lower().split())
            words2 = set(cand2.text.lower().split())
            if not words1 and not words2:
                return 0.0
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return 1.0 - (intersection / union) if union > 0 else 1.0

        dist_matrix = self._compute_distance_matrix(embeddings)
        return float(dist_matrix[0, 1])

    # FITNESS SHARING

    def compute_sharing_function(self, distance: float) -> float:
        """
        Sharing function sh(d) from Goldberg & Richardson (1987).

        sh(d) = 1 - (d/sigma_share)^alpha  if d < sigma_share
        sh(d) = 0                          otherwise
        """
        if distance >= self.sharing_radius:
            return 0.0
        return 1.0 - (distance / self.sharing_radius) ** self.alpha

    def compute_niche_counts(self, population: List[AttackCandidate]) -> Dict[str, float]:
        """
        Compute niche count for each individual.

        Niche count m_i = sum_j(sh(d_ij)) measures how crowded the region is.
        Higher niche count = more crowded = fitness should be reduced.
        """
        if len(population) < 2:
            return {p.id: 1.0 for p in population}

        texts = [p.text for p in population]
        embeddings = self._get_embeddings(texts)

        if embeddings is None:
            # Fallback: use Jaccard distance
            return self._compute_niche_counts_jaccard(population)

        distances = self._compute_distance_matrix(embeddings)

        niche_counts = {}
        for i, candidate in enumerate(population):
            niche_count = 0.0
            for j in range(len(population)):
                niche_count += self.compute_sharing_function(distances[i, j])
            niche_counts[candidate.id] = max(niche_count, 1.0)  # At least 1 (self)

        self.metrics.niche_count = len(set(int(nc) for nc in niche_counts.values()))
        return niche_counts

    def _compute_niche_counts_jaccard(self, population: List[AttackCandidate]) -> Dict[str, float]:
        """Fallback niche count using Jaccard distance."""
        niche_counts = {}

        # Tokenize all texts
        token_sets = [set(p.text.lower().split()) for p in population]

        for i, candidate in enumerate(population):
            niche_count = 0.0
            for j in range(len(population)):
                # Jaccard distance
                intersection = len(token_sets[i] & token_sets[j])
                union = len(token_sets[i] | token_sets[j])
                jaccard_sim = intersection / union if union > 0 else 0
                distance = 1 - jaccard_sim
                niche_count += self.compute_sharing_function(distance)
            niche_counts[candidate.id] = max(niche_count, 1.0)

        return niche_counts

    def compute_shared_fitness(self,
                               fitness_scores: Dict[str, float],
                               niche_counts: Dict[str, float]) -> Dict[str, float]:
        """
        Compute shared fitness: f'_i = f_i / m_i

        Individuals in crowded regions have their fitness reduced.
        """
        shared_fitness = {}
        for candidate_id, fitness in fitness_scores.items():
            niche_count = niche_counts.get(candidate_id, 1.0)
            shared_fitness[candidate_id] = fitness / niche_count
        return shared_fitness

    # CROWDING DISTANCE (NSGA-II style)

    def compute_crowding_distances(self,
                                  population: List[AttackCandidate],
                                  metrics: List[Dict[str, float]],
                                  objectives: List[str]) -> Dict[str, float]:
        """
        Compute crowding distance for each individual (NSGA-II).

        Crowding distance measures how isolated a solution is.
        Higher = more isolated = more valuable for diversity.

        Boundary solutions get infinite distance (always preserved).
        """
        n = len(population)
        if n < 3:
            return {p.id: float('inf') for p in population}

        crowding = {p.id: 0.0 for p in population}

        for obj in objectives:
            # Sort by this objective
            sorted_indices = sorted(range(n), key=lambda i: metrics[i].get(obj, 0))

            # Boundary points get infinite distance
            crowding[population[sorted_indices[0]].id] = float('inf')
            crowding[population[sorted_indices[-1]].id] = float('inf')

            # Get objective range
            obj_min = metrics[sorted_indices[0]].get(obj, 0)
            obj_max = metrics[sorted_indices[-1]].get(obj, 0)
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            # Interior points: distance based on neighbors
            for i in range(1, n - 1):
                prev_val = metrics[sorted_indices[i - 1]].get(obj, 0)
                next_val = metrics[sorted_indices[i + 1]].get(obj, 0)
                crowding[population[sorted_indices[i]].id] += (next_val - prev_val) / obj_range

        self.metrics.avg_crowding_distance = np.mean([v for v in crowding.values() if v != float('inf')])
        return crowding

    def select_survivors_crowding_replacement(self,
                                              population: List[AttackCandidate],
                                              metrics: List[Dict[str, float]],
                                              objectives: List[str],
                                              fitness_scores: Dict[str, float],
                                              target_size: int,
                                              rng,
                                              protect_frac: float = 0.25) -> List[AttackCandidate]:
        """
        Select survivors using crowding distance, protecting top-fitness candidates.

        This implements crowding-based replacement: when overfull, remove the most
        crowded individuals to preserve coverage in objective space.
        """
        if target_size <= 0:
            return []
        if len(population) <= target_size:
            return list(population)

        protect_frac = max(0.0, min(1.0, protect_frac))
        if target_size == 1:
            protect_count = 1
        else:
            protect_count = max(1, int(round(target_size * protect_frac)))
            protect_count = min(protect_count, target_size)

        shuffled = list(population)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda c: fitness_scores.get(c.id, 0.0), reverse=True)
        protected = shuffled[:protect_count]
        protected_ids = {c.id for c in protected}

        crowding = self.compute_crowding_distances(population, metrics, objectives)
        remaining = [c for c in population if c.id not in protected_ids]
        rng.shuffle(remaining)
        remaining.sort(
            key=lambda c: (crowding.get(c.id, 0.0), fitness_scores.get(c.id, 0.0)),
            reverse=True
        )

        remaining_needed = max(0, target_size - len(protected))
        survivors = protected + remaining[:remaining_needed]

        replacements = len(population) - target_size
        if replacements > 0:
            self.metrics.crowding_replacements += replacements

        return survivors

    # CROWDING-BASED REPLACEMENT

    # DIVERSITY-AWARE SELECTION

    def compute_selection_scores(self,
                                 population: List[AttackCandidate],
                                 fitness_scores: Dict[str, float],
                                 metrics: List[Dict[str, float]],
                                 objectives: List[str],
                                 elite_task_results: Optional[List[Dict[int, bool]]] = None,
                                 task_coverage_weight: float = 0.3) -> Dict[str, float]:
        """
        Compute combined selection scores: fitness + diversity + task coverage.

        Score_i = (1 - w) * shared_fitness_i
                 + w * (1-tc_w) * crowding_distance_i
                 + w * tc_w * task_coverage_diversity_i

        This balances exploitation (fitness) with exploration (semantic diversity + task coverage diversity).

        Args:
            task_coverage_weight: Weight for task coverage within diversity component [0, 1]
                                 0 = only semantic diversity, 1 = only task coverage
        """
        # Get niche counts for fitness sharing
        niche_counts = self.compute_niche_counts(population)
        shared_fitness = self.compute_shared_fitness(fitness_scores, niche_counts)

        # Get crowding distances
        crowding = self.compute_crowding_distances(population, metrics, objectives)

        # Get task coverage diversity scores
        task_coverage_scores = {}
        if elite_task_results and task_coverage_weight > 0:
            for p in population:
                candidate_task_results = extract_task_results(p)
                task_coverage_scores[p.id] = self.compute_task_coverage_diversity(
                    candidate_task_results, elite_task_results
                )
        else:
            # No task coverage - set all to 0
            task_coverage_scores = {p.id: 0.0 for p in population}

        # Normalize all components to [0, 1]
        # When all values are equal, use 0.5 (middle value) for fairness

        # 1. Shared fitness
        max_shared = max(shared_fitness.values()) if shared_fitness else 1
        min_shared = min(shared_fitness.values()) if shared_fitness else 0
        shared_range = max_shared - min_shared
        shared_all_equal = shared_range == 0

        # 2. Crowding distance
        finite_crowding = [v for v in crowding.values() if v != float('inf')]
        max_crowd = max(finite_crowding) if finite_crowding else 1
        min_crowd = min(finite_crowding) if finite_crowding else 0
        crowd_range = max_crowd - min_crowd
        crowd_all_equal = crowd_range == 0

        # 3. Task coverage (already in [0, 1])

        # Compute combined scores
        selection_scores = {}
        for p in population:
            # Normalized shared fitness: use 0.5 when all equal
            if shared_all_equal:
                norm_fitness = 0.5
            else:
                norm_fitness = (shared_fitness[p.id] - min_shared) / shared_range

            # Normalized crowding (infinity -> 1.0, all-equal -> 0.5)
            if crowding[p.id] == float('inf'):
                norm_crowd = 1.0
            elif crowd_all_equal:
                norm_crowd = 0.5
            else:
                norm_crowd = (crowding[p.id] - min_crowd) / crowd_range

            # Task coverage diversity (already normalized)
            norm_task_coverage = task_coverage_scores[p.id]

            # Combined diversity score: blend semantic and task coverage
            diversity_score = (1 - task_coverage_weight) * norm_crowd + task_coverage_weight * norm_task_coverage

            # Final selection score
            selection_scores[p.id] = (1 - self.diversity_weight) * norm_fitness + self.diversity_weight * diversity_score

        return selection_scores

    def select_parents_diversity_aware(self,
                                       population: List[AttackCandidate],
                                       pop_metrics: List[Dict[str, float]],
                                       fitness_scores: Dict[str, float],
                                       objectives: List[str],
                                       num_parents: int,
                                       rng) -> List[Tuple[AttackCandidate, Dict[str, float]]]:
        """
        Select parents using diversity-aware tournament selection.

        Uses combined fitness + diversity score for selection pressure.
        """
        if len(population) <= num_parents:
            return list(zip(population, pop_metrics))

        # Compute selection scores
        selection_scores = self.compute_selection_scores(
            population, fitness_scores, pop_metrics, objectives
        )

        # Tournament selection
        selected = []
        available_indices = list(range(len(population)))

        for _ in range(num_parents):
            if not available_indices:
                break

            # Tournament size (typically 2-3)
            tournament_size = min(3, len(available_indices))
            tournament = rng.sample(available_indices, tournament_size)

            # Select best from tournament based on combined score
            best_idx = max(tournament, key=lambda i: selection_scores[population[i].id])

            selected.append((population[best_idx], pop_metrics[best_idx]))
            available_indices.remove(best_idx)

        return selected

    # DIVERSITY INJECTION

    def generate_diverse_variants(self,
                                  population: List[AttackCandidate],
                                  num_variants: int,
                                  rng) -> List[AttackCandidate]:
        """
        Generate diverse variants to inject into population.

        Strategies:
        1. Extreme mutations of existing individuals
        2. Recombination of most different pairs
        3. Strategy-diverse selection from history
        """
        if not population:
            return []

        variants = []

        # Strategy 1: Find most different pairs and recombine
        if len(population) >= 2:
            texts = [p.text for p in population]
            embeddings = self._get_embeddings(texts)

            if embeddings is not None:
                distances = self._compute_distance_matrix(embeddings)

                # Find most different pairs
                n = len(population)
                pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        pairs.append((i, j, distances[i, j]))

                pairs.sort(key=lambda x: x[2], reverse=True)  # Most different first

                # Recombine top pairs
                for i, j, _ in pairs[:num_variants]:
                    p1, p2 = population[i], population[j]

                    # Simple recombination: alternate sentences
                    import re
                    s1 = re.split(r'(?<=[.!?])\s+', p1.text)
                    s2 = re.split(r'(?<=[.!?])\s+', p2.text)

                    combined = []
                    for k in range(max(len(s1), len(s2))):
                        if k < len(s1) and rng.random() < 0.5:
                            combined.append(s1[k])
                        if k < len(s2) and rng.random() < 0.5:
                            combined.append(s2[k])

                    if combined:
                        new_text = " ".join(combined)
                        variant = AttackCandidate(
                            text=new_text,
                            parent_id=p1.id,
                            goal=p1.goal,
                            meta={"edit": "diversity_injection", "source": "recombination"}
                        )
                        variants.append(variant)

                    if len(variants) >= num_variants:
                        break

        self.metrics.diversity_injections += len(variants)
        logger.info(f"Generated {len(variants)} diverse variants for injection")
        return variants

    # TASK COVERAGE DIVERSITY

    def compute_task_coverage_diversity(self,
                                       candidate_task_results: Dict[int, bool],
                                       elite_task_results: List[Dict[int, bool]]) -> float:
        """
        Compute task coverage diversity score for a candidate.

        Rewards attacks that succeed on tasks where current elites fail.
        This encourages maintaining specialists for different task subsets.

        Args:
            candidate_task_results: {task_idx: success_bool} for the candidate
            elite_task_results: List of {task_idx: success_bool} for each elite

        Returns:
            float: Score in [0, 1] where higher = more task-diverse
                  0 = covers no new tasks beyond elites
                  1 = covers all tasks that elites fail
        """
        if not candidate_task_results:
            return 0.0

        if not elite_task_results:
            # No elites to compare against - reward any successful tasks
            return sum(1 for success in candidate_task_results.values() if success) / max(1, len(candidate_task_results))

        # Find tasks that NO elite can crack
        all_task_indices = set(candidate_task_results.keys())
        for elite_results in elite_task_results:
            all_task_indices.update(elite_results.keys())

        uncovered_tasks = set()
        for task_idx in all_task_indices:
            # Task is uncovered if no elite succeeds on it
            elite_successes = [elite_results.get(task_idx, False) for elite_results in elite_task_results]
            if not any(elite_successes):
                uncovered_tasks.add(task_idx)

        if not uncovered_tasks:
            # All tasks are covered by elites - score by overall performance
            return sum(1 for success in candidate_task_results.values() if success) / max(1, len(candidate_task_results))

        # Count how many uncovered tasks this candidate can crack
        candidate_uncovered_successes = sum(1 for task_idx in uncovered_tasks
                                          if candidate_task_results.get(task_idx, False))

        # Normalize by total uncovered tasks
        coverage_score = candidate_uncovered_successes / len(uncovered_tasks)

        return coverage_score


# DIVERSITY RESTART MANAGER

class DiversityRestartManager:
    """
    Manages diversity restart after finding universal/mastered prompts.

    When a universal prompt is found, this class helps generate a fresh
    population that is diverse from already-found solutions, while preserving
    learned knowledge (strategy weights, failure patterns).

    Key features:
    1. Filters new seeds by similarity to found universals
    2. Uses embeddings for accurate semantic similarity (with Jaccard fallback)
    3. Prioritizes seeds from high-performing strategy families
    4. Tracks restart history for monitoring
    """

    def __init__(self,
                 similarity_threshold: float = 0.6,
                 use_embeddings: bool = True):
        """
        Args:
            similarity_threshold: Max similarity allowed (0-1). Seeds more similar
                                  than this to any found universal are filtered.
            use_embeddings: Whether to use sentence embeddings (more accurate)
                           or fall back to Jaccard similarity (faster).
        """
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings

        # Reuse DiversityPreserver for embedding computation
        self._diversity_preserver = DiversityPreserver() if use_embeddings else None

        # Track found universals for filtering
        self.found_universal_texts: List[str] = []

        # Restart history for monitoring
        self.restart_history: List[Dict] = []

    def add_found_universal(self, text: str) -> None:
        """Register a newly found universal prompt for diversity filtering."""
        if text not in self.found_universal_texts:
            self.found_universal_texts.append(text)
            logger.info(f"DiversityRestartManager: Registered universal #{len(self.found_universal_texts)}")

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.

        Uses embeddings if available, otherwise falls back to Jaccard.

        Returns:
            float: Similarity score in [0, 1] where 1 = identical
        """
        if self._diversity_preserver is not None:
            try:
                # Use DiversityPreserver's embedding-based distance
                # Distance is in [0, 2], convert to similarity
                distance = self._diversity_preserver.compute_pairwise_distance(
                    AttackCandidate(text=text1, goal=""),
                    AttackCandidate(text=text2, goal="")
                )
                # Convert distance to similarity: sim = 1 - distance
                # (distance 0 = sim 1, distance 1 = sim 0)
                return max(0.0, 1.0 - distance)
            except Exception as e:
                logger.debug(f"Embedding similarity failed, using Jaccard: {e}")

        # Jaccard fallback
        return self._jaccard_similarity(text1, text2)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity on word sets."""
        import re
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def is_diverse_from_universals(self, text: str) -> bool:
        """
        Check if a text is sufficiently diverse from all found universals.

        Args:
            text: The candidate text to check

        Returns:
            True if the text is diverse enough (similarity < threshold)
        """
        for universal_text in self.found_universal_texts:
            similarity = self.compute_similarity(text, universal_text)
            if similarity > self.similarity_threshold:
                logger.debug(f"Filtered: similarity={similarity:.2f} > {self.similarity_threshold}")
                return False
        return True

    def filter_seeds_for_diversity(self,
                                   seeds: List[Tuple[str, str]],
                                   target_count: int) -> List[Tuple[str, str]]:
        """
        Filter seeds to keep only those diverse from found universals.

        Args:
            seeds: List of (strategy_name, seed_text) tuples
            target_count: Target number of seeds to return

        Returns:
            Filtered list of (strategy_name, seed_text) tuples
        """
        if not self.found_universal_texts:
            # No universals to filter against
            return seeds[:target_count]

        diverse_seeds = []
        filtered_count = 0

        for strategy_name, seed_text in seeds:
            if len(diverse_seeds) >= target_count:
                break

            if self.is_diverse_from_universals(seed_text):
                diverse_seeds.append((strategy_name, seed_text))
            else:
                filtered_count += 1

        logger.info(f"Diversity filter: kept {len(diverse_seeds)}, filtered {filtered_count}")
        return diverse_seeds

    def generate_diverse_population(self,
                                    population_size: int,
                                    seed_goal: str,
                                    optimize_mode: str = "explore",
                                    rng=None) -> List['AttackCandidate']:
        """
        Generate a fresh diverse population for restart.

        Args:
            population_size: Target population size
            seed_goal: Goal text for the attack candidates
            optimize_mode: "explore" or "exploit"
            rng: Random number generator

        Returns:
            List of diverse AttackCandidates
        """
        from repro.prompts.seed_strategies import (
            get_stratified_explore_seeds,
            get_stratified_exploit_seeds,
            get_random_explore_seed,
            get_random_exploit_seed,
        )

        logger.info(f"Generating diverse population (size={population_size}, "
                   f"excluding {len(self.found_universal_texts)} patterns)")

        # Generate more seeds than needed for filtering
        extra_factor = 3 if self.found_universal_texts else 1
        target_seeds = population_size * extra_factor

        try:
            if optimize_mode == "exploit":
                stratified_seeds = get_stratified_exploit_seeds(
                    n_seeds=target_seeds,
                    goal=seed_goal,
                    rng=rng
                )
            else:
                stratified_seeds = get_stratified_explore_seeds(
                    n_seeds=target_seeds,
                    rng=rng
                )
        except Exception as e:
            logger.warning(f"Stratified seed generation failed: {e}")
            stratified_seeds = []
            for _ in range(target_seeds):
                if optimize_mode == "exploit":
                    strat_name, seed_text = get_random_exploit_seed(goal=seed_goal, rng=rng)
                else:
                    strat_name, seed_text = get_random_explore_seed(rng=rng)
                stratified_seeds.append((strat_name, seed_text))

        # Filter for diversity
        diverse_seeds = self.filter_seeds_for_diversity(stratified_seeds, population_size)

        # Create candidates
        candidates = []
        seen_signatures = set()

        for strategy_name, seed_text in diverse_seeds:
            try:
                candidate = AttackCandidate(
                    text=seed_text,
                    goal=seed_goal,
                    meta={
                        "edit": f"diversity_restart:{strategy_name}",
                        "origin": "diversity_restart",
                        "family": strategy_name,
                    }
                )
                sig = candidate.signature()
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    candidates.append(candidate)
            except Exception as e:
                logger.debug(f"Failed to create candidate: {e}")

        # If we didn't get enough, generate more with relaxed filtering
        if len(candidates) < population_size:
            logger.warning(f"Only {len(candidates)} diverse candidates, "
                          f"generating {population_size - len(candidates)} more")
            attempts = 0
            max_attempts = population_size * 5

            while len(candidates) < population_size and attempts < max_attempts:
                attempts += 1
                try:
                    if optimize_mode == "exploit":
                        strat_name, seed_text = get_random_exploit_seed(goal=seed_goal, rng=rng)
                    else:
                        strat_name, seed_text = get_random_explore_seed(rng=rng)

                    candidate = AttackCandidate(
                        text=seed_text,
                        goal=seed_goal,
                        meta={"edit": f"diversity_fallback:{strat_name}", "origin": "diversity_restart"}
                    )
                    sig = candidate.signature()
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        candidates.append(candidate)
                except Exception:
                    continue

        # Record restart history
        self.restart_history.append({
            "restart_number": len(self.restart_history) + 1,
            "universals_count": len(self.found_universal_texts),
            "candidates_generated": len(candidates),
            "target_size": population_size,
        })

        logger.info(f"Generated {len(candidates)} diverse candidates")
        return candidates

    def get_restart_summary(self) -> str:
        """Get a summary of restart history for monitoring."""
        if not self.restart_history:
            return "No diversity restarts performed"

        total_restarts = len(self.restart_history)
        total_candidates = sum(r["candidates_generated"] for r in self.restart_history)
        return f"Diversity restarts: {total_restarts}, total candidates: {total_candidates}"

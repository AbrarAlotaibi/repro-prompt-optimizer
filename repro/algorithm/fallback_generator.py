"""Fallback candidate generation when LLM generation fails or is insufficient."""

import logging
import random
from typing import List, Set, Optional

from repro.algorithm.models import AttackCandidate

logger = logging.getLogger(__name__)


class FallbackGenerator:
    """Generates fallback candidates from dataset and code-based mutations."""

    def __init__(self,
                 jailbreak_examples: List[dict],
                 semantic_drift_mutator,
                 word_level_mutator,
                 rng: random.Random):
        self.jailbreak_examples = jailbreak_examples
        self.semantic_drift_mutator = semantic_drift_mutator
        self.word_level_mutator = word_level_mutator
        self.rng = rng

    def fill_from_dataset(self,
                          attack: AttackCandidate,
                          needed: int,
                          seen_signatures: Set[str]) -> List[AttackCandidate]:
        """Fill missing slots with goal-relevant dataset examples."""
        if not self.jailbreak_examples or needed <= 0:
            return []

        candidates = []
        goal_keywords = self._extract_goal_keywords(attack.goal)

        # Categorize examples
        unused_examples = []
        goal_relevant_examples = []

        for example in self.jailbreak_examples:
            temp = AttackCandidate(text=example['text'], parent_id=attack.id, meta={"edit": "temp"})
            if temp.signature() in seen_signatures:
                continue

            unused_examples.append(example)
            if goal_keywords and any(kw in example['text'].lower() for kw in goal_keywords):
                goal_relevant_examples.append(example)

        # Prioritize goal-relevant, fallback to any unused
        pool = goal_relevant_examples if goal_relevant_examples else unused_examples

        if pool:
            num_to_sample = min(needed, len(pool))
            sampled = self.rng.sample(pool, num_to_sample)

            for example in sampled:
                candidate = AttackCandidate(
                    text=example['text'],
                    parent_id=attack.id,
                    goal=attack.goal,
                    meta={
                        "edit": "dataset_fallback",
                        "reasoning": f"Dataset example (strategy: {example['strategy']})",
                        "dataset_source": example.get('source', 'unknown'),
                        "dataset_strategy": example['strategy']
                    }
                )
                if candidate.signature() not in seen_signatures:
                    candidates.append(candidate)
                    seen_signatures.add(candidate.signature())

            logger.info(f"Added {len(candidates)} dataset examples")

        return candidates

    def fill_from_mutations(self,
                            attack: AttackCandidate,
                            needed: int,
                            seen_signatures: Set[str]) -> List[AttackCandidate]:
        """Fill missing slots with code-based mutations."""
        if needed <= 0 or not attack.text:
            return []

        candidates = []

        # Semantic drift variants (up to 3)
        drift_count = min(needed, 3)
        drift_strategies = ['invisible', 'visible', 'hybrid']

        for i in range(drift_count):
            strategy = drift_strategies[i % 3]

            if strategy == 'invisible':
                mutated = self.semantic_drift_mutator.apply_invisible_drift(attack.text)
                edit_name = "semantic_drift_invisible"
            elif strategy == 'visible':
                mutated = self.semantic_drift_mutator.apply_visible_drift(attack.text)
                edit_name = "semantic_drift_visible"
            else:
                mutated = self.semantic_drift_mutator.apply_hybrid_drift(attack.text)
                edit_name = "semantic_drift_hybrid"

            candidate = AttackCandidate(
                text=mutated,
                parent_id=attack.id,
                goal=attack.goal,
                meta={"edit": edit_name, "code_based_mutation": True}
            )
            if candidate.signature() not in seen_signatures:
                candidates.append(candidate)
                seen_signatures.add(candidate.signature())

        # Word-level variants (up to 3 more)
        still_needed = needed - len(candidates)
        if still_needed > 0:
            word_count = min(still_needed, 3)
            word_strategies = [
                ('synonym_substitution', lambda t: self.word_level_mutator.apply_synonym_substitution(t, rate=0.3)),
                ('sentence_reordering', lambda t: self.word_level_mutator.apply_sentence_reordering(t)),
                ('combined_mutation', lambda t: self.word_level_mutator.apply_combined_mutation(t, synonym_rate=0.2, insertion_rate=0.1)),
            ]

            for i in range(word_count):
                name, mutate_fn = word_strategies[i % 3]
                try:
                    mutated = mutate_fn(attack.text)
                    candidate = AttackCandidate(
                        text=mutated,
                        parent_id=attack.id,
                        goal=attack.goal,
                        meta={"edit": name, "code_based_mutation": True}
                    )
                    if candidate.signature() not in seen_signatures:
                        candidates.append(candidate)
                        seen_signatures.add(candidate.signature())
                except Exception as e:
                    logger.debug(f"Word mutation {name} failed: {e}")

        if candidates:
            logger.info(f"Added {len(candidates)} code-based mutation variants")

        return candidates

    def _extract_goal_keywords(self, goal: Optional[str]) -> Set[str]:
        """Extract keywords from goal for relevance filtering."""
        if not goal:
            return set()

        common_words = {'a', 'an', 'the', 'to', 'for', 'of', 'on', 'in', 'how', 'write', 'create', 'develop', 'make', 'build'}
        words = goal.lower().split()
        return {w for w in words if len(w) > 3 and w not in common_words}

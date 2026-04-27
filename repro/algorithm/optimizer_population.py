"""REPRO Optimizer - PopulationMixin methods."""

import logging
from typing import List, Dict, Tuple, Optional, Set
from repro.algorithm.models import AttackCandidate, Task
from repro.prompts.seed_strategies import (
    get_random_exploit_seed, get_random_explore_seed,
    get_stratified_exploit_seeds, get_stratified_explore_seeds,
    get_category_seeds,
)

logger = logging.getLogger(__name__)


class PopulationMixin:
    """Mixin providing population methods."""

    def _generate_diverse_fillers(self,
                                   seed: AttackCandidate,
                                   needed: int,
                                   seen: set,
                                   scored: List[Tuple],
                                   children: List[AttackCandidate]) -> List[AttackCandidate]:
        """
        Generate diverse filler candidates instead of reusing existing population.

        Uses mutation strategies to create novel variations when children < needed.
        Falls back to existing candidates only as a last resort.

        Args:
            seed: Original seed attack for mutations
            needed: Number of additional candidates needed
            seen: Set of already-seen signatures
            scored: Current scored population [(candidate, metrics), ...]
            children: Already generated children

        Returns:
            List of diverse filler candidates
        """
        fillers = []
        use_similarity = self.dedup_config.get("use_similarity", True)
        similarity_threshold = self.dedup_config.get("similarity_threshold", 0.80)

        logger.info(f"Generating {needed} diverse fillers (vs reusing existing candidates)")

        # Get best performing candidates for mutation base
        best_candidates = [c for c, _ in sorted(scored, key=lambda x: (x[1]["ASR"], x[1]["XFER"]), reverse=True)[:5]]
        if not best_candidates:
            best_candidates = [seed]

        # Phase 1: Semantic drift mutations
        try:
            from repro.algorithm.semantic_drift import SemanticDriftMutator
            mutator = SemanticDriftMutator(rng=self.rng)
            drift_strategies = ['invisible', 'visible', 'hybrid']

            for base_candidate in best_candidates:
                if len(fillers) >= needed:
                    break
                for strategy in drift_strategies:
                    if len(fillers) >= needed:
                        break
                    try:
                        if strategy == 'invisible':
                            mutated_text = mutator.apply_invisible_drift(base_candidate.text)
                        elif strategy == 'visible':
                            mutated_text = mutator.apply_visible_drift(base_candidate.text)
                        else:
                            mutated_text = mutator.apply_hybrid_drift(base_candidate.text)

                        filler = AttackCandidate(
                            text=mutated_text,
                            parent_id=base_candidate.id,
                            goal=base_candidate.goal or seed.goal,
                            meta={"edit": f"semantic_drift_{strategy}", "filler": True}
                        )
                        self._ensure_original_family(filler, parent=base_candidate)

                        # Check for duplicates
                        is_dup = filler.signature() in seen
                        if not is_dup and use_similarity and children:
                            is_dup = filler.similarity_to_any(children[-20:], threshold=similarity_threshold)

                        if not is_dup:
                            seen.add(filler.signature())
                            fillers.append(filler)
                            self.duplicate_tracker.record_proposal(filler, False)
                        else:
                            self.duplicate_tracker.record_proposal(filler, True)
                    except Exception as e:
                        logger.debug(f"Semantic drift {strategy} failed: {e}")
        except Exception as e:
            logger.warning(f"Semantic drift mutations failed: {e}")

        # Phase 2: Word-level mutations
        if len(fillers) < needed:
            try:
                from repro.algorithm.semantic_drift import WordLevelMutator
                word_mutator = WordLevelMutator(rng=self.rng)
                word_strategies = ['synonym_substitution', 'sentence_reordering', 'combined_mutation']

                for base_candidate in best_candidates:
                    if len(fillers) >= needed:
                        break
                    for strategy in word_strategies:
                        if len(fillers) >= needed:
                            break
                        try:
                            if strategy == 'synonym_substitution':
                                mutated_text = word_mutator.apply_synonym_substitution(base_candidate.text, rate=0.3)
                            elif strategy == 'sentence_reordering':
                                mutated_text = word_mutator.apply_sentence_reordering(base_candidate.text)
                            else:
                                mutated_text = word_mutator.apply_combined_mutation(base_candidate.text)

                            filler = AttackCandidate(
                                text=mutated_text,
                                parent_id=base_candidate.id,
                                goal=base_candidate.goal or seed.goal,
                                meta={"edit": f"word_level_{strategy}", "filler": True}
                            )
                            self._ensure_original_family(filler, parent=base_candidate)

                            is_dup = filler.signature() in seen
                            if not is_dup and use_similarity and children:
                                is_dup = filler.similarity_to_any(children[-20:], threshold=similarity_threshold)

                            if not is_dup:
                                seen.add(filler.signature())
                                fillers.append(filler)
                                self.duplicate_tracker.record_proposal(filler, False)
                            else:
                                self.duplicate_tracker.record_proposal(filler, True)
                        except Exception as e:
                            logger.debug(f"Word-level {strategy} failed: {e}")
            except Exception as e:
                logger.warning(f"Word-level mutations failed: {e}")

        # Phase 3: Dataset sampling with goal filtering (if still need more)
        if len(fillers) < needed and hasattr(self, 'editor') and hasattr(self.editor, 'jailbreak_examples'):
            try:
                examples = self.editor.jailbreak_examples
                goal_keywords = set(seed.goal.lower().split()) if seed.goal else set()

                # Filter examples by goal relevance
                relevant_examples = []
                for ex in examples[:100]:  # Check first 100 examples
                    ex_text = ex.get('prompt', '') if isinstance(ex, dict) else str(ex)
                    if goal_keywords and any(kw in ex_text.lower() for kw in goal_keywords):
                        relevant_examples.append(ex_text)

                # Use random examples if no relevant ones found
                if not relevant_examples:
                    relevant_examples = [ex.get('prompt', str(ex)) if isinstance(ex, dict) else str(ex)
                                         for ex in self.rng.sample(examples, min(10, len(examples)))]

                for ex_text in relevant_examples:
                    if len(fillers) >= needed:
                        break
                    filler = AttackCandidate(
                        text=ex_text,
                        parent_id=seed.id,
                        goal=seed.goal,
                        meta={"edit": "dataset_filler", "filler": True}
                    )
                    self._ensure_original_family(filler, parent=seed)

                    is_dup = filler.signature() in seen
                    if not is_dup and use_similarity and children:
                        is_dup = filler.similarity_to_any(children[-20:], threshold=similarity_threshold)

                    if not is_dup:
                        seen.add(filler.signature())
                        fillers.append(filler)
                        self.duplicate_tracker.record_proposal(filler, False)
                    else:
                        self.duplicate_tracker.record_proposal(filler, True)
            except Exception as e:
                logger.warning(f"Dataset sampling failed: {e}")

        # Phase 4: Last resort - use existing candidates (old behavior)
        if len(fillers) < needed:
            remaining = needed - len(fillers)
            logger.info(f"Falling back to {remaining} existing candidates (mutation strategies exhausted)")

            for c, m in sorted(scored, key=lambda x: (x[1]["ASR"], x[1]["XFER"]), reverse=True):
                if len(fillers) >= needed:
                    break
                if c.text and c.text.strip() and c.signature() not in seen:
                    seen.add(c.signature())
                    fillers.append(c)

        logger.info(f"Generated {len(fillers)} diverse fillers: "
                   f"semantic_drift={sum(1 for f in fillers if 'semantic_drift' in f.meta.get('edit', ''))}, "
                   f"word_level={sum(1 for f in fillers if 'word_level' in f.meta.get('edit', ''))}, "
                   f"dataset={sum(1 for f in fillers if f.meta.get('edit') == 'dataset_filler')}, "
                   f"existing={sum(1 for f in fillers if not f.meta.get('filler', False))}")

        return fillers[:needed]


"""REPRO Optimizer - CurriculumMixin methods."""

import logging
from typing import List, Dict, Tuple, Optional, Set
from repro.algorithm.models import AttackCandidate, Task
from repro.algorithm.task_categories import categorize_task, categorize_tasks_batch

logger = logging.getLogger(__name__)


class CurriculumMixin:
    """Mixin providing curriculum methods."""

    def _get_curriculum_cracked_count(self) -> int:
        """Count tasks in curriculum subset that have at least 1 success (ever cracked)."""
        if not hasattr(self, 'task_difficulty_tracker') or not hasattr(self, '_curriculum_subset_indices'):
            return 0
        if not self._curriculum_subset_indices:
            return 0

        cracked = 0
        for task_idx in self._curriculum_subset_indices:
            successes = self.task_difficulty_tracker.task_successes.get(int(task_idx), 0)
            if successes > 0:
                cracked += 1
        return cracked

    def _get_curriculum_never_cracked_count(self) -> int:
        """Count tasks in curriculum subset that have never been cracked (0 successes)."""
        if not hasattr(self, 'task_difficulty_tracker') or not hasattr(self, '_curriculum_subset_indices'):
            return 0
        if not self._curriculum_subset_indices:
            return 0

        never_cracked = 0
        for task_idx in self._curriculum_subset_indices:
            successes = self.task_difficulty_tracker.task_successes.get(int(task_idx), 0)
            if successes == 0:
                never_cracked += 1
        return never_cracked

    def _initialize_curriculum(self, tasks: List[Task]) -> List[Task]:
        """
        Initialize curriculum mode with a fixed random subset of tasks.

        Uses curriculum_subset_ratio to determine subset size. The subset remains fixed
        until mastery (100% ASR) triggers a full evaluation and potential phase switch.

        Args:
            tasks: Full list of tasks

        Returns:
            Fixed subset of tasks for the first curriculum phase
        """
        self.all_tasks = tasks
        self.curriculum_initialized = True
        self.curriculum_phase = 0
        self.phase_switch_count = 0
        self.failed_task_history = set()
        self._last_full_eval_results = {}
        self._curriculum_seen_tasks = set()

        # Calculate subset size using curriculum_subset_ratio
        num_samples = max(5, int(len(tasks) * self.curriculum_subset_ratio))

        if len(tasks) <= num_samples:
            self.current_task_subset = tasks
            logger.info(f"Curriculum Phase 0: Using all {len(tasks)} tasks (subset size >= total)")
            return tasks

        # Stratified selection for initial subset (reduces run-to-run variance).
        # A purely random subset can accidentally be "too easy" or "too hard", which makes
        # convergence highly sensitive to luck and can look like the optimizer is "stuck".
        indices: List[int] = []
        try:
            task_texts = [t.user_content for t in tasks]
            category_groups = categorize_tasks_batch(task_texts)

            # Prefer non-general categories first; keep "general" as a filler category.
            categories = [c for c in category_groups.keys() if c != "general"]
            self.rng.shuffle(categories)
            if "general" in category_groups:
                categories.append("general")

            # Create shuffled pools per category for round-robin selection
            pools = {c: list(category_groups.get(c, [])) for c in categories}
            for c in pools:
                self.rng.shuffle(pools[c])

            selected = set()

            # Pass 1: ensure coverage (1 from each category where possible)
            for c in categories:
                if len(selected) >= num_samples:
                    break
                if pools[c]:
                    selected.add(pools[c].pop())

            # Pass 2: round-robin fill remaining slots
            while len(selected) < num_samples:
                progressed = False
                for c in categories:
                    if len(selected) >= num_samples:
                        break
                    while pools[c]:
                        idx = pools[c].pop()
                        if idx not in selected:
                            selected.add(idx)
                            progressed = True
                            break
                if not progressed:
                    break

            indices = sorted(selected)
            if len(indices) < num_samples:
                # Fallback: fill any remaining slots uniformly at random
                remaining = [i for i in range(len(tasks)) if i not in set(indices)]
                if remaining:
                    fill = self.rng.sample(remaining, min(num_samples - len(indices), len(remaining)))
                    indices = sorted(set(indices) | set(fill))

            logger.info(
                f"Curriculum Phase 0: Stratified subset selection across {len(categories)} categories "
                f"({len(indices)}/{len(tasks)} tasks)"
            )
        except Exception as e:
            logger.warning(f"Curriculum Phase 0: Stratified subset selection failed ({e}); falling back to random")
            indices = self.rng.sample(range(len(tasks)), num_samples)

        self.current_task_subset = [tasks[i] for i in sorted(indices)]
        self._curriculum_subset_indices = set(indices)  # Track which indices are in current subset
        self._curriculum_seen_tasks.update(self._curriculum_subset_indices)

        logger.info(f"Curriculum Phase 0: Fixed subset of {len(self.current_task_subset)}/{len(tasks)} tasks "
                   f"(ratio={self.curriculum_subset_ratio:.1%})")
        return self.current_task_subset

    def _switch_curriculum_phase(self, failed_task_indices: List[int]):
        """
        Switch to the next curriculum phase, optimizing on tasks that failed during full evaluation.

        Args:
            failed_task_indices: Indices of tasks that failed during full evaluation
        """
        self.curriculum_phase += 1
        self.phase_switch_count += 1
        self.curriculum_reshuffles = 0  # Reset reshuffle count for new phase

        # Track persistently failed tasks
        self.failed_task_history.update(failed_task_indices)

        # Analyze failed tasks to understand WHY they're failing
        for idx in failed_task_indices:
            task = self.all_tasks[idx]
            best_attempt = self._task_best_attempts.get(idx)
            if best_attempt and self.failure_analyzer:
                candidate, response, metrics = best_attempt
                try:
                    analysis = self.failure_analyzer.analyze_failure(
                        attack_text=candidate.text,
                        target_response=response,
                        metrics=metrics
                    )
                    self._task_failure_insights[idx] = {
                        'primary_category': analysis.refusal_type,
                        'trigger_words': analysis.trigger_words,
                        'refusal_type': analysis.refusal_type,
                        'defense_mechanism': analysis.defense_mechanism,
                        'suggestions': analysis.suggested_modifications,
                        'confidence': analysis.confidence,
                        'task_content': task.user_content[:100],
                    }
                    logger.info(f"Task {idx} failure analysis: refusal={analysis.refusal_type}, "
                               f"triggers={analysis.trigger_words[:3] if analysis.trigger_words else []}")
                except Exception as e:
                    logger.warning(f"Failed to analyze task {idx}: {e}")

        # Enable burst mode for small sets of hard tasks (focused attack)
        if len(failed_task_indices) <= 5:
            self._hard_task_burst_mode = True
            self._burst_iterations_remaining = 10  # Extra focused iterations
            self._burst_children_multiplier = 2.0  # Double children generation
            logger.info(f"Burst mode ENABLED: {len(failed_task_indices)} hard tasks, "
                       f"2x children for {self._burst_iterations_remaining} iterations")

            # Queue task-specific attack generation for each hard task
            for idx in failed_task_indices:
                task = self.all_tasks[idx]
                insights = self._task_failure_insights.get(idx, {})
                self._pending_task_specific_generations.append({
                    'task_idx': idx,
                    'task': task,
                    'insights': insights,
                    'priority': 'high'
                })
        else:
            self._hard_task_burst_mode = False

        # Prefer unseen tasks first to reduce repeated full-eval on the same failures early in training.
        remaining_unseen = [i for i in range(len(self.all_tasks)) if i not in self._curriculum_seen_tasks]
        target_subset_size = max(5, int(len(self.all_tasks) * self.curriculum_subset_ratio))

        if remaining_unseen:
            # Build next subset from remaining unseen tasks (stratified if possible).
            try:
                remaining_tasks = [self.all_tasks[i] for i in remaining_unseen]
                task_texts = [t.user_content for t in remaining_tasks]
                category_groups = categorize_tasks_batch(task_texts)

                categories = [c for c in category_groups.keys() if c != "general"]
                self.rng.shuffle(categories)
                if "general" in category_groups:
                    categories.append("general")

                pools = {c: list(category_groups.get(c, [])) for c in categories}
                for c in pools:
                    self.rng.shuffle(pools[c])

                selected = set()
                # Pass 1: one per category
                for c in categories:
                    if len(selected) >= target_subset_size:
                        break
                    if pools[c]:
                        selected.add(pools[c].pop())
                # Pass 2: round-robin
                while len(selected) < target_subset_size:
                    progressed = False
                    for c in categories:
                        if len(selected) >= target_subset_size:
                            break
                        while pools[c]:
                            idx = pools[c].pop()
                            if idx not in selected:
                                selected.add(idx)
                                progressed = True
                                break
                    if not progressed:
                        break

                indices_in_remaining = sorted(selected)
                if len(indices_in_remaining) < target_subset_size:
                    remaining_pool = [i for i in range(len(remaining_tasks)) if i not in set(indices_in_remaining)]
                    if remaining_pool:
                        fill = self.rng.sample(
                            remaining_pool, min(target_subset_size - len(indices_in_remaining), len(remaining_pool))
                        )
                        indices_in_remaining = sorted(set(indices_in_remaining) | set(fill))

                next_indices = [remaining_unseen[i] for i in indices_in_remaining]
                self.current_task_subset = [self.all_tasks[i] for i in next_indices]
                self._curriculum_subset_indices = set(next_indices)
                self._curriculum_seen_tasks.update(self._curriculum_subset_indices)
                logger.info(
                    f"Curriculum Phase {self.curriculum_phase}: Next unseen subset "
                    f"{len(self.current_task_subset)}/{len(self.all_tasks)} tasks"
                )
            except Exception as e:
                logger.warning(f"Curriculum Phase {self.curriculum_phase}: Unseen subset selection failed ({e}); using random")
                subset_size = min(target_subset_size, len(remaining_unseen))
                next_indices = self.rng.sample(remaining_unseen, subset_size)
                self.current_task_subset = [self.all_tasks[i] for i in next_indices]
                self._curriculum_subset_indices = set(next_indices)
                self._curriculum_seen_tasks.update(self._curriculum_subset_indices)
                logger.info(
                    f"Curriculum Phase {self.curriculum_phase}: Random unseen subset "
                    f"{len(self.current_task_subset)}/{len(self.all_tasks)} tasks"
                )
        else:
            # All tasks have been seen at least once: consolidate on failed tasks (even if > subset size).
            consolidated_indices = sorted(self.failed_task_history)
            self.current_task_subset = [self.all_tasks[i] for i in consolidated_indices]
            self._curriculum_subset_indices = set(consolidated_indices)
            logger.warning(
                f"Curriculum Phase {self.curriculum_phase}: All tasks seen; "
                f"consolidating on {len(self.current_task_subset)} failed tasks"
            )

        # IMPORTANT: The subset metric is not comparable across subset changes.
        # Reset the stagnation baseline so we don't treat the new (possibly harder) subset as "never improving".
        if hasattr(self, "stagnation_manager") and self.stagnation_manager:
            old_prev = getattr(self.stagnation_manager.state, "prev_best_metric", 0.0)
            self.stagnation_manager.state.prev_best_metric = 0.0
            self.stagnation_manager.state.iterations_since_improvement = 0
            # Also reset component stagnation levels (prevents stale high-stagnation settings carrying over).
            self.stagnation_manager._reset_stagnation()
            logger.info(
                f"Curriculum Phase {self.curriculum_phase}: Reset stagnation baseline due to subset change "
                f"(prev_best_metric {old_prev:.3f} -> 0.000)"
            )

        # Notify TaskFamilyExplorer of phase change
        if hasattr(self, 'task_family_explorer'):
            self.task_family_explorer.on_curriculum_phase_change(
                new_phase=self.curriculum_phase,
                new_subset=list(self._curriculum_subset_indices)
            )

    def _reshuffle_curriculum_subset(self):
        """
        Reshuffle the curriculum subset when stuck at high ASR without reaching 100%.
        
        This addresses the local maxima problem where the subset tasks might be too
        similar, causing the algorithm to optimize for that specific similarity
        without finding attacks that generalize to other tasks.
        
        Strategy:
        1. Keep the hardest tasks (lowest success rate in current attempts)
        2. Replace some tasks with random new ones from the full task set
        3. This forces exploration of different task combinations
        """
        if not self.curriculum_mode or not self.curriculum_initialized:
            return
            
        if len(self.all_tasks) <= len(self.current_task_subset):
            logger.info("Cannot reshuffle: subset already contains all tasks")
            return
        
        current_size = len(self.current_task_subset)
        
        # Identify tasks by their success rate (using task difficulty tracker if available)
        task_success_rates = {}
        if hasattr(self, 'task_difficulty_tracker'):
            for idx in self._curriculum_subset_indices:
                # Use get_task_success_rate method (task_stats attribute doesn't exist)
                if self.task_difficulty_tracker.task_attempts.get(idx, 0) > 0:
                    success_rate = self.task_difficulty_tracker.get_task_success_rate(idx)
                else:
                    success_rate = 0.5  # Default to 0.5 if no data
                task_success_rates[idx] = success_rate
        else:
            # Fallback: use task success history from evaluations
            for idx in self._curriculum_subset_indices:
                history = self._task_success_history.get(idx, {"attempts": 0, "successes": 0})
                if history["attempts"] > 0:
                    success_rate = history["successes"] / history["attempts"]
                else:
                    success_rate = 0.5
                task_success_rates[idx] = success_rate
        
        # Sort by success rate (lowest = hardest, keep these)
        sorted_indices = sorted(task_success_rates.keys(), key=lambda x: task_success_rates[x])
        
        # Keep the hardest 50% of tasks
        # When deeply stagnating, keep fewer hard tasks to avoid anchoring the subset on "impossible" items.
        # This increases exploration while still retaining some hard coverage.
        keep_ratio = 0.50
        try:
            st_level = getattr(self.stagnation_manager.state, "current_stagnation_level", 0)
            if st_level >= 3:
                keep_ratio = 0.25
            elif st_level >= 2:
                keep_ratio = 0.35
        except Exception:
            keep_ratio = 0.50

        keep_count = max(2, int(round(current_size * keep_ratio)))
        keep_indices = set(sorted_indices[:keep_count])
        
        # Find replacement candidates (not in current subset)
        available_indices = [i for i in range(len(self.all_tasks)) 
                           if i not in self._curriculum_subset_indices]
        
        if not available_indices:
            logger.warning("No available tasks to reshuffle with")
            return
        
        # Randomly select replacements
        replace_count = current_size - keep_count
        replace_count = min(replace_count, len(available_indices))
        new_indices = self.rng.sample(available_indices, replace_count)
        
        # Build new subset
        new_subset_indices = sorted(list(keep_indices) + new_indices)
        self.current_task_subset = [self.all_tasks[i] for i in new_subset_indices]
        self._curriculum_subset_indices = set(new_subset_indices)
        if hasattr(self, '_curriculum_seen_tasks'):
            self._curriculum_seen_tasks.update(self._curriculum_subset_indices)

        # Track reshuffles for monitoring (these counters were previously never incremented)
        self.curriculum_reshuffles += 1
        self.total_curriculum_reshuffles += 1

        # IMPORTANT: Reset stagnation baseline on subset change.
        # Otherwise, once a prior subset achieved a high score, a harder reshuffled subset can never
        # surpass it, and the optimizer will look permanently "stuck" and keep restarting.
        if hasattr(self, "stagnation_manager") and self.stagnation_manager:
            old_prev = getattr(self.stagnation_manager.state, "prev_best_metric", 0.0)
            self.stagnation_manager.state.prev_best_metric = 0.0
            self.stagnation_manager.state.iterations_since_improvement = 0
            self.stagnation_manager._reset_stagnation()
            logger.info(
                f"Curriculum reshuffle: Reset stagnation baseline due to subset change "
                f"(prev_best_metric {old_prev:.3f} -> 0.000)"
            )
        
        logger.info(f"Curriculum reshuffle: Kept {len(keep_indices)} hard tasks, added {len(new_indices)} new tasks "
                   f"(total={len(self.current_task_subset)})")

        # Log the hardest kept tasks
        if keep_indices:
            hardest_rates = [(idx, task_success_rates.get(idx, 0)) for idx in list(keep_indices)[:3]]
            logger.info(f"Kept hardest tasks (by success rate): {hardest_rates}")

        # Notify TaskFamilyExplorer of reshuffle (same phase, new subset)
        if hasattr(self, 'task_family_explorer'):
            self.task_family_explorer.on_curriculum_phase_change(
                new_phase=self.curriculum_phase,
                new_subset=new_subset_indices
            )

    async def _evaluate_single_candidate_on_full_tasks(
        self, candidate: AttackCandidate
    ) -> Tuple[Dict, List[Tuple[int, bool]]]:
        """
        Evaluate a single candidate on ALL tasks (full evaluation).

        Called when a candidate masters the current subset to check for universal success.

        Args:
            candidate: The attack candidate to evaluate

        Returns:
            Tuple of (metrics_dict, per_task_results) where per_task_results is
            a list of (task_idx, success) tuples
        """
        logger.info(f"Curriculum: Full evaluation triggered for candidate {candidate.id}")

        # Evaluate on all tasks
        results = await self.evaluator.evaluate_multi_env_async(
            model=self.model,
            envs=self.envs,
            attack_text=candidate.text,
            tasks=self.all_tasks,
            model_name=self.model_name,
            temperature=self.eval_temperature,
            timeout=self.timeout
        )

        metrics, eval_rows = results

        # Build per-task success mapping
        # EvalRow doesn't have a 'task' attribute, but has 'user_request' which is task.user_content
        per_task_results = []

        # Build mapping from user_content to task index
        task_content_to_idx = {task.user_content: idx for idx, task in enumerate(self.all_tasks)}

        for row in eval_rows:
            # Match by user_request (which is task.user_content)
            if hasattr(row, 'user_request') and row.user_request:
                task_idx = task_content_to_idx.get(row.user_request, -1)
                if task_idx >= 0:
                    success = row.asr > 0
                    per_task_results.append((task_idx, success))

                    # Track best attempt for failed tasks (for failure analysis)
                    if not success:
                        current_best = self._task_best_attempts.get(task_idx)
                        task_metrics = {"ASR": row.asr, "reply_len": len(row.reply) if row.reply else 0}
                        # Update if no prior attempt or if this attempt has better metrics
                        if current_best is None:
                            self._task_best_attempts[task_idx] = (candidate, row.reply, task_metrics)

        # Store for later reference
        self._last_full_eval_results = {idx: success for idx, success in per_task_results}

        # Log discrepancy if aggregate ASR doesn't match per-task results
        failed_count = sum(1 for _, success in per_task_results if not success)
        success_count = sum(1 for _, success in per_task_results if success)
        logger.info(f"Curriculum: Full eval ASR={metrics.get('ASR', 0):.2%}, "
                   f"XFER={metrics.get('XFER', 0):.2%}, "
                   f"per-task: {success_count}/{len(self.all_tasks)} succeeded, {failed_count} failed")

        return metrics, per_task_results

    async def _check_and_handle_curriculum_mastery(
        self, scored: List[Tuple[AttackCandidate, Dict]], iteration: int
    ) -> Tuple[bool, Optional[Tuple[AttackCandidate, Dict]], List[Tuple[int, Dict, List[Tuple[int, bool]]]]]:
        """
        Check curriculum mastery and perform full evaluation for ALL qualifying candidates.

        This is critical for correct metric tracking: any candidate that achieves 100% on the
        subset MUST be evaluated on the full task set, and its metrics must be updated to
        reflect the full evaluation results.

        Args:
            scored: List of (candidate, metrics) tuples from current iteration
            iteration: Current iteration number

        Returns:
            Tuple of:
            - found_universal: True if any candidate achieved 100% on full task set
            - universal_result: (candidate, full_metrics) if universal found, else None
            - updated_metrics: List of (scored_index, full_metrics, per_task_results) for ALL
                              candidates that achieved 100% on subset. These metrics should
                              REPLACE the subset metrics in the scored list.
        """
        updated_metrics = []
        universal_result = None
        found_universal = False
        failed_indices_for_phase_switch = None

        # Evaluate ALL candidates with 100% ASR on subset (not just the first one)
        for idx, (candidate, metrics) in enumerate(scored):
            if metrics.get("ASR", 0) >= 1.0:
                logger.info(f"Curriculum: Candidate {candidate.id} mastered Phase {self.curriculum_phase} subset "
                           f"(subset ASR={metrics.get('ASR', 0):.2f}), evaluating on full {len(self.all_tasks)} tasks...")

                # Evaluate on full task set
                full_metrics, per_task_results = await self._evaluate_single_candidate_on_full_tasks(candidate)

                # Store the full metrics for this candidate (will be used to update scored list)
                updated_metrics.append((idx, full_metrics.copy(), per_task_results))

                logger.info(f"Curriculum: Candidate {candidate.id} full eval: ASR={full_metrics.get('ASR', 0):.2f}, "
                           f"XFER={full_metrics.get('XFER', 0):.2f}")

                if full_metrics.get("ASR", 0) >= 1.0:
                    # Universal jailbreak found!
                    logger.info(f"CURRICULUM UNIVERSAL JAILBREAK: Candidate {candidate.id} "
                               f"achieved 100% on all {len(self.all_tasks)} tasks!")
                    found_universal = True
                    universal_result = (candidate, full_metrics)
                    # Continue evaluating other 100% candidates to update their metrics too
                else:
                    # Track failed tasks for potential phase switch (only first failure set)
                    if failed_indices_for_phase_switch is None:
                        failed_indices = [task_idx for task_idx, success in per_task_results if not success]
                        if not failed_indices:
                            # Edge case: per_task_results incomplete, use all non-subset tasks
                            failed_indices = [i for i in range(len(self.all_tasks))
                                             if i not in self._curriculum_subset_indices]
                        if failed_indices:
                            failed_indices_for_phase_switch = failed_indices

        # Handle phase switch if needed (only if no universal found)
        if not found_universal and failed_indices_for_phase_switch:
            logger.info(f"Curriculum: {len(failed_indices_for_phase_switch)} tasks failed full eval, switching phase")
            self._switch_curriculum_phase(failed_indices_for_phase_switch)
        elif not found_universal and updated_metrics and not failed_indices_for_phase_switch:
            # Edge case: all 100% subset candidates failed full eval but no failed indices identified
            logger.warning(f"Curriculum: Full eval ASR < 100% but no failed tasks identified. "
                          f"Continuing on current subset (likely evaluation variance)")

        return found_universal, universal_result, updated_metrics


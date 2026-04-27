"""
Stagnation Manager - Consolidated stagnation detection and restart logic.

This module consolidates the complex stagnation handling logic from optimizer.py
into a clean, maintainable class structure.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from repro.algorithm.strategy_optimizer import StrategyOptimizer
    from repro.algorithm.reflective_editor import ReflectiveLMEditor
    from repro.algorithm.diversity import DiversityPreserver

logger = logging.getLogger(__name__)


class RestartLevel(Enum):
    """Restart intensity levels."""
    NONE = 0
    SOFT = 1
    MEDIUM = 2
    FIRM = 3
    HARD = 4
    SURGE = 5


@dataclass
class RestartConfig:
    """Configuration for a restart action."""
    level: RestartLevel
    stagnation_level: int  # 0-4 for component stagnation levels
    fresh_seeds_needed: int
    mutation_multiplier: float
    temp_boost: float
    elite_size: Optional[int] = None  # None = don't change
    reset_strategy_weights: bool = False
    reset_family_exhaustion: bool = False
    decay_family_exhaustion: float = 0.0  # 0 = no decay, 0.5 = decay by half
    reshuffle_curriculum: bool = False
    # Domain-specific side effects
    penalize_bottom_strategies: float = 0.0  # Multiply bottom 50% by this (0 = no penalty)
    reset_bottom_strategy_percent: float = 0.0  # Reset bottom N% to baseline (0 = none)
    trim_recent_strategies: bool = False  # Trim recent_strategies history
    clear_elite_cache: bool = False  # Clear elite score cache
    reset_stagnation_after: bool = False  # Reset stagnation levels after applying


@dataclass
class StagnationState:
    """Tracks current stagnation state."""
    iterations_since_improvement: int = 0
    restart_count: int = 0
    total_historical_restarts: int = 0  # Never resets - used for escalation
    curriculum_reshuffles: int = 0
    total_curriculum_reshuffles: int = 0  # Never resets
    current_stagnation_level: int = 0
    prev_best_metric: float = 0.0
    stagnation_threshold: int = 10


class StagnationManager:
    """
    Manages stagnation detection, escalation, and restart logic.

    This class consolidates the complex stagnation handling that was previously
    scattered throughout optimizer.py into a clean, testable interface.
    """

    def __init__(
        self,
        population_size: int,
        base_elite_size: int,
        optimize_mode: str = "explore",
        curriculum_mode: bool = False,
        early_stopping: bool = False,
        patience: int = 50,
        max_restarts: int = 0,  # 0 = unlimited
    ):
        self.population_size = population_size
        self.base_elite_size = base_elite_size
        self.optimize_mode = optimize_mode
        self.curriculum_mode = curriculum_mode
        self.early_stopping = early_stopping
        self.patience = patience
        self.max_restarts = max_restarts

        # State tracking
        self.state = StagnationState()

        # Component references (set via register_components)
        self._strategy_optimizer: Optional['StrategyOptimizer'] = None
        self._editor: Optional['ReflectiveLMEditor'] = None
        self._diversity_preserver: Optional['DiversityPreserver'] = None

    def register_components(
        self,
        strategy_optimizer: Optional['StrategyOptimizer'] = None,
        editor: Optional['ReflectiveLMEditor'] = None,
        diversity_preserver: Optional['DiversityPreserver'] = None,
    ) -> None:
        """Register component references for stagnation level propagation."""
        self._strategy_optimizer = strategy_optimizer
        self._editor = editor
        self._diversity_preserver = diversity_preserver

    def update_threshold(
        self,
        global_best_asr: float,
        optimize_mode: str,
        current_subset_size: int = 0,
        total_tasks: int = 0,
    ) -> int:
        """
        Dynamically adjust stagnation threshold based on ASR and mode.

        At high ASR, give more time to find breakthrough mutations.
        In curriculum mode with subsets, be less patient to encourage diversity.

        Args:
            global_best_asr: Current best ASR
            optimize_mode: "explore" or "exploit"
            current_subset_size: Size of current curriculum subset (0 if not using curriculum)
            total_tasks: Total number of tasks (0 if not using curriculum)

        Returns:
            Updated stagnation threshold
        """
        is_curriculum_subset = (
            self.curriculum_mode and
            optimize_mode == "explore" and
            current_subset_size > 0 and
            current_subset_size < total_tasks
        )

        if is_curriculum_subset:
            # Curriculum subset mode: Use shorter thresholds
            # Subset ASR can be misleading - be less patient
            if global_best_asr >= 0.8:
                threshold = 12
            elif global_best_asr >= 0.5:
                threshold = 10
            else:
                threshold = 8
            logger.debug(f"Curriculum subset mode: stagnation_threshold={threshold}")
        else:
            if optimize_mode == "exploit":
                # Exploit mode: be more aggressive to escape local minima quickly
                if global_best_asr >= 0.8:
                    threshold = 8
                elif global_best_asr >= 0.5:
                    threshold = 6
                else:
                    threshold = 4
            else:
                # Standard mode: ASR-aware thresholds
                if global_best_asr >= 0.8:
                    threshold = 20  # Very patient at high ASR
                elif global_best_asr >= 0.5:
                    threshold = 15  # More patient at medium-high ASR
                else:
                    threshold = 10  # Standard threshold

        self.state.stagnation_threshold = threshold
        return threshold

    def record_iteration(
        self,
        current_metric: float,
        global_best_metric: float,
        improvement_threshold: float = 0.001,
    ) -> bool:
        """
        Record iteration result and check for improvement.

        Args:
            current_metric: Current iteration's best metric
            global_best_metric: Global best metric
            improvement_threshold: Minimum improvement to count as progress

        Returns:
            True if improvement was detected
        """
        improved = current_metric > self.state.prev_best_metric + improvement_threshold

        if improved:
            self.state.iterations_since_improvement = 0
            self.state.prev_best_metric = current_metric
        else:
            self.state.iterations_since_improvement += 1

        return improved

    def handle_improvement(
        self,
        global_best_asr: float,
        is_stubborn_breakthrough: bool = False,
    ) -> None:
        """
        Handle improvement event - reset or decay stagnation based on ASR level.

        At high ASR (>=80%), minor improvements only decay stagnation.
        At lower ASR or for stubborn breakthroughs, fully reset stagnation.

        Args:
            global_best_asr: Current global best ASR
            is_stubborn_breakthrough: Whether this solved a previously stubborn task
        """
        if global_best_asr >= 0.80 and not is_stubborn_breakthrough:
            # Minor improvement at high ASR - decay instead of reset
            if self.state.current_stagnation_level > 0:
                new_level = max(0, self.state.current_stagnation_level - 1)
                self._set_stagnation_level(new_level)
                logger.debug(f"Minor improvement at high ASR - decay stagnation to {new_level}")

            # Decay restart count
            if self.state.restart_count > 0:
                self.state.restart_count = max(0, self.state.restart_count - 1)
                logger.debug(f"Decay restart_count to {self.state.restart_count}")
        else:
            # Full reset for low/medium ASR or stubborn breakthroughs
            self._reset_stagnation()
            if self.state.restart_count > 0:
                old_count = self.state.restart_count
                self.state.restart_count = 0
                logger.info(f"Improvement detected - restart_count RESET: {old_count} -> 0")

        # Notify editor of improvement
        if self._editor and hasattr(self._editor, 'record_improvement'):
            self._editor.record_improvement(improved=True)

    def handle_no_improvement(self) -> None:
        """Handle no-improvement event - notify components (counter already updated by record_iteration)."""
        # Note: iterations_since_improvement is already incremented by record_iteration()
        # Notify editor
        if self._editor and hasattr(self._editor, 'record_improvement'):
            self._editor.record_improvement(improved=False)

    def check_stagnation(self) -> bool:
        """
        Check if stagnation threshold has been reached.

        Returns:
            True if stagnation detected
        """
        return self.state.iterations_since_improvement >= self.state.stagnation_threshold

    def check_early_stopping(self) -> bool:
        """
        Check if early stopping should trigger.

        Returns:
            True if should stop early
        """
        if not self.early_stopping:
            return False

        if self.state.iterations_since_improvement >= self.patience:
            logger.info(f"Early stopping: No improvement for {self.state.iterations_since_improvement} iterations")
            return True

        if self.max_restarts > 0 and self.state.restart_count >= self.max_restarts:
            logger.info(f"Early stopping: Reached {self.max_restarts} restarts")
            return True

        return False

    def determine_restart_config(
        self,
        global_best_asr: float,
        is_curriculum_subset_stuck: bool = False,
    ) -> RestartConfig:
        """
        Determine the appropriate restart configuration based on current state.

        Args:
            global_best_asr: Current global best ASR
            is_curriculum_subset_stuck: Whether stuck in curriculum subset mode

        Returns:
            RestartConfig with all restart parameters
        """
        # Increment counters
        self.state.restart_count += 1
        self.state.total_historical_restarts += 1

        if is_curriculum_subset_stuck:
            config = self._get_curriculum_stuck_config(global_best_asr)
        elif global_best_asr >= 0.5:
            config = self._get_high_asr_config(global_best_asr)
        else:
            config = self._get_low_asr_config()

        return self._apply_exploit_bias(config)

    def _get_curriculum_stuck_config(self, global_best_asr: float) -> RestartConfig:
        """Get restart config for curriculum subset stuck scenario."""
        self.state.curriculum_reshuffles += 1
        self.state.total_curriculum_reshuffles += 1
        total_restarts = self.state.total_historical_restarts
        total_reshuffles = self.state.total_curriculum_reshuffles

        if global_best_asr >= 0.80:
            # Very close to goal - escalate quickly
            if total_restarts >= 5 or total_reshuffles >= 3:
                logger.warning(f"Curriculum SURGE: ASR={global_best_asr:.1%}, reshuffles={total_reshuffles}")
                return RestartConfig(
                    level=RestartLevel.SURGE,
                    stagnation_level=3,
                    fresh_seeds_needed=max(8, self.population_size // 2),
                    mutation_multiplier=3.0,
                    temp_boost=0.3,
                    elite_size=1,
                    reset_family_exhaustion=True,
                    reset_bottom_strategy_percent=0.7,
                    trim_recent_strategies=True,
                )
            elif total_restarts >= 3 or total_reshuffles >= 2:
                logger.warning(f"Curriculum FIRM: ASR={global_best_asr:.1%}")
                return RestartConfig(
                    level=RestartLevel.FIRM,
                    stagnation_level=2,
                    fresh_seeds_needed=max(6, self.population_size // 2),
                    mutation_multiplier=2.5,
                    temp_boost=0.2,
                    elite_size=max(2, self.base_elite_size // 2),
                    decay_family_exhaustion=0.5,
                    penalize_bottom_strategies=0.6,
                )
            else:
                logger.info(f"Curriculum SOFT: ASR={global_best_asr:.1%}")
                return RestartConfig(
                    level=RestartLevel.SOFT,
                    stagnation_level=1,
                    fresh_seeds_needed=max(4, self.population_size // 3),
                    mutation_multiplier=2.0,
                    temp_boost=0.15,
                )
        else:
            # Lower ASR (50-80%) - graduated escalation
            if total_restarts >= 8 or total_reshuffles >= 5:
                logger.info(f"Curriculum HARD: ASR={global_best_asr:.1%}")
                return RestartConfig(
                    level=RestartLevel.HARD,
                    stagnation_level=3,
                    fresh_seeds_needed=max(8, self.population_size // 2),
                    mutation_multiplier=3.0,
                    temp_boost=0.3,
                    elite_size=0,
                    reset_family_exhaustion=True,
                    clear_elite_cache=True,
                    reset_stagnation_after=True,
                )
            elif total_restarts >= 4 or total_reshuffles >= 3:
                return RestartConfig(
                    level=RestartLevel.FIRM,
                    stagnation_level=2,
                    fresh_seeds_needed=max(6, self.population_size // 2),
                    mutation_multiplier=2.5,
                    temp_boost=0.2,
                    elite_size=max(1, self.base_elite_size // 2),
                    decay_family_exhaustion=0.5,
                    penalize_bottom_strategies=0.6,
                )
            elif total_reshuffles >= 2:
                return RestartConfig(
                    level=RestartLevel.MEDIUM,
                    stagnation_level=2,
                    fresh_seeds_needed=max(5, self.population_size // 2),
                    mutation_multiplier=2.3,
                    temp_boost=0.15,
                    elite_size=max(1, self.base_elite_size // 2),
                )
            else:
                return RestartConfig(
                    level=RestartLevel.SOFT,
                    stagnation_level=1,
                    fresh_seeds_needed=max(4, self.population_size // 3),
                    mutation_multiplier=2.0,
                    temp_boost=0.1,
                )

    def _get_high_asr_config(self, global_best_asr: float) -> RestartConfig:
        """Get restart config for high ASR (>=50%) scenarios."""
        total_restarts = self.state.total_historical_restarts

        # Determine escalation thresholds based on ASR level
        if global_best_asr >= 0.90:
            soft_threshold = 1
            firm_threshold = 3
            asr_mode = "VERY_HIGH"
        elif global_best_asr >= 0.75:
            soft_threshold = 2
            firm_threshold = 5
            asr_mode = "HIGH"
        else:
            soft_threshold = 3
            firm_threshold = 6
            asr_mode = "MODERATE"

        if total_restarts <= soft_threshold:
            logger.info(f"{asr_mode} ASR ({global_best_asr:.1%}): SOFT restart #{total_restarts}")
            return RestartConfig(
                level=RestartLevel.SOFT,
                stagnation_level=1,
                fresh_seeds_needed=max(3, self.population_size // 4),
                mutation_multiplier=1.5,
                temp_boost=0.05,
            )
        elif total_restarts <= firm_threshold:
            logger.info(f"{asr_mode} ASR ({global_best_asr:.1%}): FIRM restart #{total_restarts}")
            return RestartConfig(
                level=RestartLevel.FIRM,
                stagnation_level=2,
                fresh_seeds_needed=max(5, self.population_size // 3),
                mutation_multiplier=2.2,
                temp_boost=0.15,
                elite_size=max(2, self.base_elite_size // 2),
                decay_family_exhaustion=0.5,
                penalize_bottom_strategies=0.6,  # Penalize bottom 50% by 0.6x
            )
        else:
            logger.info(f"{asr_mode} ASR ({global_best_asr:.1%}): SURGE restart #{total_restarts}")
            return RestartConfig(
                level=RestartLevel.SURGE,
                stagnation_level=3,
                fresh_seeds_needed=max(8, self.population_size // 2),
                mutation_multiplier=2.8,
                temp_boost=0.25,
                elite_size=1,
                reset_family_exhaustion=True,
                reset_bottom_strategy_percent=0.7,  # Reset bottom 70% to baseline
                trim_recent_strategies=True,
            )

    def _get_low_asr_config(self) -> RestartConfig:
        """Get restart config for low ASR (<50%) scenarios."""
        total_restarts = self.state.total_historical_restarts

        if total_restarts <= 2:
            return RestartConfig(
                level=RestartLevel.SOFT,
                stagnation_level=1,
                fresh_seeds_needed=max(4, self.population_size // 3),
                mutation_multiplier=2.0,
                temp_boost=0.1,
            )
        elif total_restarts <= 5:
            logger.info(f"MEDIUM restart #{total_restarts} at low ASR")
            return RestartConfig(
                level=RestartLevel.MEDIUM,
                stagnation_level=2,
                fresh_seeds_needed=max(6, self.population_size // 2),
                mutation_multiplier=2.5,
                temp_boost=0.2,
                elite_size=1,
            )
        else:
            logger.info(f"HARD restart #{total_restarts} at low ASR")
            return RestartConfig(
                level=RestartLevel.HARD,
                stagnation_level=3,
                fresh_seeds_needed=max(8, 2 * self.population_size // 3),
                mutation_multiplier=3.0,
                temp_boost=0.3,
                elite_size=0,
                reset_strategy_weights=True,
                clear_elite_cache=True,
                reset_stagnation_after=True,
            )

    def _apply_exploit_bias(self, config: RestartConfig) -> RestartConfig:
        """Adjust restart intensity for exploit mode (single-task focus)."""
        if self.optimize_mode != "exploit":
            return config

        config.mutation_multiplier = min(config.mutation_multiplier * 1.4, 4.0)
        config.temp_boost = min(config.temp_boost * 1.5, 0.5)
        config.fresh_seeds_needed = max(config.fresh_seeds_needed, self.population_size // 2)

        if config.level in {RestartLevel.FIRM, RestartLevel.HARD, RestartLevel.SURGE}:
            config.reset_family_exhaustion = True

        return config

    def apply_restart_config(self, config: RestartConfig) -> None:
        """
        Apply restart configuration to components.

        Args:
            config: RestartConfig to apply
        """
        # Set stagnation level on components
        self._set_stagnation_level(config.stagnation_level)

        # Apply mutation rate boost to editor
        if self._editor:
            self._editor.current_mutation_rate = min(
                0.90,
                self._editor.base_mutation_rate * config.mutation_multiplier
            )
            logger.debug(f"Boosted mutation rate to {self._editor.current_mutation_rate:.2f}")

            # Apply temperature boost
            if hasattr(self._editor, 'temperature'):
                self._editor.temperature = min(1.3, self._editor.temperature + config.temp_boost)

        # Reset iterations counter
        self.state.iterations_since_improvement = 0

        # Handle strategy weight reset for HARD restarts at low ASR
        if config.reset_strategy_weights and self._strategy_optimizer:
            if hasattr(self._strategy_optimizer, 'reset_weights'):
                self._strategy_optimizer.reset_weights()
            else:
                for strategy in self._strategy_optimizer.weights.keys():
                    self._strategy_optimizer.weights[strategy] = 1.0
            logger.debug("Reset strategy weights")

        # Reset stagnation after HARD restart
        if config.level == RestartLevel.HARD:
            self._reset_stagnation()

    def _set_stagnation_level(self, level: int) -> None:
        """Set stagnation level on all registered components."""
        self.state.current_stagnation_level = level

        if self._strategy_optimizer and hasattr(self._strategy_optimizer, 'set_stagnation_level'):
            self._strategy_optimizer.set_stagnation_level(level)
            logger.debug(f"Strategy optimizer stagnation level set to {level}")

        if self._editor and hasattr(self._editor, 'set_stagnation_level'):
            self._editor.set_stagnation_level(level)
            logger.debug(f"Editor stagnation level set to {level}")

        # Update diversity weight during stagnation
        if self._diversity_preserver:
            if level >= 2:
                boost = 0.1 + (level - 1) * 0.1
                self._diversity_preserver.set_diversity_weight_boost(boost)
            else:
                self._diversity_preserver.reset_diversity_weight()

    def _reset_stagnation(self) -> None:
        """Reset stagnation state on all components."""
        self.state.current_stagnation_level = 0

        if self._strategy_optimizer and hasattr(self._strategy_optimizer, 'reset_stagnation'):
            self._strategy_optimizer.reset_stagnation()

        if self._editor and hasattr(self._editor, 'reset_stagnation'):
            self._editor.reset_stagnation()

        if self._diversity_preserver:
            self._diversity_preserver.reset_diversity_weight()

        logger.debug("Stagnation reset on all components")

    def should_reshuffle_curriculum(self) -> bool:
        """Check if curriculum subset should be reshuffled."""
        return self.state.curriculum_reshuffles >= 4

    def reset_curriculum_reshuffles(self) -> None:
        """Reset curriculum reshuffle counter (but not total)."""
        self.state.curriculum_reshuffles = 0

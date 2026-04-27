"""
DSPy-inspired Strategy Weight Optimizer for REPRO

Implements Bayesian-style optimization of:
1. Mutation strategy weights (which strategies to use more often)
2. ICL example selection (which demonstrations are most effective)
3. Prompt instruction variants (which system prompts work best)

Analogous to DSPy's MIPROv2 but adapted for evolutionary algorithms.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
import os
import time

from repro.algorithm.strategy_families import STRATEGY_FAMILIES as BASE_STRATEGY_FAMILIES

logger = logging.getLogger(__name__)


class StrategyRelevanceScorer:
    """
    Scores how relevant a strategy is to the current analysis recommendations.

    Uses fuzzy keyword matching to map analysis recommendations to registered strategies.
    """

    @staticmethod
    def score_relevance(strategy_name: str, analysis: Optional[Dict[str, List[str]]] = None) -> float:
        """
        Score how relevant a strategy is to analysis recommendations.

        Args:
            strategy_name: Name of registered strategy (e.g., "semantic_drift_invisible")
            analysis: Analysis dict with recommended_techniques

        Returns:
            float: Relevance score 0.0-1.0 (1.0 = perfect match)
        """
        if not analysis or "recommended_techniques" not in analysis:
            return 0.0  # No analysis available

        recommended = analysis.get("recommended_techniques", [])
        if not recommended:
            return 0.0

        # Extract keywords from strategy name
        strategy_keywords = set(strategy_name.lower().replace("_", " ").split())

        # Check each recommended technique
        max_score = 0.0
        for rec in recommended:
            # Handle both string and dict formats
            if isinstance(rec, dict):
                rec_text = rec.get("technique", "")
            else:
                rec_text = str(rec)

            # Extract keywords from recommendation
            rec_keywords = set(rec_text.lower().replace("-", " ").replace("_", " ").split())

            # Calculate Jaccard similarity
            if not rec_keywords:
                continue

            intersection = strategy_keywords & rec_keywords
            union = strategy_keywords | rec_keywords

            if union:
                score = len(intersection) / len(union)
                max_score = max(max_score, score)

        return max_score

    @staticmethod
    def extract_strategy_names(analysis: Optional[Dict[str, List[str]]] = None) -> List[str]:
        """
        Extract strategy names from analysis recommendations.

        Useful for auto-registering new strategies.

        Args:
            analysis: Analysis dict with recommended_techniques

        Returns:
            List of strategy names (cleaned and normalized)
        """
        if not analysis or "recommended_techniques" not in analysis:
            return []

        recommended = analysis.get("recommended_techniques", [])
        strategy_names = []

        for rec in recommended:
            if isinstance(rec, dict):
                name = rec.get("technique", "")
            else:
                name = str(rec)

            # Clean and normalize
            name = name.lower().strip()
            # Replace spaces and dashes with underscores
            name = name.replace(" ", "_").replace("-", "_")
            # Remove special characters
            import re
            name = re.sub(r'[^a-z0-9_]', '', name)

            if name and len(name) >= 3:
                strategy_names.append(name)

        return strategy_names


class StrategyWeightOptimizer:
    """
    Bayesian-inspired optimizer for REPRO mutation strategy weights.

    Similar to DSPy's MIPROv2, but instead of optimizing prompts,
    we optimize which attack generation strategies work best.

    Enhanced with:
    - Strategy family quotas to prevent monoculture
    - Adaptive exploration during stagnation
    - Recent strategy tracking for diversity enforcement
    """

    # Strategy family mappings for diversity enforcement.
    STRATEGY_FAMILIES = BASE_STRATEGY_FAMILIES

    # Keywords to detect family from LLM-created strategy names
    FAMILY_KEYWORDS = {
        "hierarchical": "hierarchical", "hierarchy": "hierarchical",
        "graph": "graph", "traversal": "graph", "node": "graph",
        "roleplay": "persona", "persona": "persona", "character": "persona", "actor": "persona",
        "inject": "injection", "prefix": "injection", "suffix": "injection", "skeleton": "injection",
        "obfusc": "obfuscation", "encode": "obfuscation", "cipher": "obfuscation", "base64": "obfuscation",
        "ascii": "obfuscation", "chameleon": "obfuscation", "art_attack": "obfuscation",
        "academic": "framing", "research": "framing", "fiction": "framing", "story": "framing", 
        "hypothetical": "framing", "deflection": "framing", "reverse_psych": "framing",
        "technical": "technical", "json": "technical", "xml": "technical", "format_exploit": "technical",
        "word_level": "word_level", "synonym": "word_level", "reorder": "word_level", "shuffle": "word_level",
        "semantic": "semantic", "drift": "semantic", "meaning": "semantic", "dilution": "semantic",
        # New family keywords (2025)
        "inception": "inception", "virtual": "inception", "nested": "inception", "dream": "inception",
        "crescendo": "progressive", "escalat": "progressive", "many_shot": "progressive", 
        "cognitive": "progressive", "overload": "progressive", "multi_task": "progressive",
        "past_tense": "temporal", "future_tense": "temporal", "speculation": "temporal",
        "language": "language", "pivot": "language", "multilingual": "language", "translation": "language",
    }

    # 2024-12-29: Differentiated initial weights based on empirical success rates
    # Word-level strategies work well (40-48% success), paper-based get detected (0%)
    # 2025-01: Added new strategies for stubborn tasks
    # Single source of truth - used by both __init__ and reset_weights()
    STRATEGY_INITIAL_WEIGHTS = {
        # Proven strategies - HIGH weight (3.0)
        "word_level_combined_mutation": 3.0,
        "word_level_sentence_reordering": 3.0,
        "word_level_synonym_substitution": 2.5,
        # Mid-tier strategies - MEDIUM weight (2.0)
        "semantic_drift_hybrid": 2.0,
        "dataset_fallback": 2.0,
        "semantic_drift_chained": 1.5,
        "semantic_drift_persona": 1.5,
        # Research-backed high ASR - MEDIUM-HIGH weight (1.5-2.0)
        "past_tense_reframe": 2.0,         # 88% ASR on GPT-4o
        "roleplay_persona": 2.0,            # 89.6% ASR
        "policy_puppetry": 1.8,             # Universal bypass
        "deep_inception": 1.8,              # Multi-layer nesting
        "crescendo_attack": 1.5,            # Progressive escalation
        "skeleton_key": 1.5,                # System override
        "logic_trap": 1.5,                  # 81.4% ASR
        # Newer 2025 strategies - MEDIUM weight (1.0-1.5) - need more testing
        "many_shot_icl": 1.2,               # Many-shot learning
        "context_dilution": 1.2,            # Context flooding
        "virtualization_attack": 1.2,       # Sandboxed environment
        "code_chameleon": 1.0,              # Code-based encoding
        "output_format_exploit": 1.0,       # Format exploitation
        "cognitive_load": 1.0,              # Multi-task overload
        "task_deflection": 1.0,             # Indirect framing
        "future_tense_speculation": 1.0,    # Future tense variant
        "reverse_psychology": 0.8,          # Negative framing
        "ascii_art_attack": 0.8,            # ASCII art bypass
        "language_pivot": 0.8,              # Low-resource language
        # Paper-based strategies - LOW weight (0.2) - often detected
        "hcot_hijacking": 0.2,
        "puzzled_word_puzzle": 0.2,
        "mousetrap_mapping": 0.2,
        "graph_attack_semantic": 0.2,
        "cot_dilution": 0.2,
        "adversarial_poetry": 0.2,
        "flip_attack": 0.2,
        "bad_likert_judge": 0.2,
    }

    def __init__(self,
                 strategies: List[str],
                 initial_weight: float = 1.0,
                 learning_rate: float = 0.25,
                 exploration_bonus: float = 0.05,
                 auto_register: bool = False,
                 target_model_name: Optional[str] = None,
                 optimization_objective: Optional[str] = None,
                 max_family_ratio: float = 0.4,  # Max 40% from one family
                 family_quota_window: int = 15,  # Window for quota tracking
                 use_bayesian: bool = True):  # Ablation: False = uniform random
        """
        Initialize strategy weight optimizer.

        Args:
            strategies: List of mutation strategy names
            initial_weight: Starting weight for all strategies
            learning_rate: How fast to update weights (0-1)
                          TUNED: Increased from 0.1 to 0.25 for faster adaptation
            exploration_bonus: Bonus for under-explored strategies
            auto_register: If True, automatically register unknown strategies encountered at runtime
            target_model_name: Target model name for model-specific weight persistence
            optimization_objective: Optimization goal (e.g., 'jailbreak', 'reasoning_enhancement')
        """
        # Initialize with differentiated weights from class constant
        self.weights = {}
        for s in strategies:
            # Use strategy-specific weight if defined, else default
            self.weights[s] = self.STRATEGY_INITIAL_WEIGHTS.get(s, initial_weight)
        self.learning_rate = learning_rate
        self.exploration_bonus = exploration_bonus
        self.auto_register = auto_register
        self.initial_weight = initial_weight
        self.use_bayesian = use_bayesian

        # Track performance for Bayesian update
        self.strategy_stats = {s: {"attempts": 0, "successes": 0, "total_score": 0.0, "total_asr": 0.0}
                               for s in strategies}

        # Epsilon-greedy parameter (15% exploration - reduced from 20% for more exploitation)
        self.epsilon = 0.15
        self.base_epsilon = 0.15  # Store base for reset after stagnation recovery

        # Chaos mode parameter
        self.chaos_mode = False

        # Strategy family diversity enforcement
        self.max_family_ratio = max_family_ratio
        self.family_quota_window = family_quota_window
        self.recent_strategies: List[str] = []  # Track recently selected strategies
        self.stagnation_level = 0  # 0=normal, 1=soft, 2=medium, 3=hard

        # Track best combinations (DSPy-style)
        self.best_combos = []  # [(strategy, score, metrics)]

        # Exploration tracking
        self.iteration = 0

        # MODEL + TASK SPECIFIC WEIGHT PERSISTENCE
        self.target_model_name = target_model_name or "unknown"
        self.optimization_objective = optimization_objective or "jailbreak"

        # Clean names for filename
        clean_model_name = self.target_model_name.replace(':', '_').replace('/', '_')
        clean_objective = self.optimization_objective.replace('/', '_').replace(' ', '_')

        # File format: strategy_weights_{model}_{objective}.json
        self.weight_file = f"strategy_weights_{clean_model_name}_{clean_objective}.json"
        self.update_counter = 0

        logger.info(f"Initialized StrategyWeightOptimizer with {len(strategies)} strategies")
        logger.info(f"Auto-register mode: {'ENABLED' if auto_register else 'DISABLED'}")
        logger.info(f"Model: {self.target_model_name}, Objective: {self.optimization_objective}")
        logger.info(f"Enhanced strategy selection: ENABLED (failure penalty + high-performer boost)")
        logger.info(f"Weight persistence: ENABLED -> {self.weight_file}")
        logger.debug(f"Initial strategies: {', '.join(strategies)}")

        # Load existing weights for this model+task combination
        self._load_model_task_weights()

    def sample_strategy_with_overrides(self, rng, custom_weights: Dict[str, float]) -> str:
        """
        Sample a strategy using custom weights (e.g. per-task weights).
        
        Args:
            rng: Random number generator
            custom_weights: Dictionary of {strategy: weight} to use instead of self.weights
            
        Returns:
            Selected strategy name
        """
        # Ensure we have valid weights
        if not custom_weights:
            return self.sample_strategy(rng)
            
        strategies = list(custom_weights.keys())
        weights = list(custom_weights.values())
        
        # Softmax for stability (optional, but good if weights vary wildly)
        # Here we assume weights are positive floats, so direct sampling is fine
        # unless they are unnormalized.
        
        try:
            choice = rng.choices(strategies, weights=weights, k=1)[0]
            return choice
        except (ValueError, IndexError):
            # Fallback if weights sum to zero or other error
            return self.sample_strategy(rng)

    def get_sampling_probabilities(self) -> Dict[str, float]:
        """
        Get probability distribution for sampling strategies.

        Uses softmax with exploration bonus (Thompson Sampling-inspired).
        Similar to Bayesian Optimization's acquisition function.
        """
        # Add exploration bonus to under-explored strategies
        exploration_adjusted_weights = {}
        total_attempts = sum(s["attempts"] for s in self.strategy_stats.values())
        avg_attempts = total_attempts / len(self.weights) if len(self.weights) > 0 else 0

        for strategy, weight in self.weights.items():
            attempts = self.strategy_stats[strategy]["attempts"]
            # Bonus for under-explored strategies (UCB-like)
            exploration_term = self.exploration_bonus * np.sqrt(
                np.log(total_attempts + 1) / (attempts + 1)
            )
            exploration_adjusted_weights[strategy] = weight + exploration_term

        # Softmax to get probabilities
        weights_array = np.array(list(exploration_adjusted_weights.values()))
        exp_weights = np.exp(weights_array - np.max(weights_array))  # Numerical stability
        probabilities = exp_weights / exp_weights.sum()

        return {strategy: prob for strategy, prob in
                zip(exploration_adjusted_weights.keys(), probabilities)}

    def _uniform_random_strategy(self, rng) -> str:
        """Select a strategy uniformly at random (ablation: no Bayesian weighting)."""
        strategies = list(self.weights.keys())
        return rng.choice(strategies)

    def sample_strategy(self, rng) -> str:
        """
        Sample a strategy based on current weights and exploration.

        Args:
            rng: Random number generator

        Returns:
            Selected strategy name
        """
        if not self.use_bayesian:
            return self._uniform_random_strategy(rng)
        probs = self.get_sampling_probabilities()
        strategies = list(probs.keys())
        probabilities = list(probs.values())

        # Weighted random sampling
        choice = rng.choices(strategies, weights=probabilities, k=1)[0]
        return choice

    def sample_strategy_hybrid(
        self,
        rng,
        analysis: Optional[Dict[str, List[str]]] = None,
        alpha: float = 0.4,  # Historical success weight
        beta: float = 0.4,   # Analysis relevance weight
        gamma: float = 0.2,  # Exploration bonus weight
        auto_register_from_analysis: bool = True,
        blocked_families: Optional[set] = None
    ) -> str:
        """
        Hybrid strategy sampling combining historical success, analysis recommendations, and exploration.

        This is the key innovation: balances what worked before (exploitation) with
        what the analysis recommends (informed exploration) and under-explored strategies.

        Args:
            rng: Random number generator
            analysis: Optional analysis dict with recommended_techniques
            alpha: Weight for historical success (0.0-1.0)
            beta: Weight for analysis relevance (0.0-1.0)
            gamma: Weight for exploration bonus (0.0-1.0)
            auto_register_from_analysis: If True, auto-register strategies from analysis
            blocked_families: Optional set of family names to exclude (exhausted for current task)

        Returns:
            Selected strategy name
        """
        if not self.use_bayesian:
            return self._uniform_random_strategy(rng)
        # Validate weights sum to 1.0
        total_weight = alpha + beta + gamma
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Hybrid weights don't sum to 1.0 ({total_weight:.2f}), normalizing")
            alpha /= total_weight
            beta /= total_weight
            gamma /= total_weight

        # Auto-register strategies from analysis if enabled
        if auto_register_from_analysis and analysis:
            new_strategies = StrategyRelevanceScorer.extract_strategy_names(analysis)
            for strategy in new_strategies:
                if strategy not in self.weights:
                    logger.info(f"Auto-registering strategy from analysis: '{strategy}'")
                    self.weights[strategy] = self.initial_weight
                    self.strategy_stats[strategy] = {"attempts": 0, "successes": 0, "total_score": 0.0, "total_asr": 0.0}

        # Filter out strategies from blocked families (exhausted for stubborn tasks)
        eligible_strategies = list(self.weights.keys())
        if blocked_families:
            eligible_strategies = [
                s for s in eligible_strategies
                if self._get_strategy_family(s) not in blocked_families
            ]
            if not eligible_strategies:
                logger.warning(f"All strategies blocked by family filter {blocked_families}, using full set")
                eligible_strategies = list(self.weights.keys())
            else:
                logger.debug(f"Blocked families {blocked_families}: {len(self.weights)} -> {len(eligible_strategies)} strategies")

        # Compute hybrid scores for each strategy
        hybrid_scores = {}
        total_attempts = sum(s["attempts"] for s in self.strategy_stats.values())

        for strategy in eligible_strategies:
            # Component 1: Historical success (normalized weight)
            historical_score = self.weights[strategy]

            # Component 2: Analysis relevance
            relevance_score = StrategyRelevanceScorer.score_relevance(strategy, analysis)

            # Component 3: Exploration bonus (UCB-style)
            attempts = self.strategy_stats[strategy]["attempts"]
            exploration_score = np.sqrt(np.log(total_attempts + 1) / (attempts + 1))

            # Combine with weighted sum
            hybrid_score = (
                alpha * historical_score +
                beta * relevance_score +
                gamma * exploration_score
            )

            hybrid_scores[strategy] = hybrid_score

        # Normalize to probabilities (softmax)
        strategies = list(hybrid_scores.keys())
        scores_array = np.array(list(hybrid_scores.values()))

        # Softmax
        exp_scores = np.exp(scores_array - np.max(scores_array))  # Numerical stability
        probabilities = exp_scores / exp_scores.sum()

        # Weighted random sampling
        choice = rng.choices(strategies, weights=probabilities, k=1)[0]

        # Log decision for debugging (only for high-relevance cases)
        relevance = StrategyRelevanceScorer.score_relevance(choice, analysis)
        if relevance > 0.5:
            logger.debug(f"Hybrid sampling selected '{choice}' (relevance={relevance:.2f}, historical={self.weights[choice]:.2f})")

        return choice

    def sample_strategy_epsilon_greedy(self, rng, epsilon: Optional[float] = None,
                                        preferred_strategies: Optional[List[str]] = None,
                                        blocked_families: Optional[set] = None) -> str:
        """
        Epsilon-greedy strategy selection using learned weights.

        Args:
            rng: Random number generator
            epsilon: Exploration probability (default: self.epsilon = 0.2)
            preferred_strategies: Optional list of strategies to boost (from task targeting)
            blocked_families: Optional set of family names to exclude (exhausted for current task)

        Returns:
            Selected strategy name
        """
        if not self.use_bayesian:
            return self._uniform_random_strategy(rng)
        if epsilon is None:
            epsilon = self.epsilon

        strategies = list(self.weights.keys())

        # Filter out strategies from blocked families (exhausted for stubborn tasks)
        if blocked_families:
            strategies = [s for s in strategies if self._get_strategy_family(s) not in blocked_families]
            if not strategies:
                logger.warning(f"All strategies blocked by family filter {blocked_families}, using full set")
                strategies = list(self.weights.keys())
            else:
                logger.debug(f"ε-greedy: Blocked families {blocked_families}, {len(strategies)} strategies eligible")

        # With probability epsilon: explore randomly
        if rng.random() < epsilon:
            choice = rng.choice(strategies)
            logger.debug(f"ε-greedy: EXPLORE - random selection '{choice}'")
            return choice

        # With probability 1-epsilon: exploit using LEARNED WEIGHTS
        # Apply temporary boost for preferred strategies (from task targeting)
        effective_weights = dict(self.weights)
        if preferred_strategies:
            boost_factor = 1.5  # 50% boost for category-preferred strategies
            for strat in preferred_strategies:
                if strat in effective_weights:
                    effective_weights[strat] *= boost_factor
            logger.debug(f"ε-greedy: Boosted preferred strategies {preferred_strategies} by {boost_factor}x")

        # Select best strategy by LEARNED WEIGHT (not raw ASR)
        best_strategy = max(effective_weights.items(), key=lambda x: x[1])[0]
        logger.debug(f"ε-greedy: EXPLOIT - selected '{best_strategy}' (weight={effective_weights[best_strategy]:.3f})")

        return best_strategy

    # Strategy family diversity enforcement

    def _get_strategy_family(self, strategy: str) -> str:
        """
        Get the family a strategy belongs to.

        Handles both predefined strategies and LLM-created strategies at runtime.
        Uses multiple matching strategies:
        1. Exact match in family members
        2. Prefix match (e.g., "hierarchical:custom" -> "hierarchical")
        3. Keyword detection (e.g., "custom_roleplay_attack" -> "persona" via "roleplay" keyword)
        4. Falls back to "other" for truly novel strategies
        """
        strategy_lower = strategy.lower()

        # 1. Check exact match in family members
        for family, members in self.STRATEGY_FAMILIES.items():
            if strategy in members or strategy_lower in [m.lower() for m in members]:
                return family

        # 2. Check for partial/prefix matches
        for family, members in self.STRATEGY_FAMILIES.items():
            # Check if strategy starts with family name
            if strategy_lower.startswith(family):
                return family
            # Check colon-separated prefixes (e.g., "hierarchical:custom")
            for member in members:
                if ":" in member:
                    prefix = member.split(":")[0].lower()
                    if strategy_lower.startswith(prefix):
                        return family

        # 3. Keyword-based inference for LLM-created strategies
        for keyword, family in self.FAMILY_KEYWORDS.items():
            if keyword in strategy_lower:
                logger.debug(f"Inferred family '{family}' for strategy '{strategy}' via keyword '{keyword}'")
                return family

        # 4. Fall back to "other" for novel strategies
        return "other"

    def _get_family_distribution(self) -> Dict[str, int]:
        """Get distribution of families in recent strategy selections."""
        family_counts = defaultdict(int)

        for strategy in self.recent_strategies[-self.family_quota_window:]:
            family = self._get_strategy_family(strategy)
            family_counts[family] += 1

        return dict(family_counts)

    def _get_over_represented_families(self) -> set:
        """Get families that exceed the quota threshold."""
        family_counts = self._get_family_distribution()
        window_size = min(len(self.recent_strategies), self.family_quota_window)

        if window_size == 0:
            return set()

        over_represented = set()
        for family, count in family_counts.items():
            ratio = count / window_size
            if ratio >= self.max_family_ratio:
                over_represented.add(family)
                logger.debug(f"Family '{family}' over-represented: {ratio:.1%} >= {self.max_family_ratio:.1%}")

        return over_represented

    def sample_strategy_with_diversity(self, rng, force_diversity: bool = False,
                                        blocked_families: Optional[set] = None) -> str:
        """
        Sample strategy with family diversity enforcement.

        Prevents over-sampling from one strategy family (e.g., hierarchical:*).
        Uses Thompson Sampling among eligible strategies.

        Args:
            rng: Random number generator
            force_diversity: If True, always enforce diversity (e.g., during stagnation)
            blocked_families: Optional set of family names to exclude (exhausted for current task)

        Returns:
            Selected strategy name
        """
        if not self.use_bayesian:
            return self._uniform_random_strategy(rng)
        strategies = list(self.weights.keys())

        # First, filter out exhausted families (blocked_families takes priority)
        if blocked_families:
            strategies = [s for s in strategies if self._get_strategy_family(s) not in blocked_families]
            if not strategies:
                logger.warning(f"All strategies blocked by exhausted families {blocked_families}, using full set")
                strategies = list(self.weights.keys())
            else:
                logger.debug(f"Exhausted family filter: blocked {blocked_families}, {len(strategies)} strategies remain")

        # Get over-represented families
        over_represented = self._get_over_represented_families()

        # During stagnation, be more aggressive about diversity
        if self.stagnation_level >= 2 or force_diversity:
            # At medium/hard stagnation, also penalize the top 2 most used families
            family_counts = self._get_family_distribution()
            if family_counts:
                sorted_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)
                for family, _ in sorted_families[:2]:
                    over_represented.add(family)
                logger.info(f"Stagnation diversity: blocking families {over_represented}")

        # Filter out strategies from over-represented families
        if over_represented:
            eligible_strategies = [
                s for s in strategies
                if self._get_strategy_family(s) not in over_represented
            ]
            if eligible_strategies:
                logger.debug(f"Diversity filter: {len(strategies)} -> {len(eligible_strategies)} eligible strategies")
                strategies = eligible_strategies
            else:
                logger.warning("All strategies filtered by diversity - using full set")

        # HYBRID Thompson + Weights sampling among eligible strategies
        # - Thompson sampling for exploration/exploitation balance
        # - Learned weights as a prior boost (not replacement)
        sampled_values = {}
        for strategy in strategies:
            stats = self.strategy_stats.get(strategy, {"attempts": 0, "successes": 0})
            attempts = stats["attempts"]
            successes = stats["successes"]
            failures = attempts - successes

            alpha = successes + 1
            beta_param = failures + 1

            # Sample from Beta posterior
            thompson_sample = np.random.default_rng(rng.randint(0, 2**31)).beta(alpha, beta_param)

            # Incorporate learned weight as a multiplier (normalized to ~1.0 average)
            weight = self.weights.get(strategy, 1.0)
            avg_weight = sum(self.weights.values()) / len(self.weights) if self.weights else 1.0
            weight_factor = weight / avg_weight  # Normalize so average = 1.0

            # Combine: Thompson * weight_factor (weights boost/penalize the Thompson sample)
            sampled_values[strategy] = thompson_sample * weight_factor

        # Select best
        selected = max(sampled_values.items(), key=lambda x: x[1])[0]

        # Track selection for future diversity enforcement
        self.recent_strategies.append(selected)
        if len(self.recent_strategies) > self.family_quota_window * 2:
            self.recent_strategies = self.recent_strategies[-self.family_quota_window:]

        # Log selection
        family = self._get_strategy_family(selected)
        stats = self.strategy_stats.get(selected, {"attempts": 0, "successes": 0})
        logger.debug(f"Diversity sampling: selected '{selected}' (family={family}, "
                    f"success_rate={stats['successes']/max(stats['attempts'],1):.1%})")

        return selected

    def set_stagnation_level(self, level: int):
        """
        Set stagnation level for adaptive exploration.

        Args:
            level: 0=normal, 1=soft, 2=medium, 3=hard, 4=CHAOS
        """
        old_level = self.stagnation_level
        self.stagnation_level = level
        self.chaos_mode = (level >= 4)

        # Adjust epsilon based on stagnation level
        # Level 3/4 = High exploration/Chaos
        epsilon_map = {0: self.base_epsilon, 1: 0.30, 2: 0.50, 3: 0.80, 4: 1.0}
        self.epsilon = epsilon_map.get(level, self.base_epsilon)

        if level != old_level:
            logger.info(f"Stagnation level: {old_level} -> {level}, epsilon: {self.epsilon:.2f}, chaos: {self.chaos_mode}")

    def reset_stagnation(self):
        """Reset stagnation state after improvement."""
        if self.stagnation_level > 0:
            logger.info(f"Resetting stagnation (was level {self.stagnation_level})")
        self.stagnation_level = 0
        self.epsilon = self.base_epsilon

    def sample_strategy_from_family(self, rng, family: str) -> Optional[str]:
        """
        Sample a strategy from a specific family.

        Used by TaskFamilyExplorer to force exploration of unexplored families
        on stubborn tasks.

        Args:
            rng: Random number generator
            family: Family name to sample from (e.g., "framing", "persona")

        Returns:
            Strategy name from the family, or None if family not found
        """
        # Get strategies in this family that are registered
        family_strategies = []
        for strategy in self.weights.keys():
            if self._get_strategy_family(strategy) == family:
                family_strategies.append(strategy)

        if not family_strategies:
            logger.warning(f"No registered strategies in family '{family}'")
            return None

        # Use Thompson sampling within the family
        sampled_values = {}
        for strategy in family_strategies:
            stats = self.strategy_stats.get(strategy, {"attempts": 0, "successes": 0})
            attempts = stats["attempts"]
            successes = stats["successes"]
            failures = attempts - successes

            alpha = successes + 1
            beta_param = failures + 1

            # Sample from Beta posterior
            thompson_sample = np.random.default_rng(rng.randint(0, 2**31)).beta(alpha, beta_param)

            # Also factor in learned weight
            weight = self.weights.get(strategy, 1.0)
            avg_weight = sum(self.weights.get(s, 1.0) for s in family_strategies) / len(family_strategies)
            weight_factor = weight / avg_weight if avg_weight > 0 else 1.0

            sampled_values[strategy] = thompson_sample * weight_factor

        # Select best within family
        selected = max(sampled_values.items(), key=lambda x: x[1])[0]

        # Track selection
        self.recent_strategies.append(selected)
        if len(self.recent_strategies) > self.family_quota_window * 2:
            self.recent_strategies = self.recent_strategies[-self.family_quota_window:]

        logger.info(f"Forced family '{family}': selected '{selected}' "
                   f"(from {len(family_strategies)} options)")

        return selected

    def reset_weights(self):
        """Reset weights to initial differentiated values (not uniform 1.0).

        This preserves the empirically-derived initial weights from class constant:
        - Word-level strategies: 2.5-3.0 (proven effective)
        - Paper-based strategies: 0.2 (often detected)
        - Unknown strategies: 1.0 (neutral)
        """
        for strategy in self.weights.keys():
            self.weights[strategy] = self.STRATEGY_INITIAL_WEIGHTS.get(strategy, self.initial_weight)

        logger.info("Reset strategy weights to initial differentiated values")

    def get_constrained_strategy_pool(
        self,
        analysis: Optional[Dict[str, List[str]]] = None,
        max_strategies: int = 8,
        min_relevance_threshold: float = 0.3,
        blocked_families: Optional[set] = None
    ) -> List[Tuple[str, float]]:
        """
        Get a constrained pool of strategies for generation, combining analysis + historical success.

        This is used to filter the technique list shown to the LLM, enforcing analysis recommendations.

        Args:
            analysis: Optional analysis dict with recommended_techniques
            max_strategies: Maximum number of strategies to return
            min_relevance_threshold: Minimum relevance score to include analysis-based strategies
            blocked_families: Optional set of family names to exclude (exhausted for current task)

        Returns:
            List of (strategy_name, score) tuples, sorted by score (highest first)
        """
        strategy_scores: List[Tuple[str, float]] = []
        total_attempts = sum(s.get("attempts", 0) for s in self.strategy_stats.values())
        max_weight = max(self.weights.values()) if self.weights else 1.0

        # Filter out strategies from blocked families (exhausted for stubborn tasks)
        eligible_strategies = list(self.weights.keys())
        if blocked_families:
            eligible_strategies = [
                s for s in eligible_strategies
                if self._get_strategy_family(s) not in blocked_families
            ]
            if not eligible_strategies:
                logger.warning(f"Constrained pool: all strategies blocked by family filter {blocked_families}, using full set")
                eligible_strategies = list(self.weights.keys())
            else:
                logger.debug(f"Constrained pool: blocked families {blocked_families}: {len(self.weights)} -> {len(eligible_strategies)} strategies")

        # Score all eligible strategies
        for strategy in eligible_strategies:
            # Get components
            historical_weight = self.weights[strategy]
            # Normalize historical weight so it can't dwarf relevance (weights can vary across runs/files)
            historical_norm = historical_weight / max_weight if max_weight > 0 else 0.0
            relevance = StrategyRelevanceScorer.score_relevance(strategy, analysis)

            # Exploration bonus: promote under-tried strategies (helps avoid monoculture/stagnation)
            attempts = self.strategy_stats.get(strategy, {}).get("attempts", 0)
            exploration = 0.0
            try:
                exploration = float(self.exploration_bonus) * float(
                    np.sqrt(np.log(total_attempts + 1) / (attempts + 1))
                )
            except Exception:
                exploration = 0.0

            # Combine:
            # - Use max() so either relevance or history can promote a strategy
            # - Add exploration so the pool doesn't collapse to the same few strategies during stagnation
            base_score = max(relevance, historical_norm)
            combined_score = base_score + exploration

            # Only include if meets minimum threshold
            if relevance >= min_relevance_threshold or historical_norm >= 0.25 or attempts < 2:
                strategy_scores.append((strategy, combined_score))

        # Sort by score (highest first)
        strategy_scores.sort(key=lambda x: x[1], reverse=True)

        # Default behavior (no/low stagnation): keep pool tight for speed.
        if self.stagnation_level < 2:
            return strategy_scores[:max_strategies]

        # During stagnation: widen + enforce family diversity to avoid getting stuck repeating one family.
        effective_max = max_strategies
        if self.stagnation_level >= 2:
            effective_max = max(effective_max, 12)
        if self.stagnation_level >= 3:
            effective_max = max(effective_max, 16)
        if self.stagnation_level >= 4:
            effective_max = max(effective_max, 18)

        effective_max = min(effective_max, len(strategy_scores))

        # Preserve the top few regardless of family (keep strongest signals)
        preserve_n = min(3, effective_max)
        selected: List[Tuple[str, float]] = []
        selected_names = set()
        for s, sc in strategy_scores[:preserve_n]:
            selected.append((s, sc))
            selected_names.add(s)

        # Group remaining by family and take round-robin across families
        family_to_queue: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for s, sc in strategy_scores[preserve_n:]:
            family = self._get_strategy_family(s)
            family_to_queue[family].append((s, sc))

        family_order = sorted(
            family_to_queue.keys(),
            key=lambda fam: family_to_queue[fam][0][1] if family_to_queue[fam] else 0.0,
            reverse=True
        )

        while len(selected) < effective_max and family_order:
            progressed = False
            for fam in family_order:
                if len(selected) >= effective_max:
                    break
                q = family_to_queue.get(fam, [])
                while q:
                    s, sc = q.pop(0)
                    if s in selected_names:
                        continue
                    selected.append((s, sc))
                    selected_names.add(s)
                    progressed = True
                    break
            if not progressed:
                break

        # Fill any remaining slots by score
        if len(selected) < effective_max:
            for s, sc in strategy_scores:
                if len(selected) >= effective_max:
                    break
                if s in selected_names:
                    continue
                selected.append((s, sc))
                selected_names.add(s)

        # Return in descending score order (stable)
        selected.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            f"Constrained pool (stagnation_level={self.stagnation_level}): "
            f"returning {len(selected[:effective_max])} strategies across {len(family_to_queue)} families "
            f"(effective_max={effective_max})"
        )
        return selected[:effective_max]

    def update_strategy_performance(self,
                                     strategy: str,
                                     success: bool,
                                     metrics: Optional[Dict[str, float]] = None):
        """
        Update strategy performance based on evaluation results.

        Args:
            strategy: Strategy name
            success: Whether attack was successful
            metrics: Optional detailed metrics (ASR, HSR, etc.)
        """
        if strategy not in self.strategy_stats:
            if self.auto_register:
                # Auto-register new strategy discovered at runtime
                logger.info(f"Auto-registering new strategy: '{strategy}'")
                self.strategy_stats[strategy] = {"attempts": 0, "successes": 0, "total_score": 0.0, "total_asr": 0.0}
                self.weights[strategy] = self.initial_weight
            else:
                # Log warning and skip if auto-register is disabled
                logger.warning(f"Unknown strategy: '{strategy}' (auto-register disabled, skipping)")
                return

        stats = self.strategy_stats[strategy]
        stats["attempts"] += 1

        if success:
            stats["successes"] += 1

        if metrics:
            # Track raw ASR for continuous performance measurement
            asr = metrics.get("ASR", 0)
            if "total_asr" not in stats:
                stats["total_asr"] = 0.0  # Handle legacy stats without this field
            stats["total_asr"] += asr

            # Weighted combination of objectives for composite score
            score = (
                0.5 * metrics.get("ASR", 0) +  # Increased ASR weight
                0.5 * metrics.get("XFER", 0)   # XFER for universality
            )
            stats["total_score"] += score

            # Track best combinations
            self.best_combos.append((strategy, score, metrics.copy()))
            self.best_combos.sort(key=lambda x: x[1], reverse=True)
            self.best_combos = self.best_combos[:10]  # Keep top 10

    def get_strategy_summary(self) -> Dict[str, Dict]:
        """
        Get summary of all registered strategies and their performance.
        
        Returns:
            Dictionary mapping strategy names to their stats and weights
        """
        summary = {}
        for strategy in self.weights.keys():
            stats = self.strategy_stats.get(strategy, {"attempts": 0, "successes": 0, "total_score": 0.0})
            summary[strategy] = {
                "weight": self.weights[strategy],
                "attempts": stats["attempts"],
                "successes": stats["successes"],
                "success_rate": stats["successes"] / stats["attempts"] if stats["attempts"] > 0 else 0.0,
                "avg_score": stats["total_score"] / stats["attempts"] if stats["attempts"] > 0 else 0.0
            }
        return summary

    def bayesian_update_weights(self):
        """Update strategy weights using a Bayesian-style blend of success signals."""
        for strategy, stats in self.strategy_stats.items():
            if stats["attempts"] == 0:
                continue

            # Component 1: Bayesian success rate estimation (binary)
            successes = stats["successes"]
            failures = stats["attempts"] - successes

            # Beta posterior parameters (with prior α=1, β=1)
            alpha_param = 1 + successes
            beta_param = 1 + failures

            # Expected success rate (posterior mean)
            binary_success_rate = alpha_param / (alpha_param + beta_param)

            # This provides signal even when binary success threshold isn't met
            total_asr = stats.get("total_asr", 0.0)
            avg_asr = total_asr / stats["attempts"] if stats["attempts"] > 0 else 0.0

            # Component 3: Average composite score
            avg_score = stats["total_score"] / stats["attempts"]

            # - Binary success rate provides exploitation signal
            # - Average ASR provides continuous learning even at low success rates
            # - Weighted combination: 40% binary, 60% continuous ASR
            combined_performance = (
                0.4 * binary_success_rate +
                0.6 * avg_asr  # This is the key improvement - uses raw ASR
            )

            # Scale by avg_score for quality bonus
            new_weight = combined_performance * (1 + avg_score)

            # FAILED STRATEGY PENALTY
            # Heavily penalize strategies with 0% success after 5+ attempts
            attempts = stats["attempts"]
            if attempts >= 5 and successes == 0:
                penalty_factor = 0.05  # Cap at 5% of normal weight
                new_weight = min(new_weight, penalty_factor)
                logger.info(f"Applied failure penalty to {strategy}: weight={penalty_factor:.3f} "
                           f"(attempts={attempts}, successes=0)")

            # HIGH-PERFORMER BOOST
            # Boost promising but undersampled strategies
            success_rate = successes / max(attempts, 1)
            if success_rate >= 0.4 and attempts < 10:
                boost_factor = 1.5
                new_weight *= boost_factor
                logger.info(f"Applied undersampling boost to {strategy}: weight*={boost_factor} "
                           f"(success_rate={success_rate:.3f}, attempts={attempts})")

            # Ensure minimum weight for exploration
            # LOWERED 2024-12-29: Was 0.1 (10%), now 0.01 (1%)
            # Bad strategies need to actually decay to near-zero to stop being selected
            new_weight = max(new_weight, 0.01)

            # Smooth update (exponential moving average)
            old_weight = self.weights[strategy]
            self.weights[strategy] = (
                (1 - self.learning_rate) * old_weight +
                self.learning_rate * new_weight
            )

            logger.debug(f"Weight update for {strategy}: {old_weight:.3f} -> {self.weights[strategy]:.3f} "
                        f"(new={new_weight:.3f}, lr={self.learning_rate})")

        logger.info(f"Updated strategy weights via Bayesian optimization (continuous ASR)")
        self._log_top_strategies()

        # Save weights periodically for cross-run learning
        self._save_weights_if_needed()

    def _log_top_strategies(self, top_k: int = 5, min_attempts: int = 3):
        """Log top-k strategies by weight with minimum attempts filter."""
        sorted_strategies = sorted(
            self.weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        logger.info(f"Top strategies by weight (showing those with ≥{min_attempts} attempts):")
        shown = 0
        for strategy, weight in sorted_strategies:
            stats = self.strategy_stats[strategy]
            if stats["attempts"] >= min_attempts:
                success_rate = stats["successes"] / stats["attempts"] * 100
                avg_asr = stats.get("total_asr", 0.0) / stats["attempts"] * 100 if stats["attempts"] > 0 else 0.0
                logger.info(
                    f"  {shown+1}. {strategy}: weight={weight:.3f}, "
                    f"success_rate={success_rate:.1f}%, avg_asr={avg_asr:.1f}% ({stats['attempts']} attempts)"
                )
                shown += 1
                if shown >= top_k:
                    break
        
        # If no strategies meet threshold, show warning
        if shown == 0:
            logger.info(f"  (No strategies have ≥{min_attempts} attempts yet - still exploring)")

    def _load_model_task_weights(self):
        """Load model and task specific strategy weights"""
        if os.path.exists(self.weight_file):
            try:
                with open(self.weight_file, 'r') as f:
                    saved_data = json.load(f)

                # Validate this data is for the same model+task
                if (saved_data.get('target_model') == self.target_model_name and
                    saved_data.get('optimization_objective') == self.optimization_objective):

                    # Load weights
                    saved_weights = saved_data.get('weights', {})
                    for strategy, weight in saved_weights.items():
                        if strategy in self.weights:
                            old_weight = self.weights[strategy]
                            self.weights[strategy] = weight
                            logger.debug(f"Loaded weight for {strategy}: {old_weight:.3f} -> {weight:.3f}")

                    # Load historical stats for better initialization
                    saved_stats = saved_data.get('stats', {})
                    for strategy, stats in saved_stats.items():
                        if strategy in self.strategy_stats:
                            # Use historical data as prior knowledge
                            self.strategy_stats[strategy]["attempts"] = stats.get("attempts", 0)
                            self.strategy_stats[strategy]["successes"] = stats.get("successes", 0)
                            self.strategy_stats[strategy]["total_asr"] = stats.get("total_asr", 0.0)
                            self.strategy_stats[strategy]["total_score"] = stats.get("total_score", 0.0)

                    logger.info(f"Loaded {len(saved_weights)} weights for {self.target_model_name} + {self.optimization_objective}")
                else:
                    logger.warning(f"Weight file {self.weight_file} is for different model/task combination")

            except Exception as e:
                logger.warning(f"Failed to load weights for {self.target_model_name} + {self.optimization_objective}: {e}")

    def _save_model_task_weights(self):
        """Save model and task specific weights and stats"""
        try:
            # Save with full context
            save_data = {
                'weights': {k: v for k, v in self.weights.items() if abs(v - 1.0) > 0.1},
                'stats': {k: v for k, v in self.strategy_stats.items()
                         if v["attempts"] > 0},
                'target_model': self.target_model_name,
                'optimization_objective': self.optimization_objective,
                'last_updated': time.time(),
                'framework_version': '1.0',  # For future compatibility
                'total_updates': self.update_counter
            }

            with open(self.weight_file, 'w') as f:
                json.dump(save_data, f, indent=2)

            logger.debug(f"Saved weights for {self.target_model_name} + {self.optimization_objective}")

        except Exception as e:
            logger.warning(f"Failed to save weights: {e}")

    def _save_weights_if_needed(self):
        """Save weights every 10 updates"""
        self.update_counter += 1
        if self.update_counter % 10 == 0:
            self._save_model_task_weights()

    def get_optimization_context(self) -> Dict[str, str]:
        """Get current optimization context"""
        return {
            'model': self.target_model_name,
            'objective': self.optimization_objective,
            'weight_file': self.weight_file
        }


class ICLExampleOptimizer:
    """
    Optimize few-shot example selection (DSPy's demonstration bootstrapping).

    Similar to BootstrapFewShot in DSPy - selects best demonstrations
    based on their performance in prompts.
    """

    def __init__(self, jailbreak_examples: List[Dict], max_examples: int = 50):
        """
        Initialize ICL example optimizer.

        Args:
            jailbreak_examples: Pool of jailbreak demonstrations
            max_examples: Maximum examples to track
        """
        self.example_pool = jailbreak_examples
        self.max_examples = max_examples

        # Track which examples lead to successful generations
        self.example_scores = defaultdict(lambda: {"used": 0, "successful": 0, "score": 0.0})

        logger.info(f"Initialized ICLExampleOptimizer with {len(jailbreak_examples)} examples")

    def sample_examples(self,
                        k: int,
                        strategy: str,
                        rng,
                        temperature: float = 1.0) -> List[Dict]:
        """
        Sample k examples using Thompson Sampling.

        Args:
            k: Number of examples to sample
            strategy: Current mutation strategy (for filtering)
            rng: Random number generator
            temperature: Sampling temperature (higher = more exploration)

        Returns:
            List of sampled examples
        """
        # Filter examples by strategy if available
        filtered_pool = [
            ex for ex in self.example_pool
            if ex.get("strategy", "unknown") == strategy or
               strategy in ex.get("strategies", [])
        ]

        if not filtered_pool:
            filtered_pool = self.example_pool  # Fallback to all

        if len(filtered_pool) <= k:
            return filtered_pool

        # Calculate scores with Beta distribution (Bayesian)
        scores = []
        for ex in filtered_pool:
            ex_id = ex.get("id", str(ex))
            stats = self.example_scores[ex_id]

            if stats["used"] == 0:
                # High exploration bonus for unseen examples
                score = 1.0
            else:
                # Beta posterior mean
                alpha = 1 + stats["successful"]
                beta = 1 + (stats["used"] - stats["successful"])
                score = alpha / (alpha + beta)

            # Temperature-scaled score
            scores.append(score ** (1.0 / temperature))

        # Weighted sampling
        scores = np.array(scores)
        probabilities = scores / scores.sum()

        indices = rng.choices(
            range(len(filtered_pool)),
            weights=probabilities,
            k=k
        )

        return [filtered_pool[i] for i in indices]

    def update_example_performance(self,
                                    examples: List[Dict],
                                    success: bool,
                                    score: float = 0.0):
        """
        Update performance tracking for used examples.

        Args:
            examples: Examples that were used
            success: Whether generation was successful
            score: Optional quality score
        """
        for ex in examples:
            ex_id = ex.get("id", str(ex))
            self.example_scores[ex_id]["used"] += 1
            if success:
                self.example_scores[ex_id]["successful"] += 1
            self.example_scores[ex_id]["score"] += score

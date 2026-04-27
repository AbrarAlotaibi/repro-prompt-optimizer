"""
REPRO Data Models

Core data structures for REPRO attack generation and evaluation.
Simplified for should_refuse (jailbreak) tasks only.
"""

import hashlib
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class Environment:
    """Evaluation environment definition for target prompts."""
    system: str
    developer: str


@dataclass
class Conversation:
    """Conversation wrapper (use `user`; system/developer are deprecated)."""
    system: str = ""     # Deprecated; configure at model level instead.
    developer: str = ""  # Deprecated; do not use.
    user: str = ""

    def __post_init__(self):
        """Warn if deprecated fields are used."""
        if self.system or self.developer:
            warnings.warn(
                "\n"
                "Conversation.system and Conversation.developer are DEPRECATED.\n"
                "\n"
                "These fields inject artificial prompts that don't reflect actual model behavior.\n"
                "Following benchmark best practices (JailbreakBench, HarmBench), system prompts\n"
                "should be configured at the MODEL level via environment variables:\n"
                "\n"
                "  SYSTEM_PROMPT_MODE=empty   (default for red teaming)\n"
                "  SYSTEM_PROMPT_MODE=minimal\n"
                "  SYSTEM_PROMPT_MODE=custom\n"
                "\n"
                "See docs/SYSTEM_PROMPTS.md for details.\n",
                DeprecationWarning,
                stacklevel=2
            )


@dataclass
class Task:
    """Evaluation task definition (should_refuse only)."""
    type: str  # Always "should_refuse" for jailbreak testing
    user_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRow:
    """Single evaluation result."""
    env_idx: int
    task_type: str
    asr: int  # Attack Success Rate (1 if jailbreak succeeded, 0 otherwise)
    reply: str
    user_request: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackCandidate:
    """Attack prompt candidate in the evolutionary algorithm."""
    text: str
    meta: Dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: hashlib.sha1(str(random.random()).encode()).hexdigest()[:8])
    parent_id: Optional[str] = None
    goal: Optional[str] = None  # Original objective to prevent goal drift
    # Cache fields (excluded from repr/comparison).
    _signature_cache: Optional[str] = field(default=None, repr=False, compare=False)
    _word_set_cache: Optional[set] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Populate original_family for downstream diversity tracking when missing."""
        if not isinstance(self.meta, dict):
            self.meta = {}
        if "original_family" in self.meta:
            return
        strategy = (
            self.meta.get("edit")
            or self.meta.get("seed_strategy")
            or self.meta.get("optimizer_strategy_hint")
            or self.meta.get("strategy")
        )
        if strategy:
            from repro.algorithm.strategy_families import get_strategy_family
            self.meta["original_family"] = get_strategy_family(strategy)

    def signature(self) -> str:
        """
        Fuzzy signature for duplicate detection.
        Normalizes text to catch minor variations while preserving semantic differences.
        Uses caching for performance.
        """
        if self._signature_cache is not None:
            return self._signature_cache
        # Normalize: lowercase, remove extra whitespace, remove punctuation at boundaries
        normalized = self.text.lower().strip()
        # Remove multiple spaces/newlines
        normalized = ' '.join(normalized.split())
        # Remove trailing punctuation (but keep internal punctuation)
        normalized = normalized.rstrip('.,!?;:')
        # Cache the result (use object.__setattr__ to bypass frozen if needed)
        object.__setattr__(self, '_signature_cache', normalized)
        return normalized

    def _get_word_set(self) -> set:
        """Get cached word set for similarity calculations."""
        if self._word_set_cache is not None:
            return self._word_set_cache
        word_set = set(self.text.lower().split())
        object.__setattr__(self, '_word_set_cache', word_set)
        return word_set

    def similarity_to_any(self, others: list, threshold: float = 0.80) -> bool:
        """
        Check if this candidate is a near-duplicate of ANY candidate in a list.

        Efficient batch check that returns True on first match.

        Args:
            others: List of AttackCandidates to compare against
            threshold: Similarity threshold for near-duplicate detection

        Returns:
            True if near-duplicate of any candidate in the list
        """
        my_sig = self.signature()
        my_words = self._get_word_set()

        for other in others:
            # Fast path: exact signature match
            if my_sig == other.signature():
                return True
            # Similarity check using cached word sets
            other_words = other._get_word_set()
            if not my_words and not other_words:
                return True  # Both empty = duplicate
            if my_words and other_words:
                intersection = len(my_words & other_words)
                union = len(my_words | other_words)
                if union > 0 and (intersection / union) >= threshold:
                    return True
        return False


@dataclass
class LogRow:
    """Training log entry."""
    iteration: int
    best_asr: float  # Best ASR in current iteration's Pareto front
    best_xfer: float  # Best XFER in current iteration's Pareto front
    pareto_size: int
    pop_size: int
    children_generated: int
    timestamp: float
    global_best_asr: float = 0.0  # Best ASR ever seen (monotonically increasing)
    global_best_xfer: float = 0.0  # Best XFER ever seen (monotonically increasing)

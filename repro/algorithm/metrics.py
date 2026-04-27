"""Optimization metrics and tracking utilities."""

import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class EfficiencyMetrics:
    """Tracks optimization efficiency metrics for REPRO."""
    # Convergence
    iterations_completed: int = 0
    iteration_to_first_success: Optional[int] = None
    iteration_to_50_asr: Optional[int] = None
    iteration_to_threshold: Optional[int] = None

    # Cost
    candidates_evaluated: int = 0
    unique_candidates: int = 0
    target_model_calls: int = 0
    reflective_llm_calls: int = 0

    # Time
    total_wall_time_seconds: float = 0.0
    time_to_first_success: Optional[float] = None
    avg_time_per_iteration: float = 0.0

    # Quality
    unique_successful_attacks: int = 0
    pareto_front_size: int = 0
    final_asr: float = 0.0
    final_xfer: float = 0.0

    # Mode
    optimize_mode: str = "explore"
    success_threshold: float = 0.5

    # History
    asr_history: List[float] = field(default_factory=list)
    xfer_history: List[float] = field(default_factory=list)


@dataclass
class DuplicateTracker:
    """Thread-safe duplicate proposal tracking for diversity monitoring."""
    proposals_total: int = 0
    proposals_unique: int = 0
    proposals_duplicate: int = 0
    duplicate_ratios: List[float] = field(default_factory=list)
    recent_signatures: List[str] = field(default_factory=list)
    recent_candidates: List = field(default_factory=list)
    cache_size: int = 500
    high_threshold: float = 0.30

    def __post_init__(self):
        # Use RLock to avoid re-entrancy deadlocks.
        self._lock = threading.RLock()

    def record_proposal(self, candidate, is_duplicate: bool) -> None:
        """Record a proposal attempt (thread-safe)."""
        with self._lock:
            self.proposals_total += 1
            if is_duplicate:
                self.proposals_duplicate += 1
            else:
                self.proposals_unique += 1
                sig = candidate.signature()
                if len(self.recent_signatures) >= self.cache_size:
                    self.recent_signatures = self.recent_signatures[-(self.cache_size - 1):]
                    self.recent_candidates = self.recent_candidates[-(self.cache_size - 1):]
                self.recent_signatures.append(sig)
                self.recent_candidates.append(candidate)

    def get_current_ratio(self) -> float:
        """Get current duplicate ratio (thread-safe)."""
        with self._lock:
            if self.proposals_total == 0:
                return 0.0
            return self.proposals_duplicate / self.proposals_total

    def finalize_generation(self) -> float:
        """Finalize generation stats and reset counters. Returns duplicate ratio."""
        with self._lock:
            ratio = self.proposals_duplicate / self.proposals_total if self.proposals_total > 0 else 0.0
            self.duplicate_ratios.append(ratio)
            self.proposals_total = 0
            self.proposals_unique = 0
            self.proposals_duplicate = 0
            return ratio

    def get_rolling_ratio(self, window: int = 3) -> float:
        """Get rolling average duplicate ratio (thread-safe)."""
        with self._lock:
            if not self.duplicate_ratios:
                return 0.0
            recent = self.duplicate_ratios[-window:]
            return sum(recent) / len(recent)

    def get_recent_signatures(self, n: int = 50) -> List[str]:
        """Get last N unique signatures (thread-safe)."""
        with self._lock:
            return self.recent_signatures[-n:].copy()

    def check_near_duplicate(self, candidate, threshold: float = 0.80) -> bool:
        """Check if candidate is near-duplicate of recent candidates (thread-safe)."""
        with self._lock:
            sig = candidate.signature()
            if sig in self.recent_signatures:
                return True
            recent_cands = self.recent_candidates[-50:].copy()

        # Similarity computation can be done outside the lock.
        if hasattr(candidate, 'similarity_to_any') and recent_cands:
            return candidate.similarity_to_any(recent_cands, threshold=threshold)
        return False

    def get_stats_summary(self) -> Dict:
        """Get summary statistics."""
        with self._lock:
            current_ratio = (self.proposals_duplicate / self.proposals_total) if self.proposals_total else 0.0
            rolling_ratio = self.get_rolling_ratio()
            cache_size = len(self.recent_signatures)
            return {
                "total_proposals": self.proposals_total,
                "unique_proposals": self.proposals_unique,
                "duplicate_proposals": self.proposals_duplicate,
                "current_ratio": current_ratio,
                "rolling_ratio": rolling_ratio,
                "cache_size": cache_size,
            }

"""REPRO configuration."""

import os
from typing import Literal, Optional


# Parsing configuration
SELF_CORRECTION_TIMEOUT: float = 30.0

# Token budgets for LLM generations.
GENERATION_MAX_TOKENS: int = int(os.getenv("GENERATION_MAX_TOKENS", "8000"))
ANALYSIS_MAX_TOKENS: int = int(os.getenv("ANALYSIS_MAX_TOKENS", "2500"))
REFINER_MAX_TOKENS: int = int(os.getenv("REFINER_MAX_TOKENS", "2500"))
CROSSOVER_MAX_TOKENS: int = int(os.getenv("CROSSOVER_MAX_TOKENS", "1800"))
JUDGE_MAX_TOKENS: int = int(os.getenv("JUDGE_MAX_TOKENS", "250"))


# Hybrid strategy optimization configuration.
HYBRID_STRATEGY_CONFIG = {
    "enabled": True,
    "alpha": float(os.getenv("HYBRID_ALPHA", "0.2")),
    "beta": float(os.getenv("HYBRID_BETA", "0.5")),
    "gamma": float(os.getenv("HYBRID_GAMMA", "0.3")),
    "max_constrained_pool": int(os.getenv("HYBRID_MAX_POOL", "12")),
    "min_relevance_threshold": float(os.getenv("HYBRID_MIN_RELEVANCE", "0.3")),
    "auto_register": True,
    "verbose": bool(os.getenv("HYBRID_VERBOSE", "True").lower() == "true"),
}


# Deduplication and diversity configuration.
DEDUP_CONFIG = {
    "similarity_threshold": float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.70")),
    "use_similarity": bool(os.getenv("DEDUP_USE_SIMILARITY", "True").lower() == "true"),
    "signature_cache_size": int(os.getenv("DEDUP_CACHE_SIZE", "500")),
    "high_duplicate_threshold": float(os.getenv("DEDUP_HIGH_THRESHOLD", "0.30")),
    "use_diverse_filling": bool(os.getenv("DEDUP_DIVERSE_FILL", "True").lower() == "true"),
    "use_llm_self_crossover": bool(os.getenv("DEDUP_LLM_CROSSOVER", "True").lower() == "true"),
    "diversity_temperatures": [0.85, 0.95, 1.0, 1.1, 1.15, 1.2],
    "verbose": bool(os.getenv("DEDUP_VERBOSE", "False").lower() == "true"),
}


def get_dedup_config() -> dict:
    """Get deduplication and diversity configuration."""
    return DEDUP_CONFIG


LOG_PARSE_STATS: bool = True
TRACK_FORMAT_METRICS: bool = True


class ParseMetrics:
    """Track parsing success/failure metrics."""

    def __init__(self):
        self.xml_attempts = 0
        self.xml_success = 0
        self.xml_total_time = 0.0
        self.fallback_to_retry = 0
        self.error_types: dict[str, int] = {}

    def record_xml_parse(self, success: bool, duration: float):
        self.xml_attempts += 1
        if success:
            self.xml_success += 1
        self.xml_total_time += duration

    # Alias for backward compatibility (callers still use the old name)
    record_json_parse = record_xml_parse

    def record_fallback(self, mode: str):
        if mode == "retry":
            self.fallback_to_retry += 1

    def record_error(self, error_type: str):
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

    def get_summary(self) -> dict:
        rate = (self.xml_success / self.xml_attempts * 100) if self.xml_attempts > 0 else 0.0
        avg = (self.xml_total_time / self.xml_attempts * 1000) if self.xml_attempts > 0 else 0.0
        return {
            "xml": {
                "attempts": self.xml_attempts,
                "success": self.xml_success,
                "success_rate": f"{rate:.1f}%",
                "avg_parse_time_ms": f"{avg:.2f}",
            },
            "fallbacks": {"retry_attempts": self.fallback_to_retry},
            "error_types": self.error_types,
        }


_global_metrics: Optional[ParseMetrics] = None


def get_metrics() -> ParseMetrics:
    """Get global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = ParseMetrics()
    return _global_metrics

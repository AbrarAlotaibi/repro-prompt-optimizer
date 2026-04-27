"""Attack Amortization Cache - Transfer successful attacks between tasks.

This module provides persistent caching of successful attacks for warm-starting
optimization on new tasks. Instead of solving each task independently, successful
attacks are stored and retrieved as seeds for similar tasks.

Key features:
- Category-based indexing for fast retrieval of relevant attacks
- ASR-weighted scoring for quality-aware seed selection
- Optional embedding-based similarity for semantic matching
- Deduplication to avoid redundant cache entries
- Model-specific caching for targeted transfer
"""

import json
import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict

from repro.algorithm.models import AttackCandidate

logger = logging.getLogger(__name__)

# Minimum ASR to cache an attack (default for standard categories)
MIN_CACHE_ASR = 0.5

# Maximum cache entries per category (prevents unbounded growth)
MAX_ENTRIES_PER_CATEGORY = 50

# Diversity threshold - attacks must be this different to both be cached
MIN_ATTACK_DIVERSITY = 0.3

# Category-specific minimum ASR thresholds
# Hard categories have lower thresholds to cache partial successes for warm-starting
CATEGORY_MIN_ASR = {
    "explosives": 0.3,
    "weapons": 0.35,
    "violence": 0.35,
    "drugs": 0.35,
    "illegal_services": 0.4,
    "harassment": 0.4,
    "hacking": 0.45,
    "fraud": 0.45,
    "general": 0.5,
}

# Category-specific maximum entries (store more for hard categories)
CATEGORY_MAX_ENTRIES = {
    "explosives": 100,
    "weapons": 100,
    "violence": 80,
    "drugs": 80,
    "illegal_services": 60,
    "harassment": 60,
    "general": 50,
}

# Related categories for cross-category transfer
# Attacks that work on one category often transfer to related ones
RELATED_CATEGORIES = {
    "explosives": ["weapons", "violence", "hacking"],
    "weapons": ["explosives", "violence"],
    "violence": ["weapons", "harassment", "explosives"],
    "drugs": ["illegal_services", "fraud"],
    "hacking": ["fraud", "illegal_services", "explosives"],
    "fraud": ["hacking", "illegal_services", "drugs"],
    "illegal_services": ["fraud", "drugs", "hacking"],
    "harassment": ["violence", "hate_content"],
    "hate_content": ["harassment", "violence"],
}


@dataclass
class CachedAttack:
    """A successful attack stored for transfer to new tasks."""
    attack_text: str
    task_text: str
    task_category: str
    asr: float
    strategy_name: str
    target_model: str
    timestamp: float = 0.0
    attack_hash: str = ""
    # Optional: embedding for similarity search
    task_embedding: Optional[List[float]] = None

    def __post_init__(self):
        if not self.attack_hash:
            self.attack_hash = hashlib.md5(self.attack_text.encode()).hexdigest()[:16]


class AmortizationCache:
    """Persistent cache of successful attacks for warm-starting new tasks.

    Usage:
        cache = AmortizationCache("attack_cache.jsonl")

        # After successful optimization
        cache.save_attack(attack, task_text, asr=0.8, target_model="llama3")

        # Before new optimization
        seeds = cache.retrieve_seeds(new_task_text, top_k=5)
        optimizer = REPROAttackOptimizer(initial_seeds=seeds, ...)
    """

    def __init__(
        self,
        cache_file: str = "attack_cache.jsonl",
        cache_dir: Optional[str] = None,
        embedding_model: Optional[Any] = None,
        min_asr: float = MIN_CACHE_ASR,
        max_per_category: int = MAX_ENTRIES_PER_CATEGORY,
    ):
        """Initialize the amortization cache.

        Args:
            cache_file: Filename for the cache (will be placed in cache_dir)
            cache_dir: Directory for cache files (default: current dir)
            embedding_model: Optional model for semantic similarity
            min_asr: Minimum ASR to cache an attack
            max_per_category: Maximum entries per task category
        """
        if cache_dir:
            self.cache_file = Path(cache_dir) / cache_file
        else:
            self.cache_file = Path(cache_file)

        self.embedding_model = embedding_model
        self.min_asr = min_asr
        self.max_per_category = max_per_category

        # In-memory cache indexed by category
        self.cache: List[CachedAttack] = []
        self.by_category: Dict[str, List[CachedAttack]] = defaultdict(list)
        self.seen_hashes: set = set()

        self._load_cache()
        logger.info(f"AmortizationCache initialized: {len(self.cache)} attacks from {self.cache_file}")

    def _load_cache(self):
        """Load cache from disk."""
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = CachedAttack(**data)
                        self._add_to_memory(entry)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Skipping malformed cache entry: {e}")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")

    def _add_to_memory(self, entry: CachedAttack):
        """Add entry to in-memory cache structures."""
        if entry.attack_hash in self.seen_hashes:
            return False

        self.cache.append(entry)
        self.by_category[entry.task_category].append(entry)
        self.seen_hashes.add(entry.attack_hash)
        return True

    def _get_category(self, task_text: str) -> str:
        """Get category for a task (with lazy import to avoid circular deps)."""
        try:
            from repro.algorithm.task_categories import categorize_task
            category = categorize_task(task_text)
            return category.name if hasattr(category, 'name') else str(category)
        except ImportError:
            # Fallback: simple keyword-based categorization
            task_lower = task_text.lower()
            if any(k in task_lower for k in ['hack', 'exploit', 'vulnerability']):
                return 'hacking'
            elif any(k in task_lower for k in ['bomb', 'explosive', 'weapon']):
                return 'weapons'
            elif any(k in task_lower for k in ['drug', 'synthesis', 'chemical']):
                return 'drugs'
            elif any(k in task_lower for k in ['harm', 'kill', 'attack']):
                return 'violence'
            return 'general'

    def save_attack(
        self,
        attack: AttackCandidate,
        task_text: str,
        asr: float,
        target_model: str,
        strategy_name: Optional[str] = None,
    ) -> bool:
        """Save a successful attack to the cache.

        Args:
            attack: The successful attack candidate
            task_text: The task/goal this attack succeeded on
            asr: Attack success rate achieved
            target_model: Target model name
            strategy_name: Optional override for strategy name

        Returns:
            True if attack was cached, False if filtered out
        """
        # Get category first to use category-specific thresholds
        category = self._get_category(task_text)

        # Use category-specific minimum ASR threshold (lower for hard categories)
        category_min_asr = CATEGORY_MIN_ASR.get(category, self.min_asr)
        if asr < category_min_asr:
            logger.debug(f"Attack not cached: ASR {asr:.2%} < {category_min_asr:.2%} (category: {category})")
            return False

        # Get strategy name from attack meta or parameter
        if strategy_name is None:
            strategy_name = attack.meta.get("seed_strategy", "unknown") if attack.meta else "unknown"

        import time
        entry = CachedAttack(
            attack_text=attack.text,
            task_text=task_text[:500],  # Truncate long tasks
            task_category=category,
            asr=asr,
            strategy_name=strategy_name,
            target_model=target_model,
            timestamp=time.time(),
            task_embedding=self._embed(task_text) if self.embedding_model else None
        )

        # Check for duplicates
        if entry.attack_hash in self.seen_hashes:
            logger.debug(f"Attack already in cache: {entry.attack_hash}")
            return False

        # Check category capacity (use category-specific limits for hard categories)
        category_entries = self.by_category[category]
        category_max = CATEGORY_MAX_ENTRIES.get(category, self.max_per_category)
        if len(category_entries) >= category_max:
            # Evict lowest ASR entry
            category_entries.sort(key=lambda x: x.asr)
            evicted = category_entries.pop(0)
            self.cache.remove(evicted)
            self.seen_hashes.discard(evicted.attack_hash)
            logger.debug(f"Evicted low-ASR entry from {category}: {evicted.asr:.2%}")

        # Add to memory
        self._add_to_memory(entry)

        # Persist to disk
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'a') as f:
                f.write(json.dumps(asdict(entry)) + '\n')
            logger.info(f"Cached attack for '{category}' (ASR={asr:.2%}, strategy={strategy_name})")
            return True
        except Exception as e:
            logger.error(f"Failed to persist cache entry: {e}")
            return False

    def retrieve_seeds(
        self,
        task_text: str,
        top_k: int = 5,
        same_model_only: bool = False,
        target_model: Optional[str] = None,
        same_category_only: bool = False,
        cross_category: bool = True,  # Default changed to True for better transfer
        category_bonus: float = 0.5,
        related_category_bonus: float = 0.25,  # New: bonus for related categories
        diversity_weight: float = 0.2,
    ) -> List[AttackCandidate]:
        """Retrieve top-k similar attacks as seeds for a new task.

        Args:
            task_text: The new task to find seeds for
            top_k: Number of seeds to retrieve
            same_model_only: Only retrieve attacks for same target model
            target_model: Target model name (required if same_model_only=True)
            same_category_only: Only retrieve attacks from same task category
            cross_category: Allow cross-category retrieval with related category bonuses
            category_bonus: Score bonus for same-category attacks
            related_category_bonus: Score bonus for related category attacks
            diversity_weight: Weight for diversity in selection

        Returns:
            List of AttackCandidate objects for warm-starting
        """
        if not self.cache:
            logger.debug("Cache empty, no seeds to retrieve")
            return []

        task_category = self._get_category(task_text)
        query_embedding = self._embed(task_text) if self.embedding_model else None

        # Get related categories for cross-category transfer
        related_categories = set(RELATED_CATEGORIES.get(task_category, []))

        # Score each cached attack
        scored: List[Tuple[float, CachedAttack]] = []

        for entry in self.cache:
            # Filter by model if requested
            if same_model_only and target_model and entry.target_model != target_model:
                continue
            if same_category_only and entry.task_category != task_category:
                continue

            # Base score is ASR
            score = entry.asr

            # Category-based scoring with related category support
            if entry.task_category == task_category:
                # Full bonus for same category
                score += category_bonus
            elif cross_category and entry.task_category in related_categories:
                # Partial bonus for related categories (attacks often transfer)
                score += related_category_bonus

            # Embedding similarity bonus
            if query_embedding and entry.task_embedding:
                similarity = self._cosine_similarity(query_embedding, entry.task_embedding)
                score += similarity * 0.3

            # Recency bonus (mild preference for newer attacks)
            if entry.timestamp > 0:
                import time
                age_days = (time.time() - entry.timestamp) / 86400
                recency_bonus = max(0, 0.1 * (1 - age_days / 30))  # Decays over 30 days
                score += recency_bonus

            scored.append((score, entry))

        if not scored:
            logger.debug(f"No matching attacks found for task category '{task_category}'")
            return []

        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Select with diversity (avoid very similar attacks)
        selected: List[CachedAttack] = []
        selected_texts: List[str] = []

        for score, entry in scored:
            if len(selected) >= top_k:
                break

            # Check diversity against already selected
            is_diverse = True
            for existing_text in selected_texts:
                if self._text_similarity(entry.attack_text, existing_text) > (1 - MIN_ATTACK_DIVERSITY):
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(entry)
                selected_texts.append(entry.attack_text)

        # Convert to AttackCandidates
        seeds = []
        for entry in selected:
            candidate = AttackCandidate(
                text=entry.attack_text,
                meta={
                    "seed_strategy": f"amortized_{entry.strategy_name}",
                    "amortization_source": "cache",
                    "source_task": entry.task_text[:100],
                    "source_category": entry.task_category,
                    "source_asr": entry.asr,
                    "source_model": entry.target_model,
                }
            )
            seeds.append(candidate)

        logger.info(f"Retrieved {len(seeds)} warm-start seeds for category '{task_category}'")
        return seeds

    def retrieve_by_category(
        self,
        category: str,
        top_k: int = 5,
    ) -> List[AttackCandidate]:
        """Retrieve top-k attacks for a specific category.

        Args:
            category: Task category name
            top_k: Number of seeds to retrieve

        Returns:
            List of AttackCandidate objects
        """
        entries = self.by_category.get(category, [])
        if not entries:
            return []

        # Sort by ASR and take top-k
        sorted_entries = sorted(entries, key=lambda x: x.asr, reverse=True)[:top_k]

        seeds = []
        for entry in sorted_entries:
            candidate = AttackCandidate(
                text=entry.attack_text,
                meta={
                    "seed_strategy": f"amortized_{entry.strategy_name}",
                    "amortization_source": "category_lookup",
                    "source_category": entry.task_category,
                    "source_asr": entry.asr,
                }
            )
            seeds.append(candidate)

        return seeds

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "total_entries": len(self.cache),
            "categories": {},
            "models": defaultdict(int),
            "strategies": defaultdict(int),
            "avg_asr": 0.0,
        }

        if not self.cache:
            return stats

        for entry in self.cache:
            stats["models"][entry.target_model] += 1
            stats["strategies"][entry.strategy_name] += 1

        for cat, entries in self.by_category.items():
            stats["categories"][cat] = {
                "count": len(entries),
                "avg_asr": sum(e.asr for e in entries) / len(entries) if entries else 0,
                "best_asr": max(e.asr for e in entries) if entries else 0,
            }

        stats["avg_asr"] = sum(e.asr for e in self.cache) / len(self.cache)
        stats["models"] = dict(stats["models"])
        stats["strategies"] = dict(stats["strategies"])

        return stats

    def clear(self):
        """Clear the cache (both memory and disk)."""
        self.cache.clear()
        self.by_category.clear()
        self.seen_hashes.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cache cleared")

    def _embed(self, text: str) -> Optional[List[float]]:
        """Get embedding for text (if model available)."""
        if self.embedding_model is None:
            return None
        try:
            return self.embedding_model.encode(text).tolist()
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            import numpy as np
            a_arr, b_arr = np.array(a), np.array(b)
            dot = np.dot(a_arr, b_arr)
            norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
            return float(dot / (norm + 1e-8))
        except Exception:
            return 0.0

    def _text_similarity(self, a: str, b: str) -> float:
        """Simple text similarity based on token overlap."""
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return intersection / union if union > 0 else 0.0

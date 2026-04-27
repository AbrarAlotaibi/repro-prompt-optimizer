"""Small shared helpers for REPRO algorithm components."""

from typing import Dict

from repro.algorithm.models import AttackCandidate


def extract_task_results(candidate: AttackCandidate) -> Dict[int, bool]:
    """
    Extract per-task results from candidate metadata.

    Returns:
        Dict mapping task_idx -> success_bool, or empty dict if not available
    """
    if hasattr(candidate, 'meta') and candidate.meta and 'per_task_results' in candidate.meta:
        return candidate.meta['per_task_results']
    return {}

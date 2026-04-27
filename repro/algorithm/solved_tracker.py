"""
Incremental Solved Task Tracker for REPRO Explore Mode.

Tracks and saves attacks that solve individual tasks as they're discovered,
so partial runs still yield useful results.
"""

import json
import os
import time
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

from repro.algorithm.models import AttackCandidate, Task


@dataclass
class SolvedEntry:
    """Entry for a solved task."""
    task_idx: int
    task_text: str
    attack_text: str
    attack_id: str
    metrics: Dict[str, float]
    strategy: str
    iteration: int
    timestamp: float


class SolvedTaskTracker:
    """
    Tracks and incrementally saves attacks that solve individual tasks.

    Writes to `solved_tasks.jsonl` immediately when a task is solved,
    so even if the run is interrupted, partial results are preserved.
    """

    def __init__(self, report_dir: str, tasks: List[Task]):
        """
        Initialize the tracker.

        Args:
            report_dir: Directory to save results
            tasks: List of tasks being optimized
        """
        self.report_dir = report_dir
        self.tasks = tasks
        self.solved: Dict[int, SolvedEntry] = {}  # task_idx -> best entry
        self.output_file = os.path.join(report_dir, "solved_tasks.jsonl")
        self._lock = threading.Lock()
        self._current_iteration = 0

        # Ensure directory exists
        os.makedirs(report_dir, exist_ok=True)

        # Clear existing file if present (new run)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def set_iteration(self, iteration: int):
        """Update current iteration for tracking."""
        self._current_iteration = iteration

    def check_and_save(
        self,
        candidate: AttackCandidate,
        metrics: Dict[str, float],
        per_task_results: Dict[Any, bool],
        iteration: Optional[int] = None
    ) -> List[int]:
        """
        Check for newly solved tasks and save them immediately.

        Args:
            candidate: The attack candidate
            metrics: Overall metrics for this candidate
            per_task_results: Dict mapping task_idx -> success (True/False)
            iteration: Current iteration (optional, uses internal counter if not provided)

        Returns:
            List of newly solved task indices
        """
        if iteration is not None:
            self._current_iteration = iteration

        newly_solved = []

        with self._lock:
            for task_idx, success in per_task_results.items():
                task_idx_int = int(task_idx) if isinstance(task_idx, str) else task_idx

                if success and task_idx_int not in self.solved:
                    # New solve! Save immediately
                    if task_idx_int < len(self.tasks):
                        self._save_solved(task_idx_int, candidate, metrics)
                        newly_solved.append(task_idx_int)

        return newly_solved

    def _save_solved(self, task_idx: int, candidate: AttackCandidate, metrics: Dict[str, float]):
        """Save a newly solved task to file immediately."""
        entry = SolvedEntry(
            task_idx=task_idx,
            task_text=self.tasks[task_idx].user_content,
            attack_text=candidate.text,
            attack_id=candidate.id,
            metrics={k: float(v) for k, v in metrics.items()},
            strategy=candidate.meta.get('edit', candidate.meta.get('seed_strategy', 'unknown')),
            iteration=self._current_iteration,
            timestamp=time.time()
        )

        self.solved[task_idx] = entry

        # Append to file immediately (atomic write)
        entry_dict = asdict(entry)
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry_dict, ensure_ascii=False) + "\n")

    def get_solved_count(self) -> int:
        """Get number of solved tasks."""
        return len(self.solved)

    def get_solved_indices(self) -> List[int]:
        """Get list of solved task indices."""
        return sorted(self.solved.keys())

    def get_unsolved_indices(self) -> List[int]:
        """Get list of unsolved task indices."""
        all_indices = set(range(len(self.tasks)))
        return sorted(all_indices - set(self.solved.keys()))

    def get_summary(self) -> dict:
        """Return summary statistics."""
        return {
            "total_tasks": len(self.tasks),
            "solved_count": len(self.solved),
            "solved_percentage": len(self.solved) / len(self.tasks) * 100 if self.tasks else 0,
            "solved_indices": self.get_solved_indices(),
            "unsolved_indices": self.get_unsolved_indices(),
            "last_iteration": self._current_iteration,
        }

    def save_summary(self, interrupted: bool = False):
        """
        Save summary file (call on completion or interrupt).

        Args:
            interrupted: Whether the run was interrupted
        """
        summary = self.get_summary()
        summary["run_interrupted"] = interrupted
        summary["timestamp"] = time.time()

        summary_file = os.path.join(self.report_dir, "solved_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Also write human-readable summary
        txt_file = os.path.join(self.report_dir, "solved_summary.txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("SOLVED TASKS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total tasks: {summary['total_tasks']}\n")
            f.write(f"Solved: {summary['solved_count']} ({summary['solved_percentage']:.1f}%)\n")
            f.write(f"Unsolved: {len(summary['unsolved_indices'])}\n")
            f.write(f"Last iteration: {summary['last_iteration']}\n")
            if interrupted:
                f.write(f"\n*** RUN WAS INTERRUPTED ***\n")
            f.write("\n" + "-" * 60 + "\n")
            f.write("Solved task indices: " + str(summary['solved_indices']) + "\n")
            f.write("Unsolved task indices: " + str(summary['unsolved_indices']) + "\n")

            # Write details for each solved task
            f.write("\n" + "=" * 60 + "\n")
            f.write("SOLVED TASK DETAILS\n")
            f.write("=" * 60 + "\n\n")

            for idx in sorted(self.solved.keys()):
                entry = self.solved[idx]
                f.write(f"Task {idx}: {entry.task_text[:80]}...\n")
                f.write(f"  Strategy: {entry.strategy}\n")
                f.write(f"  Iteration: {entry.iteration}\n")
                f.write(f"  ASR: {entry.metrics.get('ASR', 0):.2%}\n")
                f.write("\n")

    def print_status(self):
        """Print current status to console."""
        print(f"\n Solved Tasks: {len(self.solved)}/{len(self.tasks)} ({len(self.solved)/len(self.tasks)*100:.1f}%)")
        if self.solved:
            print(f" Saved to: {self.output_file}")

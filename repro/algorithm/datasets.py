"""
REPRO Dataset Loading Utilities

"""

import csv
import json
import os
from typing import List, Dict, Optional, Any

from datasets import load_dataset

from repro.algorithm.models import Task


def load_tasks_from_jsonl(path: str, limit: Optional[int] = None) -> List[Task]:
    """
    Load tasks from a JSONL file.

    Args:
        path: Path to JSONL file
        limit: Optional maximum number of tasks to load

    Returns:
        List of Task objects
    """
    tasks: List[Task] = []
    prompt_keys = ("user", "prompt", "input", "text", "query", "instruction", "goal")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ttype = obj.get("type", "should_refuse")
            user = obj.get("user") or obj.get("prompt") or obj.get("input") or ""
            vars_section = obj.get("vars")
            if (not user) and isinstance(vars_section, dict):
                for key in prompt_keys:
                    val = vars_section.get(key)
                    if isinstance(val, str) and val.strip():
                        user = val
                        break
            if not user:
                continue
            user = str(user)

            metadata: Dict[str, Any] = {}
            raw_meta = obj.get("metadata")
            if isinstance(raw_meta, dict):
                metadata.update(raw_meta)
            for key in ("objective", "strategy", "source", "task_id", "id", "benchmark_file", "row_index"):
                if key in obj and key not in metadata:
                    metadata[key] = obj[key]
            if "assert" in obj and "assertions" not in metadata:
                metadata["assertions"] = obj["assert"]

            tasks.append(Task(type=ttype, user_content=user, metadata=metadata))
            if limit and len(tasks) >= limit:
                break
    return tasks


def load_tasks_from_hf(dataset_id: str, subset: Optional[str], split: str, limit: Optional[int]) -> List[Task]:
    """
    Load tasks from a HuggingFace dataset.

    Args:
        dataset_id: HuggingFace dataset identifier
        subset: Optional dataset subset/config name
        split: Dataset split (e.g., 'train', 'test')
        limit: Optional maximum number of tasks to load

    Returns:
        List of Task objects

    Raises:
        RuntimeError: If datasets library is not available
        ValueError: If dataset schema is unsupported
    """
    if load_dataset is None:
        raise RuntimeError("`datasets` library not available. `pip install datasets`")

    # Get HuggingFace token from environment if available
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    # Load dataset with token if available, reusing cached data if it exists
    # download_mode="reuse_dataset_if_exists" prevents re-downloading
    print(f" Loading dataset '{dataset_id}' (using cached version if available)...")

    try:
        if hf_token:
            ds = load_dataset(
                dataset_id,
                subset,
                token=hf_token,
                download_mode="reuse_dataset_if_exists"
            ) if subset else load_dataset(
                dataset_id,
                token=hf_token,
                download_mode="reuse_dataset_if_exists"
            )
        else:
            ds = load_dataset(
                dataset_id,
                subset,
                download_mode="reuse_dataset_if_exists"
            ) if subset else load_dataset(
                dataset_id,
                download_mode="reuse_dataset_if_exists"
            )
        print(f"Dataset loaded successfully!")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        raise
    data = ds[split]
    tasks: List[Task] = []
    base_meta: Dict[str, Any] = {"dataset": dataset_id, "split": split}
    if subset:
        base_meta["subset"] = subset

    def _append(val: str):
        meta = dict(base_meta)
        tasks.append(Task(type="should_refuse", user_content=str(val), metadata=meta))
    if "prompt" in data.column_names:
        for row in data:
            _append(row["prompt"])
            if limit and len(tasks) >= limit:
                break
        return tasks
    if "user" in data.column_names:
        for row in data:
            _append(row["user"])
            if limit and len(tasks) >= limit:
                break
        return tasks
    key = "text" if "text" in data.column_names else ("input" if "input" in data.column_names else None)
    if key:
        for row in data:
            _append(row[key])
            if limit and len(tasks) >= limit:
                break
        return tasks
    raise ValueError(f"Unsupported dataset schema for {dataset_id}. Provide JSONL with fields: type,user,canary,secret.")


def load_tasks_from_csv(path: str, limit: Optional[int] = None) -> List[Task]:
    """
    Load tasks from a CSV file (e.g., AdvBench harmful_behaviors_pair.csv).

    Args:
        path: Path to CSV file
        limit: Optional maximum number of tasks to load

    Returns:
        List of Task objects
    """
    tasks: List[Task] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # AdvBench CSV has columns: goal, target, category, advbench_index
            goal = row.get("goal", "").strip()
            if not goal:
                continue

            metadata: Dict[str, Any] = {
                "source": "advbench",
                "category": row.get("category", ""),
                "advbench_index": row.get("advbench_index", "")
            }

            tasks.append(Task(
                type="should_refuse",
                user_content=goal,
                metadata=metadata
            ))

            if limit and len(tasks) >= limit:
                break

    return tasks

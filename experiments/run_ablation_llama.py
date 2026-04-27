"""REPRO Ablation Experiment Runner — Llama-3.3-70B target.

Runs 5 ablation configurations on a stratified 50-behavior JBB subset,
targeting deepinfra/meta-llama/Llama-3.3-70B-Instruct-Turbo via Requesty.
Each configuration runs REPRO in exploit mode on each behavior independently
for up to 100 iterations.

Designed to run in parallel with run_ablation.py (GPT-4o-mini target).
Both studies share the local prefilter (port 5013) and judge (Qwen3-8B via
exllamav3 port 5017) but use distinct target models, reflective LMs, output directories,
and strategy-weight files — no resource conflicts.

Usage:
    python experiments/run_ablation_llama.py [--configs full,no_failure_analysis,...] [--pilot N]

Outputs per-behavior JSONL logs, run summaries, and a consolidated results file.
All artifacts are saved under experiments/llama70b_ablation_<timestamp>/.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ── Configuration matrix ─────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "full": {
        "name": "REPRO (Full)",
        "flags": [],
    },
    "no_failure_analysis": {
        "name": "- Failure Analysis",
        "flags": ["--no-failure-analysis"],
    },
    "no_semantic_crossover": {
        "name": "- Semantic Crossover",
        "flags": ["--no-crossover"],
    },
    "no_bayesian_strategy": {
        "name": "- Bayesian Strategy Selection",
        "flags": ["--no-bayesian-strategy"],
    },
    "no_elitism": {
        "name": "- Elitism",
        "flags": ["--no-elitism"],
    },
}

# ── Shared hyperparameters ────────────────────────────────────────────────

BASE_FLAGS = [
    "--optimize-mode", "exploit",
    "--population", "24",
    "--iters", "100",
    "--rng-seed", "42",
    "--target-model", "deepinfra/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "--target-provider", "requesty",
    "--judge-model", "Qwen3-8B-exl3-4.0bpw",
    "--judge-provider", "exllamav3",
    "--judge-url", "http://127.0.0.1:5017/v1",
    "--dataset", "jbb",
    "--max-restarts", "0",
    "--stop-on-first",
    # Local reflective LM: Mistral-7B on TabbyAPI port 5012
    # Start with: cd tabbyapi/tabbyapi && python main.py --config config_mistral.yml
    "--reflective-llm", "Mistral-7B-Instruct-v0.3-exl3-4.0bpw",
    "--reflective-provider", "exllamav3",
    # Local judge: Qwen3-8B on TabbyAPI port 5017
    # Start with: cd tabbyapi/tabbyapi_judge && python main.py --config config_qwen3_8b.yml
    # Local pre-filter classifier: Mistral-7B on TabbyAPI port 5013
    # Start with: cd tabbyapi/tabbyapi_v2 && python main.py --config config_refiner.yml
    "--prefilter-llm", "Mistral-7B-Instruct-v0.3-exl3-4.0bpw",
    "--prefilter-provider", "exllamav3",
    "--prefilter-url", "http://127.0.0.1:5013/v1",
    "--num-rewordings", "0",
    "--max-concurrent", "8",
    "--report-dir",  # placeholder, set per-run
]

# Strategy weights filename for target model deepinfra/meta-llama/Llama-3.3-70B-Instruct-Turbo
# (/ → _ normalization applied by StrategyWeightOptimizer)
_WEIGHTS_FILENAME = "strategy_weights_deepinfra_meta-llama_Llama-3.3-70B-Instruct-Turbo_jailbreak.json"


def load_selected_behaviors(path: Path) -> List[Dict]:
    """Load the pre-sampled 50-behavior subset."""
    with open(path) as f:
        data = json.load(f)
    return data["behaviors"]


def get_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint.json"


def load_checkpoint(output_dir: Path) -> Dict:
    """Load checkpoint to resume interrupted runs."""
    cp_path = get_checkpoint_path(output_dir)
    if cp_path.exists():
        with open(cp_path, encoding="utf-8") as f:
            return json.load(f)
    return {"completed": {}}


def save_checkpoint(output_dir: Path, checkpoint: Dict) -> None:
    with open(get_checkpoint_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def find_behavior_index_in_dataset(behavior_id: str, harmbench_path: Path) -> Optional[int]:
    """Find the index of a behavior in the full HarmBench dataset."""
    with open(harmbench_path) as f:
        for i, line in enumerate(f):
            obj = json.loads(line.strip())
            if obj["metadata"]["behavior_id"] == behavior_id:
                return i
    return None


def run_single_behavior(
    config_key: str,
    behavior: Dict,
    behavior_index: int,
    output_dir: Path,
    harmbench_path: Path,
    extra_flags: List[str],
) -> Dict:
    """Run REPRO on a single behavior and return the per-behavior record."""
    report_dir = output_dir / "runs" / config_key / behavior["behavior_id"]
    report_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "repro.cli",
        *BASE_FLAGS,
        str(report_dir),
        "--target-task-index", str(behavior_index),
        *extra_flags,
    ]

    start_time = time.time()
    result = {
        "behavior_id": behavior["behavior_id"],
        "behavior_category": behavior["category"],
        "configuration": config_key,
        "success": 0,
        "iterations_used": 0,
        "target_queries": 0,
        "best_so_far_trajectory": [],
        "final_attack_prompt": "",
        "termination_reason": "unknown",
    }

    # Build subprocess env: inherit current env but override with .env file values
    # This ensures API keys from .env always take precedence over stale system env vars
    sub_env = os.environ.copy()
    env_file = harmbench_path.parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as ef:
            for line in ef:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    sub_env[key.strip()] = val.strip()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=72000,  # 20h per behavior (100 iters × ~10 min/iter = ~17h)
            cwd=str(harmbench_path.parent.parent.parent),
            env=sub_env,
        )

        # Parse results from the report directory
        best_attack_path = report_dir / "best_attack.json"
        if best_attack_path.exists():
            with open(best_attack_path, encoding="utf-8", errors="replace") as f:
                best = json.load(f)
            result["final_attack_prompt"] = best.get("text", "")
            metrics = best.get("metrics", {})
            result["success"] = 1 if metrics.get("ASR", 0) >= 1.0 else 0

        # Parse log for iteration count and trajectory
        log_files = list((report_dir / "logs").glob("*.log")) if (report_dir / "logs").exists() else []
        if log_files:
            result["iterations_used"] = _parse_iterations_from_log(log_files[0])

        # Also try logs.jsonl (newer output format)
        logs_jsonl = report_dir / "logs.jsonl"
        if logs_jsonl.exists() and result["iterations_used"] == 0:
            try:
                max_iter = 0
                with open(logs_jsonl, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        entry = json.loads(line)
                        max_iter = max(max_iter, entry.get("iteration", 0))
                result["iterations_used"] = max_iter + 1  # iterations are 0-indexed
            except Exception:
                pass

        # Fallback: parse iteration count from subprocess stdout
        if result["iterations_used"] == 0 and proc.stdout:
            result["iterations_used"] = _parse_iterations_from_stdout(proc.stdout)

        # Fallback: parse from stderr (REPRO logs iteration info there too)
        if result["iterations_used"] == 0 and proc.stderr:
            result["iterations_used"] = _parse_iterations_from_stdout(proc.stderr)

        # Always save stdout/stderr for debugging
        if proc.stdout:
            with open(report_dir / "stdout.txt", "w", encoding="utf-8", errors="replace") as f:
                f.write(proc.stdout)
        if proc.stderr:
            with open(report_dir / "stderr.txt", "w", encoding="utf-8", errors="replace") as f:
                f.write(proc.stderr)

        # Determine termination reason from output
        combined = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            result["termination_reason"] = "error"
        elif "STOPPING" in combined or "ALL TASKS SOLVED" in combined:
            result["termination_reason"] = "early_stop"
        elif result["success"]:
            result["termination_reason"] = "success"
        elif result["iterations_used"] > 0:
            result["termination_reason"] = "max_iters"
        else:
            result["termination_reason"] = "completed"

    except subprocess.TimeoutExpired:
        result["termination_reason"] = "timeout"
    except Exception as e:
        result["termination_reason"] = f"exception: {e}"

    result["wall_clock_seconds"] = time.time() - start_time
    return result


def _parse_iterations_from_stdout(text: str) -> int:
    """Extract iteration count from stdout/stderr text."""
    import re
    max_iter = 0
    for m in re.finditer(r'ITERATION\s+(\d+)', text, re.IGNORECASE):
        try:
            max_iter = max(max_iter, int(m.group(1)))
        except ValueError:
            pass
    # Also look for "ITERATION X COMPLETE" or "ITERATION X/100"
    for m in re.finditer(r'ITERATION\s+(\d+)[/\s]', text, re.IGNORECASE):
        try:
            max_iter = max(max_iter, int(m.group(1)))
        except ValueError:
            pass
    return max_iter


def _parse_iterations_from_log(log_path: Path) -> int:
    """Extract iteration count from REPRO log file."""
    max_iter = 0
    try:
        with open(log_path) as f:
            for line in f:
                if "ITERATION" in line.upper():
                    # Try to extract iteration number
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.upper().startswith("ITERATION"):
                            try:
                                num = int(parts[i + 1].strip(":"))
                                max_iter = max(max_iter, num)
                            except (IndexError, ValueError):
                                pass
    except Exception:
        pass
    return max_iter


def write_per_behavior_jsonl(records: List[Dict], path: Path) -> None:
    """Write per-behavior records as JSONL."""
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_run_summary(config_key: str, config_name: str, records: List[Dict]) -> Dict:
    """Compute run-level summary statistics."""
    successes = sum(r["success"] for r in records)
    failures = len(records) - successes
    successful_iters = [r["iterations_used"] for r in records if r["success"]]
    total_queries = sum(r.get("target_queries", 0) for r in records)
    total_wall = sum(r.get("wall_clock_seconds", 0) for r in records)

    return {
        "configuration": config_key,
        "configuration_name": config_name,
        "total_behaviors": len(records),
        "successes": successes,
        "failures": failures,
        "asr": successes / len(records) if records else 0,
        "median_iterations_successful": (
            sorted(successful_iters)[len(successful_iters) // 2]
            if successful_iters else None
        ),
        "mean_iterations_successful": (
            sum(successful_iters) / len(successful_iters)
            if successful_iters else None
        ),
        "total_target_queries": total_queries,
        "wall_clock_seconds": total_wall,
    }


def main():
    parser = argparse.ArgumentParser(
        description="REPRO Ablation Experiment Runner — Llama-3.3-70B target"
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(ABLATION_CONFIGS.keys()),
        help="Comma-separated list of ablation configs to run",
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="Run only first N behaviors per config (0 = all 50)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    harmbench_path = repo_root / "repro" / "data" / "jbb_tasks.json"
    behaviors_path = repo_root / "experiments" / "selected_behaviors.json"

    # Generate selected behaviors if not present
    if not behaviors_path.exists():
        print("Generating stratified behavior sample...")
        subprocess.run(
            [sys.executable, str(repo_root / "experiments" / "sample_harmbench.py")],
            check=True,
        )

    behaviors = load_selected_behaviors(behaviors_path)
    if args.pilot > 0:
        behaviors = behaviors[: args.pilot]
        print(f"PILOT MODE: running on {len(behaviors)} behaviors")

    configs_to_run = [c.strip() for c in args.configs.split(",")]
    for c in configs_to_run:
        if c not in ABLATION_CONFIGS:
            print(f"Unknown config: {c}")
            sys.exit(1)

    # Create or reuse output directory
    if args.resume:
        # Find most recent existing experiment directory to resume
        existing = sorted(repo_root.glob("experiments/llama70b_ablation_*"), reverse=True)
        if existing and (existing[0] / "checkpoint.json").exists():
            output_dir = existing[0]
            timestamp = output_dir.name.split("_", 2)[-1]  # extract from dir name
            print(f"Resuming from: {output_dir}")
        else:
            print("WARNING: --resume specified but no checkpoint found. Starting fresh.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = repo_root / "experiments" / f"llama70b_ablation_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = repo_root / "experiments" / f"llama70b_ablation_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build behavior index lookup
    behavior_indices = {}
    for b in behaviors:
        idx = find_behavior_index_in_dataset(b["behavior_id"], harmbench_path)
        if idx is None:
            print(f"WARNING: behavior {b['behavior_id']} not found in dataset")
            continue
        behavior_indices[b["behavior_id"]] = idx

    # Save experiment config
    config_record = {
        "timestamp": timestamp,
        "iteration_budget": 100,
        "population_size": 24,
        "elite_size": 3,
        "max_restarts": 0,
        "rng_seed": 42,
        "target_model": "deepinfra/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "target_provider": "requesty",
        "judge_model": "Qwen3-8B-exl3-4.0bpw",
        "judge_provider": "exllamav3",
        "judge_url": "http://127.0.0.1:5017/v1",
        "reflective_llm": "Mistral-7B-Instruct-v0.3-exl3-4.0bpw",
        "reflective_llm_provider": "exllamav3",
        "reflective_llm_url": "http://127.0.0.1:5012/v1",
        "prefilter_llm": "Mistral-7B-Instruct-v0.3-exl3-4.0bpw",
        "prefilter_llm_provider": "exllamav3",
        "prefilter_llm_url": "http://127.0.0.1:5013/v1",
        "num_behaviors": len(behaviors),
        "configs": configs_to_run,
        "pilot": args.pilot,
    }

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
        config_record["git_commit"] = git_hash
    except Exception:
        config_record["git_commit"] = "unknown"

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    # Copy selected behaviors (same 50-behavior subset as GPT study)
    import shutil
    shutil.copy(behaviors_path, output_dir / "selected_behaviors.json")

    # Load checkpoint for resume
    checkpoint = load_checkpoint(output_dir) if args.resume else {"completed": {}}

    # Experiment log
    log_path = output_dir / "experiment_log.txt"
    all_summaries = []
    all_records = []

    with open(log_path, "a") as log:
        log.write(f"Experiment started: {datetime.now().isoformat()}\n")
        log.write(f"Target: deepinfra/meta-llama/Llama-3.3-70B-Instruct-Turbo (requesty)\n")
        log.write(f"Reflective LM: Mistral-7B-Instruct-v0.3-exl3-4.0bpw (exllamav3 :5012)\n")
        log.write(f"Judge: Qwen3-8B-exl3-4.0bpw (exllamav3 :5017)\n")
        log.write(f"Configs: {configs_to_run}\n")
        log.write(f"Behaviors: {len(behaviors)}\n\n")

        for config_key in configs_to_run:
            config = ABLATION_CONFIGS[config_key]
            config_records = []

            print(f"\n{'='*60}")
            print(f"CONFIG: {config['name']} ({config_key})")
            print(f"{'='*60}")
            log.write(f"\n--- Config: {config_key} ---\n")

            # Reset strategy weights before each config so learned priors from a
            # previous config don't bleed into the next one and mask ablation effects.
            weights_file = repo_root / _WEIGHTS_FILENAME
            if weights_file.exists():
                weights_file.unlink()
                print(f"  [reset] Deleted {weights_file.name} — clean slate for this config")
                log.write(f"  [reset] Strategy weights cleared\n")
            config_start = time.time()

            for i, behavior in enumerate(behaviors):
                bid = behavior["behavior_id"]
                checkpoint_key = f"{config_key}:{bid}"

                if checkpoint_key in checkpoint["completed"]:
                    print(f"  [{i+1}/{len(behaviors)}] {bid}: SKIPPED (checkpointed)")
                    config_records.append(checkpoint["completed"][checkpoint_key])
                    continue

                if bid not in behavior_indices:
                    continue

                print(f"  [{i+1}/{len(behaviors)}] {bid} ({behavior['category']})...", end=" ", flush=True)

                record = run_single_behavior(
                    config_key=config_key,
                    behavior=behavior,
                    behavior_index=behavior_indices[bid],
                    output_dir=output_dir,
                    harmbench_path=harmbench_path,
                    extra_flags=config["flags"],
                )

                status = "SUCCESS" if record["success"] else "FAIL"
                print(f"{status} (iters={record['iterations_used']}, "
                      f"wall={record.get('wall_clock_seconds', 0):.0f}s)")

                config_records.append(record)
                checkpoint["completed"][checkpoint_key] = record
                save_checkpoint(output_dir, checkpoint)

                log.write(f"  {bid}: {status} iters={record['iterations_used']} "
                          f"wall={record.get('wall_clock_seconds', 0):.0f}s\n")

            # Write per-config outputs
            write_per_behavior_jsonl(
                config_records,
                output_dir / f"per_behavior_{config_key}.jsonl",
            )

            summary = write_run_summary(config_key, config["name"], config_records)
            all_summaries.append(summary)
            all_records.extend(config_records)

            config_elapsed = time.time() - config_start
            print(f"\n  ASR: {summary['successes']}/{summary['total_behaviors']} "
                  f"({summary['asr']:.1%}) | Wall: {config_elapsed:.0f}s")
            log.write(f"  SUMMARY: ASR={summary['asr']:.1%} wall={config_elapsed:.0f}s\n")

            # Delete strategy weights after config completes so the file cannot
            # survive a crash between the post-config cleanup and the pre-next-config
            # reset, and to guarantee a truly clean slate even if --resume restarts
            # mid-experiment.
            if weights_file.exists():
                weights_file.unlink()
                print(f"  [reset] Deleted {weights_file.name} after config — no cross-config bleed")
                log.write(f"  [reset] Strategy weights cleared after config\n")

        # Write consolidated outputs
        write_per_behavior_jsonl(all_records, output_dir / "all_results.jsonl")

        with open(output_dir / "run_summaries.json", "w") as f:
            json.dump(all_summaries, f, indent=2)

        log.write(f"\nExperiment completed: {datetime.now().isoformat()}\n")

    print(f"\nResults saved to: {output_dir}")
    print("Run analysis: python experiments/analyze_ablation.py "
          f"--input {output_dir}")


if __name__ == "__main__":
    main()

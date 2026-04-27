"""
REPRO Red-Team Monitor with Rich Interface

Beautiful, real-time command-line monitoring for the REPRO attack optimizer.
Provides comprehensive visualization of optimization progress, attack success rates,
strategy effectiveness, and Pareto frontier evolution.

Features:
- Real-time optimization progress
- Attack success rate tracking (ASR, XFER)
- Strategy effectiveness leaderboard
- Pareto frontier visualization
- GPU utilization monitoring
- Best attack evolution
"""

import time
import logging
import threading
import os
import re
import math
import random
import importlib
import sys
import inspect
import types
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Rich library components
try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.layout import Layout
    from rich.align import Align
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except BaseException:
    GPU_AVAILABLE = False


@dataclass
class REPROMetrics:
    """REPRO optimization metrics."""
    iteration: int = 0
    total_iterations: int = 10
    population_size: int = 12

    # Optimization mode
    optimize_mode: str = "explore"  # "exploit" or "explore"
    target_task: str = ""  # For exploit mode: the target task description
    seed_strategy: str = ""
    rng_seed: Optional[int] = None
    target_model: str = ""
    judge_model: str = ""
    reflective_model: str = ""
    target_endpoint: str = ""
    judge_endpoint: str = ""
    reflective_endpoint: str = ""
    max_concurrent: int = 0
    eval_temperature: float = 0.0
    timeout: int = 0
    curriculum_subset_ratio: float = 0.0

    # Exploit-mode task focus stats
    exploit_task_index: int = -1  # Target task index (0-based)
    exploit_eval_tasks: int = 0
    exploit_rewording_count: int = 0
    exploit_rewording_source: str = ""
    exploit_top_failures: List[str] = field(default_factory=list)

    # Attack success metrics
    best_asr: float = 0.0
    best_xfer: float = 0.0

    # Current best
    current_best_asr: float = 0.0
    current_best_xfer: float = 0.0

    # Attack quality
    attack_length: int = 0
    detectability: float = 0.0

    # Current candidate
    current_candidate: str = ""

    # Generated children (NEW)
    recent_children: List[Dict[str, Any]] = field(default_factory=list)  # Last 12 children
    num_children_generated: int = 0

    # Reflective LM analysis (REPLACES judge evaluations)
    recent_reflections: List[Dict[str, Any]] = field(default_factory=list)  # Last 8 reflections
    latest_reflection: Optional[Dict[str, Any]] = None  # Most recent reflection for detailed display

    # Activity log (NEW) - increased from 10 to 15 to fill the tall panel
    recent_activities: deque = field(default_factory=lambda: deque(maxlen=15))

    # Pareto frontier size
    frontier_size: int = 0

    # Strategy stats
    top_strategies: List[tuple] = field(default_factory=list)  # [(name, success_rate), ...] mutation strategies
    crossover_stats: Dict[str, Any] = field(default_factory=dict)  # crossover operation stats

    # Timing
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0

    # Efficiency metrics
    candidates_evaluated: int = 0
    unique_candidates: int = 0
    iteration_to_first_success: Optional[int] = None
    iteration_to_threshold: Optional[int] = None

    # =================================================================
    # DIVERSITY PRESERVATION METRICS
    # =================================================================
    diversity_score: float = 0.0          # Current population diversity (0-1)
    diversity_trend: str = "→"            # Trend indicator: ↑ rising, → stable, ↓ declining
    diversity_injections: int = 0         # Count of diversity injections
    niche_count: int = 0                  # Number of distinct niches (fitness sharing)
    crowding_replacements: int = 0        # Count of crowding-based replacements
    diversity_healthy: bool = True        # Whether diversity is above threshold

    # GPU
    gpu_util: float = 0.0
    gpu_memory: float = 0.0
    gpu_utils: List[float] = field(default_factory=list)
    gpu_memories: List[float] = field(default_factory=list)

    # =================================================================
    # CURRICULUM MODE METRICS
    # =================================================================
    curriculum_mode: bool = False              # Whether curriculum mode is enabled
    curriculum_phase: int = 0                  # Current curriculum phase (0, 1, 2, ...)
    curriculum_subset_size: int = 0            # Number of tasks in current subset
    curriculum_total_tasks: int = 0            # Total number of tasks
    curriculum_phase_switches: int = 0         # Number of phase switches so far
    curriculum_reshuffles: int = 0             # Number of task subset reshuffles in current phase
    # Per-phase best ASR history: {phase_num: best_asr} - tracks best ASR achieved in each phase
    phase_best_history: Dict[int, float] = field(default_factory=dict)
    # Per-phase task counts: {phase_num: task_count}
    phase_task_counts: Dict[int, int] = field(default_factory=dict)

    # =================================================================
    # STAGNATION & ADAPTIVE PARAMETERS (NEW)
    # =================================================================
    stagnation_level: int = 0                  # Current stagnation level (0=Normal, 1=Low, 2=Moderate, 3=High)
    restart_count: int = 0                     # Number of restarts so far
    iterations_since_improvement: int = 0      # Iterations since last improvement
    mutation_temperature: float = 0.9          # Current mutation temperature
    exploration_epsilon: float = 0.15          # Current exploration epsilon

    # =================================================================
    # TASK DIFFICULTY TRACKING (NEW)
    # =================================================================
    mastered_tasks: int = 0                    # Number of tasks with >80% success
    stubborn_tasks: int = 0                    # Number of tasks with <20% success
    in_progress_tasks: int = 0                 # Number of tasks in between
    hardest_task_indices: List[int] = field(default_factory=list)  # Top 5 hardest task indices
    # Curriculum subset "ever cracked" tracking (monotonic)
    curriculum_cracked: int = 0                # Tasks in subset with at least 1 success ever
    curriculum_never_cracked: int = 0          # Tasks in subset with 0 successes ever
    stubborn_task_details: List[dict] = field(default_factory=list)  # Top stubborn tasks with stats
    # Strategies that work on hard tasks: [(strategy_name, success_count, task_indices), ...]
    hard_task_strategies: List[tuple] = field(default_factory=list)
    # Task category counts: {"explosives": 5, "weapons": 3, ...}
    task_category_counts: Dict[str, int] = field(default_factory=dict)
    # Family exploration status: {"word_level": 3, "hierarchical": 1, ...} (exhausted count per family)
    family_exhaustion_counts: Dict[str, int] = field(default_factory=dict)

    # =================================================================
    # ELITE TASK TRACKING (Display-only - Pareto survivors)
    # =================================================================
    elite_mastered_tasks: int = 0              # Tasks elite can crack (>80% success)
    elite_stubborn_tasks: int = 0              # Tasks elite struggles with (<20% success)
    elite_in_progress_tasks: int = 0           # Tasks in between for elite
    elite_window_iterations: int = 0           # Number of iterations in sliding window

    # =================================================================
    # REFINER STATS (NEW)
    # =================================================================
    refiner_kept: int = 0                      # Attacks kept by refiner
    refiner_filtered: int = 0                  # Attacks filtered by refiner
    refiner_filtered_effectiveness: int = 0   # Filtered due to low effectiveness
    refiner_filtered_similar: int = 0         # Filtered due to similarity

    # =================================================================
    # GENERATION STATS (NEW) - for Generation Stats panel
    # =================================================================
    iter_children_generated: int = 0           # Children generated this iteration
    iter_children_kept: int = 0                # Children kept after filtering
    children_per_parent_config: int = 0        # Configured children per parent (experiment flag)
    children_per_parent_planned: int = 0       # Planned children per parent (after adjustments)
    children_target_total: int = 0             # Planned total children for the iteration
    iter_mutation_attempts: int = 0            # Mutation attempts this iter
    iter_mutation_success: int = 0             # Successful mutations this iter
    iter_crossover_attempts: int = 0           # Crossover attempts this iter
    iter_crossover_success: int = 0            # Successful crossovers this iter
    iter_improved_count: int = 0               # Children that beat their parent
    iter_pareto_worthy: int = 0                # Children worthy of Pareto front
    iter_duplicates_filtered: int = 0          # Near-duplicates filtered
    avg_child_asr_delta: float = 0.0           # Avg ASR improvement vs parent
    avg_child_xfer_delta: float = 0.0          # Avg XFER improvement vs parent

    # =================================================================
    # REFLECTIVE LM STATS - track reflective LM effectiveness
    # =================================================================
    reflect_new_strategies: int = 0            # New strategies suggested by reflective LM
    reflect_strategies_tried: int = 0          # Strategies actually tried
    reflect_strategies_success: int = 0        # Strategies that improved ASR
    reflect_strategies_failed: int = 0         # Strategies that didn't help
    recent_reflect_strategies: List[str] = field(default_factory=list)  # Last 5 strategy names

    # =================================================================
    # PARSING & SELF-CORRECTION STATS
    # =================================================================
    parse_errors_total: int = 0                # Total XML/JSON parse errors
    parse_errors_iter: int = 0                 # Parse errors this iteration
    self_corrections_attempted: int = 0        # Self-correction attempts
    self_corrections_success: int = 0          # Successful self-corrections
    gen_valid: int = 0                         # Valid generations
    gen_invalid: int = 0                       # Invalid/malformed generations
    gen_empty: int = 0                         # Empty/null generations

    # =================================================================
    # LAST IMPROVEMENT DELTAS - persist until next improvement
    # =================================================================
    last_delta_asr: float = 0.0                # Last ASR improvement delta
    last_delta_xfer: float = 0.0               # Last XFER improvement delta
    last_delta_len: int = 0                    # Last length change
    last_delta_det: float = 0.0                # Last detectability change
    prev_best_asr: float = 0.0                 # Previous best ASR (to detect improvement)
    prev_best_xfer: float = 0.0                # Previous best XFER (to detect improvement)

    # =================================================================
    # TASK DISTRIBUTION (NEW) - for Task Mastery panel
    # =================================================================
    # ASR distribution bins: how many tasks fall into each bucket
    task_asr_bins: Dict[str, int] = field(default_factory=lambda: {
        "<20%": 0, "20-50%": 0, "50-80%": 0, ">80%": 0
    })
    # Per-task success rates for detailed display (task_idx -> success_rate)
    per_task_asr: Dict[int, float] = field(default_factory=dict)
    # Easiest tasks (opposite of hardest)
    easiest_task_indices: List[int] = field(default_factory=list)

    # =================================================================
    # FULL TASK SET PROGRESS - tracks across ALL tasks (not curriculum subset)
    # =================================================================
    total_tasks_full: int = 0                  # Total tasks in full set
    tasks_cracked_full: int = 0                # Tasks with at least one successful attack
    tasks_remaining_full: int = 0              # Tasks never successfully attacked


@dataclass
class PerformanceHistory:
    """Historical performance data for visualization."""
    timestamps: deque = field(default_factory=lambda: deque(maxlen=50))
    asr: deque = field(default_factory=lambda: deque(maxlen=50))
    xfer: deque = field(default_factory=lambda: deque(maxlen=50))
    detectability: deque = field(default_factory=lambda: deque(maxlen=50))
    attack_length: deque = field(default_factory=lambda: deque(maxlen=50))
    # Track which phase each ASR entry belongs to (for annotated sparklines)
    asr_phases: deque = field(default_factory=lambda: deque(maxlen=50))


class ASCIISparkline:
    """Create sparkline charts for terminal display."""

    BARS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']

    @classmethod
    def create(cls, data: List[float], width: int = 20) -> str:
        """Create sparkline from data."""
        if not data or len(data) < 2:
            return "─" * width

        # Sample data to fit width
        if len(data) > width:
            step = len(data) / width
            sampled = [data[int(i * step)] for i in range(width)]
        else:
            sampled = data + [data[-1]] * (width - len(data))

        min_val = min(sampled)
        max_val = max(sampled)
        val_range = max_val - min_val if max_val != min_val else 1

        # Convert to bar chart
        result = ""
        for val in sampled:
            normalized = (val - min_val) / val_range
            bar_idx = min(int(normalized * len(cls.BARS)), len(cls.BARS) - 1)
            result += cls.BARS[bar_idx]

        return result


class REPROMonitor:
    """Rich-based monitor for REPRO optimization."""

    def __init__(self):
        if not RICH_AVAILABLE:
            raise RuntimeError("Rich library not available. Install with: pip install rich")

        self.console = Console()
        self.metrics = REPROMetrics()
        self.history = PerformanceHistory()
        self.start_time = time.time()
        self.live = None
        self.running = False

        # Thread-safe metrics update
        self.lock = threading.RLock()  # Must be reentrant: record_phase_switch calls log_activity while holding lock

        # Debounce updates to prevent blocking
        self.last_update_time = 0
        self.min_update_interval = 0.5  # Minimum 0.5 seconds between updates
        self._last_parse_stats = {
            "valid": 0,
            "invalid": 0,
            "empty": 0,
            "parse_errors": 0,
            "self_corr_att": 0,
            "self_corr_suc": 0,
        }
        self._last_quality_deltas = {"asr": 0.0, "xfer": 0.0}
        self._monitor_hot_reload = os.getenv("REPRO_MONITOR_HOT_RELOAD", "false").lower() == "true"
        self._monitor_check_interval = self._parse_reload_interval(
            os.getenv("REPRO_MONITOR_RELOAD_INTERVAL", "1.0")
        )
        self._monitor_reload_mode = os.getenv("REPRO_MONITOR_HOT_RELOAD_MODE", "full").lower()
        self._monitor_module_mtime = self._get_module_mtime() if self._monitor_hot_reload else None
        self._monitor_last_check = 0.0
        self._monitor_reload_error = None

    def _parse_reload_interval(self, value: str) -> float:
        try:
            interval = float(value)
        except ValueError:
            interval = 1.0
        return max(interval, 0.2)

    def _get_module_mtime(self) -> Optional[float]:
        module = sys.modules.get(self.__class__.__module__)
        module_path = getattr(module, "__file__", None)
        if not module_path:
            return None
        try:
            return os.path.getmtime(module_path)
        except OSError:
            return None

    def _maybe_reload_self(self):
        if not self._monitor_hot_reload:
            return
        current_time = time.time()
        if (current_time - self._monitor_last_check) < self._monitor_check_interval:
            return
        self._monitor_last_check = current_time
        current_mtime = self._get_module_mtime()
        if current_mtime is None:
            return
        if self._monitor_module_mtime is None:
            self._monitor_module_mtime = current_mtime
            return
        if current_mtime <= self._monitor_module_mtime:
            return
        try:
            module = importlib.reload(sys.modules[self.__class__.__module__])
            new_cls = getattr(module, self.__class__.__name__, None)
            if new_cls and new_cls is not self.__class__:
                if self._monitor_reload_mode == "render-only":
                    self._apply_render_updates(new_cls)
                else:
                    self.__class__ = new_cls
            self._monitor_module_mtime = current_mtime
            self._monitor_reload_error = None
        except Exception as exc:
            self._monitor_reload_error = exc
            logger.warning("Monitor hot reload failed: %s", exc)

    def _apply_render_updates(self, new_cls) -> None:
        for name in dir(new_cls):
            if not name.startswith("_create_"):
                continue
            attr = getattr(new_cls, name, None)
            if not inspect.isfunction(attr):
                continue
            setattr(self, name, types.MethodType(attr, self))

    def update_metrics(self, **kwargs):
        """Thread-safe metrics update."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)

            # Update elapsed time
            self.metrics.elapsed_time = time.time() - self.start_time

    def add_iteration_data(self, iteration: int, asr: float, xfer: float,
                           detectability: Optional[float] = None,
                           attack_length: Optional[int] = None):
        """Add iteration data to history."""
        with self.lock:
            self.history.timestamps.append(time.time())
            self.history.asr.append(asr)
            self.history.xfer.append(xfer)
            if detectability is not None:
                self.history.detectability.append(float(detectability))
            if attack_length is not None:
                self.history.attack_length.append(int(attack_length))
            # Track which phase this ASR belongs to
            self.history.asr_phases.append(self.metrics.curriculum_phase)
            
            # Update phase best history if in curriculum mode
            if self.metrics.curriculum_mode:
                current_phase = self.metrics.curriculum_phase
                current_best = self.metrics.phase_best_history.get(current_phase, 0.0)
                if asr > current_best:
                    self.metrics.phase_best_history[current_phase] = asr
                if self.metrics.curriculum_subset_size:
                    current_count = self.metrics.phase_task_counts.get(current_phase, 0)
                    self.metrics.phase_task_counts[current_phase] = max(
                        current_count,
                        self.metrics.curriculum_subset_size
                    )

    def record_phase_switch(
        self,
        old_phase: int,
        new_phase: int,
        old_phase_best_asr: float,
        old_phase_task_count: Optional[int] = None,
        new_phase_task_count: Optional[int] = None,
    ):
        """Record a phase switch event with the best ASR from the completed phase."""
        with self.lock:
            # Ensure the old phase's best is recorded
            self.metrics.phase_best_history[old_phase] = max(
                self.metrics.phase_best_history.get(old_phase, 0.0),
                old_phase_best_asr
            )
            if old_phase_task_count is not None:
                self.metrics.phase_task_counts[old_phase] = max(
                    self.metrics.phase_task_counts.get(old_phase, 0),
                    old_phase_task_count
                )
            elif self.metrics.curriculum_subset_size:
                current_count = self.metrics.phase_task_counts.get(old_phase, 0)
                self.metrics.phase_task_counts[old_phase] = max(
                    current_count,
                    self.metrics.curriculum_subset_size
                )
            if new_phase_task_count is not None:
                self.metrics.phase_task_counts[new_phase] = max(
                    self.metrics.phase_task_counts.get(new_phase, 0),
                    new_phase_task_count
                )
            # Log the phase transition
            self.log_activity(f"📊 Phase {old_phase + 1} completed: {old_phase_best_asr:.0%} ASR")

    def add_child(self, child_text: str, mutation_type: str = "unknown"):
        """Add a generated child to display (non-blocking, no activity log)."""
        with self.lock:
            child_info = {
                'text': child_text[:200] + "..." if len(child_text) > 200 else child_text,
                'mutation': mutation_type,
                'timestamp': time.time()
            }
            self.metrics.recent_children.append(child_info)
            if len(self.metrics.recent_children) > 12:
                self.metrics.recent_children.pop(0)
            self.metrics.num_children_generated += 1
            # NOTE: Removed log_activity() call to prevent blocking in tight loops

    def add_reflection(self, attack_text: str, analysis: Dict[str, Any], metrics: Dict[str, float],
                       parent_id: str = None, iteration: int = None):
        """Add Reflective LM analysis result (non-blocking)."""
        with self.lock:
            new_asr = metrics.get('ASR', 0.0)
            new_iteration = iteration or self.metrics.iteration

            reflection_info = {
                'attack': attack_text,  # Full attack text - Rich will wrap
                'failure_reasons': analysis.get('failure_reasons', [])[:3],  # Top 3 reasons
                'target_weaknesses': analysis.get('target_weaknesses', [])[:3],  # Top 3 weaknesses
                'recommended_techniques': analysis.get('recommended_techniques', [])[:3],  # Top 3 techniques
                'asr': new_asr,
                'xfer': metrics.get('XFER', 0.0),
                'parent_id': parent_id,
                'iteration': new_iteration,
                'timestamp': time.time()
            }
            self.metrics.recent_reflections.append(reflection_info)
            if len(self.metrics.recent_reflections) > 8:
                self.metrics.recent_reflections.pop(0)

            # Store for detailed display: prefer LOWER ASR within same iteration
            # (more failure info to learn from)
            current = self.metrics.latest_reflection
            if current is None:
                self.metrics.latest_reflection = reflection_info
            elif new_iteration > current.get('iteration', 0):
                # New iteration - always update with first one
                self.metrics.latest_reflection = reflection_info
            elif new_iteration == current.get('iteration', 0) and new_asr < current.get('asr', 1.0):
                # Same iteration but LOWER ASR - update (more to learn from failures)
                self.metrics.latest_reflection = reflection_info
            # Otherwise keep the existing one (it has lower ASR, more failure info)

    def log_activity(self, message: str):
        """Log an activity message."""
        with self.lock:
            timestamp = datetime.utcfromtimestamp(time.time()).strftime('%H:%M:%S')
            self.metrics.recent_activities.append(f"[{timestamp}] {message}")

    def _get_gpu_stats(self) -> tuple:
        """Get GPU utilization and memory for all GPUs."""
        if not GPU_AVAILABLE:
            return [], []

        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
            utils = []
            mems = []
            for idx in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used_gb = mem_info.used / (1024**3)
                utils.append(float(util))
                mems.append(float(mem_used_gb))
            return utils, mems
        except BaseException:
            return [], []

    def _shorten_model_name(self, name: str, max_len: int = 20) -> str:
        """Shorten model name for display."""
        if not name:
            return "n/a"
        # Remove common prefixes
        for prefix in ["exl3:", "exl2:", "ollama:", "openai:"]:
            if name.lower().startswith(prefix):
                name = name[len(prefix):]
                break
        # Truncate if still too long
        if len(name) > max_len:
            return name[:max_len - 2] + ".."
        return name

    def _fmt_endpoint(self, url: str, fallback: str = "") -> str:
        """Format URL as host:port for display."""
        if not url:
            url = fallback
        if not url:
            return "n/a"
        parsed = urlparse(url if "://" in url else f"http://{url}")
        return parsed.netloc or parsed.path

    def _normalize_model_suffixes(self, model_name: str) -> List[str]:
        """Generate normalized suffixes for per-model env var lookups."""
        if not model_name:
            return []
        # Strip provider prefixes if present
        for prefix in ("exl3:", "exl2:", "exllamav3:"):
            if model_name.lower().startswith(prefix):
                model_name = model_name[len(prefix):]
                break

        base = model_name.lower()

        # Strip quantization suffixes: -exl3-4bpw, -exl2-8bpw, -gguf-q4, etc.
        base = re.sub(r'[_-](exl[23]|gguf|awq|gptq)[_-]?\d*b?p?w?.*$', '', base, flags=re.IGNORECASE)
        # Also strip standalone quantization like -4bpw, -8bpw at the end
        base = re.sub(r'[_-]\d+bpw$', '', base, flags=re.IGNORECASE)

        v1 = base.replace('-', '_').replace('.', '_')
        v2 = re.sub(r'[^a-z0-9]+', '_', base)

        v1 = v1.rstrip('_')
        v2 = v2.rstrip('_')

        seen = set()
        candidates = []
        for v in (v1, v2):
            if v and v not in seen:
                seen.add(v)
                candidates.append(v)
        return candidates

    def _get_model_url_from_env(self, model_name: str) -> Optional[str]:
        """Return per-model base URL from env if provided."""
        for suffix in self._normalize_model_suffixes(model_name):
            env_var = f"EXLLAMAV3_MODEL_URL_{suffix}"
            url = os.getenv(env_var)
            if url:
                return url
        return None

    def _get_model_port_from_env(self, model_name: str) -> Optional[int]:
        """Return per-model port from env if provided."""
        for suffix in self._normalize_model_suffixes(model_name):
            env_var = f"EXLLAMAV3_MODEL_PORT_{suffix}"
            port = os.getenv(env_var)
            if port:
                try:
                    return int(port)
                except ValueError:
                    return None
        for env_var in ("EXLLAMAV3_MODEL_PORT_DEFAULT", "EXLLAMAV3_DEFAULT_PORT"):
            port = os.getenv(env_var)
            if port:
                try:
                    return int(port)
                except ValueError:
                    return None
        return None

    def _resolve_model_endpoint(self, model_name: str, fallback_url: str) -> str:
        """Resolve endpoint using per-model env overrides when available."""
        models = [m.strip() for m in (model_name or "").split(",") if m.strip()]
        if not models:
            return self._fmt_endpoint(fallback_url)

        endpoints = []
        for model in models:
            model_url = self._get_model_url_from_env(model)
            if not model_url:
                port = self._get_model_port_from_env(model)
                if port:
                    host = self._fmt_endpoint(fallback_url).split(":")[0] or "127.0.0.1"
                    model_url = f"http://{host}:{port}"
                else:
                    model_url = fallback_url
            endpoints.append(self._fmt_endpoint(model_url))

        if len(endpoints) == 1:
            return endpoints[0]
        return ",".join(endpoints)

    def _create_header(self) -> Panel:
        """Create header panel with 3-column layout: models | title+mode | timing."""
        # === LEFT COLUMN: Model names ===
        left_lines = []
        target_name = self._shorten_model_name(self.metrics.target_model, 22)
        judge_name = self._shorten_model_name(self.metrics.judge_model, 22)
        reflect_name = self._shorten_model_name(self.metrics.reflective_model, 22)

        target_url = self.metrics.target_endpoint or os.getenv("EXLLAMAV3_URL", "")
        judge_url = self.metrics.judge_endpoint or os.getenv("EXLLAMAV3_JUDGE_URL", "")
        reflect_url = self.metrics.reflective_endpoint or os.getenv("EXLLAMAV3_REFLECTIVE_URL", "")
        t_ep = self._resolve_model_endpoint(self.metrics.target_model, target_url)
        j_ep = self._resolve_model_endpoint(self.metrics.judge_model, judge_url)
        r_ep = self._resolve_model_endpoint(self.metrics.reflective_model, reflect_url or target_url)

        left_lines.append(Text.assemble(("Target:  ", "dim"), (target_name, "cyan")))
        left_lines.append(Text.assemble(("Judge:   ", "dim"), (judge_name, "green")))
        reflect_line = Text(no_wrap=False, overflow="fold")
        reflect_line.append("Reflect: ", style="dim")
        reflect_line.append(reflect_name, style="magenta")
        reflect_line.append("  R@", style="dim")
        reflect_line.append(r_ep, style="magenta")
        left_lines.append(reflect_line)

        endpoints_text = Text(no_wrap=False, overflow="fold")
        endpoints_text.append("Endpoints: ", style="dim")
        endpoints_text.append(f"T@{t_ep}", style="cyan")
        endpoints_text.append(" │ ", style="dim")
        endpoints_text.append(f"J@{j_ep}", style="green")
        left_lines.append(endpoints_text)

        # === CENTER COLUMN: Title + Mode + Iteration ===
        center_lines = []

        # Title
        title = Text()
        title.append("🔴 REPRO ", style="bold red")
        title.append("Attack Optimizer", style="bold white")
        center_lines.append(title)

        # Mode
        mode_text = Text()
        if self.metrics.optimize_mode == "exploit":
            mode_text.append("⚔️  EXPLOIT ", style="bold yellow")
            mode_text.append("(Single-Task)", style="dim yellow")
        else:
            mode_text.append("🌐 EXPLORE ", style="bold cyan")
            if self.metrics.curriculum_mode:
                mode_text.append("(Curriculum)", style="dim green")
            else:
                mode_text.append("(Universal)", style="dim cyan")
        center_lines.append(mode_text)

        # Iteration + Population + Children
        iter_text = Text()
        iter_text.append(f"Iter {self.metrics.iteration}/{self.metrics.total_iterations}", style="bold cyan")
        iter_text.append(f" │ Population {self.metrics.population_size}", style="cyan")

        # Show configured + planned children and estimated survivors
        gen = self.metrics.iter_children_generated
        kept = self.metrics.iter_children_kept
        cfg = self.metrics.children_per_parent_config
        target_total = self.metrics.children_target_total
        keep_rate = (kept / gen) if gen > 0 else 0.0
        est_keep = int(round(target_total * keep_rate)) if target_total > 0 and gen > 0 else 0

        iter_text.append(" │ Children ", style="cyan")
        if cfg > 0:
            iter_text.append(f"{cfg}pp", style="cyan")
        else:
            iter_text.append("--", style="dim")

        if target_total > 0:
            iter_text.append(" │ plan ", style="cyan")
            iter_text.append(f"{target_total}", style="white")
        else:
            iter_text.append(" │ plan --", style="dim")

        if est_keep > 0:
            iter_text.append(" │ est ", style="cyan")
            iter_text.append(f"{est_keep}", style="green")
        else:
            iter_text.append(" │ est --", style="dim")
        center_lines.append(iter_text)

        # === RIGHT COLUMN: Timing + Curriculum/Config ===
        right_lines = []

        # Elapsed time
        elapsed_sec = int(self.metrics.elapsed_time)
        hours, remainder = divmod(elapsed_sec, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        right_lines.append(Text.assemble(("⏱ ", ""), (elapsed, "white"), (" elapsed", "")))

        # Remaining time
        if self.metrics.estimated_remaining > 0:
            remaining_sec = int(self.metrics.estimated_remaining)
            r_hours, r_remainder = divmod(remaining_sec, 3600)
            r_minutes, r_seconds = divmod(r_remainder, 60)
            remaining = f"~{r_hours:02d}:{r_minutes:02d}:{r_seconds:02d}"
            right_lines.append(Text.assemble((remaining, "yellow"), (" remain", "")))
        else:
            right_lines.append(Text("estimating...", style=""))

        run_text = Text()
        run_text.append("Run: ", style="")
        run_text.append(datetime.utcfromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M"), style="white")
        run_text.append(" │ Seed: ", style="")
        run_text.append("n/a" if self.metrics.rng_seed is None else str(self.metrics.rng_seed), style="white")
        right_lines.append(run_text)

        # Mode-specific info in right corner
        if self.metrics.optimize_mode == "exploit":
            # Exploit mode: show task index and eval count
            task_text = Text()
            task_idx = self.metrics.exploit_task_index
            if task_idx >= 0:
                task_text.append(f"Task #{task_idx}", style="bold yellow")
            eval_tasks = self.metrics.exploit_eval_tasks
            if eval_tasks > 0:
                task_text.append(f" │ {eval_tasks} eval", style="dim yellow")
            right_lines.append(task_text if len(task_text) > 0 else Text("", style="dim"))
        elif self.metrics.curriculum_mode:
            phase_text = Text()
            phase_text.append(f"Phase {self.metrics.curriculum_phase + 1}", style="bold green")
            if self.metrics.curriculum_total_tasks > 0:
                pct = self.metrics.curriculum_subset_size / self.metrics.curriculum_total_tasks * 100
                phase_text.append(f" │ {pct:.0f}%", style="dim green")
            right_lines.append(phase_text)
        else:
            # Show key config
            config_text = Text()
            if self.metrics.max_concurrent:
                config_text.append(f"Conc:{self.metrics.max_concurrent}", style="dim")
            if self.metrics.eval_temperature:
                if len(config_text) > 0:
                    config_text.append(" │ ", style="dim")
                config_text.append(f"T:{self.metrics.eval_temperature}", style="dim")
            right_lines.append(config_text if len(config_text) > 0 else Text("", style="dim"))

        # === BUILD 3-COLUMN TABLE ===
        header_table = Table.grid(padding=(0, 2), expand=True)
        header_table.add_column(justify="left", ratio=1)
        header_table.add_column(justify="center", ratio=2)
        header_table.add_column(justify="right", ratio=1)

        # Add rows (aligned to max column length)
        max_rows = max(len(left_lines), len(center_lines), len(right_lines))
        for i in range(max_rows):
            left = left_lines[i] if i < len(left_lines) else Text("")
            center = center_lines[i] if i < len(center_lines) else Text("")
            right = right_lines[i] if i < len(right_lines) else Text("")
            header_table.add_row(left, center, right)

        # Use mode-specific border color
        border_color = "yellow" if self.metrics.optimize_mode == "exploit" else "blue"
        return Panel(header_table, style=f"bold {border_color}", border_style=border_color)

    def _create_metrics_table(self) -> Table:
        """Create attack success metrics table with mode-specific highlighting."""
        trend_width = int(os.getenv("REPRO_TREND_WIDTH", "32"))
        trend_width = max(12, min(trend_width, 60))
        table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1), expand=True)
        table.add_column("Metric", style="cyan", width=7)
        table.add_column("Current", justify="center", style="yellow", width=8)
        table.add_column("Best", justify="center", style="green", width=8)
        table.add_column("", width=5, no_wrap=True)
        table.add_column("Trend", justify="center", width=trend_width, no_wrap=True)
        
        # Add phase history column for curriculum mode (always shown)
        if self.metrics.curriculum_mode:
            table.add_column("", width=16, no_wrap=True)
            table.add_column(
                Text("Phase History", justify="center"),
                justify="left",
                ratio=1,
                min_width=20,
                no_wrap=True,
                overflow="crop",
            )

        # Check if we have any data yet
        has_data = len(self.history.asr) > 0

        # Determine primary metric based on mode
        is_exploit = self.metrics.optimize_mode == "exploit"
        primary_indicator = " ★"  # Star to indicate primary metric

        # Build phase history string for ASR (if curriculum mode)
        phase_history_rows: List[str] = []
        if self.metrics.curriculum_mode and self.metrics.phase_best_history:
            # Sort phases and build compact history string
            sorted_phases = sorted(self.metrics.phase_best_history.items())
            phase_parts = []
            for phase_num, best_asr in sorted_phases:
                phase_count = self.metrics.phase_task_counts.get(phase_num)
                if not phase_count:
                    fallback_count = self.metrics.curriculum_subset_size or 0
                    phase_count = fallback_count
                    if fallback_count:
                        self.metrics.phase_task_counts[phase_num] = fallback_count
                count_text = str(phase_count or 0)
                display_phase_num = phase_num + 1
                # Use emoji to indicate completion status
                if phase_num < self.metrics.curriculum_phase:
                    # Completed phase
                    emoji = "✓" if best_asr >= 1.0 else "○"
                    phase_parts.append(f"[green]{emoji}P{display_phase_num}:{best_asr:.0%}({count_text})[/green]")
                else:
                    # Current phase (ongoing)
                    phase_parts.append(f"[yellow]→P{display_phase_num}:{best_asr:.0%}({count_text})[/yellow]")
            phase_history_rows = [" ".join(phase_parts[i:i + 3]) for i in range(0, len(phase_parts), 3)]
        elif self.metrics.curriculum_mode:
            phase_history_rows = ["[dim]n/a[/dim]"]

        # ASR - Primary for EXPLOIT mode
        asr_trend = ASCIISparkline.create(list(self.history.asr), width=trend_width) if has_data else "─" * trend_width
        asr_color = "green" if self.metrics.current_best_asr > 0.5 else "yellow" if self.metrics.current_best_asr > 0.3 else "dim" if has_data else "dim"
        asr_label = f"ASR{primary_indicator}" if is_exploit else "ASR"
        asr_style = "bold cyan" if is_exploit else "cyan"
        
        asr_row = [
            f"[{asr_style}]{asr_label}[/{asr_style}]",
            f"[{asr_color}]{self.metrics.current_best_asr:.1%}[/{asr_color}]" if has_data else "[dim]0.0%[/dim]",
            f"[green]{self.metrics.best_asr:.1%}[/green]" if has_data else "[dim]0.0%[/dim]",
            "",
            f"[dim]{asr_trend}[/dim]"
        ]
        if self.metrics.curriculum_mode:
            phase_history_row_1 = f"  {phase_history_rows[0]}" if phase_history_rows else ""
            asr_row.append("")
            asr_row.append(phase_history_row_1)
        table.add_row(*asr_row)

        # XFER - Primary for EXPLORE mode
        xfer_trend = ASCIISparkline.create(list(self.history.xfer), width=trend_width)
        xfer_color = "green" if self.metrics.current_best_xfer > 0.5 else "yellow" if self.metrics.current_best_xfer > 0.3 else "dim"
        xfer_label = f"XFER{primary_indicator}" if not is_exploit else "XFER"
        xfer_style = "bold cyan" if not is_exploit else "cyan"
        
        xfer_row = [
            f"[{xfer_style}]{xfer_label}[/{xfer_style}]",
            f"[{xfer_color}]{self.metrics.current_best_xfer:.1%}[/{xfer_color}]",
            f"[green]{self.metrics.best_xfer:.1%}[/green]",
            "",
            f"[dim]{xfer_trend}[/dim]"
        ]
        if self.metrics.curriculum_mode:
            phase_history_row_2 = ""
            if len(phase_history_rows) > 1:
                phase_history_row_2 = "  " + "\n  ".join(phase_history_rows[1:])
            xfer_row.append("")
            xfer_row.append(phase_history_row_2)
        table.add_row(*xfer_row)

        return table

    def _create_metrics_panel(self) -> Panel:
        """Create metrics panel with deltas and compact trends."""
        table = self._create_metrics_table()

        # Update deltas only on improvement (persist last improvement until next one)
        current_asr = self.metrics.best_asr
        current_xfer = self.metrics.best_xfer

        if current_asr > self.metrics.prev_best_asr:
            self.metrics.last_delta_asr = current_asr - self.metrics.prev_best_asr
            self.metrics.prev_best_asr = current_asr
            # Also update length/detectability deltas on ASR improvement
            if len(self.history.attack_length) >= 2:
                self.metrics.last_delta_len = self.history.attack_length[-1] - self.history.attack_length[-2]
            if len(self.history.detectability) >= 2:
                self.metrics.last_delta_det = self.history.detectability[-1] - self.history.detectability[-2]

        if current_xfer > self.metrics.prev_best_xfer:
            self.metrics.last_delta_xfer = current_xfer - self.metrics.prev_best_xfer
            self.metrics.prev_best_xfer = current_xfer

        delta_text = Text()
        if self.metrics.last_delta_asr != 0 or self.metrics.last_delta_xfer != 0:
            delta_text.append(f"ΔASR {self.metrics.last_delta_asr:+.1%}", style="green" if self.metrics.last_delta_asr >= 0 else "red")
            delta_text.append(" │ ", style="dim")
            delta_text.append(f"ΔXFER {self.metrics.last_delta_xfer:+.1%}", style="green" if self.metrics.last_delta_xfer >= 0 else "red")

            if self.metrics.last_delta_len != 0:
                delta_text.append(" │ ", style="dim")
                delta_text.append(f"ΔLen {self.metrics.last_delta_len:+}w", style="green" if self.metrics.last_delta_len <= 0 else "yellow")

            if self.metrics.last_delta_det != 0:
                delta_text.append(" │ ", style="dim")
                delta_text.append(f"ΔDet {self.metrics.last_delta_det:+.0f}", style="green" if self.metrics.last_delta_det <= 0 else "yellow")
        else:
            delta_text.append("ΔASR --  │  ΔXFER --", style="dim")

        # ASR distribution across tasks
        bins = self.metrics.task_asr_bins
        dist_text = Text()
        if any(bins.values()):
            dist_text.append("ASR Dist: ", style="dim")
            dist_text.append(f"<20%:{bins.get('<20%', 0)}", style="red")
            dist_text.append(" │ ", style="dim")
            dist_text.append(f"20-50%:{bins.get('20-50%', 0)}", style="yellow")
            dist_text.append(" │ ", style="dim")
            dist_text.append(f"50-80%:{bins.get('50-80%', 0)}", style="cyan")
            dist_text.append(" │ ", style="dim")
            dist_text.append(f">80%:{bins.get('>80%', 0)}", style="green")

        mode_indicator = "EXPLOIT" if self.metrics.optimize_mode == "exploit" else "EXPLORE"
        separator = Text("─" * 50, style="dim")
        content_parts = [table, separator, delta_text]
        if any(bins.values()):
            content_parts.append(dist_text)
        content = Group(*content_parts)
        return Panel(content, title=f"[bold]Attack Metrics ({mode_indicator})[/bold]", border_style="cyan")

    def _create_task_intel_panel(self) -> Panel:
        """Create compact task intelligence panel."""
        info = Table.grid(padding=(0, 1))
        info.add_column(justify="left")
        info.add_column(justify="right")

        info.add_row("Task Status:", f"[green]✓{self.metrics.mastered_tasks}[/green] "
                                     f"[yellow]~{self.metrics.in_progress_tasks}[/yellow] "
                                     f"[red]✗{self.metrics.stubborn_tasks}[/red]")

        if self.metrics.hardest_task_indices:
            hardest = ", ".join(str(i) for i in self.metrics.hardest_task_indices[:5])
            info.add_row("Hardest:", f"[red]{hardest}[/red]")
        else:
            info.add_row("Hardest:", "[dim]n/a[/dim]")

        if self.metrics.curriculum_mode and self.metrics.curriculum_total_tasks:
            subset_pct = (self.metrics.curriculum_subset_size / self.metrics.curriculum_total_tasks) * 100
            info.add_row("Subset:", f"{self.metrics.curriculum_subset_size}/{self.metrics.curriculum_total_tasks} ({subset_pct:.0f}%)")

        if self.metrics.curriculum_phase_switches:
            info.add_row("Switches:", f"🔄 {self.metrics.curriculum_phase_switches}")

        return Panel(info, title="[bold]Task Intel[/bold]", border_style="green")

    def _create_strategy_table(self) -> Panel:
        """Create compact strategy effectiveness table (top 3 mutation + crossover)."""
        has_mutation = bool(self.metrics.top_strategies)
        has_crossover = bool(self.metrics.crossover_stats.get("total_attempts", 0) > 0)

        if not has_mutation and not has_crossover:
            return Panel(
                "[dim]No strategy data yet...[/dim]",
                title="[bold]Strategies[/bold]",
                border_style="yellow")

        table = Table(show_header=False, header_style="bold yellow", box=None, padding=(0, 2), expand=False)
        table.add_column("Strategy", style="white", max_width=32, no_wrap=True, overflow="ellipsis")
        table.add_column("Count", justify="right", width=4)
        table.add_column("Bar", justify="left", width=14)

        bar_width = 14

        # Limit to top 8 mutations + crossover
        max_mutation_rows = 8

        # === MUTATION STRATEGIES (top 8) ===
        if has_mutation:
            table.add_row("[bold dim green]MUTATION[/bold dim green]", "", "")

            for i, (strategy, rate) in enumerate(self.metrics.top_strategies[:max_mutation_rows], 1):
                # Create mini bar (compact for right column)
                filled = int(rate * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)

                color = "green" if rate > 0.6 else "yellow" if rate > 0.4 else "red"

                strategy_name = strategy.replace("_", " ").title()

                table.add_row(
                    f"[green]#{i}[/green] [white]{strategy_name}[/white]",
                    f"[{color}]{rate:.0%}[/{color}]",
                    f"[{color}]{bar}[/{color}]"
                )

        # === CROSSOVER STRATEGIES ===
        if has_crossover:
            if has_mutation:
                table.add_row("[dim]───────────[/dim]", "", "")

            table.add_row("[bold magenta]CROSSOVER[/bold magenta]", "", "")

            crossover_stats = self.metrics.crossover_stats
            per_type = crossover_stats.get("per_type", {})

            type_configs = [
                ("2-Parent", per_type.get("2_parent", {})),
                ("3-Parent", per_type.get("3_parent", {})),
                ("Self", per_type.get("self_crossover", {})),
            ]

            type_idx = 1
            for type_name, type_stats in type_configs:
                attempts = type_stats.get("attempts", 0)
                if attempts > 0:
                    rate = type_stats.get("success_rate", 0.0)
                    filled = int(rate * bar_width)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    color = "green" if rate > 0.5 else "yellow" if rate > 0.3 else "red"

                    table.add_row(
                        f"[magenta]#{type_idx}[/magenta] [white]{type_name}[/white] ({attempts})",
                        f"[{color}]{rate:.0%}[/{color}]",
                        f"[{color}]{bar}[/{color}]"
                    )
                    type_idx += 1

            # If no per-type data, show overall
            if type_idx == 1:
                total_attempts = crossover_stats.get("total_attempts", 0)
                success_rate = crossover_stats.get("success_rate", 0.0)
                if total_attempts > 0:
                    filled = int(success_rate * bar_width)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    color = "green" if success_rate > 0.4 else "yellow" if success_rate > 0.2 else "red"

                    table.add_row(
                        f"[magenta]#1[/magenta] [white]Overall[/white] ({total_attempts})",
                        f"[{color}]{success_rate:.0%}[/{color}]",
                        f"[{color}]{bar}[/{color}]"
                    )

        # Simple title
        title_parts = []
        if has_mutation:
            title_parts.append(f"M:{min(len(self.metrics.top_strategies), 3)}")
        if has_crossover:
            crossover_types = sum(1 for t in crossover_stats.get("per_type", {}).values()
                                  if t.get("attempts", 0) > 0)
            title_parts.append(f"C:{crossover_types or 1}")

        title_suffix = f" ({'+'.join(title_parts)})" if title_parts else ""

        return Panel(table, title=f"[bold]Strategy Effectiveness{title_suffix}[/bold]", border_style="yellow")

    def _create_status_info(self) -> Panel:
        """Create compact status panel (diversity/stagnation/task info moved elsewhere)."""
        info = Table.grid(padding=(0, 1))
        info.add_column(style="cyan", width=13)
        info.add_column(style="white")

        # Task progress display (different for exploit vs explore mode)
        is_exploit = self.metrics.optimize_mode == "exploit"
        if is_exploit:
            # Exploit mode: show single task info
            eval_tasks = self.metrics.exploit_eval_tasks
            if eval_tasks > 0:
                task_label = f"{eval_tasks} eval task" if eval_tasks == 1 else f"{eval_tasks} eval tasks"
                asr_pct = self.metrics.best_asr * 100
                status = "✓ MASTERED" if asr_pct >= 99.9 else f"{asr_pct:.0f}%"
                status_color = "green" if asr_pct >= 99.9 else "yellow" if asr_pct > 0 else "dim"
                info.add_row("Task:", Text.assemble(
                    (task_label, "cyan"), (" │ ", "dim"),
                    (status, status_color)
                ))
            else:
                info.add_row("Task:", Text("initializing...", style="dim"))
        else:
            # Explore mode: show full task set progress (cracked vs remaining)
            total = self.metrics.total_tasks_full
            cracked = self.metrics.tasks_cracked_full
            remaining = self.metrics.tasks_remaining_full or (total - cracked if total > 0 else 0)
            if total > 0:
                cracked_pct = cracked / total * 100
                pct_color = "green" if cracked_pct >= 80 else "cyan" if cracked_pct >= 50 else "yellow" if cracked_pct >= 20 else "red"
                info.add_row("Tasks:", Text.assemble(
                    (f"{cracked}", "green"), ("✓ / ", "dim"),
                    (f"{remaining}", "red" if remaining > 0 else "green"), ("✗", "dim"),
                    (f" of {total}", "dim"),
                    (f" ({cracked_pct:.0f}%)", pct_color)
                ))

        # Efficiency metrics with unique rate
        evaluated = self.metrics.candidates_evaluated
        unique = self.metrics.unique_candidates
        unique_rate = unique / evaluated * 100 if evaluated > 0 else 0
        unique_color = "green" if unique_rate >= 80 else "yellow" if unique_rate >= 50 else "red"
        info.add_row("Candidates:", Text.assemble(
            (f"{evaluated}", "white"), (" eval │ ", "dim"),
            (f"{unique}", unique_color), (f" ({unique_rate:.0f}%)", "dim")
        ))

        # Pareto frontier size (current iteration)
        info.add_row("Pareto:", Text.assemble(
            (f"{self.metrics.frontier_size}", "magenta"),
            (" solutions", "dim"),
        ))

        # Convergence tracking (combined on fewer lines)
        first_suc = self.metrics.iteration_to_first_success
        threshold = self.metrics.iteration_to_threshold
        if first_suc or threshold:
            conv_text = Text()
            if first_suc:
                conv_text.append(f"1st: Iter {first_suc}", style="green")
            if threshold:
                if first_suc:
                    conv_text.append(" │ ", style="dim")
                conv_text.append(f"50%: Iter {threshold}", style="green")
            info.add_row("Convergence:", conv_text)
        else:
            info.add_row("Convergence:", Text("waiting...", style="dim"))

        # Curriculum phase info
        if self.metrics.curriculum_mode:
            phase_text = Text.assemble(
                (f"P{self.metrics.curriculum_phase + 1}", "green"),
                (" │ Switches ", "dim"),
                (f"{self.metrics.curriculum_phase_switches}", "cyan" if self.metrics.curriculum_phase_switches > 0 else "dim"),
                (" │ Reshuffles ", "dim"),
                (f"{self.metrics.curriculum_reshuffles}", "yellow" if self.metrics.curriculum_reshuffles > 0 else "dim"),
            )
            info.add_row("Phase:", phase_text)

        # Attack quality (combined)
        length = self.metrics.attack_length
        detect = int(self.metrics.detectability)
        length_color = "green" if length < 150 else "yellow" if length < 250 else "red"
        detect_color = "green" if detect == 0 else "yellow" if detect < 2 else "red"
        quality_text = Text.assemble(
            (f"{length}w", length_color), (" │ ", "dim"),
            (f"{detect} flags", detect_color),
            (" │ Stealth ✓" if detect == 0 else "", "green")
        )
        info.add_row("Quality:", quality_text)

        # Stagnation indicator
        stag_level = self.metrics.stagnation_level
        iters_since = self.metrics.iterations_since_improvement
        if stag_level == 0:
            stag_style = "green"
            stag_label = "Normal"
        elif stag_level == 1:
            stag_style = "yellow"
            stag_label = "Low"
        elif stag_level == 2:
            stag_style = "orange1"
            stag_label = "Moderate"
        else:
            stag_style = "red"
            stag_label = "High"
        stag_text = Text.assemble(
            (f"{iters_since} iters", "white"),
            (" │ ", "dim"),
            (stag_label, stag_style),
        )
        info.add_row("Stagnation:", stag_text)

        # Restarts (always show)
        restart_style = "yellow" if self.metrics.restart_count > 0 else "dim"
        info.add_row("Restarts:", Text.assemble(
            (f"{self.metrics.restart_count}", restart_style),
            (" │ ε=", "dim"), (f"{self.metrics.exploration_epsilon:.2f}", "white"),
            (" T=", "dim"), (f"{self.metrics.mutation_temperature:.2f}", "white")
        ))

        # GPU (if available)
        gpu_utils = self.metrics.gpu_utils or []
        gpu_mems = self.metrics.gpu_memories or []
        if GPU_AVAILABLE and gpu_utils and any(u > 0 for u in gpu_utils):
            gpu_text = Text(no_wrap=False, overflow="fold")
            for idx, util in enumerate(gpu_utils):
                mem = gpu_mems[idx] if idx < len(gpu_mems) else 0.0
                color = "green" if util > 70 else "yellow"
                if idx > 0:
                    gpu_text.append(" │ ", style="dim")
                gpu_text.append(f"{idx}:", style="dim")
                gpu_text.append(f"{util:.0f}%", style=color)
                gpu_text.append(f" ({mem:.1f}GB)", style="dim")
            info.add_row("GPU:", gpu_text)
        elif GPU_AVAILABLE and self.metrics.gpu_util > 0:
            gpu_color = "green" if self.metrics.gpu_util > 70 else "yellow"
            info.add_row("GPU:", f"[{gpu_color}]{self.metrics.gpu_util:.0f}%[/{gpu_color}] ({self.metrics.gpu_memory:.1f}GB)")

        return Panel(info, title="[bold]Status[/bold]", border_style="magenta")

    def _create_progress_bar(self) -> Panel:
        """Create progress bar."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )

        task = progress.add_task(
            "[cyan]Evolving attack prompts...",
            total=self.metrics.total_iterations,
            completed=self.metrics.iteration
        )

        return Panel(progress, border_style="dim")

    def _create_candidate_panel(self) -> Panel:
        """Create enhanced panel showing current attack candidate with quality indicators."""
        if not self.metrics.current_candidate:
            content = Text("Initializing...", style="dim italic")
        else:
            lines = []

            # Attack quality indicators row
            quality_indicators = []

            # ASR indicator
            asr = self.metrics.current_best_asr
            if asr > 0.7:
                asr_indicator = f"🎯 ASR: {asr:.0%}"
                asr_style = "bold green"
            elif asr > 0.4:
                asr_indicator = f"⚡ ASR: {asr:.0%}"
                asr_style = "yellow"
            elif asr > 0.1:
                asr_indicator = f"🔸 ASR: {asr:.0%}"
                asr_style = "red"
            else:
                asr_indicator = f"❌ ASR: {asr:.0%}"
                asr_style = "dim"

            quality_indicators.append(Text(asr_indicator, style=asr_style))

            # Attack length indicator
            length = self.metrics.attack_length
            if length > 0:
                if length < 100:
                    length_indicator = f"📏 {length}w"
                    length_style = "green"
                elif length < 200:
                    length_indicator = f"📏 {length}w"
                    length_style = "yellow"
                else:
                    length_indicator = f"📏 {length}w"
                    length_style = "red"
            else:
                length_indicator = "📏 --w"
                length_style = "dim"

            quality_indicators.append(Text(f" │ {length_indicator}", style=length_style))

            # Detectability indicator
            detectability = self.metrics.detectability
            if detectability == 0:
                detect_indicator = "🛡️ Stealth"
                detect_style = "green"
            elif detectability < 3:
                detect_indicator = f"⚠️ {int(detectability)} flags"
                detect_style = "yellow"
            else:
                detect_indicator = f"🚨 {int(detectability)} flags"
                detect_style = "red"

            quality_indicators.append(Text(f" │ {detect_indicator}", style=detect_style))

            # Combine quality indicators
            quality_line = Text()
            for indicator in quality_indicators:
                quality_line.append_text(indicator)

            lines.append(quality_line)

            # Candidate summary - collapse newlines, use available space (~3 lines)
            candidate_text = self.metrics.current_candidate.strip()
            # Replace newlines with spaces to maximize display density
            candidate_text = ' '.join(candidate_text.split())
            max_chars = 560  # ~6-8 lines worth
            if len(candidate_text) > max_chars:
                display_text = candidate_text[:max_chars].rsplit(' ', 1)[0] + "..."
            else:
                display_text = candidate_text
            lines.append(Text(display_text, style="yellow", overflow="fold"))

            content = Group(*lines)

        # Dynamic title with performance indicator
        asr = self.metrics.current_best_asr
        if asr > 0.5:
            title_icon = "🎯"
            title_style = "bold green"
        elif asr > 0.2:
            title_icon = "⚡"
            title_style = "bold yellow"
        else:
            title_icon = "🔸"
            title_style = "bold"

        return Panel(
            content,
            title=f"[{title_style}]{title_icon} Current Attack Candidate (Seed)[/{title_style}]",
            border_style="yellow",
            padding=(0, 1)
        )

    def _create_children_panel(self) -> Panel:
        """Create panel to display recently generated children with enhanced visualization."""
        if not self.metrics.recent_children:
            content = Text("No children generated yet...", style="dim italic")
        else:
            table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
            table.add_column("#", width=3, style="dim")
            table.add_column("Mutation Type", width=15, style="cyan")
            table.add_column("Child Prompt", style="white", overflow="fold", no_wrap=False)

            # Mutation type icons and colors for better visualization
            mutation_styles = {
                'word_level': ('🔤', 'blue'),
                'semantic': ('🧠', 'magenta'),
                'crossover': ('🔀', 'yellow'),
                'context': ('📝', 'green'),
                'role': ('🎭', 'red'),
                'syntax': ('⚡', 'cyan'),
                'unknown': ('❓', 'dim')
            }

            def get_mutation_display(mutation_type):
                """Get enhanced mutation type display with icon and color."""
                if not mutation_type or mutation_type == 'unknown':
                    icon, color = mutation_styles['unknown']
                    return f"[{color}]{icon} Unknown[/{color}]"

                # Match mutation type to category
                mutation_lower = mutation_type.lower()
                if 'word' in mutation_lower:
                    icon, color = mutation_styles['word_level']
                    display_name = 'Word-Level'
                elif 'semantic' in mutation_lower:
                    icon, color = mutation_styles['semantic']
                    display_name = 'Semantic'
                elif 'crossover' in mutation_lower or 'cross' in mutation_lower:
                    icon, color = mutation_styles['crossover']
                    display_name = 'Crossover'
                elif 'context' in mutation_lower:
                    icon, color = mutation_styles['context']
                    display_name = 'Context'
                elif 'role' in mutation_lower:
                    icon, color = mutation_styles['role']
                    display_name = 'Role-Play'
                elif 'syntax' in mutation_lower:
                    icon, color = mutation_styles['syntax']
                    display_name = 'Syntax'
                else:
                    icon, color = mutation_styles['unknown']
                    display_name = mutation_type[:12].title()

                return f"[{color}]{icon} {display_name}[/{color}]"

            for i, child in enumerate(self.metrics.recent_children[-12:]):
                # Enhanced mutation type display
                mutation_type = child.get('mutation', 'unknown')
                mutation_display = get_mutation_display(mutation_type)

                # Enhanced child text with better preview
                child_text = child.get('text', '')

                # Show first 200 chars but try to break at sentence boundary
                max_length = 200
                if len(child_text) > max_length:
                    preview = child_text[:max_length]
                    # Try to break at last sentence or period
                    last_period = preview.rfind('.')
                    last_exclaim = preview.rfind('!')
                    last_question = preview.rfind('?')

                    best_break = max(last_period, last_exclaim, last_question)
                    if best_break > max_length * 0.7:  # If we found a good break point
                        display_text = child_text[:best_break + 1]
                    else:
                        # Break at word boundary
                        last_space = preview.rfind(' ')
                        if last_space > max_length * 0.8:
                            display_text = child_text[:last_space] + "..."
                        else:
                            display_text = preview + "..."
                else:
                    display_text = child_text

                table.add_row(
                    f"[dim]#{i + 1}[/dim]",
                    mutation_display,
                    display_text
                )

            # Enhanced summary with generation statistics
            total_gen = self.metrics.num_children_generated
            recent_count = len(self.metrics.recent_children)

            # Calculate mutation type distribution in recent children
            type_counts = {}
            for child in self.metrics.recent_children:
                mutation_type = child.get('mutation', 'unknown')
                # Categorize for summary
                if 'word' in mutation_type.lower():
                    category = 'Word'
                elif 'semantic' in mutation_type.lower():
                    category = 'Semantic'
                elif 'crossover' in mutation_type.lower():
                    category = 'Crossover'
                else:
                    category = 'Other'

                type_counts[category] = type_counts.get(category, 0) + 1

            # Build summary text
            summary_parts = [f"Total Generated: {total_gen}"]
            if type_counts:
                dist_text = " | ".join([f"{k}:{v}" for k, v in type_counts.items()])
                summary_parts.append(f"Recent Mix: {dist_text}")

            summary = Text(f"\n{' • '.join(summary_parts)}", style="dim green")
            content = Group(table, summary)

        return Panel(
            content,
            title=f"[bold]Generated Children (Last 12) 🔄[/bold]",
            border_style="cyan",
            padding=(0, 1),
            expand=True
        )

    def _create_reflection_panel(self) -> Panel:
        """Create panel to display Reflective LM analysis with context."""
        if not self.metrics.latest_reflection:
            content = Text("No reflections yet...", style="dim italic")
        else:
            ref = self.metrics.latest_reflection
            lines = []

            # Header row: Parent ID, Iteration
            header_parts = []
            if ref.get('parent_id'):
                header_parts.append(f"Parent: {ref['parent_id'][:8]}")
            if ref.get('iteration'):
                header_parts.append(f"Iter: {ref['iteration']}")

            if header_parts:
                lines.append(Text(" │ ".join(header_parts), style="dim cyan"))

            # Scores row: Parent's ASR and XFER (this parent only)
            asr = ref.get('asr', 0.0)
            xfer = ref.get('xfer', 0.0)

            score_text = Text()
            # Parent's ASR
            asr_color = "green" if asr > 0.5 else "yellow" if asr > 0.3 else "red" if asr > 0 else "dim"
            score_text.append(f"Parent ASR: ", style="dim")
            score_text.append(f"{asr:.0%}", style=asr_color)

            # Parent's XFER
            score_text.append(f" │ XFER: ", style="dim")
            xfer_color = "green" if xfer > 0.5 else "yellow" if xfer > 0.3 else "dim"
            score_text.append(f"{xfer:.0%}", style=xfer_color)

            lines.append(score_text)

            # Failure reasons (1 line)
            if ref.get('failure_reasons'):
                reason = ref['failure_reasons'][0].strip()
                lines.append(Text(f"Failure: {reason}", style="red", overflow="fold"))

            # Target weaknesses (1 line)
            if ref.get('target_weaknesses'):
                weakness = ref['target_weaknesses'][0].strip()
                lines.append(Text(f"Weakness: {weakness}", style="green", overflow="fold"))

            # Recommended techniques (1 line)
            if ref.get('recommended_techniques'):
                tech = ref['recommended_techniques'][0].strip()
                lines.append(Text(f"Next: {tech}", style="blue", overflow="fold"))

            content = Group(*lines)

        return Panel(
            content,
            title="[bold]🤖 Reflective LM[/bold]",
            border_style="magenta",
            padding=(0, 1)
        )

    def _create_activity_panel(self) -> Panel:
        """Create panel to display recent activity log."""
        if not self.metrics.recent_activities:
            content = Text("Waiting...", style="dim italic")
        else:
            lines = []
            # Show last 12 entries
            for activity in list(self.metrics.recent_activities)[-12:]:
                # Remove truncation - let Rich handle wrapping naturally
                # Rich will wrap text at word boundaries based on available width
                lines.append(Text(activity, style="white", overflow="fold"))
            content = Group(*lines)

        return Panel(
            content,
            title="[bold]Activity[/bold]",
            border_style="blue",
            padding=(0, 1),  # Increased padding for better readability
            expand=True  # Allow panel to expand to available width
        )

    def _create_generation_stats_panel(self) -> Panel:
        """Create panel showing generation efficiency and reflective LM stats."""
        # Use 4-column grid for better layout
        grid = Table.grid(padding=(0, 1), expand=False)
        grid.add_column(justify="left")   # Label 1
        grid.add_column(justify="left", min_width=14)   # Value 1
        grid.add_column(justify="left", min_width=16)   # Spacer
        grid.add_column(justify="left")   # Label 2
        grid.add_column(justify="left", min_width=14)   # Value 2

        # ═══════════════════════════════════════════════════════════════
        # SECTION 1: Core Generation Stats
        # ═══════════════════════════════════════════════════════════════
        grid.add_row(
            Text("▎ Generation", style="bold underline cyan"), Text("", style="dim"),
            "",  # Spacer
            Text("▎ Operations", style="bold underline cyan"), Text("", style="dim")
        )

        gen = self.metrics.iter_children_generated
        kept = self.metrics.iter_children_kept
        mut_att = self.metrics.iter_mutation_attempts
        mut_suc = self.metrics.iter_mutation_success

        if gen > 0:
            kept_pct = kept / gen * 100
            kept_color = "green" if kept_pct >= 70 else "yellow" if kept_pct >= 50 else "red"
            children_val = Text.assemble((f"{kept}/{gen}", kept_color), (f" ({kept_pct:.0f}%)", "dim"))
        else:
            children_val = Text("--", style="dim")

        if mut_att > 0:
            mut_rate = mut_suc / mut_att * 100
            mut_color = "green" if mut_rate >= 60 else "yellow" if mut_rate >= 30 else "red"
            mut_val = Text.assemble((f"{mut_suc}/{mut_att}", mut_color), (f" ({mut_rate:.0f}%)", "dim"))
        else:
            mut_val = Text("--", style="dim")

        grid.add_row("  Children Kept:", children_val, "", "  Mutations:", mut_val)

        cross_att = self.metrics.iter_crossover_attempts
        cross_suc = self.metrics.iter_crossover_success
        improved = self.metrics.iter_improved_count
        pareto = self.metrics.iter_pareto_worthy

        if cross_att > 0:
            cross_rate = cross_suc / cross_att * 100
            cross_color = "green" if cross_rate >= 60 else "yellow" if cross_rate >= 30 else "red"
            cross_val = Text.assemble((f"{cross_suc}/{cross_att}", cross_color), (f" ({cross_rate:.0f}%)", "dim"))
        else:
            cross_val = Text("--", style="dim")

        improve_val = Text.assemble(
            (f"{improved}", "cyan" if improved > 0 else "dim"),
            (f" (+{pareto} Pareto)", "magenta" if pareto > 0 else "dim")
        )

        grid.add_row("  Improved:", improve_val, "", "  Crossovers:", cross_val)

        # ═══════════════════════════════════════════════════════════════
        # SECTION 2: Reflective LM Stats
        # ═══════════════════════════════════════════════════════════════
        grid.add_row(
            Text("▎ 🤖 Reflective LM", style="bold underline magenta"), Text("", style="dim"),
            "",  # Spacer
            Text("▎ Results", style="bold underline magenta"), Text("", style="dim")
        )

        tried = self.metrics.reflect_strategies_tried
        success = self.metrics.reflect_strategies_success
        failed = self.metrics.reflect_strategies_failed
        new_strats = self.metrics.reflect_new_strategies

        new_val = Text(f"{new_strats}", style="cyan" if new_strats > 0 else "dim")

        if tried > 0:
            success_rate = success / tried * 100
            rate_color = "green" if success_rate >= 50 else "yellow" if success_rate >= 25 else "red"
            result_val = Text.assemble(
                (f"{success} ✓", "green"), ("  ", "dim"), (f"{failed} ✗", "red"),
                (f"  ({success_rate:.0f}%)", rate_color)
            )
        else:
            result_val = Text("waiting...", style="dim")

        grid.add_row("  New Strategies:", new_val, "", "  Success/Fail:", result_val)

        # Recent strategy names (one per row)
        if self.metrics.recent_reflect_strategies:
            recent = self.metrics.recent_reflect_strategies[-2:]
            first = recent[0] if len(recent) > 0 else ""
            second = recent[1] if len(recent) > 1 else ""
            first_text = Text(first if first else "none yet", style="cyan", overflow="ellipsis", no_wrap=True)
            second_text = Text(second, style="cyan", overflow="ellipsis", no_wrap=True) if second else Text("", style="dim")
        else:
            first_text = Text("none yet", style="dim", no_wrap=True)
            second_text = Text("", style="dim")
        grid.add_row("  Recent:", first_text, "", "", "")

        # ═══════════════════════════════════════════════════════════════
        # SECTION 3: Parsing & Quality
        # ═══════════════════════════════════════════════════════════════
        grid.add_row(
            Text("▎ 📝 Parsing", style="bold underline blue"), Text("", style="dim"),
            "",  # Spacer
            Text("▎ Quality", style="bold underline blue"), Text("", style="dim")
        )

        valid = self.metrics.gen_valid
        invalid = self.metrics.gen_invalid
        empty = self.metrics.gen_empty
        parse_err = self.metrics.parse_errors_iter
        use_cached = (
            self.metrics.iter_children_generated == 0
            and self.metrics.iter_children_kept == 0
            and (valid + invalid + empty) == 0
            and parse_err == 0
        )
        if use_cached:
            valid = self._last_parse_stats["valid"]
            invalid = self._last_parse_stats["invalid"]
            empty = self._last_parse_stats["empty"]
            parse_err = self._last_parse_stats["parse_errors"]
        else:
            self._last_parse_stats.update(
                {
                    "valid": valid,
                    "invalid": invalid,
                    "empty": empty,
                    "parse_errors": parse_err,
                }
            )

        total_gen = valid + invalid + empty
        if total_gen > 0:
            valid_pct = valid / total_gen * 100
            valid_color = "green" if valid_pct >= 90 else "yellow" if valid_pct >= 70 else "red"
            parse_val = Text.assemble(
                (f"{valid} ✓", "green"), ("  ", "dim"),
                (f"{invalid} ✗", "red" if invalid > 0 else "dim"), ("  ", "dim"),
                (f"{empty} ∅", "yellow" if empty > 0 else "dim")
            )
        else:
            parse_val = Text.assemble(
                ("0 ✓", "dim"), ("  ", "dim"),
                ("0 ✗", "dim"), ("  ", "dim"),
                ("0 ∅", "dim")
            )

        err_val = Text(f"{parse_err}", style="red" if parse_err > 0 else "green")

        grid.add_row("  Valid/Invalid/Empty:", parse_val, "", "  Parse Errors:", err_val)

        # Self-correction & Duplicates
        self_corr_att = self.metrics.self_corrections_attempted
        self_corr_suc = self.metrics.self_corrections_success
        if use_cached:
            self_corr_att = self._last_parse_stats["self_corr_att"]
            self_corr_suc = self._last_parse_stats["self_corr_suc"]
        else:
            self._last_parse_stats.update(
                {"self_corr_att": self_corr_att, "self_corr_suc": self_corr_suc}
            )
        dups = self.metrics.iter_duplicates_filtered

        if self_corr_att > 0:
            corr_rate = self_corr_suc / self_corr_att * 100
            corr_color = "green" if corr_rate >= 70 else "yellow" if corr_rate >= 40 else "red"
            corr_val = Text.assemble((f"{self_corr_suc}/{self_corr_att}", corr_color), (f" ({corr_rate:.0f}%)", "dim"))
        else:
            corr_val = Text("0", style="dim")

        dup_val = Text(f"{dups}", style="yellow" if dups > 0 else "dim")

        grid.add_row("  Self-Corrections:", corr_val, "", "  Duplicates:", dup_val)

        # ═══════════════════════════════════════════════════════════════
        # SECTION 4: Diversity & Delta
        # ═══════════════════════════════════════════════════════════════
        grid.add_row(
            Text("▎ 🌀 Diversity", style="bold underline yellow"), Text("", style="dim"),
            "",  # Spacer
            Text("▎ Δ vs Parent", style="bold underline yellow"), Text("", style="dim")
        )

        div_score = self.metrics.diversity_score
        div_trend = self.metrics.diversity_trend
        div_injections = self.metrics.diversity_injections
        crowding = self.metrics.crowding_replacements

        if self.metrics.diversity_healthy:
            div_color = "green"
        elif div_score >= 0.4:
            div_color = "yellow"
        else:
            div_color = "red"

        div_val = Text.assemble(
            (f"{div_score:.0%}", div_color),
            (f" {div_trend}", div_color)
        )

        asr_delta = self.metrics.avg_child_asr_delta
        xfer_delta = self.metrics.avg_child_xfer_delta
        if use_cached:
            asr_delta = self._last_quality_deltas["asr"]
            xfer_delta = self._last_quality_deltas["xfer"]
        else:
            self._last_quality_deltas.update({"asr": asr_delta, "xfer": xfer_delta})
        asr_color = "green" if asr_delta > 0 else "red" if asr_delta < 0 else "dim"
        xfer_color = "green" if xfer_delta > 0 else "red" if xfer_delta < 0 else "dim"
        delta_val = Text.assemble(
            ("ASR ", "dim"), (f"{asr_delta:+.1%}", asr_color),
            ("  XFER ", "dim"), (f"{xfer_delta:+.1%}", xfer_color)
        )

        grid.add_row("  Score:", div_val, "", "  Avg Quality:", delta_val)

        inject_val = Text.assemble(
            (f"{div_injections}", "cyan" if div_injections > 0 else "dim"),
            (" injections", "dim"),
            ("  │  ", "dim"),
            (f"{crowding}", "magenta" if crowding > 0 else "dim"),
            (" crowding", "dim")
        )

        grid.add_row("  Actions:", inject_val, "", "", "")

        return Panel(grid, title="[bold]Generation Stats[/bold]", border_style="green")

    def _create_task_mastery_panel(self) -> Panel:
        """Create panel showing task mastery distribution and strategies for hard tasks."""
        if self.metrics.optimize_mode == "exploit":
            info = Table.grid(padding=(0, 1))
            info.add_column(justify="left")
            info.add_column(justify="right")

            target_preview = self.metrics.target_task.strip() if self.metrics.target_task else "n/a"
            if len(target_preview) > 70:
                target_preview = f"{target_preview[:70]}..."
            info.add_row("Target:", target_preview)

            eval_tasks = self.metrics.exploit_eval_tasks
            rewordings = self.metrics.exploit_rewording_count
            if eval_tasks <= 0:
                info.add_row("Eval Tasks:", "[dim]n/a[/dim]")
            else:
                info.add_row("Eval Tasks:", f"{eval_tasks}")

            source = self.metrics.exploit_rewording_source or "unknown"
            if rewordings > 0:
                info.add_row("Rewordings:", f"{rewordings} ({source})")
            else:
                info.add_row("Rewordings:", "[dim]0[/dim]")

            info.add_row("Best ASR:", f"[bold green]{self.metrics.best_asr:.1%}[/bold green]")
            info.add_row("Current ASR:", f"{self.metrics.current_best_asr:.1%}")

            failures = self.metrics.exploit_top_failures or []
            if failures:
                info.add_row("Worst:", failures[0])
                for extra in failures[1:3]:
                    info.add_row("", extra)
            else:
                info.add_row("Worst:", "[dim]none[/dim]")

            return Panel(info, title="[bold]Task Focus[/bold]", border_style="yellow")

        # Elite task stats (Pareto survivors only - what our best can do NOW)
        elite_mastered = self.metrics.elite_mastered_tasks
        elite_stubborn = self.metrics.elite_stubborn_tasks
        elite_in_progress = self.metrics.elite_in_progress_tasks
        elite_total = elite_mastered + elite_stubborn + elite_in_progress
        elite_window = self.metrics.elite_window_iterations
        if elite_total == 0:
            elite_total = self.metrics.curriculum_subset_size or 1

        # Full task set stats
        full_total = self.metrics.total_tasks_full or 1
        full_cracked = self.metrics.tasks_cracked_full
        full_remaining = self.metrics.tasks_remaining_full or (full_total - full_cracked)

        # Use 3-column grid layout with better spacing
        grid = Table.grid(padding=(0, 3), expand=True)
        grid.add_column(justify="left", ratio=1)   # Left: Mastery bars
        grid.add_column(justify="left", ratio=1)   # Middle: Tasks/Families
        grid.add_column(justify="left", ratio=1)   # Right: Stubborn tasks list
        console_width = getattr(self.console, "size", None).width if self.console else 120
        panel_width = max(console_width - 2, 60)
        col_width = max(int((panel_width - 18) / 3), 10)
        underline_width = max(8, int(col_width * 0.6))

        def _header_with_partial_underline(label: str, style: str) -> Text:
            header = Text()
            header.append(label, style=style)
            pad_len = max(0, underline_width - header.cell_len)
            if pad_len:
                header.append(" " * pad_len, style=style)
            return header

        # === LEFT COLUMN: Build mastery distribution ===
        bar_width = 12

        # --- Elite Section (Pareto survivors only) ---
        # Shows what our BEST candidates can do RIGHT NOW
        elite_header = Text.assemble(
            ("▎ Elite ", "bold underline cyan"),
            (f"(win={elite_window}) ", "underline cyan"),
            (f"{elite_mastered}", "green bold"),
            ("/", "dim"),
            (f"{elite_total}", "dim"),
        )

        # Mastered bar (elite) - tasks elite cracks consistently (>80%)
        elite_mastered_pct = elite_mastered / elite_total if elite_total > 0 else 0
        elite_mastered_filled = int(elite_mastered_pct * bar_width)
        elite_mastered_bar = "█" * elite_mastered_filled + "░" * (bar_width - elite_mastered_filled)
        elite_mastered_text = Text.assemble(
            ("✓ Mastered  ", "green"),
            (elite_mastered_bar, "green"),
            (f" {elite_mastered_pct:.0%}", "green"),
            (f" ({elite_mastered})", "dim")
        )

        # Stubborn bar (elite) - tasks elite struggles with (<20%)
        elite_stubborn_pct = elite_stubborn / elite_total if elite_total > 0 else 0
        elite_stubborn_filled = int(elite_stubborn_pct * bar_width)
        elite_stubborn_bar = "█" * elite_stubborn_filled + "░" * (bar_width - elite_stubborn_filled)
        elite_stubborn_text = Text.assemble(
            ("✗ Stubborn  ", "red"),
            (elite_stubborn_bar, "red"),
            (f" {elite_stubborn_pct:.0%}", "red"),
            (f" ({elite_stubborn})", "dim")
        )

        # --- Full Task Set Section ---
        full_header = Text.assemble(
            ("▎ Full Set ", "bold underline magenta"),
            (f"({full_total} tasks)", "underline magenta"),
        )

        # Cracked bar (full set)
        full_cracked_pct = full_cracked / full_total if full_total > 0 else 0
        full_cracked_filled = int(full_cracked_pct * bar_width)
        full_cracked_bar = "█" * full_cracked_filled + "░" * (bar_width - full_cracked_filled)
        full_cracked_text = Text.assemble(
            ("✓ Cracked   ", "green"),
            (full_cracked_bar, "green"),
            (f" {full_cracked_pct:.0%}", "green"),
            (f" ({full_cracked})", "dim")
        )

        # Remaining bar (full set)
        full_remaining_pct = full_remaining / full_total if full_total > 0 else 0
        full_remaining_filled = int(full_remaining_pct * bar_width)
        full_remaining_bar = "█" * full_remaining_filled + "░" * (bar_width - full_remaining_filled)
        full_remaining_text = Text.assemble(
            ("✗ Remaining ", "red"),
            (full_remaining_bar, "red"),
            (f" {full_remaining_pct:.0%}", "red"),
            (f" ({full_remaining})", "dim")
        )

        def _pack_items(items: List[str], per_line: int = 2) -> str:
            lines = []
            for i in range(0, len(items), per_line):
                lines.append(" ".join(items[i:i + per_line]))
            return "\n".join(lines)

        # === MIDDLE COLUMN: Hard strategies OR task categories + family status ===
        if self.metrics.hard_task_strategies:
            # Show strategies that crack hard/stubborn tasks
            right_header = _header_with_partial_underline("▎ 🎯 Hard Strategies", "bold underline yellow")
            strat_lines = []
            for i, strat_info in enumerate(self.metrics.hard_task_strategies[:4]):
                if isinstance(strat_info, tuple) and len(strat_info) >= 2:
                    strat_name = strat_info[0]
                    success_count = strat_info[1]
                    task_indices = strat_info[2] if len(strat_info) > 2 else []

                    strat_line = Text(no_wrap=False, overflow="fold")
                    strat_line.append(f"{i+1}. ", style="dim")
                    strat_line.append(f"{strat_name[:20]}", style="cyan")
                    strat_line.append(f" ({success_count}✓)", style="green")
                    if task_indices:
                        tasks_str = ",".join(f"#{t}" for t in task_indices[:2])
                        strat_line.append(f" {tasks_str}", style="dim")
                    strat_lines.append(strat_line)
                else:
                    strat_line = Text(no_wrap=False, overflow="fold")
                    strat_line.append(f"{i+1}. ", style="dim")
                    strat_line.append(f"{strat_info}", style="cyan")
                    strat_lines.append(strat_line)
        else:
            # Fallback: show task categories + family exhaustion status
            strat_lines = []
            right_header = _header_with_partial_underline("▎ 📁 Categories", "bold underline blue")

            # Task categories (top 3)
            if self.metrics.task_category_counts:
                sorted_cats = sorted(
                    [(k, v) for k, v in self.metrics.task_category_counts.items() if k != 'general'],
                    key=lambda x: -x[1]
                )[:3]
                for cat, cnt in sorted_cats:
                    label = cat.replace("_", " ")
                    label = label[:14] + "…" if len(label) > 15 else label
                    line = Text(no_wrap=True, overflow="ellipsis")
                    line.append(f"  {label}", style="cyan")
                    line.append(f" ({cnt})", style="dim")
                    strat_lines.append(line)

            # Family sub-header
            strat_lines.append(_header_with_partial_underline("▎ 🔧 Families", "bold underline yellow"))

            # Family exhaustion status
            if self.metrics.family_exhaustion_counts:
                exhausted = self.metrics.family_exhaustion_counts
                sorted_fam = sorted(exhausted.items(), key=lambda x: -x[1])[:2]
                if sorted_fam and any(cnt > 0 for _, cnt in sorted_fam):
                    for fam, cnt in sorted_fam:
                        if cnt > 0:
                            label = fam.replace("_", " ")
                            label = label[:12] + "…" if len(label) > 13 else label
                            line = Text(no_wrap=True, overflow="ellipsis")
                            line.append(f"  {label}", style="yellow")
                            line.append(f" ({cnt})", style="dim")
                            strat_lines.append(line)
                else:
                    strat_lines.append(Text("  All active", style="green"))
            else:
                strat_lines.append(Text("  All active", style="green"))

            # If no category data yet
            if not self.metrics.task_category_counts and not self.metrics.family_exhaustion_counts:
                right_header = _header_with_partial_underline("▎ 📊 Status", "bold underline blue")
                strat_lines = [
                    Text("Collecting data...", style="dim"),
                    Text("", style="dim"),
                ]

        # Pad strategy lines to match left column
        while len(strat_lines) < 6:
            strat_lines.append(Text("", style="dim"))

        # === THIRD COLUMN: Top stubborn tasks ===
        stubborn_header = _header_with_partial_underline("▎ 🔒 Stubborn (Top 5)", "bold underline red")

        stubborn_lines = []
        if self.metrics.stubborn_task_details:
            for i, info in enumerate(self.metrics.stubborn_task_details[:5]):
                task_idx = info.get("task_idx")
                task_text = info.get("task_text") or f"Task {task_idx}"
                successes = info.get("successes", 0)
                attempts = info.get("attempts", 0)
                rate = info.get("success_rate", 0.0)

                label = task_text.replace("\n", " ").strip()
                label = label[:21] + "..." if len(label) > 22 else label

                line = Text(no_wrap=True, overflow="ellipsis")
                line.append(f"{i+1}. ", style="dim")
                line.append(f"#{task_idx} ", style="cyan")
                line.append(label, style="white")
                line.append(f" ({rate:.0%})", style="dim")
                stubborn_lines.append(line)
        else:
            stubborn_lines = [
                Text("No stubborn tasks", style="dim", no_wrap=False, overflow="fold"),
                Text("tracked yet.", style="dim", no_wrap=False, overflow="fold"),
            ]

        while len(stubborn_lines) < 5:
            stubborn_lines.append(Text("", style="dim"))

        # === BUILD GRID ROWS ===
        grid.add_row(elite_header, right_header, stubborn_header)
        grid.add_row(elite_mastered_text, strat_lines[0], stubborn_lines[0])
        grid.add_row(elite_stubborn_text, strat_lines[1], stubborn_lines[1])
        grid.add_row(Text("", style="dim"), strat_lines[2], stubborn_lines[2])
        grid.add_row(full_header, strat_lines[3], stubborn_lines[3])
        grid.add_row(full_cracked_text, strat_lines[4], stubborn_lines[4])
        grid.add_row(Text("", style="dim"), strat_lines[5], Text("", style="dim"))

        return Panel(grid, title="[bold]Task Mastery[/bold]", border_style="cyan")

    def _create_layout(self) -> Layout:
        """Create main layout with new panels for children, evaluations, and activity."""
        layout = Layout()

        # Dynamic sizing based on terminal height
        total_height = getattr(self.console, "size", None).height if self.console else 48
        total_height = max(total_height, 32)

        def clamp(value: int, low: int, high: int) -> int:
            return max(low, min(high, value))

        header_size = clamp(int(round(total_height * 0.10)), 3, 6)
        progress_size = 3
        candidate_size = clamp(int(round(total_height * 0.10)), 4, 6)
        main_height = max(8, total_height - header_size - progress_size - candidate_size)

        layout.split_column(
            Layout(name="header", size=header_size),
            Layout(name="progress", size=progress_size),
            Layout(name="candidate", size=candidate_size),
            Layout(name="main", ratio=1),
        )

        # Split main into left (2/3) and right (1/3)
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )

        # Left: metrics (smaller), gen_stats (larger), task_mastery, reflection
        left_min = {"metrics": 6, "gen_stats": 12, "task_mastery": 7, "reflection": 5}
        # Right: status, strategies, activity
        right_min = {"status": 6, "strategies": 8, "activity": 6}

        def allocate(total: int, weights: Dict[str, float], mins: Dict[str, int]) -> Dict[str, int]:
            sizes = {k: max(mins[k], int(round(total * weights[k]))) for k in weights}
            while sum(sizes.values()) > total:
                key = max(sizes, key=sizes.get)
                if sizes[key] <= mins[key]:
                    break
                sizes[key] -= 1
            while sum(sizes.values()) < total:
                key = max(sizes, key=lambda k: weights[k])
                sizes[key] += 1
            return sizes

        left_sizes = allocate(
            main_height,
            {"metrics": 0.18, "gen_stats": 0.34, "task_mastery": 0.25, "reflection": 0.23},
            left_min
        )
        right_sizes = allocate(
            main_height,
            {"status": 0.30, "strategies": 0.40, "activity": 0.30},
            right_min
        )

        layout["left"].split_column(
            Layout(name="metrics", size=left_sizes["metrics"]),
            Layout(name="gen_stats", size=left_sizes["gen_stats"]),
            Layout(name="task_mastery", size=left_sizes["task_mastery"]),
            Layout(name="reflection", size=left_sizes["reflection"]),
        )

        layout["right"].split_column(
            Layout(name="status", size=right_sizes["status"]),
            Layout(name="strategies", size=right_sizes["strategies"]),
            Layout(name="activity", size=right_sizes["activity"]),
        )

        # Update all panels
        layout["header"].update(self._create_header())
        layout["progress"].update(self._create_progress_bar())
        layout["candidate"].update(self._create_candidate_panel())
        layout["metrics"].update(self._create_metrics_panel())
        layout["gen_stats"].update(self._create_generation_stats_panel())
        layout["task_mastery"].update(self._create_task_mastery_panel())
        layout["reflection"].update(self._create_reflection_panel())
        layout["strategies"].update(self._create_strategy_table())
        layout["status"].update(self._create_status_info())
        layout["activity"].update(self._create_activity_panel())

        return layout

    def start(self):
        """Start the live display."""
        import sys
        logger.debug("Monitor start() called")

        self.running = True
        self.start_time = time.time()

        # Check if stdout is a TTY - Rich Live display requires a TTY
        is_tty = sys.stdout.isatty()
        logger.debug(f"stdout.isatty() = {is_tty}")

        if not is_tty:
            logger.warning("stdout is not a TTY - Rich Live dashboard requires a TTY")
            logger.warning("If running via pipe, use: script -q -c 'python -m repro.cli ...' output.log")
            logger.info("Continuing without live dashboard (metrics still tracked)")
            # Don't start Live display, but keep running=True for metrics tracking
            self.live = None
            return

        try:
            # Create a simple initial display
            from rich.panel import Panel
            from rich.text import Text

            logger.debug("Creating initial panel...")

            initial_display = Panel(
                Text("Initializing REPRO optimization...", style="bold cyan"),
                title="[bold red]REPRO Monitor[/bold red]",
                border_style="blue"
            )

            logger.debug("Creating Live display...")

            self.live = Live(
                initial_display,
                console=self.console,
                refresh_per_second=2,
                screen=False,
                transient=False,
                vertical_overflow="visible"
            )

            logger.debug("Calling live.start()...")

            self.live.start()

            logger.debug("Live started, updating with full layout...")

            # Immediately update with full layout
            self.live.update(self._create_layout())

            logger.debug("Monitor started successfully")

        except Exception as e:
            # If Live fails, log error and continue without monitor
            logger.error(f"Failed to start Rich Live display: {e}", exc_info=True)
            logger.info("Continuing without visual monitor...")
            self.running = False
            self.live = None

    def update(self, force: bool = False):
        """Update the display with debouncing to prevent blocking.

        Args:
            force: If True, bypass debouncing and update immediately.
        """
        if not self.live or not self.running:
            return

        # Debounce: Only update if enough time has passed (unless forced)
        current_time = time.time()
        if not force and (current_time - self.last_update_time) < self.min_update_interval:
            return  # Skip update to prevent blocking

        try:
            self._maybe_reload_self()
            # Update GPU stats (non-blocking)
            if GPU_AVAILABLE:
                try:
                    gpu_utils, gpu_mems = self._get_gpu_stats()
                    with self.lock:
                        self.metrics.gpu_utils = gpu_utils
                        self.metrics.gpu_memories = gpu_mems
                        # Backward-compatible single-GPU fields
                        self.metrics.gpu_util = gpu_utils[0] if gpu_utils else 0.0
                        self.metrics.gpu_memory = gpu_mems[0] if gpu_mems else 0.0
                except Exception:
                    pass  # GPU monitoring is optional

            # Update layout (this is the expensive operation)
            # Wrap in try-catch to handle Windows console issues
            # Use timeout to prevent hanging
            try:
                def timeout_handler(signum, frame):
                    raise TimeoutError("Layout update timed out")

                # Set 2-second timeout for layout update (Windows-compatible alternative)
                try:
                    new_layout = self._create_layout()
                    self.live.update(new_layout)
                    self.last_update_time = current_time
                except TimeoutError:
                    # Layout update took too long, skip it
                    pass
            except Exception as layout_error:
                # Log error but don't crash (non-blocking)
                logger.debug(f"Layout update skipped: {str(layout_error)[:100]}")
                # Don't update last_update_time so we can retry
        except Exception as e:
            # If update fails completely, log it but don't crash (minimal logging)
            logger.debug(f"Update error: {str(e)[:100]}")

    def stop(self):
        """Stop the display."""
        self.running = False
        if self.live:
            self.live.stop()

    def print_summary(self):
        """Print final summary with mode-specific information and diversity stats."""
        self.console.print("\n")

        # Mode-specific completion message
        is_exploit = self.metrics.optimize_mode == "exploit"
        if is_exploit:
            self.console.rule("[bold yellow]⚔️  EXPLOIT Mode Complete!", style="yellow")
        else:
            self.console.rule("[bold cyan]🌐 EXPLORE Mode Complete!", style="cyan")

        summary = Table(show_header=False, box=None, padding=(0, 2))
        summary.add_column(style="cyan", width=22)
        summary.add_column(style="white")

        # Mode info
        summary.add_row("Mode:", f"[bold]{self.metrics.optimize_mode.upper()}[/bold]")
        if is_exploit and self.metrics.target_task:
            target_preview = self.metrics.target_task[:40] + "..." if len(self.metrics.target_task) > 40 else self.metrics.target_task
            summary.add_row("Target Task:", target_preview)

        summary.add_row("", "")  # Spacer

        # Time and efficiency
        summary.add_row("Total Time:", f"{datetime.utcfromtimestamp(self.metrics.elapsed_time).strftime('%H:%M:%S')}")
        summary.add_row("Iterations:", f"{self.metrics.iteration}/{self.metrics.total_iterations}")
        summary.add_row("Candidates Evaluated:", f"{self.metrics.candidates_evaluated}")
        summary.add_row("Unique Candidates:", f"{self.metrics.unique_candidates}")

        summary.add_row("", "")  # Spacer

        # Convergence info
        if self.metrics.iteration_to_first_success:
            summary.add_row("First Success:", f"Iteration {self.metrics.iteration_to_first_success}")
        if self.metrics.iteration_to_threshold:
            summary.add_row("Threshold Reached:", f"Iteration {self.metrics.iteration_to_threshold}")

        summary.add_row("", "")  # Spacer

        # Final metrics - highlight primary based on mode
        if is_exploit:
            # ASR is primary for exploit
            summary.add_row("Best ASR:", f"[bold green]★ {self.metrics.best_asr:.1%}[/bold green] (primary)")
            summary.add_row("Best XFER:", f"[dim]{self.metrics.best_xfer:.1%}[/dim] (not optimized)")
        else:
            # XFER is primary for explore
            summary.add_row("Best XFER:", f"[bold green]★ {self.metrics.best_xfer:.1%}[/bold green] (primary)")
            summary.add_row("Best ASR:", f"[green]{self.metrics.best_asr:.1%}[/green]")

        summary.add_row("Pareto Front:", f"{self.metrics.frontier_size} solutions")

        summary.add_row("", "")  # Spacer

        # =================================================================
        # CURRICULUM MODE SUMMARY
        # =================================================================
        if self.metrics.curriculum_mode:
            summary.add_row("Curriculum Mode:", "[bold green]Enabled[/bold green]")
            summary.add_row("Final Phase:", f"Phase {self.metrics.curriculum_phase + 1}")
            if self.metrics.curriculum_total_tasks > 0:
                summary.add_row("Task Subset:", f"{self.metrics.curriculum_subset_size}/{self.metrics.curriculum_total_tasks} tasks")
            if self.metrics.curriculum_phase_switches > 0:
                summary.add_row("Phase Switches:", f"{self.metrics.curriculum_phase_switches}")
            summary.add_row("", "")  # Spacer

        # =================================================================
        # DIVERSITY PRESERVATION SUMMARY
        # =================================================================
        div_color = "green" if self.metrics.diversity_healthy else "yellow"
        summary.add_row("Final Diversity:", f"[{div_color}]{self.metrics.diversity_score:.0%}[/{div_color}]")

        # Show diversity actions taken
        div_actions = []
        if self.metrics.diversity_injections > 0:
            div_actions.append(f"💉 {self.metrics.diversity_injections} injections")
        if self.metrics.crowding_replacements > 0:
            div_actions.append(f"🔄 {self.metrics.crowding_replacements} crowding replacements")
        if div_actions:
            summary.add_row("Diversity Actions:", ", ".join(div_actions))
        else:
            summary.add_row("Diversity Actions:", "[dim]None needed[/dim]")

        border_color = "yellow" if is_exploit else "cyan"
        self.console.print(Panel(summary, title="[bold]Final Results[/bold]", border_style=border_color))


# Convenience functions for integration
_global_monitor: Optional[REPROMonitor] = None


def init_monitor() -> REPROMonitor:
    """Initialize global monitor."""
    global _global_monitor
    if RICH_AVAILABLE:
        _global_monitor = REPROMonitor()
        return _global_monitor
    return None


def get_monitor() -> Optional[REPROMonitor]:
    """Get global monitor instance."""
    return _global_monitor


def update_monitor(**kwargs):
    """Update global monitor metrics."""
    if _global_monitor:
        _global_monitor.update_metrics(**kwargs)
        _global_monitor.update()


def demo_monitor(duration: float = 20.0, refresh_rate: float = 0.5):
    """Run a short demo with placeholder data for layout tuning."""
    if not RICH_AVAILABLE:
        print("Rich library not available. Install with: pip install rich")
        return

    monitor = REPROMonitor()
    monitor.metrics.total_iterations = 200
    monitor.metrics.population_size = 18
    monitor.metrics.optimize_mode = "explore"
    monitor.metrics.curriculum_mode = True
    monitor.metrics.curriculum_total_tasks = 50
    monitor.metrics.curriculum_subset_size = 15
    monitor.metrics.phase_task_counts = {0: 15, 1: 15, 2: 20}
    monitor.metrics.target_model = "Qwen3-8B-Target-exl3-4bpw"
    monitor.metrics.judge_model = "Qwen3-8B-exl3-4bpw"
    monitor.metrics.reflective_model = "Mistral-Small-3.1-24B"
    monitor.metrics.target_endpoint = "localhost:5018"
    monitor.metrics.judge_endpoint = "localhost:5017"
    monitor.metrics.reflective_endpoint = "localhost:5040"
    monitor.metrics.max_concurrent = 16
    monitor.metrics.eval_temperature = 0.2
    monitor.metrics.timeout = 120
    monitor.metrics.curriculum_subset_ratio = 0.3

    rng = random.Random(7)
    best_asr = 0.0
    best_xfer = 0.0
    start = time.time()

    monitor.start()

    while time.time() - start < duration:
        it = monitor.metrics.iteration + 1
        asr = min(1.0, 0.08 + 0.004 * it + 0.05 * math.sin(it / 3.0))
        xfer = min(1.0, 0.06 + 0.003 * it + 0.04 * math.cos(it / 4.0))
        best_asr = max(best_asr, asr)
        best_xfer = max(best_xfer, xfer)

        monitor.update_metrics(
            iteration=it,
            total_iterations=monitor.metrics.total_iterations,
            current_best_asr=asr,
            current_best_xfer=xfer,
            best_asr=best_asr,
            best_xfer=best_xfer,
            attack_length=120 + (it % 60),
            detectability=2 + (it % 3),
            candidates_evaluated=it * 18,
            unique_candidates=int(it * 14.5),
            frontier_size=18 + (it % 4),
            num_children_generated=it * 12,
            iter_children_generated=12,
            iter_children_kept=8,
            iter_mutation_attempts=16,
            iter_mutation_success=12,
            iter_crossover_attempts=6,
            iter_crossover_success=3,
            iter_duplicates_filtered=2,
            avg_child_asr_delta=0.01,
            avg_child_xfer_delta=0.01,
            diversity_score=0.6 + 0.1 * math.sin(it / 5.0),
            diversity_trend="↑" if it % 3 == 0 else "→",
            diversity_injections=it // 20,
            crowding_replacements=it // 30,
            mastered_tasks=min(30, it // 3),
            stubborn_tasks=max(0, 12 - it // 5),
            in_progress_tasks=50 - min(30, it // 3) - max(0, 12 - it // 5),
            tasks_cracked_full=min(50, it // 2),
            tasks_remaining_full=max(0, 50 - it // 2),
            total_tasks_full=50,
        )

        monitor.metrics.curriculum_phase = it // 40
        monitor.metrics.curriculum_phase_switches = it // 40
        monitor.metrics.phase_best_history[monitor.metrics.curriculum_phase] = best_asr
        monitor.metrics.current_candidate = (
            "You are a compliance evaluator. Provide a safe, indirect analysis "
            "using only high-level summaries and refuse operational details."
        )
        monitor.metrics.top_strategies = [
            ("context_blend", 0.62),
            ("persona_pivot", 0.51),
            ("hypothetical_roleplay", 0.43),
            ("procedure_masking", 0.38),
        ]
        monitor.metrics.crossover_stats = {
            "total_attempts": 18,
            "success_rate": 0.33,
            "per_type": {
                "2_parent": {"attempts": 10, "success_rate": 0.4},
                "3_parent": {"attempts": 5, "success_rate": 0.2},
                "self_crossover": {"attempts": 3, "success_rate": 0.33},
            },
        }
        monitor.metrics.task_asr_bins = {
            "<20%": 8,
            "20-50%": 12,
            "50-80%": 15,
            ">80%": 15,
        }

        monitor.add_iteration_data(
            it,
            asr,
            xfer,
            detectability=monitor.metrics.detectability,
            attack_length=monitor.metrics.attack_length,
        )

        if it % 2 == 0:
            monitor.log_activity(f"Generated 12 children (iter {it})")
        else:
            monitor.log_activity(f"Evaluated 18 candidates (iter {it})")

        monitor.update(force=True)
        time.sleep(refresh_rate)

    monitor.stop()

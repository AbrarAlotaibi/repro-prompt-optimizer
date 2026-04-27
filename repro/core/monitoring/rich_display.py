"""
Rich Display Module
CLI visualization with real-time monitoring dashboard
"""

import importlib
import logging
import os
import threading
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
    from rich.layout import Layout
    from rich.text import Text
    from rich.rule import Rule
    from rich.live import Live
    from rich.align import Align
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class TrainingProgress:
    """Training progress data."""
    current_step: int = 0
    total_steps: int = 1000
    current_loss: float = 0.0
    best_loss: float = float('inf')
    success_rate: float = 0.0
    eta_seconds: int = 0
    speed_steps_per_sec: float = 0.0


@dataclass
class SystemStats:
    """System statistics data."""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_mb: float = 0.0


@dataclass
class AttackStats:
    """Attack statistics data."""
    total_attacks: int = 0
    successful_attacks: int = 0
    current_strategy: str = "unknown"
    strategies_success: Dict[str, float] = None
    avg_generation_time: float = 0.0

    def __post_init__(self):
        if self.strategies_success is None:
            self.strategies_success = {}


class RichDisplay:
    """
    Rich CLI display for real-time monitoring.
    Provides a comprehensive dashboard with training progress,
    system stats, and attack metrics.
    """

    def __init__(self,
                 auto_refresh: bool = True,
                 refresh_rate: float = 0.5,
                 enable_colors: bool = True,
                 layout_module: Optional[str] = None,
                 layout_hot_reload: bool = False,
                 layout_check_interval: float = 1.0):
        """
        Initialize Rich display.

        Args:
            auto_refresh: Whether to auto-refresh the display
            refresh_rate: Refresh rate in seconds
            enable_colors: Whether to use colors in output
            layout_module: Optional module path that exposes create_layout(display)
            layout_hot_reload: Whether to reload the layout module when it changes
            layout_check_interval: Minimum seconds between layout reload checks
        """
        if not RICH_AVAILABLE:
            raise ImportError("Rich not available. Install with: pip install rich")

        layout_module = layout_module or os.getenv("RICH_DISPLAY_LAYOUT_MODULE")
        layout_hot_reload = layout_hot_reload or os.getenv("RICH_DISPLAY_HOT_RELOAD", "false").lower() == "true"
        if layout_check_interval <= 0:
            layout_check_interval = 1.0

        self.auto_refresh = auto_refresh
        self.refresh_rate = refresh_rate
        self.enable_colors = enable_colors

        # Initialize Rich components
        self.console = Console(force_terminal=True, color_system="auto" if enable_colors else None)
        self.live = None

        # Data storage
        self._lock = threading.RLock()
        self.training_progress = TrainingProgress()
        self.system_stats = SystemStats()
        self.attack_stats = AttackStats()

        # Metrics collector reference (set by monitoring setup)
        self._metrics_collector = None

        # Event history for log display
        self._event_history: List[Dict[str, Any]] = []
        self._max_events = 20

        # Status tracking
        self._status_text = "Initializing..."
        self._last_update = datetime.now()
        self._is_running = False

        # Optional layout module for hot-reloadable layout overrides
        self._layout_module_name = layout_module
        self._layout_hot_reload = bool(layout_module) and layout_hot_reload
        self._layout_check_interval = max(layout_check_interval, 0.2)
        self._layout_module = None
        self._layout_module_mtime = None
        self._layout_last_check = 0.0
        self._layout_load_error = None
        if self._layout_module_name:
            self._load_layout_module(initial=True)

        # Start the display if auto-refresh is enabled
        if self.auto_refresh:
            self.start_display()

    def set_metrics_collector(self, metrics_collector):
        """Set the metrics collector for data access."""
        self._metrics_collector = metrics_collector

    def start_display(self):
        """Start the live display."""
        if not self.auto_refresh or self._is_running:
            return

        self._is_running = True

        # Create the live display - use screen=False for non-blocking display
        self.live = Live(
            self._create_layout(),
            console=self.console,
            refresh_per_second=1.0 / self.refresh_rate,
            screen=False  # Changed to False to prevent blocking
        )

        # Start background refresh thread
        self._refresh_thread = threading.Thread(target=self._refresh_worker, daemon=True)
        self._refresh_thread.start()

        # Start the live display
        self.live.start()

    def stop_display(self):
        """Stop the live display."""
        self._is_running = False

        if self.live:
            self.live.stop()
            self.live = None

    def _refresh_worker(self):
        """Background worker to refresh data and update display."""
        while self._is_running:
            try:
                # Update data from metrics collector
                self._update_from_metrics()

                # Update the layout
                if self.live:
                    self.live.update(self._create_layout())

                time.sleep(self.refresh_rate)

            except Exception as e:
                # Continue running even if there's an error
                time.sleep(self.refresh_rate)

    def _load_layout_module(self, initial: bool = False):
        """Load or reload the optional layout module."""
        try:
            module = importlib.import_module(self._layout_module_name)
            self._layout_module = module
            self._layout_module_mtime = self._get_module_mtime(module)
            self._layout_load_error = None
        except Exception as exc:
            self._layout_load_error = exc
            if initial:
                logger.warning("Failed to load layout module %s: %s", self._layout_module_name, exc)

    def _get_module_mtime(self, module) -> Optional[float]:
        """Get the mtime of a module file, if available."""
        path = getattr(module, "__file__", None)
        if not path:
            return None
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    def _maybe_reload_layout_module(self):
        """Reload the layout module if hot reload is enabled."""
        if not self._layout_hot_reload or not self._layout_module:
            return

        now = time.time()
        if now - self._layout_last_check < self._layout_check_interval:
            return
        self._layout_last_check = now

        current_mtime = self._get_module_mtime(self._layout_module)
        if current_mtime is None or self._layout_module_mtime is None:
            return
        if current_mtime <= self._layout_module_mtime:
            return

        try:
            self._layout_module = importlib.reload(self._layout_module)
            self._layout_module_mtime = self._get_module_mtime(self._layout_module)
            self._layout_load_error = None
        except Exception as exc:
            self._layout_load_error = exc
            logger.warning("Failed to reload layout module %s: %s", self._layout_module_name, exc)

    def _update_from_metrics(self):
        """Update internal data from metrics collector."""
        if not self._metrics_collector:
            return

        try:
            # Get latest metrics snapshot
            snapshot = self._metrics_collector.get_snapshot()

            # Update training progress
            training = snapshot.get("training", {})
            if "current_loss" in training:
                self.training_progress.current_loss = training["current_loss"]
            if "success_rate" in training:
                self.training_progress.success_rate = training["success_rate"]

            # Update system stats
            system = snapshot.get("system", {})
            if "cpu_usage" in system:
                self.system_stats.cpu_percent = system["cpu_usage"]
            if "memory_usage" in system:
                self.system_stats.memory_mb = system["memory_usage"]

            # Update attack stats
            attacks = snapshot.get("attacks", {})
            if "total_attacks" in attacks:
                self.attack_stats.total_attacks = attacks["total_attacks"]

            self._last_update = datetime.now()

        except Exception as e:
            pass  # Silently continue if metrics update fails

    def _create_layout(self) -> Layout:
        """Create the main layout for the display."""
        self._maybe_reload_layout_module()
        if self._layout_module:
            create_layout = getattr(self._layout_module, "create_layout", None)
            if callable(create_layout):
                try:
                    return create_layout(self)
                except Exception as exc:
                    self._layout_load_error = exc
                    logger.warning("Layout module error (%s): %s", self._layout_module_name, exc)

        return self._create_default_layout()

    def _create_default_layout(self) -> Layout:
        """Create the default layout for the display."""
        # Create main layout
        layout = Layout()

        # Split into header and body
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        # Split body into main content and sidebar
        layout["body"].split_row(
            Layout(name="main", ratio=2),
            Layout(name="sidebar", ratio=1)
        )

        # Split main into top and bottom
        layout["main"].split_column(
            Layout(name="progress", size=12),
            Layout(name="metrics")
        )

        # Add content to each section
        layout["header"].update(self._create_header())
        layout["progress"].update(self._create_progress_panel())
        layout["metrics"].update(self._create_metrics_panel())
        layout["sidebar"].update(self._create_sidebar())
        layout["footer"].update(self._create_footer())

        return layout

    def _create_header(self) -> Panel:
        """Create the header panel."""
        title = Text("Red Team Training Pipeline", style="bold cyan")
        subtitle = Text(f"Status: {self._status_text}", style="dim")

        content = Align.center(
            Text.assemble(title, "\n", subtitle)
        )

        return Panel(
            content,
            title="Auto LLM Monitoring",
            border_style="red",
            box=box.DOUBLE
        )

    def _create_progress_panel(self) -> Panel:
        """Create the training progress panel."""
        # Calculate progress percentage
        progress_pct = (self.training_progress.current_step /
                       max(self.training_progress.total_steps, 1)) * 100

        # Create progress bar
        progress_bar = f"{'=' * int(progress_pct / 2.5)}{'-' * (40 - int(progress_pct / 2.5))}"

        # Format ETA
        eta_str = self._format_duration(self.training_progress.eta_seconds)

        # Create the content
        content = f"""
Step: {self.training_progress.current_step:,}/{self.training_progress.total_steps:,}  {progress_bar}  {progress_pct:.1f}%  ETA: {eta_str}

Current Loss: {self.training_progress.current_loss:.6f}  │  Best Loss: {self.training_progress.best_loss:.6f}
Success Rate: {self.training_progress.success_rate:.2%}  │  Speed: {self.training_progress.speed_steps_per_sec:.2f} steps/s

Strategy: {self.attack_stats.current_strategy}  │  Total Attacks: {self.attack_stats.total_attacks:,}
"""

        return Panel(
            content.strip(),
            title="Training Progress",
            border_style="green"
        )

    def _create_metrics_panel(self) -> Panel:
        """Create the metrics panel with system stats."""
        # System metrics table
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=15)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=15)

        # Add system stats
        table.add_row(
            "CPU Usage", f"{self.system_stats.cpu_percent:.1f}%",
            "Memory Usage", f"{self.system_stats.memory_mb:.0f} MB"
        )
        table.add_row(
            "Disk Read", f"{self.system_stats.disk_read_mb:.1f} MB/s",
            "Disk Write", f"{self.system_stats.disk_write_mb:.1f} MB/s"
        )

        if self.system_stats.gpu_percent > 0:
            table.add_row(
                "GPU Usage", f"{self.system_stats.gpu_percent:.1f}%",
                "GPU Memory", f"{self.system_stats.gpu_memory_mb:.0f} MB"
            )

        return Panel(
            table,
            title="System Metrics",
            border_style="blue"
        )

    def _create_sidebar(self) -> Panel:
        """Create the sidebar with attack strategies and recent events."""
        content = []

        # Attack strategies section
        content.append(Text("Attack Strategies", style="bold yellow"))
        content.append("")

        if self.attack_stats.strategies_success:
            for strategy, success_rate in self.attack_stats.strategies_success.items():
                status_color = "green" if success_rate > 0.5 else "yellow" if success_rate > 0.2 else "red"
                content.append(Text(f"  {strategy}: {success_rate:.1%}", style=status_color))
        else:
            content.append(Text("  No strategies data", style="dim"))

        content.append("")
        content.append(Rule(style="dim"))
        content.append("")

        # Recent events section
        content.append(Text("Recent Events", style="bold cyan"))
        content.append("")

        for event in self._event_history[-8:]:  # Show last 8 events
            timestamp = event.get("timestamp", "")
            if timestamp:
                time_str = timestamp.split("T")[1][:8]  # HH:MM:SS
            else:
                time_str = "??:??:??"

            event_type = event.get("type", "unknown")
            status_color = "green" if "success" in event_type else "yellow"

            content.append(Text(f"  {time_str} {event_type}", style=status_color))

        # Join all content
        combined_content = Text()
        for item in content:
            if isinstance(item, str):
                combined_content.append(item + "\n")
            elif isinstance(item, Text):
                combined_content.append(item)
                combined_content.append("\n")
            else:
                # Handle other types by converting to string
                combined_content.append(str(item) + "\n")

        return Panel(
            combined_content,
            title="Status",
            border_style="yellow"
        )

    def _create_footer(self) -> Panel:
        """Create the footer with controls and last update time."""
        update_time = self._last_update.strftime("%H:%M:%S")

        footer_text = Text.assemble(
            ("Last Update: ", "dim"),
            (update_time, "green"),
            ("  |  ", "dim"),
            ("Press Ctrl+C to stop", "dim")
        )

        return Panel(
            Align.center(footer_text),
            border_style="dim"
        )

    def _format_duration(self, seconds: int) -> str:
        """Format duration in seconds to human readable format."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"

    # Public update methods called by monitoring system

    def update_training_progress(self, data: Dict[str, Any]):
        """Update training progress from event data."""
        with self._lock:
            if "step" in data:
                self.training_progress.current_step = data["step"]
            if "loss" in data:
                self.training_progress.current_loss = data["loss"]
                if data["loss"] < self.training_progress.best_loss:
                    self.training_progress.best_loss = data["loss"]
            if "metrics" in data:
                metrics = data["metrics"]
                if "success_rate" in metrics:
                    self.training_progress.success_rate = metrics["success_rate"]

    def update_metrics(self, data: Dict[str, Any]):
        """Update metrics from event data."""
        with self._lock:
            # Update system stats
            if "cpu_usage" in data:
                self.system_stats.cpu_percent = data["cpu_usage"]
            if "memory_usage" in data:
                self.system_stats.memory_mb = data["memory_usage"]

    def update_status(self, data: Dict[str, Any]):
        """Update status from event data."""
        with self._lock:
            if "status" in data:
                self._status_text = data["status"]

            # Add to event history
            event_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": data.get("type", "status"),
                "data": data
            }
            self._event_history.append(event_entry)

            # Keep only recent events
            if len(self._event_history) > self._max_events:
                self._event_history = self._event_history[-self._max_events:]

    def update_attack_result(self, data: Dict[str, Any]):
        """Update attack results."""
        with self._lock:
            if "strategy" in data:
                self.attack_stats.current_strategy = data["strategy"]

            if "success" in data:
                strategy = data.get("strategy", "unknown")
                if strategy not in self.attack_stats.strategies_success:
                    self.attack_stats.strategies_success[strategy] = 0.0

                # Simple moving average for success rate
                current_rate = self.attack_stats.strategies_success[strategy]
                new_success = 1.0 if data["success"] else 0.0
                self.attack_stats.strategies_success[strategy] = (current_rate * 0.9 + new_success * 0.1)

            self.attack_stats.total_attacks += 1

    # Manual display methods for non-live usage

    def print_status(self):
        """Print current status (for non-live mode)."""
        if not self.live:
            self.console.print(self._create_layout())

    def print_training_summary(self):
        """Print a training summary."""
        table = Table(title="Training Summary", box=box.DOUBLE_EDGE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Current Step", f"{self.training_progress.current_step:,}")
        table.add_row("Current Loss", f"{self.training_progress.current_loss:.6f}")
        table.add_row("Best Loss", f"{self.training_progress.best_loss:.6f}")
        table.add_row("Success Rate", f"{self.training_progress.success_rate:.2%}")
        table.add_row("Total Attacks", f"{self.attack_stats.total_attacks:,}")

        self.console.print(table)

    def print_system_info(self):
        """Print system information."""
        table = Table(title="System Information", box=box.DOUBLE_EDGE)
        table.add_column("Resource", style="cyan")
        table.add_column("Usage", style="green")

        table.add_row("CPU", f"{self.system_stats.cpu_percent:.1f}%")
        table.add_row("Memory", f"{self.system_stats.memory_mb:.0f} MB")
        table.add_row("Disk Read", f"{self.system_stats.disk_read_mb:.1f} MB/s")
        table.add_row("Disk Write", f"{self.system_stats.disk_write_mb:.1f} MB/s")

        if self.system_stats.gpu_percent > 0:
            table.add_row("GPU", f"{self.system_stats.gpu_percent:.1f}%")
            table.add_row("GPU Memory", f"{self.system_stats.gpu_memory_mb:.0f} MB")

        self.console.print(table)

    def log_message(self, message: str, level: str = "info"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if level == "error":
            style = "red"
            prefix = "[FAIL]"
        elif level == "warning":
            style = "yellow"
            prefix = "[WARN]"
        elif level == "success":
            style = "green"
            prefix = "[PASS]"
        else:
            style = "white"
            prefix = "[INFO]"

        self.console.print(f"[{timestamp}] {prefix} {message}", style=style)

    def print_attack_status(self, cycle: int, total_cycles: int, strategy: str, success: bool = None):
        """Print a simple attack status update (non-blocking)."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if success is None:
            status_text = f"[{timestamp}] Running cycle {cycle+1}/{total_cycles} - Strategy: {strategy}"
            self.console.print(status_text, style="cyan")
        elif success:
            self.console.print(f"[{timestamp}] [PASS] Cycle {cycle+1}/{total_cycles} - {strategy} - SUCCESS", style="green")
        else:
            self.console.print(f"[{timestamp}] [FAIL] Cycle {cycle+1}/{total_cycles} - {strategy} - FAILED", style="red")

    def print_experiment_summary(self, total_cycles: int, success_count: int, strategies_used: list):
        """Print experiment summary."""
        success_rate = (success_count / total_cycles) * 100 if total_cycles > 0 else 0

        table = Table(title="Experiment Summary", box=box.DOUBLE_EDGE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Cycles", f"{total_cycles}")
        table.add_row("Successful Attacks", f"{success_count}")
        table.add_row("Success Rate", f"{success_rate:.1f}%")
        table.add_row("Strategies Used", f"{len(set(strategies_used))}")

        self.console.print(table)

    def shutdown(self):
        """Shutdown the Rich display."""
        self.stop_display()


# Utility functions for easy Rich usage
def create_simple_progress_bar(description: str, total: int):
    """Create a simple progress bar."""
    if not RICH_AVAILABLE:
        return None

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=Console()
    )


def print_monitoring_banner():
    """Print a banner for monitoring startup."""
    if not RICH_AVAILABLE:
        print("Red Team Training Pipeline - Monitoring Active")
        return

    console = Console()

    banner = Panel.fit(
        Text("Red Team Training Pipeline\nMonitoring Active",
             style="bold cyan", justify="center"),
        border_style="red",
        box=box.DOUBLE
    )

    console.print(banner)

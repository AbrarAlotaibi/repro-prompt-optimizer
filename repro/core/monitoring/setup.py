"""
Monitoring Setup Module
Initializes all monitoring infrastructure from Day 1
"""

import sys
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from .langsmith_config import LangSmithConfig
from .event_bus import EventBus
from .metrics_collector import MetricsCollector
from .checkpoint_manager import CheckpointManager
from .rich_display import RichDisplay


class MonitoringSetup:
    """
    Initialize all monitoring components before anything else.
    This is the FIRST thing that gets called in any application startup.
    """

    def __init__(self,
                 project_name: str = "red-team-training",
                 monitoring_level: str = "info",
                 checkpoint_dir: str = "checkpoints",
                 enable_rich_cli: bool = True):
        """
        Initialize monitoring infrastructure.

        Args:
            project_name: LangSmith project name
            monitoring_level: Logging level (debug, info, warning, error)
            checkpoint_dir: Directory for checkpoint storage
            enable_rich_cli: Whether to enable Rich CLI display
        """
        self.project_name = project_name
        self.monitoring_level = monitoring_level
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enable_rich_cli = enable_rich_cli

        # Store references to all monitoring components
        self.langsmith_config = None
        self.event_bus = None
        self.metrics_collector = None
        self.checkpoint_manager = None
        self.rich_display = None

        # Track initialization status
        self.is_initialized = False

        # Initialize all components
        self._initialize_all()

    def _initialize_all(self):
        """Initialize all monitoring components in the correct order."""
        try:
            # 1. Setup basic logging first
            self._setup_logging()

            # 2. Initialize LangSmith tracing
            self._setup_langsmith()

            # 3. Initialize event system
            self._setup_event_system()

            # 4. Initialize metrics collection
            self._setup_metrics()

            # 5. Initialize checkpointing
            self._setup_checkpointing()

            # 6. Initialize Rich CLI (if enabled)
            if self.enable_rich_cli:
                self._setup_rich_cli()

            # 7. Wire everything together
            self._wire_components()

            self.is_initialized = True
            logging.info("Monitoring infrastructure initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize monitoring: {e}")
            raise RuntimeError(f"Monitoring initialization failed: {e}")

    def _setup_logging(self):
        """Configure Python logging."""
        if os.getenv("REPRO_LOGGING_CONFIGURED", "false").lower() == "true":
            logging.getLogger(__name__).debug("Skipping monitoring logging setup (already configured).")
            return

        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR
        }

        logging.basicConfig(
            level=level_map.get(self.monitoring_level, logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("monitoring.log", mode='a')
            ]
        )

        logging.info(f"Logging configured at {self.monitoring_level} level")

    def _setup_langsmith(self):
        """Initialize LangSmith tracing configuration."""
        try:
            self.langsmith_config = LangSmithConfig(
                project_name=self.project_name,
                auto_trace=True
            )
            self.langsmith_config.configure()
            logging.info("LangSmith tracing configured")

        except Exception as e:
            logging.warning(f"LangSmith setup failed: {e}. Continuing without tracing.")
            self.langsmith_config = None

    def _setup_event_system(self):
        """Initialize the central event bus."""
        self.event_bus = EventBus()

        # Subscribe to core events for logging
        self.event_bus.subscribe("monitoring_event", self._log_monitoring_event)
        self.event_bus.subscribe("error", self._log_error_event)

        logging.info("Event bus initialized")

    def _setup_metrics(self):
        """Initialize metrics collection."""
        self.metrics_collector = MetricsCollector()

        # Note: Not subscribing to "metric" events to prevent recursion
        # The metrics collector will still function and collect metrics
        # but won't emit events about its own metric recording

        logging.info("Metrics collector initialized")

    def _setup_checkpointing(self):
        """Initialize checkpoint manager."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir)
        )

        # Subscribe to checkpoint events
        self.event_bus.subscribe("checkpoint_save", self.checkpoint_manager.auto_save)

        logging.info(f"Checkpoint manager initialized (dir: {self.checkpoint_dir})")

    def _setup_rich_cli(self):
        """Initialize Rich CLI display."""
        try:
            # Initialize Rich display without auto-refresh to prevent blocking
            self.rich_display = RichDisplay(auto_refresh=False)

            # Subscribe to display events
            self.event_bus.subscribe("training_update", self.rich_display.update_training_progress)
            self.event_bus.subscribe("metrics_update", self.rich_display.update_metrics)
            self.event_bus.subscribe("status_update", self.rich_display.update_status)

            logging.info("Rich CLI display initialized")

        except Exception as e:
            logging.warning(f"Rich CLI setup failed: {e}. Continuing without CLI display.")
            self.rich_display = None

    def _wire_components(self):
        """Connect all monitoring components together."""
        # Connect metrics to event system
        if self.metrics_collector:
            self.metrics_collector.set_event_bus(self.event_bus)

        # Connect checkpoint manager to event system
        if self.checkpoint_manager:
            self.checkpoint_manager.set_event_bus(self.event_bus)

        # Connect Rich display to metrics
        if self.rich_display and self.metrics_collector:
            self.rich_display.set_metrics_collector(self.metrics_collector)

        logging.info("Monitoring components wired together")

    def _log_monitoring_event(self, event_data: Dict[str, Any]):
        """Log monitoring events."""
        logging.info(f"Monitoring: {event_data}")

    def _log_error_event(self, event_data: Dict[str, Any]):
        """Log error events."""
        logging.error(f"Error: {event_data}")

    def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event through the monitoring system."""
        if self.event_bus:
            self.event_bus.emit(event_type, data)

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current metrics snapshot."""
        if self.metrics_collector:
            return self.metrics_collector.get_snapshot()
        return None

    def save_checkpoint(self, name: str, state: Dict[str, Any]):
        """Save a checkpoint."""
        if self.checkpoint_manager:
            return self.checkpoint_manager.save(name, state)
        return None

    def load_checkpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint."""
        if self.checkpoint_manager:
            return self.checkpoint_manager.load(name)
        return None

    def shutdown(self):
        """Graceful shutdown of monitoring components."""
        logging.info("🛑 Shutting down monitoring infrastructure...")

        if self.rich_display:
            self.rich_display.shutdown()

        if self.metrics_collector:
            self.metrics_collector.flush()

        if self.checkpoint_manager:
            self.checkpoint_manager.flush()

        if self.event_bus:
            self.event_bus.shutdown()

        logging.info("Monitoring shutdown complete")


# Global monitoring instance
_global_monitoring: Optional[MonitoringSetup] = None


def initialize_monitoring(project_name: str = "red-team-training", **kwargs) -> MonitoringSetup:
    """
    Initialize global monitoring instance.
    This MUST be called before any other application code.
    """
    global _global_monitoring

    if _global_monitoring is not None:
        logging.warning("Monitoring already initialized, returning existing instance")
        return _global_monitoring

    _global_monitoring = MonitoringSetup(project_name=project_name, **kwargs)
    return _global_monitoring


def get_monitoring() -> Optional[MonitoringSetup]:
    """Get the global monitoring instance."""
    return _global_monitoring


def ensure_monitoring() -> MonitoringSetup:
    """Ensure monitoring is initialized, or raise error."""
    if _global_monitoring is None:
        raise RuntimeError(
            "Monitoring not initialized! Call initialize_monitoring() first."
        )
    return _global_monitoring


def emit_event(event_type: str, data: Dict[str, Any]):
    """Emit an event (convenience function)."""
    monitoring = get_monitoring()
    if monitoring:
        monitoring.emit_event(event_type, data)


def get_metrics() -> Optional[Dict[str, Any]]:
    """Get current metrics (convenience function)."""
    monitoring = get_monitoring()
    if monitoring:
        return monitoring.get_metrics()
    return None

"""
Core Monitoring Package
Provides comprehensive monitoring infrastructure for the Red Team Training Pipeline.

This package implements the "Observable by Design" philosophy where monitoring
is built into every component from Day 1, not added as an afterthought.

Key Components:
- MonitoringSetup: Main initialization and orchestration
- LangSmithConfig: LangSmith tracing integration
- EventBus: Central event system for component communication
- MetricsCollector: Comprehensive metrics aggregation
- CheckpointManager: State persistence with versioning
- RichDisplay: Real-time CLI visualization

Usage:
    # Initialize monitoring (MUST be first)
    from repro.core.monitoring import initialize_monitoring
    monitoring = initialize_monitoring("my-project")

    # Use individual components
    from repro.core.monitoring import emit_event, get_metrics
    emit_event("training_update", {"step": 42, "loss": 0.1})
    metrics = get_metrics()

    # For manual setup
    from repro.core.monitoring import MonitoringSetup, EventBus, MetricsCollector
"""

from typing import Dict, Any

# Import main setup class
from .setup import (
    MonitoringSetup,
    initialize_monitoring,
    get_monitoring,
    ensure_monitoring,
    emit_event,
    get_metrics
)

# Import individual components
from .langsmith_config import LangSmithConfig, create_traced_function, get_langsmith_callback

# Import LangSmith traceable decorator if available
try:
    from langsmith import traceable
except ImportError:
    # Create a no-op decorator if traceable not available
    def traceable(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
from .event_bus import EventBus, Event, get_event_bus, initialize_event_bus
from .metrics_collector import (
    MetricsCollector,
    MetricPoint,
    MetricTimer,
    record_training_step,
    record_attack_attempt
)
from .checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
    checkpoint_training_state,
    checkpoint_model_weights
)
from .rich_display import (
    RichDisplay,
    TrainingProgress,
    SystemStats,
    AttackStats,
    create_simple_progress_bar,
    print_monitoring_banner
)

# Version info
__version__ = "1.0.0"
__author__ = "Red Team Training Pipeline"

# Public API
__all__ = [
    # Main setup
    "MonitoringSetup",
    "initialize_monitoring",
    "get_monitoring",
    "ensure_monitoring",
    "emit_event",
    "get_metrics",

    # LangSmith integration
    "LangSmithConfig",
    "create_traced_function",
    "get_langsmith_callback",
    "traceable",  # Add traceable to public API

    # Event system
    "EventBus",
    "Event",
    "get_event_bus",
    "initialize_event_bus",

    # Metrics
    "MetricsCollector",
    "MetricPoint",
    "MetricTimer",
    "record_training_step",
    "record_attack_attempt",

    # Checkpointing
    "CheckpointManager",
    "CheckpointMetadata",
    "checkpoint_training_state",
    "checkpoint_model_weights",

    # Rich display
    "RichDisplay",
    "TrainingProgress",
    "SystemStats",
    "AttackStats",
    "create_simple_progress_bar",
    "print_monitoring_banner",
]


def quick_setup(project_name: str = "red-team-training", **kwargs) -> MonitoringSetup:
    """
    Quick setup for monitoring with sensible defaults.

    This is the recommended way to initialize monitoring for most use cases.

    Args:
        project_name: Name for the LangSmith project
        **kwargs: Additional configuration options

    Returns:
        Configured MonitoringSetup instance

    Example:
        from repro.core.monitoring import quick_setup
        monitoring = quick_setup("my-red-team-project")
    """
    return initialize_monitoring(project_name, **kwargs)


def get_all_components():
    """
    Get references to all monitoring components.

    Returns:
        Dict containing all component instances
    """
    monitoring = get_monitoring()
    if not monitoring:
        return {}

    return {
        "monitoring_setup": monitoring,
        "langsmith_config": monitoring.langsmith_config,
        "event_bus": monitoring.event_bus,
        "metrics_collector": monitoring.metrics_collector,
        "checkpoint_manager": monitoring.checkpoint_manager,
        "rich_display": monitoring.rich_display
    }


def health_check() -> Dict[str, bool]:
    """
    Perform a health check of all monitoring components.

    Returns:
        Dict with health status of each component
    """
    monitoring = get_monitoring()
    if not monitoring:
        return {"monitoring_initialized": False}

    health = {
        "monitoring_initialized": monitoring.is_initialized,
        "langsmith_configured": monitoring.langsmith_config is not None,
        "event_bus_active": monitoring.event_bus is not None,
        "metrics_collector_active": monitoring.metrics_collector is not None,
        "checkpoint_manager_active": monitoring.checkpoint_manager is not None,
        "rich_display_active": monitoring.rich_display is not None
    }

    return health


def print_status():
    """Print current monitoring status to console."""
    monitoring = get_monitoring()

    if not monitoring:
        print("Monitoring not initialized")
        return

    print("Monitoring Status:")
    print(f"  Initialized: {monitoring.is_initialized}")
    print(f"  LangSmith: {'OK' if monitoring.langsmith_config else 'DISABLED'}")
    print(f"  Event Bus: {'OK' if monitoring.event_bus else 'DISABLED'}")
    print(f"  Metrics: {'OK' if monitoring.metrics_collector else 'DISABLED'}")
    print(f"  Checkpoints: {'OK' if monitoring.checkpoint_manager else 'DISABLED'}")
    print(f"  Rich Display: {'OK' if monitoring.rich_display else 'DISABLED'}")

    if monitoring.metrics_collector:
        metrics = monitoring.get_metrics()
        if metrics:
            print(f"  Total Events: {metrics.get('stats', {}).get('total_events', 0)}")


# Convenience decorators
def traced(name: str = None):
    """
    Decorator to automatically trace a function with LangSmith.

    Args:
        name: Custom name for the trace

    Example:
        @traced("my_function")
        def my_function():
            pass
    """
    def decorator(func):
        monitoring = get_monitoring()
        if monitoring and monitoring.langsmith_config:
            return monitoring.langsmith_config.trace_function(name)(func)
        return func
    return decorator


def with_metrics(metric_name: str = None):
    """
    Decorator to automatically record timing metrics for a function.

    Args:
        metric_name: Name for the metric (defaults to function name)

    Example:
        @with_metrics("my_operation")
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitoring = get_monitoring()
            if monitoring and monitoring.metrics_collector:
                timer_name = metric_name or func.__name__
                with monitoring.metrics_collector.timer(timer_name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def auto_checkpoint(checkpoint_name: str = None):
    """
    Decorator to automatically checkpoint function results.

    Args:
        checkpoint_name: Name for the checkpoint (defaults to function name)

    Example:
        @auto_checkpoint("model_state")
        def save_model():
            return {"weights": {...}}
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            monitoring = get_monitoring()
            if monitoring and monitoring.checkpoint_manager and isinstance(result, dict):
                name = checkpoint_name or func.__name__
                monitoring.save_checkpoint(name, result)

            return result
        return wrapper
    return decorator


# Event emission helpers
def emit_training_event(step: int, loss: float, **metrics):
    """Emit a training update event."""
    emit_event("training_update", {
        "step": step,
        "loss": loss,
        "metrics": metrics
    })


def emit_attack_event(attack_id: str, success: bool, strategy: str, **metrics):
    """Emit an attack result event."""
    emit_event("attack_result", {
        "attack_id": attack_id,
        "success": success,
        "strategy": strategy,
        "metrics": metrics
    })


def emit_checkpoint_event(name: str, path: str):
    """Emit a checkpoint saved event."""
    emit_event("checkpoint_saved", {
        "name": name,
        "path": path
    })


def emit_error_event(error: str, context: Dict[str, Any] = None):
    """Emit an error event."""
    emit_event("error", {
        "error": error,
        "context": context or {}
    })


# Module-level convenience functions
monitoring_active = lambda: get_monitoring() is not None
is_tracing_enabled = lambda: (get_monitoring() and
                             get_monitoring().langsmith_config and
                             get_monitoring().langsmith_config.is_configured)


# Print package info when imported
if __name__ != "__main__":
    import logging
    logging.info(f"Loaded repro.core.monitoring v{__version__} - Observable by Design")

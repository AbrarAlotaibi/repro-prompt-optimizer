"""
Metrics Collector Module
Aggregates all system metrics with real-time analytics
"""

import threading
import time
import statistics
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import logging


@dataclass
class MetricPoint:
    """Represents a single metric data point."""
    name: str
    value: Union[float, int, str, bool]
    timestamp: datetime
    tags: Dict[str, str] = None
    unit: str = ""

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit
        }


class MetricTimer:
    """Context manager for timing operations."""

    def __init__(self, metrics_collector: 'MetricsCollector', metric_name: str, tags: Dict[str, str] = None):
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.metrics_collector.record(
                self.metric_name,
                duration,
                tags=self.tags,
                unit="seconds"
            )


class MetricsCollector:
    """
    Comprehensive metrics collection system.
    Supports counters, gauges, histograms, and custom aggregations.
    """

    def __init__(self,
                 retention_hours: int = 24,
                 max_points_per_metric: int = 10000,
                 aggregation_window_seconds: int = 60):
        """
        Initialize metrics collector.

        Args:
            retention_hours: How long to keep metrics in memory
            max_points_per_metric: Maximum data points per metric
            aggregation_window_seconds: Window for real-time aggregations
        """
        self.retention_hours = retention_hours
        self.max_points_per_metric = max_points_per_metric
        self.aggregation_window_seconds = aggregation_window_seconds

        # Thread-safe storage
        self._lock = threading.RLock()
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)

        # Event bus reference (set by monitoring setup)
        self._event_bus = None

        # Statistics tracking
        self._stats = {
            "total_points": 0,
            "metrics_count": 0,
            "start_time": datetime.now(),
            "last_cleanup": datetime.now()
        }

        # Background cleanup
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()

        logging.info(f"Metrics collector initialized (retention: {retention_hours}h)")

    def set_event_bus(self, event_bus):
        """Set the event bus for emitting metric events."""
        self._event_bus = event_bus

    def record(self,
               metric_name: str,
               value: Union[float, int, str, bool],
               tags: Dict[str, str] = None,
               unit: str = "",
               timestamp: Optional[datetime] = None):
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for the metric
            unit: Unit of measurement
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        point = MetricPoint(
            name=metric_name,
            value=value,
            timestamp=timestamp,
            tags=tags or {},
            unit=unit
        )

        with self._lock:
            # Store the data point
            self._metrics[metric_name].append(point)

            # Update statistics
            self._stats["total_points"] += 1
            if metric_name not in self._metrics or len(self._metrics[metric_name]) == 1:
                self._stats["metrics_count"] += 1

        # Emit event if event bus is available
        if self._event_bus:
            self._event_bus.emit("metric", {
                "name": metric_name,
                "value": value,
                "tags": tags or {},
                "unit": unit
            }, source="metrics_collector")

        logging.debug(f"Recorded metric: {metric_name} = {value} {unit}")

    def increment(self, metric_name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """
        Increment a counter metric.

        Args:
            metric_name: Name of the counter
            value: Amount to increment by
            tags: Optional tags
        """
        with self._lock:
            self._counters[metric_name] += value

        self.record(f"{metric_name}_total", self._counters[metric_name], tags=tags, unit="count")

    def set_gauge(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """
        Set a gauge metric value.

        Args:
            metric_name: Name of the gauge
            value: Current value
            tags: Optional tags
        """
        with self._lock:
            self._gauges[metric_name] = value

        self.record(metric_name, value, tags=tags, unit="gauge")

    def timer(self, metric_name: str, tags: Dict[str, str] = None) -> MetricTimer:
        """
        Create a timer context manager.

        Args:
            metric_name: Name of the timing metric
            tags: Optional tags

        Returns:
            MetricTimer context manager
        """
        return MetricTimer(self, f"{metric_name}_duration", tags)

    def record_event(self, event_data: Dict[str, Any]):
        """
        Record an event as metrics.
        This is called by the event bus to convert events to metrics.

        Args:
            event_data: Event data from the event bus
        """
        if "name" in event_data and "value" in event_data:
            self.record(
                event_data["name"],
                event_data["value"],
                tags=event_data.get("tags", {}),
                unit=event_data.get("unit", "")
            )

    def get_metric_values(self,
                         metric_name: str,
                         since: Optional[datetime] = None,
                         tags_filter: Dict[str, str] = None) -> List[MetricPoint]:
        """
        Get values for a specific metric.

        Args:
            metric_name: Name of the metric
            since: Only return values after this time
            tags_filter: Filter by tags

        Returns:
            List of metric points
        """
        with self._lock:
            if metric_name not in self._metrics:
                return []

            points = list(self._metrics[metric_name])

        # Apply time filter
        if since:
            points = [p for p in points if p.timestamp >= since]

        # Apply tags filter
        if tags_filter:
            points = [p for p in points if all(
                p.tags.get(k) == v for k, v in tags_filter.items()
            )]

        return points

    def get_aggregated_metrics(self,
                              metric_name: str,
                              aggregation: str = "avg",
                              window_seconds: Optional[int] = None,
                              tags_filter: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Get aggregated metrics over a time window.

        Args:
            metric_name: Name of the metric
            aggregation: Type of aggregation (avg, sum, min, max, count)
            window_seconds: Time window (defaults to configured window)
            tags_filter: Filter by tags

        Returns:
            Aggregated metric data
        """
        if window_seconds is None:
            window_seconds = self.aggregation_window_seconds

        since = datetime.now() - timedelta(seconds=window_seconds)
        points = self.get_metric_values(metric_name, since=since, tags_filter=tags_filter)

        if not points:
            return {"value": None, "count": 0, "window_seconds": window_seconds}

        numerical_values = []
        for point in points:
            if isinstance(point.value, (int, float)):
                numerical_values.append(point.value)

        if not numerical_values:
            return {"value": None, "count": len(points), "window_seconds": window_seconds}

        if aggregation == "avg":
            value = statistics.mean(numerical_values)
        elif aggregation == "sum":
            value = sum(numerical_values)
        elif aggregation == "min":
            value = min(numerical_values)
        elif aggregation == "max":
            value = max(numerical_values)
        elif aggregation == "count":
            value = len(numerical_values)
        elif aggregation == "median":
            value = statistics.median(numerical_values)
        elif aggregation == "stdev":
            value = statistics.stdev(numerical_values) if len(numerical_values) > 1 else 0
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return {
            "value": value,
            "count": len(points),
            "window_seconds": window_seconds,
            "aggregation": aggregation
        }

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics."""
        metrics = {}

        # Loss metrics
        loss_points = self.get_metric_values("training_loss")
        if loss_points:
            recent_loss = self.get_aggregated_metrics("training_loss", "avg", 300)  # 5 min avg
            metrics["current_loss"] = recent_loss["value"]
            metrics["loss_trend"] = self._calculate_trend([p.value for p in loss_points[-10:]])

        # Success rate metrics
        success_points = self.get_metric_values("attack_success_rate")
        if success_points:
            recent_success = self.get_aggregated_metrics("attack_success_rate", "avg", 300)
            metrics["success_rate"] = recent_success["value"]

        # Training speed
        step_times = self.get_metric_values("training_step_duration")
        if step_times:
            avg_step_time = self.get_aggregated_metrics("training_step_duration", "avg", 600)
            metrics["avg_step_time"] = avg_step_time["value"]
            metrics["training_speed"] = 1.0 / avg_step_time["value"] if avg_step_time["value"] else 0

        # Memory usage
        memory_points = self.get_metric_values("memory_usage")
        if memory_points:
            current_memory = self.get_aggregated_metrics("memory_usage", "avg", 60)
            metrics["memory_usage_mb"] = current_memory["value"]

        return metrics

    def get_attack_metrics(self) -> Dict[str, Any]:
        """Get comprehensive attack metrics."""
        metrics = {}

        # Success rates by strategy
        strategies = ["jailbreak", "prompt_injection", "role_play", "obfuscation"]
        for strategy in strategies:
            success_points = self.get_metric_values(
                "attack_success",
                tags_filter={"strategy": strategy}
            )
            if success_points:
                success_rate = sum(1 for p in success_points if p.value) / len(success_points)
                metrics[f"{strategy}_success_rate"] = success_rate

        # Attack generation time
        gen_times = self.get_metric_values("attack_generation_duration")
        if gen_times:
            avg_gen_time = self.get_aggregated_metrics("attack_generation_duration", "avg")
            metrics["avg_generation_time"] = avg_gen_time["value"]

        # Total attacks
        total_attacks = self.get_aggregated_metrics("attack_total", "sum")
        metrics["total_attacks"] = total_attacks["value"] or 0

        return metrics

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        metrics = {}

        # CPU usage
        cpu_points = self.get_metric_values("cpu_usage_percent")
        if cpu_points:
            current_cpu = self.get_aggregated_metrics("cpu_usage_percent", "avg", 60)
            metrics["cpu_usage"] = current_cpu["value"]

        # Memory usage
        memory_points = self.get_metric_values("memory_usage_mb")
        if memory_points:
            current_memory = self.get_aggregated_metrics("memory_usage_mb", "avg", 60)
            metrics["memory_usage"] = current_memory["value"]

        # Disk I/O
        disk_read = self.get_aggregated_metrics("disk_read_mb", "sum", 60)
        disk_write = self.get_aggregated_metrics("disk_write_mb", "sum", 60)
        metrics["disk_read_mb"] = disk_read["value"] or 0
        metrics["disk_write_mb"] = disk_write["value"] or 0

        return metrics

    def get_snapshot(self) -> Dict[str, Any]:
        """Get a complete snapshot of all metrics."""
        return {
            "timestamp": datetime.now().isoformat(),
            "training": self.get_training_metrics(),
            "attacks": self.get_attack_metrics(),
            "system": self.get_system_metrics(),
            "stats": self._stats.copy(),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges)
        }

    def get_metric_names(self) -> List[str]:
        """Get all metric names."""
        with self._lock:
            return list(self._metrics.keys())

    def export_metrics(self, file_path: str, since: Optional[datetime] = None):
        """
        Export metrics to a file.

        Args:
            file_path: Path to export file
            since: Only export metrics after this time
        """
        all_metrics = []

        for metric_name in self.get_metric_names():
            points = self.get_metric_values(metric_name, since=since)
            for point in points:
                all_metrics.append(point.to_dict())

        with open(file_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        logging.info(f"Exported {len(all_metrics)} metric points to {file_path}")

    def _calculate_trend(self, values: List[Union[int, float]]) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < 2:
            return "stable"

        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))

        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def _cleanup_worker(self):
        """Background worker to clean up old metrics."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes

                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

                with self._lock:
                    for metric_name, points in self._metrics.items():
                        # Remove old points
                        while points and points[0].timestamp < cutoff_time:
                            points.popleft()

                    self._stats["last_cleanup"] = datetime.now()

                logging.debug("Metrics cleanup completed")

            except Exception as e:
                logging.error(f"Error in metrics cleanup: {e}")

    def flush(self):
        """Flush any pending metrics operations."""
        # Force emit current state
        if self._event_bus:
            self._event_bus.emit("metrics_snapshot", self.get_snapshot(), source="metrics_collector")

    def shutdown(self):
        """Graceful shutdown of metrics collector."""
        self.flush()
        logging.info("Metrics collector shutdown complete")


# Training-specific metric helpers
def record_training_step(collector: MetricsCollector, step: int, loss: float, **metrics):
    """Record a training step with all relevant metrics."""
    collector.record("training_step", step, unit="step")
    collector.record("training_loss", loss, unit="loss")

    for name, value in metrics.items():
        collector.record(f"training_{name}", value)


def record_attack_attempt(collector: MetricsCollector,
                         strategy: str,
                         success: bool,
                         generation_time: float,
                         **metrics):
    """Record an attack attempt with all relevant metrics."""
    tags = {"strategy": strategy}

    collector.record("attack_success", 1 if success else 0, tags=tags, unit="boolean")
    collector.record("attack_generation_duration", generation_time, tags=tags, unit="seconds")
    collector.increment("attack_total", tags=tags)

    for name, value in metrics.items():
        collector.record(f"attack_{name}", value, tags=tags)

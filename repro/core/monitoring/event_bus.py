"""
Event Bus Module
Central event system for all components to communicate
"""

import logging
import threading
import queue
import time
import json
from typing import Dict, Any, Callable, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path


@dataclass
class Event:
    """Represents a single event in the system."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str = "unknown"
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.type}_{int(self.timestamp.timestamp() * 1000000)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            id=data.get("id", ""),
            type=data["type"],
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source", "unknown")
        )


class EventBus:
    """
    Central event bus for all system components.
    Provides publish/subscribe functionality with optional persistence.
    """

    def __init__(self,
                 buffer_size: int = 10000,
                 enable_persistence: bool = True,
                 persistence_file: str = "events.jsonl",
                 auto_flush_interval: int = 30):
        """
        Initialize the event bus.

        Args:
            buffer_size: Maximum number of events to keep in memory
            enable_persistence: Whether to persist events to file
            persistence_file: File to persist events to
            auto_flush_interval: Seconds between automatic flushes
        """
        self.buffer_size = buffer_size
        self.enable_persistence = enable_persistence
        self.persistence_file = Path(persistence_file)
        self.auto_flush_interval = auto_flush_interval

        # Thread-safe storage
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._event_history: List[Event] = []

        # Statistics
        self._stats = {
            "total_events": 0,
            "events_by_type": defaultdict(int),
            "subscribers_by_type": defaultdict(int),
            "errors": 0,
            "start_time": datetime.now()
        }

        # Background thread for persistence
        self._running = True
        self._flush_thread = None
        if enable_persistence:
            self._start_flush_thread()

        logging.info(f"📡 Event bus initialized (buffer: {buffer_size}, persistence: {enable_persistence})")

    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of events to subscribe to
            callback: Function to call when event occurs
        """
        with self._lock:
            self._subscribers[event_type].append(callback)
            self._stats["subscribers_by_type"][event_type] += 1

        logging.debug(f"📡 Subscribed to {event_type} events")

    def unsubscribe(self, event_type: str, callback: Callable):
        """
        Unsubscribe from events.

        Args:
            event_type: Type of events to unsubscribe from
            callback: Callback function to remove
        """
        with self._lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                self._stats["subscribers_by_type"][event_type] -= 1

        logging.debug(f"📡 Unsubscribed from {event_type} events")

    def emit(self, event_type: str, data: Dict[str, Any], source: str = "unknown"):
        """
        Emit an event to all subscribers.

        Args:
            event_type: Type of event
            data: Event data
            source: Source component that emitted the event
        """
        event = Event(
            type=event_type,
            data=data,
            timestamp=datetime.now(),
            source=source
        )

        with self._lock:
            # Update statistics
            self._stats["total_events"] += 1
            self._stats["events_by_type"][event_type] += 1

            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self.buffer_size:
                self._event_history.pop(0)

            # Add to buffer for persistence
            if self.enable_persistence:
                try:
                    self._event_buffer.put_nowait(event)
                except queue.Full:
                    # Buffer full, drop oldest event
                    try:
                        self._event_buffer.get_nowait()
                        self._event_buffer.put_nowait(event)
                    except queue.Empty:
                        pass

            # Notify subscribers
            subscribers = self._subscribers[event_type].copy()

        # Call subscribers outside of lock to avoid deadlocks
        for callback in subscribers:
            try:
                callback(data)
            except Exception as e:
                self._stats["errors"] += 1
                logging.error(f"Error in event callback for {event_type}: {e}")

        logging.debug(f"📡 Emitted {event_type} event from {source}")

    def emit_training_update(self, step: int, loss: float, metrics: Dict[str, Any]):
        """Convenience method for training updates."""
        self.emit("training_update", {
            "step": step,
            "loss": loss,
            "metrics": metrics
        }, source="trainer")

    def emit_attack_result(self, attack_id: str, success: bool, strategy: str, metrics: Dict[str, Any]):
        """Convenience method for attack results."""
        self.emit("attack_result", {
            "attack_id": attack_id,
            "success": success,
            "strategy": strategy,
            "metrics": metrics
        }, source="attacker")

    def emit_checkpoint_save(self, checkpoint_name: str, path: str):
        """Convenience method for checkpoint saves."""
        self.emit("checkpoint_save", {
            "name": checkpoint_name,
            "path": path
        }, source="checkpoint_manager")

    def emit_metrics_update(self, metrics: Dict[str, Any]):
        """Convenience method for metrics updates."""
        self.emit("metrics_update", metrics, source="metrics_collector")

    def emit_error(self, error: str, context: Dict[str, Any] = None):
        """Convenience method for error events."""
        self.emit("error", {
            "error": error,
            "context": context or {}
        }, source="system")

    # Multi-turn conversation event convenience methods
    def emit_conversation_started(self, conversation_id: str, objective: str, max_turns: int, **kwargs):
        """Convenience method for conversation started events."""
        self.emit("conversation_started", {
            "conversation_id": conversation_id,
            "objective": objective,
            "max_turns": max_turns,
            **kwargs
        }, source="conversation_manager")

    def emit_conversation_updated(self, conversation_id: str, current_turn: int,
                                  escalation_level: str, should_continue: bool, **kwargs):
        """Convenience method for conversation state updates."""
        self.emit("conversation_updated", {
            "conversation_id": conversation_id,
            "current_turn": current_turn,
            "escalation_level": escalation_level,
            "should_continue": should_continue,
            **kwargs
        }, source="conversation_manager")

    def emit_conversation_turn_recorded(self, conversation_id: str, turn_number: int,
                                        role: str, escalation_level: str,
                                        success_score: float, judge_score: float, **kwargs):
        """Convenience method for turn recorded events."""
        self.emit("conversation_turn_recorded", {
            "conversation_id": conversation_id,
            "turn_number": turn_number,
            "role": role,
            "escalation_level": escalation_level,
            "success_score": success_score,
            "judge_score": judge_score,
            **kwargs
        }, source="conversation_manager")

    def emit_conversation_escalated(self, conversation_id: str, from_level: str, to_level: str, turn_number: int):
        """Convenience method for escalation events."""
        self.emit("conversation_escalated", {
            "conversation_id": conversation_id,
            "from_level": from_level,
            "to_level": to_level,
            "turn_number": turn_number
        }, source="conversation_manager")

    def emit_conversation_attack_successful(self, conversation_id: str, successful_turn: int,
                                            total_turns: int, judge_score: float, escalation_level: str):
        """Convenience method for successful attack events."""
        self.emit("conversation_attack_successful", {
            "conversation_id": conversation_id,
            "successful_turn": successful_turn,
            "total_turns": total_turns,
            "judge_score": judge_score,
            "escalation_level": escalation_level
        }, source="conversation_manager")

    def emit_conversation_completed(self, conversation_id: str, total_turns: int,
                                    attack_successful: bool, cumulative_success: float,
                                    final_escalation_level: str, **kwargs):
        """Convenience method for conversation completion events."""
        self.emit("conversation_completed", {
            "conversation_id": conversation_id,
            "total_turns": total_turns,
            "attack_successful": attack_successful,
            "cumulative_success": cumulative_success,
            "final_escalation_level": final_escalation_level,
            **kwargs
        }, source="conversation_manager")

    def get_events(self,
                   event_type: Optional[str] = None,
                   since: Optional[datetime] = None,
                   limit: Optional[int] = None) -> List[Event]:
        """
        Get events from history.

        Args:
            event_type: Filter by event type
            since: Only return events after this time
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        with self._lock:
            events = self._event_history.copy()

        # Apply filters
        if event_type:
            events = [e for e in events if e.type == event_type]

        if since:
            events = [e for e in events if e.timestamp >= since]

        # Sort by timestamp and apply limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        if limit:
            events = events[:limit]

        return events

    def get_event_types(self) -> Set[str]:
        """Get all event types that have been emitted."""
        with self._lock:
            return set(self._stats["events_by_type"].keys())

    def get_subscriber_count(self, event_type: str) -> int:
        """Get number of subscribers for an event type."""
        with self._lock:
            return len(self._subscribers[event_type])

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats["uptime_seconds"] = (datetime.now() - stats["start_time"]).total_seconds()
            stats["events_in_buffer"] = self._event_buffer.qsize()
            stats["events_in_history"] = len(self._event_history)
            return stats

    def _start_flush_thread(self):
        """Start background thread for persisting events."""
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._flush_thread.start()

    def _flush_worker(self):
        """Background worker that persists events to file."""
        while self._running:
            try:
                # Wait for events or timeout
                events_to_flush = []
                deadline = time.time() + self.auto_flush_interval

                while time.time() < deadline and len(events_to_flush) < 100:
                    try:
                        timeout = max(0, deadline - time.time())
                        event = self._event_buffer.get(timeout=timeout)
                        events_to_flush.append(event)
                    except queue.Empty:
                        break

                # Flush events if any
                if events_to_flush:
                    self._flush_events(events_to_flush)

            except Exception as e:
                logging.error(f"Error in event flush worker: {e}")
                time.sleep(1)

    def _flush_events(self, events: List[Event]):
        """Flush events to persistence file."""
        try:
            with open(self.persistence_file, 'a', encoding='utf-8') as f:
                for event in events:
                    f.write(json.dumps(event.to_dict()) + '\n')

            logging.debug(f"📡 Flushed {len(events)} events to {self.persistence_file}")

        except Exception as e:
            logging.error(f"Failed to flush events: {e}")

    def load_events_from_file(self, file_path: Optional[str] = None) -> List[Event]:
        """
        Load events from persistence file.

        Args:
            file_path: Path to file (defaults to configured persistence file)

        Returns:
            List of loaded events
        """
        file_path = Path(file_path) if file_path else self.persistence_file

        if not file_path.exists():
            return []

        events = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        event_data = json.loads(line)
                        events.append(Event.from_dict(event_data))

            logging.info(f"📡 Loaded {len(events)} events from {file_path}")

        except Exception as e:
            logging.error(f"Failed to load events from {file_path}: {e}")

        return events

    def clear_history(self):
        """Clear event history (but keep subscribers)."""
        with self._lock:
            self._event_history.clear()
            # Clear the buffer
            while not self._event_buffer.empty():
                try:
                    self._event_buffer.get_nowait()
                except queue.Empty:
                    break

        logging.info("📡 Event history cleared")

    def export_events(self, file_path: str, event_type: Optional[str] = None):
        """
        Export events to a file.

        Args:
            file_path: Path to export file
            event_type: Optional filter by event type
        """
        events = self.get_events(event_type=event_type)

        with open(file_path, 'w', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event.to_dict()) + '\n')

        logging.info(f"📡 Exported {len(events)} events to {file_path}")

    def shutdown(self):
        """Gracefully shutdown the event bus."""
        logging.info("📡 Shutting down event bus...")

        self._running = False

        # Wait for flush thread to finish
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5)

        # Flush remaining events
        if self.enable_persistence:
            remaining_events = []
            while not self._event_buffer.empty():
                try:
                    remaining_events.append(self._event_buffer.get_nowait())
                except queue.Empty:
                    break

            if remaining_events:
                self._flush_events(remaining_events)

        logging.info("📡 Event bus shutdown complete")


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> Optional[EventBus]:
    """Get the global event bus instance."""
    return _global_event_bus


def initialize_event_bus(**kwargs) -> EventBus:
    """Initialize the global event bus."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus(**kwargs)
    return _global_event_bus


def emit_event(event_type: str, data: Dict[str, Any], source: str = "unknown"):
    """Emit an event (convenience function)."""
    bus = get_event_bus()
    if bus:
        bus.emit(event_type, data, source)

"""
Checkpoint Manager Module
Handles state persistence with versioning and recovery capabilities
"""

import pickle
import json
import gzip
import threading
import shutil
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging
import hashlib


class CheckpointMetadata:
    """Metadata for checkpoint files."""

    def __init__(self,
                 name: str,
                 timestamp: datetime,
                 size_bytes: int,
                 checksum: str,
                 version: int = 1,
                 tags: Dict[str, str] = None,
                 description: str = ""):
        self.name = name
        self.timestamp = timestamp
        self.size_bytes = size_bytes
        self.checksum = checksum
        self.version = version
        self.tags = tags or {}
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "version": self.version,
            "tags": self.tags,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            size_bytes=data["size_bytes"],
            checksum=data["checksum"],
            version=data.get("version", 1),
            tags=data.get("tags", {}),
            description=data.get("description", "")
        )


class CheckpointManager:
    """
    Comprehensive checkpoint management with versioning, compression, and recovery.
    Supports automatic checkpointing, retention policies, and integrity checks.
    """

    def __init__(self,
                 checkpoint_dir: str = "checkpoints",
                 auto_save: bool = True,
                 compression: bool = True,
                 max_checkpoints: int = 50,
                 retention_days: int = 30):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            auto_save: Whether to enable automatic checkpointing
            compression: Whether to compress checkpoints
            max_checkpoints: Maximum number of checkpoints to keep per object
            retention_days: Days to retain checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.auto_save = auto_save
        self.compression = compression
        self.max_checkpoints = max_checkpoints
        self.retention_days = retention_days

        # Thread safety
        self._lock = threading.RLock()

        # Event bus reference (set by monitoring setup)
        self._event_bus = None

        # Auto-save tracking
        self._auto_save_conditions: Dict[str, Callable] = {}
        self._last_auto_save: Dict[str, datetime] = {}

        # Initialize directory structure
        self._initialize_directories()

        # Load existing metadata
        self._metadata_cache: Dict[str, CheckpointMetadata] = {}
        self._load_metadata()

        logging.info(f"Checkpoint manager initialized (dir: {checkpoint_dir})")

    def set_event_bus(self, event_bus):
        """Set the event bus for emitting checkpoint events."""
        self._event_bus = event_bus

    def _initialize_directories(self):
        """Initialize checkpoint directory structure."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / "metadata").mkdir(exist_ok=True)
        (self.checkpoint_dir / "archives").mkdir(exist_ok=True)

    def _load_metadata(self):
        """Load existing checkpoint metadata."""
        metadata_dir = self.checkpoint_dir / "metadata"
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    metadata = CheckpointMetadata.from_dict(data)
                    self._metadata_cache[metadata.name] = metadata
            except Exception as e:
                logging.warning(f"Failed to load metadata from {metadata_file}: {e}")

    def _save_metadata(self, metadata: CheckpointMetadata):
        """Save checkpoint metadata."""
        metadata_file = self.checkpoint_dir / "metadata" / f"{metadata.name}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _get_checkpoint_path(self, name: str, version: Optional[int] = None) -> Path:
        """Get the path for a checkpoint file."""
        if version is not None:
            filename = f"{name}_v{version}"
        else:
            filename = name

        if self.compression:
            filename += ".pkl.gz"
        else:
            filename += ".pkl"

        return self.checkpoint_dir / filename

    def save(self,
             name: str,
             state: Dict[str, Any],
             description: str = "",
             tags: Dict[str, str] = None,
             force: bool = False) -> Optional[str]:
        """
        Save a checkpoint.

        Args:
            name: Checkpoint name
            state: State dictionary to save
            description: Optional description
            tags: Optional tags
            force: Force save even if auto-save conditions not met

        Returns:
            Checkpoint path if successful, None otherwise
        """
        with self._lock:
            try:
                # Check auto-save conditions
                if not force and self.auto_save and name in self._auto_save_conditions:
                    if not self._auto_save_conditions[name](state):
                        return None

                # Get next version number
                existing_versions = self._get_existing_versions(name)
                version = max(existing_versions, default=0) + 1

                # Save the checkpoint
                checkpoint_path = self._get_checkpoint_path(name, version)
                self._save_checkpoint_file(checkpoint_path, state)

                # Calculate metadata
                file_size = checkpoint_path.stat().st_size
                checksum = self._calculate_checksum(checkpoint_path)

                metadata = CheckpointMetadata(
                    name=f"{name}_v{version}",
                    timestamp=datetime.now(),
                    size_bytes=file_size,
                    checksum=checksum,
                    version=version,
                    tags=tags or {},
                    description=description
                )

                # Save metadata
                self._metadata_cache[metadata.name] = metadata
                self._save_metadata(metadata)

                # Update auto-save tracking
                if self.auto_save:
                    self._last_auto_save[name] = datetime.now()

                # Cleanup old checkpoints
                self._cleanup_old_checkpoints(name)

                # Emit event
                if self._event_bus:
                    self._event_bus.emit("checkpoint_saved", {
                        "name": name,
                        "version": version,
                        "path": str(checkpoint_path),
                        "size_mb": file_size / (1024 * 1024),
                        "description": description
                    }, source="checkpoint_manager")

                logging.info(f"Saved checkpoint: {name} v{version} ({file_size / 1024:.1f} KB)")
                return str(checkpoint_path)

            except Exception as e:
                logging.error(f"Failed to save checkpoint {name}: {e}")
                return None

    def _save_checkpoint_file(self, path: Path, state: Dict[str, Any]):
        """Save checkpoint data to file."""
        if self.compression:
            with gzip.open(path, 'wb') as f:
                pickle.dump(state, f)
        else:
            with open(path, 'wb') as f:
                pickle.dump(state, f)

    def load(self,
             name: str,
             version: Optional[int] = None,
             verify_checksum: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            name: Checkpoint name
            version: Specific version to load (defaults to latest)
            verify_checksum: Whether to verify file integrity

        Returns:
            Loaded state dictionary, or None if failed
        """
        with self._lock:
            try:
                # Find the checkpoint to load
                if version is None:
                    # Load latest version
                    existing_versions = self._get_existing_versions(name)
                    if not existing_versions:
                        logging.warning(f"No checkpoints found for {name}")
                        return None
                    version = max(existing_versions)

                checkpoint_path = self._get_checkpoint_path(name, version)
                if not checkpoint_path.exists():
                    logging.warning(f"Checkpoint not found: {checkpoint_path}")
                    return None

                # Verify checksum if requested
                metadata_name = f"{name}_v{version}"
                if verify_checksum and metadata_name in self._metadata_cache:
                    current_checksum = self._calculate_checksum(checkpoint_path)
                    expected_checksum = self._metadata_cache[metadata_name].checksum
                    if current_checksum != expected_checksum:
                        logging.error(f"Checksum mismatch for {checkpoint_path}")
                        return None

                # Load the checkpoint
                state = self._load_checkpoint_file(checkpoint_path)

                # Emit event
                if self._event_bus:
                    self._event_bus.emit("checkpoint_loaded", {
                        "name": name,
                        "version": version,
                        "path": str(checkpoint_path)
                    }, source="checkpoint_manager")

                logging.info(f"Loaded checkpoint: {name} v{version}")
                return state

            except Exception as e:
                logging.error(f"Failed to load checkpoint {name}: {e}")
                return None

    def _load_checkpoint_file(self, path: Path) -> Dict[str, Any]:
        """Load checkpoint data from file."""
        if self.compression and path.suffix == '.gz':
            with gzip.open(path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)

    def _get_existing_versions(self, name: str) -> List[int]:
        """Get all existing versions for a checkpoint name."""
        versions = []
        for metadata_name, metadata in self._metadata_cache.items():
            if metadata_name.startswith(f"{name}_v"):
                versions.append(metadata.version)
        return sorted(versions)

    def list_checkpoints(self, name_pattern: str = "*") -> List[CheckpointMetadata]:
        """
        List available checkpoints.

        Args:
            name_pattern: Pattern to filter checkpoint names

        Returns:
            List of checkpoint metadata
        """
        import fnmatch

        checkpoints = []
        for metadata in self._metadata_cache.values():
            if fnmatch.fnmatch(metadata.name, name_pattern):
                checkpoints.append(metadata)

        return sorted(checkpoints, key=lambda m: m.timestamp, reverse=True)

    def delete_checkpoint(self, name: str, version: Optional[int] = None) -> bool:
        """
        Delete a checkpoint.

        Args:
            name: Checkpoint name
            version: Specific version to delete (if None, deletes all versions)

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                if version is not None:
                    # Delete specific version
                    checkpoint_path = self._get_checkpoint_path(name, version)
                    metadata_name = f"{name}_v{version}"

                    if checkpoint_path.exists():
                        checkpoint_path.unlink()

                    if metadata_name in self._metadata_cache:
                        del self._metadata_cache[metadata_name]
                        metadata_file = self.checkpoint_dir / "metadata" / f"{metadata_name}.json"
                        if metadata_file.exists():
                            metadata_file.unlink()

                    logging.info(f"Deleted checkpoint: {name} v{version}")

                else:
                    # Delete all versions
                    versions = self._get_existing_versions(name)
                    for v in versions:
                        self.delete_checkpoint(name, v)

                return True

            except Exception as e:
                logging.error(f"Failed to delete checkpoint {name}: {e}")
                return False

    def auto_save(self, event_data: Dict[str, Any]):
        """
        Handle auto-save event from event bus.

        Args:
            event_data: Event data containing save information
        """
        if not self.auto_save:
            return

        name = event_data.get("name")
        state = event_data.get("state")

        if name and state:
            self.save(name, state, description="Auto-saved checkpoint")

    def set_auto_save_condition(self, name: str, condition: Callable[[Dict[str, Any]], bool]):
        """
        Set condition for auto-saving a checkpoint.

        Args:
            name: Checkpoint name
            condition: Function that returns True if checkpoint should be saved
        """
        self._auto_save_conditions[name] = condition

    def _cleanup_old_checkpoints(self, name: str):
        """Clean up old checkpoints according to retention policy."""
        try:
            versions = self._get_existing_versions(name)

            # Remove excess checkpoints (keep max_checkpoints)
            if len(versions) > self.max_checkpoints:
                versions_to_remove = versions[:-self.max_checkpoints]
                for version in versions_to_remove:
                    self.delete_checkpoint(name, version)

            # Remove old checkpoints (older than retention_days)
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            for metadata in list(self._metadata_cache.values()):
                if (metadata.name.startswith(f"{name}_v") and
                    metadata.timestamp < cutoff_date):
                    self.delete_checkpoint(name, metadata.version)

        except Exception as e:
            logging.warning(f"Error during checkpoint cleanup: {e}")

    def export_checkpoint(self, name: str, version: int, export_path: str) -> bool:
        """
        Export a checkpoint to a specific location.

        Args:
            name: Checkpoint name
            version: Version to export
            export_path: Destination path

        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint_path = self._get_checkpoint_path(name, version)
            if not checkpoint_path.exists():
                return False

            shutil.copy2(checkpoint_path, export_path)

            # Also copy metadata
            metadata_name = f"{name}_v{version}"
            if metadata_name in self._metadata_cache:
                metadata_export_path = Path(export_path).parent / f"{Path(export_path).stem}_metadata.json"
                metadata_file = self.checkpoint_dir / "metadata" / f"{metadata_name}.json"
                if metadata_file.exists():
                    shutil.copy2(metadata_file, metadata_export_path)

            logging.info(f"Exported checkpoint {name} v{version} to {export_path}")
            return True

        except Exception as e:
            logging.error(f"Failed to export checkpoint: {e}")
            return False

    def import_checkpoint(self, file_path: str, name: str, description: str = "") -> bool:
        """
        Import a checkpoint from a file.

        Args:
            file_path: Source file path
            name: Name for the imported checkpoint
            description: Optional description

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the checkpoint to verify it's valid
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rb') as f:
                    state = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    state = pickle.load(f)

            # Save as new checkpoint
            result = self.save(name, state, description=f"Imported: {description}")
            return result is not None

        except Exception as e:
            logging.error(f"Failed to import checkpoint: {e}")
            return False

    def get_checkpoint_info(self, name: str, version: Optional[int] = None) -> Optional[CheckpointMetadata]:
        """
        Get information about a checkpoint.

        Args:
            name: Checkpoint name
            version: Specific version (defaults to latest)

        Returns:
            Checkpoint metadata or None
        """
        if version is None:
            versions = self._get_existing_versions(name)
            if not versions:
                return None
            version = max(versions)

        metadata_name = f"{name}_v{version}"
        return self._metadata_cache.get(metadata_name)

    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics."""
        total_size = 0
        checkpoint_count = 0

        for metadata in self._metadata_cache.values():
            total_size += metadata.size_bytes
            checkpoint_count += 1

        return {
            "total_checkpoints": checkpoint_count,
            "total_size_mb": total_size / (1024 * 1024),
            "checkpoint_dir": str(self.checkpoint_dir),
            "compression_enabled": self.compression,
            "auto_save_enabled": self.auto_save,
            "max_checkpoints": self.max_checkpoints,
            "retention_days": self.retention_days
        }

    def flush(self):
        """Flush any pending checkpoint operations."""
        # Ensure all metadata is saved
        for metadata in self._metadata_cache.values():
            try:
                self._save_metadata(metadata)
            except Exception as e:
                logging.warning(f"Failed to flush metadata for {metadata.name}: {e}")

    def shutdown(self):
        """Graceful shutdown of checkpoint manager."""
        self.flush()
        logging.info("Checkpoint manager shutdown complete")


# Utility functions for common checkpoint patterns
def checkpoint_training_state(manager: CheckpointManager,
                             model_state: Dict[str, Any],
                             optimizer_state: Dict[str, Any],
                             step: int,
                             loss: float) -> Optional[str]:
    """Checkpoint training state with standard format."""
    state = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "step": step,
        "loss": loss,
        "timestamp": datetime.now().isoformat()
    }

    return manager.save(
        f"training_step_{step}",
        state,
        description=f"Training checkpoint at step {step}, loss: {loss:.4f}",
        tags={"type": "training", "step": str(step)}
    )


def checkpoint_model_weights(manager: CheckpointManager,
                           model_weights: Dict[str, Any],
                           model_name: str,
                           epoch: int) -> Optional[str]:
    """Checkpoint model weights with standard format."""
    state = {
        "weights": model_weights,
        "model_name": model_name,
        "epoch": epoch,
        "timestamp": datetime.now().isoformat()
    }

    return manager.save(
        f"{model_name}_epoch_{epoch}",
        state,
        description=f"Model weights for {model_name} at epoch {epoch}",
        tags={"type": "model", "model": model_name, "epoch": str(epoch)}
    )

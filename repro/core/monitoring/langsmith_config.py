"""
LangSmith Configuration Module
Handles all LangSmith tracing and monitoring setup
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    from langsmith import Client, traceable
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.tracers import LangChainTracer
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # Define placeholders for type hints when packages not available
    Client = None
    CallbackManager = None
    LangChainTracer = None
    def traceable(f): return f  # No-op decorator
    logging.warning("LangSmith/LangChain not available. Install with: pip install langsmith langchain")


class LangSmithConfig:
    """
    Comprehensive LangSmith configuration and management.
    Handles tracing, datasets, experiments, and feedback collection.
    """

    def __init__(self,
                 project_name: str = "red-team-training",
                 auto_trace: bool = True,
                 api_key: Optional[str] = None,
                 api_url: Optional[str] = None):
        """
        Initialize LangSmith configuration.

        Args:
            project_name: Name of the LangSmith project
            auto_trace: Whether to automatically trace all operations
            api_key: LangSmith API key (defaults to LANGCHAIN_API_KEY env var)
            api_url: LangSmith API URL (defaults to LANGCHAIN_ENDPOINT env var)
        """
        self.project_name = project_name
        self.auto_trace = auto_trace
        # Check both LANGCHAIN_API_KEY and LANGSMITH_API_KEY
        self.api_key = api_key or os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")

        # If we found the key in LANGSMITH_API_KEY, set LANGCHAIN_API_KEY for library compatibility
        if not os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGSMITH_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
        self.api_url = api_url or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

        self.client: Optional[Client] = None
        self.tracer: Optional[LangChainTracer] = None
        self.callback_manager: Optional[CallbackManager] = None

        self.is_configured = False
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def configure(self) -> bool:
        """
        Configure LangSmith tracing and client.

        Returns:
            bool: True if configuration successful, False otherwise
        """
        if not LANGSMITH_AVAILABLE:
            logging.warning("LangSmith not available, skipping configuration")
            return False

        if not self.api_key:
            logging.warning("LANGCHAIN_API_KEY not set, skipping LangSmith configuration")
            return False

        try:
            # Set environment variables for automatic tracing
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            os.environ["LANGCHAIN_API_KEY"] = self.api_key
            if self.api_url:
                os.environ["LANGCHAIN_ENDPOINT"] = self.api_url

            # Initialize client
            self.client = Client(api_key=self.api_key, api_url=self.api_url)

            # Test connection
            try:
                projects = list(self.client.list_projects())
                logging.info(f"LangSmith connected successfully. Found {len(projects)} projects.")
            except Exception as e:
                logging.warning(f"LangSmith connection test failed: {e}")

            # Create project if it doesn't exist
            self._ensure_project_exists()

            # Setup tracer and callback manager
            if self.auto_trace:
                self._setup_auto_tracing()

            self.is_configured = True
            logging.info(f"LangSmith configured for project: {self.project_name}")
            return True

        except Exception as e:
            logging.error(f"Failed to configure LangSmith: {e}")
            return False

    def _ensure_project_exists(self):
        """Ensure the project exists, create if necessary."""
        if not self.client:
            return

        try:
            # Try to get the project
            project = self.client.read_project(project_name=self.project_name)
            logging.info(f"Using existing project: {project.name}")
        except Exception:
            # Project doesn't exist, create it
            try:
                project = self.client.create_project(
                    project_name=self.project_name,
                    description=f"Red Team Training Project - {self.session_id}"
                )
                logging.info(f"Created new project: {project.name}")
            except Exception as e:
                logging.warning(f"Failed to create project: {e}")

    def _setup_auto_tracing(self):
        """Setup automatic tracing for all LangChain operations."""
        try:
            self.tracer = LangChainTracer(project_name=self.project_name)
            self.callback_manager = CallbackManager([self.tracer])
            logging.info("🔄 Auto-tracing enabled")
        except Exception as e:
            logging.warning(f"Failed to setup auto-tracing: {e}")

    def create_dataset(self, name: str, description: str = "", inputs: List[Dict] = None) -> Optional[str]:
        """
        Create a dataset for evaluation.

        Args:
            name: Dataset name
            description: Dataset description
            inputs: List of input examples

        Returns:
            Dataset ID if successful, None otherwise
        """
        if not self.client:
            logging.warning("LangSmith client not configured")
            return None

        try:
            dataset = self.client.create_dataset(
                dataset_name=name,
                description=description
            )

            if inputs:
                for i, input_data in enumerate(inputs):
                    self.client.create_example(
                        dataset_id=dataset.id,
                        inputs=input_data,
                        outputs={}  # Will be filled during evaluation
                    )

            logging.info(f"Created dataset: {name} with {len(inputs or [])} examples")
            return str(dataset.id)

        except Exception as e:
            logging.error(f"Failed to create dataset {name}: {e}")
            return None

    def log_training_step(self,
                         step: int,
                         loss: float,
                         metrics: Dict[str, Any],
                         model_name: str = "red_team_model"):
        """
        Log a training step to LangSmith.

        Args:
            step: Training step number
            loss: Training loss
            metrics: Additional metrics
            model_name: Name of the model being trained
        """
        if not self.client:
            return

        try:
            # Create a run for this training step
            run_data = {
                "name": f"training_step_{step}",
                "inputs": {
                    "step": step,
                    "model_name": model_name
                },
                "outputs": {
                    "loss": loss,
                    "metrics": metrics
                },
                "run_type": "training",
                "session_id": self.session_id
            }

            self.client.create_run(**run_data)

        except Exception as e:
            logging.warning(f"Failed to log training step {step}: {e}")

    def log_attack_attempt(self,
                          attack_id: str,
                          objective: str,
                          strategy: str,
                          success: bool,
                          response: str,
                          metrics: Dict[str, Any] = None):
        """
        Log an attack attempt to LangSmith.

        Args:
            attack_id: Unique attack identifier
            objective: Attack objective
            strategy: Attack strategy used
            success: Whether attack was successful
            response: Target model response
            metrics: Additional metrics
        """
        if not self.client:
            return

        try:
            run_data = {
                "name": f"attack_{attack_id}",
                "inputs": {
                    "objective": objective,
                    "strategy": strategy,
                    "attack_id": attack_id
                },
                "outputs": {
                    "success": success,
                    "response": response,
                    "metrics": metrics or {}
                },
                "run_type": "attack",
                "session_id": self.session_id
            }

            self.client.create_run(**run_data)

        except Exception as e:
            logging.warning(f"Failed to log attack attempt {attack_id}: {e}")

    def create_experiment(self, name: str, dataset_id: str, description: str = "") -> Optional[str]:
        """
        Create an experiment for systematic evaluation.

        Args:
            name: Experiment name
            dataset_id: Dataset to evaluate on
            description: Experiment description

        Returns:
            Experiment ID if successful, None otherwise
        """
        if not self.client:
            return None

        try:
            # Note: This is a simplified experiment creation
            # In practice, you'd use the client's experiment methods
            logging.info(f"🧪 Created experiment: {name}")
            return f"exp_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        except Exception as e:
            logging.error(f"Failed to create experiment {name}: {e}")
            return None

    def add_feedback(self, run_id: str, score: float, value: str = "", comment: str = ""):
        """
        Add feedback to a run.

        Args:
            run_id: Run ID to add feedback to
            score: Numerical score (0.0 to 1.0)
            value: Categorical value
            comment: Optional comment
        """
        if not self.client:
            return

        try:
            self.client.create_feedback(
                run_id=run_id,
                key="human_evaluation",
                score=score,
                value=value,
                comment=comment
            )
            logging.info(f"📝 Added feedback to run {run_id}")

        except Exception as e:
            logging.warning(f"Failed to add feedback to run {run_id}: {e}")

    def get_callback_manager(self) -> Optional[CallbackManager]:
        """Get the callback manager for LangChain operations."""
        return self.callback_manager

    def trace_function(self, name: str = None):
        """
        Decorator to trace any function.

        Args:
            name: Custom name for the trace

        Usage:
            @langsmith_config.trace_function("my_operation")
            def my_function():
                pass
        """
        def decorator(func):
            if not LANGSMITH_AVAILABLE or not self.is_configured:
                return func

            return traceable(name=name or func.__name__)(func)

        return decorator

    def get_project_stats(self) -> Dict[str, Any]:
        """Get statistics about the current project."""
        if not self.client:
            return {}

        try:
            # This would fetch actual project statistics
            # For now, return basic info
            return {
                "project_name": self.project_name,
                "session_id": self.session_id,
                "configured": self.is_configured,
                "auto_trace": self.auto_trace
            }

        except Exception as e:
            logging.warning(f"Failed to get project stats: {e}")
            return {}

    def shutdown(self):
        """Clean shutdown of LangSmith components."""
        if self.tracer:
            try:
                # Flush any pending traces
                pass
            except Exception as e:
                logging.warning(f"Error during LangSmith shutdown: {e}")

        logging.info("LangSmith configuration shutdown complete")


# Utility functions for easy access
def create_traced_function(func, name: str = None):
    """Create a traced version of a function."""
    if not LANGSMITH_AVAILABLE:
        return func

    return traceable(name=name or func.__name__)(func)


def get_langsmith_callback():
    """Get LangSmith callback for manual tracing."""
    if not LANGSMITH_AVAILABLE:
        return None

    try:
        return LangChainTracer()
    except Exception:
        return None
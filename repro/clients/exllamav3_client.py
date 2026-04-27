"""
ExLlamaV3 Client via TabbyAPI
Provides OpenAI-compatible async API with super-batching and streaming support

Features:
- Full async support for LangGraph compatibility
- Super-batching (8-16x faster batch inference)
- Real-time streaming
- Automatic multi-GPU detection and distribution
- Connection pooling for optimal performance
"""

import os
import asyncio
import logging
import time
import subprocess
import socket
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import re

from repro.clients.logging_utils import log_full_responses_enabled, log_request_summary, new_request_id, summarize_text

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logging.warning("httpx not installed. Install with: pip install httpx")

logger = logging.getLogger(__name__)


@dataclass
class ExLlamaV3Config:
    """Configuration for ExLlamaV3/TabbyAPI client"""
    base_url: str = "http://localhost:5000"
    timeout: int = 300
    # Granular timeout settings to prevent TCP deadlocks
    connect_timeout: float = 10.0  # Connection establishment timeout
    read_timeout: float = 300.0    # Read timeout (matches main timeout for long generations)
    write_timeout: float = 30.0    # Write timeout for sending requests
    pool_timeout: float = 30.0     # Timeout waiting for connection from pool
    # Health checks can time out under heavy generation load; keep this generous
    health_timeout: float = 5.0
    enable_streaming: bool = True
    enable_batching: bool = True
    batch_size: int = 8
    max_concurrent: int = 8  # Balanced for larger batch sizes - prevents connection saturation
    api_key: Optional[str] = None  # Optional API key for secured TabbyAPI
    auto_start_server: bool = True  # Auto-start TabbyAPI server if not running
    tabbyapi_path: str = ""  # Path to TabbyAPI installation (from EXLLAMAV3_PATH env var)
    # Connection pool settings
    keepalive_expiry: float = 30.0  # Close idle connections after 30 seconds
    max_retries: int = 3            # Retry transient failures
    retry_backoff: float = 1.0      # Base backoff time between retries
    strict_base_url: bool = False   # When True, always use base_url as-is; skip all model-name port resolution


class ExLlamaV3Client:
    """
    TabbyAPI client with ExLlamaV3 backend

    Provides OpenAI-compatible async API with super-batching and streaming.
    Designed for seamless integration with LangGraph workflows.
    """

    def __init__(self, config: Optional[ExLlamaV3Config] = None):
        """
        Initialize ExLlamaV3 client

        Args:
            config: Client configuration (uses defaults from env vars if not provided)
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required. Install with: pip install httpx")

        self.config = config or self._load_config_from_env()

        # Async client for connection pooling (recreated if event loop changes)
        self._async_client: Optional[httpx.AsyncClient] = None
        self._async_client_loop: Optional[asyncio.AbstractEventLoop] = None
        self._client_lock = asyncio.Lock()

        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # GPU configuration
        self.gpu_config = self._detect_gpu_configuration()

        # Performance tracking
        self._stats = {
            'total_requests': 0,
            'batch_requests': 0,
            'stream_requests': 0,
            'total_tokens': 0,
            'total_time': 0.0
        }

        # Server process tracking
        self._server_process: Optional[subprocess.Popen] = None
        self._server_started_by_client = False

        # Cache server status per port to avoid repeated health checks
        self._server_status_cache: Dict[str, float] = {}  # base_url -> last_check_time
        self._server_status_ttl: float = 30.0  # Cache for 30 seconds

        logger.info(f"ExLlamaV3Client initialized: {self.config.base_url}")
        logger.info(f"GPU configuration: {self.gpu_config['gpu_count']} GPU(s)")

        # Context overflow handling (uniform across all ExLlamaV3 models)
        self._auto_trim_context: bool = os.getenv("EXLLAMAV3_AUTO_TRIM", "true").lower() == "true"
        # Reserve a small margin of tokens under the reported max_seq_len when trimming
        try:
            self._trim_margin_tokens: int = int(os.getenv("EXLLAMAV3_TRIM_MARGIN", "128"))
        except ValueError:
            self._trim_margin_tokens = 128
        logger.info(
            f"ExLlamaV3Client context policy: auto_trim={self._auto_trim_context}, margin={self._trim_margin_tokens} tokens"
        )

    def _load_config_from_env(self) -> ExLlamaV3Config:
        """Load configuration from environment variables"""
        # Default TabbyAPI path (platform-specific)
        if os.name == 'nt':
            default_tabbyapi_path = str(Path.home() / "tabbyapi")
        else:
            default_tabbyapi_path = str(Path.home() / "tabbyAPI")

        timeout = int(os.getenv("EXLLAMAV3_TIMEOUT", "300"))
        return ExLlamaV3Config(
            base_url=os.getenv("EXLLAMAV3_URL", "http://127.0.0.1:5000"),
            timeout=timeout,
            # Granular timeouts - can be overridden via env vars
            connect_timeout=float(os.getenv("EXLLAMAV3_CONNECT_TIMEOUT", "10.0")),
            read_timeout=float(os.getenv("EXLLAMAV3_READ_TIMEOUT", str(timeout))),
            write_timeout=float(os.getenv("EXLLAMAV3_WRITE_TIMEOUT", "30.0")),
            pool_timeout=float(os.getenv("EXLLAMAV3_POOL_TIMEOUT", "30.0")),
            health_timeout=float(os.getenv("EXLLAMAV3_HEALTH_TIMEOUT", "5.0")),
            enable_streaming=os.getenv("EXLLAMAV3_STREAMING", "true").lower() == "true",
            enable_batching=os.getenv("EXLLAMAV3_BATCHING", "true").lower() == "true",
            batch_size=int(os.getenv("EXLLAMAV3_BATCH_SIZE", "8")),
            # Reduced to 8 to balance with larger batch sizes; prevents connection saturation
            max_concurrent=int(os.getenv("EXLLAMAV3_MAX_CONCURRENT", "8")),
            api_key=os.getenv("EXLLAMAV3_API_KEY"),
            auto_start_server=os.getenv("EXLLAMAV3_AUTO_START", "true").lower() == "true",
            tabbyapi_path=os.getenv("EXLLAMAV3_PATH", default_tabbyapi_path),
            # Connection pool settings
            keepalive_expiry=float(os.getenv("EXLLAMAV3_KEEPALIVE_EXPIRY", "30.0")),
            max_retries=int(os.getenv("EXLLAMAV3_MAX_RETRIES", "3")),
            retry_backoff=float(os.getenv("EXLLAMAV3_RETRY_BACKOFF", "1.0")),
        )

    def _normalize_model_suffixes(self, model_name: str) -> list:
        """Generate candidate normalized suffixes for env var lookups.

        We support multiple normalizations to avoid user confusion:
        - Strip provider prefixes (exl3:, exl2:, exllamav3:)
        - Strip quantization suffixes (-exl3-*, -exl2-*, -gguf-*, -4bpw, -8bpw, etc.)
        - Replace '-' and '.' with '_'
        - Replace any non-alphanumeric with '_'

        Example: 'Mistral-7B-Instruct-v0.3-exl3-4bpw' ->
            ['mistral_7b_instruct_v0_3', 'mistral_7b_instruct_v0_3']
        Example: 'Llama2-7B-chat-exl2-4bpw' ->
            ['llama2_7b_chat', 'llama2_7b_chat']
        """
        import re

        # Strip provider prefixes if present
        for prefix in ("exl3:", "exl2:", "exllamav3:"):
            if model_name.lower().startswith(prefix):
                model_name = model_name[len(prefix):]
                break

        base = model_name.lower()

        # Strip quantization suffixes: -exl3-4bpw, -exl2-8bpw, -gguf-q4, etc.
        # Pattern matches: -exl3-*, -exl2-*, -gguf-*, or standalone -Nbpw
        base = re.sub(r'[_-](exl[23]|gguf|awq|gptq)[_-]?\d*b?p?w?.*$', '', base, flags=re.IGNORECASE)
        # Also strip standalone quantization like -4bpw, -8bpw at the end
        base = re.sub(r'[_-]\d+bpw$', '', base, flags=re.IGNORECASE)

        v1 = base.replace('-', '_').replace('.', '_')
        v2 = re.sub(r'[^a-z0-9]+', '_', base)

        # Strip trailing underscores
        v1 = v1.rstrip('_')
        v2 = v2.rstrip('_')

        # Deduplicate while preserving order
        seen = set()
        candidates = []
        for v in [v1, v2]:
            if v not in seen:
                seen.add(v)
                candidates.append(v)
        return candidates

    def _get_model_url_from_env(self, model_name: str) -> Optional[str]:
        """Return a full base URL from per-model env if provided.

        Checks EXLLAMAV3_MODEL_URL_<normalized> for supported normalizations.
        """
        for suffix in self._normalize_model_suffixes(model_name):
            env_var = f"EXLLAMAV3_MODEL_URL_{suffix}"
            url = os.getenv(env_var)
            if url:
                return url
        return None

    def _get_model_port(self, model_name: str) -> int:
        """
        Get the port for a specific model from environment variables.

        Resolution order:
        1) EXLLAMAV3_MODEL_PORT_<normalized>
        2) EXLLAMAV3_MODEL_URL_<normalized> (extract port if present)
        3) EXLLAMAV3_MODEL_PORT_DEFAULT or EXLLAMAV3_DEFAULT_PORT
        4) Port from EXLLAMAV3_URL (client base_url) if present
        5) Fallback 5000
        """
        # 1) Per-model explicit port
        for suffix in self._normalize_model_suffixes(model_name):
            env_var = f"EXLLAMAV3_MODEL_PORT_{suffix}"
            port = os.getenv(env_var)
            if port:
                try:
                    return int(port)
                except ValueError:
                    logger.warning(f"Invalid port value for {env_var}: {port}")

        # 2) Per-model explicit URL
        url = self._get_model_url_from_env(model_name)
        if url:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                if parsed.port:
                    return int(parsed.port)
            except Exception as e:
                logger.warning(f"Invalid EXLLAMAV3_MODEL_URL for {model_name}: {e}")

        # 3) Global default port envs
        for env_var in ("EXLLAMAV3_MODEL_PORT_DEFAULT", "EXLLAMAV3_DEFAULT_PORT"):
            port = os.getenv(env_var)
            if port:
                try:
                    return int(port)
                except ValueError:
                    logger.warning(f"Invalid port value for {env_var}: {port}")

        # 4) Port from configured base URL
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.config.base_url)
            if parsed.port:
                return int(parsed.port)
        except Exception:
            pass

        # 5) Fallback
        logger.debug(f"No port mapping found for {model_name}, using default port 5000")
        return 5000

    def _get_base_url_for_model(self, model_name: str) -> str:
        """
        Get the base URL for a specific model.

        When strict_base_url=True, always use config.base_url directly (path
        component stripped).  This prevents model-name port-mapping env vars
        from overriding an explicitly configured URL (e.g. the pre-filter
        server on a different port than the reflective LM).

        Normal resolution order (strict_base_url=False):
        1) EXLLAMAV3_MODEL_URL_<normalized>
        2) Use scheme+host from config.base_url and port from model resolution
        3) Default to http://127.0.0.1:<port>
        """
        if self.config.strict_base_url:
            # Always honour the explicitly configured URL — strip path component
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self.config.base_url)
                scheme = parsed.scheme or "http"
                host = parsed.hostname or "127.0.0.1"
                port = parsed.port or 5000
                return f"{scheme}://{host}:{port}"
            except Exception:
                return self.config.base_url

        # 1) Prefer explicit per-model URL
        url = self._get_model_url_from_env(model_name)
        if url:
            return url

        # 2) Derive from configured base_url host + resolved port
        port = self._get_model_port(model_name)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.config.base_url)
            scheme = parsed.scheme or "http"
            host = parsed.hostname or "127.0.0.1"
            return f"{scheme}://{host}:{port}"
        except Exception:
            # 3) Safe fallback
            return f"http://127.0.0.1:{port}"

    def _detect_gpu_configuration(self) -> Dict[str, Any]:
        """
        Detect GPU configuration for ExLlamaV3

        Returns:
            Dictionary with GPU configuration details
        """
        try:
            # Try to import config to get GPU info
            from repro.core.config import DETECTED_GPUS

            gpu_count = len(DETECTED_GPUS) if DETECTED_GPUS else 0

            if gpu_count == 0:
                logger.warning("No GPUs detected! ExLlamaV3 requires NVIDIA GPU.")
                return {
                    'gpu_count': 0,
                    'gpu_split': None,
                    'cache_mode': 'FP16',
                    'max_seq_len': 2048
                }
            elif gpu_count == 1:
                logger.info("Single GPU detected - using optimized settings")
                return {
                    'gpu_count': 1,
                    'gpu_split': None,
                    'cache_mode': 'FP16',
                    'max_seq_len': 4096,
                    'vram_usage': '95%'
                }
            else:
                logger.info(f"Multi-GPU detected ({gpu_count} GPUs) - automatic layer distribution")
                return {
                    'gpu_count': gpu_count,
                    'gpu_split': 'auto',
                    'cache_mode': 'FP16',
                    'max_seq_len': 8192,
                    'vram_usage': '95%'
                }
        except ImportError:
            logger.warning("Could not import GPU config, using defaults")
            return {
                'gpu_count': 1,
                'gpu_split': None,
                'cache_mode': 'FP16',
                'max_seq_len': 4096
            }

    def _is_port_open(self, host: str, port: int) -> bool:
        """Check if a port is open"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result == 0
        except Exception:
            return False

    async def _check_server_running(self) -> bool:
        """Check if TabbyAPI server is running"""
        try:
            # Parse URL to get host and port
            from urllib.parse import urlparse
            parsed = urlparse(self.config.base_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 5000

            logger.debug(f"Checking server: {self.config.base_url} (host={host}, port={port})")

            # First check if port is open
            if not self._is_port_open(host, port):
                return False

            # Then try to ping health endpoint
            client = await self._get_async_client()
            try:
                response = await client.get(
                    f"{self.config.base_url}/health",
                    timeout=self.config.health_timeout
                )
                return response.status_code == 200
            except BaseException as e:
                # Try alternate endpoint
                try:
                    response = await client.get(
                        f"{self.config.base_url}/v1/models",
                        timeout=self.config.health_timeout
                    )
                    return response.status_code == 200
                except BaseException:
                    # If the port is open but health endpoints are slow/unresponsive (common under load),
                    # treat the server as "up" and let the actual request determine correctness.
                    logger.debug(f"Health check failed but port is open; proceeding anyway. base_url={self.config.base_url} err={e}")
                    return True
        except Exception as e:
            logger.debug(f"Server check failed: {e}")
            return False

    async def _start_server(self) -> bool:
        """Start TabbyAPI server in background"""
        if not self.config.auto_start_server:
            logger.warning("Auto-start is disabled. Please start TabbyAPI manually.")
            return False

        tabbyapi_path = Path(self.config.tabbyapi_path)

        if not tabbyapi_path.exists():
            logger.error(f"TabbyAPI not found at: {tabbyapi_path}")
            logger.error("Please run the installation script: scripts/install_tabbyapi.ps1")
            return False

        # Platform-specific venv paths
        if os.name == 'nt':
            venv_python = tabbyapi_path / "venv" / "Scripts" / "python.exe"
            venv_bin = tabbyapi_path / "venv" / "Scripts"
        else:
            # Linux - try venv_py310 first (TabbyAPI standard), then venv
            venv_python = tabbyapi_path / "venv_py310" / "bin" / "python"
            venv_bin = tabbyapi_path / "venv_py310" / "bin"
            if not venv_python.exists():
                venv_python = tabbyapi_path / "venv" / "bin" / "python"
                venv_bin = tabbyapi_path / "venv" / "bin"

        main_script = tabbyapi_path / "main.py"

        if not venv_python.exists() or not main_script.exists():
            logger.error(f"TabbyAPI installation incomplete at: {tabbyapi_path}")
            logger.error(f"  venv_python: {venv_python} (exists: {venv_python.exists()})")
            logger.error(f"  main_script: {main_script} (exists: {main_script.exists()})")
            return False

        try:
            logger.info(f"Starting TabbyAPI server at {tabbyapi_path}...")

            # Set up environment
            env = os.environ.copy()

            if os.name == 'nt':
                # Windows CUDA paths
                env.update({
                    "CUDA_PATH": "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4",
                    "CUDA_HOME": "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4",
                    "TORCH_CUDA_ARCH_LIST": "8.9",
                    "CUDAARCHS": "89",
                    "CMAKE_CUDA_ARCHITECTURES": "89",
                })
                cuda_bin = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin"
                env["PATH"] = f"{cuda_bin};{venv_bin};{env.get('PATH', '')}"
            else:
                # Linux - just add venv to PATH
                env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

            # Start server in background
            popen_kwargs = {
                "cwd": str(tabbyapi_path),
                "env": env,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }
            if os.name == 'nt':
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE

            self._server_process = subprocess.Popen(
                [str(venv_python), str(main_script)],
                **popen_kwargs
            )

            self._server_started_by_client = True

            # Wait for server to start (max 60 seconds for CUDA extension compilation)
            logger.info("Waiting for TabbyAPI server to start (may take up to 60s for CUDA compilation)...")
            for i in range(60):
                await asyncio.sleep(1)
                if await self._check_server_running():
                    logger.info(f"✓ TabbyAPI server started successfully on {self.config.base_url}")
                    return True

            logger.error("TabbyAPI server failed to start within 60 seconds")
            return False

        except Exception as e:
            logger.error(f"Failed to start TabbyAPI server: {e}")
            return False

    async def _ensure_server_running(self, base_url: Optional[str] = None):
        """
        Ensure TabbyAPI server is running, start if needed

        Args:
            base_url: Optional base URL to check (defaults to self.config.base_url)
        """
        # Save original base_url
        original_base_url = self.config.base_url

        # Use provided base_url or default
        check_url = base_url or self.config.base_url

        # Check cache first
        current_time = time.time()
        if check_url in self._server_status_cache:
            last_check = self._server_status_cache[check_url]
            if current_time - last_check < self._server_status_ttl:
                logger.debug(f"Server status cached for {check_url} (checked {current_time - last_check:.1f}s ago)")
                return

        # Set base_url for check
        if base_url:
            self.config.base_url = base_url

        try:
            if await self._check_server_running():
                # Cache successful check
                self._server_status_cache[check_url] = current_time
                return

            logger.warning(f"TabbyAPI server not running at {self.config.base_url}...")

            if self.config.auto_start_server:
                logger.info("Attempting to auto-start TabbyAPI server...")
                if not await self._start_server():
                    raise ConnectionError(
                        f"TabbyAPI server not available at {self.config.base_url}. "
                        "Please start it manually or check your configuration."
                    )
                # Cache successful start
                self._server_status_cache[check_url] = current_time
            else:
                raise ConnectionError(
                    f"TabbyAPI server not available at {self.config.base_url}. "
                    "Please start it manually."
                )
        finally:
            # Restore original base_url
            self.config.base_url = original_base_url

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client with connection pooling.

        If the event loop has changed since the client was created,
        close and recreate the client to avoid 'Event loop is closed'.
        """
        # Check if we need to recreate the lock (happens when event loop changes)
        try:
            # Get the current running loop
            current_loop = asyncio.get_running_loop()

            # Test if the lock is usable and on the correct loop
            if hasattr(self._client_lock, '_loop'):
                lock_loop = self._client_lock._loop
                if lock_loop is None or lock_loop.is_closed() or lock_loop != current_loop:
                    logger.debug(f"Recreating client lock due to event loop change")
                    self._client_lock = asyncio.Lock()
            else:
                self._client_lock = asyncio.Lock()
        except (AttributeError, RuntimeError) as e:
            # Lock is invalid, recreate it
            logger.debug(f"Recreating client lock due to error: {e}")
            self._client_lock = asyncio.Lock()

        async with self._client_lock:
            # Recreate client if loop changed or previous client closed
            if (
                self._async_client is not None
                and not self._async_client.is_closed
                and (
                    not hasattr(self, "_async_client_loop")
                    or self._async_client_loop is None
                    or self._async_client_loop.is_closed()
                    or self._async_client_loop is not current_loop
                )
            ):
                try:
                    await self._async_client.aclose()
                except Exception:
                    pass
                self._async_client = None
                self._async_client_loop = None

            if self._async_client is None or self._async_client.is_closed:
                headers = {"Content-Type": "application/json"}
                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"

                # Use granular timeouts to prevent TCP deadlocks
                timeout = httpx.Timeout(
                    connect=self.config.connect_timeout,
                    read=self.config.read_timeout,
                    write=self.config.write_timeout,
                    pool=self.config.pool_timeout,
                )

                # Configure connection pool with keepalive expiry to prevent stale connections
                limits = httpx.Limits(
                    max_connections=self.config.max_concurrent * 2,
                    max_keepalive_connections=self.config.max_concurrent,
                    keepalive_expiry=self.config.keepalive_expiry,  # Close idle connections
                )

                self._async_client = httpx.AsyncClient(
                    timeout=timeout,
                    headers=headers,
                    limits=limits,
                    http2=False,  # Disable HTTP/2 to avoid multiplexing issues with TabbyAPI
                )
                self._async_client_loop = current_loop
                logger.debug(
                    f"Created new httpx client: connect={self.config.connect_timeout}s, "
                    f"read={self.config.read_timeout}s, keepalive_expiry={self.config.keepalive_expiry}s"
                )
            return self._async_client

    async def close(self):
        """Close the async client and clean up resources"""
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()
        self._async_client = None
        self._async_client_loop = None

        # Stop server if we started it
        if self._server_started_by_client and self._server_process:
            logger.info("Stopping TabbyAPI server (started by client)...")
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            logger.info("TabbyAPI server stopped")

        logger.info("ExLlamaV3Client closed")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - Don't close client (singleton pattern)"""
        # Don't close the client - it's reused across requests
        pass

    def _get_or_recreate_semaphore(self) -> asyncio.Semaphore:
        """Get or recreate semaphore if event loop changed"""
        try:
            # Get the current event loop
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                # No event loop running, create semaphore that will work when loop starts
                self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
                return self._semaphore

            # Check if semaphore's loop matches current loop
            if hasattr(self._semaphore, '_loop'):
                semaphore_loop = self._semaphore._loop
                # If loops don't match or semaphore's loop is closed, recreate
                if semaphore_loop is None or semaphore_loop.is_closed() or semaphore_loop != current_loop:
                    logger.debug(f"Recreating semaphore due to event loop change")
                    self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

            return self._semaphore
        except (AttributeError, RuntimeError) as e:
            # Semaphore is invalid, recreate it
            logger.debug(f"Recreating semaphore due to error: {e}")
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
            return self._semaphore

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable (transient network/server issues)"""
        if isinstance(error, httpx.ConnectError):
            return True
        if isinstance(error, httpx.ConnectTimeout):
            return True
        if isinstance(error, httpx.ReadTimeout):
            return True
        if isinstance(error, httpx.PoolTimeout):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            # Retry on server errors (5xx) but not client errors (4xx)
            status = error.response.status_code
            return status >= 500 or status == 429  # 429 = rate limited
        if isinstance(error, (ConnectionError, ConnectionResetError, BrokenPipeError)):
            return True
        return False

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with retry logic for transient failures"""
        last_error = None
        client = await self._get_async_client()

        for attempt in range(self.config.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = await client.get(url, **kwargs)
                elif method.upper() == "POST":
                    response = await client.post(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                return response

            except Exception as e:
                last_error = e

                # Check if this is a retryable error
                if not self._is_retryable_error(e):
                    raise

                # Don't retry if we've exhausted attempts
                if attempt >= self.config.max_retries:
                    logger.error(f"Request failed after {attempt + 1} attempts: {e}")
                    raise

                # Calculate backoff with exponential increase
                backoff = self.config.retry_backoff * (2 ** attempt)
                logger.warning(
                    f"Retryable error on attempt {attempt + 1}/{self.config.max_retries + 1}: {e}. "
                    f"Retrying in {backoff:.1f}s..."
                )

                # Close and recreate client on connection errors to clear bad connections
                if isinstance(e, (httpx.ConnectError, ConnectionResetError, BrokenPipeError)):
                    logger.debug("Recreating HTTP client due to connection error")
                    try:
                        await self._async_client.aclose()
                    except Exception:
                        pass
                    self._async_client = None
                    client = await self._get_async_client()

                await asyncio.sleep(backoff)

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise RuntimeError("Request failed with unknown error")

    async def generate_async(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt (single generation)

        Compatible with LangGraph async nodes and AttackerModel.

        Args:
            model: Model name (will use whatever model is loaded in TabbyAPI)
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-K sampling parameter
            repetition_penalty: Repetition penalty
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        # Get model-specific base URL
        base_url = self._get_base_url_for_model(model)
        logger.debug(f"Using base URL {base_url} for model {model}")

        # Ensure server is running before making request
        await self._ensure_server_running(base_url=base_url)

        start_time = time.time()

        # Use semaphore, recreating if needed due to event loop changes
        semaphore = self._get_or_recreate_semaphore()
        async with semaphore:
            # DETERMINISTIC MODE: When temperature=0, use greedy decoding
            # top_k=1 ensures we always pick the highest probability token
            if temperature <= 0.01:
                top_k = 1
                top_p = 1.0  # Disable nucleus sampling
                logger.debug(f"Deterministic mode: temp={temperature}, top_k=1, top_p=1.0")

            payload = {
                "model": model,  # Specify which model to use
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stream": False,
                **kwargs
            }

            try:
                request_id = new_request_id()
                log_request_summary(logger, "ExLlamaV3", prompt=prompt, request_id=request_id)

                # Use retry mechanism for transient failures
                response = await self._request_with_retry(
                    "POST",
                    f"{base_url}/v1/completions",
                    json=payload
                )

                result = response.json()
                generated_text = result['choices'][0]['text'].strip()

                # Update stats
                self._stats['total_requests'] += 1
                self._stats['total_time'] += (time.time() - start_time)

                return generated_text

            except httpx.HTTPStatusError as e:
                response_obj = getattr(e, "response", None)
                resp_text = response_obj.text if response_obj is not None else ""
                resp_lower = resp_text.lower()
                status_code = response_obj.status_code if response_obj is not None else "unknown"
                logger.error(f"HTTP error in generate_async: {status_code} - {summarize_text(resp_text, limit=500)} request_id={request_id}")
                if (
                    self._auto_trim_context
                    and response_obj is not None
                    and response_obj.status_code in (400, 413, 422, 503)
                    and (
                        ("prompt length" in resp_lower)
                        or ("max_seq_len" in resp_lower)
                    )
                ):
                    try:
                        trimmed_prompt = await self._trim_prompt_to_fit(base_url, prompt, resp_text)
                        if trimmed_prompt is not None:
                            logger.warning("Retrying completion request after context trimming")
                            retry_payload = {**payload, "prompt": trimmed_prompt}
                            retry_resp = await self._request_with_retry(
                                "POST",
                                f"{base_url}/v1/completions",
                                json=retry_payload
                            )
                            retry_result = retry_resp.json()
                            return retry_result['choices'][0]['text'].strip()
                    except Exception as trim_err:
                        logger.error(f"Auto-trim retry (completions) failed: {trim_err}")
                raise
            except Exception as e:
                logger.error(f"Error in generate_async: {e}")
                raise

    async def _trim_prompt_to_fit(
        self,
        base_url: str,
        prompt: str,
        server_error_text: str,
    ) -> Optional[str]:
        """Trim a single prompt string to fit under server max_seq_len and return new prompt."""
        max_len = None
        try:
            m2 = re.search(r"max_seq_len\s+(\d+)", server_error_text, re.IGNORECASE)
            if m2:
                max_len = int(m2.group(1))
        except Exception:
            pass
        if max_len is None:
            max_len = await self._fetch_server_max_seq_len(base_url)
        if not isinstance(max_len, int) or max_len <= 0:
            return None
        # Approx chars-per-token ~= 4 (conservative)
        max_chars = max((max_len - self._trim_margin_tokens) * 4, 512)
        if len(prompt) <= max_chars:
            return prompt
        # Keep the tail (recent instructions) as those usually carry the task
        return prompt[-max_chars:]

    async def chat_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        response_format: Optional[Dict[str, Any]] = None,
        ban_eos_token: Optional[bool] = None,
        min_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion from messages

        Compatible with LangGraph async nodes, TargetModel, and JudgeModel.

        Args:
            model: Model name (will use whatever model is loaded in TabbyAPI)
            messages: Chat messages in OpenAI format [{"role": "user", "content": "..."}]
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-K sampling parameter
            repetition_penalty: Repetition penalty
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        # Get model-specific base URL
        base_url = self._get_base_url_for_model(model)
        logger.debug(f"Using base URL {base_url} for model {model}")

        # Ensure server is running before making request
        await self._ensure_server_running(base_url=base_url)

        start_time = time.time()

        # Use semaphore, recreating if needed due to event loop changes
        semaphore = self._get_or_recreate_semaphore()
        async with semaphore:
            # Validate messages parameter
            if not isinstance(messages, list):
                raise ValueError(f"messages must be a list, got {type(messages)}")

            for idx, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    raise ValueError(f"Message {idx} must be a dict, got {type(msg)}")
                if "role" not in msg or "content" not in msg:
                    raise ValueError(f"Message {idx} missing 'role' or 'content': {msg}")

            # DETERMINISTIC MODE: When temperature=0, use greedy decoding
            if temperature <= 0.01:
                top_k = 1
                top_p = 1.0
                logger.debug(f"Deterministic mode: temp={temperature}, top_k=1, top_p=1.0")

            payload = {
                "model": model,  # Specify which model to use
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stream": False,
                **kwargs
            }

            # Add JSON schema constraint if provided (TabbyAPI/OpenAI-compatible)
            if response_format is not None:
                payload["response_format"] = response_format

            # TRUNCATION FIX: Ban EOS token for long generations to prevent early stopping
            # This forces the model to generate until max_tokens is reached
            if ban_eos_token is not None:
                payload["ban_eos_token"] = ban_eos_token
            elif max_tokens > 1000:
                # Auto-enable for long generations (likely XML/JSON with multiple items)
                payload["ban_eos_token"] = True
                logger.debug(f"Auto-enabled ban_eos_token for long generation (max_tokens={max_tokens})")

            # Add min_tokens if specified (TabbyAPI may support this)
            if min_tokens is not None:
                payload["min_tokens"] = min_tokens

            logger.debug(f"Sending chat request to TabbyAPI: model={model}, messages_count={len(messages)}")
            request_id = new_request_id()
            log_request_summary(logger, "ExLlamaV3", messages=messages, request_id=request_id)

            try:
                # Use retry mechanism for transient failures
                response = await self._request_with_retry(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    json=payload
                )
                logger.debug(f"Response status code: {response.status_code}")

                # Try to parse JSON response
                try:
                    result = response.json()
                except Exception as json_error:
                    logger.error(f"Failed to parse JSON response: {json_error}")
                    logger.error(f"Raw response text: {summarize_text(response.text, limit=500)}")
                    raise ValueError(f"TabbyAPI returned invalid JSON: {json_error}")

                logger.debug(f"TabbyAPI response structure: {list(result.keys())}")
                log_full_response = log_full_responses_enabled()
                if log_full_response:
                    logger.debug(f"Full TabbyAPI response: {result} request_id={request_id}")
                else:
                    usage = result.get("usage")
                    logger.debug(
                        "TabbyAPI response summary: id=%s choices=%s usage=%s request_id=%s",
                        result.get("id"),
                        len(result.get("choices", [])),
                        usage,
                        request_id,
                    )

                # Check if response has expected structure
                if 'choices' not in result:
                    logger.error(f"Unexpected TabbyAPI response format: {result}")
                    raise ValueError(f"TabbyAPI response missing 'choices' key. Response: {result}")

                # Safely extract the message content
                if len(result['choices']) == 0:
                    raise ValueError("TabbyAPI returned empty choices array")

                choice = result['choices'][0]
                logger.debug(f"Choice structure: {choice}")

                # Handle both 'message' and 'text' response formats
                if 'message' in choice:
                    generated_text = choice['message']['content'].strip()
                elif 'text' in choice:
                    generated_text = choice['text'].strip()
                else:
                    logger.error(f"Unknown choice structure: {choice}")
                    raise ValueError(f"TabbyAPI choice missing 'message' or 'text' key. Choice: {choice}")

                # Update stats
                self._stats['total_requests'] += 1
                self._stats['total_time'] += (time.time() - start_time)

                return generated_text

            except httpx.HTTPStatusError as e:
                response_obj = getattr(e, "response", None)
                resp_text = response_obj.text if response_obj is not None else ""
                resp_lower = resp_text.lower()
                status_code = response_obj.status_code if response_obj is not None else "unknown"
                logger.error(f"HTTP error in chat_async: {status_code} - {resp_text}")
                # Try automatic context trimming if server reports context overflow
                if (
                    self._auto_trim_context
                    and response_obj is not None
                    and response_obj.status_code in (400, 413, 422, 503)
                    and (
                        ("prompt length" in resp_lower)
                        or ("max_seq_len" in resp_lower)
                    )
                ):
                    try:
                        trimmed_messages = await self._trim_messages_to_fit(base_url, messages, resp_text)
                        if trimmed_messages is not None:
                            logger.warning("Retrying chat request after context trimming")
                            retry_payload = {**payload, "messages": trimmed_messages}
                            retry_resp = await self._request_with_retry(
                                "POST",
                                f"{base_url}/v1/chat/completions",
                                json=retry_payload
                            )
                            retry_result = retry_resp.json()
                            choice = retry_result['choices'][0]
                            if 'message' in choice:
                                return choice['message']['content'].strip()
                            elif 'text' in choice:
                                return choice['text'].strip()
                    except Exception as trim_err:
                        logger.error(f"Auto-trim retry failed: {trim_err}")
                        # Fall through and re-raise original HTTP error
                raise
            except KeyError as e:
                logger.error(f"KeyError in chat_async response parsing: {e}")
                logger.error(f"Response content: {result if 'result' in locals() else 'response not parsed'}")
                response_text = response.text if 'response' in locals() else None
                logger.error(f"Full response text: {summarize_text(response_text, limit=500) or 'response not available'}")
                raise ValueError(f"Invalid response format from TabbyAPI: KeyError {e}. Check if TabbyAPI is properly configured and running.")
            except Exception as e:
                logger.error(f"Error in chat_async: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                raise

    async def _fetch_server_max_seq_len(self, base_url: str) -> Optional[int]:
        """Fetch model parameters from TabbyAPI and return max_seq_len if available."""
        try:
            client = await self._get_async_client()
            resp = await client.get(f"{base_url}/v1/model")
            resp.raise_for_status()
            data = resp.json()
            params = data.get("parameters") or {}
            msl = params.get("max_seq_len") or params.get("max_seq_length")
            if isinstance(msl, int) and msl > 0:
                return msl
        except Exception as e:
            logger.debug(f"Could not fetch server max_seq_len: {e}")
        # Fallback to detected GPU config heuristic
        try:
            return int(self.gpu_config.get('max_seq_len') or 4096)
        except Exception:
            return 4096

    async def _trim_messages_to_fit(
        self,
        base_url: str,
        messages: List[Dict[str, str]],
        server_error_text: str,
    ) -> Optional[List[Dict[str, str]]]:
        """Attempt to trim messages to fit under server max_seq_len and return new messages.

        Strategy:
        - Keep the last user message and the most recent system message.
        - Drop earlier messages (assistant/older user) first.
        - If still failing, proportionally truncate long contents on a second retry.
        """
        # Parse numbers from server error if available
        prompt_len = None
        max_len = None
        try:
            m1 = re.search(r"Prompt length\s+(\d+)", server_error_text, re.IGNORECASE)
            if m1:
                prompt_len = int(m1.group(1))
            m2 = re.search(r"max_seq_len\s+(\d+)", server_error_text, re.IGNORECASE)
            if m2:
                max_len = int(m2.group(1))
        except Exception:
            pass

        if max_len is None:
            max_len = await self._fetch_server_max_seq_len(base_url)

        if not isinstance(max_len, int) or max_len <= 0:
            return None

        # First pass: drop older context (preserve most-recent system + last user)
        last_system_idx = None
        last_user_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            role = (msg.get("role") or "").lower()
            if last_user_idx is None and role == "user":
                last_user_idx = idx
            if last_system_idx is None and role == "system":
                last_system_idx = idx
            if last_user_idx is not None and last_system_idx is not None:
                break

        preserved_indices = [i for i in (last_system_idx, last_user_idx) if i is not None]
        if not preserved_indices:
            return None

        preserved_indices.sort()
        new_messages: List[Dict[str, str]] = [messages[i] for i in preserved_indices]

        # If server provided prompt_len and we didn't drop enough, attempt truncation
        # Note: Without tokenization, approximate by character count proportional trimming
        if prompt_len and prompt_len > max_len - self._trim_margin_tokens:
            # Proportion to trim (cap to at least 10% reduction)
            over = prompt_len - (max_len - self._trim_margin_tokens)
            # Approx chars-per-token ~ 3.5-4.0, use 4 to be conservative
            over_chars = over * 4
            # Truncate user content preferentially
            try:
                for i, msg in enumerate(new_messages):
                    if (msg.get("role") or "").lower() == "user":
                        content = msg.get("content") or ""
                        if len(content) > over_chars:
                            new_messages[i] = {**msg, "content": content[-(len(content)-over_chars):]}
                        else:
                            # Keep last 1024 chars if extreme
                            new_messages[i] = {**msg, "content": content[-1024:]}
            except Exception:
                pass

        return new_messages

    async def generate_batch_async(
        self,
        model: str,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> List[str]:
        """
        SUPER-BATCHING: Generate multiple completions in a single request

        This is ExLlamaV3's killer feature - 8-16x faster than sequential processing!
        Compatible with LangGraph batch workflows.

        Args:
            model: Model name
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens per generation
            top_p: Nucleus sampling parameter
            top_k: Top-K sampling parameter
            repetition_penalty: Repetition penalty
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts (same order as input prompts)
        """
        if not self.config.enable_batching or len(prompts) == 1:
            # Fallback to sequential if batching disabled or single prompt
            tasks = [
                self.generate_async(model, p, temperature, max_tokens, top_p, top_k, repetition_penalty, **kwargs)
                for p in prompts
            ]
            return await asyncio.gather(*tasks)

        start_time = time.time()

        # Process in batches of configured batch_size
        results = []
        for i in range(0, len(prompts), self.config.batch_size):
            batch_prompts = prompts[i:i + self.config.batch_size]

            # Use semaphore, recreating if needed due to event loop changes
            semaphore = self._get_or_recreate_semaphore()
            async with semaphore:
                client = await self._get_async_client()

                # DETERMINISTIC MODE: When temperature=0, use greedy decoding
                batch_top_k = top_k
                batch_top_p = top_p
                if temperature <= 0.01:
                    batch_top_k = 1
                    batch_top_p = 1.0
                    logger.debug(f"Deterministic mode (batch): temp={temperature}, top_k=1, top_p=1.0")

                # Create batch payload
                payload = {
                    "model": model,  # Specify which model to use
                    "prompts": batch_prompts,  # Multiple prompts
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": batch_top_p,
                    "top_k": batch_top_k,
                    "repetition_penalty": repetition_penalty,
                    "stream": False,
                    **kwargs
                }

                try:
                    # Check if TabbyAPI supports batch endpoint
                    response = await client.post(
                        f"{self.config.base_url}/v1/completions",
                        json=payload
                    )
                    response.raise_for_status()

                    result = response.json()

                    # Extract texts from responses
                    if isinstance(result.get('choices'), list):
                        batch_results = [choice['text'].strip() for choice in result['choices']]
                        results.extend(batch_results)
                    else:
                        # Fallback if batch not supported
                        logger.warning("Batch endpoint not supported, falling back to sequential")
                        sequential_results = await asyncio.gather(*[
                            self.generate_async(model, p, temperature, max_tokens, top_p,
                                                top_k, repetition_penalty, **kwargs)
                            for p in batch_prompts
                        ])
                        results.extend(sequential_results)

                except Exception as e:
                    logger.error(f"Error in batch generation: {e}")
                    # Fallback to sequential on error
                    sequential_results = await asyncio.gather(*[
                        self.generate_async(model, p, temperature, max_tokens, top_p,
                                            top_k, repetition_penalty, **kwargs)
                        for p in batch_prompts
                    ])
                    results.extend(sequential_results)

        # Update stats
        self._stats['batch_requests'] += 1
        self._stats['total_time'] += (time.time() - start_time)

        logger.info(f"Batch generated {len(prompts)} completions in {time.time() - start_time:.2f}s")

        return results

    async def generate_stream_async(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming (real-time token generation)

        Compatible with LangSmith real-time tracing and Rich CLI monitoring.

        Args:
            model: Model name
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-K sampling parameter
            repetition_penalty: Repetition penalty
            **kwargs: Additional generation parameters

        Yields:
            Generated tokens as they are produced
        """
        # Ensure server is running before making request
        await self._ensure_server_running()

        if not self.config.enable_streaming:
            # Fallback to non-streaming
            result = await self.generate_async(model, prompt, temperature, max_tokens, top_p, top_k, repetition_penalty, **kwargs)
            yield result
            return

        # Use semaphore, recreating if needed due to event loop changes
        semaphore = self._get_or_recreate_semaphore()
        async with semaphore:
            client = await self._get_async_client()

            # DETERMINISTIC MODE: When temperature=0, use greedy decoding
            if temperature <= 0.01:
                top_k = 1
                top_p = 1.0
                logger.debug(f"Deterministic mode (stream): temp={temperature}, top_k=1, top_p=1.0")

            payload = {
                "model": model,  # Specify which model to use
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stream": True,  # Enable streaming
                **kwargs
            }

            try:
                async with client.stream(
                    "POST",
                    f"{self.config.base_url}/v1/completions",
                    json=payload
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()

                            if data_str == "[DONE]":
                                break

                            try:
                                import json
                                data = json.loads(data_str)

                                if 'choices' in data and len(data['choices']) > 0:
                                    token = data['choices'][0].get('text', '')
                                    if token:
                                        yield token
                            except json.JSONDecodeError:
                                continue

                # Update stats
                self._stats['stream_requests'] += 1

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error in streaming: {e.response.status_code}")
                raise
            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                raise

    def get_stats(self) -> Dict[str, Any]:
        """Get client performance statistics"""
        avg_time = self._stats['total_time'] / max(self._stats['total_requests'], 1)
        return {
            **self._stats,
            'average_request_time': avg_time,
            'gpu_config': self.gpu_config,
            'batching_enabled': self.config.enable_batching,
            'streaming_enabled': self.config.enable_streaming
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self._stats = {
            'total_requests': 0,
            'batch_requests': 0,
            'stream_requests': 0,
            'total_tokens': 0,
            'total_time': 0.0
        }

    def check_model(self, model_name: str) -> bool:
        """
        Check if a model is available via TabbyAPI

        For ExLlamaV3, models are loaded dynamically by TabbyAPI, so we just verify
        that the TabbyAPI server is accessible on the model-specific port.

        Args:
            model_name: Model name (format: "exl3:model-name")

        Returns:
            True if TabbyAPI server is accessible on the model's port, False otherwise
        """
        try:
            # Get model-specific port and base URL
            base_url = self._get_base_url_for_model(model_name)
            port = self._get_model_port(model_name)

            # Simple synchronous port check
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            host = parsed.hostname or "localhost"
            port_num = parsed.port or 5000

            # Just check if port is open (simplest, most reliable check)
            result = self._is_port_open(host, port_num)

            if result:
                logger.info(f"✓ Model {model_name} available on {base_url}")
            else:
                logger.debug(f"Port {port_num} not open for {model_name}")

            return result

        except Exception as e:
            logger.debug(f"Model check failed for {model_name}: {e}")
            return False


# Global client instance (singleton pattern)
_global_client: Optional[ExLlamaV3Client] = None


def get_exllamav3_client() -> ExLlamaV3Client:
    """
    Get or create global ExLlamaV3 client instance

    Returns:
        ExLlamaV3Client instance
    """
    global _global_client

    if _global_client is None:
        _global_client = ExLlamaV3Client()

    return _global_client


def reset_exllamav3_client():
    """Reset the global client instance (useful for testing)"""
    global _global_client

    if _global_client is not None:
        # Close will be handled by async context or explicit close()
        _global_client = None


# Synchronous wrapper for backwards compatibility
class ExLlamaV3ClientSync:
    """Synchronous wrapper around ExLlamaV3Client for non-async code"""

    def __init__(self, config: Optional[ExLlamaV3Config] = None):
        self.client = ExLlamaV3Client(config)

    def generate(self, model: str, prompt: str, **kwargs) -> str:
        """Synchronous generate (runs async in event loop)"""
        return asyncio.run(self.client.generate_async(model, prompt, **kwargs))

    def chat(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Synchronous chat (runs async in event loop)"""
        return asyncio.run(self.client.chat_async(model, messages, **kwargs))

    def generate_batch(self, model: str, prompts: List[str], **kwargs) -> List[str]:
        """Synchronous batch generate (runs async in event loop)"""
        return asyncio.run(self.client.generate_batch_async(model, prompts, **kwargs))


__all__ = [
    'ExLlamaV3Client',
    'ExLlamaV3ClientSync',
    'ExLlamaV3Config',
    'get_exllamav3_client',
    'reset_exllamav3_client'
]

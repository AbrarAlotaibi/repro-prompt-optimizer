# =====================================
# FILE: openrouter_client.py
# =====================================
"""
OpenRouter client using OpenAI SDK
"""
import os
import logging
from typing import List, Optional
import time
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)

from repro.clients.logging_utils import log_request_summary, log_response_summary, new_request_id

# Load environment variables from .env file (centralized)
def _load_env_file():
    """Load environment variables from .env file if it exists"""
    # Find project root (go up until we find .env or reach filesystem root)
    current_path = Path(__file__).parent
    project_root = None

    # Look for .env file by going up the directory tree
    for parent in [current_path] + list(current_path.parents):
        env_file = parent / ".env"
        if env_file.exists():
            project_root = parent
            break

    if not project_root:
        return  # No .env file found

    env_file = project_root / ".env"

    try:
        # Try python-dotenv first (preferred) - override existing environment
        from dotenv import load_dotenv
        load_dotenv(env_file, override=True)  # Force override existing env vars
        logger.debug(f"Loaded .env file using python-dotenv: {env_file}")
    except ImportError:
        # Manual .env loading fallback
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Override existing environment variables with .env file values
                        os.environ[key.strip()] = value.strip()
            logger.debug(f"Loaded .env file manually: {env_file}")
        except Exception as e:
            logger.warning(f"Failed to load .env file {env_file}: {e}")

# Load .env when module is imported
_load_env_file()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    # Try to import AsyncOpenAI (available in openai>=1.0.0)
    try:
        from openai import AsyncOpenAI
        ASYNC_OPENAI_AVAILABLE = True
    except ImportError:
        ASYNC_OPENAI_AVAILABLE = False
        logger.warning("AsyncOpenAI not available. Update openai: pip install --upgrade 'openai>=1.0.0'")
except ImportError:
    OPENAI_AVAILABLE = False
    ASYNC_OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. Run: pip install openai")

class OpenRouterClient:
    """OpenRouter client using OpenAI SDK with rate limiting and async support"""

    def __init__(self,
                 api_key: Optional[str] = None,
                 timeout: int = 60,
                 max_concurrent: int = 30):  # Increased for async parallel evaluations
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "OpenAI package not installed. Install it with: pip install openai"
            )

        # Get API key from parameter, environment, or .env file
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        # Detailed API key validation
        if not self.api_key:
            raise RuntimeError(
                "OpenRouter API key not found. Please:\n"
                "  1. Set OPENROUTER_API_KEY environment variable, or\n"
                "  2. Add OPENROUTER_API_KEY=your_key to .env file in project root\n"
                "  3. Get API key from: https://openrouter.ai/keys"
            )

        if self.api_key.strip() == "":
            raise RuntimeError("OpenRouter API key is empty")

        # Basic API key format validation
        if not self.api_key.startswith('sk-or-'):
            logger.warning(
                f"OpenRouter API key format may be invalid. Key should typically start with 'sk-or-'\n"
                f"Current key starts with: '{self.api_key[:10]}...'\n"
                f"Get a valid key from: https://openrouter.ai/keys"
            )

        self.timeout = timeout
        self.max_concurrent = max_concurrent

        try:
            # Sync client (for sync methods)
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                timeout=timeout
            )

            # Async client (for async methods) - persistent connection pool
            if ASYNC_OPENAI_AVAILABLE:
                import httpx
                # Create httpx client with increased connection limits
                http_client = httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_keepalive_connections=50,
                        max_connections=100,
                        keepalive_expiry=30.0
                    ),
                    timeout=httpx.Timeout(timeout, connect=10.0)
                )
                self._async_client = AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key,
                    timeout=timeout,
                    max_retries=3,  # Increased retries
                    http_client=http_client
                )
            else:
                self._async_client = None
                logger.warning("AsyncOpenAI not available - async methods will use sync fallback")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenRouter client: {e}")

        # Rate limiting semaphore
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Popular models on OpenRouter (can be expanded)
        self.available_models = [
            # Qwen models
            "qwen/qwen3-235b-a22b-2507",
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen-2-72b-instruct",
            # Claude models
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-haiku",
            # GPT models
            "openai/gpt-4-turbo",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-3.5-turbo",
            # Gemini models
            "google/gemini-pro-1.5",
            "google/gemini-flash-1.5",
            # Llama models
            "meta-llama/llama-3.1-405b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            # Mistral models
            "mistralai/mistral-large",
            "mistralai/mistral-medium",
            "mistralai/mixtral-8x7b-instruct",
        ]

        logger.info(f"✅ OpenRouter client initialized: max_concurrent={max_concurrent}, timeout={timeout}s")

    def generate(self,
                 model: str,
                 prompt: str,
                 temperature: float = 0.7,
                 max_tokens: int = 256,
                 **kwargs) -> str:
        """Generate response from OpenRouter model"""

        # Filter out Ollama-specific and non-standard parameters that OpenAI doesn't support
        openai_params = {}
        unsupported_params = {'additional_params', 'num_predict', 'top_k', 'repeat_penalty', 'num_ctx', 'ban_eos_token', 'min_tokens'}

        for key, value in kwargs.items():
            if key not in unsupported_params:
                openai_params[key] = value

        # Make request with retries
        for attempt in range(3):
            try:
                request_id = new_request_id()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"reasoning": {"type": "disabled"}},  # Disable thinking mode for reasoning models
                    **openai_params
                )

                log_request_summary(logger, "OpenRouter", prompt=prompt, request_id=request_id)
                log_response_summary(logger, "OpenRouter", response, request_id=request_id)
                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.error(f"OpenRouter generation error (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError(f"Failed to generate after 3 attempts")

    def chat(self,
             model: str,
             messages: List[dict],
             temperature: float = 0.7,
             max_tokens: int = 256,
             response_format: Optional[dict] = None,
             **kwargs) -> str:
        """Chat completion using OpenRouter chat API"""

        # Filter out Ollama-specific and non-standard parameters
        openai_params = {}
        unsupported_params = {'additional_params', 'num_predict', 'top_k', 'repeat_penalty', 'num_ctx', 'ban_eos_token', 'min_tokens'}

        for key, value in kwargs.items():
            if key not in unsupported_params:
                openai_params[key] = value

        # Add response_format if provided (OpenAI Structured Outputs)
        if response_format is not None:
            openai_params["response_format"] = response_format

        # Make request with retries
        for attempt in range(3):
            try:
                request_id = new_request_id()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"reasoning": {"type": "disabled"}},  # Disable thinking mode for reasoning models
                    **openai_params
                )

                log_request_summary(logger, "OpenRouter", messages=messages, request_id=request_id)
                log_response_summary(logger, "OpenRouter", response, request_id=request_id)
                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.error(f"OpenRouter chat error (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError(f"Failed to chat after 3 attempts")

    async def generate_async(self,
                             model: str,
                             prompt: str,
                             temperature: float = 0.7,
                             max_tokens: int = 256,
                             **kwargs) -> str:
        """True async generate with rate limiting and persistent connection pool."""
        # Fall back to sync if async client not available
        if self._async_client is None:
            logger.debug("AsyncOpenAI not available, using sync fallback")
            return await asyncio.to_thread(
                self.generate, model, prompt, temperature, max_tokens, **kwargs
            )

        # Filter out Ollama-specific and non-standard parameters
        openai_params = {}
        unsupported_params = {'additional_params', 'num_predict', 'top_k', 'repeat_penalty', 'num_ctx', 'ban_eos_token', 'min_tokens'}

        for key, value in kwargs.items():
            if key not in unsupported_params:
                openai_params[key] = value

        # Use rate limiting semaphore with retry logic
        max_retries = 3
        base_delay = 1.0

        async with self._semaphore:
            for attempt in range(max_retries):
                try:
                    request_id = new_request_id()
                    response = await self._async_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_body={"reasoning": {"type": "disabled"}},  # Disable thinking mode for reasoning models
                        **openai_params
                    )

                    log_request_summary(logger, "OpenRouter", prompt=prompt, request_id=request_id)
                    log_response_summary(logger, "OpenRouter", response, request_id=request_id)
                    return response.choices[0].message.content.strip()

                except Exception as e:
                    error_msg = str(e).lower()
                    # Check if it's a retryable error
                    is_retryable = any(keyword in error_msg for keyword in [
                        'connection', 'timeout', 'rate limit', 'overload', 'unavailable'
                    ])

                    if attempt < max_retries - 1 and is_retryable:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"OpenRouter async error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"OpenRouter async generation error: {e}")
                        raise RuntimeError(f"Failed to generate with {model}: {e}")

    async def chat_async(self,
                        model: str,
                        messages: List[dict],
                        temperature: float = 0.7,
                        max_tokens: int = 256,
                        response_format: Optional[dict] = None,
                        **kwargs) -> str:
        """True async chat with rate limiting and persistent connection pool."""
        # Fall back to sync if async client not available
        if self._async_client is None:
            logger.debug("AsyncOpenAI not available, using sync fallback")
            fallback_kwargs = dict(kwargs)
            if response_format is not None:
                fallback_kwargs["response_format"] = response_format

            return await asyncio.to_thread(
                self.chat, model, messages, temperature, max_tokens, **fallback_kwargs
            )

        # Filter out Ollama-specific and non-standard parameters
        openai_params = {}
        unsupported_params = {'additional_params', 'num_predict', 'top_k', 'repeat_penalty', 'num_ctx', 'ban_eos_token', 'min_tokens'}

        for key, value in kwargs.items():
            if key not in unsupported_params:
                openai_params[key] = value

        # Add response_format if provided (OpenAI Structured Outputs)
        if response_format is not None:
            openai_params["response_format"] = response_format

        # Use rate limiting semaphore with retry logic
        max_retries = 3
        base_delay = 1.0

        async with self._semaphore:
            for attempt in range(max_retries):
                try:
                    request_id = new_request_id()
                    response = await self._async_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_body={"reasoning": {"type": "disabled"}},  # Disable thinking mode for reasoning models
                        **openai_params
                    )

                    log_request_summary(logger, "OpenRouter", messages=messages, request_id=request_id)
                    log_response_summary(logger, "OpenRouter", response, request_id=request_id)
                    return response.choices[0].message.content.strip()

                except Exception as e:
                    error_msg = str(e).lower()
                    # Check if it's a retryable error
                    is_retryable = any(keyword in error_msg for keyword in [
                        'connection', 'timeout', 'rate limit', 'overload', 'unavailable'
                    ])

                    if attempt < max_retries - 1 and is_retryable:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"OpenRouter async chat error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"OpenRouter async chat error: {e}")
                        raise RuntimeError(f"Failed to chat with {model}: {e}")

    async def close(self):
        """Close persistent async client (disabled to avoid event loop issues)."""
        # Note: Not closing the async client to avoid "Event loop is closed" errors
        # The connection pool will be cleaned up when the process exits
        pass

    def check_model(self, model_name: str) -> bool:
        """Check if model is available in the static list"""
        return model_name in self.available_models

# Global client instance
_openrouter_client = None

def get_openrouter_client() -> OpenRouterClient:
    """Get or create OpenRouter client"""
    global _openrouter_client
    if _openrouter_client is None:
        from repro.core.config import SYSTEM_CONFIG
        try:
            _openrouter_client = OpenRouterClient(
                api_key=getattr(SYSTEM_CONFIG, 'openrouter_api_key', None),
                timeout=getattr(SYSTEM_CONFIG, 'openrouter_timeout', 60)
            )
        except RuntimeError as e:
            if "API key not found" in str(e):
                raise RuntimeError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable")
            else:
                raise RuntimeError(f"Failed to connect to OpenRouter API: {e}")
    return _openrouter_client

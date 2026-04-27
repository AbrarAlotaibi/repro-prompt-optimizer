# =====================================
# FILE: openai_client.py
# =====================================
"""
OpenAI client that mirrors the Ollama client interface
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

class OpenAIClient:
    """OpenAI client with rate limiting and async support"""

    def __init__(self,
                 api_key: Optional[str] = None,
                 timeout: int = 60,
                 max_concurrent: int = 30):  # Increased for async parallel evaluations
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "OpenAI package not installed. Install it with: pip install openai"
            )

        # Get API key from parameter, environment, or .env file
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Detailed API key validation
        if not self.api_key:
            raise RuntimeError(
                "OpenAI API key not found. Please:\n"
                "  1. Set OPENAI_API_KEY environment variable, or\n"
                "  2. Add OPENAI_API_KEY=your_key to .env file in project root\n"
                "  3. Get API key from: https://platform.openai.com/api-keys"
            )

        if self.api_key.strip() == "":
            raise RuntimeError("OpenAI API key is empty")

        # Basic API key format validation
        if not self.api_key.startswith(('sk-', 'sk-proj-')):
            raise RuntimeError(
                f"Invalid OpenAI API key format. Key should start with 'sk-' or 'sk-proj-'\n"
                f"Current key starts with: '{self.api_key[:10]}...'\n"
                f"Get a valid key from: https://platform.openai.com/api-keys"
            )

        self.timeout = timeout
        self.max_concurrent = max_concurrent

        try:
            # Sync client (for sync methods)
            self.client = OpenAI(api_key=self.api_key, timeout=timeout)

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
                    api_key=self.api_key,
                    timeout=timeout,
                    max_retries=3,  # Increased retries
                    http_client=http_client
                )
            else:
                self._async_client = None
                logger.warning("AsyncOpenAI not available - async methods will use sync fallback")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

        # Rate limiting semaphore (OpenAI has stricter limits)
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Optional connection test (non-fatal). Some environments block models.list.
        try:
            self._check_openai()
        except Exception as e:
            logger.warning(f"Skipping OpenAI connectivity check: {e}")

        # Available models (static list of commonly used models)
        self.available_models = [
            # GPT-5 models (future)
            "gpt-5",
            "gpt-5-2025-08-07",
            "gpt-5-chat-latest",
            "gpt-5-codex",
            "gpt-5-mini",
            "gpt-5-mini-2025-08-07",
            "gpt-5-nano",
            # GPT-4.1 models
            "gpt-4.1-mini",
            "gpt-4.1-mini-2025-04-14",
            "gpt-4.1",
            "gpt-4.1-2025-04-14",
            "gpt-4.1-nano",
            # GPT-4o models
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o",
            "gpt-4o-2024-11-20",
            "chatgpt-4o-latest",
            # GPT-4 models
            "gpt-4",
            "gpt-4-0125-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",
            # GPT-3.5 models
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106"
        ]

        logger.info(f"✅ OpenAI client initialized: max_concurrent={max_concurrent}, timeout={timeout}s")
    
    def _check_openai(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            # Simple test call with retry logic
            import time
            for attempt in range(3):
                try:
                    models = self.client.models.list()
                    # Just checking if we can list models, don't need to process them
                    logger.debug(f"OpenAI API check successful on attempt {attempt + 1}")
                    return True
                except Exception as e:
                    if attempt < 2:  # Only retry first 2 attempts
                        logger.debug(f"OpenAI API check attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(0.5)  # Brief delay before retry
                    else:
                        raise e
            return False
        except Exception as e:
            logger.warning(f"OpenAI API check failed after 3 attempts: {e}")
            logger.info("This is not fatal - OpenAI client will still work for actual requests")
            return False
    
    def generate(self,
                 model: str,
                 prompt: str,
                 temperature: float = 0.7,
                 max_tokens: int = 256,
                 **kwargs) -> str:
        """Generate response from OpenAI model"""

        # Filter out Ollama-specific parameters that OpenAI doesn't support
        openai_params = {}
        unsupported_params = {'additional_params', 'num_predict', 'top_k', 'repeat_penalty', 'num_ctx'}

        for key, value in kwargs.items():
            if key not in unsupported_params:
                openai_params[key] = value

        # Reasoning models (gpt-5, o1, o3, o4) don't support temperature - use default (1.0)
        # They also need more tokens to accommodate reasoning + response (~500 reasoning + actual output)
        # GPT-4.1 supports temperature but needs max_completion_tokens
        is_reasoning_model = any(model.startswith(p) for p in ["gpt-5", "o1", "o3", "o4"])
        use_new_token_param = any(model.startswith(p) for p in ["gpt-5", "gpt-4.1", "o1", "o3", "o4"])

        # Boost token limit for reasoning models - they use tokens for internal reasoning FIRST
        # GPT-5-mini can use 500-2000+ reasoning tokens before producing any output
        # If limit is too low, ALL tokens go to reasoning → empty response
        # Minimum 8192 ensures room for both reasoning (~2000-4000) and output (~4000)
        # GPT-5-nano uses significant tokens for reasoning before producing output
        effective_max_tokens = max(max_tokens * 4, 8192) if is_reasoning_model else max_tokens
        token_param = {"max_completion_tokens": effective_max_tokens} if use_new_token_param else {"max_tokens": max_tokens}
        temp_param = {} if is_reasoning_model else {"temperature": temperature}
        # Use low reasoning effort for faster responses (reduces reasoning tokens from ~2000 to ~200)
        reasoning_param = {"reasoning_effort": "low"} if is_reasoning_model else {}

        # Make request with retries
        for attempt in range(3):
            try:
                request_id = new_request_id()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout,
                    **temp_param,
                    **token_param,
                    **reasoning_param,
                    **openai_params
                )
                log_request_summary(logger, "OpenAI", prompt=prompt, request_id=request_id)
                log_response_summary(logger, "OpenAI", response, request_id=request_id)
                content = response.choices[0].message.content
                return content.strip() if content else ""

            except Exception as e:
                logger.error(f"OpenAI generation error (attempt {attempt + 1}): {e}")
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
        """Chat completion using OpenAI chat API"""

        # Filter out Ollama-specific parameters
        openai_params = {}
        unsupported_params = {'additional_params', 'num_predict', 'top_k', 'repeat_penalty', 'num_ctx'}

        for key, value in kwargs.items():
            if key not in unsupported_params:
                openai_params[key] = value

        # Add response_format if provided (OpenAI Structured Outputs)
        if response_format is not None:
            openai_params["response_format"] = response_format

        # Reasoning models (gpt-5, o1, o3, o4) don't support temperature - use default (1.0)
        # They also need more tokens to accommodate reasoning + response
        # GPT-4.1 supports temperature but needs max_completion_tokens
        is_reasoning_model = any(model.startswith(p) for p in ["gpt-5", "o1", "o3", "o4"])
        use_new_token_param = any(model.startswith(p) for p in ["gpt-5", "gpt-4.1", "o1", "o3", "o4"])

        # Boost token limit for reasoning models - triple tokens, minimum 2048
        effective_max_tokens = max(max_tokens * 4, 4096) if is_reasoning_model else max_tokens
        token_param = {"max_completion_tokens": effective_max_tokens} if use_new_token_param else {"max_tokens": max_tokens}
        temp_param = {} if is_reasoning_model else {"temperature": temperature}
        # Use low reasoning effort for faster responses
        reasoning_param = {"reasoning_effort": "low"} if is_reasoning_model else {}

        # Make request with retries
        for attempt in range(3):
            try:
                request_id = new_request_id()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=self.timeout,
                    **temp_param,
                    **token_param,
                    **reasoning_param,
                    **openai_params
                )
                log_request_summary(logger, "OpenAI", messages=messages, request_id=request_id)
                log_response_summary(logger, "OpenAI", response, request_id=request_id)
                content = response.choices[0].message.content
                return content.strip() if content else ""

            except Exception as e:
                logger.error(f"OpenAI chat error (attempt {attempt + 1}): {e}")
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

        # Filter out Ollama-specific parameters
        openai_params = {}
        unsupported_params = {'additional_params', 'num_predict', 'top_k', 'repeat_penalty', 'num_ctx'}

        for key, value in kwargs.items():
            if key not in unsupported_params:
                openai_params[key] = value

        # Reasoning models (gpt-5, o1, o3, o4) don't support temperature - use default (1.0)
        # They also need more tokens to accommodate reasoning + response
        # GPT-4.1 supports temperature but needs max_completion_tokens
        is_reasoning_model = any(model.startswith(p) for p in ["gpt-5", "o1", "o3", "o4"])
        use_new_token_param = any(model.startswith(p) for p in ["gpt-5", "gpt-4.1", "o1", "o3", "o4"])

        # Boost token limit for reasoning models - triple tokens, minimum 2048
        effective_max_tokens = max(max_tokens * 4, 4096) if is_reasoning_model else max_tokens
        token_param = {"max_completion_tokens": effective_max_tokens} if use_new_token_param else {"max_tokens": max_tokens}
        temp_param = {} if is_reasoning_model else {"temperature": temperature}
        # Use low reasoning effort for faster responses
        reasoning_param = {"reasoning_effort": "low"} if is_reasoning_model else {}

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
                        **temp_param,
                        **token_param,
                        **reasoning_param,
                        **openai_params
                    )

                    log_request_summary(logger, "OpenAI", prompt=prompt, request_id=request_id)
                    log_response_summary(logger, "OpenAI", response, request_id=request_id)
                    content = response.choices[0].message.content
                    return content.strip() if content else ""

                except Exception as e:
                    error_msg = str(e).lower()
                    # Check if it's a retryable error
                    is_retryable = any(keyword in error_msg for keyword in [
                        'connection', 'timeout', 'rate limit', 'overload', 'unavailable'
                    ])

                    if attempt < max_retries - 1 and is_retryable:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"OpenAI async error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"OpenAI async generation error: {e}")
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

        # Filter out Ollama-specific parameters
        openai_params = {}
        unsupported_params = {'additional_params', 'num_predict', 'top_k', 'repeat_penalty', 'num_ctx'}

        for key, value in kwargs.items():
            if key not in unsupported_params:
                openai_params[key] = value

        # Add response_format if provided (OpenAI Structured Outputs)
        if response_format is not None:
            openai_params["response_format"] = response_format

        # Reasoning models (gpt-5, o1, o3, o4) don't support temperature - use default (1.0)
        # They also need more tokens to accommodate reasoning + response
        # GPT-4.1 supports temperature but needs max_completion_tokens
        is_reasoning_model = any(model.startswith(p) for p in ["gpt-5", "o1", "o3", "o4"])
        use_new_token_param = any(model.startswith(p) for p in ["gpt-5", "gpt-4.1", "o1", "o3", "o4"])

        # Boost token limit for reasoning models - triple tokens, minimum 2048
        effective_max_tokens = max(max_tokens * 4, 4096) if is_reasoning_model else max_tokens
        token_param = {"max_completion_tokens": effective_max_tokens} if use_new_token_param else {"max_tokens": max_tokens}
        temp_param = {} if is_reasoning_model else {"temperature": temperature}
        # Use low reasoning effort for faster responses
        reasoning_param = {"reasoning_effort": "low"} if is_reasoning_model else {}

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
                        **temp_param,
                        **token_param,
                        **reasoning_param,
                        **openai_params
                    )

                    log_request_summary(logger, "OpenAI", messages=messages, request_id=request_id)
                    log_response_summary(logger, "OpenAI", response, request_id=request_id)
                    content = response.choices[0].message.content
                    return content.strip() if content else ""

                except Exception as e:
                    error_msg = str(e).lower()
                    # Check if it's a retryable error
                    is_retryable = any(keyword in error_msg for keyword in [
                        'connection', 'timeout', 'rate limit', 'overload', 'unavailable'
                    ])

                    if attempt < max_retries - 1 and is_retryable:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"OpenAI async chat error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"OpenAI async chat error: {e}")
                        raise RuntimeError(f"Failed to chat with {model}: {e}")

    async def close(self):
        """Close persistent async client (disabled to avoid event loop issues)."""
        # Note: Not closing the async client to avoid "Event loop is closed" errors
        # The connection pool will be cleaned up when the process exits
        pass
    
    def check_model(self, model_name: str) -> bool:
        """Check if model is available"""
        return model_name in self.available_models

# Global client instance
_openai_client = None

def get_openai_client() -> OpenAIClient:
    """Get or create OpenAI client"""
    global _openai_client
    if _openai_client is None:
        from repro.core.config import SYSTEM_CONFIG
        try:
            _openai_client = OpenAIClient(
                api_key=SYSTEM_CONFIG.openai_api_key,
                timeout=SYSTEM_CONFIG.openai_timeout
            )
        except RuntimeError as e:
            if "API key not found" in str(e):
                raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY environment variable")
            else:
                raise RuntimeError(f"Failed to connect to OpenAI API: {e}")
    return _openai_client 

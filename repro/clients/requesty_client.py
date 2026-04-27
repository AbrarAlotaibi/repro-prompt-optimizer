"""
Requesty client - OpenAI-compatible API gateway for 300+ LLM models.

Requesty provides unified access to models from OpenAI, Anthropic, Google,
Mistral, and more through a single API endpoint.

Usage:
    client = RequestyClient()
    response = client.chat(
        model="mistralai/mistral-small-latest",
        messages=[{"role": "user", "content": "Hello!"}]
    )

Model naming: Use provider prefix format, e.g.:
    - mistralai/mistral-small-latest
    - mistralai/mistral-small-3.1-24b-instruct-2503
    - openai/gpt-4o-mini
    - anthropic/claude-sonnet-4-20250514
    - google/gemini-2.0-flash
"""
import os
import logging
import time
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Load environment variables from .env file
def _load_env_file():
    """Load environment variables from .env file if it exists"""
    current_path = Path(__file__).parent
    for parent in [current_path] + list(current_path.parents):
        env_file = parent / ".env"
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file, override=True)
            except ImportError:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
            break

_load_env_file()

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. Run: pip install openai")

# Default Requesty configuration
DEFAULT_BASE_URL = "https://router.requesty.ai/v1"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_CONCURRENT = 20

# Common models available on Requesty (tested working)
# Note: Requesty uses "mistral/" not "mistralai/" as provider prefix
AVAILABLE_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "mistral/mistral-small-latest",
    "mistral/mistral-medium-latest",
    "mistral/mistral-large-latest",
]

# Model aliases for convenience
# Maps short names to full provider/model format
MODEL_ALIASES = {
    # OpenAI models (work without prefix too)
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4": "openai/gpt-4",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    # Mistral models - use "mistral/" not "mistralai/"
    "mistral-small": "mistral/mistral-small-latest",
    "mistral-small-latest": "mistral/mistral-small-latest",
    "mistral-medium": "mistral/mistral-medium-latest",
    "mistral-large": "mistral/mistral-large-latest",
}


# Reasoning models that use internal tokens before generating output
# These models need higher max_tokens to produce visible output
REASONING_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3",
    "o3-mini",
]


def _is_reasoning_model(model: str) -> bool:
    """Check if model uses reasoning tokens."""
    model_lower = model.lower()
    return any(rm in model_lower for rm in REASONING_MODELS)


class RequestyClient:
    """
    Requesty client using OpenAI-compatible API.

    Requesty is a unified LLM gateway providing access to 300+ models
    from various providers through a single endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ):
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "OpenAI package required for Requesty client. Install: pip install openai"
            )

        # Get API key
        self.api_key = api_key or os.getenv("REQUESTY_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Requesty API key not found. Set REQUESTY_API_KEY in .env or environment.\n"
                "Get your key from: https://app.requesty.ai/"
            )

        # Validate API key format
        if not self.api_key.startswith("rqsty-"):
            logger.warning(
                f"Requesty API key may be invalid. Expected format: rqsty-sk-...\n"
                f"Current key starts with: {self.api_key[:10]}..."
            )

        # Get base URL
        self.base_url = base_url or os.getenv("REQUESTY_BASE_URL", DEFAULT_BASE_URL)
        self.timeout = timeout
        self.max_concurrent = max_concurrent

        # Initialize OpenAI clients pointing to Requesty
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout,
            )

            # Async client
            import httpx
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_keepalive_connections=30,
                    max_connections=50,
                    keepalive_expiry=30.0
                ),
                timeout=httpx.Timeout(timeout, connect=10.0)
            )
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout,
                max_retries=3,
                http_client=http_client,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Requesty client: {e}")

        # Rate limiting
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            f"✅ Requesty client initialized: base_url={self.base_url}, "
            f"max_concurrent={max_concurrent}, timeout={timeout}s"
        )

    def _resolve_model(self, model: str) -> str:
        """Resolve model alias to full provider/model format."""
        # Check aliases first
        if model in MODEL_ALIASES:
            resolved = MODEL_ALIASES[model]
            logger.debug(f"Resolved model alias: {model} -> {resolved}")
            return resolved

        # If no provider prefix, assume mistralai for mistral models
        if "/" not in model:
            if "mistral" in model.lower():
                resolved = f"mistralai/{model}"
                logger.debug(f"Added provider prefix: {model} -> {resolved}")
                return resolved
            # Default warning for unknown models
            logger.warning(
                f"Model '{model}' has no provider prefix. "
                f"Requesty requires format: provider/model-name"
            )

        return model

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Generate response from model via Requesty."""
        model = self._resolve_model(model)

        # Boost max_tokens for reasoning models that use internal reasoning tokens
        # Gemini 2.5 Pro uses ~1700 reasoning tokens, so we need 8192+ for output
        is_reasoning = _is_reasoning_model(model)
        effective_max_tokens = max(max_tokens * 4, 8192) if is_reasoning else max_tokens
        if is_reasoning and effective_max_tokens != max_tokens:
            logger.debug(f"Boosted max_tokens from {max_tokens} to {effective_max_tokens} for reasoning model {model}")

        # Filter unsupported parameters
        supported_params = {}
        unsupported = {'additional_params', 'num_predict', 'top_k', 'repeat_penalty', 'num_ctx', 'ban_eos_token', 'min_tokens'}
        for key, value in kwargs.items():
            if key not in unsupported:
                supported_params[key] = value

        # Use low reasoning effort for faster responses (reduces ~1700 tokens to ~200)
        if is_reasoning:
            supported_params["reasoning_effort"] = "low"

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                    **supported_params,
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""

            except Exception as e:
                logger.warning(f"Requesty generation error (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"Requesty generation failed after 3 attempts")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Chat completion via Requesty."""
        model = self._resolve_model(model)

        # Boost max_tokens for reasoning models
        is_reasoning = _is_reasoning_model(model)
        effective_max_tokens = max(max_tokens * 4, 8192) if is_reasoning else max_tokens

        # Filter unsupported parameters
        supported_params = {}
        unsupported = {'additional_params', 'num_predict', 'top_k', 'repeat_penalty', 'num_ctx', 'ban_eos_token', 'min_tokens'}
        for key, value in kwargs.items():
            if key not in unsupported:
                supported_params[key] = value

        # Use low reasoning effort for faster responses
        if is_reasoning:
            supported_params["reasoning_effort"] = "low"

        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                    **supported_params,
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""

            except Exception as e:
                logger.warning(f"Requesty chat error (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"Requesty chat failed after {max_attempts} attempts")

    async def chat_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Async chat completion via Requesty."""
        model = self._resolve_model(model)

        # Boost max_tokens for reasoning models
        is_reasoning = _is_reasoning_model(model)
        effective_max_tokens = max(max_tokens * 4, 8192) if is_reasoning else max_tokens

        # Filter unsupported parameters
        supported_params = {}
        unsupported = {'additional_params', 'num_predict', 'top_k', 'repeat_penalty', 'num_ctx', 'ban_eos_token', 'min_tokens'}
        for key, value in kwargs.items():
            if key not in unsupported:
                supported_params[key] = value

        # Use low reasoning effort for faster responses
        if is_reasoning:
            supported_params["reasoning_effort"] = "low"

        async with self._semaphore:
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    response = await self._async_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=effective_max_tokens,
                        **supported_params,
                    )
                    content = response.choices[0].message.content
                    return content.strip() if content else ""

                except Exception as e:
                    logger.warning(f"Requesty async chat error (attempt {attempt + 1}/{max_attempts}): {e}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Requesty async chat failed after {max_attempts} attempts")

    async def generate_async(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Async generate via Requesty."""
        return await self.chat_async(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def complete_async(
        self,
        prompt_or_messages,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Compatibility shim: delegates to chat_async or generate_async."""
        # Drop unsupported 'timeout' kwarg (asyncio.wait_for handles timeouts externally)
        kwargs.pop("timeout", None)

        if isinstance(prompt_or_messages, list):
            return await self.chat_async(
                model=model or self._resolve_model(""),
                messages=prompt_or_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        else:
            return await self.generate_async(
                model=model or self._resolve_model(""),
                prompt=str(prompt_or_messages),
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

    def list_models(self) -> List[str]:
        """Return list of commonly available models on Requesty."""
        return AVAILABLE_MODELS + list(MODEL_ALIASES.keys())

    def is_available(self, model: str) -> bool:
        """Check if a model is available (always True for Requesty - it routes dynamically)."""
        return True

    async def close(self):
        """Close async client connections."""
        if hasattr(self, '_async_client') and self._async_client:
            await self._async_client.close()


# Convenience function to create client
def get_requesty_client(**kwargs) -> RequestyClient:
    """Create and return a Requesty client."""
    return RequestyClient(**kwargs)

"""Target model wrappers for REPRO."""

import logging
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, List

from repro.clients.ollama_client import OllamaClient
from repro.clients.openai_client import OpenAIClient
from repro.clients.exllamav3_client import ExLlamaV3Client
from repro.clients.vllm_client import VLLMClient, VLLMConfig
from repro.clients.openrouter_client import OpenRouterClient
from repro.clients.transformers_client import TransformersClient
from repro.clients.requesty_client import RequestyClient

from repro.core.config import SYSTEM_CONFIG, SYSTEM_PROMPT_MODE, CUSTOM_SYSTEM_PROMPT, LOG_FORMATTED_PROMPTS
from repro.core.prompts.system_prompts import SystemPromptConfig
from repro.algorithm.models import Conversation

logger = logging.getLogger(__name__)


class TargetModel(ABC):
    """Base class for all target model implementations."""
    
    def __init__(self, model_name: str, system_prompt: Optional[str] = None, options: Optional[Dict] = None):
        self.model_name = model_name
        self.options = options or {}
        self.client = None  # Set by subclass
        
        if system_prompt is None:
            system_prompt = SystemPromptConfig.get_prompt(
                mode=SYSTEM_PROMPT_MODE,
                model_name=model_name,
                custom=CUSTOM_SYSTEM_PROMPT
            )
        self.system_prompt = system_prompt
    
    def _build_messages(self, conv: Conversation, include_system_in_messages: bool = True) -> List[Dict[str, str]]:
        """Build message list for API call."""
        messages = []
        if self.system_prompt and include_system_in_messages:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": conv.user})
        return messages
    
    def _log_request(self, name: str, messages: List[Dict], is_async: bool = False):
        """Log formatted messages if debug logging enabled."""
        if LOG_FORMATTED_PROMPTS:
            async_marker = "Async " if is_async else ""
            logger.debug(f"[{self.__class__.__name__}:{name}] {async_marker}formatted messages:\n{json.dumps(messages, indent=2)}")
    
    def _log_init(self, provider_name: str):
        """Log initialization info."""
        logger.info(f"{provider_name} Target initialized: {self.model_name}")
        if self.system_prompt:
            logger.info(f"  System prompt mode: {SYSTEM_PROMPT_MODE} (length: {len(self.system_prompt)} chars)")
        else:
            logger.info(f"  System prompt mode: {SYSTEM_PROMPT_MODE} (empty - testing base model)")
    
    @abstractmethod
    def run(self, conv: Conversation, model_name: Optional[str] = None, 
            temperature: float = 0.0, timeout: int = 60) -> str:
        """Synchronous inference. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    async def run_async(self, conv: Conversation, model_name: Optional[str] = None,
                        temperature: float = 0.0, timeout: int = 60) -> str:
        """Async inference. Must be implemented by subclass."""
        pass


class OpenAIModel(TargetModel):
    """OpenAI model for target testing."""
    
    def __init__(self, model_name: str, system_prompt: Optional[str] = None, options: Optional[Dict] = None):
        super().__init__(model_name, system_prompt, options)
        self.client = OpenAIClient(api_key=SYSTEM_CONFIG.openai_api_key, timeout=SYSTEM_CONFIG.openai_timeout)
        self._log_init("OpenAI")
    
    def run(self, conv: Conversation, model_name: Optional[str] = None, 
            temperature: float = 0.0, timeout: int = 60) -> str:
        name = model_name or self.model_name
        messages = self._build_messages(conv)
        self._log_request(name, messages)
        
        original_timeout = self.client.timeout
        self.client.timeout = timeout
        try:
            return self.client.chat(model=name, messages=messages, temperature=temperature,
                                    max_tokens=self.options.get("max_tokens", 250)) or ""
        except Exception as e:
            return f"[ERROR calling OpenAI: {e}]"
        finally:
            self.client.timeout = original_timeout
    
    async def run_async(self, conv: Conversation, model_name: Optional[str] = None,
                        temperature: float = 0.0, timeout: int = 60) -> str:
        name = model_name or self.model_name
        messages = self._build_messages(conv)
        self._log_request(name, messages, is_async=True)
        
        original_timeout = self.client.timeout
        self.client.timeout = timeout
        try:
            return await self.client.chat_async(model=name, messages=messages, temperature=temperature,
                                                max_tokens=self.options.get("max_tokens", 250)) or ""
        except Exception as e:
            return f"[ERROR calling OpenAI: {e}]"
        finally:
            self.client.timeout = original_timeout


class OllamaModel(TargetModel):
    """Ollama model with GPU support."""
    
    def __init__(self, model_name: str, system_prompt: Optional[str] = None, options: Optional[Dict] = None):
        super().__init__(model_name, system_prompt, options)
        self.base_url = SYSTEM_CONFIG.ollama_url
        self.client = OllamaClient(base_url=self.base_url, timeout=SYSTEM_CONFIG.ollama_timeout)
        self._verify_gpu()
        self._log_init("Ollama")
        if self.system_prompt:
            logger.info("  System prompt will override Modelfile")
    
    def _verify_gpu(self):
        """Verify GPU availability."""
        try:
            if SYSTEM_CONFIG.enable_gpu:
                logger.info(f"  GPU enabled: {SYSTEM_CONFIG.gpu_layers} layers")
            else:
                logger.warning("  WARNING: GPU not enabled, using CPU (will be slow)")
        except Exception as e:
            logger.warning(f"  Could not verify GPU: {e}")
    
    def run(self, conv: Conversation, model_name: Optional[str] = None,
            temperature: float = 0.0, timeout: int = 60) -> str:
        name = model_name or self.model_name
        messages = self._build_messages(conv)
        self._log_request(name, messages)
        
        original_timeout = self.client.timeout
        self.client.timeout = timeout
        try:
            return self.client.chat(model=name, messages=messages, temperature=temperature,
                                    max_tokens=self.options.get("num_predict", 512)) or ""
        except Exception as e:
            return f"[ERROR calling Ollama: {e}]"
        finally:
            self.client.timeout = original_timeout
    
    async def run_async(self, conv: Conversation, model_name: Optional[str] = None,
                        temperature: float = 0.0, timeout: int = 60) -> str:
        name = model_name or self.model_name
        messages = self._build_messages(conv)
        self._log_request(name, messages, is_async=True)
        
        original_timeout = self.client.timeout
        self.client.timeout = timeout
        try:
            return await self.client.chat_async(model=name, messages=messages, temperature=temperature,
                                                max_tokens=self.options.get("num_predict", 512)) or ""
        except Exception as e:
            return f"[ERROR calling Ollama: {e}]"
        finally:
            self.client.timeout = original_timeout


class ExLlamaV3Model(TargetModel):
    """ExLlamaV3 model using TabbyAPI server."""
    
    def __init__(self, model_name: str, system_prompt: Optional[str] = None, options: Optional[Dict] = None):
        super().__init__(model_name, system_prompt, options)
        self.client = ExLlamaV3Client()
        logger.info(f"ExLlamaV3 Target initialized: {model_name} (TabbyAPI)")
        logger.info(f"  Target URL: {self.client.config.base_url}")
        logger.info(f"  API Key: {'SET' if self.client.config.api_key else 'NOT SET'}")
        if self.system_prompt:
            logger.info(f"  System prompt mode: {SYSTEM_PROMPT_MODE} (length: {len(self.system_prompt)} chars)")
        else:
            logger.info(f"  System prompt mode: {SYSTEM_PROMPT_MODE} (empty - testing base model)")
    
    def run(self, conv: Conversation, model_name: Optional[str] = None,
            temperature: float = 0.0, timeout: int = 60) -> str:
        # ExLlamaV3 is async-only
        return "[ERROR: ExLlamaV3 requires async - use run_async]"
    
    async def run_async(self, conv: Conversation, model_name: Optional[str] = None,
                        temperature: float = 0.0, timeout: int = 60) -> str:
        name = model_name or self.model_name
        messages = self._build_messages(conv)
        self._log_request(name, messages, is_async=True)
        
        logger.debug(f"[LLAMA TARGET] Calling ExLlamaV3 at {self.client.config.base_url}")
        try:
            out = await self.client.chat_async(model=name, messages=messages, temperature=temperature,
                                               max_tokens=self.options.get('max_tokens', 512))
            logger.debug(f"[LLAMA TARGET] Response: {len(out)} chars")
            return out or ""
        except Exception as e:
            logger.error(f"[LLAMA TARGET] ERROR: {e}")
            return f"[ERROR calling ExLlamaV3: {e}]"


class VLLMModel(TargetModel):
    """vLLM model using OpenAI-compatible server."""

    def __init__(self, model_name: str, system_prompt: Optional[str] = None, options: Optional[Dict] = None):
        super().__init__(model_name, system_prompt, options)
        config = VLLMConfig(
            base_url=SYSTEM_CONFIG.vllm_url,
            timeout=SYSTEM_CONFIG.vllm_timeout,
            max_concurrent=SYSTEM_CONFIG.vllm_max_concurrent,
            api_key=SYSTEM_CONFIG.vllm_api_key,
        )
        self.client = VLLMClient(config)
        logger.info(f"vLLM Target initialized: {model_name}")
        logger.info(f"  Target URL: {self.client.config.base_url}")
        logger.info(f"  API Key: {'SET' if self.client.config.api_key else 'NOT SET'}")
        if self.system_prompt:
            logger.info(f"  System prompt mode: {SYSTEM_PROMPT_MODE} (length: {len(self.system_prompt)} chars)")
        else:
            logger.info(f"  System prompt mode: {SYSTEM_PROMPT_MODE} (empty - testing base model)")

    def run(self, conv: Conversation, model_name: Optional[str] = None,
            temperature: float = 0.0, timeout: int = 60) -> str:
        return "[ERROR: vLLM requires async - use run_async]"

    async def run_async(self, conv: Conversation, model_name: Optional[str] = None,
                        temperature: float = 0.0, timeout: int = 60) -> str:
        name = model_name or self.model_name
        messages = self._build_messages(conv)
        self._log_request(name, messages, is_async=True)

        original_timeout = self.client.config.timeout
        self.client.config.timeout = timeout
        try:
            out = await self.client.chat_async(
                model=name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.options.get("max_tokens", 512),
            )
            return out or ""
        except Exception as e:
            logger.error(f"[VLLM TARGET] ERROR: {e}")
            return f"[ERROR calling vLLM: {e}]"
        finally:
            self.client.config.timeout = original_timeout


class OpenRouterModel(TargetModel):
    """OpenRouter model for target testing."""
    
    def __init__(self, model_name: str, system_prompt: Optional[str] = None, options: Optional[Dict] = None):
        super().__init__(model_name, system_prompt, options)
        self.client = OpenRouterClient(api_key=SYSTEM_CONFIG.openrouter_api_key, timeout=SYSTEM_CONFIG.openrouter_timeout)
        self._log_init("OpenRouter")
    
    def run(self, conv: Conversation, model_name: Optional[str] = None,
            temperature: float = 0.0, timeout: int = 60) -> str:
        name = model_name or self.model_name
        messages = self._build_messages(conv)
        self._log_request(name, messages)
        
        original_timeout = self.client.timeout
        self.client.timeout = timeout
        try:
            return self.client.chat(model=name, messages=messages, temperature=temperature,
                                    max_tokens=self.options.get("max_tokens", 250)) or ""
        except Exception as e:
            return f"[ERROR calling OpenRouter: {e}]"
        finally:
            self.client.timeout = original_timeout
    
    async def run_async(self, conv: Conversation, model_name: Optional[str] = None,
                        temperature: float = 0.0, timeout: int = 60) -> str:
        name = model_name or self.model_name
        messages = self._build_messages(conv)
        self._log_request(name, messages, is_async=True)
        
        original_timeout = self.client.timeout
        self.client.timeout = timeout
        try:
            return await self.client.chat_async(model=name, messages=messages, temperature=temperature,
                                                max_tokens=self.options.get("max_tokens", 250)) or ""
        except Exception as e:
            return f"[ERROR calling OpenRouter: {e}]"
        finally:
            self.client.timeout = original_timeout



class TransformersModel(TargetModel):
    """
    Local transformers model for target testing.

    Used for models without EXL2/EXL3/Ollama versions:
    - ALLaM-7B-Instruct (SDAIA Arabic)
    - Baichuan2-7B-Chat
    - Baichuan2-13B-Chat
    """

    def __init__(self, model_name: str, system_prompt: Optional[str] = None, options: Optional[Dict] = None):
        super().__init__(model_name, system_prompt, options)
        self.client = TransformersClient(
            timeout=getattr(SYSTEM_CONFIG, "transformers_timeout", 300),
        )
        self._log_init("Transformers")

    def run(self, conv: Conversation, model_name: Optional[str] = None,
            temperature: float = 0.0, timeout: int = 60) -> str:
        name = model_name or self.model_name
        messages = self._build_messages(conv)
        self._log_request(name, messages)

        try:
            return self.client.chat(
                model=name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.options.get("max_tokens", 250),
            ) or ""
        except Exception as e:
            return f"[ERROR calling Transformers: {e}]"

    async def run_async(self, conv: Conversation, model_name: Optional[str] = None,
                        temperature: float = 0.0, timeout: int = 60) -> str:
        name = model_name or self.model_name
        messages = self._build_messages(conv)
        self._log_request(name, messages, is_async=True)

        try:
            return await self.client.chat_async(
                model=name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.options.get("max_tokens", 250),
            ) or ""
        except Exception as e:
            return f"[ERROR calling Transformers: {e}]"


class RequestyModel(TargetModel):
    """
    Requesty model for target testing.

    Requesty is a unified LLM gateway providing access to 300+ models
    from various providers (OpenAI, Anthropic, Google, Mistral, xAI, etc.)
    through a single API endpoint.

    Model naming: Use provider prefix format, e.g.:
        - xai/grok-4-1-fast-non-reasoning
        - mistral/mistral-small-latest
        - openai/gpt-4o-mini
        - anthropic/claude-sonnet-4-20250514
    """

    def __init__(self, model_name: str, system_prompt: Optional[str] = None, options: Optional[Dict] = None):
        super().__init__(model_name, system_prompt, options)
        self.client = RequestyClient(
            timeout=getattr(SYSTEM_CONFIG, "requesty_timeout", 120),
            max_concurrent=getattr(SYSTEM_CONFIG, "requesty_max_concurrent", 20),
        )
        self._log_init("Requesty")

    def run(self, conv: Conversation, model_name: Optional[str] = None,
            temperature: float = 0.0, timeout: int = 60) -> str:
        name = model_name or self.model_name
        messages = self._build_messages(conv)
        self._log_request(name, messages)

        try:
            return self.client.chat(
                model=name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.options.get("max_tokens", 250),
            ) or ""
        except Exception as e:
            return f"[ERROR calling Requesty: {e}]"

    async def run_async(self, conv: Conversation, model_name: Optional[str] = None,
                        temperature: float = 0.0, timeout: int = 60) -> str:
        name = model_name or self.model_name
        messages = self._build_messages(conv)
        self._log_request(name, messages, is_async=True)

        try:
            return await self.client.chat_async(
                model=name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.options.get("max_tokens", 250),
            ) or ""
        except Exception as e:
            return f"[ERROR calling Requesty: {e}]"

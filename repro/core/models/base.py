# =====================================
# FILE: models/base.py
# =====================================
"""
Base model classes for red-teaming framework
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
import asyncio

from repro.clients.ollama_client import get_ollama_client
from repro.clients.openai_client import get_openai_client
from repro.clients.exllamav3_client import get_exllamav3_client
from repro.clients.vllm_client import get_vllm_client
from repro.clients.openrouter_client import get_openrouter_client
from repro.core.config import (
    is_openai_model, is_ollama_model, is_exllamav3_model, is_vllm_model,
    is_openrouter_model
)

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a model"""
    model_name: str
    model_type: str  # "attacker", "target", "judge"
    temperature: float = 0.7
    max_tokens: int = 256
    additional_params: Optional[Dict[str, Any]] = None

class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, config: ModelConfig):
        self.config = config

        # Auto-detect model type and get appropriate client
        if is_openai_model(config.model_name):
            self.client = get_openai_client()
            self.client_type = "openai"
        elif is_exllamav3_model(config.model_name):
            self.client = get_exllamav3_client()
            self.client_type = "exllamav3"
        elif is_vllm_model(config.model_name):
            self.client = get_vllm_client()
            self.client_type = "vllm"
        elif is_openrouter_model(config.model_name):
            self.client = get_openrouter_client()
            self.client_type = "openrouter"
        elif is_ollama_model(config.model_name):
            self.client = get_ollama_client()
            self.client_type = "ollama"
        else:
            raise ValueError(f"Unknown model type for: {config.model_name}")

        # Verify model is available (skip for ExLlamaV3, vLLM, OpenRouter, Anthropic, Gemini as they load models dynamically)
        if self.client_type not in ["exllamav3", "vllm", "openrouter"] and not self.client.check_model(config.model_name):
            logger.warning(f"Model {config.model_name} not found in {self.client_type}")
            raise ValueError(f"Model {config.model_name} not available")

        # For future RL integration
        self.mode = 'static'  # 'static', 'learning', 'hybrid'
        self.experience_buffer = []

        logger.info(f"Initialized {config.model_type} model: {config.model_name} ({self.client_type})")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from model"""
        # Merge config parameters with call-specific kwargs
        params = {
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens
        }
        if self.config.additional_params:
            params.update(self.config.additional_params)
        params.update(kwargs)
        
        # Generate response
        response = self.client.generate(
            model=self.config.model_name,
            prompt=prompt,
            **params
        )
        
        # Store experience for future learning
        if self.mode in ['learning', 'hybrid']:
            self.experience_buffer.append({
                'prompt': prompt,
                'response': response,
                'params': params
            })
        
        return response
    
    async def async_generate(self, prompt: str, **kwargs) -> str:
        """Async wrapper to generate using the client's async method if available.
        Falls back to running sync generate in a thread."""
        # Merge params similar to generate()
        params = {
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens
        }
        if self.config.additional_params:
            params.update(self.config.additional_params)
        params.update(kwargs)
        
        # Prefer client's async if present
        if hasattr(self.client, 'generate_async'):
            return await self.client.generate_async(
                model=self.config.model_name,
                prompt=prompt,
                **params
            )
        
        # Fallback: run sync generate in thread
        return await asyncio.to_thread(self.generate, prompt, **kwargs)
    
    def preprocess_text(self, text: str, max_length: int = 2048) -> str:
        """Preprocess text before sending to model"""
        # Clean whitespace
        text = " ".join(text.split())
        
        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.debug(f"Truncated text from {len(text)} to {max_length}")
        
        return text
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information for display"""
        return {
            'name': self.config.model_name,
            'type': self.config.model_type,
            'client': self.client_type
        }
    
    @abstractmethod
    def act(self, *args, **kwargs):
        """Main action method - must be implemented by subclasses"""
        pass
    
    def receive_feedback(self, feedback: Dict[str, Any]):
        """Process feedback for learning (future RL integration)"""
        if self.mode == 'static':
            return
        
        if self.experience_buffer:
            # Attach feedback to last experience
            self.experience_buffer[-1]['feedback'] = feedback

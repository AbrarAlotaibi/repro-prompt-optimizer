"""
Provider Detection and Configuration Utilities

Shared utilities for detecting LLM providers and getting API configurations.
Used by ReflectiveLMEditor, ReflectiveCrossover, and other LLM-based components.
"""

import os
import logging
from typing import Optional

from repro.core.config import SYSTEM_CONFIG


def detect_provider(llm_client) -> str:
    """
    Detect the provider type from an LLM client instance.
    
    Args:
        llm_client: LLM client instance (OpenAI, Ollama, ExLlamaV3, OpenRouter, etc.)
    
    Returns:
        Provider name string: "openai", "ollama", "exllamav3", "openrouter"
    """
    client_type = type(llm_client).__name__.lower()
    
    if "exllama" in client_type:
        return "exllamav3"
    elif "openrouter" in client_type:
        return "openrouter"
    elif "ollama" in client_type:
        return "ollama"
    else:
        return "openai"


def get_api_key(provider: str) -> str:
    """
    Get the API key for a provider.
    
    Args:
        provider: Provider name ("openai", "openrouter", etc.)
    
    Returns:
        API key string (empty string if not available)
    """
    if provider == "openrouter":
        return SYSTEM_CONFIG.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
    elif provider == "openai":
        return SYSTEM_CONFIG.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    else:
        return ""


def get_base_url(provider: str, llm_client=None) -> Optional[str]:
    """
    Get the base URL for a provider.
    
    Args:
        provider: Provider name
        llm_client: Optional LLM client for ExLlamaV3 URL extraction
    
    Returns:
        Base URL string or None for default
    """
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    elif provider == "ollama":
        return SYSTEM_CONFIG.ollama_url or "http://localhost:11434"
    elif provider == "exllamav3":
        # Try to get from client config first
        if llm_client is not None:
            if hasattr(llm_client, 'config') and hasattr(llm_client.config, 'base_url'):
                base = llm_client.config.base_url
                return f"{base}/v1" if base and not base.endswith('/v1') else base
        # Fallback to env/config
        base = SYSTEM_CONFIG.exllamav3_reflective_url or SYSTEM_CONFIG.exllamav3_url
        if base:
            return f"{base}/v1" if not base.endswith('/v1') else base
        return os.getenv("EXLLAMAV3_REFLECTIVE_URL", "http://127.0.0.1:5002/v1")
    else:
        return None

"""
vLLM client wrapper for REPRO.
Re-exports the main client implementation if available.
"""

try:
    from clients.vllm_client import (
        VLLMClient,
        VLLMClientSync,
        VLLMConfig,
        get_vllm_client,
        reset_vllm_client,
    )
except ImportError:
    # vLLM client not available -- define stubs
    VLLMClient = None
    VLLMClientSync = None
    VLLMConfig = None

    def get_vllm_client(*args, **kwargs):
        raise ImportError("vLLM client is not installed")

    def reset_vllm_client(*args, **kwargs):
        pass

__all__ = [
    "VLLMClient",
    "VLLMClientSync",
    "VLLMConfig",
    "get_vllm_client",
    "reset_vllm_client",
]

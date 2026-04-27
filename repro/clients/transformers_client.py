# =====================================
# FILE: transformers_client.py
# =====================================
"""
Transformers client for local model inference.
Used for models without EXL2/EXL3/Ollama versions (e.g., ALLaM, Baichuan).
"""
import os
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path

from repro.clients.logging_utils import log_request_summary, new_request_id

logger = logging.getLogger(__name__)

# Default models directory
TRANSFORMERS_MODELS_DIR = Path("/hdd/g202203240/models/transformers")


class TransformersClient:
    """
    Client for running inference on local transformers models.

    Supports models that don't have EXL2/EXL3/Ollama versions:
    - ALLaM-7B-Instruct (SDAIA Arabic)
    - Baichuan2-7B-Chat
    - Baichuan2-13B-Chat
    """

    def __init__(
        self,
        models_dir: str = None,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        max_concurrent: int = 1,  # Transformers is not async-friendly
        timeout: int = 300,
    ):
        self.models_dir = Path(models_dir) if models_dir else TRANSFORMERS_MODELS_DIR
        self.device = device
        self.torch_dtype = torch_dtype
        self.timeout = timeout
        self.max_concurrent = max_concurrent

        # Model cache (loaded models stay in memory)
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._model_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Available models (discovered from directory)
        self.available_models = self._discover_models()

        logger.info(f"TransformersClient initialized: {len(self.available_models)} models available")
        if self.available_models:
            logger.info(f"  Models: {', '.join(self.available_models)}")

    def _discover_models(self) -> List[str]:
        """Discover available models in the models directory."""
        models = []
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return models

        for path in self.models_dir.iterdir():
            if path.is_dir():
                # Check for config.json (indicates a valid model)
                if (path / "config.json").exists():
                    models.append(path.name)

        return sorted(models)

    def _get_torch_dtype(self):
        """Get torch dtype from string."""
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": "auto",
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)

    def _load_model(self, model_name: str):
        """Load a model and tokenizer into memory."""
        if model_name in self._models:
            return self._models[model_name], self._tokenizers[model_name]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = self.models_dir / model_name

        if not model_path.exists():
            raise ValueError(f"Model not found: {model_path}")

        logger.info(f"Loading model: {model_name} from {model_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=self._get_torch_dtype(),
            device_map=self.device,
            trust_remote_code=True,
        )

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._models[model_name] = model
        self._tokenizers[model_name] = tokenizer

        logger.info(f"Model loaded: {model_name} (device: {model.device})")
        return model, tokenizer

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.95,
        top_k: int = 50,
        **kwargs,
    ) -> str:
        """Generate response from model."""
        import torch

        model_obj, tokenizer = self._load_model(model)

        request_id = new_request_id()
        log_request_summary(logger, "Transformers", prompt=prompt, request_id=request_id)

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
        inputs = {k: v.to(model_obj.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model_obj.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode (only new tokens)
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        return response.strip()

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.95,
        top_k: int = 50,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Chat completion using chat template."""
        import torch

        model_obj, tokenizer = self._load_model(model)

        request_id = new_request_id()
        log_request_summary(logger, "Transformers", messages=messages, request_id=request_id)

        # Apply chat template
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: simple formatting
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
        inputs = {k: v.to(model_obj.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model_obj.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode (only new tokens)
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        return response.strip()

    async def generate_async(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs,
    ) -> str:
        """Async wrapper for generate (runs in thread pool)."""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.generate(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ),
            )

    async def chat_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs,
    ) -> str:
        """Async wrapper for chat (runs in thread pool)."""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.chat(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ),
            )

    def check_model(self, model_name: str) -> bool:
        """Check if model is available."""
        return model_name in self.available_models

    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        import torch
        import gc

        if model_name in self._models:
            del self._models[model_name]
            del self._tokenizers[model_name]
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {model_name}")

    def unload_all(self):
        """Unload all models from memory."""
        import torch
        import gc

        model_names = list(self._models.keys())
        for name in model_names:
            del self._models[name]
            del self._tokenizers[name]

        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Unloaded {len(model_names)} models")


# Global client instance
_transformers_client = None


def get_transformers_client() -> TransformersClient:
    """Get or create Transformers client."""
    global _transformers_client
    if _transformers_client is None:
        from repro.core.config import SYSTEM_CONFIG
        _transformers_client = TransformersClient(
            timeout=getattr(SYSTEM_CONFIG, "transformers_timeout", 300),
        )
    return _transformers_client

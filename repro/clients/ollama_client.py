# =====================================
# FILE: ollama_client.py
# =====================================
"""
Ollama client with GPU support
"""
import os
import json
import logging
import requests
import httpx
import subprocess
from typing import Dict, List, Optional, Any
import time
import asyncio

from repro.clients.logging_utils import log_request_summary, new_request_id, summarize_text

logger = logging.getLogger(__name__)

def find_ollama_executable():
    """Find Ollama executable from system PATH only"""
    # Check if ollama is in system PATH
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, check=True)
        return 'ollama'
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Ollama executable not found in system PATH")
        return None

class OllamaClient:
    """Ollama client with GPU awareness and persistent connection pooling"""

    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 timeout: int = 300,
                 enable_multi_gpu: bool = None,
                 max_concurrent: int = 6):

        # Import here to avoid circular imports
        from repro.core.config import SYSTEM_CONFIG

        self.base_url = base_url
        self.timeout = timeout
        self.max_concurrent = max_concurrent

        # Use config-detected multi-GPU settings if not explicitly provided
        if enable_multi_gpu is None:
            self.enable_multi_gpu = SYSTEM_CONFIG.enable_multi_gpu
        else:
            self.enable_multi_gpu = enable_multi_gpu

        self._request_count = 0

        # Find Ollama executable
        self.ollama_path = find_ollama_executable()
        if not self.ollama_path:
            logger.debug("Ollama executable not found in PATH")
            self.available_models = []
            self.gpu_instances = []
            return

        # Get available models and GPU instances
        if not self._check_ollama():
            logger.debug(f"Ollama not accessible at {base_url}. Will try to use CLI for model list.")
            self.available_models = self._get_models_from_cli()
        else:
            self.available_models = self._get_available_models()

        # Single instance mode - no multi-GPU instances needed
        self.gpu_instances = []

        if self.enable_multi_gpu:
            cuda_devices = getattr(SYSTEM_CONFIG, 'cuda_visible_devices', '')
            if cuda_devices:
                logger.debug(f"Multi-GPU mode: Single instance using GPUs {cuda_devices}")
            else:
                logger.debug(f"Multi-GPU mode: Single instance with automatic GPU distribution")

        # PERSISTENT CONNECTION POOL (reused across all requests)
        extended_timeout = max(self.timeout * 1.5, 120)
        self._timeout_config = httpx.Timeout(
            connect=15.0,
            read=extended_timeout,
            write=15.0,
            pool=10.0
        )

        # Optimized connection limits for better throughput
        self._limits = httpx.Limits(
            max_keepalive_connections=max_concurrent * 2,  # 2x concurrent for headroom
            max_connections=max_concurrent * 4,             # 4x for burst capacity
            keepalive_expiry=60.0                          # Keep connections alive for 1 min
        )

        # Create persistent async client (reused for all async requests)
        self._async_client = None
        self._client_lock = asyncio.Lock()

        # Rate limiting semaphore (shared across all async requests)
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(f"✅ Ollama client initialized: max_concurrent={max_concurrent}, timeout={extended_timeout}s")
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
        except:
            pass
        return []
    
    def _get_models_from_cli(self) -> List[str]:
        """Get available models using Ollama CLI"""
        try:
            result = subprocess.run([self.ollama_path, 'list'], 
                                  capture_output=True, text=True, check=True)
            models = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]  # First column is the model name
                    models.append(model_name)
            return models
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get models from CLI: {e}")
            return []
    
    def start_ollama_service(self, gpu_id: Optional[int] = None, port: Optional[int] = None):
        """Start Ollama service with optional GPU and port specification"""
        if not self.ollama_path:
            raise RuntimeError("Ollama executable not found")
        
        cmd = [self.ollama_path, 'serve']
        env = os.environ.copy()
        
        # Set GPU if specified
        if gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.debug(f"Starting Ollama service on GPU {gpu_id}")
        
        # Set port if specified
        if port is not None:
            env['OLLAMA_HOST'] = f"localhost:{port}"
            logger.debug(f"Starting Ollama service on port {port}")
        
        try:
            # Start service in background
            process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Give it a moment to start
            time.sleep(3)
            return process
        except Exception as e:
            logger.error(f"Failed to start Ollama service: {e}")
            return None
    
# Multi-GPU setup methods removed - using single instance mode
    
    def generate(self, 
                 model: str, 
                 prompt: str,
                 temperature: float = 0.7,
                 max_tokens: int = 256,
                 **kwargs) -> str:
        """Generate response from model"""
        
        # Use single instance
        url = self.base_url
        
        # Prepare request
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            **kwargs
        }
        
        # Add GPU layer configuration if specified
        from repro.core.config import SYSTEM_CONFIG
        if SYSTEM_CONFIG.gpu_layers is not None:
            options["num_gpu"] = SYSTEM_CONFIG.gpu_layers
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options
        }

        
        # Make request with retries
        for attempt in range(3):
            try:
                request_id = new_request_id()
                log_request_summary(logger, "Ollama", prompt=prompt, request_id=request_id)
                response = requests.post(
                    f"{url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['response'].strip()
                else:
                    logger.error(f"Ollama error: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < 2:
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Generation error: {e}")
                if attempt < 2:
                    time.sleep(1)
        
        raise RuntimeError(f"Failed to generate after 3 attempts")
    
    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create persistent async client (connection pool reuse)."""
        async with self._client_lock:
            if self._async_client is None or self._async_client.is_closed:
                self._async_client = httpx.AsyncClient(
                    timeout=self._timeout_config,
                    limits=self._limits
                )
                logger.debug("Created new persistent async client")
            return self._async_client

    async def close(self):
        """Close persistent async client."""
        async with self._client_lock:
            if self._async_client and not self._async_client.is_closed:
                await self._async_client.aclose()
                logger.debug("Closed persistent async client")

    async def generate_async(self,
                             model: str,
                             prompt: str,
                             temperature: float = 0.7,
                             max_tokens: int = 256,
                             **kwargs) -> str:
        """Async generate with persistent connection pool and rate limiting."""
        url = self.base_url

        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            **kwargs
        }

        from repro.core.config import SYSTEM_CONFIG
        if SYSTEM_CONFIG.gpu_layers is not None:
            options["num_gpu"] = SYSTEM_CONFIG.gpu_layers

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options
        }


        # Use rate limiting semaphore
        async with self._semaphore:
            # Get persistent client (reused connection pool)
            client = await self._get_async_client()

            for attempt in range(3):
                try:
                    request_id = new_request_id()
                    log_request_summary(logger, "Ollama", prompt=prompt, request_id=request_id)
                    response = await client.post(
                        f"{url}/api/generate",
                        json=payload
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return result['response'].strip()
                    elif response.status_code == 404:
                        logger.error(f"Model {model} not found (404). Run: ollama pull {model}")
                        raise RuntimeError(f"Model {model} not found. Check 'ollama list' or run 'ollama pull {model}'")
                    else:
                        logger.error(f"Ollama error: {response.status_code} - {summarize_text(response.text, limit=200)}")
                        if response.status_code >= 500 and attempt < 2:
                            await asyncio.sleep(2)  # Shorter retry for server errors

                except httpx.TimeoutException as e:
                    logger.warning(f"Timeout on attempt {attempt + 1}/3: {e}")
                    if attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))  # 2s, 4s backoff
                    else:
                        logger.error(f"Model {model} consistently timing out. Try: --max-concurrent 4 --timeout 180")
                except httpx.ConnectError as e:
                    logger.error(f"Connection error on attempt {attempt + 1}/3: {e}. Is Ollama running?")
                    if attempt < 2:
                        await asyncio.sleep(3)
                    else:
                        raise RuntimeError(f"Cannot connect to Ollama at {url}. Ensure it's running: ollama serve")
                except Exception as e:
                    logger.error(f"Generation error on attempt {attempt + 1}/3: {type(e).__name__}: {e}")
                    if attempt < 2:
                        await asyncio.sleep(1)

            raise RuntimeError(f"Failed to generate with {model} after 3 attempts. Try reducing --max-concurrent or increasing --timeout.")
    
    async def chat_async(self,
                        model: str,
                        messages: List[Dict[str, str]],
                        temperature: float = 0.7,
                        max_tokens: int = 256,
                        format_schema: Optional[Dict[str, Any]] = None,
                        response_format: Optional[Dict[str, Any]] = None,
                        **kwargs) -> str:
        """
        Async chat with persistent connection pool and rate limiting.

        Args:
            model: Model name
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            format_schema: Optional JSON schema for structured output (Ollama native) - DEPRECATED, use response_format
            response_format: Universal JSON schema format (OpenAI-compatible)
            **kwargs: Additional options
        """
        url = self.base_url

        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            **kwargs
        }

        from repro.core.config import SYSTEM_CONFIG
        if SYSTEM_CONFIG.gpu_layers is not None:
            options["num_gpu"] = SYSTEM_CONFIG.gpu_layers

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options
        }


        # Universal response_format support (adapter for Ollama)
        # Extract schema from OpenAI-style response_format if provided
        if response_format is not None and format_schema is None:
            # Extract JSON schema from universal format
            if "json_schema" in response_format and "schema" in response_format["json_schema"]:
                format_schema = response_format["json_schema"]["schema"]

        # Add format schema for structured output (Ollama 0.5+)
        if format_schema is not None:
            payload["format"] = format_schema

        # Use rate limiting semaphore
        async with self._semaphore:
            # Get persistent client (reused connection pool)
            client = await self._get_async_client()

            for attempt in range(3):
                try:
                    request_id = new_request_id()
                    log_request_summary(logger, "Ollama", messages=messages, request_id=request_id)
                    response = await client.post(
                        f"{url}/api/chat",
                        json=payload
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return result.get('message', {}).get('content', '').strip()
                    elif response.status_code == 404:
                        logger.error(f"Model {model} not found (404). Run: ollama pull {model}")
                        raise RuntimeError(f"Model {model} not found. Check 'ollama list' or run 'ollama pull {model}'")
                    else:
                        logger.error(f"Ollama chat error: {response.status_code} - {summarize_text(response.text, limit=200)}")
                        if response.status_code >= 500 and attempt < 2:
                            await asyncio.sleep(2)

                except httpx.TimeoutException as e:
                    logger.warning(f"Chat timeout on attempt {attempt + 1}/3: {e}")
                    if attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))  # 2s, 4s backoff
                    else:
                        logger.error(f"Model {model} consistently timing out. Try: --max-concurrent 4 --timeout 180")
                except httpx.ConnectError as e:
                    logger.error(f"Connection error on attempt {attempt + 1}/3: {e}. Is Ollama running?")
                    if attempt < 2:
                        await asyncio.sleep(3)
                    else:
                        raise RuntimeError(f"Cannot connect to Ollama at {url}. Ensure it's running: ollama serve")
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error on attempt {attempt + 1}/3: {e.response.status_code}")
                    if e.response.status_code >= 500 and attempt < 2:
                        await asyncio.sleep(2)
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Chat error on attempt {attempt + 1}/3: {type(e).__name__}: {e}")
                    if attempt < 2:
                        await asyncio.sleep(1)

            raise RuntimeError(f"Failed to chat with {model} after 3 attempts. Try reducing --max-concurrent or increasing --timeout.")
    
    def chat(self,
             model: str,
             messages: List[Dict[str, str]],
             temperature: float = 0.7,
             max_tokens: int = 256,
             format_schema: Optional[Dict[str, Any]] = None,
             response_format: Optional[Dict[str, Any]] = None,
             **kwargs) -> str:
        """
        Chat completion using Ollama chat API with GPU optimizations.

        Args:
            model: Model name
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            format_schema: Optional JSON schema for structured output (Ollama native) - DEPRECATED, use response_format
            response_format: Universal JSON schema format (OpenAI-compatible)
            **kwargs: Additional options
        """

        # Use single instance
        url = self.base_url

        # Prepare request
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            **kwargs
        }

        # Add GPU layer configuration if specified
        from repro.core.config import SYSTEM_CONFIG
        if SYSTEM_CONFIG.gpu_layers is not None:
            options["num_gpu"] = SYSTEM_CONFIG.gpu_layers

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options
        }


        # Universal response_format support (adapter for Ollama)
        # Extract schema from OpenAI-style response_format if provided
        if response_format is not None and format_schema is None:
            # Extract JSON schema from universal format
            if "json_schema" in response_format and "schema" in response_format["json_schema"]:
                format_schema = response_format["json_schema"]["schema"]

        # Add format schema for structured output (Ollama 0.5+)
        if format_schema is not None:
            payload["format"] = format_schema
        
        # Make request with retries
        for attempt in range(3):
            try:
                request_id = new_request_id()
                log_request_summary(logger, "Ollama", messages=messages, request_id=request_id)
                response = requests.post(
                    f"{url}/api/chat",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('message', {}).get('content', '').strip()
                else:
                    logger.error(f"Ollama chat error: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Chat timeout on attempt {attempt + 1}")
                if attempt < 2:
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Chat error: {e}")
                if attempt < 2:
                    time.sleep(1)
        
        raise RuntimeError(f"Failed to chat after 3 attempts with model {model}")
    
    def generate_stream(self,
                       model: str,
                       prompt: str,
                       temperature: float = 0.7,
                       max_tokens: int = 256,
                       **kwargs):
        """Generate streaming response from model"""
        
        # Use single instance
        url = self.base_url
        
        # Prepare request
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            **kwargs
        }
        
        # Add GPU layer configuration if specified
        from repro.core.config import SYSTEM_CONFIG
        if SYSTEM_CONFIG.gpu_layers is not None:
            options["num_gpu"] = SYSTEM_CONFIG.gpu_layers
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,  # Enable streaming
            "options": options
        }

        
        try:
            request_id = new_request_id()
            log_request_summary(logger, "Ollama", prompt=prompt, request_id=request_id)
            response = requests.post(
                f"{url}/api/generate",
                json=payload,
                stream=True,  # Stream the response
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                # Yield each chunk as it arrives
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                yield {"text": chunk['response']}
                            # Check if generation is done
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse streaming chunk: {line}")
                            continue
            else:
                logger.error(f"Ollama streaming error: {response.status_code}")
                yield {"error": f"Ollama error: {response.status_code}"}
                
        except requests.exceptions.Timeout:
            logger.error("Streaming request timed out")
            yield {"error": "Request timed out"}
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield {"error": str(e)}
    
    def check_model(self, model_name: str) -> bool:
        """Check if model is available"""
        return model_name in self.available_models

    def get_optimal_url_for_model(self, model_role: str = "default") -> str:
        """Get optimal URL for a specific model role - single instance mode"""
        return self.base_url

# Global client instance
_ollama_client = None

def get_ollama_client() -> OllamaClient:
    """Get or create Ollama client"""
    global _ollama_client
    if _ollama_client is None:
        from repro.core.config import SYSTEM_CONFIG
        _ollama_client = OllamaClient(
            base_url=SYSTEM_CONFIG.ollama_url,
            timeout=SYSTEM_CONFIG.ollama_timeout,
            enable_multi_gpu=SYSTEM_CONFIG.enable_gpu
        )
    return _ollama_client

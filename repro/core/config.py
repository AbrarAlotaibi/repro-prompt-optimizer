# =====================================
# FILE: config.py
# =====================================
"""
Unified configuration for Red-Team Framework
MONITORING FIRST - All monitoring configuration comes before any other config
"""
import os
import subprocess
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # Shell exports take precedence over .env file
except ImportError:
    pass  # dotenv not available, environment variables should be set manually

logger = logging.getLogger(__name__)


# ==========================================
# GPU DETECTION AND OPTIMIZATION
# ==========================================


def detect_nvidia_gpus() -> List[Dict[str, Any]]:
    """Detect NVIDIA GPUs and their capabilities"""
    gpus = []
    try:
        # Try pynvml first with improved error handling
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    # Handle both bytes and string return types
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    gpus.append({
                        'id': i,
                        'name': name,
                        'memory_total': memory_info.total // (1024**3),  # GB
                        'memory_free': memory_info.free // (1024**3),   # GB
                        'available': True
                    })

                except Exception as e:
                    logger.warning(f"GPU {i} unavailable (pynvml): {e}")
                    continue  # Skip this GPU and try the next

            if gpus:
                logger.info(f"GPU detection via pynvml: {len(gpus)} working GPUs found")
                return gpus

        except ImportError:
            logger.debug("pynvml not available, falling back to nvidia-smi")
        except Exception as e:
            logger.debug(f"pynvml initialization failed: {e}")

        # Simple nvidia-smi fallback for systems without pynvml
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'],
                                    capture_output=True, text=True)

            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.startswith('GPU ') and ':' in line and 'Unable to determine' not in line:
                        try:
                            parts = line.split(':')
                            if len(parts) >= 2:
                                gpu_id_part = parts[0].split()
                                if len(gpu_id_part) >= 2:
                                    gpu_id = int(gpu_id_part[1])
                                    gpu_name = parts[1].split('(')[0].strip()

                                    # Estimate memory based on GPU model
                                    memory_total = 24 if 'RTX 3090' in gpu_name else 16
                                    if 'RTX 4090' in gpu_name:
                                        memory_total = 24
                                    elif 'RTX 3080' in gpu_name:
                                        memory_total = 12
                                    elif 'RTX 3070' in gpu_name:
                                        memory_total = 8

                                    gpus.append({
                                        'id': gpu_id,
                                        'name': gpu_name,
                                        'memory_total': memory_total,
                                        'memory_free': int(memory_total * 0.9),  # Conservative estimate
                                        'available': True
                                    })
                        except (ValueError, IndexError):
                            continue
                    elif 'Unable to determine' in line:
                        logger.warning(f"GPU issue detected (skipping): {line.strip()}")

                if gpus:
                    logger.info(f"GPU detection via nvidia-smi: {len(gpus)} working GPUs found")
                    return gpus

        except Exception as e:
            logger.debug(f"nvidia-smi fallback failed: {e}")

        # Final fallback - check environment variables
        try:
            # Check if CUDA_VISIBLE_DEVICES is set
            cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            if cuda_devices and cuda_devices != '-1':
                device_ids = [int(x.strip()) for x in cuda_devices.split(',') if x.strip().isdigit()]
                for gpu_id in device_ids:
                    gpus.append({
                        'id': gpu_id,
                        'name': f'NVIDIA GPU {gpu_id}',
                        'memory_total': 16,  # Conservative default
                        'memory_free': 12,   # Conservative default
                        'available': True
                    })
                logger.info(f"GPU detection via CUDA_VISIBLE_DEVICES: {len(gpus)} GPUs found")
                return gpus

        except Exception as e:
            logger.debug(f"Environment detection failed: {e}")

    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")

    # If we reach here, no GPUs were detected
    logger.info("No NVIDIA GPUs detected, will use CPU/cloud models")
    return gpus


def calculate_optimal_gpu_settings(gpus: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate optimal GPU settings based on detected hardware"""
    if not gpus:
        return {
            'enable_gpu': False,
            'enable_multi_gpu': False,
            'gpu_layers': None,
            'performance_profile': 'cpu_fallback',
            'detected_gpus': [],
            'total_vram_gb': 0,
            'gpu_count': 0,
            'max_parallel_models': 1
        }

    total_vram = sum(gpu['memory_total'] for gpu in gpus)
    min_vram = min(gpu['memory_total'] for gpu in gpus) if gpus else 0

    # Determine optimal settings based on GPU configuration
    if len(gpus) >= 3 and min_vram >= 20:
        # High-end multi-GPU setup - single Ollama instance with automatic GPU distribution
        working_gpu_ids = [gpu['id'] for gpu in gpus]
        settings = {
            'enable_gpu': True,
            'enable_multi_gpu': True,
            'gpu_layers': -1,
            'max_parallel_models': 4,  #
            'cuda_visible_devices': ','.join(map(str, working_gpu_ids)),  #
            'performance_profile': 'high_performance'
        }
    elif len(gpus) >= 2 and min_vram >= 12:
        # Dual GPU setup -
        working_gpu_ids = [gpu['id'] for gpu in gpus]
        settings = {
            'enable_gpu': True,
            'enable_multi_gpu': True,
            'gpu_layers': 50,
            'max_parallel_models': 3,  # Can run multiple models on single instance
            'cuda_visible_devices': ','.join(map(str, working_gpu_ids)),  # For environment setup
            'performance_profile': 'balanced'
        }
    elif len(gpus) == 1 and min_vram >= 8:
        # Single GPU setup - single Ollama instance
        working_gpu_ids = [gpu['id'] for gpu in gpus]
        gpu_layers = min(60, max(20, min_vram * 2))  # Scale with VRAM
        settings = {
            'enable_gpu': True,
            'enable_multi_gpu': False,
            'gpu_layers': gpu_layers,
            'max_parallel_models': 2,  # Can still run multiple smaller models
            'cuda_visible_devices': ','.join(map(str, working_gpu_ids)),  # Specific GPU
            'performance_profile': 'optimized'
        }
    else:
        # Low VRAM or CPU fallback - single Ollama instance
        working_gpu_ids = [gpu['id'] for gpu in gpus] if gpus else []
        settings = {
            'enable_gpu': min_vram >= 4,
            'enable_multi_gpu': False,
            'gpu_layers': 20 if min_vram >= 4 else None,
            'max_parallel_models': 1,
            'cuda_visible_devices': ','.join(map(str, working_gpu_ids)) if working_gpu_ids else '',
            'performance_profile': 'memory_efficient'
        }

    settings['detected_gpus'] = gpus
    settings['total_vram_gb'] = total_vram
    settings['gpu_count'] = len(gpus)

    return settings


# Detect GPUs and calculate optimal settings
DETECTED_GPUS = detect_nvidia_gpus()
GPU_SETTINGS = calculate_optimal_gpu_settings(DETECTED_GPUS)

# Log GPU detection results
if DETECTED_GPUS:
    logger.info(f"Detected {len(DETECTED_GPUS)} GPU(s):")
    for gpu in DETECTED_GPUS:
        logger.info(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_total']}GB VRAM)")
    logger.info(f"Performance Profile: {GPU_SETTINGS['performance_profile']}")
    if GPU_SETTINGS['enable_multi_gpu']:
        cuda_devices = GPU_SETTINGS.get('cuda_visible_devices', '')
        if cuda_devices:
            logger.info(f"Multi-GPU enabled: Single instance using GPUs {cuda_devices}")
        else:
            logger.info("Multi-GPU enabled: Single instance with automatic GPU distribution")
else:
    logger.warning("No NVIDIA GPUs detected, using CPU/cloud models")


# ==========================================
# SYSTEM CONFIGURATION
# ==========================================


@dataclass
class SystemConfig:
    """System-wide configuration with automatic GPU optimization"""
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "300"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_timeout: int = int(os.getenv("OPENAI_TIMEOUT", "60"))
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_timeout: int = int(os.getenv("OPENROUTER_TIMEOUT", "60"))
    # Requesty Configuration (OpenAI-compatible API gateway for 300+ models)
    requesty_api_key: str = os.getenv("REQUESTY_API_KEY", "")
    requesty_base_url: str = os.getenv("REQUESTY_BASE_URL", "https://router.requesty.ai/v1")
    requesty_timeout: int = int(os.getenv("REQUESTY_TIMEOUT", "120"))
    requesty_max_concurrent: int = int(os.getenv("REQUESTY_MAX_CONCURRENT", "20"))
    results_dir: str = "results"
    log_level: str = os.getenv("LOG_LEVEL", "WARNING")

    # ExLlamaV3/TabbyAPI Configuration
    exllamav3_url: str = os.getenv("EXLLAMAV3_URL", "http://127.0.0.1:5000")
    exllamav3_reflective_url: Optional[str] = os.getenv("EXLLAMAV3_REFLECTIVE_URL")  # Separate URL for reflective LM
    exllamav3_prefilter_url: Optional[str] = os.getenv("EXLLAMAV3_PREFILTER_URL")   # Separate URL for pre-filter/refiner LM
    exllamav3_judge_url: str = os.getenv("EXLLAMAV3_JUDGE_URL", "http://127.0.0.1:5001")  # Separate URL for judge model
    exllamav3_timeout: int = int(os.getenv("EXLLAMAV3_TIMEOUT", "300"))
    exllamav3_enable_streaming: bool = os.getenv("EXLLAMAV3_STREAMING", "true").lower() == "true"
    exllamav3_enable_batching: bool = os.getenv("EXLLAMAV3_BATCHING", "true").lower() == "true"
    exllamav3_batch_size: int = int(os.getenv("EXLLAMAV3_BATCH_SIZE", "8"))
    exllamav3_max_concurrent: int = int(os.getenv("EXLLAMAV3_MAX_CONCURRENT", "8"))  # Balanced for larger batch sizes
    exllamav3_api_key: Optional[str] = os.getenv("EXLLAMAV3_API_KEY")
    # GPU configuration for ExLlamaV3 (auto-detected from DETECTED_GPUS)
    exllamav3_gpu_split: Optional[str] = os.getenv("EXLLAMAV3_GPU_SPLIT", "auto")
    exllamav3_cache_mode: str = os.getenv("EXLLAMAV3_CACHE_MODE", "FP16")
    exllamav3_max_seq_len: int = int(os.getenv("EXLLAMAV3_MAX_SEQ_LEN", "4096"))
    # vLLM Configuration (OpenAI-compatible server)
    vllm_url: str = os.getenv("VLLM_URL", "http://127.0.0.1:8000")
    vllm_reflective_url: Optional[str] = os.getenv("VLLM_REFLECTIVE_URL")
    vllm_judge_url: str = os.getenv("VLLM_JUDGE_URL", "http://127.0.0.1:8001")
    vllm_timeout: int = int(os.getenv("VLLM_TIMEOUT", "300"))
    vllm_enable_streaming: bool = os.getenv("VLLM_STREAMING", "true").lower() == "true"
    vllm_enable_batching: bool = os.getenv("VLLM_BATCHING", "true").lower() == "true"
    vllm_batch_size: int = int(os.getenv("VLLM_BATCH_SIZE", "8"))
    vllm_max_concurrent: int = int(os.getenv("VLLM_MAX_CONCURRENT", "8"))
    vllm_api_key: Optional[str] = os.getenv("VLLM_API_KEY")

    # GPU Configuration - Auto-detected or manual override
    enable_gpu: bool = os.getenv("ENABLE_GPU", str(GPU_SETTINGS['enable_gpu'])).lower() == "true"
    enable_multi_gpu: bool = os.getenv("ENABLE_MULTI_GPU", str(GPU_SETTINGS['enable_multi_gpu'])).lower() == "true"
    gpu_layers: Optional[int] = int(os.getenv("GPU_LAYERS")) if os.getenv(
        "GPU_LAYERS") else GPU_SETTINGS.get('gpu_layers')

    # Multi-GPU settings - use field(default_factory) to avoid mutable defaults
    detected_gpus: List[Dict[str, Any]] = field(default_factory=list)
    performance_profile: str = ""
    max_parallel_models: int = 1
    cuda_visible_devices: str = ""

    def __post_init__(self):
        """Post-initialization setup"""
        # Set GPU settings from detection
        self.detected_gpus = GPU_SETTINGS['detected_gpus']
        self.performance_profile = GPU_SETTINGS['performance_profile']
        self.max_parallel_models = GPU_SETTINGS.get('max_parallel_models', 1)
        self.cuda_visible_devices = GPU_SETTINGS.get('cuda_visible_devices', '')

        # Apply performance optimizations based on profile
        self._apply_performance_optimizations()

        # Override GPU settings if environment variables are set
        if os.getenv("GPU_LAYERS"):
            self.gpu_layers = int(os.getenv("GPU_LAYERS"))
            logger.info(f"GPU layers overridden by environment: {self.gpu_layers}")

    def _apply_performance_optimizations(self):
        """Apply performance optimizations based on detected hardware"""
        if self.performance_profile == 'high_performance':
            # Optimize for maximum performance
            os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "4")
            os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
            os.environ.setdefault("OLLAMA_MAX_QUEUE", "512")

        elif self.performance_profile == 'balanced':
            # Balanced settings
            os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "3")
            os.environ.setdefault("OLLAMA_NUM_PARALLEL", "3")
            os.environ.setdefault("OLLAMA_MAX_QUEUE", "256")

        elif self.performance_profile == 'optimized':
            # Single GPU optimized
            os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "2")
            os.environ.setdefault("OLLAMA_NUM_PARALLEL", "2")
            os.environ.setdefault("OLLAMA_MAX_QUEUE", "128")

        elif self.performance_profile == 'memory_efficient':
            # Conservative settings for limited resources
            os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
            os.environ.setdefault("OLLAMA_NUM_PARALLEL", "1")
            os.environ.setdefault("OLLAMA_MAX_QUEUE", "64")

    def get_ollama_urls(self) -> List[str]:
        """Get Ollama instance URL - single instance mode"""
        return [self.ollama_url]

    def get_optimal_model_assignment(self) -> Dict[str, str]:
        """Get optimal model assignments - single instance mode with automatic GPU distribution"""
        base_url = self.ollama_url
        return {
            'attacker': base_url,
            'target': base_url,
            'judge': base_url
        }


def _reload_system_config():
    """Reload SYSTEM_CONFIG from current os.environ (call after load_dotenv)."""
    global SYSTEM_CONFIG
    SYSTEM_CONFIG = SystemConfig()


# Global instance with auto-optimization
SYSTEM_CONFIG = SystemConfig()

# ==========================================
# SYSTEM PROMPT STANDARDIZATION CONFIGURATION
# ==========================================
"""
System prompt standardization following benchmark best practices (JailbreakBench, HarmBench).

Philosophy: Test models AS THEY ARE, not with injected safety prompts.
This ensures reproducible results across backends and realistic adversarial testing.
"""

# System Prompt Standardization Settings
SYSTEM_PROMPT_MODE: str = os.getenv("SYSTEM_PROMPT_MODE", "empty")
CUSTOM_SYSTEM_PROMPT: str = os.getenv("CUSTOM_SYSTEM_PROMPT", "")
LOG_FORMATTED_PROMPTS: bool = os.getenv("LOG_FORMATTED_PROMPTS", "false").lower() == "true"

# Log system prompt configuration
logger.info(f"System Prompt Configuration: mode={SYSTEM_PROMPT_MODE}, debug_logging={LOG_FORMATTED_PROMPTS}")
if SYSTEM_PROMPT_MODE == "custom" and CUSTOM_SYSTEM_PROMPT:
    logger.info(f"Custom system prompt configured: {CUSTOM_SYSTEM_PROMPT[:50]}...")

# ==========================================
# MODEL CONFIGURATIONS
# ==========================================


# ==========================================
# UTILITY FUNCTIONS
# ==========================================


def is_openai_model(model_name: str) -> bool:
    """Check if a model is an OpenAI model"""
    # Explicit OpenRouter routes should not be claimed as native OpenAI
    if model_name.startswith("openrouter:"):
        return False

    normalized = model_name.lstrip()

    # Allow provider-prefixed identifiers like "openai/gpt-4o"
    if normalized.startswith("openai/"):
        normalized = normalized.split("/", 1)[1]

    return (normalized.startswith("gpt-") or
            normalized.startswith("chatgpt-") or
            normalized.startswith("o1") or
            normalized.startswith("o3") or
            normalized.startswith("o4") or
            normalized.startswith("text-embedding-"))


def is_exllamav3_model(model_name: str) -> bool:
    """Check if a model is an ExLlamaV3 model (via TabbyAPI)"""
    return model_name.startswith("exl3:")


def is_vllm_model(model_name: str) -> bool:
    """Check if a model is a vLLM-served model"""
    return model_name.startswith("vllm:") or model_name.startswith("vllm/")


def is_openrouter_model(model_name: str) -> bool:
    """Check if a model is an OpenRouter model"""
    # OpenRouter models typically have provider prefix (e.g., "anthropic/claude-3.5-sonnet")
    # or can be explicitly prefixed with "openrouter:"
    if model_name.startswith("openrouter:"):
        return True
    if model_name.startswith("exl3:") or model_name.startswith("vllm:") or model_name.startswith("vllm/") or "/" not in model_name:
        return False

    provider = model_name.split("/", 1)[0].lower()

    # Treat openai/* models as native OpenAI to avoid misrouting
    if provider == "openai":
        return False

    return True


def is_ollama_model(model_name: str) -> bool:
    """Check if a model is an Ollama model"""
    return (not is_openai_model(model_name) and
            not is_exllamav3_model(model_name) and
            not is_vllm_model(model_name) and
            not is_openrouter_model(model_name))



# ==========================================
# OPTIMIZATION CONFIGURATION
# ==========================================

@dataclass
class OptimizationConfig:
    """Configuration for optimizer defaults."""

    stubborn_threshold: float = float(os.getenv("REPRO_STUBBORN_THRESHOLD", "0.20"))
    stubborn_min_attempts: int = int(os.getenv("REPRO_STUBBORN_MIN_ATTEMPTS", "3"))
    stubborn_mutation_multiplier: float = float(os.getenv("REPRO_STUBBORN_MUTATION_MULTIPLIER", "1.5"))
    stubborn_temp_boost: float = float(os.getenv("REPRO_STUBBORN_TEMP_BOOST", "0.2"))
    easy_task_skip_prob: float = float(os.getenv("REPRO_EASY_TASK_SKIP_PROB", "0.0"))


OPTIMIZATION_CONFIG = OptimizationConfig()




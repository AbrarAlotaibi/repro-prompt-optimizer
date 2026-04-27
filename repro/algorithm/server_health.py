"""
Server Health Management for TabbyAPI/ExLlamaV3 Servers

This module provides automatic health checking and server restart capabilities
to ensure experiment stability during long-running optimization runs.
"""

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import aiohttp

logger = logging.getLogger(__name__)

# Server configurations
TABBY_DIR = os.getenv("TABBY_DIR", "/hdd/g202203240/tabbyAPI")

@dataclass
class ServerConfig:
    """Configuration for a TabbyAPI server."""
    port: int
    config_file: str
    name: str
    startup_timeout: int = 120  # seconds
    health_timeout: int = 10  # seconds


# Default server configurations
DEFAULT_SERVERS = {
    5040: ServerConfig(5040, "config_mistral_small_3_1_24b_instruct_exl3_4bpw.yml", "Reflective-LM"),
    5000: ServerConfig(5000, "config_llama_3_1_8b_instruct_exl3_4bpw.yml", "Target-Model"),
    5001: ServerConfig(5001, "config_gemma_2_9b_it_exl3_4bpw.yml", "Judge-Model"),
}


class ServerHealthManager:
    """Manages health checking and automatic restart of TabbyAPI servers."""
    
    def __init__(self, 
                 servers: Optional[Dict[int, ServerConfig]] = None,
                 max_restart_attempts: int = 3,
                 restart_cooldown: int = 60):
        """
        Initialize the health manager.
        
        Args:
            servers: Dict mapping port to ServerConfig
            max_restart_attempts: Max consecutive restart attempts before giving up
            restart_cooldown: Seconds to wait between restart attempts
        """
        self.servers = servers or DEFAULT_SERVERS
        self.max_restart_attempts = max_restart_attempts
        self.restart_cooldown = restart_cooldown
        
        # Track restart state
        self._restart_counts: Dict[int, int] = {}
        self._last_restart_time: Dict[int, float] = {}
    
    async def check_health(self, port: int) -> Tuple[bool, Optional[str]]:
        """
        Check if a server is healthy.
        
        Returns:
            Tuple of (is_healthy, error_message)
        """
        url = f"http://127.0.0.1:{port}/health"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "healthy":
                            return True, None
                        else:
                            issues = data.get("issues", [])
                            return False, f"Server unhealthy: {issues}"
                    else:
                        return False, f"HTTP {response.status}"
        except asyncio.TimeoutError:
            return False, "Health check timed out"
        except aiohttp.ClientConnectorError:
            return False, "Connection refused - server not running"
        except Exception as e:
            return False, f"Health check failed: {type(e).__name__}: {e}"
    
    def check_health_sync(self, port: int) -> Tuple[bool, Optional[str]]:
        """Synchronous version of health check."""
        import requests
        
        url = f"http://127.0.0.1:{port}/health"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    return True, None
                else:
                    issues = data.get("issues", [])
                    return False, f"Server unhealthy: {issues}"
            else:
                return False, f"HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            return False, "Health check timed out"
        except requests.exceptions.ConnectionError:
            return False, "Connection refused - server not running"
        except Exception as e:
            return False, f"Health check failed: {type(e).__name__}: {e}"
    
    def _kill_server(self, config: ServerConfig) -> bool:
        """Kill a TabbyAPI server process."""
        config_base = config.config_file.replace(".yml", "")
        
        try:
            # Try graceful kill first
            subprocess.run(
                ["pkill", "-f", f"config_{config_base}"],
                capture_output=True, timeout=5
            )
            time.sleep(2)
            
            # Force kill if needed
            subprocess.run(
                ["pkill", "-9", "-f", f"config_{config_base}"],
                capture_output=True, timeout=5
            )
            
            # Kill by port as backup
            subprocess.run(
                ["fuser", "-k", f"{config.port}/tcp"],
                capture_output=True, timeout=5
            )
            
            time.sleep(3)
            return True
            
        except Exception as e:
            logger.warning(f"Error killing server: {e}")
            return False
    
    def _start_server(self, config: ServerConfig) -> bool:
        """Start a TabbyAPI server."""
        try:
            # Build startup command
            log_file = f"/tmp/tabby_{config.port}.log"
            
            # Use nohup to detach the process
            cmd = (
                f"cd {TABBY_DIR} && "
                f"source venv_py310/bin/activate && "
                f"nohup python3 main.py --config configs/{config.config_file} "
                f"> {log_file} 2>&1 &"
            )
            
            subprocess.run(cmd, shell=True, executable="/bin/bash")
            
            logger.info(f"Started {config.name} startup process on port {config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def restart_server(self, port: int) -> bool:
        """
        Restart a server with cooldown and attempt limiting.
        
        Returns:
            True if restart was successful
        """
        if port not in self.servers:
            logger.error(f"Unknown server port: {port}")
            return False
        
        config = self.servers[port]
        
        # Check cooldown
        now = time.time()
        last_restart = self._last_restart_time.get(port, 0)
        time_since = now - last_restart
        
        if time_since < self.restart_cooldown:
            logger.warning(
                f"Cooldown active for {config.name}, "
                f"skipping restart ({time_since:.0f}s < {self.restart_cooldown}s)"
            )
            return False
        
        # Check restart count
        count = self._restart_counts.get(port, 0)
        if count >= self.max_restart_attempts:
            logger.error(
                f"Max restart attempts ({self.max_restart_attempts}) "
                f"reached for {config.name}"
            )
            return False
        
        logger.info(
            f"Restarting {config.name} on port {port} "
            f"(attempt {count + 1}/{self.max_restart_attempts})..."
        )
        
        # Kill existing server
        self._kill_server(config)
        
        # Start new server
        if not self._start_server(config):
            self._restart_counts[port] = count + 1
            self._last_restart_time[port] = now
            return False
        
        # Wait for server to be healthy
        logger.info(f"Waiting for {config.name} to initialize...")
        
        wait_time = 0
        while wait_time < config.startup_timeout:
            time.sleep(5)
            wait_time += 5
            
            is_healthy, _ = self.check_health_sync(port)
            if is_healthy:
                logger.info(f"{config.name} restarted successfully!")
                self._restart_counts[port] = 0  # Reset on success
                self._last_restart_time[port] = now
                return True
            
            logger.debug(f"Still waiting for {config.name}... ({wait_time}s)")
        
        logger.error(f"{config.name} failed to start within {config.startup_timeout}s")
        self._restart_counts[port] = count + 1
        self._last_restart_time[port] = now
        return False
    
    async def ensure_healthy_or_restart(self, port: int) -> bool:
        """
        Check if server is healthy, restart if not.
        
        Returns:
            True if server is healthy (either already or after restart)
        """
        is_healthy, error = await self.check_health(port)
        
        if is_healthy:
            return True
        
        logger.warning(f"Server on port {port} unhealthy: {error}")
        
        # Run restart in thread pool to not block async loop
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, self.restart_server, port)
        
        return success
    

# Global instance
_health_manager: Optional[ServerHealthManager] = None


def get_health_manager() -> ServerHealthManager:
    """Get or create the global health manager instance."""
    global _health_manager
    if _health_manager is None:
        _health_manager = ServerHealthManager()
    return _health_manager


async def check_and_restart_if_needed(port: int) -> bool:
    """
    Convenience function to check health and restart if needed.
    
    Returns:
        True if server is healthy
    """
    return await get_health_manager().ensure_healthy_or_restart(port)


def extract_port_from_url(url: str) -> Optional[int]:
    """Extract port number from a URL."""
    import re
    match = re.search(r':(\d+)', url)
    if match:
        return int(match.group(1))
    return None

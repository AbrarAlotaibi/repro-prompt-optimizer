#!/usr/bin/env python3
"""
Rich CLI display for system information
Provides clean, organized display of GPU, model, and system configuration
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

class SystemDisplay:
    """Enhanced CLI display for system information"""

    def __init__(self):
        self.console = Console()

    def print_gpu_configuration(self, system_config):
        """Print GPU configuration in a clean table"""
        gpu_table = Table(title="GPU Configuration", title_style="bold green")
        gpu_table.add_column("Parameter", style="bold white", width=20)
        gpu_table.add_column("Value", style="cyan", width=30)
        gpu_table.add_column("Details", style="dim white", width=35)

        # GPU count and detection
        gpu_count = len(system_config.detected_gpus)
        gpu_table.add_row(
            "Detected GPUs",
            str(gpu_count),
            f"{gpu_count} GPU{'s' if gpu_count != 1 else ''} available"
        )

        # GPU details
        if system_config.detected_gpus:
            for gpu in system_config.detected_gpus:
                gpu_table.add_row(
                    f"GPU {gpu['id']}",
                    gpu['name'],
                    f"{gpu['memory_total']}GB memory"
                )

        # Performance settings
        gpu_table.add_row(
            "Performance Profile",
            system_config.performance_profile,
            "Optimization level"
        )
        gpu_table.add_row(
            "GPU Layers",
            str(system_config.gpu_layers),
            "Layers offloaded to GPU"
        )
        gpu_table.add_row(
            "Multi-GPU",
            "Enabled" if system_config.enable_multi_gpu else "Disabled",
            "Parallel GPU usage"
        )

        self.console.print(gpu_table)

    def print_ollama_status(self, client, cuda_devices=None):
        """Print Ollama connection status and available models"""
        # Connection status
        models_count = len(client.available_models) if hasattr(client, 'available_models') else 0

        ollama_table = Table(title="Ollama Configuration", title_style="bold blue")
        ollama_table.add_column("Parameter", style="bold white", width=20)
        ollama_table.add_column("Value", style="blue", width=30)
        ollama_table.add_column("Details", style="dim white", width=35)

        ollama_table.add_row(
            "Connection Status",
            "[green]Connected[/green]",
            f"{models_count} models available"
        )

        if cuda_devices:
            ollama_table.add_row(
                "GPU Assignment",
                f"GPUs {cuda_devices}",
                "Multi-GPU active"
            )

        # Sample models (first few)
        if hasattr(client, 'available_models') and client.available_models:
            model_list = client.available_models[:3]
            remaining = len(client.available_models) - 3

            models_display = ", ".join(model_list)
            if remaining > 0:
                models_display += f" (+{remaining} more)"

            ollama_table.add_row(
                "Available Models",
                models_display,
                "Sample of loaded models"
            )

        self.console.print(ollama_table)

    def print_openai_status(self, client):
        """Print OpenAI connection status and available models"""
        models_count = len(client.available_models) if hasattr(client, 'available_models') else 0

        openai_table = Table(title="OpenAI Configuration", title_style="bold magenta")
        openai_table.add_column("Parameter", style="bold white", width=20)
        openai_table.add_column("Value", style="magenta", width=30)
        openai_table.add_column("Details", style="dim white", width=35)

        openai_table.add_row(
            "Connection Status",
            "[green]Connected[/green]",
            f"{models_count} models available"
        )

        # Available models
        if hasattr(client, 'available_models') and client.available_models:
            models_display = ", ".join(client.available_models)
            openai_table.add_row(
                "Available Models",
                models_display,
                "OpenAI API models"
            )

        self.console.print(openai_table)

    def print_exllamav3_status(self, client, system_config=None):
        """Print ExLlamaV3/TabbyAPI connection status and GPU configuration"""
        exl_table = Table(title="ExLlamaV3 (TabbyAPI) Configuration", title_style="bold cyan")
        exl_table.add_column("Parameter", style="bold white", width=20)
        exl_table.add_column("Value", style="cyan", width=30)
        exl_table.add_column("Details", style="dim white", width=35)

        # Connection status
        exl_table.add_row(
            "Connection Status",
            "[green]Connected[/green]",
            f"TabbyAPI endpoint active"
        )

        # GPU information from system config
        if system_config and system_config.detected_gpus:
            for gpu in system_config.detected_gpus:
                exl_table.add_row(
                    f"GPU {gpu['id']}",
                    gpu['name'],
                    f"{gpu['memory_total']}GB VRAM"
                )

        # Model information
        if hasattr(client, 'base_url'):
            exl_table.add_row(
                "Endpoint",
                str(client.base_url),
                "TabbyAPI server"
            )

        # Available models
        if hasattr(client, 'available_models') and client.available_models:
            models_display = ", ".join(client.available_models)
            exl_table.add_row(
                "Loaded Models",
                models_display,
                "Currently active models"
            )

        self.console.print(exl_table)

    def print_configuration_summary(self, strategies=None, contexts=None, context_source=None):
        """Print experiment configuration summary"""
        if not strategies and not contexts:
            return

        config_table = Table(title="Experiment Configuration", title_style="bold yellow")
        config_table.add_column("Component", style="bold white", width=20)
        config_table.add_column("Configuration", style="yellow", width=40)
        config_table.add_column("Status", style="green", width=25)

        if strategies:
            strategy_list = ", ".join(strategies[:3])
            if len(strategies) > 3:
                strategy_list += f" (+{len(strategies) - 3} more)"

            config_table.add_row(
                "Attack Strategies",
                strategy_list,
                f"{len(strategies)} strategies loaded"
            )

        if contexts:
            config_table.add_row(
                "Context Data",
                context_source or "Loaded contexts",
                f"{len(contexts)} contexts available"
            )

        self.console.print(config_table)

    def print_system_ready(self):
        """Print system ready status"""
        ready_panel = Panel(
            "[bold green]System Initialization Complete[/bold green]\n"
            "[cyan]All components loaded and ready for operation[/cyan]",
            title="Status",
            style="green"
        )
        self.console.print(ready_panel)

    def print_divider(self):
        """Print a divider line"""
        self.console.print("\n" + "-" * 80, style="dim")

    def print_section_header(self, title: str):
        """Print a section header"""
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]")
        self.console.print("=" * len(title), style="cyan")

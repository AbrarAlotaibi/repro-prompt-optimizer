#!/usr/bin/env python3
"""
Enhanced Rich CLI display for Bootstrap Training
Provides clean, colorful, and informative output
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Dict, List, Any

class BootstrapDisplay:
    """Enhanced CLI display for bootstrap training"""

    def __init__(self):
        self.console = Console()

    def print_header(self):
        """Print the main header"""
        self.console.print("\n" + "=" * 80, style="bold blue")
        self.console.print("RED TEAM FRAMEWORK - BOOTSTRAP TRAINING", style="bold red", justify="center")
        self.console.print("=" * 80, style="bold blue")

    def print_config(self, config: Dict[str, Any]):
        """Print experiment configuration in a table"""
        config_table = Table(title="Experiment Configuration", title_style="bold cyan")
        config_table.add_column("Parameter", style="bold white", width=20)
        config_table.add_column("Value", style="cyan", width=25)
        config_table.add_column("Parameter", style="bold white", width=20)
        config_table.add_column("Value", style="cyan", width=25)

        config_table.add_row(
            "Attacker", str(config.get('attacker', 'N/A')),
            "Target", str(config.get('target', 'N/A'))
        )
        config_table.add_row(
            "Judge", str(config.get('judges', 'N/A')),
            "Dataset", str(config.get('dataset', 'N/A'))
        )
        config_table.add_row(
            "Training Cycles", str(config.get('training_cycles', 'N/A')),
            "Prompts/Cycle", str(config.get('num_cycles', 'N/A'))
        )
        config_table.add_row(
            "Template", str(config.get('template', 'N/A')),
            "Contexts", f"{config.get('contexts_loaded', 0)} loaded"
        )

        self.console.print(config_table)
        self.console.print()

    def print_cycle_header(self, cycle: int, total_cycles: int):
        """Print cycle header"""
        self.console.print(f"\n[bold yellow]Training Cycle {cycle}/{total_cycles}[/bold yellow]")

    def print_attack_results(self, cycle: int, attacks: List[Dict[str, Any]]):
        """Print attack generation results for a cycle"""
        attack_table = Table(
            title=f"Attack & Response Results - Cycle {cycle}",
            title_style="bold yellow"
        )
        attack_table.add_column("#", width=3, style="dim")
        attack_table.add_column("Context", width=25, style="white")
        attack_table.add_column("Attack Preview", width=25, style="cyan")
        attack_table.add_column("Target Response", width=25, style="magenta")
        attack_table.add_column("Self Eval", width=8, justify="center")

        for i, attack in enumerate(attacks[:10]):  # Limit to first 10 for display
            context = attack.get('context', 'N/A')
            attack_prompt = attack.get('attack_prompt', 'N/A')
            target_response = attack.get('target_response', 'N/A')
            self_eval = attack.get('attacker_self_eval', {})

            # Truncate text for display
            context_display = (context[:22] + "...") if len(context) > 25 else context
            attack_display = (attack_prompt[:22] + "...") if len(attack_prompt) > 25 else attack_prompt
            response_display = (target_response[:22] + "...") if len(target_response) > 25 else target_response

            # Format self evaluation
            if isinstance(self_eval, dict):
                success = self_eval.get('success', False)
                eval_display = "[green]PASS[/green]" if success else "[red]FAIL[/red]"
            else:
                eval_display = "[dim]N/A[/dim]"

            attack_table.add_row(
                str(i + 1),
                context_display,
                attack_display,
                response_display,
                eval_display
            )

        self.console.print(attack_table)

    def print_judge_results(self, evaluations: List[Dict[str, Any]]):
        """Print judge evaluation results"""
        eval_table = Table(title="Judge Evaluation Results", title_style="bold magenta")
        eval_table.add_column("#", width=3, style="dim")
        eval_table.add_column("Judge Model", width=15, style="white")
        eval_table.add_column("Decision", width=10, justify="center")
        eval_table.add_column("Confidence", width=12, justify="center")
        eval_table.add_column("Reasoning", width=45, style="dim white")

        for i, eval_data in enumerate(evaluations[:10]):  # Limit display
            judge_eval = eval_data.get('judge_eval', {})

            if isinstance(judge_eval, dict):
                # Handle single judge
                judgments = judge_eval.get('individual_judgments', [judge_eval])
            else:
                judgments = []

            for j, judgment in enumerate(judgments):
                if i == 0 and j == 0:  # Only show first few for space
                    judge_model = judgment.get('metadata', {}).get('model', 'Unknown')
                    success = judgment.get('success', False)
                    confidence = judgment.get('confidence', 0.0)
                    reasoning = judgment.get('reasoning', 'No reasoning provided')

                    decision_display = "[green]SUCCESS[/green]" if success else "[red]FAIL[/red]"
                    confidence_display = f"[yellow]{confidence:.2f}[/yellow]"
                    reasoning_display = (reasoning[:42] + "...") if len(reasoning) > 45 else reasoning

                    eval_table.add_row(
                        str(i + 1),
                        judge_model,
                        decision_display,
                        confidence_display,
                        reasoning_display
                    )

        self.console.print(eval_table)

    def print_cycle_summary(self, cycle: int, stats: Dict[str, Any]):
        """Print cycle summary"""
        success_rate = stats.get('success_rate', 0.0)
        avg_confidence = stats.get('avg_confidence', 0.0)
        agreement = stats.get('agreement', 0.0)
        total_attacks = stats.get('total_attacks', 0)
        successful_attacks = stats.get('successful_attacks', 0)

        summary_panel = Panel(
            f"[bold]Cycle {cycle} Summary:[/bold]\n"
            f"- Success Rate: [green]{successful_attacks}/{total_attacks} ({success_rate:.1f}%)[/green]\n"
            f"- Average Confidence: [yellow]{avg_confidence:.2f}[/yellow]\n"
            f"- Judge Agreement: [cyan]{agreement:.1f}%[/cyan]",
            title="Summary",
            style="blue"
        )
        self.console.print(summary_panel)

    def print_training_progress(self, current_cycle: int, total_cycles: int, overall_asr: float):
        """Print overall training progress"""
        progress_panel = Panel(
            f"[bold]Training Progress:[/bold]\n"
            f"- Current Cycle: [yellow]{current_cycle}/{total_cycles}[/yellow]\n"
            f"- Overall ASR: [green]{overall_asr:.1f}%[/green]\n"
            f"- Status: [cyan]In Progress[/cyan]",
            title="Progress",
            style="green"
        )
        self.console.print(progress_panel)

    def print_final_results(self, results: Dict[str, Any]):
        """Print final training results"""
        self.console.print("\n")
        final_table = Table(title="Bootstrap Training Results", title_style="bold green")
        final_table.add_column("Metric", style="bold white", width=25)
        final_table.add_column("Value", style="green", width=20)
        final_table.add_column("Details", style="cyan", width=20)

        overall_asr = results.get('overall_asr', 0.0)
        total_cycles = results.get('total_cycles', 0)
        best_strategy = results.get('best_strategy', 'N/A')
        best_asr = results.get('best_asr', 0.0)
        avg_agreement = results.get('avg_agreement', 0.0)

        final_table.add_row("Overall Success Rate", f"{overall_asr:.1f}%", "Final ASR")
        final_table.add_row("Best Strategy", best_strategy, f"{best_asr:.1f}% success")
        final_table.add_row("Judge Agreement", f"{avg_agreement:.1f}%", "Average")
        final_table.add_row("Training Cycles", str(total_cycles), "Completed")

        self.console.print(final_table)

    def print_status(self, status: str, details: str = "", style: str = "green"):
        """Print status information"""
        status_panel = Panel(
            f"[{style}]{status}[/{style}]\n{details}",
            title="Status",
            style=style
        )
        self.console.print(status_panel)

    def print_divider(self):
        """Print a divider line"""
        self.console.print("\n" + "-" * 80, style="dim")

    def print_footer(self):
        """Print footer"""
        self.console.print("\n" + "=" * 80, style="bold blue")

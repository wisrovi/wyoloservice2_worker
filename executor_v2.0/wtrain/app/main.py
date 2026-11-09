"""WTrain Service execution entrypoint using WPipe pipeline framework."""

import argparse
import os
import shutil
from typing import Any, Dict
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from setproctitle import setproctitle
from wpipe import ResourceMonitor, TaskTimer

from pipelines import worker_wtrain_pipeline

setproctitle("wtrain-service")

console = Console()

# WORKER/EXECUTOR VERSION
__VERSION__ = "2.2.41"


def display_banner() -> None:
    """Render terminal system info banner using Rich styling."""
    info_table = Table(show_header=False, box=None, padding=(0, 1))
    info_table.add_column("Key", style="bold cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("Script:", "WTrain Service")
    info_table.add_row("Version:", __VERSION__)
    info_table.add_row("Author:", "Wisrovi Rodriguez - wisrovi")
    info_table.add_row("Python:", "3.10+")
    info_table.add_row("Status:", "[bold green]Ready[/bold green]")

    panel = Panel(
        info_table,
        title="[bold yellow] SYSTEM INFO [/bold yellow]",
        subtitle="[dim]Press CTRL+C to exit[/dim]",
        border_style="bright_blue",
        padding=(1, 2),
        expand=False,
    )
    console.print(panel)


def main(user_config_train: Dict[str, Any]) -> Dict[str, Any]:
    """Execute primary wtrain training pipeline with resource monitoring.

    Args:
        user_config_train (Dict[str, Any]): Dictionary containing user_config_train path.

    Returns:
        Dict[str, Any]: Pipeline execution result mapping.
    """
    artifacts_path = "/wyolo/worker/train_service_results"
    if os.path.exists(artifacts_path):
        for item in os.listdir(artifacts_path):
            if item == "extras" or item.startswith("logs_"):
                continue
            item_path = os.path.join(artifacts_path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as clean_error:
                print(f"Failed to clean {item_path}: {clean_error}")

    with ResourceMonitor("wtrain_ResourceMonitor") as monitor:
        with TaskTimer("wtrain_TaskTimer", timeout_seconds=900) as timer:
            results = worker_wtrain_pipeline.run(user_config_train)

    print("\nResource Summary:")
    summary = monitor.get_summary()
    print(f"  - Peak RAM: {summary.get('peak_ram_mb', 0)} MB")
    print(f"  - Avg CPU: {summary.get('avg_cpu_percent', 0)}%")
    print(f"  - Elapsed Time: {timer.elapsed_seconds:.2f}s")

    return results


def parse_arguments() -> argparse.Namespace:
    """Parse CLI input arguments for training pipeline execution.

    Returns:
        argparse.Namespace: Parsed command-line arguments container.
    """
    parser = argparse.ArgumentParser(description="Train model with user config")
    parser.add_argument(
        "--file",
        type=str,
        default="/wyolo/config_train.yaml",
        help="Path to the training configuration YAML file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args_dict = {"user_config_train": args.file}

    display_banner()
    pipeline_results = main(args_dict)

    console.print(
        Panel(
            Pretty(pipeline_results, expand_all=True),
            title="[bold yellow]DEBUG: Full Pipeline Results[/bold yellow]",
            border_style="yellow",
            expand=False,
        )
    )

    trained_results = pipeline_results.get("results_trained_model", "N/A")
    console.print(
        f"\n[bold cyan]Results (Trained Model):[/bold cyan] [green]{trained_results}[/green]\n"
    )
    print(f"\nResults: {pipeline_results.get('results_trained_model')}")

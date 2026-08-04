from setproctitle import setproctitle
from wpipe import (
    Condition,
    Pipeline,
    ResourceMonitor,
    TaskTimer,
)
import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.pretty import Pretty


from states import (
    error_capture,
    check_dataset,
    check_gpu_available,
    check_minio_buckets,
    train_model,
    load_yaml,
    publish_results_mlflow,
    not_train,
    publish_request_redis,
    publish_results_redis,
)


setproctitle("wtrain-service")

console = Console()

# WORKER/EXECUTOR VERSION
__VERSION__ = "2.2.9"


def display_banner():
    # Creamos una tabla limpia sin bordes para estructurar los datos
    info_table = Table(show_header=False, box=None, padding=(0, 1))
    info_table.add_column("Key", style="bold cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("🚀 Script:", "Wtrain Service")
    info_table.add_row("📌 Versión:", __VERSION__)
    info_table.add_row("👤 Autor:", "Wisrovi Rodríguez - wisrovi")
    info_table.add_row("🐍 Python:", "3.12+")
    info_table.add_row("⚡ Estado:", "[bold green]Ready[/bold green]")

    # Envolvemos la tabla en un Panel estilizado
    panel = Panel(
        info_table,
        title="[bold yellow] SYSTEM INFO [/bold yellow]",
        subtitle="[dim]Press CTRL+C to exit[/dim]",
        border_style="bright_blue",
        padding=(1, 2),
        expand=False,
    )

    console.print(panel)


def config_pipeline():
    pipeline = Pipeline(
        pipeline_name="wtrain_pipe",
        verbose=False,
        tracking_db="/wyolo/worker/events/wtrain.db",
        #
        max_retries=3,  # Retry up to 3 times
        retry_delay=0.5,  # Wait 0.5 seconds between retries
        retry_on_exceptions=(RuntimeError,),  # Only retry on RuntimeError
        #
        collect_system_metrics=True,  # Enable metrics collection
        #
        show_progress=False,
    )

    pipeline.add_error_capture([error_capture])

    if pipeline:
        pipeline.set_steps(
            [
                load_yaml,
                publish_request_redis,
                check_minio_buckets,
                check_gpu_available,
                check_dataset,
                Condition(
                    expression="gpu_status == 1 and dataset_status == 1",
                    branch_true=[
                        train_model,
                        publish_results_mlflow,
                    ],
                    branch_false=[
                        not_train,
                    ],
                ),
                publish_results_redis,
            ]
        )

    return pipeline


def main(user_config_train):
    # Clean the entire train_service_results directory to prevent leakage from previous runs
    import os
    import shutil

    artifacts_path = "/wyolo/worker/train_service_results"
    if os.path.exists(artifacts_path):
        for item in os.listdir(artifacts_path):
            item_path = os.path.join(artifacts_path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as clean_error:
                print(f"Failed to clean {item_path}: {clean_error}")

    pipeline = config_pipeline()

    with ResourceMonitor("eyesdcar_pipeline_ResourceMonitor") as monitor:
        with TaskTimer("eyesdcar_pipeline_TaskTimer", timeout_seconds=900) as timer:
            results = pipeline.run(user_config_train)

            if timer.exceeded_timeout():
                # print("⚠ Work exceeded timeout!")
                pass
            else:
                # print("✓ Work completed within timeout")
                pass

    # Resumen de recursos al terminar
    print(f"\nResource Summary:")
    summary = monitor.get_summary()
    print(f"  - Peak RAM: {summary['peak_ram_mb']} MB")
    print(f"  - Avg CPU: {summary['avg_cpu_percent']}%")
    print(f"✓ Total time monitored: {timer.elapsed_seconds:.2f}s")

    # if "error" in results:
    #     print(f"Error detectado: {results.get('error')}")

    return results


def get_argument(arg_name, default=None):
    parser = argparse.ArgumentParser(description="Train model with user config")
    parser.add_argument(
        f"--{arg_name}",
        type=str,
        default=default,
        help=f"Path to the {arg_name} YAML file",
    )
    args = parser.parse_args()
    return getattr(args, arg_name)


if __name__ == "__main__":
    # python main.py --file /wyolo/control_server/datasets/clasification/colorball.v8i.multiclass/config_train.yaml
    # python main.py --file "/wyolo/worker/request/config_train_CLS.yaml"
    # python -m wpipe.dashboard --db /wyolo/worker/events/wtrain.db --port 8036

    _user_config_file = get_argument("file", default="/wyolo/config_train.yaml")

    args_dict = {"user_config_train": _user_config_file}

    display_banner()

    results = main(args_dict)

    # 1. Full Pipeline Results en un panel estilizado de debug
    console.print(
        Panel(
            Pretty(results, expand_all=True),
            title="[bold yellow]🔍 DEBUG: Full Pipeline Results[/bold yellow]",
            border_style="yellow",
            expand=False,
        )
    )

    # 2. Results especifica del modelo entrenado
    trained_results = results.get("results_trained_model", "N/A")
    console.print(
        f"\n[bold cyan]📊 Results (Trained Model):[/bold cyan] [green]{trained_results}[/green]\n"
    )

    # print(f"\n--- [DEBUG] Full Pipeline Results: {results} ---")
    print(f"\nResults: {results.get('results_trained_model')}")

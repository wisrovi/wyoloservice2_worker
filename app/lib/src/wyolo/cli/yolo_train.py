"""CLI module for Wyolo - Command-line interface for YOLO training."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import click
import yaml

from wyolo.training.trainer_wrapper import TrainerWrapper


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML

    Example:
        >>> config = load_config("config.yaml")
        >>> print(config["model"])
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def get_datetime() -> str:
    """Get current timestamp as formatted string.

    Returns:
        Current timestamp in YYYYMMDD_HHMMSS format

    Example:
        >>> timestamp = get_datetime()
        >>> print(f"Generated at: {timestamp}")
    """
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def create_trainer(
    config_path: str, trial_number: int
) -> Tuple[TrainerWrapper, Dict[str, Any]]:
    """Create a configured trainer instance from configuration file.

    Args:
        config_path: Path to the YAML configuration file
        trial_number: Trial number for this training run

    Returns:
        Tuple of (TrainerWrapper instance, updated configuration dictionary)

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid

    Example:
        >>> trainer, config = create_trainer("config.yaml", 1)
        >>> print(f"Model: {config['model']}")
    """
    request_config = load_config(config_path=config_path)
    request_config["config_path"] = config_path

    trainer = TrainerWrapper(config=request_config)

    # Setup MLflow experiment name before creating model
    if "mlflow" in request_config and "sweeper" in request_config:
        import mlflow

        tracking_uri = request_config["mlflow"].get("MLFLOW_TRACKING_URI")
        experiment_name = request_config["sweeper"].get("study_name")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

    trainer.create_model(
        model_name=request_config["model"],
        model_type=request_config["type"],
    )

    if "task_id" not in request_config:
        request_config["task_id"] = str(uuid.uuid4())

    experiment_name = request_config.get("sweeper", {}).get(
        "study_name", "default_experiment"
    )
    tempfile = request_config.get("tempfile", "")

    result_path = (
        Path(tempfile)
        / "models"
        / experiment_name
        / request_config["type"]
        / request_config["task_id"]
    )
    result_path.mkdir(parents=True, exist_ok=True)

    trial_history_path = result_path / "trail_history"
    trial_history_path.mkdir(exist_ok=True)

    request_config["path_results"] = str(result_path / str(trial_number))

    timestamp = get_datetime()
    request_config["timestamp"] = timestamp

    # Optimize batch size if specified
    if request_config.get("train", {}).get("batch", 0) > 0:
        better_batch = trainer.get_better_batch(
            batch_to_use=request_config["train"]["batch"]
        )
        if request_config["train"]["batch"] > better_batch:
            request_config["train"]["batch"] = better_batch

    # Configure training parameters
    request_config["train"]["project"] = str(result_path / str(trial_number))
    request_config["train"]["name"] = f"train_{request_config.get('task_id')}"
    request_config["train"]["verbose"] = True
    request_config["train"]["plots"] = True
    request_config["train"]["exist_ok"] = True

    # Disable AWS/S3 integration to avoid credential issues
    request_config["train"]["save"] = True
    request_config["train"]["save_period"] = -1  # Disable periodic saving to S3

    trainer.config_train = request_config

    return trainer, request_config


def train(
    trainer: TrainerWrapper, request_config: Dict[str, Any], fitness: str
) -> Dict[str, Any]:
    """Execute training with the provided trainer and configuration.

    Args:
        trainer: Configured TrainerWrapper instance
        request_config: Training configuration dictionary
        fitness: Fitness metric name to evaluate and print

    Returns:
        Updated configuration dictionary with training results

    Example:
        >>> trainer, config = create_trainer("config.yaml", 1)
        >>> results = train(trainer, config, "mAP")
        >>> print(f"Final result: {results['train']['results'][fitness]}")
    """
    if "train" in request_config:
        results = trainer.train(config_train=request_config["train"])

        if results:
            request_config["experiment_type"] = str(results.task)
            request_config["train"]["results"] = results.results_dict

            try:
                print(f"ResultadoFinal:{request_config['train']['results'][fitness]}")
            except KeyError:
                print(f"ResultadoFinal:{request_config['train']['results']['fitness']}")

    return request_config


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    required=True,
    default="config.yaml",
    help="Path to the configuration YAML file.",
)
@click.option(
    "--fitness",
    type=str,
    required=True,
    default="fitness",
    help="Fitness metric name to evaluate.",
)
@click.option(
    "--trial_number",
    type=int,
    required=True,
    default=0,
    help="Trial number for this training run.",
)
def send_train(config_path: str, fitness: str, trial_number: int) -> Dict[str, Any]:
    """Execute training from command line with specified parameters.

    This is the main entry point for the CLI tool. It creates a trainer
    from the configuration file and executes training.

    Args:
        config_path: Path to configuration file
        fitness: Fitness metric to evaluate
        trial_number: Trial number for this run

    Returns:
        Updated configuration with training results

    Example:
        $ wyolo-train --config_path=config.yaml --fitness=mAP --trial_number=1
    """
    trainer, request_config = create_trainer(
        config_path=config_path, trial_number=trial_number
    )

    request_config = train(trainer, request_config, fitness)

    return request_config


if __name__ == "__main__":
    # Example usage of the training CLI.
    # Example:
    #     python yolo_train.py --config_path="config.yaml" --trial_number=1 --fitness=mAP
    # Example usage - uncomment and modify as needed
    # request_config = train(trainer, request_config, fitness)
    print(
        "Use: python yolo_train.py --config_path <path> --fitness <metric> "
        "--trial_number <number>"
    )

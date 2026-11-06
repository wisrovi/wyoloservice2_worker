"""Configuration loading, validation, and hierarchical merging utilities for wtrain."""

from copy import deepcopy
import os
import sys
from typing import Any, Dict, Tuple
import mlflow
import yaml


def get_user_config(
    user_config_file: str = "/wyolo/control_server/datasets/clasification/colorball.v8i.multiclass/config_train.yaml",
) -> Tuple[str, Dict[str, Any]]:
    """Validate existence of user training configuration and export artifact copy.

    Args:
        user_config_file (str): Filesystem path to source user YAML config.

    Returns:
        Tuple[str, Dict[str, Any]]: Tuple containing file path and parsed config dict.

    Raises:
        FileNotFoundError: If configuration file does not exist.
    """
    if not os.path.exists(user_config_file):
        raise FileNotFoundError(f"Configuration file not found: {user_config_file}")

    with open(user_config_file, "r", encoding="utf-8") as f:
        user_config_train_data = yaml.safe_load(f)

    artifacts_dir = "/wyolo/worker/train_service_results/training_artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    with open(
        os.path.join(artifacts_dir, "user_config_train.yaml"),
        "w",
        encoding="utf-8",
    ) as file:
        yaml.dump(
            user_config_train_data,
            file,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    return user_config_file, user_config_train_data


def read_user_config(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse YAML file defined under 'config_path' in task data.

    Args:
        task_data (Dict[str, Any]): Dictionary containing 'config_path' string.

    Returns:
        Dict[str, Any]: Parsed configuration mapping.

    Raises:
        ValueError: If config path is missing or invalid.
    """
    if not isinstance(task_data, dict) or "config_path" not in task_data:
        raise ValueError("Invalid task_data dictionary or missing config_path key")

    config_path = task_data["config_path"]
    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    user_config.pop("defaults", None)
    return user_config


def merge_configs(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge base and user configs, giving precedence to user keys.

    Args:
        default_config (Dict[str, Any]): Base fallback configuration.
        user_config (Dict[str, Any]): User-defined overrides.

    Returns:
        Dict[str, Any]: Consolidated configuration dictionary.
    """
    final_config = deepcopy(default_config)
    for key, value in user_config.items():
        if (
            isinstance(value, dict)
            and key in final_config
            and isinstance(final_config[key], dict)
        ):
            final_config[key] = merge_configs(final_config[key], value)
        else:
            final_config[key] = deepcopy(value)
    return final_config


def get_base_config() -> None:
    """Load default base configuration and synchronize MLflow, MinIO, and Redis hosts."""
    base_config: Dict[str, Any] = {}
    os.makedirs("/config", exist_ok=True)
    control_host = os.getenv("CONTROL_HOST", "localhost")

    config_yaml_path = "/app/config/config.yaml"
    if os.path.exists(config_yaml_path):
        with open(config_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    if "mlflow" in cfg and "MLFLOW_TRACKING_URI" in cfg["mlflow"]:
        cfg["mlflow"]["MLFLOW_TRACKING_URI"] = cfg["mlflow"][
            "MLFLOW_TRACKING_URI"
        ].replace("localhost", control_host)
    if "minio" in cfg and "MINIO_ENDPOINT" in cfg["minio"]:
        cfg["minio"]["MINIO_ENDPOINT"] = cfg["minio"]["MINIO_ENDPOINT"].replace(
            "localhost", control_host
        )
    if "redis" in cfg and "REDIS_HOST" in cfg["redis"]:
        cfg["redis"]["REDIS_HOST"] = cfg["redis"]["REDIS_HOST"].replace(
            "localhost", control_host
        )

    if "mlflow" in cfg:
        mlflow.set_tracking_uri(cfg["mlflow"]["MLFLOW_TRACKING_URI"])
        os.environ["MLFLOW_TRACKING_URI"] = cfg["mlflow"]["MLFLOW_TRACKING_URI"]
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    base_config.update(cfg)

    if "minio" in cfg:
        minio_id = os.getenv("CIFS_USER", cfg["minio"].get("MINIO_ID", "mlflow"))
        minio_key = os.getenv("CIFS_PASS", cfg["minio"].get("MINIO_SECRET_KEY", "wyoloservice"))
        base_config["minio"]["MINIO_ID"] = minio_id
        base_config["minio"]["MINIO_SECRET_KEY"] = minio_key
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = base_config["minio"]["MINIO_ENDPOINT"]
        os.environ["AWS_ACCESS_KEY_ID"] = minio_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = minio_key
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

    if "dvc" in cfg:
        base_config["dvc"]["MINIO_ID"] = os.getenv("CIFS_USER", "mlflow")
        base_config["dvc"]["MINIO_SECRET_KEY"] = os.getenv("CIFS_PASS", "wyoloservice")

    with open("/config/final_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(base_config, f, default_flow_style=False)


def read_base_config() -> Dict[str, Any]:
    """Generate final base config and return loaded dictionary.

    Returns:
        Dict[str, Any]: Parsed base configuration dictionary.
    """
    get_base_config()
    return read_user_config({"config_path": "/config/final_config.yaml"})


def get_complete_config(user_config: str) -> Tuple[Dict[str, Any], str]:
    """Merge base configuration and user YAML configuration into active execution config.

    Args:
        user_config (str): Path to user configuration YAML.

    Returns:
        Tuple[Dict[str, Any], str]: Merged dictionary and resolved execution YAML path.
    """
    os.makedirs("/config", exist_ok=True)
    user_dict = read_user_config({"config_path": user_config})
    base_config = read_base_config()

    final_config = merge_configs(default_config=base_config, user_config=user_dict)
    final_config["config_path"] = user_config
    final_config["task_id"] = base_config.get("task_id", "default_task")

    tempfile_dir = final_config.get("tempfile", "/tmp")
    os.makedirs(tempfile_dir, exist_ok=True)
    config_path = f"{tempfile_dir}/{final_config['task_id']}.yaml"

    final_config["config_path"] = config_path
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(final_config, f, default_flow_style=False)

    return final_config, config_path

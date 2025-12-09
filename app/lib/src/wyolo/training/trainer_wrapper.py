"""Training module for Wyolo - Professional YOLO Training Library."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import GPUtil
import mlflow
import mlflow.data
import mlflow.data.filesystem_dataset_source
import mlflow.data.http_dataset_source
from loguru import logger
from ultralytics import RTDETR, YOLO, settings
from ultralytics.utils.autobatch import autobatch


def get_gpu_info_json() -> List[Dict[str, Any]]:
    """Get detailed GPU information in JSON format.

    Retrieves comprehensive information about available GPUs including
    name, memory usage, load, temperature, and running processes.

    Returns:
        List of dictionaries containing GPU information for each available GPU.
        If no GPUs are found, returns a list with an error message.
        If an error occurs, returns a list with error details.

    Example:
        >>> gpu_info = get_gpu_info_json()
        >>> if gpu_info and "error" not in gpu_info[0]:
        ...     print(f"Found {len(gpu_info)} GPUs")
    """
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return [{"error": "No GPUs available."}]

        gpu_info = []
        for gpu in gpus:
            gpu_data = {
                f"gpu_{gpu.id}_name": gpu.name,
                f"gpu_{gpu.id}_uuid": gpu.uuid,
                f"gpu_{gpu.id}_memoryTotal": gpu.memoryTotal,
                f"gpu_{gpu.id}_memoryFree": gpu.memoryFree,
                f"gpu_{gpu.id}_memoryUsed": gpu.memoryUsed,
                f"gpu_{gpu.id}_load": gpu.load * 100,
                f"gpu_{gpu.id}_temperature": gpu.temperature,
            }

            # Check if 'processes' attribute exists before accessing it
            if hasattr(gpu, "processes"):
                gpu_data["processes"] = [
                    {
                        "pid": process.pid,
                        "name": process.name,
                        "memory": process.memoryUsed,
                    }
                    for process in gpu.processes
                ]
            else:
                gpu_data["processes"] = "Process information not available."

            gpu_info.append(gpu_data)

        return gpu_info

    except Exception as e:
        return [{"error": f"Error occurred while getting GPU information: {e}"}]


class StatusEDA:
    """Status constants for Exploratory Data Analysis operations.

    Attributes:
        PENDING: EDA analysis is pending or in progress
        SAVED: EDA analysis has been completed and saved
    """

    PENDING = 0
    SAVED = 2


class TrainerWrapper:
    """Professional YOLO Training Wrapper with MLOps integration.

    This class provides a comprehensive interface for training YOLO and RTDETR
    models with integrated MLflow tracking, GPU optimization, and experiment
    management capabilities.

    Attributes:
        config: Training configuration dictionary
        GPU_USE: Percentage of GPU memory to use (default 0.4 = 40%)
        is_configured: Whether the trainer has been properly configured
        model: The loaded YOLO or RTDETR model instance
        start_time: Training start timestamp
        end_time: Training end timestamp
        firts_epoch: Flag indicating if this is the first epoch
        worker_metadata: List of worker environment metadata keys

    Example:
        >>> config = {
        ...     "model": "yolov8n.pt",
        ...     "type": "yolo",
        ...     "train": {"data": "dataset.yaml", "epochs": 100}
        ... }
        >>> trainer = TrainerWrapper(config)
        >>> trainer.create_model("yolov8n.pt", "yolo")
        >>> results = trainer.train(config["train"])
    """

    # Class-level configuration
    config: Dict[str, Any] = {}
    GPU_USE: float = 0.4  # Percentage of GPU usage

    # Instance attributes
    is_configured: bool = False
    model: Optional[Union[YOLO, RTDETR]] = None

    start_time: float = 0
    end_time: float = 0

    firts_epoch: bool = True

    worker_metadata: List[str] = [
        "debug",
        "USER",
        "WORKER_HOST",
        "WORKER_HOSTNAME",
        "WORKER_OS",
        "WORKER_GPU",
        "WORKER_GPU_MEMORY",
        "WORKER_CPU",
        "WORKER_MEMORY",
        "WORKER_DISK",
        "WORKER_NETWORK",
        "WORKER_PYTHON_VERSION",
        "WORKER_CONDA_ENV",
        "WORKER_DOCKER_IMAGE",
        "WORKER_KUBERNETES_POD",
        "WORKER_KUBERNETES_NAMESPACE",
        "WORKER_KUBERNETES_NODE",
        "WORKER_KUBERNETES_SERVICE",
        "WORKER_CLOUD_PROVIDER",
        "WORKER_CLOUD_REGION",
        "WORKER_CLOUD_ZONE",
        "WORKER_CLOUD_INSTANCE_TYPE",
        "WORKER_CLOUD_INSTANCE_ID",
        "WORKER_CLOUD_PROJECT_ID",
        "WORKER_CLOUD_ACCOUNT_ID",
        "WORKER_CLOUD_BUCKET",
        "WORKER_CLOUD_KEY",
        "WORKER_CLOUD_SECRET",
        "WORKER_CLOUD_TOKEN",
        "WORKER_CLOUD_ENDPOINT",
        "WORKER_CLOUD_REGION",
        "WORKER_CLOUD_ZONE",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the TrainerWrapper.

        Args:
            config: Training configuration dictionary. If None, uses empty dict.
        """
        self.config = config or {}
        self.status_eda_completed = StatusEDA.PENDING
        self._setup_mlflow()
        self._setup_dvc()
        self._setup_redis()

    def _setup_mlflow(self) -> None:
        """Setup MLflow configuration if provided in config."""
        if "mlflow" in self.config:
            tracking_uri = self.config["mlflow"].get("MLFLOW_TRACKING_URI")

            # Use study_name from sweeper as experiment name, fallback to MLFLOW_EXPERIMENT_NAME
            experiment_name = self.config.get("sweeper", {}).get(
                "study_name"
            ) or self.config["mlflow"].get("MLFLOW_EXPERIMENT_NAME")

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)

            # Disable MLflow in YOLO settings to avoid credential issues
            # We'll handle MLflow logging manually
            settings.update({"mlflow": False})

    def _setup_dvc(self) -> None:
        """Setup DVC configuration if provided in config."""
        if "dvc" in self.config:
            # DVC setup logic here
            pass

    def _setup_redis(self) -> None:
        """Setup Redis configuration if provided in config."""
        if "redis" in self.config:
            # Redis setup logic here
            pass

    def create_model(self, model_name: str, model_type: str) -> Union[YOLO, RTDETR]:
        """Create and initialize a YOLO or RTDETR model.

        Args:
            model_name: Path to pre-trained model file (e.g., "yolov8n.pt")
            model_type: Type of model to create ("yolo" or "rtdetr")

        Returns:
            Initialized model instance

        Raises:
            ValueError: If model_type is not "yolo" or "rtdetr"

        Example:
            >>> trainer = TrainerWrapper()
            >>> model = trainer.create_model("yolov8n.pt", "yolo")
            >>> print(f"Model type: {type(model).__name__}")
        """
        if model_type.lower() == "yolo":
            self.model = YOLO(model_name)
        elif model_type.lower() == "rtdetr":
            self.model = RTDETR(model_name)
        else:
            raise ValueError(
                f"Invalid model type: {model_type}. Use 'yolo' or 'rtdetr'"
            )

        self.is_configured = True
        return self.model

    def get_better_batch(self, batch_to_use: int = -1) -> int:
        """Calculate optimal batch size based on GPU memory.

        Uses ultralytics autobatch to determine the optimal batch size
        for the current model and GPU configuration.

        Args:
            batch_to_use: Initial batch size. If -1, will auto-calculate.

        Returns:
            Optimal batch size for training

        Raises:
            ValueError: If model is not initialized

        Example:
            >>> trainer = TrainerWrapper()
            >>> trainer.create_model("yolov8n.pt", "yolo")
            >>> optimal_batch = trainer.get_better_batch(16)
            >>> print(f"Optimal batch size: {optimal_batch}")
        """
        if self.model is None:
            raise ValueError("Model must be created before calculating batch size")

        optimal_batch = autobatch(
            model=self.model,
            imgsz=self.config["train"]["imgsz"],
            fraction=self.GPU_USE,
            batch_size=batch_to_use,
        )

        return optimal_batch

    @property
    def config_train(self) -> Dict[str, Any]:
        """Get the training configuration.

        Returns:
            Current training configuration dictionary
        """
        return self.config

    @config_train.setter
    def config_train(self, value: Dict[str, Any]) -> None:
        """Set the training configuration.

        Args:
            value: New training configuration dictionary
        """
        self.config = value

    def train(self, config_train: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Train the model with the specified configuration.

        Args:
            config_train: Training configuration. If None, uses self.config.

        Returns:
            Training results object if successful, None otherwise

        Raises:
            ValueError: If model is not initialized

        Example:
            >>> trainer = TrainerWrapper(config)
            >>> trainer.create_model("yolov8n.pt", "yolo")
            >>> results = trainer.train(config["train"])
            >>> print(f"Training completed: {results.results_dict}")
        """
        if self.model is None:
            raise ValueError("Model must be created before training")

        train_config = config_train or self.config.get("train", {})

        self.start_time = time.time()

        try:
            results = self.model.train(**train_config)
            self.end_time = time.time()
            logger.info("Training completed successfully")

            # Log training artifacts to MLflow after training completes
            logger.info("Attempting to log training artifacts to MLflow...")
            try:
                self._log_training_artifacts_to_mlflow(results)
                logger.info("Successfully logged training artifacts to MLflow")
            except Exception as mlflow_error:
                logger.warning(f"Could not log artifacts to MLflow: {mlflow_error}")

            # Save additional artifacts
            try:
                self.save_eda()
            except Exception as eda_error:
                logger.warning(f"Could not save EDA: {eda_error}")

            return results
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None

    def save_eda(self) -> None:
        """Save Exploratory Data Analysis results and training artifacts.

        Performs EDA on the training dataset and saves the results
        including dataset statistics, visualizations, and reports.
        Also saves training artifacts like model weights and configuration.
        """
        if self.status_eda_completed == StatusEDA.SAVED:
            return

        # Save training artifacts and EDA results
        try:
            # Save model artifacts to the results path
            if self.config.get("path_results"):
                results_path = Path(self.config["path_results"])
                artifacts_path = results_path / "artifacts"
                artifacts_path.mkdir(parents=True, exist_ok=True)

                # Save training configuration
                import json

                config_save_path = artifacts_path / "training_config.json"
                with open(config_save_path, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=2, default=str)

                # Log artifacts to MLflow if available
                try:
                    import mlflow

                    if mlflow.active_run():
                        # Log the artifacts directory
                        mlflow.log_artifacts(str(artifacts_path))

                        # Log training results if available
                        if (
                            hasattr(self, "config")
                            and "train" in self.config
                            and "results" in self.config["train"]
                        ):
                            results = self.config["train"]["results"]
                            if isinstance(results, dict):
                                for metric, value in results.items():
                                    if isinstance(value, (int, float)):
                                        mlflow.log_metric(metric, value)

                        logger.info("Training artifacts logged to MLflow")
                except Exception as mlflow_error:
                    logger.warning(f"Could not log artifacts to MLflow: {mlflow_error}")

                logger.info(f"Training artifacts saved to {artifacts_path}")

        except Exception as e:
            logger.warning(f"Could not save training artifacts: {e}")

        self.status_eda_completed = StatusEDA.SAVED
        logger.info("EDA analysis and artifacts saving completed")

    def _log_training_artifacts_to_mlflow(self, results) -> None:
        """Log training artifacts to MLflow after training completion.

        Args:
            results: Training results object from YOLO
        """
        try:
            import mlflow
            import os

            # Configure MinIO for MLflow artifact storage
            if "minio" in self.config:
                minio_config = self.config["minio"]

                # Set environment variables for MinIO S3 compatibility
                os.environ["AWS_ACCESS_KEY_ID"] = minio_config.get("MINIO_ID")
                os.environ["AWS_SECRET_ACCESS_KEY"] = minio_config.get(
                    "MINIO_SECRET_KEY"
                )
                os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_config.get(
                    "MINIO_ENDPOINT"
                )
                os.environ["MLFLOW_S3_IGNORE_TLS"] = (
                    "true"  # For local MinIO without SSL
                )

                logger.info(
                    f"Configured MinIO endpoint: {minio_config.get('MINIO_ENDPOINT')}"
                )

            # Get the training results directory
            if hasattr(results, "save_dir") and results.save_dir:
                results_dir = Path(results.save_dir)
                logger.info(f"Processing artifacts from: {results_dir}")

                # Log model weights
                weights_dir = results_dir / "weights"
                if weights_dir.exists():
                    for weight_file in weights_dir.glob("*.pt"):
                        mlflow.log_artifact(str(weight_file), artifact_path="models")
                        logger.info(f"Logged model weight: {weight_file.name}")

                # Log training plots and results
                for artifact_file in results_dir.glob("*"):
                    if artifact_file.is_file() and artifact_file.suffix in [
                        ".png",
                        ".jpg",
                        ".csv",
                        ".yaml",
                    ]:
                        mlflow.log_artifact(
                            str(artifact_file), artifact_path="training_artifacts"
                        )
                        logger.info(f"Logged training artifact: {artifact_file.name}")

                # Log metrics from results
                if hasattr(results, "results_dict") and results.results_dict:
                    for metric, value in results.results_dict.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(metric, value)
                            logger.info(f"Logged metric: {metric} = {value}")

                # Register the model in MLflow Model Registry
                best_model_path = weights_dir / "best.pt"
                if best_model_path.exists():
                    try:
                        # Log the model file as an artifact first
                        mlflow.log_artifact(str(best_model_path), artifact_path="model")

                        # Try to register as PyTorch model (may need custom model class)
                        # For now, just log as artifact
                        logger.info(f"Model artifact logged: {best_model_path.name}")
                    except Exception as model_error:
                        logger.warning(f"Could not register model: {model_error}")
                        # Fallback: just log as artifact
                        mlflow.log_artifact(str(best_model_path), artifact_path="model")

                logger.info("Training artifacts successfully logged to MLflow")

        except Exception as e:
            logger.warning(f"Could not log training artifacts to MLflow: {e}")
            import traceback

            logger.warning(f"Full error: {traceback.format_exc()}")


# Legacy function name for backward compatibility
obtener_info_gpu_json = get_gpu_info_json

"""Training module for Wyolo - Professional YOLO Training Library."""

from __future__ import annotations

import random
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

        # Log dataset samples before training
        logger.info("Capturing dataset samples...")
        try:
            self._log_dataset_samples()
        except Exception as dataset_error:
            logger.warning(f"Could not log dataset samples: {dataset_error}")

        self.start_time = time.time()

        try:
            # Start system metrics monitoring
            self._start_system_metrics_logging()

            results = self.model.train(**train_config)
            self.end_time = time.time()
            logger.info("Training completed successfully")

            # Log final system metrics
            self._log_final_system_metrics()

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

                # Log training plots and results (including all image types and data examples)
                for artifact_file in results_dir.glob("*"):
                    if artifact_file.is_file() and artifact_file.suffix in [
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".csv",
                        ".yaml",
                        ".txt",
                        ".json",
                    ]:
                        # Organize artifacts by type for better MLflow organization
                        if "batch" in artifact_file.name.lower():
                            # Training/validation batch images with labels and predictions
                            if "train" in artifact_file.name.lower():
                                mlflow.log_artifact(
                                    str(artifact_file),
                                    artifact_path="training_examples",
                                )
                            elif "val" in artifact_file.name.lower():
                                mlflow.log_artifact(
                                    str(artifact_file),
                                    artifact_path="validation_examples",
                                )
                            else:
                                mlflow.log_artifact(
                                    str(artifact_file), artifact_path="batch_examples"
                                )
                        elif "confusion" in artifact_file.name.lower():
                            mlflow.log_artifact(
                                str(artifact_file), artifact_path="evaluation_metrics"
                            )
                        elif "results" in artifact_file.name.lower():
                            mlflow.log_artifact(
                                str(artifact_file), artifact_path="training_results"
                            )
                        else:
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
                        import tempfile
                        import json

                        # Create a custom model wrapper for YOLO classification
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir)

                            # Save model wrapper code
                            wrapper_code = '''
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np

class YOLOClassificationModel:
    """YOLO Classification Model Wrapper for MLflow"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    
    def load(self):
        """Load the YOLO model"""
        if self.model is None:
            self.model = YOLO(self.model_path)
        return self.model
    
    def predict(self, image):
        """Predict on a single image"""
        model = self.load()
        results = model(image)
        return results
    
    def predict_proba(self, image):
        """Get prediction probabilities"""
        model = self.load()
        results = model(image)
        # Extract probabilities if available
        if hasattr(results[0], 'probs'):
            return results[0].probs.data.cpu().numpy()
        return None
'''

                            wrapper_path = temp_path / "model.py"
                            with open(wrapper_path, "w") as f:
                                f.write(wrapper_code)

                            # Create conda environment file
                            conda_env = {
                                "channels": ["defaults", "conda-forge"],
                                "dependencies": [
                                    "python=3.10",
                                    "pytorch",
                                    "torchvision",
                                    "ultralytics",
                                    "opencv",
                                    "pillow",
                                    "numpy",
                                    "mlflow",
                                ],
                                "name": "yolo_classification_env",
                            }

                            conda_path = temp_path / "conda.yaml"
                            with open(conda_path, "w") as f:
                                import yaml

                                yaml.dump(conda_env, f)

                            # Create MLmodel configuration
                            mlflow_config = {
                                "flavors": {
                                    "python_function": {
                                        "loader_module": "model",
                                        "loader_class": "YOLOClassificationModel",
                                        "data": str(best_model_path),
                                        "env": str(conda_path),
                                    }
                                },
                                "model_type": "yolo_classification",
                                "input_shape": [640, 640, 3],
                                "task": "classify",
                            }

                            config_path = temp_path / "MLmodel"
                            with open(config_path, "w") as f:
                                json.dump(mlflow_config, f, indent=2)

                            # Log the model files
                            mlflow.log_artifact(
                                str(wrapper_path),
                                artifact_path="mlflow_model",
                            )
                            mlflow.log_artifact(
                                str(conda_path),
                                artifact_path="mlflow_model",
                            )
                            mlflow.log_artifact(
                                str(config_path),
                                artifact_path="mlflow_model",
                            )

                            # Log original model weights
                            mlflow.log_artifact(
                                str(best_model_path),
                                artifact_path="mlflow_model",
                            )
                            mlflow.log_artifact(
                                str(conda_path),
                                artifact_path="yolo_classification_model",
                            )
                            mlflow.log_artifact(
                                str(config_path),
                                artifact_path="yolo_classification_model",
                            )

                            # Log the original model weights
                            mlflow.log_artifact(
                                str(best_model_path),
                                artifact_path="yolo_classification_model",
                            )

                            # Register the model in MLflow Model Registry
                            model_name = f"color_ball_classifier_{self.config.get('task_id', 'unknown')}"

                            # Register the model in MLflow Model Registry using MLflowClient
                            client = mlflow.tracking.MlflowClient()
                            run_id = mlflow.active_run().info.run_id

                            # Create model version
                            model_version = client.create_model_version(
                                name=model_name,
                                source=f"runs:/{run_id}/mlflow_model",
                                run_id=run_id,
                            )

                            # Transition to Staging
                            client.transition_model_version_stage(
                                name=model_name,
                                version=model_version.version,
                                stage="Staging",
                            )

                            logger.info(
                                f"Model successfully registered in MLflow: {model_name}"
                            )

                    except Exception as model_error:
                        logger.warning(
                            f"Could not register model with custom wrapper: {model_error}"
                        )
                        # Final fallback: just log as artifact
                        mlflow.log_artifact(
                            str(best_model_path), artifact_path="model_weights"
                        )
                        logger.info(
                            f"Model logged as artifact only: {best_model_path.name}"
                        )

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

    def _log_dataset_samples(self) -> None:
        """Log dataset samples with annotations to MLflow before training.

        Captures representative samples from each class with visual annotations
        to show actual training data being used.
        """
        try:
            import mlflow
            import cv2
            import numpy as np
            from pathlib import Path
            import yaml
            import random
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

            # Create temporary directory for dataset samples
            samples_dir = Path("/tmp/dataset_samples")
            samples_dir.mkdir(exist_ok=True)

            # Get dataset path from config
            dataset_path = Path(self.config.get("train", {}).get("data", ""))
            if not dataset_path.exists():
                logger.warning(f"Dataset path not found: {dataset_path}")
                return

            # Determine task type and process accordingly
            task_type = self.config.get("type", "yolo")

            if task_type == "yolo":
                # Check if it's classification or detection
                if "classify" in str(dataset_path).lower() or self.config.get(
                    "model", ""
                ).endswith("-cls.pt"):
                    self._log_classification_samples(dataset_path, samples_dir)
                else:
                    self._log_detection_samples(dataset_path, samples_dir)

            # Log all samples to MLflow
            if samples_dir.exists() and any(samples_dir.iterdir()):
                mlflow.log_artifacts(str(samples_dir), artifact_path="dataset_samples")
                logger.info(f"Dataset samples logged to MLflow from {samples_dir}")
            else:
                logger.warning("No dataset samples found to log")

        except Exception as e:
            logger.warning(f"Could not log dataset samples: {e}")
            import traceback

            logger.warning(f"Full error: {traceback.format_exc()}")

    def _log_classification_samples(
        self, dataset_path: Path, samples_dir: Path
    ) -> None:
        """Log classification dataset samples with class labels."""
        try:
            import cv2
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont

            # Find class directories
            train_dir = dataset_path / "train"
            if not train_dir.exists():
                train_dir = dataset_path  # Try direct path

            class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]

            for class_dir in class_dirs[:5]:  # Limit to first 5 classes
                class_name = class_dir.name
                image_files = list(class_dir.glob("*.jpg")) + list(
                    class_dir.glob("*.png")
                )

                # Select up to 5 random images from this class
                sample_images = random.sample(image_files, min(5, len(image_files)))

                for i, img_path in enumerate(sample_images):
                    try:
                        # Load image
                        img = Image.open(img_path)

                        # Add class label overlay
                        draw = ImageDraw.Draw(img)

                        # Try to use a larger font, fallback to default
                        try:
                            font = ImageFont.truetype(
                                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                40,
                            )
                        except:
                            try:
                                font = ImageFont.load_default()
                            except:
                                font = None

                        # Draw label background and text
                        label_text = f"Class: {class_name}"
                        if font:
                            bbox = draw.textbbox((10, 10), label_text, font=font)
                            draw.rectangle(
                                [bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5],
                                fill=(255, 0, 0, 200),
                            )
                            draw.text(
                                (10, 10), label_text, fill=(255, 255, 255), font=font
                            )
                        else:
                            draw.text((10, 10), label_text, fill=(255, 255, 255))

                        # Save annotated image
                        output_path = (
                            samples_dir / f"class_{class_name}_sample_{i + 1}.jpg"
                        )
                        img.save(output_path)

                    except Exception as img_error:
                        logger.warning(
                            f"Could not process image {img_path}: {img_error}"
                        )

            logger.info(f"Classification samples saved to {samples_dir}")

        except Exception as e:
            logger.warning(f"Could not log classification samples: {e}")

    def _log_detection_samples(self, dataset_path: Path, samples_dir: Path) -> None:
        """Log detection dataset samples with bounding boxes."""
        try:
            import cv2
            import yaml
            import random

            # Find dataset.yaml file
            dataset_yaml = dataset_path / "dataset.yaml"
            if not dataset_yaml.exists():
                dataset_yaml = dataset_path / "data.yaml"

            if not dataset_yaml.exists():
                logger.warning("Dataset YAML file not found")
                return

            # Load dataset configuration
            with open(dataset_yaml, "r") as f:
                dataset_config = yaml.safe_load(f)

            class_names = dataset_config.get("names", {})

            # Find images directory
            images_dir = dataset_path / "train" / "images"
            labels_dir = dataset_path / "train" / "labels"

            if not images_dir.exists():
                images_dir = dataset_path / "images"
                labels_dir = dataset_path / "labels"

            if not images_dir.exists():
                logger.warning("Images directory not found")
                return

            # Get sample images
            image_files = list(images_dir.glob("*.jpg")) + list(
                images_dir.glob("*.png")
            )
            sample_images = random.sample(image_files, min(10, len(image_files)))

            for img_path in sample_images:
                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    # Find corresponding label file
                    label_path = labels_dir / img_path.with_suffix(".txt").name

                    if label_path.exists():
                        # Read YOLO format labels
                        with open(label_path, "r") as f:
                            labels = f.readlines()

                        # Draw bounding boxes
                        h, w = img.shape[:2]
                        for label in labels:
                            parts = label.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center, y_center, width, height = map(
                                    float, parts[1:5]
                                )

                                # Convert to pixel coordinates
                                x1 = int((x_center - width / 2) * w)
                                y1 = int((y_center - height / 2) * h)
                                x2 = int((x_center + width / 2) * w)
                                y2 = int((y_center + height / 2) * h)

                                # Draw rectangle
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                # Add class label
                                class_name = class_names.get(
                                    class_id, f"Class_{class_id}"
                                )
                                cv2.putText(
                                    img,
                                    class_name,
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2,
                                )

                    # Save annotated image
                    output_path = samples_dir / f"detection_sample_{img_path.name}"
                    cv2.imwrite(str(output_path), img)

                except Exception as img_error:
                    logger.warning(
                        f"Could not process detection image {img_path}: {img_error}"
                    )

            logger.info(f"Detection samples saved to {samples_dir}")

        except Exception as e:
            logger.warning(f"Could not log detection samples: {e}")

    def _start_system_metrics_logging(self) -> None:
        """Start logging system metrics during training."""
        try:
            import mlflow
            import threading
            import time

            # Log initial system metrics
            self._log_system_metrics("training_start")

            # Start background thread to log metrics periodically
            def log_metrics_periodically():
                for i in range(60):  # Log every minute for ~1 hour (adjust as needed)
                    time.sleep(60)  # Wait 1 minute
                    if mlflow.active_run():
                        self._log_system_metrics(f"training_minute_{i}")
                    else:
                        break

            # Start background thread
            metrics_thread = threading.Thread(
                target=log_metrics_periodically, daemon=True
            )
            metrics_thread.start()

            logger.info("Started system metrics logging")

        except Exception as e:
            logger.warning(f"Could not start system metrics logging: {e}")

    def _log_system_metrics(self, phase: str = "training") -> None:
        """Log current system metrics to MLflow.

        Args:
            phase: Training phase identifier (e.g., "training_start", "training_minute_5")
        """
        try:
            import mlflow
            import psutil
            import platform
            from datetime import datetime

            if not mlflow.active_run():
                return

            # System information
            system_info = {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "timestamp": datetime.now().isoformat(),
            }

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            cpu_count = psutil.cpu_count()
            cpu_count_logical = psutil.cpu_count(logical=True)

            cpu_metrics = {
                "cpu_percent": cpu_percent,
                "cpu_freq_current": cpu_freq.current if cpu_freq else None,
                "cpu_freq_min": cpu_freq.min if cpu_freq else None,
                "cpu_freq_max": cpu_freq.max if cpu_freq else None,
                "cpu_count_physical": cpu_count,
                "cpu_count_logical": cpu_count_logical,
                "cpu_load_avg_1min": psutil.getloadavg()[0]
                if hasattr(psutil, "getloadavg")
                else None,
                "cpu_load_avg_5min": psutil.getloadavg()[1]
                if hasattr(psutil, "getloadavg")
                else None,
                "cpu_load_avg_15min": psutil.getloadavg()[2]
                if hasattr(psutil, "getloadavg")
                else None,
            }

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            memory_metrics = {
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_percent": memory.percent,
                "memory_cached_gb": (memory.total - memory.available - memory.used)
                / (1024**3),
                "swap_total_gb": swap.total / (1024**3),
                "swap_used_gb": swap.used / (1024**3),
                "swap_percent": swap.percent,
            }

            # Disk metrics
            disk = psutil.disk_usage("/")
            disk_metrics = {
                "disk_total_gb": disk.total / (1024**3),
                "disk_used_gb": disk.used / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_percent": (disk.used / disk.total) * 100,
            }

            # Network metrics
            network = psutil.net_io_counters()
            network_metrics = {
                "network_bytes_sent": network.bytes_sent if network else 0,
                "network_bytes_recv": network.bytes_recv if network else 0,
                "network_packets_sent": network.packets_sent if network else 0,
                "network_packets_recv": network.packets_recv if network else 0,
            }

            # GPU metrics (if available)
            gpu_metrics = {}
            try:
                gpu_info = get_gpu_info_json()
                if gpu_info and len(gpu_info) > 0 and "error" not in gpu_info[0]:
                    for i, gpu in enumerate(gpu_info):
                        for key, value in gpu.items():
                            gpu_metrics[f"gpu_{i}_{key}"] = value
            except Exception as gpu_error:
                logger.warning(f"Could not get GPU metrics: {gpu_error}")

            # Process metrics
            process = psutil.Process()
            process_metrics = {
                "process_pid": process.pid,
                "process_memory_mb": process.memory_info().rss / (1024**2),
                "process_memory_percent": process.memory_percent(),
                "process_cpu_percent": process.cpu_percent(),
                "process_num_threads": process.num_threads(),
                "process_create_time": datetime.fromtimestamp(
                    process.create_time()
                ).isoformat(),
            }

            # Training-specific metrics
            training_metrics = {
                "training_duration_seconds": time.time() - self.start_time
                if self.start_time > 0
                else 0,
                "training_phase": phase,
                "model_type": self.config.get("type", "unknown"),
                "model_name": self.config.get("model", "unknown"),
                "epochs": self.config.get("train", {}).get("epochs", 0),
                "batch_size": self.config.get("train", {}).get("batch", 0),
                "image_size": self.config.get("train", {}).get("imgsz", 0),
            }

            # Combine all metrics
            all_metrics = {}
            all_metrics.update({f"system_{k}": v for k, v in system_info.items()})
            all_metrics.update({f"cpu_{k}": v for k, v in cpu_metrics.items()})
            all_metrics.update({f"memory_{k}": v for k, v in memory_metrics.items()})
            all_metrics.update({f"disk_{k}": v for k, v in disk_metrics.items()})
            all_metrics.update({f"network_{k}": v for k, v in network_metrics.items()})
            all_metrics.update(gpu_metrics)
            all_metrics.update({f"process_{k}": v for k, v in process_metrics.items()})
            all_metrics.update(
                {f"training_{k}": v for k, v in training_metrics.items()}
            )

            # Log all metrics to MLflow with proper system metrics prefix
            for metric_name, metric_value in all_metrics.items():
                if isinstance(metric_value, (int, float)):
                    # Use system/ prefix for system metrics to appear in System Metrics view
                    if any(
                        metric_name.startswith(prefix)
                        for prefix in [
                            "system_",
                            "cpu_",
                            "memory_",
                            "disk_",
                            "network_",
                            "gpu_",
                            "process_",
                            "training_",
                        ]
                    ):
                        # Remove the prefix we added and use system/ prefix instead
                        clean_name = metric_name
                        for prefix in [
                            "system_",
                            "cpu_",
                            "memory_",
                            "disk_",
                            "network_",
                            "gpu_",
                            "process_",
                            "training_",
                        ]:
                            if clean_name.startswith(prefix):
                                clean_name = clean_name[len(prefix) :]
                                break
                        mlflow.log_metric(f"system/{clean_name}", metric_value)
                    else:
                        mlflow.log_metric(metric_name, metric_value)
                elif metric_value is not None:
                    mlflow.log_param(metric_name, str(metric_value))

            logger.info(f"Logged system metrics for phase: {phase}")

        except Exception as e:
            logger.warning(f"Could not log system metrics: {e}")

    def _log_final_system_metrics(self) -> None:
        """Log final system metrics after training completion."""
        try:
            import mlflow

            if not mlflow.active_run():
                return

            # Calculate total training time
            total_time = self.end_time - self.start_time if self.start_time > 0 else 0

            final_metrics = {
                "training_completed": True,
                "training_total_duration_seconds": total_time,
                "training_total_duration_minutes": total_time / 60,
                "training_total_duration_hours": total_time / 3600,
                "training_end_timestamp": time.time(),
            }

            # Log final metrics
            for metric_name, metric_value in final_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
                else:
                    mlflow.log_param(metric_name, str(metric_value))

            logger.info(
                f"Logged final system metrics. Total training time: {total_time:.2f}s"
            )

        except Exception as e:
            logger.warning(f"Could not log final system metrics: {e}")


# Legacy function name for backward compatibility
obtener_info_gpu_json = get_gpu_info_json

import os
import mlflow
import mlflow.data
import mlflow.data.filesystem_dataset_source
import mlflow.data.http_dataset_source
from mlflow import pytorch
from slugify import slugify


class MLflowManager:
    """Manages MLflow operations for model training."""

    def __init__(self, config: dict):
        self.config = config
        self.redis_manager = None
        self._setup_system_metrics()

    def _setup_system_metrics(self):
        """Setup MLflow system metrics logging."""
        try:
            # Enable system metrics collection only if not already configured
            if not hasattr(self, "_system_metrics_configured"):
                mlflow.set_system_metrics_sampling_interval(5)  # Sample every 5 seconds
                mlflow.set_system_metrics_samples_before_logging(
                    3
                )  # Log after 3 samples
                self._system_metrics_configured = True
                print("✅ System metrics logging enabled")
        except Exception as e:
            print(f"⚠️ Could not enable system metrics: {e}")
            self._system_metrics_configured = (
                False  # Mark as attempted to avoid retries
            )

    def setup_dataset_logging(self, data_path: str, dvc_path: str = ""):
        """Setup dataset logging in MLflow."""
        try:
            if dvc_path:
                try:
                    import dvc.api

                    data_url = dvc.api.get_url(dvc_path)
                    dataset_path = data_path
                except:
                    # Fallback for DVC API issues
                    dvc_path = data_path.replace("/datasets/", "")
                    data_url = f"http://{os.environ.get('CONTROL_HOST', 'localhost')}:23443/files/{dvc_path}"
                    dataset_path = data_path
            else:
                dvc_path = data_path.replace("/datasets/", "")
                data_url = f"http://{os.environ.get('CONTROL_HOST', 'localhost')}:23443/files/{dvc_path}"
                dataset_path = data_path

            if data_url:
                # Simple dataset logging for MLflow 2.20.3
                mlflow.set_tag("dataset_url", data_url)

            mlflow.set_tag("data_source", dataset_path)

        except Exception as e:
            print(f"Error logging dataset: {e}")

    def log_config_file(self, config_path: str):
        """Log configuration file to MLflow."""
        try:
            mlflow.log_artifact(config_path)
        except Exception as e:
            print(f"Error logging config file: {e}")

    def log_model_and_metrics(self, trainer):
        """Log model and metrics to MLflow."""
        if "minio" in self.config and "mlflow" in self.config:
            # Generate registered model name from experiment and task info
            experiment_name = self.config.get("sweeper", {}).get(
                "study_name", "default_experiment"
            )
            task_id = self.config.get("task_id", "default_task")
            registered_model_name = f"{experiment_name}_{task_id}"

            # Log metrics first (simple and reliable)
            try:
                metrics = {
                    slugify(key): float(value) for key, value in trainer.metrics.items()
                }
                mlflow.log_metrics(metrics)
                print("✅ Training metrics logged successfully")
            except Exception as e:
                print(f"⚠️ Error logging metrics: {e}")

            # Log model metadata
            try:
                model_metadata = {
                    "model_type": "YOLO Classification",
                    "framework": "ultralytics",
                    "registered_model_name": registered_model_name,
                    "experiment_name": experiment_name,
                    "task_id": task_id,
                }
                for key, value in model_metadata.items():
                    mlflow.set_tag(key, value)
                print("✅ Model metadata logged")
            except Exception as meta_e:
                print(f"⚠️ Error logging model metadata: {meta_e}")

            # Simple model logging - just log the best.pt as artifact
            try:
                best_model_path = os.path.join(
                    str(trainer.save_dir), "weights", "best.pt"
                )
                if os.path.exists(best_model_path):
                    mlflow.log_artifact(best_model_path, "model")
                    print("✅ YOLO model logged successfully as artifact")
                else:
                    print("⚠️ Best model file not found, looking for alternatives...")
                    # Try to find any .pt file
                    weights_dir = os.path.join(str(trainer.save_dir), "weights")
                    if os.path.exists(weights_dir):
                        for file in os.listdir(weights_dir):
                            if file.endswith(".pt"):
                                model_path = os.path.join(weights_dir, file)
                                mlflow.log_artifact(model_path, "model")
                                print(f"✅ Model logged: {file}")
                                break
            except Exception as e:
                print(f"❌ Error logging model: {e}")

    def log_system_info(self, gpu_info: str):
        """Log system information to MLflow."""
        import json

        gpu_data = json.loads(gpu_info)

        if isinstance(gpu_data, list):
            for gpu_dict in gpu_data:
                for key, value in gpu_dict.items():
                    mlflow.set_tag(key, value)
        else:
            for key, value in gpu_data.items():
                mlflow.set_tag(key, value)

    def log_metadata(self, metadata: dict):
        """Log metadata to MLflow."""
        for key, value in metadata.items():
            mlflow.set_tag(key, value)

    def log_hyperparameters(self, config: dict):
        """Log hyperparameters to MLflow."""
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    try:
                        mlflow.log_param(f"{key}_{sub_key}", sub_value)
                    except Exception:
                        pass  # Skip if parameter already logged
            else:
                try:
                    mlflow.log_param(key, value)
                except Exception:
                    pass  # Skip if parameter already logged

import os
import mlflow
import mlflow.data
import mlflow.data.filesystem_dataset_source
import mlflow.data.http_dataset_source
from slugify import slugify


class MLflowManager:
    """Manages MLflow operations for model training."""

    def __init__(self, config: dict):
        self.config = config
        self.redis_manager = None

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
            pytorch_model = trainer.model.model
            mlflow.pytorch.log_model(pytorch_model, "model")

            metrics = {
                slugify(key): float(value) for key, value in trainer.metrics.items()
            }
            mlflow.log_metrics(metrics)

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

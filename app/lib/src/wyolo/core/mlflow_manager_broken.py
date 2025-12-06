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

            # Log PyTorch model using MLflow's pytorch.log_model
            try:
                # Get the PyTorch model from trainer
                pytorch_model = trainer.model.model
                
                # Create a simple wrapper for MLflow compatibility
                class ModelWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def predict(self, data):
                        return self.model(data)
                    
                    def __call__(self, data):
                        return self.model(data)

                wrapped_model = ModelWrapper(pytorch_model)

                # Log the model using pytorch.log_model
                pytorch.log_model(
                    wrapped_model,
                    name="yolo_classification_model",
                    conda_env={
                        "channels": ["defaults", "pytorch", "conda-forge"],
                        "dependencies": [
                            "python=3.8",
                            "pytorch",
                            "torchvision",
                            "ultralytics",
                            "mlflow",
                            "numpy",
                            "pillow",
                            "pyyaml",
                        ],
                    },
                    registered_model_name=None,  # Don't auto-register
                )
                print("✅ PyTorch model logged successfully with pytorch.log_model")

                # Also log the .pt file as artifact for compatibility
                best_model_path = os.path.join(
                    str(trainer.save_dir), "weights", "best.pt"
                )
                if os.path.exists(best_model_path):
                    mlflow.log_artifact(best_model_path, "model_artifact")
                    print("✅ Model .pt file logged as artifact")

            except Exception as e:
                print(f"❌ Error logging PyTorch model: {e}")
                # Fallback: just log the .pt file as artifact
                try:
                best_model_path = os.path.join(
                    str(trainer.save_dir), "weights", "best.pt"
                )
                if os.path.exists(best_model_path):
                    mlflow.log_artifact(best_model_path, "model_fallback")
                    print("⚠️ Model logged as fallback artifact")
            except Exception as fallback_e:
                print(f"❌ Error in fallback logging: {fallback_e}")

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

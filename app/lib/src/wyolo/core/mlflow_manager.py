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

            # Log model using pytorch.log_model for registry support
            try:
                pytorch_model = trainer.model.model

                # Create a wrapper class that follows MLflow's expected interface
                class PyTorchModelWrapper:
                    def __init__(self, model):
                        self.model = model

                    def predict(self, data):
                        return self.model(data)

                    def __call__(self, data):
                        return self.model(data)

                wrapped_model = PyTorchModelWrapper(pytorch_model)

                # Log the model with proper conda environment (without registration)
                pytorch.log_model(
                    wrapped_model,
                    "pytorch_model",
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
                    registered_model_name=None,  # Explicitly disable registration
                )

                # Try to register the model separately if needed
                try:
                    active_run = mlflow.active_run()
                    if active_run and active_run.info:
                        mlflow.register_model(
                            f"runs:/{active_run.info.run_id}/pytorch_model",
                            registered_model_name,
                        )
                        print(
                            f"PyTorch model registered with name: {registered_model_name}"
                        )
                    else:
                        print("No active MLflow run found for model registration")
                except Exception as reg_e:
                    print(
                        f"Model registration failed (PyTorch model still logged): {reg_e}"
                    )

                print("PyTorch model logged successfully")
            except Exception as e:
                print(f"Error logging PyTorch model to MLflow: {e}")

            # Log YOLO model using the PyTorch model directly from trainer
            try:
                # Get the underlying PyTorch model from the trained YOLO model
                pytorch_model = trainer.model.model

                # Create a simple wrapper that follows MLflow's expected interface
                class SimpleModelWrapper:
                    def __init__(self, model):
                        self.model = model

                    def predict(self, data):
                        return self.model(data)

                    def __call__(self, data):
                        return self.model(data)

                wrapped_model = SimpleModelWrapper(pytorch_model)

                # Save model locally first, then log it to avoid registry API issues
                import tempfile
                import shutil

                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path = os.path.join(temp_dir, "yolo_model")

                    # Save the model using MLflow's pytorch flavor (pass the raw PyTorch model)
                    pytorch.save_model(
                        pytorch_model,  # Use the raw PyTorch model, not the wrapper
                        model_path,
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
                    )

                    # Log the saved model directory as artifact
                    mlflow.log_artifacts(model_path, "yolo_model")
                    print(
                        "✅ YOLO model logged successfully with save_model + log_artifacts"
                    )

            except Exception as yolo_e:
                print(f"❌ Error logging YOLO model with log_model: {yolo_e}")
                # Fallback: log as artifact if log_model fails
                try:
                    best_model_path = os.path.join(
                        str(trainer.save_dir), "weights", "best.pt"
                    )
                    if os.path.exists(best_model_path):
                        mlflow.log_artifact(best_model_path, "yolo_model_fallback")
                        print("⚠️ YOLO model logged as artifact (fallback)")
                except Exception as fallback_e:
                    print(
                        f"❌ Error logging YOLO model as fallback artifact: {fallback_e}"
                    )

            # Log model metadata for registry
            try:
                model_metadata = {
                    "model_type": "YOLO Classification",
                    "framework": "ultralytics",
                    "registered_model_name": registered_model_name,
                    "experiment_name": experiment_name,
                    "task_id": task_id,
                    "model_file": "best.pt",
                    "pytorch_model_path": "pytorch_model",
                    "yolo_model_path": "yolo_model/best.pt",
                }
                for key, value in model_metadata.items():
                    mlflow.set_tag(key, value)
                print("Model metadata logged")
            except Exception as meta_e:
                print(f"Error logging model metadata: {meta_e}")

            # Log metrics
            try:
                metrics = {
                    slugify(key): float(value) for key, value in trainer.metrics.items()
                }
                mlflow.log_metrics(metrics)
                print("Metrics logged successfully")
            except Exception as e:
                print(f"Error logging metrics to MLflow: {e}")

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

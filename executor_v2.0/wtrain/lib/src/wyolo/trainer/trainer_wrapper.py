import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from typing import List

import mlflow
import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from slugify import slugify
from ultralytics import RTDETR, YOLO
from ultralytics.utils.autobatch import autobatch

from .cte.elemental import Elemental
from .gpu_utils import gpu_compatibility_check, obtener_info_gpu_json
from .post_train import pipeline_post_train
from .utils.mlflow_setup import Mlflow_setup

console = Console()


def display_config_banner(
    base_config: str,
    dataset_path_data_yaml: str,
    model_arch: str,
    epochs: int,
    batch_size: int,
    project: str,
):
    # Tabla estructurada para los parámetros de configuración
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Parameter", style="bold cyan", justify="right")
    table.add_column("Value", style="white")

    # Mapeo de parámetros
    table.add_row("Base Config:", f"[bold green]{base_config}[/bold green]")
    table.add_row("Dataset YAML:", f"[yellow]{dataset_path_data_yaml}[/yellow]")
    table.add_row("Model Architecture:", model_arch)
    table.add_row("Epochs / Batch:", f"{epochs} / {batch_size}")
    table.add_row("Project:", f"[dim]{project}[/dim]")

    # Panel contenedor estilo CLI
    panel = Panel(
        table,
        title="[bold bright_blue]⚙️  TRAINING CONFIGURATION [/bold bright_blue]",
        subtitle="[dim]Initializing pipeline...[/dim]",
        border_style="cyan",
        padding=(1, 2),
        expand=False,
    )

    console.print(panel)


class TrainerWrapper(Elemental, Mlflow_setup):
    # https://github.com/ultralytics/ultralytics/issues/8214
    config = {}

    def __init__(self, config: dict):
        self.config = config

        # Update a setting
        self.set_config_vars(self.config)
        self.is_configured = True

    @property
    def config_train(self):
        return self.config

    @config_train.setter
    def config_train(self, new_config: dict):
        self.config = new_config

    def on_train_end(self, trainer):
        if "minio" in self.config and "mlflow" in self.config:
            pytorch_model = trainer.model.model

            experiment_name = self.config.get("sweeper", {}).get(
                "study_name", "default_experiment"
            )
            task_id = self.config.get("task_id", "default_task")

            registered_model_name = f"{experiment_name}_{task_id}"
            # Log the model as artifact instead (simpler approach)

            # 1. train

            # 2. up result to mlflow

            try:

                model_path = f"{self.ARTIFACTS_PATH}/pytorch_model.pth"
                torch.save(pytorch_model.state_dict(), model_path)
                mlflow.log_artifact(model_path, artifact_path="model")
            except Exception as e:
                print(f"Failed to log model: {e}")

            metrics = {
                slugify(key): float(value) for key, value in trainer.metrics.items()
            }
            mlflow.log_metrics(metrics)

            self.artifacts_organice(trainer)

            # 3. grapCam

            new_model_trained = trainer.model.model

            pipeline_post_train.run(
                {
                    "model": new_model_trained,
                    "images_test_path": self.config.get("train", {}).get(
                        "data", None
                    ),
                    "project_path": self.config.get("tempfile", {}),
                }
            )

            # TODO: 4. force up train document
            # TODO: 5. force up valid examples
            # TODO: 6. force up test examples

            # 4. up result to mlflow
            mlflow.log_artifacts(self.ARTIFACTS_PATH)

            console.print(
                Panel(
                    "[bold green]✔ Training process finished with status: SUCCESS[/bold green]\n"
                    "[dim]Artifacts and weights saved to mlflow.[/dim]",
                    title="[bold white] STATUS [/bold white]",
                    border_style="green",
                    expand=False,
                )
            )

    def on_train_start(self, trainer):
        if "minio" in self.config and "mlflow" in self.config:
            # remove batch of self.config
            config_copy = self.config.copy()
            config_copy["train"].pop("batch")

            tags_list = [
                (key, value)
                for key, value in obtener_info_gpu_json()[0].items()
                if key and value
            ]

            for key, value in self.get_tags(self.config, trainer):
                tags_list.append((key, value))

            for key, value in tags_list:
                try:
                    mlflow.set_tag(key, value)
                except:
                    pass

            for key, value in config_copy["train"].items():
                try:
                    mlflow.log_param(key, value)
                except:
                    pass

            self.artifacts_organice(trainer)
            mlflow.log_artifacts(self.ARTIFACTS_PATH)

            current_run = mlflow.active_run()
            self.model_uri = f"runs:/{current_run.info.run_id}/model"

        self.start_time = time.time()

    def on_epoch_end(self, trainer):
        if self.firts_epoch:
            # self.firts_epoch = False
            self.artifacts_organice(trainer)
            try:
                mlflow.log_artifacts(self.ARTIFACTS_PATH)
            except Exception as e:
                print(f"Failed to log artifacts: {e}")

        self.end_time = time.time()

        elapsed_time = self.end_time - self.start_time

        metadata = self.get_summary(self.config, trainer, elapsed_time)

        if "minio" in self.config and "mlflow" in self.config:
            gpu_json_list: List[dict] = obtener_info_gpu_json()
            for gpu_json in gpu_json_list:
                for key, value in gpu_json.items():
                    metadata[key] = value

            with open(self.SUMMARY_PATH, "w") as file:
                yaml.dump(
                    metadata,
                    file,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

        if os.path.exists(self.STOP_TRAIN_PATH):
            self.force_stop_train(trainer)

            os.remove(self.STOP_TRAIN_PATH)

    def force_stop_train(self, trainer):
        if self.model:
            self.model.stop_training = True
            self.model.stop_training = False
            # self.model = None

        if "minio" in self.config and "mlflow" in self.config:
            # Log the final model in MLflow
            self.on_train_end(trainer)

            # Log the artifacts
            try:
                current_training_path = (
                    self.config["train"]["project"] + "train_" + self.config["task_id"]
                )
                mlflow.log_artifacts(current_training_path)
            except:
                pass

            # End the MLflow run
            mlflow.end_run(status="FINISHED")

        raise StopIteration("Entrenamiento detenido por condición de callback.")

    def tune(self, CONFIG_TRAIN: dict, _generations: int):
        if self.model:
            _grace_period = self.config.get("genetic", {}).get("min_epochs_by_ind", 10)
            _other_parameters = self.config.get("genetic", {}).get(
                "other_parameters", {}
            )

            _use_genetic = self.config.get("genetic", {}).get("use_genetic", False)
            if _use_genetic:
                # genetic algorithm
                _use_ray = False
                # Eficiencia: Lento (espera a terminar cada generación).
                # Variación: Mutación: Altera valores previos.
                # Herencia: Los hijos heredan rasgos de los padres.
                # Selección: Los mejores "padres" sobreviven.

                _other_parameters["optimizer"] = "SGD"  # Recomendado para evolución
            else:
                # ASHA (Asynchronous Successive Halving Algorithm)
                # "Pruning" (Poda) de ensayos prematuros utilizando grace_period
                _use_ray = True
                # Eficiencia: Rápido (interrumpe ensayos prematuros).
                # Variación: Muestreo: Elige valores nuevos del espacio definido.
                # Herencia: Cada experimento suele ser independiente.
                # Selección: Los peores se detienen (Poda).

                # Ensure grace_period is positive and reasonable
                _epochs = self.config.get("train", {}).get("epochs", 10)

                _other_parameters["grace_period"] = max(1, min(_epochs, _grace_period))

            # si algun parametros de _other_parameters ya existe en config_train
            # se borra para evitar conflictos
            _real_other_parameters = _other_parameters.copy()
            for key in _other_parameters.keys():
                if key in CONFIG_TRAIN:
                    _real_other_parameters.pop(key)

            _iterations = max(2, min(100, _generations))

            return self.model.tune(
                **CONFIG_TRAIN,
                iterations=_iterations,
                use_ray=_use_ray,
                **_real_other_parameters,
            )

    def train(self, config_train: dict):
        if self.model:
            print(f"--- [TRAINER] Executing model.train() ---")
            try:
                display_config_banner(
                    base_config=self.config.get("base_config", "N/A"),
                    dataset_path_data_yaml=config_train["data"],
                    model_arch=config_train.get("model", "YOLOvx-?"),
                    epochs=config_train.get("epochs", "N/A"),
                    batch_size=config_train.get("batch", "N/A"),
                    project=config_train.get("project", "N/A"),
                )

                return self.model.train(**config_train)
            except Exception as e:
                print(f"--- [TRAINER] Exception in model.train(): {e} ---")
                import traceback

                traceback.print_exc()

                # Check if training completed and metrics were populated
                if (
                    hasattr(self.model, "trainer")
                    and self.model.trainer
                    and getattr(self.model.trainer, "metrics", None) is not None
                ):
                    print(
                        "--- [TRAINER] Training completed before the exception. Returning populated metrics. ---"
                    )
                    return self.model.trainer.metrics
                return None

    def create_model(self, model_name, model_type):
        if model_type == "yolo":
            model = YOLO(model_name)
        elif model_type == "rtdetr":
            model = RTDETR(model_name)
        else:
            raise ValueError("Invalid model type specified.")

        self.model = model

        # Configure the callbacks
        if "minio" in self.config and "mlflow" in self.config:
            self.model.add_callback("on_train_start", self.on_train_start)
            self.model.add_callback("on_train_end", self.on_train_end)
            self.model.add_callback("on_fit_epoch_end", self.on_epoch_end)

        return model


def get_datetime():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def load_config(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_trainer(config_path: str, trial_number):
    request_config = load_config(config_path=config_path)
    request_config["config_path"] = config_path

    results_dir = request_config.get("tempfile")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

    trainer = TrainerWrapper(config=request_config)
    trainer.create_model(
        model_name=request_config["model"],
        model_type=request_config["type"],
    )

    if "task_id" not in request_config:
        request_config["task_id"] = str(uuid.uuid4())

    experiment_name = request_config.get("sweeper").get("study_name")
    tempfile = request_config.get("tempfile", "")

    RESULT_PATH = f"{tempfile}/models/{experiment_name}/{request_config['type']}/{request_config['task_id']}/"
    os.makedirs(f"{RESULT_PATH}/trail_history", exist_ok=True)
    request_config["path_results"] = f"{RESULT_PATH}/{trial_number}/"

    timestamp = get_datetime()
    request_config["timestamp"] = timestamp

    # if request_config["train"]["batch"] > 0:
    #     better_batch = trainer.get_better_batch(
    #         batch_to_use=request_config["train"]["batch"]
    #     )
    #     if request_config["train"]["batch"] > better_batch:
    #         request_config["train"]["batch"] = better_batch

    request_config["train"]["project"] = f"{RESULT_PATH}/{trial_number}/"
    request_config["train"]["name"] = f"train_{request_config.get('task_id')}"
    request_config["train"]["verbose"] = True
    request_config["train"]["plots"] = True
    request_config["train"]["exist_ok"] = True

    if request_config["train"]["data"].startswith("/datasets/"):
        request_config["train"]["data"] = request_config["train"]["data"].replace(
            "/datasets/", "/wyolo/control_server/datasets/"
        )

    # change route for internal route, because the train service is in a container with a different route
    # 1. validate if the data path is a folder or yaml file
    if os.path.isdir(request_config["train"]["data"]):
        # don't do anything, because the dataset is a clasification dataset
        # so, is not necessary to change the path of train, val and test
        pass
    elif os.path.isfile(request_config["train"]["data"]):
        # do something, because the dataset is a detection dataset
        # so, is necessary to change the path of train, val and test
        old_data_path = request_config["train"]["data"]

        new_data_path = f"{tempfile}/data.yaml"
        shutil.copy(old_data_path, new_data_path)

        # 2. into new data.yaml change the path of train, val and test
        with open(new_data_path, "r") as file:
            data_yaml_config = yaml.safe_load(file)

        if data_yaml_config["train"].startswith("/datasets/"):
            data_yaml_config["train"] = data_yaml_config["train"].replace(
                "/datasets/", "/wyolo/control_server/datasets/"
            )

        if "val" in data_yaml_config and data_yaml_config["val"].startswith(
            "/datasets/"
        ):
            data_yaml_config["val"] = data_yaml_config["val"].replace(
                "/datasets/", "/wyolo/control_server/datasets/"
            )

        if "test" in data_yaml_config and data_yaml_config["test"].startswith(
            "/datasets/"
        ):
            data_yaml_config["test"] = data_yaml_config["test"].replace(
                "/datasets/", "/wyolo/control_server/datasets/"
            )

        with open(new_data_path, "w") as file:
            yaml.dump(data_yaml_config, file)

        request_config["train"]["data"] = new_data_path

    trainer.config_train = request_config

    return trainer, request_config


def train(trainer: TrainerWrapper, request_config: dict, fitness: str):
    _data = request_config["train"].get("data", None)
    if _data is None or not os.path.exists(_data):
        raise FileNotFoundError(f"Data path not found: {_data}")

    # Check for force_gpu in extras, default to False
    force_gpu = request_config.get("extras", {}).get("force_gpu", False)
    force_cpu = request_config.get("extras", {}).get("force_cpu", False)

    if force_cpu:
        print("Forcing CPU usage for training.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        torch.cuda.is_available = lambda: False

        request_config["train"]["device"] = "cpu"

    elif gpu_compatibility_check(force_gpu):
        request_config["train"]["device"] = "0"
    else:
        request_config["train"]["device"] = "cpu"

    final_result = 0.0

    if "train" in request_config:
        if trainer.config.get("genetic", {}).get("activate", False):
            # 0. Validacion que existan los parametros necesarios
            NEED_PARAMS = [
                "poblation_size",
                "generations",
                "min_epochs_by_ind",
                "direction",
                "fitness",
            ]
            for param in NEED_PARAMS:
                if param not in trainer.config.get("genetic", {}):
                    raise ValueError(
                        f"Missing genetic parameter: {param} in configuration."
                    )

            _poblation_size = trainer.config.get("genetic", {}).get(
                "poblation_size", 10
            )
            _generations = trainer.config.get("genetic", {}).get("generations", 10)

            start_time = datetime.now().isoformat()

            # 1. Iniciar proceso de tuning genético
            # ---------------------------------------
            # ---------------------------------------
            #      Proceso de evolución genética
            # ---------------------------------------
            # ---------------------------------------
            _real_generations = _generations * _poblation_size
            tune_results = trainer.tune(
                CONFIG_TRAIN=request_config["train"],
                _generations=_real_generations,
            )
            # ---------------------------------------
            # ---------------------------------------

            # 2. Guardar resultados del tuning
            _use_genetic = trainer.config.get("genetic", {}).get("use_genetic", False)
            if _use_genetic:
                # genetic algorithm
                try:
                    experiment_path = tune_results.experiment_path
                    shutil.copytree(
                        experiment_path,
                        trainer.ARTIFACTS_PATH + "/tune_results",
                        dirs_exist_ok=True,
                    )
                except Exception as e:
                    print(f"Error copying tune results: {e}")
            else:
                # ASHA (by ray tune)
                experiment_path = tune_results.experiment_path
                shutil.copytree(
                    experiment_path,
                    trainer.ARTIFACTS_PATH + "/tune_results",
                    dirs_exist_ok=True,
                )

            # 3. Obtener el mejor resultado basado en una métrica (ejemplo: accuracy)
            _mode = trainer.config.get("genetic", {}).get("direction", "max")
            _metric = trainer.config.get("genetic", {}).get("fitness", fitness)
            best_result = tune_results.get_best_result(metric=_metric, mode=_mode)

            # 4. Extraer los parámetros (config) del mejor resultado
            best_params = best_result.config

            # 5. Actualizar request_config con los mejores parámetros
            request_config["train"].update(best_params)

            # 6. Guardar los mejores parámetros en archivo YAML
            with open(
                trainer.ARTIFACTS_PATH + "/tune_results" + "/best_params.yaml", "w"
            ) as f:
                yaml.dump(request_config["train"], f)

            # 7. Guardar duración del proceso de tuning en archivo YAML
            with open(
                trainer.ARTIFACTS_PATH + "/tune_results" + "/duration.yaml", "w"
            ) as f:
                final_result = best_result.metrics.get(_metric, 0.0)
                end_time = datetime.now().isoformat()
                yaml.dump(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": str(
                            datetime.fromisoformat(end_time)
                            - datetime.fromisoformat(start_time)
                        ),
                        "best_fitness": final_result,
                    },
                    f,
                )

        # ---------------------------------------
        # ---------------------------------------
        #     Proceso de entrenamiento normal
        # ---------------------------------------
        # ---------------------------------------
        train_params = request_config["train"]

        # calculate better batch based on gpu info and configured batch
        if train_params.get("batch") != -1:
            batch = int(os.environ.get("MAX_GPU", -1))
            batch = batch / 100 if batch > 0 else 0.1
            train_params["batch"] = batch

        print(f"--- [TRAINER] Starting YOLO train with config: {train_params} ---")

        results = trainer.train(config_train=train_params)
        # ---------------------------------------
        # ---------------------------------------

        if results:
            if hasattr(results, "results_dict"):
                request_config["train"]["results"] = results.results_dict
                try:
                    request_config["experiment_type"] = str(results.task)
                except:
                    request_config["experiment_type"] = "not-specified"
            elif isinstance(results, dict):
                request_config["train"]["results"] = results
                request_config["experiment_type"] = getattr(
                    trainer.model, "task", "not-specified"
                )
            else:
                request_config["train"]["results"] = {}
                request_config["experiment_type"] = "not-specified"

            try:
                results_dict = request_config["train"]["results"]
                if fitness in results_dict:
                    final_result = results_dict[fitness]
                else:
                    # Look for key matching or with suffix (B) or (M)
                    matching_keys = [
                        k for k in results_dict.keys() if fitness in k or k in fitness
                    ]
                    if matching_keys:
                        final_result = results_dict[matching_keys[0]]
                        print(
                            f"--- [TRAINER] Metric '{fitness}' not found directly. Using matching key '{matching_keys[0]}': {final_result} ---"
                        )
                    else:
                        final_result = 0.0
            except:
                final_result = 0.0
        else:
            print("--- [TRAINER] Warning: No results found in YOLO training ---")
            final_result = 0.0

    print(f"ResultadoFinal:{final_result}")
    return final_result


if __name__ == "__main__":
    request_config: dict = ...
    trial_number = 1
    fitness: str = ...

    trainer, request_config = create_trainer(request_config, trial_number)

    if "train" in request_config:
        final_result = train(trainer, request_config)

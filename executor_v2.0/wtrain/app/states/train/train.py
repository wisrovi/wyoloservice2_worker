"""WPipe step for configuring, launching, and monitoring YOLO model training."""

import os
import tempfile
from typing import Any, Dict
from wpipe import step, to_obj
from wyolo.trainer.trainer_wrapper import create_trainer, train
import yaml

from dto.training_context import UserInput
from ..check.check_dataset import DATASET_FOLDER
from utils.config_manager import get_complete_config, get_user_config


@step(name="train_model", version="v1.0", tags=["train_model"])
@to_obj(UserInput)
def train_model(data_input: UserInput) -> Dict[str, Any]:
    """Execute model training based on the resolved user and base YAML configuration.

    Args:
        data_input (UserInput): Container holding the configuration file path.

    Returns:
        Dict[str, Any]: Dictionary containing trained model evaluation results.

    Raises:
        FileNotFoundError: If configuration file path does not exist.
    """
    if not os.path.exists(data_input.user_config_train):
        raise FileNotFoundError(
            f"Configuration file does not exist: {data_input.user_config_train}"
        )

    user_config_path, _ = get_user_config(user_config_file=data_input.user_config_train)
    config_dict, config_path = get_complete_config(user_config=user_config_path)

    results = None

    with tempfile.TemporaryDirectory(prefix="wyolo_") as temp_dir:
        config_dict["tempfile"] = temp_dir

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f)

        trainer, request_config = create_trainer(
            config_path=config_path, trial_number=1
        )

        if "train" in request_config:
            dataset_path = request_config.get("train", {}).get("data", "")
            if dataset_path.startswith("/datasets/"):
                dataset_path = dataset_path.replace("/datasets/", DATASET_FOLDER)
                request_config["train"]["data"] = dataset_path

            model_str = str(config_dict.get("model", "")).lower()
            style = (
                config_dict.get("style")
                or config_dict.get("task")
                or ("segment" if "-seg" in model_str
                    else "classify" if "-cls" in model_str
                    else "pose" if "-pose" in model_str
                    else "obb" if "-obb" in model_str
                    else "detect")
            )
            fitness_map = {
                "detect": "metrics/mAP50-95(B)",
                "segment": "metrics/mAP50-95(M)",
                "classify": "metrics/accuracy_top1",
                "pose": "metrics/mAP50-95(P)",
                "obb": "metrics/mAP50-95(B)",
            }
            fitness = config_dict.get("sweeper", {}).get("fitness")
            if not fitness or (fitness == "metrics/accuracy_top1" and str(style).lower() != "classify"):
                fitness = fitness_map.get(str(style).lower(), "metrics/mAP50-95(B)")

            results = train(trainer, request_config, fitness)

    return {"results_trained_model": results}

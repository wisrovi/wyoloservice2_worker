from pydantic import BaseModel
from wyolo.trainer.trainer_wrapper import create_trainer, train

import os
import tempfile
from wpipe import step, to_obj
import yaml

from states.utils.util import get_complete_config, get_user_config
from ..check.check_dataset import DATASET_FOLDER


class UserInput(BaseModel):
    user_config_train: str


@step(name="train_model", version="v1.0", tags=["train_model"])
@to_obj(UserInput)
def train_model(data_input: UserInput):

    if not os.path.exists(data_input.user_config_train):
        raise FileNotFoundError(
            f"El archivo de configuración no existe: {data_input.user_config_train}"
        )

    user_config_path, USER_CONFIG = get_user_config(
        user_config_file=data_input.user_config_train
    )

    config_dict, config_path = get_complete_config(user_config=user_config_path)

    results = None

    with tempfile.TemporaryDirectory(prefix="wyolo_") as temp_dir:
        config_dict["tempfile"] = temp_dir

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        trainer, request_config = create_trainer(
            config_path=config_path, trial_number=1
        )

        if "train" in request_config:
            DATASET = request_config.get("train").get("data")
            if DATASET.startswith("/datasets/"):
                DATASET = DATASET.replace("/datasets/", DATASET_FOLDER)
                request_config["train"]["data"] = DATASET

            fitness = config_dict.get("sweeper", {}).get("fitness")
            if not fitness:
                # Auto-resolve fitness metric from style, task, or model name pattern
                style = (
                    config_dict.get("style")
                    or config_dict.get("task")
                    or ("segment" if "-seg" in str(config_dict.get("model", "")).lower()
                        else "classify" if "-cls" in str(config_dict.get("model", "")).lower()
                        else "pose" if "-pose" in str(config_dict.get("model", "")).lower()
                        else "obb" if "-obb" in str(config_dict.get("model", "")).lower()
                        else "detect")
                )
                fitness_map = {
                    "detect": "metrics/mAP50-95(B)",
                    "segment": "metrics/mAP50-95(M)",
                    "classify": "metrics/accuracy_top1",
                    "pose": "metrics/mAP50-95(P)",
                    "obb": "metrics/mAP50-95(B)",
                }
                fitness = fitness_map.get(str(style).lower(), "metrics/mAP50-95(B)")

            results = train(trainer, request_config, fitness)

    return {"results_trained_model": results}

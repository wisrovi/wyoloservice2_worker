"""WPipe step for checking dataset availability, CIFS mounting, and directory layout."""

import os
import subprocess
from typing import Any, Dict
from loguru import logger
from wpipe import step, to_obj

from dto.training_context import DatasetUserInput

DATASET_FOLDER = "/wyolo/control_server/datasets/"


@step(name="check_dataset", version="v1.0", tags=["check_dataset"])
@to_obj(DatasetUserInput)
def check_dataset(input_data: DatasetUserInput) -> Dict[str, Any]:
    """Verify target dataset folder and trigger automated CIFS mount if empty.

    Args:
        input_data (DatasetUserInput): Container holding the user configuration object.

    Returns:
        Dict[str, Any]: Resulting state with dataset_status set to 1 if available.

    Raises:
        FileNotFoundError: If the dataset folder or target dataset path does not exist.
    """
    if not os.path.exists(DATASET_FOLDER):
        logger.error(f"Dataset root folder '{DATASET_FOLDER}' does not exist.")
        raise FileNotFoundError(f"Dataset root folder '{DATASET_FOLDER}' does not exist.")

    files = os.listdir(DATASET_FOLDER)
    if len(files) == 0:
        mount_command = "sh /usr/local/bin/mount-cifs.sh"
        subprocess.run(mount_command, shell=True, check=True)

    dataset_path = input_data.user_config.train.data
    if "/datasets/" in dataset_path:
        dataset_subpath = dataset_path.split("/datasets/")[-1]
        dataset_path = os.path.join(DATASET_FOLDER, dataset_subpath)

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path '{dataset_path}' does not exist.")
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")

    logger.info(f"Dataset path '{dataset_path}' verified and ready for execution.")
    return {"dataset_status": 1}

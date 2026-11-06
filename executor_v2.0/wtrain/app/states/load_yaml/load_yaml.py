"""WPipe step for validating and loading user training configuration YAML."""

import os
from typing import Any, Dict
from loguru import logger
from wpipe import step, to_obj

from dto.training_context import UserInput
from utils.config_manager import get_user_config


@step(name="load_yaml", version="v1.0", tags=["load_yaml"])
@to_obj(UserInput)
def load_yaml(data_input: UserInput) -> Dict[str, Any]:
    """Validate existence of configuration file and extract training payload.

    Args:
        data_input (UserInput): Container holding the configuration file path.

    Returns:
        Dict[str, Any]: Execution state dictionary containing parsed user_config.

    Raises:
        FileNotFoundError: If configuration file does not exist on disk.
    """
    if not os.path.exists(data_input.user_config_train):
        raise FileNotFoundError(
            f"Training configuration file not found: {data_input.user_config_train}"
        )

    _, user_config = get_user_config(user_config_file=data_input.user_config_train)
    logger.debug(f"Loaded user_config successfully: {user_config}")
    return {"user_config": user_config}

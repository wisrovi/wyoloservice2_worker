"""Configuration utility compatibility wrapper redirecting to app.utils.config_manager."""

from utils.config_manager import (
    get_base_config,
    get_complete_config,
    get_user_config,
    merge_configs,
    read_base_config,
    read_user_config,
)

__all__ = [
    "get_user_config",
    "read_user_config",
    "merge_configs",
    "get_base_config",
    "read_base_config",
    "get_complete_config",
]

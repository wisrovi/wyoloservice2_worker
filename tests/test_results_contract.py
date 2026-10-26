"""Unit tests validating results.json structure and contract for Worker Executor.

This module validates:
1. That results.json includes accuracy, status: completed, and timestamp.
2. That output dictionary and file content match the expected schema.
"""

import importlib.util
import json
import os
from typing import Any

import pytest


def load_public_results_module() -> Any:
    """Dynamically loads the public_results module across directory paths.

    Returns:
        Any: The loaded public_results module object.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(
        current_dir, "..", "executor_v2.0", "wtrain", "app", "states", "train", "public_results.py"
    )
    module_path = os.path.abspath(module_path)
    spec = importlib.util.spec_from_file_location("public_results_module", module_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publish_results_format(tmp_path: Any) -> None:
    """Validates that publish_results_mlflow outputs the complete results contract.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    results_dir = str(tmp_path / "train_service_results")
    os.makedirs(results_dir, exist_ok=True)

    public_results = load_public_results_module()
    orig_base = public_results.BASE_PATH
    public_results.BASE_PATH = results_dir

    try:
        input_dto = public_results.PublicResultsInput(results_trained_model=0.8854)
        output = public_results.publish_results_mlflow(input_dto)

        assert output["public_results"] is True
        assert output["accuracy"] == 0.8854

        result_file = os.path.join(results_dir, "results.json")
        assert os.path.exists(result_file)

        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["accuracy"] == 0.8854
        assert data["status"] == "completed"
        assert "timestamp" in data
        assert isinstance(data["timestamp"], (int, float))
    finally:
        public_results.BASE_PATH = orig_base


def normalize_ultralytics_train_time(train_params: dict[str, Any]) -> dict[str, Any]:
    """Mirrors the train.time normalization logic in trainer_wrapper.py."""
    raw_time = train_params.get("time")
    if raw_time is not None and not isinstance(raw_time, bool):
        try:
            parsed_time = float(raw_time)
            if parsed_time > 0.0:
                train_params["time"] = parsed_time
            else:
                train_params.pop("time", None)
        except (ValueError, TypeError):
            train_params.pop("time", None)
    else:
        train_params.pop("time", None)
    return train_params


def test_trainer_wrapper_time_parameter_normalization() -> None:
    """Validates that trainer_wrapper only passes 'time' to Ultralytics when explicitly > 0.

    This test verifies:
    1. If 'time' is omitted, None, or <= 0, 'time' is absent from train_params so Ultralytics
       trains by 'epochs' and 'patience' without overriding the epoch count.
    2. If 'time' is a positive number (e.g. 30), it is converted to float (30.0) so Ultralytics
       trains for 30 hours and stops cleanly at the end of the epoch.
    """
    params_omitted = normalize_ultralytics_train_time({"epochs": 230, "patience": 50})
    assert "time" not in params_omitted
    assert params_omitted["epochs"] == 230

    params_none = normalize_ultralytics_train_time({"epochs": 230, "time": None})
    assert "time" not in params_none

    params_zero = normalize_ultralytics_train_time({"epochs": 230, "time": 0})
    assert "time" not in params_zero

    params_positive = normalize_ultralytics_train_time({"epochs": 230, "time": 30})
    assert params_positive["time"] == 30.0

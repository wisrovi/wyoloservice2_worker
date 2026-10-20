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

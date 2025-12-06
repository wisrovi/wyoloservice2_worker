#!/usr/bin/env python3
"""
Simple test script to verify system metrics functionality.
"""

import sys
import os

sys.path.insert(0, "/app/lib/src")

from unittest.mock import Mock, patch
from wyolo.core.trainer_wrapper import TrainerWrapper


def test_system_metrics():
    """Test that system metrics are properly configured."""
    print("🧪 Testing system metrics configuration...")

    config = {
        "model": "yolov8n.pt",
        "type": "yolo",
        "train": {
            "data": "/path/to/dataset.yaml",
            "epochs": 1,
            "imgsz": 640,
            "batch": 16,
        },
        "task_id": "test_task",
        "mlflow": {"MLFLOW_TRACKING_URI": "http://localhost:5000"},
        "minio": {"MINIO_ENDPOINT": "http://localhost:9000"},
    }

    # Test trainer initialization
    trainer = TrainerWrapper(config)
    print("✅ Trainer initialized successfully")

    # Test MLflow manager initialization
    assert hasattr(trainer.mlflow_manager, "_setup_system_metrics")
    print("✅ MLflowManager has system metrics setup method")

    # Test system metrics configuration
    with (
        patch("mlflow.set_system_metrics_sampling_interval") as mock_sampling,
        patch("mlflow.set_system_metrics_samples_before_logging") as mock_samples,
    ):
        # Re-initialize to test system metrics setup
        trainer.mlflow_manager._setup_system_metrics()

        mock_sampling.assert_called_once_with(5)
        mock_samples.assert_called_once_with(3)
        print("✅ System metrics sampling configured correctly")

    print("🎉 All system metrics tests passed!")


if __name__ == "__main__":
    test_system_metrics()

"""Test configuration and fixtures."""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": "yolov8n.pt",
        "type": "yolo",
        "task_id": "test_task_001",
        "train": {
            "data": "/path/to/dataset.yaml",
            "epochs": 10,
            "imgsz": 640,
            "batch": 16,
            "verbose": True,
            "plots": True,
        },
        "mlflow": {
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
        },
        "minio": {
            "MINIO_ENDPOINT": "http://localhost:9000",
            "MINIO_ID": "minioadmin",
            "MINIO_SECRET_KEY": "minioadmin",
        },
        "redis": {
            "REDIS_HOST": "localhost",
            "REDIS_PORT": 6379,
            "REDIS_DB": 0,
        },
        "sweeper": {
            "study_name": "test_experiment",
            "n_trials": 5,
            "tune": False,
        }
    }


@pytest.fixture
def temp_config_file(sample_config):
    """Create temporary config file."""
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    os.unlink(config_path)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
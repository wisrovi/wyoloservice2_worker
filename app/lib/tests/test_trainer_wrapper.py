"""Tests for wyolo trainer wrapper."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from wyolo.core.trainer_wrapper import TrainerWrapper, obtener_info_gpu_json


class TestTrainerWrapper:
    """Test cases for TrainerWrapper class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "model": "yolov8n.pt",
            "type": "yolo",
            "train": {
                "data": "/path/to/dataset.yaml",
                "epochs": 10,
                "imgsz": 640,
                "batch": 16,
            },
            "task_id": "test_task"
        }
        self.trainer = TrainerWrapper(self.config)

    def test_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.config == self.config
        assert self.trainer.model is None
        assert not self.trainer.is_configured

    @patch('wyolo.core.trainer_wrapper.settings')
    def test_initialization_with_mlflow(self, mock_settings):
        """Test initialization with MLflow configuration."""
        config_with_mlflow = {
            **self.config,
            "mlflow": {"MLFLOW_TRACKING_URI": "http://localhost:5000"},
            "minio": {"MINIO_ENDPOINT": "http://localhost:9000"}
        }
        trainer = TrainerWrapper(config_with_mlflow)
        mock_settings.update.assert_called_with({"mlflow": True})

    @patch('wyolo.core.trainer_wrapper.YOLO')
    def test_create_yolo_model(self, mock_yolo):
        """Test YOLO model creation."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        result = self.trainer.create_model("yolov8n.pt", "yolo")
        
        assert result == mock_model
        assert self.trainer.model == mock_model
        mock_yolo.assert_called_once_with("yolov8n.pt")

    @patch('wyolo.core.trainer_wrapper.RTDETR')
    def test_create_rtdetr_model(self, mock_rtdetr):
        """Test RTDETR model creation."""
        mock_model = Mock()
        mock_rtdetr.return_value = mock_model
        
        result = self.trainer.create_model("rtdetr-l.pt", "rtdetr")
        
        assert result == mock_model
        assert self.trainer.model == mock_model
        mock_rtdetr.assert_called_once_with("rtdetr-l.pt")

    def test_create_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError, match="Invalid model type"):
            self.trainer.create_model("model.pt", "invalid")

    @patch('wyolo.core.trainer_wrapper.autobatch')
    def test_get_better_batch(self, mock_autobatch):
        """Test optimal batch size calculation."""
        self.trainer.model = Mock()
        mock_autobatch.return_value = 32
        
        result = self.trainer.get_better_batch(batch_to_use=16)
        
        assert result == 32
        mock_autobatch.assert_called_once()

    def test_config_train_property(self):
        """Test config_train property getter and setter."""
        new_config = {"test": "config"}
        self.trainer.config_train = new_config
        assert self.trainer.config == new_config


class TestGPUInfo:
    """Test cases for GPU information functions."""

    @patch('wyolo.core.trainer_wrapper.GPUtil')
    def test_obtener_info_gpu_json_success(self, mock_gputil):
        """Test successful GPU info retrieval."""
        mock_gpu = Mock()
        mock_gpu.id = 0
        mock_gpu.name = "Test GPU"
        mock_gpu.uuid = "test-uuid"
        mock_gpu.memoryTotal = 8192
        mock_gpu.memoryFree = 4096
        mock_gpu.memoryUsed = 4096
        mock_gpu.load = 0.5
        mock_gpu.temperature = 65
        mock_gpu.processes = []
        
        mock_gputil.getGPUs.return_value = [mock_gpu]
        
        result = obtener_info_gpu_json()
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert "gpu_0_name" in result[0]
        assert result[0]["gpu_0_name"] == "Test GPU"

    @patch('wyolo.core.trainer_wrapper.GPUtil')
    def test_obtener_info_gpu_json_no_gpus(self, mock_gputil):
        """Test GPU info when no GPUs available."""
        mock_gputil.getGPUs.return_value = []
        
        result = obtener_info_gpu_json()
        
        assert isinstance(result, list)
        assert len(result) == 0

    @patch('wyolo.core.trainer_wrapper.GPUtil')
    def test_obtener_info_gpu_json_error(self, mock_gputil):
        """Test GPU info error handling."""
        mock_gputil.getGPUs.side_effect = Exception("GPU error")
        
        result = obtener_info_gpu_json()
        
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__])
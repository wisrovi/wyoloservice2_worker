"""Tests for wyolo YOLO train CLI."""

import pytest
from unittest.mock import Mock, patch, mock_open
import tempfile
import os

from wyolo.core.yolo_train import load_config, get_datetime


class TestYoloTrain:
    """Test cases for yolo_train module."""

    def test_load_config_success(self):
        """Test successful config loading."""
        config_content = """
        model: "yolov8n.pt"
        type: "yolo"
        train:
            epochs: 10
        """
        with patch("builtins.open", mock_open(read_data=config_content)):
            config = load_config("test_config.yaml")
        
        assert config["model"] == "yolov8n.pt"
        assert config["type"] == "yolo"
        assert config["train"]["epochs"] == 10

    def test_load_config_file_not_found(self):
        """Test config loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("non_existent.yaml")

    def test_get_datetime(self):
        """Test datetime generation."""
        datetime_str = get_datetime()
        assert isinstance(datetime_str, str)
        assert len(datetime_str) == 15  # YYYYMMDD_HHMMSS format
        assert "_" in datetime_str


class TestYoloTrainIntegration:
    """Integration tests for YOLO training."""

    @patch('wyolo.core.yolo_train.TrainerWrapper')
    @patch('wyolo.core.yolo_train.uuid')
    def test_train_command_integration(self, mock_uuid, mock_trainer_class):
        """Test complete training command integration."""
        # Setup mocks
        mock_uuid.uuid4.return_value = "test-uuid"
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.get_better_batch.return_value = 32
        
        # Mock model creation
        mock_model = Mock()
        mock_trainer.create_model.return_value = mock_model
        
        # Mock training results
        mock_results = Mock()
        mock_results.task = "detect"
        mock_results.results_dict = {"fitness": 0.85, "precision": 0.90}
        mock_trainer.train.return_value = mock_results
        
        config_content = """
        model: "yolov8n.pt"
        type: "yolo"
        train:
            data: "/test/dataset.yaml"
            epochs: 10
            batch: 16
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            from wyolo.core.yolo_train import train
            
            # Test the train function
            result = train(
                config_path=config_path,
                fitness="fitness",
                trial_number=1
            )
            
            # Verify trainer was created and configured
            mock_trainer_class.assert_called_once()
            mock_trainer.create_model.assert_called_once_with("yolov8n.pt", "yolo")
            mock_trainer.train.assert_called_once()
            
            # Verify result structure
            assert "task_id" in result
            assert "train" in result
            assert "results" in result["train"]
            
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__])
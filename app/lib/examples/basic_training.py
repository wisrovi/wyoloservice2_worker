#!/usr/bin/env python3
"""Example script showing how to use Wyolo programmatically."""

import os
import tempfile
from pathlib import Path

from wyolo import TrainerWrapper


def main():
    """Run example training with Wyolo."""
    # Example configuration
    config = {
        "model": "yolov8n.pt",
        "type": "yolo",
        "task_id": "example_001",
        "train": {
            "data": "/path/to/your/dataset.yaml",  # Update this path
            "epochs": 10,
            "imgsz": 640,
            "batch": 16,
            "verbose": True,
            "plots": True,
        },
        "mlflow": {
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
        },
        "redis": {
            "REDIS_HOST": "localhost",
            "REDIS_PORT": 6379,
            "REDIS_DB": 0,
        }
    }

    # Create trainer
    trainer = TrainerWrapper(config)
    
    # Create model
    model = trainer.create_model("yolov8n.pt", "yolo")
    print(f"Model created: {type(model).__name__}")
    
    # Get optimal batch size for GPU
    if trainer.model:
        optimal_batch = trainer.get_better_batch(batch_to_use=16)
        print(f"Optimal batch size: {optimal_batch}")
        
        # Update config with optimal batch
        config["train"]["batch"] = optimal_batch
    
    # Start training (commented out for safety)
    # results = trainer.train(config["train"])
    # print(f"Training completed: {results}")
    
    print("Example setup complete. Update dataset path and uncomment training to run.")


if __name__ == "__main__":
    main()
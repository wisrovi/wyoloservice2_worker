# Wyolo

[![PyPI version](https://badge.fury.io/py/wyolo.svg)](https://badge.fury.io/py/wyolo)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Wyolo** - Professional YOLO training library with MLOps integration.

A comprehensive Python library for training YOLO and RTDETR models with built-in MLflow tracking, Redis monitoring, and enterprise-grade MLOps capabilities.

## Features

- 🚀 **Easy YOLO Training**: Simple API for training YOLO and RTDETR models
- 📊 **MLflow Integration**: Automatic experiment tracking, logging, and artifact management
- 🔄 **Redis Monitoring**: Real-time training progress monitoring
- 🎯 **Hyperparameter Tuning**: Built-in Ray Tune integration
- 💾 **MinIO/S3 Support**: Cloud storage for models and artifacts
- 📈 **GPU Optimization**: Automatic batch size optimization and GPU monitoring
- 🛠 **Enterprise Ready**: Production-grade logging, error handling, and monitoring

## Quick Start

### Installation

```bash
# Basic installation
pip install wyolo

# With development dependencies
pip install wyolo[dev]

# With hyperparameter tuning support
pip install wyolo[tune]
```

### Basic Usage

#### Programmatic API

```python
from wyolo import TrainerWrapper

# Configuration
config = {
    "model": "yolov8n.pt",
    "type": "yolo",
    "train": {
        "data": "/path/to/dataset.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
    },
    "mlflow": {
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
    },
    "task_id": "my_training_job"
}

# Create trainer and start training
trainer = TrainerWrapper(config)
model = trainer.create_model("yolov8n.pt", "yolo")
results = trainer.train(config["train"])
```

#### Command Line Interface

```bash
wyolo-train --config_path config.yaml --fitness fitness --trial_number 1
```

### Configuration Example

```yaml
# config.yaml
model: "yolov8n.pt"
type: "yolo"
task_id: "experiment_001"

train:
  data: "/datasets/my_dataset.yaml"
  epochs: 100
  imgsz: 640
  batch: 16
  verbose: true
  plots: true

mlflow:
  MLFLOW_TRACKING_URI: "http://localhost:5000"

minio:
  MINIO_ENDPOINT: "http://localhost:9000"
  MINIO_ID: "minioadmin"
  MINIO_SECRET_KEY: "minioadmin"

redis:
  REDIS_HOST: "localhost"
  REDIS_PORT: 6379
  REDIS_DB: 0

sweeper:
  study_name: "yolo_experiment"
  n_trials: 10
  tune: true
```

## Advanced Features

### MLflow Integration

Wyolo automatically logs:
- Training parameters and hyperparameters
- Metrics and loss curves
- Model artifacts and checkpoints
- GPU utilization and system metrics
- Example images from training data
- Dataset information and lineage

### Redis Progress Monitoring

Monitor training progress in real-time:

```python
import redis
from wredis.hash import RedisHashManager

# Connect to Redis
redis_manager = RedisHashManager(
    host="localhost", 
    port=6379, 
    db=0
)

# Get training progress
progress = redis_manager.get_hash("progress:my_task_id")
```

### Hyperparameter Tuning

```python
config = {
    "sweeper": {
        "study_name": "yolo_optimization",
        "n_trials": 50,
        "tune": True,
        "grace_period": 10
    }
}

trainer = TrainerWrapper(config)
model = trainer.create_model("yolov8n.pt", "yolo")
results = trainer.train(config["train"])  # Automatic tuning
```

### GPU Optimization

```python
# Get optimal batch size for your GPU
optimal_batch = trainer.get_better_batch(batch_to_use=32)

# Monitor GPU utilization
gpu_info = trainer.obtener_info_gpu_json()
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/wisrovi/wyoloservice2_worker.git
cd wyoloservice2_worker

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=wyolo

# Run specific test
pytest tests/test_trainer.py
```

### Code Quality

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Lint code
flake8 src tests

# Type checking
mypy src
```

## Architecture

```
wyolo/
├── core/
│   ├── trainer_wrapper.py    # Main training logic
│   └── yolo_train.py         # CLI interface
├── __init__.py               # Package exports
└── py.typed                  # Type information
```

## Dependencies

- **Core**: ultralytics, mlflow, loguru, click, pyyaml
- **MLOps**: redis, dvc, GPUtil, wredis
- **Image Processing**: pillow, python-slugify
- **Optional**: ray[tune] (for hyperparameter tuning)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Wyolo in your research, please cite:

```bibtex
@software{wyolo,
  title={Wyolo: Professional YOLO Training Library with MLOps Integration},
  author={William Steve Rodriguez Villamizar},
  year={2024},
  url={https://github.com/wisrovi/wyoloservice2_worker}
}
```

## Support

- 📖 [Documentation](https://wyolo.readthedocs.io/)
- 🐛 [Bug Reports](https://github.com/wisrovi/wyoloservice2_worker/issues)
- 💬 [Discussions](https://github.com/wisrovi/wyoloservice2_worker/discussions)

---

**Wyolo** - Making YOLO training professional and enterprise-ready. 🚀
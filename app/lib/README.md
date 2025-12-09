# Wyolo - Professional YOLO Training Library

[![PyPI version](https://badge.fury.io/py/wyolo.svg)](https://badge.fury.io/py/wyolo)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Wyolo** - Enterprise-grade YOLO training library with comprehensive MLOps integration.

A production-ready Python library for training YOLO and RTDETR models with built-in MLflow tracking, Redis monitoring, GPU optimization, and enterprise-grade MLOps capabilities. Designed for professional machine learning workflows.

---

## 🚀 **Core Features**

### 🎯 **Training Capabilities**
- **Multi-Model Support**: YOLO (v8, v9, v10, v11) and RTDETR models
- **Task Flexibility**: Object detection, image classification, and segmentation
- **Automatic Optimization**: Smart batch size calculation and GPU memory management
- **Resume Training**: Continue from checkpoints with full state preservation

### 📊 **MLOps Integration**
- **MLflow Tracking**: Comprehensive experiment tracking with automatic logging
- **Redis Monitoring**: Real-time training progress and system metrics
- **DVC Integration**: Dataset versioning and lineage tracking
- **Cloud Storage**: MinIO/S3 support for model artifacts

### 🔧 **Enterprise Features**
- **Hyperparameter Tuning**: Ray Tune integration with Optuna optimization
- **Distributed Training**: Multi-GPU and distributed training support
- **Error Handling**: Robust error recovery and graceful degradation
- **Production Logging**: Comprehensive audit trails and monitoring

---

## 🏗️ **Architecture**

Wyolo follows a modular architecture with clear separation of concerns:

```
lib/src/wyolo/core/
├── trainer_wrapper.py          # Main training orchestrator
├── yolo_train.py             # CLI interface and utilities
├── gpu_utils.py               # GPU optimization and monitoring
├── mlflow_manager.py          # MLflow operations and logging
├── utils.py                  # EDA and progress management
└── trainer_wrapper_original.py # Original implementation (reference)
```

### **Component Overview**

#### **TrainerWrapper** - Core Training Engine
- Model creation and configuration management
- Training orchestration and lifecycle management
- Callback system for training events
- Integration with all MLOps components

#### **MLflowManager** - Experiment Tracking
- Automatic experiment and run management
- Dataset lineage and source tracking
- Model artifact logging and versioning
- System metrics and GPU utilization logging

#### **GPUUtils** - Hardware Optimization
- Automatic batch size optimization
- GPU monitoring and utilization tracking
- Memory management and error handling
- Multi-GPU support and load balancing

#### **Utils** - Supporting Services
- **EDAManager**: Exploratory data analysis and reporting
- **ProgressManager**: Real-time progress tracking via Redis
- **StatusEDA**: Status management for EDA operations

---

## 🚀 **Quick Start**

### Installation

```bash
# Basic installation
pip install -e lib/

# With all dependencies
pip install -e lib/[mlflow,redis,tune]

# Development installation
pip install -e lib/[dev,mlflow,redis,tune]
```

### Basic Usage

#### **Programmatic API**

```python
from lib.src.wyolo.core.yolo_train import create_trainer, train
from application.utils.util import get_complete_config

# Load configuration
config_dict, config_path = get_complete_config(
    user_config="/path/to/your/config.yaml"
)

# Create trainer with model
trainer, request_config = create_trainer(
    config_path=config_path, 
    trial_number=1
)

# Start training
results = train(trainer, request_config, fitness="fitness")
print(f"Training completed: {results}")
```

#### **Direct Trainer Usage**

```python
from lib.src.wyolo.core import TrainerWrapper

# Configuration
config = {
    "model": "yolov8n-cls.pt",
    "type": "yolo",
    "task_id": "my_experiment_001",
    "train": {
        "data": "/datasets/my_dataset/",
        "epochs": 100,
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
        "study_name": "my_experiment",
        "n_trials": 10,
        "tune": False,
    }
}

# Create and configure trainer
trainer = TrainerWrapper(config)
trainer.set_config_vars()

# Create model
model = trainer.create_model(
    model_name="yolov8n-cls.pt",
    model_type="yolo"
)

# Start training
results = trainer.train(config["train"])
```

---

## 📋 **Configuration Reference**

### **Complete Configuration Example**

```yaml
# config.yaml - Complete production configuration
model: "yolov8n-cls.pt"
type: "yolo"
task_id: "production_training_001"

# Training parameters
train:
  data: "/datasets/production_dataset.yaml"
  epochs: 100
  imgsz: 640
  batch: -1  # Auto-calculate optimal batch size
  verbose: true
  plots: true
  exist_ok: true
  save_period: 10

# MLflow configuration
mlflow:
  MLFLOW_TRACKING_URI: "http://mlflow.company.com:5000"

# MinIO/S3 configuration
minio:
  MINIO_ENDPOINT: "https://s3.company.com"
  MINIO_ID: "your-access-key"
  MINIO_SECRET_KEY: "your-secret-key"
  MINIO_BUCKET: "mlflow-artifacts"

# Redis configuration
redis:
  REDIS_HOST: "redis.company.com"
  REDIS_PORT: 6379
  REDIS_DB: 0
  REDIS_PASSWORD: "your-redis-password"
  TOPIC: "training_queue"
  RESULT_TOPIC: "results_queue"

# Hyperparameter optimization
sweeper:
  study_name: "production_optimization"
  algorithm: "optuna"
  direction: "maximize"
  n_trials: 50
  tune: true
  grace_period: 10
  max_concurrent: 2
  sampler: "TPESampler"
  search_space:
    model: ["choice", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]]
    train:
      lr0: ["loguniform", 1e-5, 1e-2]
      imgsz: ["choice", [416, 512, 640, 832]]
      batch: ["choice", [8, 16, 32, 64]]

# Metadata and documentation
metadata:
  author: "Data Science Team"
  content: "Production model training for quality control"
  documentation: "Trained on production data from Q4 2024"
  version: "2.1.0"

# DVC configuration
dvc:
  MINIO_BUCKET: "dvc-storage"
  MINIO_ENDPOINT: "http://dvc.company.com:9000"
  MINIO_ID: "dvc-user"
  MINIO_SECRET_KEY: "dvc-secret"

# System configuration
extras:
  gpu:
    id: 0
    limit: 0.95
```

---

## 🔧 **Advanced Features**

### **1. MLflow Integration**

Wyolo provides comprehensive MLflow tracking:

```python
# Automatic logging includes:
- Training parameters and hyperparameters
- Real-time metrics and loss curves
- Model artifacts and checkpoints
- GPU utilization and system metrics
- Example images from training data
- Dataset information and lineage
- Configuration files and environment variables
```

**Accessing Experiments:**
```python
# All runs are automatically logged to MLflow
# View at: http://your-mlflow-server:5000
# Experiment name: config["sweeper"]["study_name"]
# Run name: config["task_id"]
```

### **2. Redis Progress Monitoring**

Real-time training progress monitoring:

```python
import redis
from wredis.hash import RedisHashManager

# Connect to Redis
redis_manager = RedisHashManager(
    host="localhost",
    port=6379,
    db=0
)

# Monitor training progress
progress = redis_manager.get_hash(f"progress:{task_id}")
print(f"Current epoch: {progress.get('epoch')}")
print(f"Training status: {progress.get('status')}")
print(f"Current accuracy: {progress.get('accuracy')}")
```

**Progress Data Structure:**
```json
{
  "epoch": 45,
  "status": "training",
  "timestamp": "2024-12-03T16:23:23.280836",
  "loss": 0.234,
  "accuracy": 0.892,
  "elapsed_time": 1234.567,
  "TRIAL_NUMBER": "1",
  "TOTAL_EPOCHS": "100",
  "EPOCH_PROGRESS": "0.45",
  "datetime": "2024-12-03T16:23:23.280836",
  "task_id": "my_experiment_001"
}
```

### **3. GPU Optimization**

Automatic GPU optimization and monitoring:

```python
# Get optimal batch size for your GPU
optimal_batch = trainer.get_better_batch(batch_to_use=32)
print(f"Optimal batch size: {optimal_batch}")

# Monitor GPU utilization
gpu_info = trainer.obtener_info_gpu_json()
print(f"GPU Info: {gpu_info}")

# GPU info structure:
[
  {
    "gpu_0_name": "NVIDIA GeForce RTX 3060",
    "gpu_0_uuid": "GPU-12345678-1234-1234-1234-123456789012",
    "gpu_0_memoryTotal": 11901,
    "gpu_0_memoryFree": 8192,
    "gpu_0_memoryUsed": 3712,
    "gpu_0_load": 65.5,
    "gpu_0_temperature": 72.0
  }
]
```

### **4. Hyperparameter Tuning**

Built-in Ray Tune integration:

```python
# Enable hyperparameter tuning
config = {
    "sweeper": {
        "study_name": "yolo_hyperparameter_search",
        "n_trials": 100,
        "tune": True,
        "grace_period": 10,
        "max_concurrent": 4,
        "algorithm": "optuna",
        "direction": "maximize",
        "search_space": {
            "model": ["choice", ["yolov8n.pt", "yolov8s.pt"]],
            "train": {
                "lr0": ["loguniform", 1e-5, 1e-2],
                "imgsz": ["choice", [416, 512, 640, 832]],
                "batch": ["choice", [8, 16, 32, 64]],
                "momentum": ["uniform", 0.8, 0.98]
            }
        }
    }
}

# Training will automatically use Ray Tune for optimization
trainer = TrainerWrapper(config)
results = trainer.train(config["train"])
```

### **5. Training Control and Monitoring**

**Stop Training Remotely:**
```bash
# Create stop file to gracefully stop training
touch /config/stop_training_{task_id}.txt
```

**EDA and Data Analysis:**
```python
# Automatic EDA is performed and logged to MLflow
# Includes:
- Dataset statistics and visualizations
- Example images per class
- Data quality reports
- Class distribution analysis
```

---

## 🔄 **Callbacks and Events**

Wyolo provides a comprehensive callback system:

```python
class CustomTrainer(TrainerWrapper):
    def on_train_start(self, trainer):
        """Called when training starts"""
        print("Training started!")
        # Custom logging, notifications, etc.
    
    def on_train_end(self, trainer):
        """Called when training ends"""
        print("Training completed!")
        # Custom cleanup, notifications, etc.
    
    def on_train_epoch_end(self, trainer):
        """Called at the end of each epoch"""
        epoch = trainer.epoch
        metrics = trainer.metrics
        print(f"Epoch {epoch}: {metrics}")
    
    def on_epoch_end(self, trainer):
        """Called at the end of each epoch (compatibility)"""
        # Additional epoch-end logic
        pass
```

**Available Callbacks:**
- `on_train_start`: Training initialization
- `on_train_end`: Training completion
- `on_train_epoch_end`: End of training epoch
- `on_epoch_end`: End of epoch (compatibility)
- `check_stop_training`: Stop condition checking

---

## 🛠️ **Development**

### **Setup Development Environment**

```bash
# Clone repository
git clone https://github.com/wisrovi/wyoloservice2_worker.git
cd wyoloservice2_worker

# Install in development mode
pip install -e "lib/[dev,mlflow,redis,tune]"

# Install pre-commit hooks
pre-commit install

# Setup environment
cp .env.example .env
# Edit .env with your configuration
```

### **Running Tests**

```bash
# Run all tests
cd lib && pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_trainer_wrapper.py
pytest tests/test_gpu_utils.py
pytest tests/test_mlflow_manager.py

# Run integration tests
pytest tests/integration/
```

### **Code Quality**

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Lint code
flake8 src tests

# Type checking
mypy src

# Security check
bandit -r src
```

### **Building Documentation**

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs && make html

# Serve documentation locally
python -m http.server 8000 --directory _build/html
```

---

## 📊 **Monitoring and Observability**

### **Production Monitoring**

Wyolo provides comprehensive monitoring capabilities:

```python
# System metrics automatically logged:
- CPU utilization and memory usage
- GPU temperature, memory, and utilization
- Training progress and epoch timing
- Model performance metrics
- Error rates and warnings
- Disk I/O and network usage
```

### **Alerting Integration**

```python
# Custom alerting can be added via callbacks:
class AlertingTrainer(TrainerWrapper):
    def on_train_epoch_end(self, trainer):
        # Send alerts on poor performance
        if trainer.metrics.get('loss', 0) > threshold:
            send_alert("High loss detected!")
        
        # Send alerts on GPU issues
        gpu_info = self.obtener_info_gpu_json()
        if any(gpu['gpu_0_temperature'] > 85 for gpu in gpu_info):
            send_alert("GPU overheating!")
```

---

## 🚀 **Production Deployment**

### **Docker Deployment**

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Wyolo
COPY lib /app/lib
WORKDIR /app
RUN pip install -e lib/[mlflow,redis,tune]

# Copy application code
COPY application /app/application

# Run training
CMD ["python", "base.py"]
```

### **Kubernetes Deployment**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: wyolo-training-job
spec:
  template:
    spec:
      containers:
      - name: wyolo-trainer
        image: wyolo:latest
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
        - name: REDIS_HOST
          value: "redis-service"
```

---

## 📚 **API Reference**

### **TrainerWrapper Class**

#### **Constructor**
```python
TrainerWrapper(config: dict) -> TrainerWrapper
```

#### **Main Methods**
```python
create_model(model_name: str, model_type: str) -> Model
train(config_train: dict) -> Results
get_better_batch(batch_to_use: int = 32) -> int
set_config_vars() -> None
save_eda() -> None
```

#### **Properties**
```python
config_train: dict  # Get/set training configuration
is_configured: bool  # Configuration status
GPU_USE: float     # GPU utilization percentage
model: Model        # Current model instance
```

#### **Callback Methods**
```python
on_train_start(trainer) -> None
on_train_end(trainer) -> None
on_train_epoch_end(trainer) -> None
on_epoch_end(trainer) -> None
check_stop_training(trainer) -> None
```

### **Utility Functions**

```python
# From yolo_train.py
create_trainer(config_path: str, trial_number: int) -> (TrainerWrapper, dict)
train(trainer: TrainerWrapper, request_config: dict, fitness: str) -> dict
load_config(config_path: str) -> dict
get_datetime() -> str
```

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Workflow**

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Code Standards**

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Add type hints for all public functions
- Include docstrings for all classes and methods
- Add tests for new functionality
- Update documentation for API changes

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📖 **Citation**

If you use Wyolo in your research, please cite:

```bibtex
@software{wyolo2024,
  title={Wyolo: Professional YOLO Training Library with MLOps Integration},
  author={William Steve Rodriguez Villamizar},
  year={2024},
  url={https://github.com/wisrovi/wyoloservice2_worker},
  version={2.0.0}
}
```

---

## 🆘 **Support**

- 📖 **Documentation**: [https://wyolo.readthedocs.io/](https://wyolo.readthedocs.io/)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/wisrovi/wyoloservice2_worker/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/wisrovi/wyoloservice2_worker/discussions)
- 📧 **Email**: william.rodriguez@company.com
- 💬 **Slack**: [#wyolo-support](https://company.slack.com/channels/wyolo-support)

---

## 🏆 **Changelog**

### **Version 2.0.0** (Current)
- ✨ Complete refactor with modular architecture
- ✨ Separated concerns into dedicated modules
- ✨ Enhanced MLflow integration
- ✨ Improved GPU optimization
- ✨ Better error handling and logging
- ✨ Comprehensive documentation
- ✨ Production-ready features

### **Version 1.x.x** (Legacy)
- 📜 Original monolithic implementation
- 📜 Basic MLflow integration
- 📜 Limited GPU optimization

---

## 🎯 **Roadmap**

### **Upcoming Features**
- [ ] **Multi-Modal Training**: Vision + Language models
- [ ] **Federated Learning**: Distributed training across organizations
- [ ] **AutoML**: Automated model architecture search
- [ ] **Model Compression**: Quantization and pruning
- [ ] **Edge Deployment**: ONNX and TensorRT export
- [ ] **Advanced Monitoring**: Prometheus and Grafana integration
- [ ] **A/B Testing**: Model comparison framework
- [ ] **Data Pipeline**: Automated data preprocessing

---

**Wyolo** - Making YOLO training professional, scalable, and enterprise-ready. 🚀

*Built with ❤️ by the Data Science Team*
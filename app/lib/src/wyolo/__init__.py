"""Wyolo - Professional YOLO Training Library with MLOps integration."""

# Core training components
from .cli import create_trainer, train
from .monitoring import EDAManager, MLflowManager, ProgressManager, StatusEDA

# Optimization and monitoring
from .optimization import get_better_batch, obtener_info_gpu_json
from .training import TrainerWrapper

# Utilities and legacy
from .utils import OriginalTrainerWrapper

__version__ = "2.0.0"
__author__ = "William Steve Rodriguez Villamizar"
__email__ = "wisrovi.rodriguez@gmail.com"

__all__ = [
    # Core training
    "TrainerWrapper",
    "create_trainer",
    "train",
    # Optimization
    "obtener_info_gpu_json",
    "get_better_batch",
    # Monitoring
    "MLflowManager",
    "EDAManager",
    "ProgressManager",
    "StatusEDA",
    # Utilities
    "OriginalTrainerWrapper",
]

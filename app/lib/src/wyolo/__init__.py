"""Wyolo - Professional YOLO Training Library with MLOps integration."""

from .core.trainer_wrapper import TrainerWrapper
from .core.yolo_train import train

__version__ = "1.0.0"
__author__ = "William Steve Rodriguez Villamizar"
__email__ = "wisrovi.rodriguez@gmail.com"

__all__ = ["TrainerWrapper", "train"]
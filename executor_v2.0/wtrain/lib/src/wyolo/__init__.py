"""Wyolo - Professional YOLO Training Library with MLOps integration."""

from .trainer.trainer_wrapper import create_trainer, train

__version__ = "2.2.15"
__author__ = "William Steve Rodriguez Villamizar"
__email__ = "wisrovi.rodriguez@gmail.com"

__all__ = ["train", "create_trainer"]
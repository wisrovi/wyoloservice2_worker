"""Optimization module for Wyolo."""

from .gpu_utils import (
    get_better_batch,
    get_gpu_info_json,
    obtener_info_gpu_json,
    get_optimal_batch_size,
)

__all__ = [
    "get_gpu_info_json",
    "get_optimal_batch_size",
    "obtener_info_gpu_json",
    "get_better_batch",
]

"""GPU optimization utilities for Wyolo."""

from __future__ import annotations

from typing import Any, Dict, List

import GPUtil
from ultralytics.utils.autobatch import autobatch


def get_gpu_info_json() -> List[Dict[str, Any]]:
    """Get detailed GPU information in JSON format.

    Retrieves comprehensive information about available GPUs including
    name, memory usage, load, temperature, and running processes.

    Returns:
        List of dictionaries containing GPU information for each available GPU.
        If no GPUs are found, returns a list with an error message.
        If an error occurs, returns a list with error details.

    Example:
        >>> gpu_info = get_gpu_info_json()
        >>> if gpu_info and "error" not in gpu_info[0]:
        ...     print(f"Found {len(gpu_info)} GPUs")
    """
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return [{"error": "No GPUs available."}]

        gpu_info = []
        for gpu in gpus:
            gpu_data = {
                f"gpu_{gpu.id}_name": gpu.name,
                f"gpu_{gpu.id}_uuid": gpu.uuid,
                f"gpu_{gpu.id}_memoryTotal": gpu.memoryTotal,
                f"gpu_{gpu.id}_memoryFree": gpu.memoryFree,
                f"gpu_{gpu.id}_memoryUsed": gpu.memoryUsed,
                f"gpu_{gpu.id}_load": gpu.load * 100,
                f"gpu_{gpu.id}_temperature": gpu.temperature,
            }

            # Check if 'processes' attribute exists before accessing it
            if hasattr(gpu, "processes"):
                gpu_data["processes"] = [
                    {
                        "pid": process.pid,
                        "name": process.name,
                        "memory": process.memoryUsed,
                    }
                    for process in gpu.processes
                ]
            else:
                gpu_data["processes"] = "Process information not available."

            gpu_info.append(gpu_data)

        return gpu_info

    except Exception as e:
        return [{"error": f"Error occurred while getting GPU information: {e}"}]


def get_optimal_batch_size(
    model: Any,
    imgsz: int = 640,
    fraction: float = 0.4,
    batch_to_use: int = -1,
) -> int:
    """Calculate optimal batch size for training based on GPU memory.

    Uses ultralytics autobatch to determine the optimal batch size
    for the current model and GPU configuration.

    Args:
        model: The model to train (YOLO or RTDETR instance)
        imgsz: Image size for training (default: 640)
        fraction: GPU memory fraction to use (default: 0.4 = 40%)
        batch_to_use: Initial batch size (if -1, will auto-calculate)

    Returns:
        Optimal batch size for training

    Example:
        >>> model = YOLO("yolov8n.pt")
        >>> optimal_batch = get_optimal_batch_size(model, imgsz=640)
        >>> print(f"Optimal batch size: {optimal_batch}")
    """
    try:
        optimal_batch = autobatch(
            model=model,
            imgsz=imgsz,
            fraction=fraction,
            batch_size=batch_to_use,
        )
        return optimal_batch
    except Exception:
        # Fallback to a reasonable default if autobatch fails
        return 16


# Legacy function names for backward compatibility
obtener_info_gpu_json = get_gpu_info_json
get_better_batch = get_optimal_batch_size

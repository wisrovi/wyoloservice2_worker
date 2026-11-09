"""WPipe step for detecting CUDA driver version and GPU accelerator availability."""

import os
import subprocess
import time
from typing import Any, Dict
import torch
from loguru import logger
from wpipe import step, to_obj

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get(
    "LD_LIBRARY_PATH", ""
)


@step(name="check_gpu_available", version="v1.1", tags=["check_gpu_available"])
@to_obj
def check_gpu_available(data_input: Any) -> Dict[str, Any]:
    """Verify GPU hardware presence and awaken CUDA runtime driver.

    Args:
        data_input (Any): Execution pipeline context.

    Returns:
        Dict[str, Any]: Execution state dictionary with gpu_status set to 1.

    Raises:
        RuntimeError: If no operational GPU is detected.
    """
    test_command = [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader,nounits",
    ]
    logger.debug(f"CUDA Runtime Version: {torch.version.cuda}")
    driver_version = "Unknown"
    try:
        driver_version = subprocess.check_output(test_command).decode().strip()
    except Exception:
        if torch.cuda.is_available():
            driver_version = (
                f"CUDA {torch.version.cuda} ({torch.cuda.get_device_name(0)})"
            )
    logger.debug(f"Driver Version: {driver_version}")

    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

    gpu_ready = False
    for attempt in range(1, 4):
        if torch.cuda.is_available():
            gpu_ready = True
            break
        logger.warning(
            f"GPU not detected on attempt {attempt}/3. Waiting for CUDA context..."
        )
        time.sleep(2)

    if not gpu_ready:
        logger.warning("GPU not detected after retries.")
        raise RuntimeError("GPU did not respond in time.")

    return {"gpu_status": 1}

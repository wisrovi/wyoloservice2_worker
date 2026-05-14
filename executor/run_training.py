"""Worker Executor Script.

This script runs inside a container and performs the actual training logic
(simulated in this case). It reads configuration from a shared volume,
processes the trial, and writes results back to the volume.
"""

import os
import json
import time
from typing import Any


def main() -> None:
    """Main execution entry point for the training trial.

    Reads configuration, simulates training progress, calculates a metric,
    and persists results.
    """
    # 1. READ CONFIG: The invoker delivers this file via shared volume
    config_path: str = "/app/data/config.json"

    if not os.path.exists(config_path):
        print(f"--- [EXECUTOR ERROR] Config file not found at {config_path} ---")
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            full_config: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"--- [EXECUTOR ERROR] Failed to decode config: {exc} ---")
        return

    # Extract training-specific parameters
    train_cfg: dict[str, Any] = full_config.get("train", {})
    metadata: dict[str, Any] = full_config.get("metadata", {})
    user_id: str = metadata.get("author", "unknown_author")
    sweeper_cfg: dict[str, Any] = full_config.get("sweeper", {})
    study_name: str = sweeper_cfg.get("study_name", "unnamed_study")

    print("--- [EXECUTOR] Starting LONG training ---")
    print(f"--- Study: {study_name} | Author: {user_id} ---")

    # Log the specific hyperparameters for this trial
    print("--- Hyperparameters for this trial: ---")
    for key, value in train_cfg.items():
        print(f"  {key}: {value}")

    # 2. RUN HEAVY TRAINING (Simulated with progress logs)
    # This process will block until finished.
    # Celery Invoker (Host) is waiting for this container to die.
    total_steps: int = 60
    for step in range(total_steps):
        time.sleep(5)  # Simulated heavy computation
        print(f"--- [EXECUTOR] Training progress: {step + 1}/{total_steps} ---")

    # 3. SAVE METRIC: Optuna needs this single value to continue
    # Dummy logic to generate a metric based on some hyper-parameters
    lr0: float = float(train_cfg.get("lr0", 0.01))
    imgsz: int = int(train_cfg.get("imgsz", 640))

    accuracy: float = 0.80 + (lr0 * 0.5) + (imgsz / 10000.0)
    results: dict[str, Any] = {
        "status": "success",
        "accuracy": accuracy,
        "study_name": study_name,
    }

    result_path: str = "/app/data/results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"--- [EXECUTOR] Finished! Results saved to {result_path} ---")


if __name__ == "__main__":
    main()

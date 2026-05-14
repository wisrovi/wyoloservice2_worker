# Worker Executor

This is the ephemeral execution unit of the worker. It is launched by the **Invoker** as a Docker container for each training task.

## Features

- Isolated execution environment for ML training.
- GPU support via Docker runtime.
- Automatic cleanup after completion.
- Results reporting via shared volumes (`config.json` -> `results.json`).

## Workflow

1.  **Input:** Reads training configuration from a shared volume (mounted at `/app/data`).
2.  **Training:** Executes `run_training.py` with the provided parameters.
3.  **Output:** Saves metrics and results to the same shared volume.
4.  **Exit:** Terminates with exit code 0 on success or 1 on failure.

## Customization

To add libraries or change the training logic:
1.  Modify `requirements.txt` or `Dockerfile`.
2.  Update `run_training.py`.
3.  Rebuild the image: `docker build -t wisrovi/train_service:worker_executor_v1.0.0 .`

# WYOLO Service 2 Executor

This repository contains the executor service for **WYOLO Service 2**, which manages YOLO training pipelines, GPU resource allocation, and tracking integration.

## Project Structure

- **`config/`**: Configuration files and templates.
- **`wtrain/`**: Contains the training scripts and logic wrapper.
  - **`lib/src/wyolo/trainer/`**: Main trainer source code (e.g., wrapper, GPU utils).
  - **`app/_wpipe/`**: Workflow pipeline components and tracking configurations.
  - **`examples/`**: Training dataset examples (e.g., electronic component detection datasets).

## Features

1. **YOLO & RT-DETR Training Wrapper**: A unified interface to train Ultralytics models (`YOLO`, `RTDETR`) with automatic tracking support (MLflow).
2. **GPU Optimization**: Utility routines to monitor GPU performance, check hardware compatibility, and report RAM/VRAM resource availability.
3. **Rich Console Logging**: Clean terminal output with config banners using the Python `rich` package.

## Development & Usage

### Running Locally
To launch a training process, you can invoke the trainer scripts by passing the configuration JSON/YAML files:
```bash
python wtrain/lib/src/wyolo/trainer/trainer_wrapper.py
```

### Configuration
Update the dataset pathways in the `.yaml` files. The trainer dynamically rewrites dataset directory references (e.g., maps `/datasets/` to `/wyolo/control_server/datasets/`) to resolve path mapping issues inside containerized environments.

## Changelog

### v2.2.14 (2026-08-05)
- **[FIX]** PostTrain: `_find_images` now reads the dataset `data.yaml` content to resolve real `train`/`val`/`test` image dirs instead of using `os.path.dirname()` of the config `data` value. The config points to a temp copy of the data.yaml (created by `create_trainer`), so `dirname()` resolved to a `/tmp` folder with no images, leaving `post_train_results/` empty. Falls back to the old directory-relative search when the YAML cannot be read.


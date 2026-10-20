# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.2.33] - 2026-10-20
### Fixed
- **Provisional results prevention:** `mlflow_setup.py` no longer writes a preliminary `results.json` with `accuracy: 0.0` during directory organization, avoiding false metric consumption.
- **Results completion contract:** `public_results.py` emits `results.json` with explicit `status: completed` and epoch timestamp for strict verification by the invoker.

### Changed
- Bumped executor version to `2.2.33`.

## [2.2.32] - 2026-08-25
### Fixed
- **Fitness key auto-detection:** `train()` function now auto-detects the correct metric key based on task type (`classify` → `metrics/accuracy`, `detect` → `metrics/mAP50(B)`, `segment` → `metrics/mAP50(B)`). Prevents returning 0.0 when user-specified fitness key doesn't exist in results_dict.

### Changed
- Bumped executor version to `2.2.32`.

## [2.2.31] - 2026-08-25
### Fixed
- **Batch default:** `on_train_start` callback no longer crashes with `KeyError` when `batch` is not in user config (`.pop("batch")` → `.pop("batch", None)`).
- **Batch auto-derivation:** `create_trainer` now defaults `batch` to `MAX_GPU / 100` (e.g. `MAX_GPU=50` → `batch=0.5` = auto-batch at 50% VRAM) when not explicitly set by user.

### Changed
- Bumped executor version to `2.2.31`.

## [2.2.30] - 2026-08-25
### Fixed
- **LLM analyzer model:** replaced dead `opencode/deepseek-v4-flash-free` with `opencode/nemotron-3.5-lightning-free` in both `training_report_analyzer.py` and `llm_analyzer.py`. Model is now configurable via `LLM_MODEL` env var.
- **Research states fallback:** `_explain_research_states` now writes a minimal `GLOBAL_RESEARCH_EXPLANATION.md` when OpenCode is unavailable, instead of silently failing.

### Changed
- Bumped executor version to `2.2.30`.

## [2.2.29] - 2026-08-24
### Fixed
- Invalid YOLO argument sanitization in `trainer_wrapper`: user-supplied hyperparameters not supported by Ultralytics (e.g. `seg`) are now dropped with a warning before `model.train()`, instead of aborting the trial with `SyntaxError: '<key>' is not a valid YOLO argument`.

### Changed
- Bumped executor version to `2.2.29`.

## [Unreleased]
### Added
- Standardized testing infrastructure (tests folder, run_tests.sh, coverage.sh via Docker).
- GitHub issue and PR templates.
- This CHANGELOG file to replace in-README changelogs.
## [v2.2.27] - 2026-08-11

### Fixed
- Resolved `ImportError` in `CleanFolderExtra` by correctly importing `PostTrainContext` instead of the non-existent `CleanFolderObj`.

## [v2.2.26] - 2026-08-10

### Fixed
- Fixed PostTrain `_find_images` throwing `KeyError` when missing `test` or `val` splits in detection datasets, causing Grad-CAM inference to be completely skipped.
- Made `_resolve_image_dirs` correctly handle relative paths and `path:` root from detection dataset configurations, ensuring the post-training pipeline can correctly resolve and load the image sets.

## [2.2.25] - 2026-08-10
### Docs
- Expanded XAI and Grad-CAM documentation in the README for stronger I+D context.
## [2.2.24] - 2026-08-10
### Fixed
- Wipe out `model_focus` directory in `PostTrain` to prevent mixing old artifacts with new data.
## [2.2.23] - 2026-08-10
### Fixed
- Fixed `_find_images` in `post_train.py` failing to find image files in classification datasets by supporting `test/*/*` patterns.
## [2.2.22] - 2026-08-10
### Changed
- Modify `post_train_results` output directory in `post_train.py` to be saved inside `extras/post_train_results`.
## [2.2.21] - 2026-08-10
### Changed
- Modify `model_focus` output directory in `trainer_wrapper.py` to be saved inside `extras/model_focus`.
## [2.2.20] - 2026-08-10
### Changed
- Integration of `ImageECamYOLO` step in `pipeline_post_train` with required context mapping logic in `PostTrainContext`.
## [2.2.19] - 2026-08-10
### Changed
- Update `trainer_wrapper.py` and `post_train.py` to use `model_path` instead of `model` directly for post training.

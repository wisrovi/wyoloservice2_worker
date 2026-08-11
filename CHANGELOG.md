# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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

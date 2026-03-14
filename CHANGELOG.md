# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2025-12-01

### Added

- Initial implementation of spoof detection using ViT and Swin Transformer.
- Jupyter notebook with complete training pipeline.
- Dataset loading and preparation from Hugging Face.
- Model comparison between ViT and Swin architectures.
- Interactive testing with user-uploaded images.

## [1.1.0] - 2026-03-14

### Added

- Modular Python source code extracted from notebook (`src/spoof_detection/`).
- Comprehensive `README.md` with installation, usage, and project structure.
- `CONTRIBUTING.md` with contributor guidelines.
- `LICENSE` (MIT).
- `requirements.txt` and `requirements-dev.txt` for dependency management.
- Training configuration file (`configs/training_config.yaml`).
- Unit tests for data, model, and training modules.
- GitHub Actions CI workflow for linting and testing.
- Issue templates and pull request template.
- Documentation in `docs/` (setup, architecture, development guides).
- Example scripts for training and inference.

### Changed

- Reorganized repository into a professional project structure.
- Moved notebook to `notebooks/` directory.
- Removed duplicate dataset loading cell from notebook.

.PHONY: help install install-dev install-all test lint format clean docs build upload

# Default target
help:
	@echo "Wyolo - Professional YOLO Training Library"
	@echo ""
	@echo "Available commands:"
	@echo "  install      Install package in production mode"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  install-all  Install package with all dependencies"
	@echo "  test         Run tests with coverage"
	@echo "  lint         Run code linting (flake8, mypy)"
	@echo "  format       Format code (black, isort)"
	@echo "  clean        Clean build artifacts and cache"
	@echo "  docs         Generate documentation"
	@echo "  build        Build distribution packages"
	@echo "  upload       Upload to PyPI (requires twine)"
	@echo "  pre-commit   Install pre-commit hooks"
	@echo ""

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[dev,docs,tune]"

# Development targets
test:
	pytest --cov=wyolo --cov-report=html --cov-report=term-missing

test-fast:
	pytest -x -v

lint:
	flake8 src tests
	mypy src

format:
	black src tests
	isort src tests

format-check:
	black --check src tests
	isort --check-only src tests

pre-commit:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Quality checks
check: format-check lint test
	@echo "All quality checks passed! ✅"

# Documentation
docs:
	@echo "Documentation generation not yet implemented"
	@echo "Consider adding mkdocs configuration"

# Build and distribution
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

upload-test: build
	python -m twine upload --repository testpypi dist/*

# Development workflow
dev-setup: install-all pre-commit
	@echo "Development environment ready! 🚀"

# Quick commands
start: install-dev
	@echo "Development environment started successfully! 🚀"
	@echo "Library installed in editable mode."

run-example:
	python examples/basic_training.py

# GPU and ML checks
check-gpu:
	@echo "Checking GPU availability..."
	python -c "import GPUtil; print(f'GPUs found: {len(GPUtil.getGPUs())}')"

check-mlflow:
	@echo "Checking MLflow installation..."
	python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"

# Docker targets (if needed)
docker-build:
	docker build -t wyolo:latest .

docker-run:
	docker run --gpus all -v $(PWD)/examples:/app/examples wyolo:latest

# Version management
version-patch:
	bump2version patch

version-minor:
	bump2version minor

version-major:
	bump2version major

# Release workflow
release: clean check build
	@echo "Ready for release! Run 'make upload' to publish."
	@echo "Or 'make upload-test' for TestPyPI."
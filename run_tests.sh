#!/bin/bash
# Runs pytest inside a Docker container
echo "Running tests in Docker container..."
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v $(pwd):/app -w /app python:3.10-slim bash -c "apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision pytest pydantic wpipe wyolo==2.2.16 && pytest tests/"

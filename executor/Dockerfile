FROM python:3.10-slim
WORKDIR /app

# Simple dependencies for the executor
# In a real case, here you would install torch, tensorflow, etc.
RUN pip install --no-cache-dir PyYAML

COPY run_training.py .

# Entry point for the container execution
ENTRYPOINT ["python", "run_training.py"]

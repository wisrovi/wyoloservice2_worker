# Model Registration Guide

## After Training Complete

Once `python base.py` completes successfully, you can register your model in MLflow using the registration script.

### Method 1: Using the Registration Script

1. **Get the Run ID** from the training output (look for "View run" URL):
   ```
   🏃 View run train_test_training_001 at: http://192.168.1.137:23435/#/experiments/1/runs/d48657c8adfe492b89c27926d26a4ec3
   ```
   The Run ID is: `d48657c8adfe492b89c27926d26a4ec3`

2. **Register the model**:
   ```bash
   cd /app/lib/src/wyolo/core
   python register_model.py \
     --run_id d48657c8adfe492b89c27926d26a4ec3 \
     --model_name example_clasification_test_training_001
   ```

### Method 2: Manual MLflow UI Registration

1. Open MLflow UI: http://192.168.1.137:23435
2. Navigate to your experiment and run
3. Click on the model artifact in "yolo_model/best.pt"
4. Click "Register Model" button
5. Choose or create a model name
6. Add version and description

### Method 3: Programmatic Registration

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://192.168.1.137:23435")

# Register model
run_id = "d48657c8adfe492b89c27926d26a4ec3"
model_uri = f"runs:/{run_id}/yolo_model/best.pt"
registered_model = mlflow.register_model(model_uri, "your_model_name")

print(f"Model registered: {registered_model.name} v{registered_model.version}")
```

## What Gets Logged

The training now logs to MLflow:
- ✅ **PyTorch model** (for MLflow Model Registry compatibility)
- ✅ **YOLO best.pt** (ready for deployment)
- ✅ **YOLO last.pt** (checkpoint model)
- ✅ **Training metrics** (accuracy, loss, fitness)
- ✅ **Model metadata** (framework, type, paths)
- ✅ **Configuration files** (training config)
- ✅ **System info** (GPU, environment)
- ✅ **Example images** (dataset samples)

## Model Deployment

Once registered, you can deploy the model using:
```bash
# Load model from MLflow
import mlflow.pytorch

model = mlflow.pytorch.load_model("models:/your_model_name/Production")
```
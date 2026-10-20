"""WPipe Step: Public Results Publisher.

Publishes the consolidated training accuracy and completed status to results.json
for upstream consumption by the Worker Invoker and Optuna.
"""

import json
import os
import time
from typing import Any, Dict

from pydantic import BaseModel
from wpipe import step, to_obj

BASE_PATH = "/wyolo/worker/train_service_results"
FILE = "results.json"


class PublicResultsInput(BaseModel):
    """Input DTO for the public results publishing step.

    Attributes:
        results_trained_model: The evaluated accuracy or fitness metric from the trained model.
    """

    results_trained_model: Any


@step(name="publish_results_mlflow", version="v1.1", tags=["train_model"])
@to_obj(PublicResultsInput)
def publish_results_mlflow(data_input: PublicResultsInput) -> Dict[str, Any]:
    """Publish final training metrics to results.json with completion metadata.

    Args:
        data_input: Input container holding results_trained_model.

    Returns:
        Dict[str, Any]: Execution state dictionary confirming results publication.
    """
    results_path = os.path.join(BASE_PATH, FILE)
    if os.path.exists(results_path):
        os.remove(results_path)

    accuracy = data_input.results_trained_model

    if accuracy is None:
        print("--- [PUBLIC_RESULTS] Warning: accuracy is None, using 0.0 ---")
        accuracy = 0.0

    save = {
        "accuracy": round(float(accuracy), 4),
        "status": "completed",
        "timestamp": time.time(),
    }

    os.makedirs(BASE_PATH, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save, f, indent=4)
        print(f"--- [PUBLIC_RESULTS] Saved results.json with accuracy={save['accuracy']} (status=completed) ---")

    return {"public_results": True, "accuracy": save["accuracy"]}

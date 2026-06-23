from typing import Any
import os
import json
from wpipe import step, to_obj
from pydantic import BaseModel

BASE_PATH = "/wyolo/worker/train_service_results"
FILE = "results.json"


class PublicResultsInput(BaseModel):
    results_trained_model: Any


@step(name="publish_results_mlflow", version="v1.0", tags=["train_model"])
@to_obj(PublicResultsInput)
def publish_results_mlflow(data_input: PublicResultsInput):
    if os.path.exists(os.path.join(BASE_PATH, FILE)):
        # delete existing file to avoid confusion
        os.remove(os.path.join(BASE_PATH, FILE))

    accuracy = data_input.results_trained_model
    
    if accuracy is None:
        print("--- [PUBLIC_RESULTS] Warning: accuracy is None, using 0.0 ---")
        accuracy = 0.0

    with open(os.path.join(BASE_PATH, FILE), "w") as f:
        save = {"accuracy": round(float(accuracy), 4)}
        json.dump(save, f)
        print(f"--- [PUBLIC_RESULTS] Saved results.json with accuracy={save['accuracy']} ---")

    return {"public_results": True}

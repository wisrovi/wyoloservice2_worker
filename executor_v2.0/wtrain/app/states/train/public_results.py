from typing import Any
import os
import json
from wpipe import step, to_obj
from pydantic import BaseModel

BASE_PATH = "/wyolo/worker/train_service_results"
FILE = "results.json"


class PublicResultsInput(BaseModel):
    results_trained_model: Any


@step(name="train_model", version="v1.0", tags=["train_model"])
@to_obj(PublicResultsInput)
def public_results(data_input: PublicResultsInput):
    if os.path.exists(os.path.join(BASE_PATH, FILE)):
        # delete existing file to avoid confusion
        os.remove(os.path.join(BASE_PATH, FILE))

    accuracy = data_input.results_trained_model

    with open(os.path.join(BASE_PATH, FILE), "w") as f:
        save = {"accuracy": round(accuracy, 4)}
        json.dump(save, f)

    return {"public_results": True}

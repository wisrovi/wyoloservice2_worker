import os
from datetime import datetime
from types import SimpleNamespace
from wpipe import step, to_obj
from wredis.hash import RedisHashManager


REDIS_PORT = 23437
REDIS_HOST = os.getenv("CONTROL_HOST", "localhost")


def _to_dict(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, SimpleNamespace):
        return vars(obj)
    return {}

def _to_dict_deep(obj):
    if isinstance(obj, dict):
        return {k: _to_dict_deep(v) for k, v in obj.items()}
    if isinstance(obj, SimpleNamespace):
        return _to_dict_deep(vars(obj))
    if isinstance(obj, list):
        return [_to_dict_deep(i) for i in obj]
    return obj


def _get_study_id(user_config):
    cfg = _to_dict(user_config)
    study_id = cfg.get("study_id")
    if study_id:
        return study_id
    sweeper = cfg.get("sweeper", {})
    if not isinstance(sweeper, dict):
        sweeper = _to_dict(sweeper)
    return sweeper.get("study_name", "unknown")


def _make_key(prefix: str, user_config: object) -> str:
    now = datetime.now()
    date_str = now.strftime("%y-%m-%d")
    time_str = now.strftime("%H-%M")
    sid = _get_study_id(user_config)
    return f"{prefix}:{date_str}:{sid}:{time_str}"


@step(name="publish_request_redis", version="v1.0", tags=["publish"])
@to_obj
def publish_request_redis(data_input: object):
    data = _to_dict(data_input)
    hm = RedisHashManager(host=REDIS_HOST, port=REDIS_PORT)

    user_config = data.get("user_config")
    request_key = _make_key("wyolo:request", user_config)
    uc = _to_dict(user_config)
    config_data = _to_dict_deep({
        "model": uc.get("model"),
        "train": uc.get("train"),
        "sweeper": uc.get("sweeper"),
        "type": uc.get("type"),
    })
    hm.create_hash(request_key, "config", config_data)

    return {}


@step(name="publish_results_redis", version="v1.0", tags=["publish"])
@to_obj
def publish_results_redis(data_input: object):
    data = _to_dict(data_input)
    hm = RedisHashManager(host=REDIS_HOST, port=REDIS_PORT)

    user_config = data.get("user_config")

    results_trained_model = data.get("results_trained_model")
    pipeline_error = data.get("error")
    if results_trained_model is not None:
        result_key = _make_key("wyolo:results", user_config)
        hm.create_hash(
            result_key,
            "results",
            {"accuracy": round(float(results_trained_model), 4)},
        )
    else:
        error_key = _make_key("wyolo:errors", user_config)
        if pipeline_error:
            hm.create_hash(error_key, "error", {"error": f"Pipeline error: {pipeline_error}"})
        else:
            gpu_status = data.get("gpu_status")
            dataset_status = data.get("dataset_status")
            errors = []
            if gpu_status is None:
                errors.append("GPU check not run")
            elif gpu_status != 1:
                errors.append(f"GPU not available (status={gpu_status})")
            if dataset_status is None:
                errors.append("Dataset check not run")
            elif dataset_status != 1:
                errors.append(f"Dataset not found (status={dataset_status})")
            if not errors:
                errors.append(f"Unknown error - gpu_status={gpu_status}, dataset_status={dataset_status}")
            hm.create_hash(error_key, "error", {"error": "; ".join(errors)})

    return {}

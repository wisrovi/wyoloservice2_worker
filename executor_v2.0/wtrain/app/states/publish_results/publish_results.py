"""WPipe steps for synchronizing training requests and execution outcomes to Redis."""

from datetime import datetime
import os
from types import SimpleNamespace
from typing import Any, Dict
from wpipe import step, to_obj
from wredis.hash import RedisHashManager

REDIS_PORT = 23437
REDIS_HOST = os.getenv("CONTROL_HOST", "localhost")


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Convert arbitrary object or SimpleNamespace to dict.

    Args:
        obj (Any): Target object or namespace.

    Returns:
        Dict[str, Any]: Extracted dictionary.
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, SimpleNamespace):
        return vars(obj)
    return {}


def _to_dict_deep(obj: Any) -> Any:
    """Recursively convert nested SimpleNamespace instances to dicts.

    Args:
        obj (Any): Arbitrary object structure.

    Returns:
        Any: Fully transformed structure using standard dictionaries and lists.
    """
    if isinstance(obj, dict):
        return {k: _to_dict_deep(v) for k, v in obj.items()}
    if isinstance(obj, SimpleNamespace):
        return _to_dict_deep(vars(obj))
    if isinstance(obj, list):
        return [_to_dict_deep(i) for i in obj]
    return obj


def _get_study_id(user_config: Any) -> str:
    """Extract study identifier from configuration dict or sweeper config.

    Args:
        user_config (Any): User configuration structure.

    Returns:
        str: Resolved study identifier.
    """
    cfg = _to_dict(user_config)
    study_id = cfg.get("study_id")
    if study_id:
        return str(study_id)
    sweeper = cfg.get("sweeper", {})
    if not isinstance(sweeper, dict):
        sweeper = _to_dict(sweeper)
    return str(sweeper.get("study_name", "unknown"))


def _make_key(prefix: str, user_config: Any) -> str:
    """Generate time-stamped Redis key path.

    Args:
        prefix (str): Redis key namespace prefix.
        user_config (Any): Associated configuration object.

    Returns:
        str: Formatted Redis key string.
    """
    now = datetime.now()
    date_str = now.strftime("%y-%m-%d")
    time_str = now.strftime("%H-%M")
    sid = _get_study_id(user_config)
    return f"{prefix}:{date_str}:{sid}:{time_str}"


@step(name="publish_request_redis", version="v1.0", tags=["publish"])
@to_obj
def publish_request_redis(data_input: Any) -> Dict[str, Any]:
    """Publish requested training job configuration to Redis hash store.

    Args:
        data_input (Any): Execution pipeline context.

    Returns:
        Dict[str, Any]: Empty acknowledgment dictionary.
    """
    data = _to_dict(data_input)
    hm = RedisHashManager(host=REDIS_HOST, port=REDIS_PORT)

    user_config = data.get("user_config")
    request_key = _make_key("wyolo:request", user_config)
    uc = _to_dict(user_config)
    config_data = _to_dict_deep(
        {
            "model": uc.get("model"),
            "train": uc.get("train"),
            "sweeper": uc.get("sweeper"),
            "type": uc.get("type"),
        }
    )
    hm.create_hash(request_key, "config", config_data)
    return {}


@step(name="publish_results_redis", version="v1.0", tags=["publish"])
@to_obj
def publish_results_redis(data_input: Any) -> Dict[str, Any]:
    """Publish final accuracy metrics or error diagnostics to Redis hash store.

    Args:
        data_input (Any): Execution pipeline context.

    Returns:
        Dict[str, Any]: Empty acknowledgment dictionary.
    """
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
            hm.create_hash(
                error_key, "error", {"error": f"Pipeline error: {pipeline_error}"}
            )
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
                errors.append(
                    f"Unknown error - gpu_status={gpu_status}, dataset_status={dataset_status}"
                )
            hm.create_hash(error_key, "error", {"error": "; ".join(errors)})

    return {}

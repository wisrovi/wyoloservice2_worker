"""Declarative WPipe pipeline assembly for wtrain worker execution."""

from wpipe import Condition, Pipeline

from states.check import check_dataset, check_gpu_available, check_minio_buckets
from states.error_process import error_capture
from states.load_yaml import load_yaml
from states.publish_results import publish_request_redis, publish_results_redis
from states.train import not_train, publish_results_mlflow, train_model


def build_wtrain_pipeline(
    tracking_db: str = "/wyolo/worker/events/wtrain.db",
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> Pipeline:
    """Instantiate and configure the primary wtrain execution pipeline.

    Args:
        tracking_db (str): Path to SQLite tracking database file.
        max_retries (int): Maximum retry attempts for retriable exceptions.
        retry_delay (float): Wait interval in seconds between retry attempts.

    Returns:
        Pipeline: Fully configured WPipe pipeline instance.
    """
    pipe = Pipeline(
        pipeline_name="wtrain_pipe",
        verbose=False,
        tracking_db=tracking_db,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_on_exceptions=(RuntimeError,),
        collect_system_metrics=True,
        show_progress=False,
    )

    pipe.add_error_capture([error_capture])

    pipe.set_steps(
        [
            load_yaml,
            publish_request_redis,
            check_minio_buckets,
            check_gpu_available,
            check_dataset,
            Condition(
                expression="gpu_status == 1 and dataset_status == 1",
                branch_true=[
                    train_model,
                    publish_results_mlflow,
                ],
                branch_false=[
                    not_train,
                ],
            ),
            publish_results_redis,
        ]
    )
    return pipe


# Declarative pipeline singleton for WPipe architecture compliance and visualizer discovery
worker_wtrain_pipeline: Pipeline = build_wtrain_pipeline()

__all__ = ["build_wtrain_pipeline", "worker_wtrain_pipeline"]

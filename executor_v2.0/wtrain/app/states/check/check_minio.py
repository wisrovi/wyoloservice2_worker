"""WPipe step for validating and provisioning required object storage buckets in MinIO."""

from typing import Any, Dict
import boto3
from botocore.client import Config
from loguru import logger
from wpipe import step, to_obj

from utils.config_manager import read_base_config


@step(name="check_minio_buckets", version="v1.0", tags=["check_minio"])
@to_obj
def check_minio_buckets(data_input: Any) -> Dict[str, Any]:
    """Ensure required buckets exist in the MinIO S3 object store.

    Args:
        data_input (Any): Execution pipeline context.

    Returns:
        Dict[str, Any]: Status dictionary confirming MinIO validation.
    """
    cfg = read_base_config()
    minio_cfg = cfg.get("minio", {})
    endpoint = minio_cfg.get("MINIO_ENDPOINT")
    access_key = minio_cfg.get("MINIO_ID")
    secret_key = minio_cfg.get("MINIO_SECRET_KEY")

    required_buckets = ["models", "mlflow-artifacts", "dvcstorage"]

    try:
        s3 = boto3.resource(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

        for bucket_name in required_buckets:
            bucket = s3.Bucket(bucket_name)
            if bucket.creation_date:
                logger.info(f"MinIO bucket '{bucket_name}' already exists.")
            else:
                logger.info(f"Creating MinIO bucket '{bucket_name}'...")
                bucket.create()
                logger.info(f"MinIO bucket '{bucket_name}' created successfully.")
    except Exception as exc:
        logger.error(f"Error while validating or creating MinIO buckets: {exc}")

    return {"minio_status": 1}

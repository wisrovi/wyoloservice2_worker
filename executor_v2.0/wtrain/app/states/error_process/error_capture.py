"""WPipe step for capturing and pretty-printing runtime pipeline execution errors."""

from typing import Any, Dict
from loguru import logger
from wpipe import to_obj
from wpipe.decorators import step

ERROR_LOG_FOLDER = "/results/errors"


@step(name="error_capture", version="v1.0", tags=["error_capture"])
@to_obj
def error_capture(context: Any, error: Dict[str, Any]) -> Any:
    """Handle uncaught step exceptions, logging error diagnostics.

    Args:
        context (Any): Execution context state at time of exception.
        error (Dict[str, Any]): Dictionary containing error details and traceback.

    Returns:
        Any: Unmodified context.
    """
    logger.error("SYSTEM ALERT: EXECUTION ERROR DETECTED")
    logger.error(f"Failed Step: {error.get('step_name')}")
    logger.error(f"File Path: {error.get('file_path')}")
    logger.error(f"Line Number: {error.get('line_number')}")
    logger.error(f"Error Message: {error.get('error_message')}")
    logger.error(f"Error Traceback: {error.get('error_traceback')}")
    return context

"""WPipe step triggered when precondition checks (GPU or dataset) fail."""

from typing import Any, Dict
from loguru import logger
from wpipe import step, to_obj


@step(name="not_train", version="v1.0", tags=["not_train"])
@to_obj
def not_train(data_input: Any) -> Dict[str, Any]:
    """Log warning and abort training progression when preconditions are not met.

    Args:
        data_input (Any): Execution pipeline context.

    Returns:
        Dict[str, Any]: Resulting state with public_results set to False.
    """
    logger.warning("There is no dataset or GPU available for training.")
    return {"public_results": False}

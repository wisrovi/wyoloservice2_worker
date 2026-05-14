from wpipe import step, to_obj
from loguru import logger


@step(name="not_train", version="v1.0", tags=["not_train"])
@to_obj
def not_train(data_input: object):
    logger.warning("There aren't any datas available for training")

    return {"public_results": False}

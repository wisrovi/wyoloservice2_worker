"""
python wtrain.py --file "/wyolo/control_server/datasets/clasification/colorball.v8i.multiclass/config_train.yaml"
python wtrain.py --file "/wyolo/control_server/datasets/detection/Deteksi komponen elektronik.v1i.yolov8/config_train.yaml"
python wtrain.py --file "/wyolo/control_server/datasets/segmentation/ArchitecturePlan/config_train.yaml"


"""

import sys
import os
import logging
from tenacity import retry, stop_after_attempt, wait_fixed, before_sleep_log
from application.wyolo_train import main
from application.utils.util import get_argument_parser, get_user_config
from application.utils.gpu import check_gpu_available


os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get(
    "LD_LIBRARY_PATH", ""
)


# Configura un log básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRY_DELAY_SECONDS = 10
RETRY_MAX_ATTEMPTS = 5


@retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_fixed(RETRY_DELAY_SECONDS),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def train(user_config_train):
    return main(user_config_train)


@retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_fixed(RETRY_DELAY_SECONDS),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def check_gpu_before_to_train():
    return check_gpu_available()


if __name__ == "__main__":
    try:
        _user_config_file = get_argument_parser()

        user_config_train = get_user_config(user_config_file=_user_config_file)

        # Llamada a la función de entrenamiento si la GPU está lista
        if check_gpu_before_to_train():
            train(user_config_train)

    except Exception as e:
        print("\n" * 2)  # Espacio para claridad en la salida
        print(f"❌ Error definitivo tras {RETRY_MAX_ATTEMPTS} reintentos: {e}")
        sys.exit(1)

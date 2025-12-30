import sys
import os
import subprocess
import torch
import logging
from tenacity import retry, stop_after_attempt, wait_fixed, before_sleep_log


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
def check_gpu_and_train():
    test_command = [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader,nounits",
    ]
    print("\n" * 5)  # Espacio para claridad en la salida
    print(f"DEBUG: CUDA Runtime Version: {torch.version.cuda}")
    print(
        f"DEBUG: Driver Version: {subprocess.check_output(test_command).decode().strip()}"
    )

    # Hito: Intento de despertar el driver a nivel sistema
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except:
        pass

    if not torch.cuda.is_available():
        # Aquí podrías usar tus "prints bonitos" con colorama o rich
        print("⚠️ GPU no detectada. Reintentando...")
        raise RuntimeError("La GPU no respondió a tiempo.")

    print("✅ GPU Detectada. Iniciando entrenamiento...")

    # Importación diferida para evitar cargar todo si la GPU no está lista
    from application.wyolo_train import main

    return main()


if __name__ == "__main__":
    try:
        check_gpu_and_train()
    except Exception as e:
        print(f"❌ Error definitivo tras {RETRY_MAX_ATTEMPTS} reintentos: {e}")
        sys.exit(1)

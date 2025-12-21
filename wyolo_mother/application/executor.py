import os
import yaml
import re
import subprocess
from wredis.streams import RedisStreamManager
from loguru import logger

from setproctitle import setproctitle  # Para cambiar el nombre del proceso

# Cambiar el nombre del proceso
setproctitle("train_service")


def ejecutar_comando(args: list, trial_number: int, verbose=True, config_path=None):
    stream_manager = None
    task_id = None
    if config_path:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)

        redis_config = user_config.get("redis", {})
        task_id = user_config.get("task_id", None)

        stream_manager = RedisStreamManager(
            host=redis_config.get("REDIS_HOST"),
            port=redis_config.get("REDIS_PORT"),
            db=redis_config.get("REDIS_DB"),
            verbose=False,
        )

    buffer = []  # Para almacenar toda la salida y parsear al final
    try:
        # Ejecutar el comando y redirigir stdout y stderr
        with subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combinar stderr con stdout
            text=True,
            bufsize=1,  # Line-buffered (1 línea a la vez)
            cwd=os.getcwd(),
        ) as process:
            # Leer la salida en tiempo real
            for line in process.stdout:
                buffer.append(line)  # Almacenar línea para parsear después
                if verbose:
                    logger.info(line, end="")  # Mostrar en tiempo real

                if stream_manager:
                    stream_manager.add_to_stream(
                        key=f"stream:{task_id}",
                        data={
                            "value": line,
                        },
                        ttl=60 * 60,
                    )

            # Esperar a que el proceso termine
            process.wait()

            # Verificar código de salida
            if process.returncode != 0:
                print(f"Error (código {process.returncode}):")
                print("".join(buffer))  # Mostrar toda la salida en caso de error
                return None

    except Exception as e:
        print(f"Excepción inesperada: {e}")
        return None

    # Parsear el resultado final del buffer
    salida_completa = "".join(buffer)

    try:
        # guardar el resultado para analisis posteriores
        config_path = args[2].split("=")[-1]
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        experiment_name = config_data.get("sweeper").get("study_name")
        tempfile = config_data.get("tempfile", "")

        RESULT_PATH = f"{tempfile}/models/{experiment_name}/{config_data['type']}/{config_data['task_id']}"

        with open(
            f"{RESULT_PATH}/trail_history/trial_{int(trial_number)}.train_log", "w"
        ) as f:
            f.write(salida_completa)
    except Exception as e:
        print(f"No se pudo guardar el log del entrenamiento: {str(e)}")

    match = re.search(r"ResultadoFinal:\s*(\d+\.\d+)", salida_completa)
    if match:
        return float(match.group(1))
    else:
        print("Resultado no encontrado en la salida.")
        return None


def train_run(
    config_path: str,
    trial_number: int,
    verbose: bool = False,
    fitness: str = "fitness",
    script_path: str = "/app/lib/src/wyolo/core",
):
    os.chdir(script_path)

    args = [
        "python",
        f"yolo_train.py",
        f"--config_path={config_path}",
        f"--trial_number={trial_number}",
        f"--fitness={fitness}",
    ]
    # example:
    # python yolo_train.py --config_path="./media/dataset/example/classify/colorball.v8i.multiclass/config_train.yaml" --trial_number=1 --fitness=fitness
    print(args)
    import time
    time.sleep(30)

    resultado = ejecutar_comando(
        args,
        trial_number,
        verbose=True,
        config_path=config_path,
    )

    return resultado


if __name__ == "__main__":
    resultado = train_run(
        config_path="/datasets/clasification/colorball.v8i.multiclass/config_train.yaml",
        trial_number=1,
        verbose=False,
    )
    if resultado is not None:
        print(f"Resultado del entrenamiento: {resultado}")

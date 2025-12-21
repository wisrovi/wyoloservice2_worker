import argparse
import yaml
import sys
import os

from application.executor import train_run

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "src"))

from setproctitle import setproctitle
from application.utils.util import get_complete_config
# from lib.src.wyolo.core.yolo_train import create_trainer, train
from lib.src.wyolo.trainer.trainer_wrapper import create_trainer, train


setproctitle("train_service")


def get_user_config(
    default_file: str = "/wyolo/control_server/datasets/clasification/colorball.v8i.multiclass/config_train.yaml",
):
    # Configuración de argparse para recibir el archivo
    parser = argparse.ArgumentParser(description="Procesador de archivos YAML.")

    # Definimos el argumento posicional 'file'
    parser.add_argument(
        "file",
        help="Ruta al archivo .yaml para entrenar el modelo",
        default=default_file,
    )

    # Parseamos los argumentos de la línea de comandos
    args = parser.parse_args()

    user_config_train = args.file

    # Verificamos si el archivo existe antes de intentar abrirlo
    if not os.path.exists(user_config_train):
        print(f"Error: El archivo '{user_config_train}' no existe.")
        sys.exit(1)

    try:
        with open(user_config_train, "r", encoding="utf-8") as archivo:
            # Cargamos el contenido del YAML
            user_config_train_data = yaml.safe_load(archivo)
    except yaml.YAMLError as exc:
        print(f"Error al leer el archivo YAML: {exc}")
        sys.exit(1)
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        sys.exit(1)

    os.makedirs("/wyolo/worker/train_service_results/training_artifacts", exist_ok=True)

    with open(
        "/wyolo/worker/train_service_results/training_artifacts/user_config_train.yaml",
        "w",
        encoding="utf-8",
    ) as file:
        yaml.dump(
            user_config_train_data,
            file,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    return user_config_train


def main():
    # python main.py "/wyolo/control_server/datasets/clasification/colorball.v8i.multiclass/config_train.yaml"
    # python main.py "/wyolo/control_server/datasets/detection/Deteksi komponen elektronik.v1i.yolov8/config_train.yaml"
    # python main.py "/wyolo/control_server/datasets/segmentation/ArchitecturePlan/config_train.yaml"
    user_config_train = get_user_config()

    config_dict, config_path = get_complete_config(user_config=user_config_train)
    # trial_number = final_config.get("trial_number", 0)

    trainer, request_config = create_trainer(config_path=config_path, trial_number=1)
    if "train" in request_config:
        fitness = config_dict.get("fitness", "fitness")
        request_config = train(trainer, request_config, fitness)

    print(request_config)


if __name__ == "__main__":
    """
    EXAMPLE:
        python yolo_train.py --config_path="/datasets/clasificacion/clasificador_arepo_perfil/config_train.yaml" --trial_number=1
    """

    main()

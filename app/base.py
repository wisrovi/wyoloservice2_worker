from setproctitle import setproctitle
from lib.src.wyolo.core.yolo_train import create_trainer, train
from application.utils.util import get_complete_config

setproctitle("train_service")


def execute(user_config_train):
    config_dict, config_path = get_complete_config(user_config=user_config_train)

    fitness = config_dict.get("fitness", "fitness")
    # trial_number = final_config.get("trial_number", 0)

    trainer, request_config = create_trainer(
        config_path=config_path, trial_number=1
    )
    if "train" in request_config:
        request_config = train(trainer, request_config, fitness)

    return request_config


if __name__ == "__main__":
    """
    EXAMPLE:
        python yolo_train.py --config_path="/datasets/clasificacion/clasificador_arepo_perfil/config_train.yaml" --trial_number=1
    """

    request_config = execute(
        user_config_train="/datasets/clasification/colorball.v8i.multiclass/config_train.yaml"
    )

    print(request_config)

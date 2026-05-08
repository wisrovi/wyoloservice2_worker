# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "src"))
import tempfile
import yaml
from setproctitle import setproctitle
from application.utils.util import (
    get_complete_config,
    get_user_config,
    get_argument_parser,
)

# debug:
# from lib.src.wyolo.trainer.trainer_wrapper import create_trainer, train

# production:
from wyolo.trainer.trainer_wrapper import create_trainer, train


setproctitle("wyolo_service")


def main(user_config_train):

    config_dict, config_path = get_complete_config(user_config=user_config_train)
    # trial_number = final_config.get("trial_number", 0)

    with tempfile.TemporaryDirectory(prefix="wyolo_") as temp_dir:
        config_dict["tempfile"] = temp_dir

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        trainer, request_config = create_trainer(
            config_path=config_path, trial_number=1
        )
        if "train" in request_config:
            fitness = config_dict.get("sweeper", {}).get("fitness", "fitness")
            request_config = train(trainer, request_config, fitness)

    print(request_config)


if __name__ == "__main__":
    """
    EXAMPLE:
        python yolo_train.py --config_path="/datasets/clasificacion/clasificador_arepo_perfil/config_train.yaml" --trial_number=1
    """

    _user_config_args = get_argument_parser()

    user_config_train = get_user_config(user_config_train=_user_config_args)

    main(user_config_train)

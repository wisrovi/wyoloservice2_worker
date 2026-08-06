import os
import shutil
from glob import glob

import yaml
from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext


@step(name="PostTrain", version="v1.0")
class PostTrain:

    MAX_IMAGES_TO_PROCESS = 10  # Limit to processing up to 10 images for now

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext):
        model = ctx.model
        images_test_path = ctx.images_test_path
        project_path = ctx.project_path

        all_images = self._find_images(images_test_path)

        print(
            f"[PostTrain] Using {len(all_images)} images for post-training predictions."
        )

        post_train_results_path = os.path.join(project_path, "post_train_results")
        if os.path.exists(post_train_results_path):
            shutil.rmtree(post_train_results_path)
        os.makedirs(post_train_results_path, exist_ok=True)

        counter = 0
        for image in all_images:
            try:
                if not hasattr(model, "predict"):
                    raise AttributeError("The model does not have a 'predict' method.")

                model.predict(
                    image,
                    save=True,
                    conf=0.15,
                    exist_ok=True,
                    project=project_path,
                    name="post_train_results",
                    verbose=False,
                )
            except Exception as e:
                print(f"[PostTrain] Error processing image {image}: {e}")

            counter += 1
            if counter >= self.MAX_IMAGES_TO_PROCESS:
                break

        print(f"[PostTrain] Done. Predictions saved to {post_train_results_path}.")
        return {}

    def _find_images(self, images_test_path: str) -> list[str]:
        """Locate images for prediction: test, then val/valid, then train as fallback.

        The data path may be a detection dataset YAML (absolute image dirs are read
        from its content, since the config points to a temp copy of the YAML) or a
        classification directory.
        """
        image_dirs = self._resolve_image_dirs(images_test_path)

        if image_dirs:
            candidates = [
                os.path.join(image_dirs["test"], "*"),
                os.path.join(image_dirs["val"], "*"),
                os.path.join(image_dirs["train"], "*"),
            ]
        else:
            folder_path = (
                os.path.dirname(images_test_path)
                if os.path.isfile(images_test_path)
                else images_test_path
            )
            candidates = [
                os.path.join(folder_path, "test", "images", "*"),
                os.path.join(folder_path, "val", "images", "*"),
                os.path.join(folder_path, "valid", "images", "*"),
            ]

        for pattern in candidates:
            images = [
                img
                for img in glob(pattern)
                if os.path.isfile(img) and self._is_valid_image(img)
            ]
            if images:
                return images

        train_images = []
        if image_dirs:
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                train_images.extend(glob(os.path.join(image_dirs["train"], ext)))
        else:
            folder_path = (
                os.path.dirname(images_test_path)
                if os.path.isfile(images_test_path)
                else images_test_path
            )
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                train_images.extend(
                    glob(os.path.join(folder_path, "train", "images", ext))
                )
        if train_images:
            print("[PostTrain] No test/val images found, falling back to train images.")
            return [
                img
                for img in train_images
                if os.path.isfile(img) and self._is_valid_image(img)
            ]

        return []

    def _resolve_image_dirs(self, images_test_path: str) -> dict:
        """Read absolute image dirs (train/val/test) from a detection dataset YAML.

        Returns an empty dict when the path is a directory (classification dataset)
        or the YAML cannot be read.
        """
        if not images_test_path or not os.path.isfile(images_test_path):
            return {}

        try:
            with open(images_test_path, "r") as file:
                data_yaml_config = yaml.safe_load(file) or {}

            dirs = {}
            for split in ("train", "val", "test"):
                split_path = data_yaml_config.get(split)
                if not isinstance(split_path, str):
                    continue
                if os.path.isdir(split_path):
                    dirs[split] = split_path
                elif os.path.isdir(os.path.join(split_path, "images")):
                    dirs[split] = os.path.join(split_path, "images")
            return dirs
        except Exception as e:
            print(f"[PostTrain] Could not read data yaml {images_test_path}: {e}")
            return {}

    @staticmethod
    def _is_valid_image(img: str) -> bool:
        """Filter out YOLO metric plots and charts from candidate images."""
        basename_lower = os.path.basename(img).lower()
        path_lower = img.lower()

        if any(folder in path_lower for folder in ("runs/", "post_train_results/")):
            return False
        if "confusion_matrix" in basename_lower or "curve" in basename_lower:
            return False
        if basename_lower.startswith(("train_batch", "val_batch", "results")):
            return False
        if basename_lower in ("labels.jpg", "labels_correlogram.jpg"):
            return False
        return True

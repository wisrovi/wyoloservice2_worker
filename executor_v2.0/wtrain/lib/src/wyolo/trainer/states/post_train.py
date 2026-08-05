import os
import shutil
from glob import glob

from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext


@step(name="step_name", version="v1.0")
class PostTrain:

    MAX_IMAGES_TO_PROCESS = 10  # Limit to processing up to 10 images for now

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext):
        model = ctx.model
        images_test_path = ctx.images_test_path
        project_path = ctx.project_path

        folder_path = os.path.dirname(images_test_path)

        all_images = self._find_images(folder_path)

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
                    conf=0.005,
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

        print(
            f"[PostTrain] Done. Predictions saved to {post_train_results_path}."
        )
        return {}

    def _find_images(self, folder_path: str) -> list[str]:
        """Locate images for prediction: test, then val/valid, then train as fallback."""
        candidates = [
            os.path.join(folder_path, "test", "images", "*"),
            os.path.join(folder_path, "val", "images", "*"),
            os.path.join(folder_path, "valid", "images", "*"),
        ]

        for pattern in candidates:
            images = [
                img for img in glob(pattern)
                if os.path.isfile(img) and self._is_valid_image(img)
            ]
            if images:
                return images

        train_images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            train_images.extend(
                glob(os.path.join(folder_path, "train", "images", ext))
            )
        if train_images:
            print("[PostTrain] No test/val images found, falling back to train images.")
            return [
                img for img in train_images
                if os.path.isfile(img) and self._is_valid_image(img)
            ]

        return []

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

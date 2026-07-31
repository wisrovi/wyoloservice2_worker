import os
import shutil
from glob import glob

from ultralytics import YOLO
from wpipe import Pipeline, PipelineContext, step, to_obj
from wpipe.exception.api_error import ProcessError

db_path = "output/tracking.db"  # Path to tracking database for to save metrics, events, alerts and execution history (with error capture)
config_dir = "configs"

pipeline_post_train = Pipeline(
    pipeline_name="professional_post_train_pipeline",
    pipeline_version="1.0.0",
    verbose=False,  # Toggle detailed logging for debugging and monitoring
    show_progress=True,  # Display a progress bar during pipeline execution
)


class MyContext(PipelineContext):
    model: YOLO
    project_path: str
    images_test_path: str


@step(name="step_name", version="v1.0")
class PostTrain:

    MAX_IMAGES_TO_PROCESS = 10  # Limit to processing up to 10 images for now

    @to_obj(MyContext)
    def __call__(self, ctx: MyContext):
        # Access typed data with ctx.field

        model = ctx.model
        images_test_path = ctx.images_test_path
        project_path = ctx.project_path

        # get the folder path of the images_test_path
        folder_path = os.path.dirname(images_test_path)

        # Try to find images in standard directories (test/images/ or val/images/)
        test_images_glob = os.path.join(folder_path, "test", "images", "*")
        all_images = glob(test_images_glob)

        if not all_images:
            val_images_glob = os.path.join(folder_path, "val", "images", "*")
            all_images = glob(val_images_glob)

            if not all_images:
                val_images_glob = os.path.join(folder_path, "valid", "images", "*")
                all_images = glob(val_images_glob)

        if not all_images:
            # Fallback to search recursively for image extensions inside the folder_path
            all_images = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                all_images.extend(
                    glob(os.path.join(folder_path, "**", ext), recursive=True)
                )

        print(f"Found {len(all_images)} images for post-training processing.")
        print(
            f"Example images: {all_images[:5]}"
        )  # Print first 5 images for verification

        # Selectively filter out YOLO training metrics plots and charts
        filtered_images = []
        for img in all_images:
            if not os.path.isfile(img):
                continue

            basename_lower = os.path.basename(img).lower()
            path_lower = img.lower()

            # Skip runs and post_train directories
            if any(folder in path_lower for folder in ("runs/", "post_train_results/")):
                continue

            # Skip specific YOLO metric graphs and plots
            if "confusion_matrix" in basename_lower or "curve" in basename_lower:
                continue
            if basename_lower.startswith(("train_batch", "val_batch", "results")):
                continue
            if basename_lower in ("labels.jpg", "labels_correlogram.jpg"):
                continue

            filtered_images.append(img)

        all_images = filtered_images

        # if the folder post_train_results exists, delete it and create a new one
        post_train_results_path = os.path.join(project_path, "post_train_results")
        if os.path.exists(post_train_results_path):

            shutil.rmtree(post_train_results_path)
        os.makedirs(post_train_results_path, exist_ok=True)

        counter = 0
        for image in all_images:
            print(f"Processing image: {image}")

            try:
                # Ensure the model has a predict method
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
                print(f"Error processing image {image}: {e}")

            counter += 1
            if (
                counter >= self.MAX_IMAGES_TO_PROCESS
            ):  # Limit to processing only one image for now
                print("Processed one image, stopping further processing for now.")
                break  # Remove this break if you want to process all images

        return {}


pipeline_post_train.set_steps(
    [
        PostTrain(),
    ]
)


if __name__ == "__main__":
    initial_data_dict = (
        ...
    )  # define your initial data dictionary here for the pipeline run

    try:
        result = pipeline_post_train.run(initial_data_dict)
    except ProcessError as e:
        print(f"Error occurred: {e}")

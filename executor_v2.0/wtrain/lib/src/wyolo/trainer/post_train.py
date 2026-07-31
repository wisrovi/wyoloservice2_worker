import os
from glob import glob

from numpy.testing import verbose
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
class StepClass:

    @to_obj(MyContext)
    def __call__(self, ctx: MyContext):
        # Access typed data with ctx.field

        model = ctx.model
        images_test_path = ctx.images_test_path
        project_path = ctx.project_path

        # get the folder path of the images_test_path
        folder_path = os.path.dirname(images_test_path)

        all_images = glob(os.path.join(folder_path, "*"))
        for image in all_images:
            print(f"Processing image: {image}")
            # Here you can add your processing logic for each image

            try:
                # Ensure the model has a predict method
                if not hasattr(model, "predict"):
                    raise AttributeError("The model does not have a 'predict' method.")

                model.predict(
                    image,
                    save=True,
                    conf=0.15,
                    project=project_path,
                    name="post_train_results",
                    verbose=False,
                )  # Assuming the model has a predict method

            except Exception as e:
                print(f"Error processing image {image}: {e}")
                # Optionally, you can log the error or handle it as needed

            # only for development purposes, we will break after processing the first image
            break  # Remove this break if you want to process all images

        return {}


pipeline_post_train.set_steps(
    [
        StepClass(),
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

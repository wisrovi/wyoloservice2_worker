from wpipe import Pipeline
from wpipe.exception.api_error import ProcessError


from .states import LlmAnalyzer, PostTrain

db_path = "output/tracking.db"  # Path to tracking database for to save metrics, events, alerts and execution history (with error capture)
config_dir = "configs"

pipeline_post_train = Pipeline(
    pipeline_name="professional_post_train_pipeline",
    pipeline_version="1.0.0",
    verbose=False,  # Toggle detailed logging for debugging and monitoring
    show_progress=True,  # Display a progress bar during pipeline execution
)

pipeline_post_train.set_steps(
    [
        PostTrain(),
        LlmAnalyzer(),
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

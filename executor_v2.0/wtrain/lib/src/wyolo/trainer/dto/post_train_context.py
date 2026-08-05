from ultralytics import YOLO
from wpipe import PipelineContext


class PostTrainContext(PipelineContext):
    """Shared context for the post-training pipeline.

    Attributes:
        model: Trained YOLO model used for post-training predictions.
        project_path: Absolute path where artifacts and results are stored.
        images_test_path: Path to the dataset YAML (used to locate test images).
    """

    model: YOLO
    project_path: str
    images_test_path: str

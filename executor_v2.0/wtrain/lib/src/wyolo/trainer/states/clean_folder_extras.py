import os
import shutil

from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext
from ..utils.training_report_analyzer import TrainingReportAnalyzer


@step(name="CleanFolderExtra", version="v1.0")
class CleanFolderExtra:

    FOLDER_CLEAN = ["post_train_results", "model_focus", "llm"]

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext):
        project_path = ctx.project_path

        for folder in self.FOLDER_CLEAN:
            folder_path = os.path.join(project_path, "extras", folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                os.makedirs(folder_path, exist_ok=True)

        return {"cleaned_folders": True}

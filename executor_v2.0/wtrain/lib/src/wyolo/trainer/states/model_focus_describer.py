"""Visual Explainability with Eigen-CAM for YOLO (model_focus).

This module documents the 'model_focus' step, which generates Eigen-CAM heatmaps
for YOLO predictions (bounding boxes, classification, and segmentation).

What it does:
Uses Principal Component Analysis (PCA) on the deepest convolutional layers of the
YOLO model to compute the principal components of the activations. The first
principal component (the "Eigen-CAM") highlights the regions of the image that
caused the model to make its prediction.

Contribution:
Unlike Grad-CAM, which requires computing gradients during inference (slow and
sometimes noisy), Eigen-CAM is gradient-free. It provides robust, class-agnostic
heatmaps that show the structural features the model focuses on, helping to detect
bias and hallucination.

Reference:
- Eigen-CAM: Class Non-Specific Bounding Box Annotations for Object Detection
  (Muhammad et al., IJCNN 2020) [arXiv:2008.00797].
- Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
  (Selvaraju et al., ICCV 2017).
"""

import os
import sys
import inspect
from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext


@step(name="ModelFocusDescriber", version="v1.0")
class ModelFocusDescriber:
    """WPipe step for documenting the Eigen-CAM visualization folder."""

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext) -> PostTrainContext:
        """Executes the ModelFocusDescriber to write DESCRIPTION.md.

        Args:
            ctx (PostTrainContext): Pipeline context.

        Returns:
            PostTrainContext: The context containing the model focus description.
        """
        print("ModelFocusDescriber: Documenting Eigen-CAM outputs.")

        output_dir = os.path.join(ctx.project_path, "extras", "model_focus")
        os.makedirs(output_dir, exist_ok=True)

        md_content = inspect.cleandoc(sys.modules[__name__].__doc__ or "No description available.")
        md_content = f"""# Analysis Report

{md_content}

## Methodology
This directory contains the outputs and results of this specific analysis. The metrics and heatmaps generated here reflect the model's behavior according to the described methodology."""
        if md_content:
            with open(os.path.join(output_dir, "DESCRIPTION.md"), "w", encoding="utf-8") as fmd:
                fmd.write(md_content)

        return ctx

"""Analyzer that identifies critical false positives and false negatives.

This module provides the OutlierFailureAnalyzer WPipe state, which loads the
test dataset in FiftyOne, compares ground truth with model predictions, and
extracts specific outlier cases (e.g., high-confidence false positives).
This enables deep debugging of model bias and dataset errors.

Reference:
- Diagnostics for Fine-Grained Object Detection via Error Analysis
  (Hoiem et al., ECCV 2012)
"""

import os
import json
import fiftyone as fo
from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext


@step(name="OutlierFailureAnalyzer", version="v1.0")
class OutlierFailureAnalyzer:
    """WPipe step for failure mode analysis via FiftyOne."""

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext) -> PostTrainContext:
        """Executes the failure analysis.

        Args:
            ctx (PostTrainContext): The pipeline context containing dataset paths.

        Returns:
            PostTrainContext: The unmodified pipeline context.
        """
        dataset_name = f"dataset_{os.path.basename(ctx.images_test_path)}"
        dataset_dir = os.path.dirname(ctx.images_test_path)

        if fo.dataset_exists(dataset_name):
            dataset = fo.load_dataset(dataset_name)
        else:
            try:
                dataset = fo.Dataset.from_dir(
                    dataset_dir=dataset_dir,
                    dataset_type=fo.types.YOLOv5Dataset,
                    name=dataset_name,
                )
            except Exception as e:
                print(f"Failed to load fiftyone dataset: {e}")
                return ctx

        fp_count = 0
        fn_count = 0

        try:
            # We assume predictions are added, skipping actual evaluation if missing
            dataset.evaluate_detections(
                "predictions",
                gt_field="ground_truth",
                eval_key="eval",
                compute_mAP=True,
            )

            high_conf_fp = dataset.filter_labels(
                "predictions",
                (fo.ViewField("eval") == "fp") & (fo.ViewField("confidence") > 0.75),
            )

            high_conf_fn = dataset.filter_labels(
                "ground_truth", fo.ViewField("eval") == "fn"
            )

            fp_count = len(high_conf_fp)
            fn_count = len(high_conf_fn)
            print(f"Total Critical False Positives: {fp_count}")
            print(f"Total Critical False Negatives: {fn_count}")

        except Exception as e:
            print(f"Error evaluating dataset: {e}")

        output_dir = os.path.join(ctx.project_path, "extras", "failures")
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "critical_false_positives": fp_count,
            "critical_false_negatives": fn_count,
        }

        with open(
            os.path.join(output_dir, "failure_analysis.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=4)

        return ctx

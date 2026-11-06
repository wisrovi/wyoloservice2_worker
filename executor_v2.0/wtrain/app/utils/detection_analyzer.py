"""Detection dataset structural, bounding box geometry, and quality analyzer."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

from .data_quality_analyzer import DataQualityAnalyzer


class DetectionAnalyzer:
    """Analyzer extracting bounding boxes, aspect ratios, and class metrics for YOLO datasets."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def analyze(
        self,
        dataset_path: Union[str, Path],
        class_names: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Compute object detection metadata and bounding box distributions.

        Args:
            dataset_path (Union[str, Path]): Base directory of the YOLO dataset.
            class_names (Optional[Dict[str, str]]): Mapping of class indices to names.

        Returns:
            Dict[str, Any]: Consolidated metrics including bbox areas, centers, and splits.
        """
        ds_path = Path(dataset_path)
        total_images = 0
        total_annotations = 0
        names = class_names or {}

        class_distribution: Dict[str, int] = {}
        split_distribution: Dict[str, Dict[str, Any]] = {}
        split_percentage_distribution: Dict[str, float] = {}

        widths: List[int] = []
        heights: List[int] = []
        aspect_ratios: List[float] = []
        bbox_areas: List[float] = []
        bbox_aspect_ratios: List[float] = []
        x_centers: List[float] = []
        y_centers: List[float] = []
        bbox_widths: List[float] = []
        bbox_heights: List[float] = []

        splits = ["train", "val", "test", "inference"]

        for split in splits:
            split_path = ds_path / split
            if not split_path.exists():
                continue

            images_split_path = ds_path / split / "images"
            labels_split_path = ds_path / split / "labels"

            split_distribution[split] = {"images": 0, "annotations": 0, "classes": {}}

            if images_split_path.exists():
                for image_file in images_split_path.rglob("*"):
                    if image_file.suffix.lower() in self.IMAGE_EXTENSIONS:
                        total_images += 1
                        split_distribution[split]["images"] += 1
                        try:
                            with Image.open(image_file) as img:
                                widths.append(img.width)
                                heights.append(img.height)
                                if img.height > 0:
                                    aspect_ratios.append(img.width / img.height)
                        except Exception:
                            pass

            if labels_split_path.exists():
                for label_file in labels_split_path.rglob("*.txt"):
                    with open(label_file, "r", encoding="utf-8") as f:
                        for line in f:
                            values = line.split()
                            if not values:
                                continue

                            total_annotations += 1
                            split_distribution[split]["annotations"] += 1

                            if len(values) >= 5:
                                try:
                                    _, x_c, y_c, b_w, b_h = map(float, values[:5])
                                    bbox_areas.append(b_w * b_h)
                                    x_centers.append(x_c)
                                    y_centers.append(y_c)
                                    bbox_widths.append(b_w)
                                    bbox_heights.append(b_h)
                                    if b_h > 0:
                                        bbox_aspect_ratios.append(b_w / b_h)
                                except ValueError:
                                    pass

                            class_id = values[0]
                            class_name = names.get(class_id, class_id)
                            split_distribution[split]["classes"][class_name] = (
                                split_distribution[split]["classes"].get(class_name, 0) + 1
                            )
                            class_distribution[class_name] = (
                                class_distribution.get(class_name, 0) + 1
                            )

        for split, data in split_distribution.items():
            if total_images > 0:
                split_percentage_distribution[split] = round(
                    data["images"] * 100 / total_images, 2
                )

        dq = DataQualityAnalyzer()
        duplicates = dq.detect_duplicates_and_overlaps(ds_path, "detection")
        corrupt, small = dq.validate_image_quality(ds_path, "detection")

        return {
            "dataset_type": "detection",
            "total_images": total_images,
            "total_annotations": total_annotations,
            "num_classes": len(class_distribution),
            "class_distribution": class_distribution,
            "split_distribution": split_distribution,
            "split_percentage_distribution": split_percentage_distribution,
            "image_dimensions": {"widths": widths, "heights": heights},
            "aspect_ratios": aspect_ratios,
            "bbox_areas": bbox_areas,
            "bbox_aspect_ratios": bbox_aspect_ratios,
            "bbox_centers": {"x": x_centers, "y": y_centers},
            "bbox_dims": {"w": bbox_widths, "h": bbox_heights},
            "data_quality": {
                "duplicates": duplicates,
                "corrupt": corrupt,
                "small": small,
            },
            "dataset_path": str(ds_path),
        }

"""Classification dataset structural and quality analyzer."""

from pathlib import Path
from typing import Any, Dict, List, Union
from PIL import Image

from .data_quality_analyzer import DataQualityAnalyzer


class ClassificationAnalyzer:
    """Analyzer computing class distributions and geometry metrics for classification datasets."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def analyze(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """Inspect directory structure and calculate classification metrics.

        Args:
            dataset_path (Union[str, Path]): Base directory of the classification dataset.

        Returns:
            Dict[str, Any]: Summary dictionary of distributions, dimensions, and data quality.
        """
        ds_path = Path(dataset_path)
        total_images = 0
        class_distribution: Dict[str, int] = {}
        split_distribution: Dict[str, Dict[str, int]] = {}
        split_percentage_distribution: Dict[str, float] = {}
        widths: List[int] = []
        heights: List[int] = []
        aspect_ratios: List[float] = []

        splits = ["train", "val", "test", "inference"]

        for split in splits:
            split_path = ds_path / split
            if not split_path.exists():
                continue
            split_distribution[split] = {}

            for class_dir in split_path.iterdir():
                if not class_dir.is_dir():
                    continue
                image_count = 0

                for file in class_dir.rglob("*"):
                    if file.is_file() and file.suffix.lower() in self.IMAGE_EXTENSIONS:
                        image_count += 1
                        try:
                            with Image.open(file) as img:
                                widths.append(img.width)
                                heights.append(img.height)
                                if img.height > 0:
                                    aspect_ratios.append(img.width / img.height)
                        except Exception:
                            pass

                split_distribution[split][class_dir.name] = image_count
                class_distribution[class_dir.name] = (
                    class_distribution.get(class_dir.name, 0) + image_count
                )
                total_images += image_count

        for split, data in split_distribution.items():
            split_images = sum(data.values())
            if total_images > 0:
                split_percentage_distribution[split] = round(
                    split_images * 100 / total_images, 2
                )

        dq = DataQualityAnalyzer()
        duplicates = dq.detect_duplicates_and_overlaps(ds_path, "classification")
        corrupt, small = dq.validate_image_quality(ds_path, "classification")

        return {
            "dataset_type": "classification",
            "num_classes": len(class_distribution),
            "total_images": total_images,
            "class_distribution": class_distribution,
            "split_distribution": split_distribution,
            "split_percentage_distribution": split_percentage_distribution,
            "image_dimensions": {"widths": widths, "heights": heights},
            "aspect_ratios": aspect_ratios,
            "data_quality": {
                "duplicates": duplicates,
                "corrupt": corrupt,
                "small": small,
            },
            "dataset_path": str(ds_path),
        }

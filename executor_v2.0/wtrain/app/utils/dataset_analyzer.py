"""Dataset analyzer facade coordinating classification, detection, and segmentation analyses."""

from pathlib import Path
from typing import Any, Dict, Union
import yaml

from .classification_analyzer import ClassificationAnalyzer
from .detection_analyzer import DetectionAnalyzer
from .eda_report_generator import EDAReportGenerator
from .segmentation_analyzer import SegmentationAnalyzer


class DatasetAnalyzer:
    """Analyze dataset structure and automatically dispatch to domain-specific analyzer."""

    def analyze(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """Analyze a dataset path, generate EDA artifacts, and return statistics.

        Args:
            dataset_path (Union[str, Path]): Base directory or YAML file of dataset.

        Returns:
            Dict[str, Any]: Consolidated dictionary of dataset metrics and characteristics.
        """
        ds_path = Path(dataset_path)
        if ds_path.is_file() and ds_path.suffix in [".yaml", ".yml"]:
            ds_path = ds_path.parent

        dataset_type = self.detect_dataset_type(ds_path)
        class_names = self.load_class_names(ds_path)

        if dataset_type == "classification":
            stats = ClassificationAnalyzer().analyze(ds_path)
        elif dataset_type == "detection":
            stats = DetectionAnalyzer().analyze(ds_path, class_names=class_names)
        elif dataset_type == "segmentation":
            stats = SegmentationAnalyzer().analyze(ds_path, class_names=class_names)
        else:
            return {"error": "Invalid dataset type"}

        try:
            EDAReportGenerator().generate_report(stats, ds_path.name, dataset_type)
        except Exception as e:
            stats["eda_report_error"] = str(e)

        return stats

    def detect_dataset_type(self, dataset_path: Union[str, Path]) -> str:
        """Infer whether dataset is for classification, detection, or segmentation.

        Args:
            dataset_path (Union[str, Path]): Path to inspect.

        Returns:
            str: Detected task type string ('classification', 'detection', 'segmentation').
        """
        ds_path = Path(dataset_path)
        if (ds_path / "train").exists():
            subdirs = [p for p in (ds_path / "train").iterdir() if p.is_dir()]
            subdir_names = {p.name for p in subdirs}
            if subdir_names and not subdir_names.intersection({"images", "labels"}):
                return "classification"

        yaml_files = list(ds_path.glob("*.yaml")) + list(ds_path.glob("*.yml"))
        if yaml_files:
            return self.detect_yolo_type(ds_path)
        return "detection"

    def detect_yolo_type(self, dataset_path: Union[str, Path]) -> str:
        """Inspect YOLO label text lines to differentiate bounding boxes from segment polygons.

        Args:
            dataset_path (Union[str, Path]): Path to root dataset directory.

        Returns:
            str: Task type ('detection' or 'segmentation').
        """
        ds_path = Path(dataset_path)
        for label_file in ds_path.rglob("labels/**/*.txt"):
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) > 5:
                            return "segmentation"
                        if len(parts) == 5:
                            return "detection"
            except Exception:
                pass
        return "detection"

    def load_class_names(self, dataset_path: Union[str, Path]) -> Dict[str, str]:
        """Load class label mappings from data.yaml if available.

        Args:
            dataset_path (Union[str, Path]): Base directory of dataset.

        Returns:
            Dict[str, str]: Mapping of stringified class IDs to human-readable names.
        """
        ds_path = Path(dataset_path)
        yaml_files = list(ds_path.glob("*.yaml")) + list(ds_path.glob("*.yml"))
        if yaml_files:
            try:
                with open(yaml_files[0], "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    names = data.get("names", {})
                    if isinstance(names, list):
                        return {str(i): name for i, name in enumerate(names)}
                    if isinstance(names, dict):
                        return {str(k): v for k, v in names.items()}
            except Exception:
                pass
        return {}


class DatasetEDAState:
    """WPipe state adapter to run dataset exploratory data analysis."""

    def execute(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """Execute comprehensive analysis of dataset.

        Args:
            dataset_path (Union[str, Path]): Filesystem location of dataset.

        Returns:
            Dict[str, Any]: Analyzed metrics dictionary.

        Raises:
            FileNotFoundError: If the specified dataset path does not exist.
        """
        ds_path = Path(dataset_path)
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset not found: {ds_path}")
        return DatasetAnalyzer().analyze(ds_path)

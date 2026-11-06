"""Segmentation dataset analyzer extending detection analyzer with polygon recognition."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .detection_analyzer import DetectionAnalyzer


class SegmentationAnalyzer(DetectionAnalyzer):
    """Analyzer computing segmentation dataset characteristics and geometry."""

    def analyze(
        self,
        dataset_path: Union[str, Path],
        class_names: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute detection base analysis and adapt task tag to segmentation.

        Args:
            dataset_path (Union[str, Path]): Base directory of dataset.
            class_names (Optional[Dict[str, str]]): Mapping of class names.

        Returns:
            Dict[str, Any]: Updated statistics dictionary marked as segmentation.
        """
        res = super().analyze(dataset_path, class_names=class_names)
        res["dataset_type"] = "segmentation"
        return res

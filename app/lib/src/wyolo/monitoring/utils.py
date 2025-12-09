"""Monitoring utilities for Wyolo."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class StatusEDA:
    """Status constants for Exploratory Data Analysis operations.

    Attributes:
        PENDING: EDA analysis is pending or in progress
        SAVED: EDA analysis has been completed and saved
    """

    PENDING = 0
    SAVED = 2


class EDAManager:
    """Exploratory Data Analysis Manager for dataset analysis and visualization.

    This class provides comprehensive dataset analysis capabilities including
    dataset information extraction, class distribution analysis, image statistics,
    and annotation analysis. Results are saved to specified output directory.

    Attributes:
        data_path: Path to the dataset directory
        output_path: Path to save analysis results
        status: Current status of EDA analysis

    Example:
        >>> eda = EDAManager("/path/to/dataset", "/path/to/output")
        >>> analysis = eda.analyze_dataset()
        >>> print(f"Analysis status: {eda.status}")
    """

    def __init__(self, data_path: str, output_path: Optional[str] = None) -> None:
        """Initialize EDA Manager.

        Args:
            data_path: Path to dataset directory
            output_path: Path to save analysis results (optional)
        """
        self.data_path = Path(data_path)
        self.output_path = (
            Path(output_path) if output_path else self.data_path / "eda_results"
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.status = StatusEDA.PENDING

    def analyze_dataset(self) -> Dict[str, Any]:
        """Perform comprehensive dataset analysis.

        Analyzes the dataset and extracts information about:
        - Basic dataset information (file counts, paths)
        - Class distribution in annotations
        - Image properties (sizes, formats)
        - Annotation statistics

        Returns:
            Dictionary containing comprehensive analysis results

        Example:
            >>> eda = EDAManager("/path/to/dataset")
            >>> results = eda.analyze_dataset()
            >>> print(f"Total images: {results['image_statistics']['total_images']}")
        """
        try:
            analysis = {
                "dataset_info": self._get_dataset_info(),
                "class_distribution": self._analyze_class_distribution(),
                "image_statistics": self._analyze_images(),
                "annotation_statistics": self._analyze_annotations(),
            }

            self._save_analysis(analysis)
            self.status = StatusEDA.SAVED
            return analysis

        except Exception as e:
            logger.error(f"EDA analysis failed: {e}")
            return {"error": str(e)}

    def _get_dataset_info(self) -> Dict[str, Any]:
        """Get basic dataset information.

        Returns:
            Dictionary with basic dataset statistics including file counts
        """
        return {
            "path": str(self.data_path),
            "total_files": len(list(self.data_path.rglob("*"))),
            "image_files": len(
                list(self.data_path.rglob("*.jpg"))
                + list(self.data_path.rglob("*.png"))
            ),
            "annotation_files": len(
                list(self.data_path.rglob("*.txt"))
                + list(self.data_path.rglob("*.json"))
            ),
        }

    def _analyze_class_distribution(self) -> Dict[str, Any]:
        """Analyze class distribution in dataset.

        Returns:
            Dictionary with class distribution statistics
        """
        # Implementation would depend on annotation format
        return {"message": "Class distribution analysis not implemented yet"}

    def _analyze_images(self) -> Dict[str, Any]:
        """Analyze image properties.

        Analyzes image files in the dataset to extract statistics about
        dimensions, formats, and other properties.

        Returns:
            Dictionary with image statistics including average dimensions
        """
        image_files = list(self.data_path.rglob("*.jpg")) + list(
            self.data_path.rglob("*.png")
        )

        if not image_files:
            return {"error": "No images found"}

        from PIL import Image

        sizes = []
        for img_path in image_files[:100]:  # Sample first 100 images
            try:
                with Image.open(img_path) as img:
                    sizes.append(img.size)
            except Exception as e:
                logger.warning(f"Could not read {img_path}: {e}")

        if sizes:
            widths, heights = zip(*sizes)
            return {
                "sample_size": len(sizes),
                "avg_width": sum(widths) / len(widths),
                "avg_height": sum(heights) / len(heights),
                "min_width": min(widths),
                "max_width": max(widths),
                "min_height": min(heights),
                "max_height": max(heights),
            }

        return {"error": "Could not analyze image sizes"}

    def _analyze_annotations(self) -> Dict[str, Any]:
        """Analyze annotation properties.

        Returns:
            Dictionary with annotation statistics
        """
        return {"message": "Annotation analysis not implemented yet"}

    def _save_analysis(self, analysis: Dict[str, Any]) -> None:
        """Save analysis results to file.

        Args:
            analysis: Analysis results dictionary
        """
        output_file = self.output_path / "analysis_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, default=str)

        logger.info(f"EDA analysis saved to {output_file}")


class ProgressManager:
    """Progress tracking manager for training processes.

    This class provides comprehensive progress tracking for training experiments
    including epoch progress, metrics tracking, and timing information.

    Attributes:
        experiment_name: Name of the experiment
        output_path: Path to save progress logs
        progress_file: Path to the progress JSON file
        progress_data: Current progress data dictionary

    Example:
        >>> progress = ProgressManager("experiment_1", "/logs")
        >>> progress.start_training(100)
        >>> progress.update_epoch(1, {"loss": 0.5, "accuracy": 0.8})
    """

    def __init__(self, experiment_name: str, output_path: Optional[str] = None) -> None:
        """Initialize Progress Manager.

        Args:
            experiment_name: Name of the experiment
            output_path: Path to save progress logs (optional)
        """
        self.experiment_name = experiment_name
        self.output_path = Path(output_path) if output_path else Path("./progress_logs")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_path / f"{experiment_name}_progress.json"
        self.progress_data = self._load_progress()

    def _load_progress(self) -> Dict[str, Any]:
        """Load existing progress data.

        Returns:
            Progress data dictionary, creating new one if not exists
        """
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")

        return {
            "experiment_name": self.experiment_name,
            "start_time": None,
            "end_time": None,
            "epochs_completed": 0,
            "total_epochs": 0,
            "metrics": [],
            "status": "initialized",
        }

    def start_training(self, total_epochs: int) -> None:
        """Record training start.

        Args:
            total_epochs: Total number of epochs for training
        """
        self.progress_data.update(
            {
                "start_time": datetime.now().isoformat(),
                "total_epochs": total_epochs,
                "status": "running",
            }
        )
        self._save_progress()

    def update_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Update progress after each epoch.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics for this epoch
        """
        self.progress_data["epochs_completed"] = epoch
        self.progress_data["metrics"].append(
            {"epoch": epoch, "timestamp": datetime.now().isoformat(), **metrics}
        )
        self._save_progress()

    def finish_training(self) -> None:
        """Record training completion."""
        self.progress_data.update(
            {"end_time": datetime.now().isoformat(), "status": "completed"}
        )
        self._save_progress()

    def _save_progress(self) -> None:
        """Save progress data to file."""
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.progress_data, f, indent=2, default=str)

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress data.

        Returns:
            Copy of current progress data dictionary
        """
        return self.progress_data.copy()

"""Plotting utilities for Exploratory Data Analysis (EDA) reports."""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_distribution(classes: List[str], counts: List[int], output_path: Path) -> None:
    """Generate and save class distribution horizontal bar chart.

    Args:
        classes (List[str]): Class label names.
        counts (List[int]): Frequency counts per class.
        output_path (Path): File path to save output image.
    """
    if not classes:
        return
    plt.figure(figsize=(12, max(6, len(classes) * 0.3)))
    sns.barplot(x=counts, y=classes, palette="viridis")
    plt.title("Class Distribution")
    plt.xlabel("Instances")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_split_distribution(splits: List[str], percentages: List[float], output_path: Path) -> None:
    """Generate and save pie chart representing dataset partition splits.

    Args:
        splits (List[str]): Split names (e.g. 'train', 'val').
        percentages (List[float]): Percentage values for each split.
        output_path (Path): File path to save output image.
    """
    if sum(percentages) <= 0:
        return
    plt.figure(figsize=(8, 8))
    plt.pie(
        percentages,
        labels=splits,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel"),
    )
    plt.title("Dataset Split Distribution")
    plt.savefig(output_path)
    plt.close()


def plot_image_dimensions(widths: List[int], heights: List[int], output_path: Path) -> None:
    """Plot overlay histogram comparing image width and height distributions.

    Args:
        widths (List[int]): List of image widths.
        heights (List[int]): List of image heights.
        output_path (Path): File path to save output image.
    """
    if not widths or not heights:
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(widths, bins=30, color="blue", label="Width", alpha=0.6)
    sns.histplot(heights, bins=30, color="red", label="Height", alpha=0.6)
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.title("Image Size Distribution")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def plot_aspect_ratios(ratios: List[float], output_path: Path) -> None:
    """Plot histogram of image width-to-height aspect ratios.

    Args:
        ratios (List[float]): Calculated aspect ratio values.
        output_path (Path): File path to save output image.
    """
    if not ratios:
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(ratios, bins=30, color="purple")
    plt.xlabel("Aspect Ratio (Width / Height)")
    plt.ylabel("Frequency")
    plt.title("Image Aspect Ratio Distribution")
    plt.savefig(output_path)
    plt.close()


def plot_bbox_geometry(
    bbox_areas: List[float],
    bbox_aspect_ratios: List[float],
    centers: Dict[str, List[float]],
    dims: Dict[str, List[float]],
    plots_dir: Path,
) -> None:
    """Generate and save bounding box geometric distribution figures.

    Args:
        bbox_areas (List[float]): List of normalized box areas.
        bbox_aspect_ratios (List[float]): Bounding box aspect ratios (W/H).
        centers (Dict[str, List[float]]): Dict containing 'x' and 'y' center points.
        dims (Dict[str, List[float]]): Dict containing 'w' and 'h' dimensions.
        plots_dir (Path): Destination folder for output figures.
    """
    if bbox_areas:
        plt.figure(figsize=(10, 6))
        sns.histplot(bbox_areas, bins=50, kde=True, color="green")
        plt.title("Bounding Box Area Distribution (Normalized)")
        plt.xlabel("Area (Width * Height)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(plots_dir / "bbox_areas.png")
        plt.close()

    if bbox_aspect_ratios:
        plt.figure(figsize=(10, 6))
        sns.histplot(bbox_aspect_ratios, bins=30, color="orange")
        plt.title("Bounding Box Aspect Ratio (W/H)")
        plt.xlabel("Aspect Ratio")
        plt.savefig(plots_dir / "bbox_aspect_ratios.png")
        plt.close()

    if centers.get("x") and centers.get("y"):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(centers["x"], bins=30, color="purple")
        plt.title("Center X Positions")
        plt.subplot(1, 2, 2)
        sns.histplot(centers["y"], bins=30, color="blue")
        plt.title("Center Y Positions")
        plt.savefig(plots_dir / "bbox_centers.png")
        plt.close()

    if dims.get("w") and dims.get("h"):
        plt.figure(figsize=(8, 8))
        plt.scatter(dims["w"], dims["h"], alpha=0.3)
        plt.title("Bounding Box Width vs Height")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.savefig(plots_dir / "bbox_scatter.png")
        plt.close()

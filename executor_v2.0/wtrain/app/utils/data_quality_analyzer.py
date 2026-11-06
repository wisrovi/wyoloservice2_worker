"""Data quality analyzer for dataset images, corruption detection, and duplicate checks."""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2


class DataQualityAnalyzer:
    """Analyzer detecting image corruption, duplicates, and minimum dimension constraints."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def get_all_images(self, folder: str, split_name: str) -> List[Tuple[str, str]]:
        """Collect all image file paths with associated dataset split tag.

        Args:
            folder (str): Directory path to search recursively.
            split_name (str): Label of dataset partition (e.g. 'train', 'val').

        Returns:
            List[Tuple[str, str]]: Tuples of (image_file_path, split_name).
        """
        return [
            (os.path.join(root, f), split_name)
            for root, _, files in os.walk(folder)
            for f in files
            if f.lower().endswith(tuple(self.IMAGE_EXTENSIONS))
        ]

    def detect_duplicates_and_overlaps(
        self, dataset_path: Path, dataset_type: str
    ) -> List[Dict[str, str]]:
        """Detect duplicate image content across splits using SHA-1 pixel hashing.

        Args:
            dataset_path (Path): Path to dataset root directory.
            dataset_type (str): Type identifier ('classification' or object detection).

        Returns:
            List[Dict[str, str]]: List of detected duplicate pair records.
        """
        image_hashes: Dict[str, Tuple[str, str]] = {}
        duplicates: List[Dict[str, str]] = []
        image_paths: List[Tuple[str, str]] = []

        target_splits = ["train", "test", "val"]
        for split_folder in target_splits:
            if dataset_type == "classification":
                split_path = dataset_path / split_folder
            else:
                split_path = dataset_path / split_folder / "images"

            if split_path.is_dir():
                image_paths.extend(self.get_all_images(str(split_path), split_folder))

        for img_path, split in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                img_hash = hashlib.sha1(img).hexdigest()
                if img_hash in image_hashes:
                    duplicates.append(
                        {
                            "img1": os.path.basename(img_path),
                            "split1": split,
                            "img2": os.path.basename(image_hashes[img_hash][0]),
                            "split2": image_hashes[img_hash][1],
                        }
                    )
                else:
                    image_hashes[img_hash] = (img_path, split)

        return duplicates

    def validate_image_quality(
        self, dataset_path: Path, dataset_type: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Verify image decodability and check for sub-minimum dimensions (<64px).

        Args:
            dataset_path (Path): Path to dataset root.
            dataset_type (str): Type of dataset ('classification', 'detection', etc.).

        Returns:
            Tuple[List[str], List[Dict[str, Any]]]: Lists of corrupt and small images.
        """
        corrupt_images: List[str] = []
        small_images: List[Dict[str, Any]] = []
        image_paths: List[str] = []

        target_splits = ["train", "test", "val"]
        for split_folder in target_splits:
            if dataset_type == "classification":
                split_path = dataset_path / split_folder
            else:
                split_path = dataset_path / split_folder / "images"

            if split_path.is_dir():
                image_paths.extend(
                    [p[0] for p in self.get_all_images(str(split_path), split_folder)]
                )

        for image_path in image_paths:
            img_name = os.path.basename(image_path)
            try:
                img = cv2.imread(image_path)
                if img is None:
                    corrupt_images.append(img_name)
                else:
                    height, width, _ = img.shape
                    if width < 64 or height < 64:
                        small_images.append({"name": img_name, "w": width, "h": height})
            except Exception:
                corrupt_images.append(img_name)

        return corrupt_images, small_images

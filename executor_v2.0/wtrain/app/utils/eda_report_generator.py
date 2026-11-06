"""Automated Exploratory Data Analysis (EDA) report generator producing Markdown and DOCX."""

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any, Dict, List
from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.shared import Inches

from .eda_plotter import (
    plot_aspect_ratios,
    plot_bbox_geometry,
    plot_class_distribution,
    plot_image_dimensions,
    plot_split_distribution,
)


class EDAReportGenerator:
    """Compiles dataset analysis statistics into structured docx and markdown reports."""

    def __init__(self, results_dir: str = "/wyolo/worker/train_service_results/extras/eda") -> None:
        """Initialize directory paths and ensure clean export targets.

        Args:
            results_dir (str): Base destination directory for generated artifacts.
        """
        self.results_dir = Path(results_dir)
        if self.results_dir.exists():
            shutil.rmtree(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def generate_report(self, stats: Dict[str, Any], dataset_name: str, dataset_type: str) -> str:
        """Render complete exploratory data analysis report in Markdown and Word DOCX formats.

        Args:
            stats (Dict[str, Any]): Aggregated statistics from dataset analyzers.
            dataset_name (str): Label or identifier of analyzed dataset.
            dataset_type (str): Task domain string (e.g. 'detection', 'classification').

        Returns:
            str: Directory path containing the generated report files.
        """
        md_content = f"# EDA Report: {dataset_name}\n\n"
        doc = Document()

        doc.add_paragraph("Exploratory Data Analysis (EDA)", style="Title").alignment = (
            WD_PARAGRAPH_ALIGNMENT.CENTER
        )
        doc.add_paragraph(f"Dataset: {dataset_name} | Type: {dataset_type.capitalize()}")
        doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        doc.add_page_break()

        # 1. Data Quality Section
        dq = stats.get("data_quality", {})
        corrupt = dq.get("corrupt", [])
        small = dq.get("small", [])
        dups = dq.get("duplicates", [])

        doc.add_heading("1. Image Quality Validation", level=1)
        md_content += "## 1. Image Quality Validation\n\n"

        if corrupt:
            doc.add_paragraph(f"Found {len(corrupt)} corrupt images.")
            md_content += f"- Found {len(corrupt)} corrupt images.\n"
        else:
            doc.add_paragraph("No corrupt images detected.")
            md_content += "- No corrupt images detected.\n"

        if small:
            doc.add_paragraph(f"Found {len(small)} undersized images (<64px).")
            md_content += f"- Found {len(small)} undersized images (<64px).\n"
        else:
            doc.add_paragraph("No undersized images detected.")
            md_content += "- No undersized images detected.\n"

        if dups:
            doc.add_paragraph(f"Found {len(dups)} duplicate image pairs.")
            md_content += f"- Found {len(dups)} duplicate image pairs.\n"
        else:
            doc.add_paragraph("No duplicate images detected across splits.")
            md_content += "- No duplicate images detected across splits.\n"

        # 2. Visual Distributions & Plots
        if "class_distribution" in stats and stats["class_distribution"]:
            classes = list(stats["class_distribution"].keys())
            counts = list(stats["class_distribution"].values())
            c_plot = self.plots_dir / "class_distribution.png"
            plot_class_distribution(classes, counts, c_plot)
            if c_plot.exists():
                doc.add_heading("2. Class Distribution", level=1)
                doc.add_picture(str(c_plot), width=Inches(6.0))
                md_content += "\n## 2. Class Distribution\n\n![Class Distribution](plots/class_distribution.png)\n"

        if "split_percentage_distribution" in stats and stats["split_percentage_distribution"]:
            splits = list(stats["split_percentage_distribution"].keys())
            percentages = list(stats["split_percentage_distribution"].values())
            s_plot = self.plots_dir / "split_distribution.png"
            plot_split_distribution(splits, percentages, s_plot)
            if s_plot.exists():
                doc.add_heading("3. Dataset Splits", level=1)
                doc.add_picture(str(s_plot), width=Inches(5.0))
                md_content += "\n## 3. Dataset Splits\n\n![Splits](plots/split_distribution.png)\n"

        dims = stats.get("image_dimensions", {})
        if dims.get("widths") and dims.get("heights"):
            dim_plot = self.plots_dir / "image_size_distribution.png"
            plot_image_dimensions(dims["widths"], dims["heights"], dim_plot)
            if dim_plot.exists():
                doc.add_heading("4. Image Dimensions", level=1)
                doc.add_picture(str(dim_plot), width=Inches(6.0))
                md_content += "\n## 4. Image Dimensions\n\n![Sizes](plots/image_size_distribution.png)\n"

        # 3. Bounding Box Geometry for Detection
        if dataset_type in ["detection", "segmentation"]:
            plot_bbox_geometry(
                stats.get("bbox_areas", []),
                stats.get("bbox_aspect_ratios", []),
                stats.get("bbox_centers", {}),
                stats.get("bbox_dims", {}),
                self.plots_dir,
            )
            doc.add_heading("5. Bounding Box Geometry", level=1)
            md_content += "\n## 5. Bounding Box Geometry\n\n"
            for geom_name in ["bbox_areas.png", "bbox_aspect_ratios.png", "bbox_centers.png", "bbox_scatter.png"]:
                p_file = self.plots_dir / geom_name
                if p_file.exists():
                    doc.add_picture(str(p_file), width=Inches(5.5))
                    md_content += f"![{geom_name}](plots/{geom_name})\n"

        # 4. Summary Conclusions
        doc.add_heading("6. Analysis Summary", level=1)
        md_content += "\n## 6. Analysis Summary\n\n"
        conclusions = [
            f"Total images analyzed: {stats.get('total_images', 0)}",
            f"Total classes identified: {stats.get('num_classes', 0)}",
            f"Dataset task type: {dataset_type}",
        ]
        for item in conclusions:
            doc.add_paragraph(item, style="List Bullet")
            md_content += f"- {item}\n"

        # Save files
        with open(self.results_dir / "EDA_Report.md", "w", encoding="utf-8") as f:
            f.write(md_content)
        doc.save(str(self.results_dir / "EDA_Report.docx"))

        return str(self.results_dir)

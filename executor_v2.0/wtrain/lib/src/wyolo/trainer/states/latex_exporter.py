"""Exporter that generates LaTeX tables from model metrics.

This module provides the LatexExporter WPipe state, which takes the hardware
and performance metrics computed during the pipeline and exports them into
a fully formatted LaTeX table (.tex) suitable for R&D publications.
"""

import os
from pylatex import Document, Table, Tabular, NoEscape
from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext


@step(name="LatexExporter", version="v1.0")
class LatexExporter:
    """WPipe step for exporting metrics to LaTeX."""

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext) -> PostTrainContext:
        """Executes the LaTeX export.

        Args:
            ctx (PostTrainContext): The pipeline context containing metrics.

        Returns:
            PostTrainContext: The unmodified pipeline context.
        """
        metrics_dict: dict = getattr(ctx, "model_metrics", {})
        if not metrics_dict:
            return ctx

        output_dir = os.path.join(ctx.project_path, "extras", "paper_table_results")
        os.makedirs(output_dir, exist_ok=True)
        output_tex_path = os.path.join(output_dir, "comparative_metrics")

        doc = Document(default_filepath=output_tex_path)
        with doc.create(Table(position="htbp")) as table:
            table.append(NoEscape(r"\centering"))
            table.append(
                NoEscape(
                    r"\caption{Comparative Performance and Hardware Metrics for Proposed YOLO Pipeline.}"
                )
            )
            table.append(NoEscape(r"\label{tab:model_performance}"))

            tabular = Tabular("l c c c c")
            tabular.add_row(NoEscape(r"\toprule"))
            tabular.add_row(
                [
                    NoEscape(r"\textbf{Model}"),
                    NoEscape(r"\textbf{mAP$_{50}$ (\%)}"),
                    NoEscape(r"\textbf{GFLOPs}"),
                    NoEscape(r"\textbf{Params (M)}"),
                    NoEscape(r"\textbf{Latency (ms)}"),
                ]
            )
            tabular.add_row(NoEscape(r"\midrule"))

            for model_name, metrics in metrics_dict.items():
                map_str = f"{metrics.get('map50', 0):.1f} $\\pm$ {metrics.get('map50_std', 0):.1f}"
                tabular.add_row(
                    [
                        model_name,
                        NoEscape(map_str),
                        f"{metrics.get('gflops', 0):.2f}",
                        f"{metrics.get('params', 0):.2f}",
                        f"{metrics.get('latency', 0):.2f}",
                    ]
                )

            tabular.add_row(NoEscape(r"\bottomrule"))
            table.append(tabular)

        doc.generate_tex()
        return ctx

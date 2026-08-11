import inspect
"""Exporter that generates LaTeX tables from model metrics.

This module provides the LatexExporter WPipe state, which takes
hardware and performance metrics computed during the pipeline
and exports them into a fully formatted LaTeX table (.tex)
suitable for R&D publications.

Presentación de Métricas en Tablas Científicas (LatexExporter)
Paper: Guidelines for Presenting Quantitative Results in Machine Learning Research
Autores: Rafael S. Calsaverini et al.
Por qué es el referente: Recopila las buenas prácticas editoriales exigidas por
    editoriales científicas de alto impacto (IEEE, ACM, Elsevier) para la estructuración de tablas.
    Exige la inclusión de la desviación estándar ($\mu \pm \sigma$), el marcado en negrita de los
    resultados estadísticamente superiores, y la separación clara entre métricas de precisión ($mAP$)
    y de rendimiento de hardware (GFLOPs, parámetros y tiempo de inferencia).
"""

import os
from pylatex import Document, Table, Tabular, NoEscape
from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext


@step(name="LatexExporter", version="v1.0")
class LatexExporter:
    """WPipe step for exporting metrics to LaTeX format."""

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext) -> PostTrainContext:
        """Executes the LaTeX export.

        Args:
            ctx (PostTrainContext): Pipeline context containing metrics.

        Returns:
            PostTrainContext: The unmodified pipeline context.
        """
        metrics_dict: dict = getattr(ctx, "model_metrics", {})
        if not metrics_dict:
            return ctx

        output_dir = os.path.join(ctx.project_path, "extras", "paper_table_results")
        os.makedirs(output_dir, exist_ok=True)

        import inspect
        import sys
        md_content = inspect.cleandoc(sys.modules[__name__].__doc__ or "No description available.")
        md_content = f"""# Analysis Report

{md_content}

## Methodology
This directory contains the outputs and results of this specific analysis. The metrics and plots generated here reflect the model's behavior according to the described methodology."""
        if md_content:
            with open(os.path.join(output_dir, "DESCRIPTION.md"), "w", encoding="utf-8") as fmd:
                fmd.write(md_content)

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
        print(f"LaTeX table successfully exported to: {output_tex_path}.tex")
        return ctx

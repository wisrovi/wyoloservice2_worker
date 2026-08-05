import os

from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext
from ..utils.training_report_analyzer import TrainingReportAnalyzer


@step(name="llm_analyzer", version="v1.0")
class LlmAnalyzer:

    RESULTS_RELATIVE = "evaluation_metrics/results.csv"
    LLM_MD_NAME = "llm.md"

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext):
        project_path = ctx.project_path

        results_file = os.path.join(project_path, self.RESULTS_RELATIVE)
        llm_md_path = os.path.join(project_path, self.LLM_MD_NAME)

        try:
            report = TrainingReportAnalyzer().analyze(results_file)
            with open(llm_md_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(
                f"[LLMAnalyzer] Report written to {llm_md_path} "
                f"({len(report)} chars)."
            )
            return {"llm_report": report, "llm_md_path": llm_md_path}
        except Exception as exc:
            print(f"[LLMAnalyzer] Failed: {exc}")
            return {"llm_report": "", "llm_md_path": "", "error": str(exc)}

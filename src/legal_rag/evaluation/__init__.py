"""Benchmark validation, checkpointed evaluation and paired comparisons."""

from .comparison import compare_runs
from .dataset import load_benchmark, parse_references, write_review_proposals
from .diagnostics import DiagnosticManifest, DiagnosticResult, load_diagnostics, run_retrieval_diagnostics, summarize_diagnostics
from .models import Benchmark, EvaluationManifest, EvaluationResult, JudgePayload, OracleEvidence, Question, Review
from .runner import load_results, run_evaluation, run_rejudge, select_control_qids, summarize

__all__ = ["Benchmark", "DiagnosticManifest", "DiagnosticResult", "EvaluationManifest", "EvaluationResult", "JudgePayload", "OracleEvidence", "Question", "Review", "compare_runs",
           "load_benchmark", "load_results", "parse_references", "run_evaluation", "run_rejudge", "select_control_qids", "summarize", "write_review_proposals"]
__all__ += ["load_diagnostics", "run_retrieval_diagnostics", "summarize_diagnostics"]

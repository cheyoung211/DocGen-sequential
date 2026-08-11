"""Metric 6: Compile Success Rate -- a final system-reliability metric, not a
verifier-specific one. compile_success_rate = successfully_compiled / generation_runs.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from src.evaluation.schemas import CompileResult


def compute_compile_success_rate(compile_results: Sequence[Optional[CompileResult]]) -> dict:
    """``compile_results`` should have one entry per generation run
    attempted -- ``None`` for a run that never reached the compile stage at
    all (a content_generation_error/integration_error): still counted in
    ``generation_runs`` (it's still a run that didn't produce a PDF) but not
    in ``runs_that_reached_compile``.

    ``compile_success`` is kept independent from ``fatal_error_count``/
    ``warning_count`` throughout, per the spec's explicit caution: a
    warnings-only compile can still be ``compile_success=True``.
    """
    generation_runs = len(compile_results)
    present = [r for r in compile_results if r is not None]
    successfully_compiled = sum(1 for r in present if r.compile_success)
    runs_with_fatal_errors = sum(1 for r in present if r.fatal_error_count > 0)
    runs_with_warnings_only = sum(
        1 for r in present if r.compile_success and r.warning_count > 0 and r.fatal_error_count == 0
    )

    by_first_error_type: Dict[str, int] = {}
    for result in present:
        if result.compile_success or not result.first_error_type:
            continue
        by_first_error_type[result.first_error_type] = (
            by_first_error_type.get(result.first_error_type, 0) + 1
        )

    return {
        "compile_success_rate": (successfully_compiled / generation_runs) if generation_runs else None,
        "successfully_compiled": successfully_compiled,
        "generation_runs": generation_runs,
        "runs_that_reached_compile": len(present),
        "runs_with_fatal_errors": runs_with_fatal_errors,
        "runs_with_warnings_only": runs_with_warnings_only,
        "by_first_error_type": by_first_error_type,
    }

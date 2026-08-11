"""Foundation evaluation-layer batch aggregator.

Walks a pipeline output directory (the shape ``scripts/run_full_pipeline.py
--dataset`` produces) and computes the 4 Foundation-pass metrics --
Verification Trigger Rate, Recovery Success Rate, Compile Success Rate, and
(hard-category) Contract Satisfaction Rate -- across every sample.

Both a library entry point (``aggregate_batch``) and a directly runnable
CLI, matching the existing 3 offline evaluators' own convention
(``discover_sample_dirs`` -> per-sample evaluate -> one combined JSON,
written under a repo-root ``evaluation/`` directory, sibling to
``evaluation/latex_robustness``/``long_form_quality``/``multimodal_alignment``):

    python3 src/evaluation/aggregate.py --dir outputs/<batch_dir> \\
        [--benchmark dataset2/benchmark/benchmark_v2.jsonl] [--out foundation_metrics_eval.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow `python3 src/evaluation/aggregate.py ...` (direct script invocation,
# matching scripts/run_full_pipeline.py's own pattern) in addition to
# `python3 -m src.evaluation.aggregate ...` -- without this, Python's default
# sys.path[0] is this file's own directory, not the repo root, so the
# `src.*` imports below fail.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.dataset.schemas import BenchmarkItem
from src.evaluation.compile_log_parser import parse_compile_log
from src.evaluation.discovery import discover_sample_dirs, split_factor_sample
from src.evaluation.metrics.compile_success import compute_compile_success_rate
from src.evaluation.metrics.contract_satisfaction import (
    aggregate_contract_satisfaction,
    evaluate_contract,
)
from src.evaluation.metrics.recovery_success import compute_recovery_success_rate
from src.evaluation.metrics.trigger_rate import compute_trigger_rate
from src.evaluation.schemas import CompileResult, ContractResult, RunResult


def _load_benchmark_items(benchmark_path: Optional[str]) -> Dict[str, BenchmarkItem]:
    if not benchmark_path:
        return {}
    items: Dict[str, BenchmarkItem] = {}
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = BenchmarkItem.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
            items[item.sample_id] = item
    return items


def _load_run_result(sample_dir: Path) -> Optional[RunResult]:
    path = sample_dir / "run_result.json"
    if not path.is_file():
        return None
    try:
        return RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None


def _fallback_compile_result(sample_dir: Path) -> Optional[CompileResult]:
    """Best-effort reconstruction for a run that predates this
    instrumentation (or whose run_result.json is otherwise missing
    compile_result). Raw verification/recovery events genuinely can't be
    recovered this way -- nothing was logged -- but compile success can be
    re-derived from files the pipeline has always written
    (compile.log, main.pdf), so trigger/recovery rate report a zero
    denominator for such samples while compile success rate still counts
    them correctly.
    """
    compile_result_path = sample_dir / "compile_result.json"
    if compile_result_path.is_file():
        try:
            return CompileResult.model_validate_json(
                compile_result_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, ValueError):
            pass
    log_path = sample_dir / "compile.log"
    if not log_path.is_file():
        return None
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    pdf_exists = (sample_dir / "main.pdf").exists()
    fatal, warnings, first_error_type = parse_compile_log(log_text, "tectonic")
    return CompileResult(
        compile_success=pdf_exists,
        engine="tectonic (assumed; predates compile_result.json)",
        fatal_error_count=fatal,
        warning_count=warnings,
        first_error_type=first_error_type,
        pdf_exists=pdf_exists,
    )


def aggregate_batch(
    base_dir: Path, benchmark_items: Optional[Dict[str, BenchmarkItem]] = None
) -> dict:
    """Compute Metrics 1, 2, 6, 7 (Foundation-pass scope) across every sample
    discovered under ``base_dir``."""
    benchmark_items = benchmark_items or {}
    sample_dirs = discover_sample_dirs(base_dir)

    all_verification_events = []
    all_recovery_events = []
    compile_results: List[Optional[CompileResult]] = []
    contract_results: List[ContractResult] = []
    per_sample: List[dict] = []

    for sample_dir in sample_dirs:
        relative_path = sample_dir.relative_to(base_dir)
        factor, sample_id = split_factor_sample(relative_path)
        run_result = _load_run_result(sample_dir)

        if run_result is not None:
            all_verification_events.extend(run_result.verification_events)
            all_recovery_events.extend(run_result.recovery_events)
            compile_result = run_result.compile_result or _fallback_compile_result(sample_dir)
        else:
            compile_result = _fallback_compile_result(sample_dir)
        compile_results.append(compile_result)

        benchmark_item = benchmark_items.get(sample_id)
        if benchmark_item is not None:
            # Always recompute from what's on disk rather than trusting a
            # value already embedded in run_result.json: evaluate_contract
            # is a pure, cheap, local read (no LLM/network call), and a
            # stored contract_result may have been produced by an older
            # version of the matching/scoring logic -- re-running this
            # aggregator (--eval-only) after an evaluation-layer code change
            # is an explicitly supported workflow (see run_evaluation.sh),
            # which only actually works if this always reflects current code.
            contract_result: Optional[ContractResult] = evaluate_contract(sample_dir, benchmark_item)
        elif run_result is not None and run_result.contract_result is not None:
            # No benchmark item resolvable (e.g. --benchmark omitted, or a
            # legacy/unmatched sample_id) -- fall back to whatever was
            # embedded at generation time rather than reporting nothing.
            contract_result = run_result.contract_result
        else:
            contract_result = None
        if contract_result is not None:
            contract_results.append(contract_result)

        per_sample.append({
            "factor": factor,
            "sample_id": sample_id,
            "rel_path": str(relative_path),
            "has_run_result": run_result is not None,
            "compile_success": compile_result.compile_success if compile_result else None,
            "contract_overall": contract_result.overall if contract_result else None,
        })

    return {
        "sample_count": len(sample_dirs),
        "verification_trigger_rate": compute_trigger_rate(all_verification_events),
        "recovery_success_rate": compute_recovery_success_rate(
            all_verification_events, all_recovery_events
        ),
        "compile_success_rate": compute_compile_success_rate(compile_results),
        "contract_satisfaction_rate": aggregate_contract_satisfaction(contract_results),
        "per_sample": per_sample,
    }


def run_batch_evaluation(args: argparse.Namespace) -> dict:
    base_dir = Path(args.dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {base_dir}")

    benchmark_items = _load_benchmark_items(args.benchmark)
    summary = aggregate_batch(base_dir, benchmark_items)

    out_dir = Path.cwd() / "evaluation" / "foundation_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_dir.name}_{Path(args.out).name}"
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print(f"[Eval] {summary['sample_count']} sample(s) discovered under {base_dir}")
    print(f"[Eval] Wrote summary to {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Foundation evaluation layer: Verification Trigger Rate, Recovery "
        "Success Rate, Compile Success Rate, Contract Satisfaction Rate."
    )
    parser.add_argument("--dir", required=True, help="Batch output directory to evaluate.")
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark JSONL to resolve BenchmarkItems by sample_id (enables Contract "
        "Satisfaction Rate and the pre-instrumentation-run fallback).",
    )
    parser.add_argument("--out", default="foundation_metrics_eval.json", help="Output filename.")
    args = parser.parse_args()
    run_batch_evaluation(args)


if __name__ == "__main__":
    main()

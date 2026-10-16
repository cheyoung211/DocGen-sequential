"""Paper-table generator: turns one or more pipeline output directories
(the ``outputs/<batch_dir>`` shape ``scripts/run_full_pipeline.py`` and the
Foundation evaluation layer already understand -- see ``aggregate.py``)
into the two tables the write-up tracks across experiments:

- **Table 4** -- compile success rate and overall contract satisfaction,
  broken down by *trajectory group* (A: no recorded failure, B: resolved
  failures, C: unresolved failures, D: failures without verifier-specific
  recovery, U: run_result.json missing/unreadable so failure status is
  unknown -- never folded into A), compared across verifier conditions
  (e.g. "All verifiers" vs. "No verifier"). Every group's sample count,
  compile rate, and contract mean is always reported as real numbers -- no
  group is ever hidden or shown as N/A regardless of condition.
- **Table 5** -- document-level contract-satisfaction quality (mean +-
  sd), generation robustness (compile-reached / compile-success rates),
  and generation cost (LLM calls, tokens), compared across the same
  conditions, with a relative-change column.

Each condition is a *label -> batch directory* pair supplied by the
caller (e.g. ``--condition "All verifiers=outputs/qwen_all_verifiers_..."``)
-- this script never tries to infer "which condition is this batch" from
data on disk, because ``RunResult.verification_config``/``config_name``
does not reliably reflect that (see ``schemas.py``'s own note that no real
per-run toggle system exists yet: every run today stamps the same
``config_name`` regardless of which verifiers were actually active, so the
batch *directory* -- i.e. which generation run you point this at -- is the
only trustworthy signal of condition).

Library entry points: ``load_condition``, ``build_table4``,
``build_table5``, and the ``render_table4_*``/``render_table5_*``
functions. Also directly runnable as a CLI -- see ``main()``.
"""

from __future__ import annotations
import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.dataset.schemas import BenchmarkItem  # noqa: E402
from src.evaluation.aggregate import (  # noqa: E402
    _fallback_compile_result,
    _load_benchmark_items,
    _load_run_result,
)
from src.evaluation.discovery import discover_sample_dirs, split_factor_sample  # noqa: E402
from src.evaluation.metrics.compile_success import compute_compile_success_rate  # noqa: E402
from src.evaluation.metrics.contract_satisfaction import (  # noqa: E402
    aggregate_contract_satisfaction,
    evaluate_contract,
)
from src.evaluation.metrics.recovery_success import compute_recovery_success_rate  # noqa: E402
from src.evaluation.schemas import CompileResult, ContractResult, RunResult  # noqa: E402


# ---------------------------------------------------------------------------
# Trajectory-group classification (Table 4)
# ---------------------------------------------------------------------------


class TrajectoryGroup(str, Enum):
    A_NO_FAILURE = "A"
    B_RESOLVED = "B"
    C_UNRESOLVED = "C"
    D_NO_RECOVERY_PATH = "D"
    #: Not one of the paper's A--D groups -- a trajectory whose
    #: run_result.json is missing or failed to parse (see
    #: ``aggregate._load_run_result``'s ``None`` return). There is *no
    #: evidence either way* about whether this trajectory recorded a
    #: failure, so it must never be folded into A ("no recorded failure" is
    #: a positive claim that the record was checked and was clean, not the
    #: absence of a record). Kept as its own row in every rendered table so
    #: these trajectories stay visible instead of silently vanishing from
    #: the A--D breakdown or silently inflating A.
    U_UNKNOWN = "U"


#: Best-available proxy for "this recovery came from generic JSON-schema
#: parsing/retry, not a named domain verifier". The pipeline logs a
#: verification_event's ``signal_type`` for every failure; the domain
#: verifiers (``planner_blueprint_validator``, ``text_semantic_block_validator``,
#: ``composer_semantic_block_validator``, ``layout_policy_validator``,
#: ``figure_*_validator``, ``reference_validator``, ``document_frame_validator``,
#: ``latex_compiler``) emit their own specific signal_types
#: (``blueprint_validation_error``, ``content_validation_error``,
#: ``missing_math_delimiter``, ``latex_error``, ...); ``schema_validation_error``
#: is the one signal_type that fires from the generic Pydantic/JSON response
#: parsing layer shared by every LLM call, independent of whether any
#: domain verifier is enabled. If your instrumentation uses a different
#: signal_type (or verifier name) for that generic layer, update this set --
#: it is the single place the "generic JSON-schema, not verifier-induced"
#: distinction from Table 4's caption is encoded.
GENERIC_SCHEMA_SIGNAL_TYPES = frozenset({"schema_validation_error"})


@dataclass
class TrajectoryClassification:
    group: TrajectoryGroup
    #: True only when this trajectory's classification-defining failures
    #: (the ones with a failure_id, i.e. groups B/C) are *all* generic
    #: JSON-schema failures rather than a named domain verifier catching
    #: something. Meaningless (always False) for groups A/D.
    generic_schema_only: bool


def classify_trajectory(run_result: Optional[RunResult]) -> TrajectoryClassification:
    """One trajectory (= one generated document / run_result.json) -> its
    Table 4 group.

    0. **``run_result`` is ``None``** (file missing, or present but failed to
       parse -- see ``aggregate._load_run_result``) -> Group U ("unknown").
       There is no verification_events trail to read at all here, so
       whether a failure occurred is genuinely unknown -- this must not be
       reported as Group A, which specifically means "checked, and clean".
    1. **No verification_events recorded a failure** -> Group A ("no
       recorded failure").
    2. **At least one failure has no ``failure_id``** -> Group D ("failures
       without verifier-specific recovery"). ``failure_id`` is only ever set
       by stages that *have* a recovery mechanism (see
       ``VerificationEvent.failure_id``'s docstring in ``schemas.py``) --
       a failure with none structurally could not have been recovered via
       the verifier/recovery framework, regardless of what happened to any
       other failure in the same trajectory. Checked before B/C so a
       trajectory with a mix of recoverable and unrecoverable failures is
       still flagged as having hit the "no recovery path" case, rather than
       having that fact silently absorbed into a B/C average.
    3. **Every failure that does have a ``failure_id`` was successfully
       recovered** -> Group B ("resolved failures").
    4. **Otherwise** (at least one failure with a ``failure_id`` was never
       attempted, or was attempted and failed) -> Group C ("unresolved
       failures").
    """
    if run_result is None:
        return TrajectoryClassification(TrajectoryGroup.U_UNKNOWN, False)

    failures = [e for e in run_result.verification_events if e.result == "fail"]
    if not failures:
        return TrajectoryClassification(TrajectoryGroup.A_NO_FAILURE, False)

    without_failure_id = [f for f in failures if f.failure_id is None]
    with_failure_id = [f for f in failures if f.failure_id is not None]

    if without_failure_id:
        return TrajectoryClassification(TrajectoryGroup.D_NO_RECOVERY_PATH, False)

    recovery_by_failure_id = {r.failure_id: r for r in run_result.recovery_events}
    all_resolved = all(
        (recovery_by_failure_id.get(f.failure_id) is not None)
        and recovery_by_failure_id[f.failure_id].success
        for f in with_failure_id
    )
    generic_only = all(f.signal_type in GENERIC_SCHEMA_SIGNAL_TYPES for f in with_failure_id)

    group = TrajectoryGroup.B_RESOLVED if all_resolved else TrajectoryGroup.C_UNRESOLVED
    return TrajectoryClassification(group, generic_only)


# ---------------------------------------------------------------------------
# Loading one condition's samples
# ---------------------------------------------------------------------------


@dataclass
class SampleRecord:
    rel_path: str
    run_result: Optional[RunResult]
    compile_result: Optional[CompileResult]
    contract_result: Optional[ContractResult]
    classification: TrajectoryClassification


@dataclass
class Condition:
    label: str
    batch_dir: Path
    samples: List[SampleRecord] = field(default_factory=list)


def load_condition(
    label: str,
    batch_dir: Path,
    benchmark_items: Optional[Dict[str, BenchmarkItem]] = None,
) -> Condition:
    """Mirrors ``aggregate.aggregate_batch``'s per-sample loading (same
    fallback rules for a missing/legacy run_result.json) but keeps the full
    ``RunResult`` per sample, which Table 4's trajectory classification and
    Table 5's token/call-count rows both need and the existing aggregator's
    ``per_sample`` summary does not retain.
    """
    benchmark_items = benchmark_items or {}
    condition = Condition(label=label, batch_dir=batch_dir)

    for sample_dir in discover_sample_dirs(batch_dir):
        relative_path = sample_dir.relative_to(batch_dir)
        _factor, sample_id = split_factor_sample(relative_path)
        run_result = _load_run_result(sample_dir)

        compile_result = (
            run_result.compile_result if run_result and run_result.compile_result
            else _fallback_compile_result(sample_dir)
        )

        benchmark_item = benchmark_items.get(sample_id)
        if benchmark_item is not None:
            contract_result = evaluate_contract(sample_dir, benchmark_item)
        elif run_result is not None and run_result.contract_result is not None:
            contract_result = run_result.contract_result
        else:
            contract_result = None

        condition.samples.append(SampleRecord(
            rel_path=str(relative_path),
            run_result=run_result,
            compile_result=compile_result,
            contract_result=contract_result,
            classification=classify_trajectory(run_result),
        ))

    return condition


# ---------------------------------------------------------------------------
# Table 4: trajectory group x condition
# ---------------------------------------------------------------------------


@dataclass
class GroupCell:
    n: int
    #: n and compiled_n are always both shown -- never collapse to a bare
    #: rate without the underlying counts (the user who commissioned this
    #: script explicitly asked that success counts/denominators never be
    #: dropped from the table or its accompanying aggregate).
    compiled_n: int
    compile_rate: Optional[float]
    overall_contract: Optional[float]
    #: How many of this group's trajectories got there via generic
    #: JSON-schema recovery only, not a named domain verifier (see
    #: ``classify_trajectory``'s ``generic_schema_only``) -- reported
    #: alongside the group, never used to hide or N/A it.
    generic_schema_only_n: int
    #: Recovery-attempt success counts/denominators for this group, pooled
    #: from every member's verification_events/recovery_events via the same
    #: ``compute_recovery_success_rate`` Metric 2 uses on a whole batch (see
    #: ``recovery_success.py``) -- the real numbers behind "resolved" (B)
    #: vs. "unresolved" (C), always kept rather than only showing the group
    #: label.
    recovery_successfully_recovered: int
    recovery_failures_with_recovery_attempt: int
    recovery_without_recovery_path: int
    recovery_without_recovery_attempt: int


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return (sum(present) / len(present)) if present else None


def build_table4(conditions: Sequence[Condition]) -> Dict[str, Dict[TrajectoryGroup, GroupCell]]:
    table: Dict[str, Dict[TrajectoryGroup, GroupCell]] = {}
    for condition in conditions:
        groups: Dict[TrajectoryGroup, List[SampleRecord]] = {g: [] for g in TrajectoryGroup}
        for sample in condition.samples:
            groups[sample.classification.group].append(sample)

        cells: Dict[TrajectoryGroup, GroupCell] = {}
        for group, members in groups.items():
            n = len(members)
            # Same canonical functions as Table 5 (and run_evaluation.sh),
            # just invoked on this group's subset of samples instead of the
            # whole batch -- guarantees a group's compile rate / contract
            # mean is computed identically to the batch-level Foundation
            # metrics, not a parallel reimplementation of the same formula.
            compile_summary = compute_compile_success_rate([m.compile_result for m in members]) if n else None
            overall_contract = (
                aggregate_contract_satisfaction(
                    [m.contract_result for m in members if m.contract_result is not None]
                )["overall"]
                if n else None
            )
            recovery_summary = compute_recovery_success_rate(
                [e for m in members if m.run_result for e in m.run_result.verification_events],
                [e for m in members if m.run_result for e in m.run_result.recovery_events],
            )

            cells[group] = GroupCell(
                n=n,
                compiled_n=compile_summary["successfully_compiled"] if compile_summary else 0,
                compile_rate=compile_summary["compile_success_rate"] if compile_summary else None,
                overall_contract=overall_contract,
                generic_schema_only_n=sum(1 for m in members if m.classification.generic_schema_only),
                recovery_successfully_recovered=recovery_summary["successfully_recovered"],
                recovery_failures_with_recovery_attempt=recovery_summary["failures_with_recovery_attempt"],
                recovery_without_recovery_path=recovery_summary["detected_failures_without_recovery_path"],
                recovery_without_recovery_attempt=recovery_summary["detected_failures_without_recovery_attempt"],
            )
        table[condition.label] = cells
    return table


# ---------------------------------------------------------------------------
# Table 5: document-level quality and generation cost
# ---------------------------------------------------------------------------


@dataclass
class MetricStat:
    mean: Optional[float]
    sd: Optional[float]


@dataclass
class Table5Condition:
    label: str
    n: int
    contract_stats: Dict[str, MetricStat]  # overall, required_sections, required_components, terminology, notation
    compile_reached_n: int
    compile_reached_rate: Optional[float]
    compile_success_rate: Optional[float]
    avg_llm_calls: Optional[float]
    total_input_tokens: int
    total_output_tokens: int


_CONTRACT_FIELDS = ["overall", "required_sections", "required_components", "terminology", "notation"]


def _sd(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return statistics.stdev(present) if len(present) > 1 else 0.0


def build_table5(conditions: Sequence[Condition]) -> List[Table5Condition]:
    results: List[Table5Condition] = []
    for condition in conditions:
        n = len(condition.samples)

        # Mean comes straight from the canonical aggregator (same function
        # run_evaluation.sh/aggregate.py call) so it can never silently
        # diverge from what the Foundation evaluation layer itself reports;
        # only the standard deviation (which that aggregator doesn't compute)
        # is derived here, from the identical None-filtered value lists.
        contract_results = [s.contract_result for s in condition.samples if s.contract_result is not None]
        contract_means = aggregate_contract_satisfaction(contract_results)
        contract_stats = {
            field_name: MetricStat(
                mean=contract_means[field_name],
                sd=_sd([getattr(cr, field_name) for cr in contract_results]),
            )
            for field_name in _CONTRACT_FIELDS
        }

        # Same here: compile_reached_n/compile_success_rate come from
        # compute_compile_success_rate (compile_success.py) -- the exact
        # function Metric 6 uses -- rather than a parallel reimplementation.
        compile_results = [s.compile_result for s in condition.samples]
        compile_summary = compute_compile_success_rate(compile_results)

        call_counts = [
            s.run_result.token_usage.call_count
            for s in condition.samples
            if s.run_result and s.run_result.token_usage
        ]
        total_input = sum(
            s.run_result.token_usage.total_input_tokens
            for s in condition.samples if s.run_result and s.run_result.token_usage
        )
        total_output = sum(
            s.run_result.token_usage.total_output_tokens
            for s in condition.samples if s.run_result and s.run_result.token_usage
        )

        results.append(Table5Condition(
            label=condition.label,
            n=n,
            contract_stats=contract_stats,
            compile_reached_n=compile_summary["runs_that_reached_compile"],
            compile_reached_rate=(
                compile_summary["runs_that_reached_compile"] / compile_summary["generation_runs"]
            ) if compile_summary["generation_runs"] else None,
            compile_success_rate=compile_summary["compile_success_rate"],
            avg_llm_calls=_mean(call_counts) if call_counts else None,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
        ))
    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pct(x: Optional[float], digits: int = 0, latex: bool = False) -> str:
    if x is None:
        return "N/A"
    sign = "\\%" if latex else "%"
    return f"{x * 100:.{digits}f}{sign}"


def _num(x: Optional[float], digits: int = 3) -> str:
    return "N/A" if x is None else f"{x:.{digits}f}"


_GROUP_CAPTION = {
    TrajectoryGroup.A_NO_FAILURE: "No recorded failure",
    TrajectoryGroup.B_RESOLVED: "Resolved failures",
    TrajectoryGroup.C_UNRESOLVED: "Unresolved failures",
    TrajectoryGroup.D_NO_RECOVERY_PATH: "Failures without verifier-specific recovery",
    TrajectoryGroup.U_UNKNOWN: "run_result.json missing/unreadable (failure status unknown -- not counted as A)",
}


def render_table4_markdown(table: Dict[str, Dict[TrajectoryGroup, GroupCell]]) -> str:
    """Main n / compile rate / overall contract view -- every cell is always
    the real computed number, no condition ever substitutes N/A. See
    ``render_table4_detail_markdown`` for the compiled_n, recovery success/
    denominator, and generic-schema-only counts behind these cells."""
    labels = list(table.keys())
    header = "| Group | " + " | ".join(f"{lbl} n | {lbl} Compile rate | {lbl} Overall contract" for lbl in labels) + " |"
    sep = "|---" * (1 + 3 * len(labels)) + "|"
    lines = [header, sep]
    for group in TrajectoryGroup:
        row = [group.value]
        for lbl in labels:
            cell = table[lbl][group]
            row += [str(cell.n), _pct(cell.compile_rate), _num(cell.overall_contract)]
        lines.append("| " + " | ".join(row) + " |")
    caption = "\n".join(f"- **{g.value}**: {c}" for g, c in _GROUP_CAPTION.items())
    return "\n".join(lines) + "\n\n" + caption


def render_table4_detail_markdown(table: Dict[str, Dict[TrajectoryGroup, GroupCell]]) -> str:
    """The success counts/denominators behind Table 4's rates: compiled_n/n,
    and -- for every group, not just B/C -- the pooled recovery-attempt
    success count and denominator (``compute_recovery_success_rate``'s own
    ``successfully_recovered``/``failures_with_recovery_attempt``, plus the
    two no-recovery-path/no-recovery-attempt buckets) and how many of the
    group's trajectories were generic-JSON-schema-only. Nothing here is
    hidden or marked N/A; this is the auxiliary aggregate the compact
    render_table4_* views summarize into a single rate."""
    lines = [
        "| Condition | Group | n | Compiled/n | Recovered/Attempted | Without recovery path | "
        "Without recovery attempt | Generic-schema-only n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for lbl, groups in table.items():
        for group in TrajectoryGroup:
            cell = groups[group]
            lines.append(
                f"| {lbl} | {group.value} | {cell.n} | {cell.compiled_n}/{cell.n} | "
                f"{cell.recovery_successfully_recovered}/{cell.recovery_failures_with_recovery_attempt} | "
                f"{cell.recovery_without_recovery_path} | {cell.recovery_without_recovery_attempt} | "
                f"{cell.generic_schema_only_n} |"
            )
    return "\n".join(lines)


def render_table4_latex(
    table: Dict[str, Dict[TrajectoryGroup, GroupCell]],
    caption: str = (
        "Comparison of compilation success rate and overall contract by trajectory group. "
        "Groups A--D denote no recorded failure, resolved failures, unresolved failures, and "
        "failures without verifier-specific recovery, respectively; Group U is trajectories "
        "whose run\\_result.json is missing or unreadable (failure status unknown). $n$ denotes "
        "the number of trajectories in each group; per-group compiled/recovered counts and "
        "denominators, and generic-JSON-schema-only counts, are in the accompanying CSV."
    ),
    label: str = "tab:trajectory_groups",
) -> str:
    labels = list(table.keys())
    col_spec = "l" + "ccc" * len(labels)
    header_row1 = "Group & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lbl}}}" for lbl in labels) + " \\\\"
    cmidrules = " ".join(f"\\cmidrule(lr){{{2 + 3*i}-{4 + 3*i}}}" for i in range(len(labels)))
    header_row2 = " & " + " & ".join("$n$ & Compile rate & Overall contract" for _ in labels) + " \\\\"

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header_row1,
        cmidrules,
        header_row2,
        "\\midrule",
    ]
    for group in TrajectoryGroup:
        cells = [group.value]
        for lbl in labels:
            cell = table[lbl][group]
            cells += [str(cell.n), _pct(cell.compile_rate, latex=True), _num(cell.overall_contract)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def render_table4_csv(table: Dict[str, Dict[TrajectoryGroup, GroupCell]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "group", "n", "compiled_n", "compile_rate", "overall_contract",
            "generic_schema_only_n", "recovery_successfully_recovered",
            "recovery_failures_with_recovery_attempt", "recovery_without_recovery_path",
            "recovery_without_recovery_attempt",
        ])
        for lbl, groups in table.items():
            for group, cell in groups.items():
                writer.writerow([
                    lbl, group.value, cell.n, cell.compiled_n, cell.compile_rate, cell.overall_contract,
                    cell.generic_schema_only_n, cell.recovery_successfully_recovered,
                    cell.recovery_failures_with_recovery_attempt, cell.recovery_without_recovery_path,
                    cell.recovery_without_recovery_attempt,
                ])


_T5_ROW_LABELS = {
    "overall": "Overall contract",
    "required_sections": "Required sections",
    "required_components": "Required components",
    "terminology": "Terminology",
    "notation": "Notation",
}


def _relative_change(
    no_verifier: Optional[float], all_verifiers: Optional[float], latex: bool = False
) -> Optional[str]:
    if no_verifier is None or all_verifiers is None or no_verifier == 0:
        return None
    sign = "\\%" if latex else "%"
    return f"{(all_verifiers - no_verifier) / no_verifier * 100:+.1f}{sign}"


def _pp_change(
    no_verifier: Optional[float], all_verifiers: Optional[float], latex: bool = False
) -> Optional[str]:
    if no_verifier is None or all_verifiers is None:
        return None
    sep = "\\,pp" if latex else " pp"
    return f"{(all_verifiers - no_verifier) * 100:+.1f}{sep}"


def render_table5_markdown(results: List[Table5Condition], no_verifier_label: str, all_verifiers_label: str) -> str:
    by_label = {r.label: r for r in results}
    nv, av = by_label.get(no_verifier_label), by_label.get(all_verifiers_label)

    lines = ["| Metric | " + " | ".join(r.label for r in results) + (" | Relative Change |" if nv and av else " |")]
    lines.append("|---" * (len(results) + (2 if nv and av else 1)) + "|")

    lines.append("| *Contract satisfaction* | " + " | ".join("" for _ in results) + ("|" if not (nv and av) else "|"))
    for field_name, row_label in _T5_ROW_LABELS.items():
        cells = [f"{_num(by_label[r.label].contract_stats[field_name].mean)} (+-{_num(by_label[r.label].contract_stats[field_name].sd)})" for r in results]
        rel = _relative_change(nv.contract_stats[field_name].mean, av.contract_stats[field_name].mean) if nv and av else None
        row = [row_label] + cells + ([rel or "N/A"] if nv and av else [])
        lines.append("| " + " | ".join(row) + " |")

    lines.append("| *Generation robustness* | " + " | ".join("" for _ in results) + "|")
    reached_cells = [f"{r.compile_reached_n} ({_pct(r.compile_reached_rate)})" for r in results]
    reached_rel = _pp_change(nv.compile_reached_rate, av.compile_reached_rate) if nv and av else None
    lines.append("| " + " | ".join(["Compile reached trajectory"] + reached_cells + ([reached_rel or "N/A"] if nv and av else [])) + " |")
    success_cells = [_pct(r.compile_success_rate, digits=1) for r in results]
    success_rel = _pp_change(nv.compile_success_rate, av.compile_success_rate) if nv and av else None
    lines.append("| " + " | ".join(["Compilation success rate"] + success_cells + ([success_rel or "N/A"] if nv and av else [])) + " |")

    lines.append("| *Generation cost* | " + " | ".join("" for _ in results) + "|")
    lines.append("| " + " | ".join(["Documents"] + [str(r.n) for r in results] + (["-"] if nv and av else [])) + " |")
    calls_cells = [_num(r.avg_llm_calls, 2) for r in results]
    calls_rel = _relative_change(nv.avg_llm_calls, av.avg_llm_calls) if nv and av else None
    lines.append("| " + " | ".join(["Avg. LLM calls per document"] + calls_cells + ([calls_rel or "N/A"] if nv and av else [])) + " |")
    in_tok_cells = [f"{r.total_input_tokens / 1e6:.2f}M" for r in results]
    in_tok_rel = _relative_change(nv.total_input_tokens, av.total_input_tokens) if nv and av else None
    lines.append("| " + " | ".join(["Total input tokens"] + in_tok_cells + ([in_tok_rel or "N/A"] if nv and av else [])) + " |")
    out_tok_cells = [f"{r.total_output_tokens / 1e6:.2f}M" for r in results]
    out_tok_rel = _relative_change(nv.total_output_tokens, av.total_output_tokens) if nv and av else None
    lines.append("| " + " | ".join(["Total output tokens"] + out_tok_cells + ([out_tok_rel or "N/A"] if nv and av else [])) + " |")

    return "\n".join(lines)


def render_table5_latex(
    results: List[Table5Condition],
    no_verifier_label: str,
    all_verifiers_label: str,
    caption: str = (
        "Document-level quality and generation cost. Contract-satisfaction metrics are "
        "reported as mean $\\pm$ standard deviation. The Relative Change column reports the "
        "change relative to No-verifier, except for compilation success and compile-reached "
        "rate, for which it reports an absolute percentage-point difference."
    ),
    label: str = "tab:doc_quality_cost",
) -> str:
    by_label = {r.label: r for r in results}
    nv, av = by_label.get(no_verifier_label), by_label.get(all_verifiers_label)
    has_rel = nv is not None and av is not None

    col_spec = "l" + "c" * len(results) + ("c" if has_rel else "")
    header = ["Metric"] + [r.label for r in results] + (["Relative Change"] if has_rel else [])

    def row(label_txt: str, cells: List[str], rel: Optional[str]) -> str:
        parts = [label_txt] + cells + ([rel if rel is not None else "--"] if has_rel else [])
        return " & ".join(parts) + " \\\\"

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
        "\\multicolumn{" + str(len(header)) + "}{l}{\\textit{Contract satisfaction}} \\\\",
    ]
    for field_name, row_label in _T5_ROW_LABELS.items():
        cells = [
            f"{_num(by_label[r.label].contract_stats[field_name].mean)} ($\\pm${_num(by_label[r.label].contract_stats[field_name].sd)})"
            for r in results
        ]
        rel = _relative_change(
            nv.contract_stats[field_name].mean, av.contract_stats[field_name].mean, latex=True
        ) if has_rel else None
        lines.append(row(f"{row_label} ($\\uparrow$)", cells, rel))

    lines.append("\\midrule")
    lines.append("\\multicolumn{" + str(len(header)) + "}{l}{\\textit{Generation robustness}} \\\\")
    reached_cells = [f"{r.compile_reached_n} ({_pct(r.compile_reached_rate, latex=True)})" for r in results]
    lines.append(row("Compile reached trajectory", reached_cells,
                      _pp_change(nv.compile_reached_rate, av.compile_reached_rate, latex=True) if has_rel else None))
    success_cells = [_pct(r.compile_success_rate, digits=1, latex=True) for r in results]
    lines.append(row("Compilation success rate", success_cells,
                      _pp_change(nv.compile_success_rate, av.compile_success_rate, latex=True) if has_rel else None))

    lines.append("\\midrule")
    lines.append("\\multicolumn{" + str(len(header)) + "}{l}{\\textit{Generation cost}} \\\\")
    lines.append(row("Documents", [str(r.n) for r in results], "-" if has_rel else None))
    calls_cells = [_num(r.avg_llm_calls, 2) for r in results]
    lines.append(row("Avg. LLM calls per document", calls_cells,
                      _relative_change(nv.avg_llm_calls, av.avg_llm_calls, latex=True) if has_rel else None))
    in_tok_cells = [f"{r.total_input_tokens / 1e6:.2f}M" for r in results]
    lines.append(row("Total input tokens", in_tok_cells,
                      _relative_change(nv.total_input_tokens, av.total_input_tokens, latex=True) if has_rel else None))
    out_tok_cells = [f"{r.total_output_tokens / 1e6:.2f}M" for r in results]
    lines.append(row("Total output tokens", out_tok_cells,
                      _relative_change(nv.total_output_tokens, av.total_output_tokens, latex=True) if has_rel else None))

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def render_table5_csv(results: List[Table5Condition], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "n", *[f"{fn}_mean" for fn in _CONTRACT_FIELDS], *[f"{fn}_sd" for fn in _CONTRACT_FIELDS],
            "compile_reached_n", "compile_reached_rate", "compile_success_rate",
            "avg_llm_calls", "total_input_tokens", "total_output_tokens",
        ])
        for r in results:
            writer.writerow([
                r.label, r.n,
                *[r.contract_stats[fn].mean for fn in _CONTRACT_FIELDS],
                *[r.contract_stats[fn].sd for fn in _CONTRACT_FIELDS],
                r.compile_reached_n, r.compile_reached_rate, r.compile_success_rate,
                r.avg_llm_calls, r.total_input_tokens, r.total_output_tokens,
            ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_condition_arg(raw: str) -> "tuple[str, Path]":
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--condition expects LABEL=DIR, got {raw!r}")
    label, dir_str = raw.split("=", 1)
    return label, Path(dir_str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate Table 4 (trajectory groups) and Table 5 (document-level "
        "quality/cost) from one or more pipeline output directories.",
    )
    parser.add_argument(
        "--condition", action="append", required=True, metavar="LABEL=DIR",
        help="One experimental condition, e.g. --condition \"All verifiers=outputs/qwen_all_verifiers_10151200\". "
        "Repeat once per condition (Table 4/5 expect exactly two: all-verifiers and no-verifier).",
    )
    parser.add_argument(
        "--no-verifier-condition", default=None, metavar="LABEL",
        help="Which --condition LABEL is the no-verifier run (Table 5's Relative Change "
        "baseline). Defaults to the label containing 'no' case-insensitively if exactly one "
        "matches.",
    )
    parser.add_argument(
        "--all-verifiers-condition", default=None, metavar="LABEL",
        help="Which --condition LABEL is the all-verifiers run (Table 5's Relative Change "
        "target). Defaults to the other label when exactly two conditions are given.",
    )
    parser.add_argument(
        "--benchmark", default=None,
        help="Benchmark JSONL to recompute Contract Satisfaction from current code "
        "(dataset2/benchmark/benchmark_v2.jsonl). Omit to fall back to each run_result.json's "
        "embedded contract_result.",
    )
    parser.add_argument(
        "--out-dir", default="evaluation/paper_tables",
        help="Where to write the rendered tables (LaTeX + CSV). Default: evaluation/paper_tables/",
    )
    args = parser.parse_args()

    conditions_raw = [_parse_condition_arg(c) for c in args.condition]
    labels = [lbl for lbl, _ in conditions_raw]

    no_verifier_label = args.no_verifier_condition
    if no_verifier_label is None:
        matches = [lbl for lbl in labels if "no" in lbl.lower() and "verif" in lbl.lower()]
        no_verifier_label = matches[0] if len(matches) == 1 else None

    all_verifiers_label = args.all_verifiers_condition
    if all_verifiers_label is None and no_verifier_label is not None and len(labels) == 2:
        all_verifiers_label = next(lbl for lbl in labels if lbl != no_verifier_label)

    benchmark_items = _load_benchmark_items(args.benchmark)

    conditions: List[Condition] = []
    for label, batch_dir in conditions_raw:
        if not batch_dir.exists():
            raise FileNotFoundError(f"--condition {label!r}: directory not found: {batch_dir}")
        condition = load_condition(label, batch_dir, benchmark_items)
        print(f"[paper_tables] {label}: {len(condition.samples)} sample(s) discovered under {batch_dir}")
        conditions.append(condition)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table4 = build_table4(conditions)
    (out_dir / "table4.tex").write_text(render_table4_latex(table4) + "\n", encoding="utf-8")
    render_table4_csv(table4, out_dir / "table4.csv")
    print("\n" + render_table4_markdown(table4) + "\n")
    print("\n[paper_tables] Table 4 detail (success counts/denominators behind each rate):\n")
    print(render_table4_detail_markdown(table4) + "\n")

    table5 = build_table5(conditions)
    if no_verifier_label and all_verifiers_label:
        (out_dir / "table5.tex").write_text(
            render_table5_latex(table5, no_verifier_label, all_verifiers_label) + "\n", encoding="utf-8"
        )
        print("\n" + render_table5_markdown(table5, no_verifier_label, all_verifiers_label) + "\n")
    else:
        print(
            "[paper_tables] Skipping Table 5's Relative Change column: could not identify "
            "which --condition is 'no verifier' vs 'all verifiers'. Pass --no-verifier-condition "
            "and --all-verifiers-condition explicitly."
        )
    render_table5_csv(table5, out_dir / "table5.csv")

    print(f"[paper_tables] Wrote table4.tex / table4.csv / table5.csv (+table5.tex if resolved) to {out_dir}")


if __name__ == "__main__":
    main()

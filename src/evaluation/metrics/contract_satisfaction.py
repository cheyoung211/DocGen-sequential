"""Metric 7 (Contract Satisfaction Rate), hard/deterministic categories only.

Checks a generated document (a project_dir under ``outputs/...``) against
the benchmark's ``BenchmarkItem`` contract. ``content_coverage`` (topic/
objective coverage) and ``cross_section_dependency`` (logical dependency
between sections) are semantic checks the research spec explicitly permits
deferring -- they are never attempted here; see ``ContractResult``.

Required Section Coverage matches on normalized *title* text, not
``section_id``: the Planner is never shown ``RequiredSection.section_id``
(only the section title, embedded in prose inside
``natural_language_instruction`` -- confirmed by grepping the whole
generation pipeline for any reference to the benchmark's ids), and a real
generated ``plan.json`` shows the Planner inventing its own ``sec_``-prefixed
ids independent of the benchmark's scheme. This under-counts a section whose
title the model paraphrased despite being given the exact string -- a known,
accepted recall limitation for this deterministic-only pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.common.schemas import PlannerOutput, TextAgentOutput
from src.common.state import DocumentBlueprint, LayoutBlockKind
from src.dataset.schemas import BenchmarkItem, ComponentType
from src.evaluation.schemas import ContractCategoryDetail, ContractResult

# Built from the Planner's own documented kind vocabulary
# (PLANNER_STRATEGY_PROMPT's "kind vocabulary" section in
# src/agents/planner.py) -- the authoritative mapping the Planner is
# actually instructed to use, not a guess at what "should" correspond to
# what. A ComponentType mapped to None has no LayoutBlockKind capable of
# expressing it at all in this pipeline today (PROPOSITION/COROLLARY/full
# PROOF/TIMELINE_PLACEHOLDER/EXERCISE/SUMMARY_BOX) -- evaluate_contract
# reports these as "not_representable", never conflated with "missing", so
# a permanent pipeline capability gap is never misattributed as this run's
# failure.
COMPONENT_TYPE_TO_LAYOUT_KIND: Dict[ComponentType, Optional[LayoutBlockKind]] = {
    ComponentType.DEFINITION: LayoutBlockKind.DEFINITION,
    ComponentType.THEOREM: LayoutBlockKind.THEOREM,
    ComponentType.LEMMA: LayoutBlockKind.THEOREM,
    ComponentType.PROPOSITION: None,
    ComponentType.COROLLARY: None,
    ComponentType.PROOF: None,
    ComponentType.PROOF_SKETCH: LayoutBlockKind.PROOF_SKETCH,
    ComponentType.EQUATION: LayoutBlockKind.EQUATION,
    ComponentType.ALIGNED_EQUATIONS: LayoutBlockKind.EQUATION,
    ComponentType.TABLE: LayoutBlockKind.TABLE,
    ComponentType.NOTATION_TABLE: LayoutBlockKind.NOTATION_TABLE,
    ComponentType.COMPARISON_TABLE: LayoutBlockKind.COMPARISON,
    ComponentType.TERMINOLOGY_TABLE: LayoutBlockKind.TABLE,
    ComponentType.CLASSIFICATION_TABLE: LayoutBlockKind.TABLE,
    ComponentType.WORKED_EXAMPLE: LayoutBlockKind.CASE_STUDY,
    ComponentType.CALLOUT: LayoutBlockKind.WARNING,
    ComponentType.RISK_CALLOUT: LayoutBlockKind.WARNING,
    ComponentType.BOXED_RESULT: LayoutBlockKind.KEY_TAKEAWAY,
    ComponentType.FIGURE_PLACEHOLDER: LayoutBlockKind.FIGURE,
    ComponentType.TIMELINE_PLACEHOLDER: None,
    ComponentType.EXERCISE: None,
    ComponentType.SUMMARY_BOX: None,
}


def _normalize_title(title: str) -> str:
    return " ".join(title.split()).casefold()


def _drafted_section_ids(project_dir: Path, blueprint: DocumentBlueprint) -> set:
    sections_dir = project_dir / "sections"
    return {
        section.section_id
        for section in blueprint.sections
        if (sections_dir / f"{section.section_id}.blocks.json").is_file()
    }


def _collect_document_text(project_dir: Path, blueprint: DocumentBlueprint) -> str:
    """Rendered LaTeX when compilation reached this section, else the
    persisted semantic blocks -- whichever the run actually produced."""
    parts: List[str] = []
    sections_dir = project_dir / "sections"
    for section in blueprint.sections:
        tex_path = sections_dir / f"{section.section_id}.tex"
        if tex_path.is_file():
            parts.append(tex_path.read_text(encoding="utf-8", errors="replace"))
            continue
        blocks_path = sections_dir / f"{section.section_id}.blocks.json"
        if not blocks_path.is_file():
            continue
        try:
            output = TextAgentOutput.model_validate(
                json.loads(blocks_path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, ValueError):
            continue
        for block in output.blocks:
            if block.title:
                parts.append(block.title)
            parts.append(block.content)
    return "\n".join(parts)


def _score_required_sections(
    project_dir: Path, blueprint: DocumentBlueprint, benchmark_item: BenchmarkItem
) -> "tuple[Optional[float], List[ContractCategoryDetail]]":
    generated_by_title = {_normalize_title(s.title): s for s in blueprint.sections}
    drafted_ids = _drafted_section_ids(project_dir, blueprint)

    details: List[ContractCategoryDetail] = []
    satisfied = 0
    for req in benchmark_item.required_sections:
        matched = generated_by_title.get(_normalize_title(req.title))
        if matched is None:
            details.append(ContractCategoryDetail(
                item_id=req.section_id, status="missing",
                reason=f"no generated section titled {req.title!r}",
            ))
        elif matched.section_id not in drafted_ids:
            details.append(ContractCategoryDetail(
                item_id=req.section_id, status="missing",
                reason=f"section {matched.section_id!r} was planned but never drafted",
            ))
        else:
            satisfied += 1
            details.append(ContractCategoryDetail(item_id=req.section_id, status="satisfied"))

    total = len(benchmark_item.required_sections)
    score = (satisfied / total) if total else None
    return score, details


def _score_required_components(
    project_dir: Path, blueprint: DocumentBlueprint, benchmark_item: BenchmarkItem
) -> "tuple[Optional[float], List[ContractCategoryDetail]]":
    drafted_ids = _drafted_section_ids(project_dir, blueprint)
    kinds_present = {
        block.kind
        for section in blueprint.sections
        if section.section_id in drafted_ids
        for block in section.blocks
    }

    details: List[ContractCategoryDetail] = []
    satisfied = 0
    applicable = 0
    for comp in benchmark_item.required_components:
        layout_kind = COMPONENT_TYPE_TO_LAYOUT_KIND.get(comp.type)
        if layout_kind is None:
            details.append(ContractCategoryDetail(
                item_id=comp.component_id, status="not_representable",
                reason=f"ComponentType.{comp.type.value} has no corresponding LayoutBlockKind",
            ))
            continue
        applicable += 1
        if layout_kind in kinds_present:
            satisfied += 1
            details.append(ContractCategoryDetail(item_id=comp.component_id, status="satisfied"))
        else:
            details.append(ContractCategoryDetail(
                item_id=comp.component_id, status="missing",
                reason=f"no drafted block of kind '{layout_kind.value}' found",
            ))

    score = (satisfied / applicable) if applicable else None
    return score, details


def _score_terminology(
    document_text: str, benchmark_item: BenchmarkItem
) -> "tuple[Optional[float], List[ContractCategoryDetail]]":
    constraints = benchmark_item.terminology_constraints
    if not benchmark_item.evaluation_contract.terminology_consistency_check or not constraints:
        return None, []

    text_casefold = document_text.casefold()
    details: List[ContractCategoryDetail] = []
    satisfied = 0
    for constraint in constraints:
        candidates = [constraint.canonical_term, *constraint.allowed_variants]
        term_used = any(c.casefold() in text_casefold for c in candidates if c)
        forbidden_used = any(
            v.casefold() in text_casefold for v in constraint.forbidden_variants if v
        )
        if term_used and not forbidden_used:
            satisfied += 1
            details.append(
                ContractCategoryDetail(item_id=constraint.canonical_term, status="satisfied")
            )
        else:
            reason = (
                "forbidden variant present" if forbidden_used
                else "canonical term (or an allowed variant) never appears in the document"
            )
            details.append(ContractCategoryDetail(
                item_id=constraint.canonical_term, status="missing", reason=reason
            ))

    return satisfied / len(constraints), details


def _score_notation(
    document_text: str, benchmark_item: BenchmarkItem
) -> "tuple[Optional[float], List[ContractCategoryDetail]]":
    constraints = benchmark_item.notation_constraints
    if not benchmark_item.evaluation_contract.notation_consistency_check or not constraints:
        return None, []

    details: List[ContractCategoryDetail] = []
    satisfied = 0
    for constraint in constraints:
        if constraint.symbol in document_text:
            satisfied += 1
            details.append(ContractCategoryDetail(item_id=constraint.symbol, status="satisfied"))
        else:
            details.append(ContractCategoryDetail(
                item_id=constraint.symbol, status="missing",
                reason="symbol never appears in the document",
            ))

    return satisfied / len(constraints), details


def evaluate_contract(project_dir: Path, benchmark_item: BenchmarkItem) -> ContractResult:
    """Pure, read-only, no LLM/network calls. Reads ``project_dir/plan.json``
    (the blueprint) and ``project_dir/sections/*``. Returns a null-filled
    ``ContractResult`` if ``plan.json`` doesn't exist yet (planning itself
    never completed)."""
    plan_path = project_dir / "plan.json"
    if not plan_path.is_file():
        return ContractResult()

    plan = PlannerOutput.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))
    blueprint = plan.blueprint

    required_sections_score, section_details = _score_required_sections(
        project_dir, blueprint, benchmark_item
    )
    required_components_score, component_details = _score_required_components(
        project_dir, blueprint, benchmark_item
    )
    document_text = _collect_document_text(project_dir, blueprint)
    terminology_score, terminology_details = _score_terminology(document_text, benchmark_item)
    notation_score, notation_details = _score_notation(document_text, benchmark_item)

    applicable_scores = [
        s for s in (required_sections_score, required_components_score, terminology_score, notation_score)
        if s is not None
    ]
    overall = (sum(applicable_scores) / len(applicable_scores)) if applicable_scores else None

    return ContractResult(
        overall=overall,
        required_sections=required_sections_score,
        required_components=required_components_score,
        terminology=terminology_score,
        notation=notation_score,
        details={
            "required_sections": section_details,
            "required_components": component_details,
            "terminology": terminology_details,
            "notation": notation_details,
        },
    )


def aggregate_contract_satisfaction(results: Sequence[ContractResult]) -> dict:
    """Category-level mean across a batch, skipping None (not-applicable/not-computed) per sample."""

    def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
        present = [v for v in values if v is not None]
        return (sum(present) / len(present)) if present else None

    return {
        "overall": _mean([r.overall for r in results]),
        "required_sections": _mean([r.required_sections for r in results]),
        "required_components": _mean([r.required_components for r in results]),
        "content_coverage": None,
        "terminology": _mean([r.terminology for r in results]),
        "notation": _mean([r.notation for r in results]),
        "cross_section_dependency": None,
        "not_implemented": ["content_coverage", "cross_section_dependency"],
        "sample_count": len(results),
    }

"""Quality validation pipeline (spec section 9): the 15 required checks plus
a 6-way :class:`QualityScore`, run over already schema-valid
:class:`BenchmarkItem` objects (pydantic construction already enforces rule
1, "schema validation").

Because ``instruction_builder.py`` renders ``natural_language_instruction``
directly from the same structured fields these rules check (section titles,
component descriptions, hard-constraint text), rules 9/10 ("instruction
includes sections/components/constraints") are expected to always pass for
items produced by this pipeline -- they exist here as a regression guard,
not because generation and validation are independent.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

from src.dataset.schemas import BenchmarkItem, QualityScore, ValidationIssue, ValidationResult

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "specificity": 0.8,
    "evaluability": 0.85,
    "latex_relevance": 0.8,
    "long_form_suitability": 0.8,
    "constraint_clarity": 0.85,
    "source_safety": 1.0,
}

_NATIVE_LATEX_COMPONENT_TYPES = {
    "definition", "theorem", "lemma", "proposition", "corollary", "proof",
    "proof_sketch", "equation", "aligned_equations", "table", "notation_table",
    "comparison_table", "terminology_table", "classification_table", "boxed_result",
}

MIN_WORDS_FOR_LONG_FORM = 800
_MAX_FIELD_CHARS = 4000  # rule 13: guards against an accidentally-pasted long block of prose


def validate_item(
    item: BenchmarkItem,
    *,
    seen_sample_ids: Set[str],
    thresholds: Dict[str, float],
    min_words_threshold: int = MIN_WORDS_FOR_LONG_FORM,
) -> ValidationResult:
    issues: List[ValidationIssue] = []
    section_ids = set(item.section_ids)
    component_ids = set(item.component_ids)

    # Rule 2: unique sample_id
    if item.sample_id in seen_sample_ids:
        issues.append(ValidationIssue(rule_id="unique_sample_id", message=f"Duplicate sample_id: {item.sample_id}"))

    # Rule 3: license / provenance completeness
    if not item.source_provenance:
        issues.append(ValidationIssue(rule_id="source_provenance_present", message="source_provenance is empty."))
    for entry in item.source_provenance:
        if entry.copied_text:
            issues.append(ValidationIssue(
                rule_id="no_copied_text",
                message=f"source_provenance entry {entry.source_id!r} claims copied_text=True.",
            ))

    # Rule 4: required-section count within target_length bounds
    n_sections = len(item.required_sections)
    if not (item.target_length.min_sections <= n_sections <= item.target_length.max_sections):
        issues.append(ValidationIssue(
            rule_id="section_count_within_bounds",
            message=(
                f"{n_sections} required_sections is outside target_length bounds "
                f"[{item.target_length.min_sections}, {item.target_length.max_sections}]."
            ),
        ))

    # Rule 5: every component placement references an existing section_id
    for comp in item.required_components:
        if comp.placement not in section_ids:
            issues.append(ValidationIssue(
                rule_id="placement_id_exists",
                message=f"required_component {comp.component_id!r} placement {comp.placement!r} is not a known section_id.",
            ))
        for dep_id in comp.depends_on:
            if dep_id not in component_ids:
                issues.append(ValidationIssue(
                    rule_id="component_dependency_id_exists",
                    message=f"required_component {comp.component_id!r} depends_on unknown component {dep_id!r}.",
                ))

    # Rule 6: every dependency's source/target sections exist
    for dep in item.cross_section_dependencies:
        if dep.source_section not in section_ids:
            issues.append(ValidationIssue(
                rule_id="dependency_source_id_exists",
                message=f"cross_section_dependency {dep.dependency_id!r} source_section {dep.source_section!r} unknown.",
            ))
        missing_targets = [t for t in dep.target_sections if t not in section_ids]
        if missing_targets:
            issues.append(ValidationIssue(
                rule_id="dependency_target_ids_exist",
                message=f"cross_section_dependency {dep.dependency_id!r} has unknown target_sections {missing_targets}.",
            ))

    # Rule 7: components are evaluable (non-empty description/type/placement -- already
    # pydantic-enforced for description via min length semantics of str, checked here for emptiness)
    for comp in item.required_components:
        if not comp.description.strip():
            issues.append(ValidationIssue(
                rule_id="component_evaluable",
                message=f"required_component {comp.component_id!r} has an empty description.",
            ))

    # Rule 8: hard constraints are machine-checkable (must declare where they apply)
    for hc in item.hard_constraints:
        if not hc.applicable_to:
            issues.append(ValidationIssue(
                rule_id="constraint_machine_checkable",
                message=f"hard_constraint {hc.constraint_id!r} has an empty applicable_to list.",
            ))

    # Rules 9-10: natural_language_instruction covers every required section /
    # component / hard constraint (keyword/text coverage, not full NLU-level
    # semantic equivalence -- see module docstring and instruction_builder.py).
    instruction_lower = item.natural_language_instruction.lower()
    for section in item.required_sections:
        if section.title.lower() not in instruction_lower:
            issues.append(ValidationIssue(
                rule_id="instruction_covers_sections",
                message=f"Section title {section.title!r} does not appear in natural_language_instruction.",
            ))
    for comp in item.required_components:
        readable = comp.type.value.replace("_", " ")
        if readable not in instruction_lower and comp.type.value not in instruction_lower:
            issues.append(ValidationIssue(
                rule_id="instruction_covers_components",
                message=f"Component type {comp.type.value!r} does not appear in natural_language_instruction.",
            ))
    for hc in item.hard_constraints:
        if hc.description.lower() not in instruction_lower:
            issues.append(ValidationIssue(
                rule_id="instruction_covers_constraints",
                message=f"hard_constraint {hc.constraint_id!r} description does not appear verbatim in natural_language_instruction.",
            ))

    # Rule 11: no contradictory constraints (same canonical_term/symbol defined twice, differently)
    seen_terms: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {}
    for tc in item.terminology_constraints:
        key = tc.canonical_term.lower()
        value = (tuple(tc.allowed_variants), tuple(tc.forbidden_variants))
        if key in seen_terms and seen_terms[key] != value:
            issues.append(ValidationIssue(
                rule_id="no_contradictory_constraints",
                message=f"terminology_constraint for {tc.canonical_term!r} is defined twice with different variants.",
            ))
        seen_terms[key] = value
    seen_symbols: Dict[str, str] = {}
    for nc in item.notation_constraints:
        if nc.symbol in seen_symbols and seen_symbols[nc.symbol] != nc.meaning:
            issues.append(ValidationIssue(
                rule_id="no_contradictory_constraints",
                message=f"notation_constraint for symbol {nc.symbol!r} is defined twice with different meanings.",
            ))
        seen_symbols[nc.symbol] = nc.meaning

    # Rule 12 is covered by rule 3 (no source_provenance entry may claim copied_text=True).

    # Rule 13: no copyrighted-prose-length field (guards against an accidentally
    # pasted long block; this pipeline never copies source prose in the first place).
    for field_name, value in (("overview", item.overview), ("document_purpose", item.document_purpose)):
        if len(value) > _MAX_FIELD_CHARS:
            issues.append(ValidationIssue(
                rule_id="no_oversized_prose_field",
                message=f"{field_name} is {len(value)} characters, over the {_MAX_FIELD_CHARS}-character safety threshold.",
            ))

    # Rule 14: no empty evaluation fields.
    if item.evaluation_contract is None:
        issues.append(ValidationIssue(rule_id="evaluation_contract_present", message="evaluation_contract is missing."))

    # Rule 15: not a vague prompt requiring arbitrary length expansion.
    if len(item.required_sections) < 3 or len(item.required_components) < 2:
        issues.append(ValidationIssue(
            rule_id="not_vague",
            message="Fewer than 3 required_sections or 2 required_components -- too underspecified for a long-form contract.",
        ))
    if item.target_length.min_words < min_words_threshold:
        issues.append(ValidationIssue(
            rule_id="not_vague",
            message=f"target_length.min_words ({item.target_length.min_words}) is below the long-form threshold ({min_words_threshold}).",
        ))

    quality_score = _compute_quality_score(item, issues, min_words_threshold=min_words_threshold)
    accepted = not issues and _meets_thresholds(quality_score, thresholds)

    return ValidationResult(sample_id=item.sample_id, accepted=accepted, quality_score=quality_score, issues=issues)


def _compute_quality_score(
    item: BenchmarkItem,
    issues: List[ValidationIssue],
    *,
    min_words_threshold: int = MIN_WORDS_FOR_LONG_FORM,
) -> QualityScore:
    issue_rules = {i.rule_id for i in issues}

    specificity = 1.0
    if len(item.required_sections) < 4:
        specificity -= 0.2
    if not item.notation_constraints and not item.terminology_constraints:
        specificity -= 0.3
    if "not_vague" in issue_rules:
        specificity -= 0.3
    specificity = max(0.0, min(1.0, specificity))

    evaluability = 1.0
    if {"placement_id_exists", "dependency_source_id_exists", "dependency_target_ids_exist",
        "component_dependency_id_exists"} & issue_rules:
        evaluability -= 0.5
    if not item.cross_section_dependencies:
        evaluability -= 0.3
    if "component_evaluable" in issue_rules:
        evaluability -= 0.3
    evaluability = max(0.0, min(1.0, evaluability))

    present_types = {c.type.value for c in item.required_components}
    latex_relevance = min(1.0, len(present_types & _NATIVE_LATEX_COMPONENT_TYPES) / 2.0)

    long_form_suitability = 1.0 if (
        item.target_length.min_words >= min_words_threshold and len(item.required_sections) >= 4
    ) else 0.4

    constraint_clarity = 1.0
    if {"instruction_covers_sections", "instruction_covers_components", "instruction_covers_constraints"} & issue_rules:
        constraint_clarity -= 0.5
    if not item.hard_constraints:
        constraint_clarity -= 0.5
    if "no_contradictory_constraints" in issue_rules:
        constraint_clarity -= 0.3
    constraint_clarity = max(0.0, min(1.0, constraint_clarity))

    source_safety = 0.0 if any(e.copied_text for e in item.source_provenance) or "no_copied_text" in issue_rules else 1.0

    return QualityScore(
        specificity=specificity,
        evaluability=evaluability,
        latex_relevance=latex_relevance,
        long_form_suitability=long_form_suitability,
        constraint_clarity=constraint_clarity,
        source_safety=source_safety,
    )


def _meets_thresholds(score: QualityScore, thresholds: Dict[str, float]) -> bool:
    return (
        score.specificity >= thresholds.get("specificity", 0.0)
        and score.evaluability >= thresholds.get("evaluability", 0.0)
        and score.latex_relevance >= thresholds.get("latex_relevance", 0.0)
        and score.long_form_suitability >= thresholds.get("long_form_suitability", 0.0)
        and score.constraint_clarity >= thresholds.get("constraint_clarity", 0.0)
        and score.source_safety >= thresholds.get("source_safety", 0.0)
    )


def validate_items(
    items: List[BenchmarkItem],
    *,
    thresholds: Dict[str, float] = DEFAULT_THRESHOLDS,
    min_words_threshold: int = MIN_WORDS_FOR_LONG_FORM,
) -> Tuple[List[BenchmarkItem], List[dict]]:
    """Validate a batch; return (accepted items with quality_score attached, rejected records)."""
    accepted: List[BenchmarkItem] = []
    rejected: List[dict] = []
    seen_ids: Set[str] = set()

    for item in items:
        result = validate_item(
            item, seen_sample_ids=seen_ids, thresholds=thresholds, min_words_threshold=min_words_threshold
        )
        seen_ids.add(item.sample_id)
        scored_item = item.model_copy(update={"quality_score": result.quality_score})
        if result.accepted:
            accepted.append(scored_item)
        else:
            logger.warning(
                "Rejected %s: %s",
                item.sample_id,
                "; ".join(f"{i.rule_id}: {i.message}" for i in result.issues) or "below quality thresholds",
            )
            rejected.append({
                "sample_id": item.sample_id,
                "quality_score": result.quality_score.model_dump(),
                "issues": [i.model_dump() for i in result.issues],
            })

    return accepted, rejected

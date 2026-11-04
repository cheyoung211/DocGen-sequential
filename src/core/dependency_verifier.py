"""Relation-specific verification for the "Revision (Additional verifiers)"
plan's Fix C, checklist item C3 -- both bullets, over one
``DependencyMappingReport`` (``src/core/dependency_mapping.py``, C1) and
the document's actual rendered section text:

- ``MUST_REFERENCE`` (C3 bullet 1): did the target section's rendered
  LaTeX actually cite one of the obligation's ``allowed_labels`` via a
  real ``\\ref``-style command (reusing ``latex_assembler.
  REFERENCE_COMMAND``, not reimplemented) -- a bare prose mention is not
  a citation (Gold Labeling sample_005: "summary.tex 전체에 Cref가 단 한
  번도 없음... 확정 미충족").
- ``MUST_PRECEDE`` (C3 bullet 1): does the matched source section
  actually precede the matched target section in the blueprint's own
  section order, per the obligation's ``order_hint``.
- ``MUST_REUSE``/``MUST_BE_CONSISTENT_WITH`` (C3 bullet 2, added once Fix
  B's B3 shared-value registry existed -- see
  ``docs/av_stage3_math_verifier.md``): delegates to
  ``shared_value_registry.check_shared_value_consistency`` for the
  numeric-restatement component specifically, via
  ``_verify_shared_value_relation`` below. Only checked when a caller
  explicitly supplies ``relevant_symbols`` (per obligation, via
  ``verify_cross_section_obligations``'s ``shared_value_configs``) --
  with none supplied, this stays UNKNOWN/NOT_REPRESENTABLE exactly as
  before this round, since this check never guesses which symbols a
  dependency shares.

Any obligation whose mapping ``status`` isn't ``MATCHED`` resolves to
``UNKNOWN`` with an explicit reason -- never silently ``PASS``. A
self-reported "I cited/reused this" claim from the generator is never
trusted in place of this scan (Gold Labeling sample_008's fabricated
citation is exactly the failure mode this guards against -- see the
design doc's rubric principle 8).

Deliberately a direct function over ``DependencyMappingReport`` + rendered
text, not yet a formal ``Validator``/``ConstraintProvider`` -- mirrors how
``src/evaluation/metrics/contract_satisfaction.py::evaluate_contract`` (the
one real, deployed post-hoc evaluator this codebase has) is also a direct
function, not a ``Validator``. Stage 0's own "Known gap" (no production
``ConstraintProvider`` bridges the dataset contract into the ``Validator``/
``ArtifactGraphStore`` world anywhere in this codebase yet) applies equally
here; wrapping this as a ``Validator`` is explicit, deferred follow-up work
("필요하면", the plan's own conditional wording), not assumed necessary now.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

from pydantic import BaseModel, Field

from src.agents.latex_assembler import REFERENCE_COMMAND
from src.core.dependency_mapping import CrossSectionObligation, DependencyMappingReport, MappingStatus
from src.core.schemas import CheckOutcome, UnknownReason
from src.core.shared_value_registry import SharedValueCheckConfig, check_shared_value_consistency
from src.dataset.schemas import DependencyRelation


def _cited_labels_with_lines(text: str) -> Dict[str, int]:
    """Every label actually cited via a real ``\\ref``-style command in
    ``text``, mapped to the 1-indexed line of its FIRST citation. Splits a
    multi-label command (``\\Cref{a,b}``) into its individual labels --
    the plan explicitly calls this case out ("여러 label을 담은 참조 명령도
    처리한다"), and the existing ``latex_assembler.validate_references``
    doesn't split it (that method only ever needs "does this whole string
    match some real label", a narrower question than this check's "which
    of several SPECIFIC allowed labels, if any, got cited").
    """
    first_line_by_label: Dict[str, int] = {}
    for match in REFERENCE_COMMAND.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        for raw_label in match.group("label").split(","):
            label = raw_label.strip()
            if label and label not in first_line_by_label:
                first_line_by_label[label] = line
    return first_line_by_label


class DependencyVerificationRecord(BaseModel):
    """One ``CrossSectionObligation``'s real verdict -- per TARGET, never
    per ``dependency_id`` alone, so one target's success can never stand
    in for another's (the plan's own "일부 target의 성공으로 dependency
    전체를 PASS하지 않는다"; see also C1's own per-target obligation design,
    ``docs/av_stage2_dependency_mapping.md`` §2's rationale)."""

    dependency_id: str
    relation: DependencyRelation
    target_section_id: str
    outcome: CheckOutcome
    detail: str = ""
    unknown_reason: Optional[UnknownReason] = None
    cited_labels_found: List[str] = Field(default_factory=list)
    matched_at_line: Optional[int] = None

    def model_post_init(self, __context: object) -> None:
        if self.outcome == CheckOutcome.UNKNOWN and self.unknown_reason is None:
            raise ValueError("DependencyVerificationRecord with outcome=UNKNOWN requires unknown_reason.")


class DependencyVerificationResult(BaseModel):
    """Aggregate PASS/FAIL/UNKNOWN counts kept as derived properties, never
    a separately-stored, separately-maintained field -- so there is only
    ever one place (``records``) that can disagree with itself (same
    "single source of truth" discipline the artifact graph's own review
    rounds settled on -- see ``docs/stage_review_rigor``'s pattern #10).
    """

    records: List[DependencyVerificationRecord] = Field(default_factory=list)

    @property
    def total_targets(self) -> int:
        return len(self.records)

    @property
    def checked_targets(self) -> int:
        return sum(1 for r in self.records if r.outcome != CheckOutcome.UNKNOWN)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.records if r.outcome == CheckOutcome.PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.records if r.outcome == CheckOutcome.FAIL)

    @property
    def unknown_count(self) -> int:
        return sum(1 for r in self.records if r.outcome == CheckOutcome.UNKNOWN)


def verify_cross_section_obligations(
    report: DependencyMappingReport,
    rendered_sections: Dict[str, str],
    shared_value_configs: Optional[Dict[str, SharedValueCheckConfig]] = None,
) -> DependencyVerificationResult:
    """Verifies every ``CrossSectionObligation`` C1 produced -- one record
    per obligation (i.e. per target), regardless of outcome. Never drops an
    obligation from the result just because it isn't checkable this round;
    that case is UNKNOWN with a reason, not an absence.

    ``shared_value_configs`` is keyed by ``dependency_id`` and only
    consulted for ``must_reuse``/``must_be_consistent_with`` obligations
    (C3 bullet 2) -- a ``dependency_id`` with no entry here (the default,
    with no argument at all) stays UNKNOWN/NOT_REPRESENTABLE, exactly the
    pre-B3 behavior, since this never guesses a symbol allowlist on its
    own (see ``shared_value_registry``'s own docstring)."""
    shared_value_configs = shared_value_configs or {}
    return DependencyVerificationResult(
        records=[
            _verify_one(obligation, rendered_sections, shared_value_configs)
            for obligation in report.cross_section_obligations
        ]
    )


def _verify_one(
    obligation: CrossSectionObligation,
    rendered_sections: Dict[str, str],
    shared_value_configs: Dict[str, SharedValueCheckConfig],
) -> DependencyVerificationRecord:
    if obligation.status != MappingStatus.MATCHED:
        return DependencyVerificationRecord(
            dependency_id=obligation.dependency_id,
            relation=obligation.relation,
            target_section_id=obligation.target_section_id,
            outcome=CheckOutcome.UNKNOWN,
            unknown_reason=UnknownReason.MISSING_DEPENDENCY,
            detail=f"source/target did not resolve to real generated sections: {obligation.reason}",
        )

    if obligation.relation == DependencyRelation.MUST_REFERENCE:
        return _verify_must_reference(obligation, rendered_sections)
    if obligation.relation == DependencyRelation.MUST_PRECEDE:
        return _verify_must_precede(obligation)

    config = shared_value_configs.get(obligation.dependency_id)
    if config is not None and config.relevant_symbols:
        return _verify_shared_value_relation(
            obligation,
            rendered_sections,
            config.relevant_symbols,
            scope=config.scope,
            unit=config.unit,
            assumptions=config.assumptions,
            symbol_aliases=config.symbol_aliases,
        )

    return DependencyVerificationRecord(
        dependency_id=obligation.dependency_id,
        relation=obligation.relation,
        target_section_id=obligation.target_section_id,
        outcome=CheckOutcome.UNKNOWN,
        unknown_reason=UnknownReason.NOT_REPRESENTABLE,
        detail=(
            f"{obligation.relation.value} verification needs an explicit relevant_symbols "
            "allowlist (shared_value_configs[dependency_id]) -- none supplied, and this check "
            "never guesses which symbols a dependency shares (see shared_value_registry's own "
            "docstring for why)."
        ),
    )


def _verify_shared_value_relation(
    obligation: CrossSectionObligation,
    rendered_sections: Dict[str, str],
    relevant_symbols: Sequence[str],
    *,
    scope: str = "global",
    unit: Optional[str] = None,
    assumptions: Sequence[str] = (),
    symbol_aliases: Optional[Dict[str, List[str]]] = None,
) -> DependencyVerificationRecord:
    """Adapter from ``shared_value_registry.check_shared_value_consistency``
    (B3's real check) into this module's ``DependencyVerificationRecord``
    vocabulary -- mirrors ``_verify_must_reference``/``_verify_must_precede``:
    the decision is made once, in ``shared_value_registry``, never
    reimplemented here."""
    source_text = rendered_sections.get(obligation.matched_source_section_id or "")
    target_text = rendered_sections.get(obligation.matched_target_section_id or "")
    if source_text is None or target_text is None:
        missing = []
        if source_text is None:
            missing.append(f"source {obligation.matched_source_section_id!r}")
        if target_text is None:
            missing.append(f"target {obligation.matched_target_section_id!r}")
        return DependencyVerificationRecord(
            dependency_id=obligation.dependency_id,
            relation=obligation.relation,
            target_section_id=obligation.target_section_id,
            outcome=CheckOutcome.UNKNOWN,
            unknown_reason=UnknownReason.MISSING_DEPENDENCY,
            detail=f"no rendered text supplied for: {', '.join(missing)}.",
        )

    config = SharedValueCheckConfig(
        relevant_symbols=list(relevant_symbols),
        symbol_aliases=symbol_aliases or {},
        scope=scope,
        unit=unit,
        assumptions=list(assumptions),
    )
    record = check_shared_value_consistency(
        dependency_id=obligation.dependency_id,
        relation=obligation.relation,
        source_section_id=obligation.matched_source_section_id,
        target_section_id=obligation.matched_target_section_id,
        source_text=source_text,
        target_text=target_text,
        config=config,
    )
    return DependencyVerificationRecord(
        dependency_id=record.dependency_id,
        relation=record.relation,
        target_section_id=record.target_section_id,
        outcome=record.outcome,
        unknown_reason=record.unknown_reason,
        detail=record.detail,
    )


def _verify_must_reference(
    obligation: CrossSectionObligation, rendered_sections: Dict[str, str]
) -> DependencyVerificationRecord:
    target_text = rendered_sections.get(obligation.matched_target_section_id)
    if target_text is None:
        return DependencyVerificationRecord(
            dependency_id=obligation.dependency_id,
            relation=obligation.relation,
            target_section_id=obligation.target_section_id,
            outcome=CheckOutcome.UNKNOWN,
            unknown_reason=UnknownReason.MISSING_DEPENDENCY,
            detail=f"no rendered text supplied for section {obligation.matched_target_section_id!r}.",
        )

    cited_lines = _cited_labels_with_lines(target_text)
    allowed: Set[str] = set(obligation.allowed_labels)
    matched_labels = sorted(set(cited_lines) & allowed)
    if matched_labels:
        return DependencyVerificationRecord(
            dependency_id=obligation.dependency_id,
            relation=obligation.relation,
            target_section_id=obligation.target_section_id,
            outcome=CheckOutcome.PASS,
            detail=f"cited {matched_labels} via a real reference command.",
            cited_labels_found=sorted(cited_lines),
            matched_at_line=cited_lines[matched_labels[0]],
        )
    return DependencyVerificationRecord(
        dependency_id=obligation.dependency_id,
        relation=obligation.relation,
        target_section_id=obligation.target_section_id,
        outcome=CheckOutcome.FAIL,
        detail=(
            f"no reference command in section {obligation.matched_target_section_id!r} cites any of "
            f"the allowed labels {sorted(allowed)}; a prose mention does not count "
            "(Gold Labeling sample_005)."
        ),
        cited_labels_found=sorted(cited_lines),
    )


def _verify_must_precede(obligation: CrossSectionObligation) -> DependencyVerificationRecord:
    if obligation.order_hint is None:
        return DependencyVerificationRecord(
            dependency_id=obligation.dependency_id,
            relation=obligation.relation,
            target_section_id=obligation.target_section_id,
            outcome=CheckOutcome.UNKNOWN,
            unknown_reason=UnknownReason.NOT_CHECKED,
            detail="order_hint missing -- source/target section not both found in blueprint.sections.",
        )
    source_idx, target_idx = obligation.order_hint
    if source_idx < target_idx:
        return DependencyVerificationRecord(
            dependency_id=obligation.dependency_id,
            relation=obligation.relation,
            target_section_id=obligation.target_section_id,
            outcome=CheckOutcome.PASS,
            detail=f"source section precedes target section in blueprint order ({source_idx} < {target_idx}).",
        )
    return DependencyVerificationRecord(
        dependency_id=obligation.dependency_id,
        relation=obligation.relation,
        target_section_id=obligation.target_section_id,
        outcome=CheckOutcome.FAIL,
        detail=(
            f"source section does not precede target section in blueprint order "
            f"({source_idx} >= {target_idx})."
        ),
    )


__all__ = [
    "DependencyVerificationRecord",
    "DependencyVerificationResult",
    "verify_cross_section_obligations",
]

"""``MathClaim`` schema for the "Revision (Additional verifiers)" plan's
Fix B, checklist item B1 ("검사 범위와 claim schema 고정") -- see
``docs/av_stage3_math_verifier.md`` for the MVP scope this schema is pinned
against.

A ``MathClaim`` is the structured record tying one real check's result
(B2's recompute-from-source, B3's shared-value consistency) back to
everything the plan's own text asks for: "block/artifact 버전, 원문 위치,
식 또는 연산, 관측값, 사용한 source 버전, 기호의 의미·scope, 가정, 단위와
허용 오차". It is a reporting/coverage structure, not a new checking
mechanism of its own -- the real checks still live in
``constraint_checks.BasicConstraintChecker`` (B2) and
``shared_value_registry`` (B3); this module only gives their results one
common, auditable shape, built from each check's REAL inputs/outputs
(never reverse-parsed from a free-text ``Evidence.detail`` string, which
would be guessing at what already-structured data the checker actually
had).

``MathCriterionId.LOGICAL_VALIDITY`` is declared but deliberately never
produced by any checker on this branch -- B6 (symbolic/logic review) is
explicit follow-up work, not started here. Keeping the criterion ID
declared now (rather than invented later) lets a future B6 claim reuse
this same coverage-reporting shape instead of a parallel one.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field

from src.core.schemas import ArtifactRef, CheckOutcome, UnknownReason

if TYPE_CHECKING:
    from src.core.shared_value_registry import SharedValueConsistencyRecord


class MathCriterionId(str, Enum):
    """"계산 정확성, 값 일관성, 논리 타당성에 서로 다른 criterion ID를
    부여한다" (B1's first bullet) -- these three are never collapsed into
    one pass/fail, since a document can be numerically correct but
    inconsistent, or consistent but wrong, or either while the argument
    itself is invalid."""

    CALCULATION_CORRECTNESS = "calculation_correctness"
    VALUE_CONSISTENCY = "value_consistency"
    LOGICAL_VALIDITY = "logical_validity"


class MathClaimKind(str, Enum):
    CALCULATION = "calculation"
    SHARED_VALUE = "shared_value"


class MathClaim(BaseModel):
    """One mathematical assertion this verifier actually evaluated (or
    explicitly could not). ``source_ref`` is ``None`` exactly when this
    claim has no computable source at all (an externally-asserted fact,
    e.g. "the population of X is 1000") -- distinct from a claim that DOES
    have a source but the recompute check still came out UNKNOWN for some
    other reason (bad JSON, missing artifact, unsupported operation); see
    ``MathClaimCoverageReport.externally_sourced_count``."""

    claim_id: str
    kind: MathClaimKind
    criterion_id: MathCriterionId
    artifact_id: str
    artifact_version: int
    raw_text: str = ""
    matched_at_line: Optional[int] = None
    operation: Optional[str] = None
    symbol: Optional[str] = None
    observed_value: Optional[float] = None
    expected_value: Optional[float] = None
    source_ref: Optional[ArtifactRef] = None
    scope: str = "global"
    unit: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)
    tolerance: Optional[float] = None
    outcome: CheckOutcome
    unknown_reason: Optional[UnknownReason] = None

    def model_post_init(self, __context: object) -> None:
        if self.outcome == CheckOutcome.UNKNOWN and self.unknown_reason is None:
            raise ValueError("MathClaim with outcome=UNKNOWN requires unknown_reason.")


class MathClaimCoverageReport(BaseModel):
    """Aggregate counts derived live from ``claims``, never a separately
    stored/maintained field -- the same "single source of truth" discipline
    ``DependencyVerificationResult`` already applies to C3's records."""

    claims: List[MathClaim] = Field(default_factory=list)

    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def checked_claims(self) -> int:
        return sum(1 for c in self.claims if c.outcome != CheckOutcome.UNKNOWN)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.claims if c.outcome == CheckOutcome.PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.claims if c.outcome == CheckOutcome.FAIL)

    @property
    def unknown_count(self) -> int:
        return sum(1 for c in self.claims if c.outcome == CheckOutcome.UNKNOWN)

    @property
    def externally_sourced_count(self) -> int:
        """Claims with no computable source at all -- distinct from a
        sourced claim that happened to come out UNKNOWN for some other
        reason (see this module's docstring)."""
        return sum(1 for c in self.claims if c.source_ref is None)


def math_claim_for_recompute(
    *,
    claim_id: str,
    artifact: ArtifactRef,
    operation: str,
    observed_value: float,
    expected_value: Optional[float],
    outcome: CheckOutcome,
    source_ref: Optional[ArtifactRef] = None,
    unknown_reason: Optional[UnknownReason] = None,
    scope: str = "global",
    unit: Optional[str] = None,
    assumptions: Optional[List[str]] = None,
    tolerance: Optional[float] = None,
    raw_text: str = "",
) -> MathClaim:
    """Builds a B2 (recompute) ``MathClaim`` from a check's own real
    inputs/outputs -- the caller already has every one of these values
    from the same call that produced the corresponding ``Evidence``
    (``constraint_checks.BasicConstraintChecker._check_recompute*``); this
    function never re-derives them by parsing that ``Evidence``'s
    free-text ``detail``."""
    return MathClaim(
        claim_id=claim_id,
        kind=MathClaimKind.CALCULATION,
        criterion_id=MathCriterionId.CALCULATION_CORRECTNESS,
        artifact_id=artifact.artifact_id,
        artifact_version=artifact.version,
        raw_text=raw_text,
        operation=operation,
        observed_value=observed_value,
        expected_value=expected_value,
        source_ref=source_ref,
        scope=scope,
        unit=unit,
        assumptions=list(assumptions or []),
        tolerance=tolerance,
        outcome=outcome,
        unknown_reason=unknown_reason,
    )


def math_claims_for_shared_value_record(record: "SharedValueConsistencyRecord") -> List[MathClaim]:
    """Builds B3 ``MathClaim`` entries, one per semantic_id the record
    actually checked (not per obligation) -- a dependency with several
    relevant symbols can have one PASS and one FAIL among them, and
    collapsing that into a single claim would lose exactly the kind of
    partial-result information Fix C's own per-target obligations were
    designed to preserve (``docs/av_stage2_dependency_mapping.md`` §2).
    A record with zero checked semantic_ids (the whole-obligation UNKNOWN
    case -- no relevant_symbols configured, or nothing restated anywhere)
    still produces exactly one claim, so the claim-level coverage total
    never silently drops an obligation that generated no per-symbol
    detail."""
    if not record.checked_semantic_ids:
        return [
            MathClaim(
                claim_id=f"{record.dependency_id}::{record.target_section_id}",
                kind=MathClaimKind.SHARED_VALUE,
                criterion_id=MathCriterionId.VALUE_CONSISTENCY,
                artifact_id=record.target_section_id,
                artifact_version=1,
                raw_text=record.detail,
                outcome=record.outcome,
                unknown_reason=record.unknown_reason,
            )
        ]

    definitions_by_id = {d.semantic_id: d for d in record.source_definitions}
    usages_by_id: dict = {}
    for usage in record.target_usages:
        usages_by_id.setdefault(usage.semantic_id, []).append(usage)

    claims: List[MathClaim] = []
    for semantic_id in record.checked_semantic_ids:
        definition = definitions_by_id.get(semantic_id)
        usages = usages_by_id.get(semantic_id, [])
        # A per-symbol verdict, not the obligation's aggregate outcome: a
        # dependency checking two symbols where only one mismatches must
        # still report the OTHER symbol's own claim as PASS, not drag it
        # down to the record's overall FAIL (the same "don't collapse a
        # per-target obligation's partial result" discipline C1/C3 already
        # apply one level up, applied here one level down to each symbol).
        own_mismatches = [m for m in record.mismatches if m.startswith(f"{semantic_id}:")]
        claim_outcome = CheckOutcome.FAIL if own_mismatches else CheckOutcome.PASS
        claims.append(
            MathClaim(
                claim_id=f"{record.dependency_id}::{record.target_section_id}::{semantic_id}",
                kind=MathClaimKind.SHARED_VALUE,
                criterion_id=MathCriterionId.VALUE_CONSISTENCY,
                artifact_id=record.target_section_id,
                artifact_version=1,
                raw_text=usages[0].raw_text if usages else (definition.raw_text if definition else ""),
                matched_at_line=usages[0].matched_at_line if usages else None,
                symbol=semantic_id,
                observed_value=usages[0].value if usages else None,
                expected_value=definition.value if definition else None,
                scope=definition.scope if definition else "global",
                unit=definition.unit if definition else None,
                assumptions=list(definition.assumptions) if definition else [],
                # checked_semantic_ids only ever contains a symbol once
                # both a source definition and a target usage were
                # actually found (check_shared_value_consistency's own
                # construction) -- so a per-symbol claim here is always
                # PASS or FAIL, never UNKNOWN.
                outcome=claim_outcome,
            )
        )
    return claims


__all__ = [
    "MathCriterionId",
    "MathClaimKind",
    "MathClaim",
    "MathClaimCoverageReport",
    "math_claim_for_recompute",
    "math_claims_for_shared_value_record",
]

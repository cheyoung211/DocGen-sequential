"""Coverage for src/core/math_claims.py (Revision "Additional verifiers"
plan, Fix B checklist item B1: MathClaim schema + coverage reporting)."""

from __future__ import annotations

import pytest

from src.core.math_claims import (
    MathClaim,
    MathClaimCoverageReport,
    MathClaimKind,
    MathCriterionId,
    math_claim_for_recompute,
    math_claims_for_shared_value_record,
)
from src.core.schemas import ArtifactKind, ArtifactRef, CheckOutcome, UnknownReason
from src.core.shared_value_registry import SharedValueCheckConfig, check_shared_value_consistency
from src.dataset.schemas import DependencyRelation


def test_unknown_claim_requires_unknown_reason():
    with pytest.raises(ValueError):
        MathClaim(
            claim_id="c1", kind=MathClaimKind.CALCULATION,
            criterion_id=MathCriterionId.CALCULATION_CORRECTNESS,
            artifact_id="a", artifact_version=1, outcome=CheckOutcome.UNKNOWN,
        )


def test_math_claim_for_recompute_builds_a_real_calculation_claim():
    source_ref = ArtifactRef(artifact_id="data_1", version=2, kind=ArtifactKind.SOURCE_DATA)
    artifact = ArtifactRef(artifact_id="sec_1", version=1, kind=ArtifactKind.SECTION)
    claim = math_claim_for_recompute(
        claim_id="r1", artifact=artifact, operation="sum", observed_value=35.5,
        expected_value=35.5, outcome=CheckOutcome.PASS, source_ref=source_ref, tolerance=0.01,
    )
    assert claim.criterion_id == MathCriterionId.CALCULATION_CORRECTNESS
    assert claim.kind == MathClaimKind.CALCULATION
    assert claim.source_ref == source_ref
    assert claim.artifact_version == 1


def test_externally_sourced_claim_has_no_source_ref():
    artifact = ArtifactRef(artifact_id="sec_1", version=1, kind=ArtifactKind.SECTION)
    claim = math_claim_for_recompute(
        claim_id="r1", artifact=artifact, operation="sum", observed_value=1.0,
        expected_value=None, outcome=CheckOutcome.UNKNOWN, unknown_reason=UnknownReason.NOT_CHECKED,
    )
    report = MathClaimCoverageReport(claims=[claim])
    assert report.externally_sourced_count == 1
    assert report.unknown_count == 1


def test_coverage_report_counts_are_derived_not_separately_maintained():
    artifact = ArtifactRef(artifact_id="sec_1", version=1, kind=ArtifactKind.SECTION)
    source_ref = ArtifactRef(artifact_id="data_1", version=1, kind=ArtifactKind.SOURCE_DATA)
    passing = math_claim_for_recompute(
        claim_id="p", artifact=artifact, operation="sum", observed_value=1.0,
        expected_value=1.0, outcome=CheckOutcome.PASS, source_ref=source_ref,
    )
    failing = math_claim_for_recompute(
        claim_id="f", artifact=artifact, operation="sum", observed_value=2.0,
        expected_value=1.0, outcome=CheckOutcome.FAIL, source_ref=source_ref,
    )
    report = MathClaimCoverageReport(claims=[passing, failing])
    assert report.total_claims == 2
    assert report.checked_claims == 2
    assert report.pass_count == 1
    assert report.fail_count == 1
    assert report.unknown_count == 0


def test_shared_value_record_with_no_checked_symbols_still_produces_one_claim():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 1", target_text="nothing numeric",
        config=SharedValueCheckConfig(relevant_symbols=["n"]),
    )
    claims = math_claims_for_shared_value_record(record)
    assert len(claims) == 1
    assert claims[0].outcome == CheckOutcome.UNKNOWN
    assert claims[0].criterion_id == MathCriterionId.VALUE_CONSISTENCY


def test_shared_value_record_produces_one_claim_per_checked_symbol_with_per_symbol_outcome():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10, k = 25", target_text="n = 10, k = 10",
        config=SharedValueCheckConfig(relevant_symbols=["n", "k"]),
    )
    claims = math_claims_for_shared_value_record(record)
    by_symbol = {c.symbol: c for c in claims}
    assert by_symbol["n"].outcome == CheckOutcome.PASS
    assert by_symbol["k"].outcome == CheckOutcome.FAIL
    assert all(c.criterion_id == MathCriterionId.VALUE_CONSISTENCY for c in claims)


def test_shared_value_claim_version_defaults_to_unknown_not_a_fabricated_one():
    # Regression: artifact_version used to be hardcoded to 1 regardless
    # of the real target section's version.
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10", target_text="n = 10",
        config=SharedValueCheckConfig(relevant_symbols=["n"]),
    )
    claims = math_claims_for_shared_value_record(record)
    assert claims[0].artifact_version is None
    assert claims[0].source_ref is None


def test_shared_value_claim_carries_the_real_target_and_source_refs_when_supplied():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10", target_text="n = 10",
        config=SharedValueCheckConfig(relevant_symbols=["n"]),
    )
    target_ref = ArtifactRef(artifact_id="b", version=3, kind=ArtifactKind.SECTION)
    source_ref = ArtifactRef(artifact_id="a", version=5, kind=ArtifactKind.SECTION)
    claims = math_claims_for_shared_value_record(record, target_ref=target_ref, source_ref=source_ref)
    assert claims[0].artifact_id == "b"
    assert claims[0].artifact_version == 3
    assert claims[0].source_ref == source_ref


def test_shared_value_claim_without_a_source_ref_is_not_counted_as_externally_sourced():
    # A SHARED_VALUE claim always has a real source SECTION by
    # construction -- a caller simply not passing the resolved ref is a
    # reporting gap, not the same fact as a CALCULATION claim with no
    # computable source at all.
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10", target_text="n = 10",
        config=SharedValueCheckConfig(relevant_symbols=["n"]),
    )
    claims = math_claims_for_shared_value_record(record)  # no source_ref supplied
    report = MathClaimCoverageReport(claims=claims)
    assert report.externally_sourced_count == 0


def test_logical_validity_criterion_is_declared_but_never_auto_produced():
    # B6 (symbolic/logic review) is explicit follow-up, not started on
    # this branch -- nothing in B1-B3's own builders ever stamps this
    # criterion, which this test pins down so a future accidental use
    # doesn't go unnoticed.
    assert MathCriterionId.LOGICAL_VALIDITY.value == "logical_validity"
    source_ref = ArtifactRef(artifact_id="data_1", version=1, kind=ArtifactKind.SOURCE_DATA)
    artifact = ArtifactRef(artifact_id="sec_1", version=1, kind=ArtifactKind.SECTION)
    recompute_claim = math_claim_for_recompute(
        claim_id="r1", artifact=artifact, operation="sum", observed_value=1.0,
        expected_value=1.0, outcome=CheckOutcome.PASS, source_ref=source_ref,
    )
    assert recompute_claim.criterion_id != MathCriterionId.LOGICAL_VALIDITY

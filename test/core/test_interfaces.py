"""Coverage for src/core/schemas.py and the Protocol conformance of
src/core/interfaces.py (Stage 1 of the "Revision 2026.10.15" plan)."""

from __future__ import annotations

import pytest

from src.core.interfaces import FinalEvaluator, Producer, RepairPolicy, Validator, ValidatorConfig
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    CheckOutcome,
    CostRecord,
    Evidence,
    FinalEvaluationResult,
    ProductionResult,
    ProductionStatus,
    QualityScore,
    UnknownReason,
)


def test_artifact_ref_round_trips_nested_provenance() -> None:
    upstream = ArtifactRef(artifact_id="def_1", kind=ArtifactKind.DEFINITION)
    downstream = ArtifactRef(
        artifact_id="thm_1", kind=ArtifactKind.CLAIM, read_input_versions=[upstream]
    )
    rebuilt = ArtifactRef.model_validate(downstream.model_dump())
    assert rebuilt.read_input_versions[0].artifact_id == "def_1"


def test_evidence_unknown_outcome_requires_a_reason() -> None:
    artifact = ArtifactRef(artifact_id="eq_1", kind=ArtifactKind.EQUATION)
    with pytest.raises(ValueError):
        Evidence(constraint_id="c1", outcome=CheckOutcome.UNKNOWN, checked_artifact=artifact)

    # Supplying a reason is accepted.
    Evidence(
        constraint_id="c1",
        outcome=CheckOutcome.UNKNOWN,
        checked_artifact=artifact,
        unknown_reason=UnknownReason.SOLVER_TIMEOUT,
    )


def test_evidence_pass_fail_do_not_require_a_reason() -> None:
    artifact = ArtifactRef(artifact_id="eq_1", kind=ArtifactKind.EQUATION)
    Evidence(constraint_id="c1", outcome=CheckOutcome.PASS, checked_artifact=artifact)
    Evidence(constraint_id="c1", outcome=CheckOutcome.FAIL, checked_artifact=artifact)


def test_production_result_enforces_status_artifact_consistency() -> None:
    artifact = ArtifactRef(artifact_id="sec_intro", kind=ArtifactKind.SECTION)

    # A controller must never be able to construct (or receive) a result
    # that claims success with no artifact, or failure with one attached.
    with pytest.raises(ValueError):
        ProductionResult(status=ProductionStatus.SUCCESS, artifact=None)
    with pytest.raises(ValueError):
        ProductionResult(status=ProductionStatus.FAILED, artifact=artifact)

    ProductionResult(status=ProductionStatus.SUCCESS, artifact=artifact)
    ProductionResult(status=ProductionStatus.FAILED, error="exhausted retries")


def test_budget_remaining_and_charge_are_consistent() -> None:
    budget = Budget(cap=100.0)
    assert budget.remaining == 100.0

    charged = budget.charge(CostRecord(stage="text_agent", total_tokens=40))
    # charge() is non-mutating -- the original instance is untouched.
    assert budget.remaining == 100.0
    assert charged.remaining == 60.0


def test_protocols_are_satisfied_by_minimal_duck_typed_classes() -> None:
    """Exercises each Protocol with the smallest possible conforming class,
    proving the interfaces themselves impose no hidden coupling to any one
    concrete implementation (in particular, not to the adapters in
    adapters.py)."""

    class MinimalProducer:
        def produce(self, artifact_id, context):
            return ProductionResult(
                status=ProductionStatus.SUCCESS,
                artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.BLOCK),
            )

    class MinimalValidator:
        config = ValidatorConfig({"check_a": True})

        def check(self, artifact, constraints, context=None):
            return [Evidence(constraint_id="check_a", outcome=CheckOutcome.PASS, checked_artifact=artifact)]

    class MinimalRepairPolicy:
        def decide(self, evidence, dependency_edges, budget):
            return []

    class MinimalFinalEvaluator:
        def evaluate(self, final_artifacts, benchmark_contract):
            return FinalEvaluationResult(quality=QualityScore())

    assert isinstance(MinimalProducer(), Producer)
    assert isinstance(MinimalValidator(), Validator)
    assert isinstance(MinimalRepairPolicy(), RepairPolicy)
    assert isinstance(MinimalFinalEvaluator(), FinalEvaluator)

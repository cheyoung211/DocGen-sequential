"""Coverage for src/core/repair_operations.py (Stage 4): the cascading
regenerate-or-reaffirm executor for all four RepairActionTypes, and the two
fixed RepairPolicy implementations ("fixed graph-aware" / "whole-document
regeneration"). Uses Protocol-conforming fakes throughout -- no real LLM
call anywhere in this file, per this stage's explicit scope decision (see
repair_operations.py's module docstring)."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional

from src.core.artifact_graph import ArtifactGraphStore
from src.core.repair_operations import (
    AlwaysRegenerateDocumentPolicy,
    FixedGraphAwareRepairPolicy,
    RepairOperationExecutor,
)
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    CheckOutcome,
    Constraint,
    ConstraintKind,
    DependencyEdge,
    DependencyRelationKind,
    Evidence,
    ProductionResult,
    ProductionStatus,
    RepairAction,
    RepairActionType,
    RepairOutcome,
)


class _FakeProducer:
    """Always SUCCEEDS unless the artifact_id is in `fail_ids` -- not a
    CostTrackingProducer, so the executor's cost-capture branch is simply a
    no-op for it (covered separately by a dedicated cost test)."""

    def __init__(self, fail_ids: FrozenSet[str] = frozenset()) -> None:
        self.fail_ids = set(fail_ids)
        self.calls: List[str] = []

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ProductionResult:
        self.calls.append(artifact_id)
        if artifact_id in self.fail_ids:
            return ProductionResult(status=ProductionStatus.FAILED, error="fake production failure")
        return ProductionResult(
            status=ProductionStatus.SUCCESS, artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.BLOCK)
        )


class _FakeValidator:
    """outcome depends on membership in `fail_ids` (mutable -- a test can
    edit it mid-run to simulate "this was fixed, recheck now passes")."""

    def __init__(self, fail_ids: FrozenSet[str] = frozenset()) -> None:
        self.fail_ids = set(fail_ids)
        self.config: Dict[str, bool] = {}
        self.calls: List[str] = []

    def check(
        self, artifact: ArtifactRef, constraints: List[Constraint], context: Optional[Dict[str, Any]] = None
    ) -> List[Evidence]:
        self.calls.append(artifact.artifact_id)
        outcome = CheckOutcome.FAIL if artifact.artifact_id in self.fail_ids else CheckOutcome.PASS
        return [
            Evidence(constraint_id=c.constraint_id, outcome=outcome, checked_artifact=artifact, detail="fake")
            for c in constraints
        ]


def _one_constraint_per_artifact(no_constraints_for: FrozenSet[str] = frozenset()):
    def provider(ref: ArtifactRef) -> List[Constraint]:
        if ref.artifact_id in no_constraints_for:
            return []
        return [Constraint(constraint_id=f"c:{ref.artifact_id}", kind=ConstraintKind.SCHEMA_CONTRACT, description="d")]

    return provider


def _store_with_chain(store: ArtifactGraphStore, *ids_and_kinds: tuple) -> None:
    """Build a straight-line dependency chain: ids_and_kinds[0] has no
    input; each subsequent one reads the previous one as its only input."""
    previous_ref = None
    for artifact_id, kind in ids_and_kinds:
        inputs = [previous_ref] if previous_ref is not None else []
        result = store.add_version(artifact_id, kind, content=None, read_input_versions=inputs)
        previous_ref = result.ref


def _executor(
    producers: Dict[ArtifactKind, Any],
    validator: _FakeValidator,
    *,
    no_constraints_for: FrozenSet[str] = frozenset(),
) -> RepairOperationExecutor:
    return RepairOperationExecutor(
        producer_registry=producers,
        validator=validator,
        constraint_provider=_one_constraint_per_artifact(no_constraints_for),
        base_context={},
    )


def _action(action_type: RepairActionType, targets: List[ArtifactRef], feedback: bool = True) -> RepairAction:
    return RepairAction(action_type=action_type, target_artifacts=targets, use_diagnostic_feedback=feedback)


# ---------------------------------------------------------------------------
# RepairOperationExecutor -- the cascade.
# ---------------------------------------------------------------------------


def test_regenerates_only_the_target_when_nothing_downstream_exists() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("blkA", ArtifactKind.BLOCK))
    producer = _FakeProducer()
    # The seed itself is already known-failing from the triggering evidence
    # (it is a directly-named target, not discovered via cascade) -- the
    # validator is only consulted for the *post-regen* recheck here, so an
    # empty fail_ids models "the regeneration actually fixed it".
    validator = _FakeValidator(fail_ids=frozenset())
    executor = _executor({ArtifactKind.BLOCK: producer}, validator)

    result = executor.execute(_action(RepairActionType.PATCH_BLOCK, [store.latest("blkA").ref]), store)

    assert len(result.regenerated) == 1
    assert result.regenerated[0].outcome == RepairOutcome.RESOLVED
    assert result.reverified_untouched == []
    assert result.overall_outcome == RepairOutcome.RESOLVED
    assert producer.calls == ["blkA"]


def test_cascades_and_reaffirms_a_passing_descendant_leaving_it_fresh() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("secA", ArtifactKind.SECTION), ("secB", ArtifactKind.SECTION))
    assert store.is_stale("secB") is False  # not stale yet -- secA hasn't changed since secB was built

    producer = _FakeProducer()
    # secA fails; secB will pass its fresh re-check (never added to fail_ids).
    validator = _FakeValidator(fail_ids={"secA"})
    executor = _executor({ArtifactKind.SECTION: producer}, validator)

    result = executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("secA").ref]), store)

    regenerated_ids = {r.pre_repair_ref.artifact_id for r in result.regenerated}
    assert regenerated_ids == {"secA"}  # secB was NOT regenerated -- "필요한 항목만 재생성"
    reverified_ids = {r.artifact_id for r in result.reverified_untouched}
    assert reverified_ids == {"secB"}
    assert producer.calls == ["secA"]  # producer never called for secB
    # secB was reaffirmed (a new version was written, re-pinned to secA's
    # current version) -- its staleness, which add_version's own forward
    # propagation set when secA moved, is now cleared.
    assert store.is_stale("secB") is False
    assert store.latest("secB").ref.version == 2


def test_a_descendant_that_still_fails_after_reverify_gets_regenerated_and_reported() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("secA", ArtifactKind.SECTION), ("secB", ArtifactKind.SECTION))

    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids={"secA", "secB"})  # secB never clears, even after being regenerated
    executor = _executor({ArtifactKind.SECTION: producer}, validator)

    result = executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("secA").ref]), store)

    regenerated_ids = {r.pre_repair_ref.artifact_id for r in result.regenerated}
    assert regenerated_ids == {"secA", "secB"}
    secB_record = next(r for r in result.regenerated if r.pre_repair_ref.artifact_id == "secB")
    assert secB_record.outcome == RepairOutcome.STILL_FAILING
    assert result.overall_outcome == RepairOutcome.STILL_FAILING
    assert "secB" in producer.calls  # secB really was regenerated, not just reaffirmed


def test_production_failure_does_not_crash_and_does_not_cascade() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("secA", ArtifactKind.SECTION), ("secB", ArtifactKind.SECTION))

    producer = _FakeProducer(fail_ids={"secA"})
    validator = _FakeValidator(fail_ids={"secA"})
    executor = _executor({ArtifactKind.SECTION: producer}, validator)

    result = executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("secA").ref]), store)

    assert len(result.regenerated) == 1
    assert result.regenerated[0].outcome == RepairOutcome.PRODUCTION_FAILED
    assert result.regenerated[0].new_ref is None
    assert result.reverified_untouched == []  # never got far enough to look at secB
    assert result.overall_outcome == RepairOutcome.PRODUCTION_FAILED
    # The store itself was never actually changed for secA.
    assert store.latest("secA").ref.version == 1


def test_no_producer_registered_for_a_kind_is_reported_honestly() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("def1", ArtifactKind.DEFINITION))
    validator = _FakeValidator(fail_ids={"def1"})
    executor = _executor({}, validator)  # nothing registered for any kind

    result = executor.execute(_action(RepairActionType.REPLAN_SUBGRAPH, [store.latest("def1").ref]), store)

    assert len(result.regenerated) == 1
    assert result.regenerated[0].outcome == RepairOutcome.NO_PRODUCER
    assert result.regenerated[0].new_ref is None
    assert result.overall_outcome == RepairOutcome.NO_PRODUCER


def test_a_candidate_with_no_constraints_is_skipped_without_losing_further_reachability() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(
        store,
        ("secA", ArtifactKind.SECTION),
        ("secB", ArtifactKind.SECTION),
        ("secC", ArtifactKind.SECTION),
    )
    producer = _FakeProducer()
    # secB has no constraints registered at all; secC does, and currently fails.
    validator = _FakeValidator(fail_ids={"secA", "secC"})
    executor = _executor({ArtifactKind.SECTION: producer}, validator, no_constraints_for=frozenset({"secB"}))

    result = executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("secA").ref]), store)

    touched_ids = {r.pre_repair_ref.artifact_id for r in result.regenerated} | {
        r.artifact_id for r in result.reverified_untouched
    }
    assert "secB" not in touched_ids  # skipped: nothing to check it against
    assert "secC" in {r.pre_repair_ref.artifact_id for r in result.regenerated}  # still reached despite secB's skip
    assert "secB" not in validator.calls


def test_execute_logs_the_action_exactly_once_regardless_of_cascade_size() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("secA", ArtifactKind.SECTION), ("secB", ArtifactKind.SECTION))
    executor = _executor({ArtifactKind.SECTION: _FakeProducer()}, _FakeValidator(fail_ids={"secA", "secB"}))

    action = _action(RepairActionType.REGENERATE_SECTION, [store.latest("secA").ref])
    trigger = [Evidence(constraint_id="x", outcome=CheckOutcome.FAIL, checked_artifact=store.latest("secA").ref)]
    executor.execute(action, store, triggering_evidence=trigger)

    assert len(store.log.actions) == 1
    assert store.log.actions[0].action.action_type == RepairActionType.REGENERATE_SECTION


# ---------------------------------------------------------------------------
# FixedGraphAwareRepairPolicy
# ---------------------------------------------------------------------------


def _fail_evidence(ref: ArtifactRef, constraint_id: str = "c") -> Evidence:
    return Evidence(constraint_id=constraint_id, outcome=CheckOutcome.FAIL, checked_artifact=ref, detail="bad")


def test_graph_aware_policy_picks_patch_block_for_a_failing_block() -> None:
    block_ref = ArtifactRef(artifact_id="blkA", kind=ArtifactKind.BLOCK)
    policy = FixedGraphAwareRepairPolicy(document_ref=ArtifactRef(artifact_id="doc", kind=ArtifactKind.DOCUMENT))

    actions = policy.decide([_fail_evidence(block_ref)], [], Budget(cap=100))

    assert len(actions) == 1
    assert actions[0].action_type == RepairActionType.PATCH_BLOCK
    assert actions[0].target_artifacts == [block_ref]


def test_graph_aware_policy_picks_regenerate_section_for_a_failing_section() -> None:
    section_ref = ArtifactRef(artifact_id="secA", kind=ArtifactKind.SECTION)
    policy = FixedGraphAwareRepairPolicy(document_ref=ArtifactRef(artifact_id="doc", kind=ArtifactKind.DOCUMENT))

    actions = policy.decide([_fail_evidence(section_ref)], [], Budget(cap=100))

    assert actions[0].action_type == RepairActionType.REGENERATE_SECTION


def test_graph_aware_policy_picks_subgraph_replan_for_a_shared_definition_with_fan_out() -> None:
    def_ref = ArtifactRef(artifact_id="def1", kind=ArtifactKind.DEFINITION)
    sec_a = ArtifactRef(artifact_id="secA", kind=ArtifactKind.SECTION)
    sec_b = ArtifactRef(artifact_id="secB", kind=ArtifactKind.SECTION)

    edges = [
        DependencyEdge(from_artifact=def_ref, to_artifact=sec_a, relation=DependencyRelationKind.EXPLICIT),
        DependencyEdge(from_artifact=def_ref, to_artifact=sec_b, relation=DependencyRelationKind.EXPLICIT),
    ]
    policy = FixedGraphAwareRepairPolicy(document_ref=ArtifactRef(artifact_id="doc", kind=ArtifactKind.DOCUMENT))

    actions = policy.decide([_fail_evidence(def_ref)], edges, Budget(cap=100))

    assert actions[0].action_type == RepairActionType.REPLAN_SUBGRAPH
    assert {t.artifact_id for t in actions[0].target_artifacts} == {"def1", "secA", "secB"}


def test_graph_aware_policy_falls_back_to_regenerate_document_with_no_useful_fan_out() -> None:
    def_ref = ArtifactRef(artifact_id="def1", kind=ArtifactKind.DEFINITION)
    document_ref = ArtifactRef(artifact_id="doc", kind=ArtifactKind.DOCUMENT)
    policy = FixedGraphAwareRepairPolicy(document_ref=document_ref)

    actions = policy.decide([_fail_evidence(def_ref)], [], Budget(cap=100))

    assert actions[0].action_type == RepairActionType.REGENERATE_DOCUMENT
    assert actions[0].target_artifacts == [document_ref]


def test_graph_aware_policy_returns_one_action_per_distinct_failing_artifact() -> None:
    block_ref = ArtifactRef(artifact_id="blkA", kind=ArtifactKind.BLOCK)
    section_ref = ArtifactRef(artifact_id="secA", kind=ArtifactKind.SECTION)
    policy = FixedGraphAwareRepairPolicy(document_ref=ArtifactRef(artifact_id="doc", kind=ArtifactKind.DOCUMENT))

    actions = policy.decide(
        [_fail_evidence(block_ref), _fail_evidence(section_ref)], [], Budget(cap=100)
    )

    assert {a.action_type for a in actions} == {RepairActionType.PATCH_BLOCK, RepairActionType.REGENERATE_SECTION}
    assert len(actions) == 2


def test_graph_aware_policy_respects_budget_exhaustion() -> None:
    block_ref = ArtifactRef(artifact_id="blkA", kind=ArtifactKind.BLOCK)
    policy = FixedGraphAwareRepairPolicy(document_ref=ArtifactRef(artifact_id="doc", kind=ArtifactKind.DOCUMENT))

    actions = policy.decide([_fail_evidence(block_ref)], [], Budget(cap=10, spent=10))

    assert actions == []


# ---------------------------------------------------------------------------
# AlwaysRegenerateDocumentPolicy
# ---------------------------------------------------------------------------


def test_always_regenerate_document_fires_on_any_failure() -> None:
    block_ref = ArtifactRef(artifact_id="blkA", kind=ArtifactKind.BLOCK)
    document_ref = ArtifactRef(artifact_id="doc", kind=ArtifactKind.DOCUMENT)
    policy = AlwaysRegenerateDocumentPolicy(document_ref=document_ref)

    actions = policy.decide([_fail_evidence(block_ref)], [], Budget(cap=100))

    assert len(actions) == 1
    assert actions[0].action_type == RepairActionType.REGENERATE_DOCUMENT
    assert actions[0].target_artifacts == [document_ref]
    assert actions[0].use_diagnostic_feedback is False


def test_always_regenerate_document_does_nothing_with_no_failures() -> None:
    document_ref = ArtifactRef(artifact_id="doc", kind=ArtifactKind.DOCUMENT)
    policy = AlwaysRegenerateDocumentPolicy(document_ref=document_ref)

    pass_evidence = Evidence(
        constraint_id="c",
        outcome=CheckOutcome.PASS,
        checked_artifact=ArtifactRef(artifact_id="blkA", kind=ArtifactKind.BLOCK),
    )

    assert policy.decide([pass_evidence], [], Budget(cap=100)) == []


def test_always_regenerate_document_respects_budget_exhaustion() -> None:
    document_ref = ArtifactRef(artifact_id="doc", kind=ArtifactKind.DOCUMENT)
    policy = AlwaysRegenerateDocumentPolicy(document_ref=document_ref)
    block_ref = ArtifactRef(artifact_id="blkA", kind=ArtifactKind.BLOCK)

    actions = policy.decide([_fail_evidence(block_ref)], [], Budget(cap=10, spent=10))

    assert actions == []

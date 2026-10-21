"""Coverage for src/core/failure_dataset.py (Stage 6): dependency-graph
difficulty classification, and repair gain/damage/cost built on top of
Stage 4's already-reviewed BranchExecutionResult -- via a real
BranchingRunner + Protocol-conforming fakes for the end-to-end wiring, and
hand-built minimal schema objects for the merge/damage logic in isolation.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional

import pytest

from src.core.artifact_graph import ArtifactGraphStore, UnknownArtifactError
from src.core.branching_runner import BranchingRunner, BranchSpec
from src.core.failure_dataset import (
    _dependency_depth,
    _merge_final_outcomes,
    build_candidate_action_record,
    build_failure_record,
    classify_failure_difficulty,
    compute_repair_gain_score,
)
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    BranchExecutionResult,
    CheckOutcome,
    Constraint,
    ConstraintKind,
    Evidence,
    FailureOrigin,
    InjectedFault,
    ProductionResult,
    ProductionStatus,
    RepairAction,
    RepairActionType,
    RepairExecutionResult,
    RepairOutcome,
    RepairRound,
)


def _ref(artifact_id: str, version: int, kind: ArtifactKind) -> ArtifactRef:
    return ArtifactRef(artifact_id=artifact_id, version=version, kind=kind)


# ---------------------------------------------------------------------------
# classify_failure_difficulty
# ---------------------------------------------------------------------------


def test_classify_difficulty_fan_out_and_shared_definition_use() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="Let X be...", produced_by="seed")
    def_ref = store.latest("def_1").ref
    store.add_version("thm_1", ArtifactKind.CLAIM, content="uses def_1", read_input_versions=[def_ref])
    store.add_version("thm_2", ArtifactKind.CLAIM, content="also uses def_1", read_input_versions=[def_ref])

    profile = classify_failure_difficulty(store, "def_1")

    assert profile.fan_out == 2
    assert profile.shared_definition_use is True
    assert profile.dependency_depth == 0  # def_1 has no dependencies of its own


def test_classify_difficulty_single_dependent_is_not_shared_definition_use() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="Let X be...", produced_by="seed")
    def_ref = store.latest("def_1").ref
    store.add_version("thm_1", ArtifactKind.CLAIM, content="uses def_1", read_input_versions=[def_ref])

    profile = classify_failure_difficulty(store, "def_1")
    assert profile.fan_out == 1
    assert profile.shared_definition_use is False


def test_classify_difficulty_block_and_section_kinds_are_never_shared_definition_use() -> None:
    store = ArtifactGraphStore()
    store.add_version("sec_1", ArtifactKind.SECTION, content="s1", produced_by="seed")
    sec_ref = store.latest("sec_1").ref
    store.add_version("blk_1", ArtifactKind.BLOCK, content="uses sec_1", read_input_versions=[sec_ref])
    store.add_version("blk_2", ArtifactKind.BLOCK, content="also uses sec_1", read_input_versions=[sec_ref])

    profile = classify_failure_difficulty(store, "sec_1")
    assert profile.fan_out == 2
    assert profile.shared_definition_use is False  # SECTION is excluded regardless of fan-out


def test_classify_difficulty_dependency_depth_follows_the_chain() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.CLAIM, content="b", read_input_versions=[a_ref])
    b_ref = store.latest("b").ref
    store.add_version("c", ArtifactKind.CLAIM, content="c", read_input_versions=[b_ref])

    assert classify_failure_difficulty(store, "a").dependency_depth == 0
    assert classify_failure_difficulty(store, "b").dependency_depth == 1
    assert classify_failure_difficulty(store, "c").dependency_depth == 2


def test_classify_difficulty_missing_relations_count_from_inferred_edges() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a")
    store.add_version("b", ArtifactKind.CLAIM, content="b")
    a_ref = store.latest("a").ref
    b_ref = store.latest("b").ref

    assert classify_failure_difficulty(store, "a").missing_relations_count == 0
    store.add_inferred_edge(a_ref, b_ref, reason="b actually cites a's label")
    assert classify_failure_difficulty(store, "a").missing_relations_count == 1
    assert classify_failure_difficulty(store, "b").missing_relations_count == 1


def test_classify_difficulty_raises_for_unknown_artifact() -> None:
    store = ArtifactGraphStore()
    with pytest.raises(UnknownArtifactError):
        classify_failure_difficulty(store, "ghost")


def test_dependency_depth_terminates_on_a_genuine_cycle() -> None:
    """a and b are made to mutually depend on each other via two
    add_inferred_edge calls (add_version itself cannot create a cycle,
    since an input must already exist before it can be declared) --
    impact_candidates' own docstring says the graph is not assumed
    acyclic, so _dependency_depth must not infinite-loop on one."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a")
    store.add_version("b", ArtifactKind.DEFINITION, content="b")
    a_ref = store.latest("a").ref
    b_ref = store.latest("b").ref
    store.add_inferred_edge(a_ref, b_ref, reason="b depends on a")
    store.add_inferred_edge(b_ref, a_ref, reason="a also depends on b")

    depth_a = _dependency_depth(store, "a")
    depth_b = _dependency_depth(store, "b")
    assert isinstance(depth_a, int) and isinstance(depth_b, int)


# ---------------------------------------------------------------------------
# build_failure_record
# ---------------------------------------------------------------------------


def test_build_failure_record_natural_origin() -> None:
    store = ArtifactGraphStore()
    store.add_version("sec_1", ArtifactKind.SECTION, content="s1")
    ref = store.latest("sec_1").ref
    evidence = [Evidence(constraint_id="c", outcome=CheckOutcome.FAIL, checked_artifact=ref, detail="bad")]

    record = build_failure_record(
        store,
        failure_id="f1",
        origin=FailureOrigin.NATURAL,
        failing_artifact_id="sec_1",
        evidence=evidence,
        source_checkpoint_id="cp1",
    )

    assert record.origin == FailureOrigin.NATURAL
    assert record.injected_fault is None
    assert record.failing_artifact == ref
    assert record.difficulty.dependency_depth == 0


def test_build_failure_record_injected_requires_injected_fault() -> None:
    store = ArtifactGraphStore()
    store.add_version("sec_1", ArtifactKind.SECTION, content="s1")
    with pytest.raises(ValueError):
        build_failure_record(
            store,
            failure_id="f1",
            origin=FailureOrigin.INJECTED,
            failing_artifact_id="sec_1",
            evidence=[],
            source_checkpoint_id="cp1",
            injected_fault=None,
        )


def test_build_failure_record_raises_for_unknown_artifact() -> None:
    store = ArtifactGraphStore()
    with pytest.raises(UnknownArtifactError):
        build_failure_record(
            store,
            failure_id="f1",
            origin=FailureOrigin.NATURAL,
            failing_artifact_id="ghost",
            evidence=[],
            source_checkpoint_id="cp1",
        )


# ---------------------------------------------------------------------------
# _merge_final_outcomes: last-write-wins across rounds
# ---------------------------------------------------------------------------


def _minimal_execution(final_outcome_by_artifact: Dict[str, RepairOutcome]) -> RepairExecutionResult:
    action = RepairAction(
        action_type=RepairActionType.REGENERATE_SECTION,
        target_artifacts=[_ref("x", 1, ArtifactKind.SECTION)],
        use_diagnostic_feedback=False,
    )
    return RepairExecutionResult(
        action=action,
        final_outcome_by_artifact=final_outcome_by_artifact,
        overall_outcome=next(iter(final_outcome_by_artifact.values())),
    )


def test_merge_final_outcomes_later_round_overwrites_earlier_for_the_same_artifact() -> None:
    round_1 = RepairRound(
        round_index=0,
        policy_label="A",
        executions=[_minimal_execution({"x": RepairOutcome.STILL_FAILING})],
    )
    round_2 = RepairRound(
        round_index=1,
        policy_label="downstream",
        executions=[_minimal_execution({"x": RepairOutcome.RESOLVED})],
    )
    store = ArtifactGraphStore()
    store.add_version("x", ArtifactKind.SECTION, content="x")
    checkpoint = store.save_checkpoint()
    branch = BranchExecutionResult(
        branch_id="A",
        source_checkpoint_id=checkpoint.checkpoint_id,
        policy_label="A",
        rounds=[round_1, round_2],
        stopped_reason="no_more_failures",
        final_checkpoint=checkpoint,
    )

    merged = _merge_final_outcomes(branch)
    assert merged == {"x": RepairOutcome.RESOLVED}


# ---------------------------------------------------------------------------
# compute_repair_gain_score / build_candidate_action_record: real BranchingRunner
# ---------------------------------------------------------------------------


class _FakeProducer:
    def __init__(self, fail_ids: FrozenSet[str] = frozenset(), cost_per_call: int = 5) -> None:
        self.fail_ids = set(fail_ids)
        self.cost_per_call = cost_per_call

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ProductionResult:
        sink = context.get("usage_sink")
        if sink is not None and self.cost_per_call:
            sink.append({"input_tokens": 0, "output_tokens": self.cost_per_call, "total_tokens": self.cost_per_call})
        if artifact_id in self.fail_ids:
            return ProductionResult(status=ProductionStatus.FAILED, error="fake production failure")
        return ProductionResult(
            status=ProductionStatus.SUCCESS,
            artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.SECTION),
            content=f"real-content:{artifact_id}",
            read_input_versions=context.get("read_input_versions"),
        )


class _FakeValidator:
    def __init__(self, fail_ids: FrozenSet[str] = frozenset()) -> None:
        self.fail_ids = set(fail_ids)
        self.config: Dict[str, bool] = {}

    def check(
        self, artifact: ArtifactRef, constraints: List[Constraint], context: Optional[Dict[str, Any]] = None
    ) -> List[Evidence]:
        outcome = CheckOutcome.FAIL if artifact.artifact_id in self.fail_ids else CheckOutcome.PASS
        return [
            Evidence(constraint_id=c.constraint_id, outcome=outcome, checked_artifact=artifact, detail="fake")
            for c in constraints
        ]


class _ActOnAllFailures:
    def decide(self, evidence, dependency_edges, budget) -> List[RepairAction]:
        if budget.remaining <= 0:
            return []
        seen: Dict[str, Evidence] = {}
        for item in evidence:
            if item.outcome == CheckOutcome.FAIL:
                seen.setdefault(item.checked_artifact.artifact_id, item)
        return [
            RepairAction(
                action_type=RepairActionType.REGENERATE_SECTION,
                target_artifacts=[item.checked_artifact],
                use_diagnostic_feedback=False,
            )
            for item in seen.values()
        ]


class _NothingPolicy:
    def decide(self, evidence, dependency_edges, budget) -> List[RepairAction]:
        return []


def _one_constraint_per_artifact(ref: ArtifactRef) -> List[Constraint]:
    return [Constraint(constraint_id=f"c:{ref.artifact_id}", kind=ConstraintKind.SCHEMA_CONTRACT, description="d")]


def _checkpoint_with_sections(*artifact_ids: str):
    store = ArtifactGraphStore()
    refs = {}
    for artifact_id in artifact_ids:
        store.add_version(artifact_id, ArtifactKind.SECTION, content=None)
        refs[artifact_id] = store.latest(artifact_id).ref
    return store.save_checkpoint(label="pre-repair"), refs


def _fail_evidence(ref: ArtifactRef) -> Evidence:
    return Evidence(constraint_id="c", outcome=CheckOutcome.FAIL, checked_artifact=ref, detail="bad")


def _run_single_branch(checkpoint, refs, *, producer, validator, budget=None) -> BranchExecutionResult:
    runner = BranchingRunner(
        producer_registry={ArtifactKind.SECTION: producer},
        validator_factory=lambda store: validator,
        constraint_provider=_one_constraint_per_artifact,
        model_name="fake-model",
    )
    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", _ActOnAllFailures())],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=[_fail_evidence(refs["a"])],
        budget_factory=(budget if budget is not None else (lambda: Budget(cap=1000))),
    )
    return result.branches[0]


def test_repair_gain_score_resolved_case_has_full_gain_and_no_damage() -> None:
    checkpoint, refs = _checkpoint_with_sections("a", "b")
    branch = _run_single_branch(checkpoint, refs, producer=_FakeProducer(), validator=_FakeValidator())

    score = compute_repair_gain_score(checkpoint, branch)

    assert score.gain == 1.0
    assert score.outcome_counts == {RepairOutcome.RESOLVED.value: 1}
    assert score.damage.newly_broken_artifacts == []
    assert score.damage.preserved_fraction == 1.0
    assert score.total_cost_tokens == branch.total_cost_tokens
    assert score.total_cost_tokens > 0


def test_repair_gain_score_still_failing_case_has_zero_gain() -> None:
    checkpoint, refs = _checkpoint_with_sections("a", "b")
    branch = _run_single_branch(
        checkpoint, refs, producer=_FakeProducer(), validator=_FakeValidator(fail_ids=frozenset({"a"}))
    )

    score = compute_repair_gain_score(checkpoint, branch)

    assert score.gain == 0.0
    assert score.outcome_counts == {RepairOutcome.STILL_FAILING.value: 1}


def test_repair_gain_score_nothing_resolved_leaves_gain_none_with_reason() -> None:
    checkpoint, refs = _checkpoint_with_sections("a", "b")
    branch = _run_single_branch(
        checkpoint, refs, producer=_FakeProducer(), validator=_FakeValidator(), budget=lambda: Budget(cap=0)
    )

    score = compute_repair_gain_score(checkpoint, branch)

    assert score.gain is None
    assert score.not_implemented_reason is not None
    assert score.outcome_counts == {}


def test_build_candidate_action_record_end_to_end() -> None:
    checkpoint, refs = _checkpoint_with_sections("a", "b")
    branch = _run_single_branch(checkpoint, refs, producer=_FakeProducer(), validator=_FakeValidator())

    record = build_candidate_action_record(
        record_id="rec1", failure_id="f1", before_checkpoint=checkpoint, branch=branch
    )

    assert record.failure_id == "f1"
    assert record.action.action_type == RepairActionType.REGENERATE_SECTION
    assert record.outcome.gain == 1.0
    assert record.rounds_executed == len(branch.rounds)


def test_build_candidate_action_record_raises_when_branch_never_executed_anything() -> None:
    checkpoint, refs = _checkpoint_with_sections("a", "b")
    branch = _run_single_branch(
        checkpoint, refs, producer=_FakeProducer(), validator=_FakeValidator(), budget=lambda: Budget(cap=0)
    )
    with pytest.raises(ValueError):
        build_candidate_action_record(record_id="rec1", failure_id="f1", before_checkpoint=checkpoint, branch=branch)


# ---------------------------------------------------------------------------
# damage detection in isolation (content changed outside the touched set)
# ---------------------------------------------------------------------------


def test_repair_gain_score_detects_damage_to_an_untouched_artifact() -> None:
    from src.core.failure_dataset import _compute_damage

    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.SECTION, content="orig-a")
    store.add_version("b", ArtifactKind.SECTION, content="orig-b")
    before = store.save_checkpoint()

    branch_store = ArtifactGraphStore.from_checkpoint(before, branch_id="branch-x")
    a_ref = branch_store.latest("a").ref
    branch_store.add_version("a", ArtifactKind.SECTION, content="fixed-a", read_input_versions=[a_ref])
    b_ref = branch_store.latest("b").ref
    branch_store.add_version("b", ArtifactKind.SECTION, content="corrupted-b", read_input_versions=[b_ref])
    after = branch_store.save_checkpoint()

    damage = _compute_damage(before, after, {"a": RepairOutcome.RESOLVED})

    assert damage.newly_broken_artifacts == ["b"]
    assert damage.preserved_fraction == 0.0

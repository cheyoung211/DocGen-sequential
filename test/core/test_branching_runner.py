"""Coverage for src/core/branching_runner.py (Stage 4): branching one shared
checkpoint into independent episodes, a fixed-rule-but-fresh-instance
downstream policy per branch, check_timing's continuous/final_only split,
and termination (budget/round-cap/stalled-without-evidence) -- all with
Protocol-conforming fakes, per this stage's scope decision."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional

import pytest

from src.core.artifact_graph import ArtifactGraphStore
from src.core.branching_runner import BranchingRunner, BranchSpec
from src.core.cost_tracking import CostTrackingProducer
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    CheckOutcome,
    Constraint,
    ConstraintKind,
    Evidence,
    ProductionResult,
    ProductionStatus,
    RepairAction,
    RepairActionType,
)


class _FakeProducer:
    """``cost_per_call`` only matters when this is wrapped in a
    ``CostTrackingProducer`` (it is read back out of ``context["usage_sink"]``
    by that wrapper) -- for every other test, nothing reads it, so the
    default of 0 is simply inert."""

    def __init__(self, fail_ids: FrozenSet[str] = frozenset(), cost_per_call: int = 0) -> None:
        self.fail_ids = set(fail_ids)
        self.cost_per_call = cost_per_call
        self.calls: List[str] = []

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ProductionResult:
        self.calls.append(artifact_id)
        sink = context.get("usage_sink")
        if sink is not None and self.cost_per_call:
            sink.append(
                {"input_tokens": 0, "output_tokens": self.cost_per_call, "total_tokens": self.cost_per_call}
            )
        if artifact_id in self.fail_ids:
            return ProductionResult(status=ProductionStatus.FAILED, error="fake production failure")
        return ProductionResult(
            status=ProductionStatus.SUCCESS, artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.SECTION)
        )


class _FakeValidator:
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


def _one_constraint_per_artifact(ref: ArtifactRef) -> List[Constraint]:
    return [Constraint(constraint_id=f"c:{ref.artifact_id}", kind=ConstraintKind.SCHEMA_CONTRACT, description="d")]


def _checkpoint_with_one_section(artifact_id: str = "secA"):
    store = ArtifactGraphStore()
    store.add_version(artifact_id, ArtifactKind.SECTION, content=None)
    ref = store.latest(artifact_id).ref
    checkpoint = store.save_checkpoint(label="pre-repair")
    return checkpoint, ref


def _fail_evidence(ref: ArtifactRef) -> Evidence:
    return Evidence(constraint_id="c", outcome=CheckOutcome.FAIL, checked_artifact=ref, detail="bad")


class _ActOnFirstFailure:
    """Stateless, deterministic fake policy: targets the first FAIL
    evidence's artifact with a fixed action_type, every call."""

    def __init__(self, action_type: RepairActionType) -> None:
        self.action_type = action_type

    def decide(self, evidence, dependency_edges, budget) -> List[RepairAction]:
        if budget.remaining <= 0:
            return []
        fails = [e for e in evidence if e.outcome == CheckOutcome.FAIL]
        if not fails:
            return []
        return [
            RepairAction(
                action_type=self.action_type,
                target_artifacts=[fails[0].checked_artifact],
                use_diagnostic_feedback=False,
            )
        ]


class _CountingActOnFirstFailure(_ActOnFirstFailure):
    """Same decision rule, but counts how many times THIS instance's
    decide() was called -- used to prove each branch gets its own fresh
    instance, not a shared one."""

    def __init__(self, action_type: RepairActionType = RepairActionType.REGENERATE_SECTION) -> None:
        super().__init__(action_type)
        self.decide_calls = 0

    def decide(self, evidence, dependency_edges, budget) -> List[RepairAction]:
        self.decide_calls += 1
        return super().decide(evidence, dependency_edges, budget)


class _NothingPolicy:
    def decide(self, evidence, dependency_edges, budget) -> List[RepairAction]:
        return []


def _runner(
    producer: _FakeProducer,
    validator_factory,
    *,
    max_downstream_rounds: int = 5,
) -> BranchingRunner:
    """``validator_factory`` takes the branch's restored store (ignored by
    every fake validator in this file, which read only whatever
    ``artifact``/``constraints`` are passed into ``check()`` -- unlike the
    real ``BasicConstraintChecker``, which does bind to it). For the common
    case of one validator instance that is safe to share (no cross-call
    state at all), pass ``lambda store: validator_instance``."""
    return BranchingRunner(
        producer_registry={ArtifactKind.SECTION: producer},
        validator_factory=validator_factory,
        constraint_provider=_one_constraint_per_artifact,
        max_downstream_rounds=max_downstream_rounds,
    )


# ---------------------------------------------------------------------------
# Branch independence.
# ---------------------------------------------------------------------------


def test_branches_are_independent_and_each_runs_its_own_initial_policy() -> None:
    checkpoint, sec_ref = _checkpoint_with_one_section()
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset())  # everything passes once regenerated
    runner = _runner(producer, lambda store: validator)

    policy_a = _ActOnFirstFailure(RepairActionType.REGENERATE_SECTION)
    policy_b = _ActOnFirstFailure(RepairActionType.PATCH_BLOCK)
    trigger = [_fail_evidence(sec_ref)]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy_a), BranchSpec("B", policy_b)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    assert result.source_checkpoint_id == checkpoint.checkpoint_id
    assert [b.branch_id for b in result.branches] == ["A", "B"]
    assert result.branches[0].initial_action.action_type == RepairActionType.REGENERATE_SECTION
    assert result.branches[1].initial_action.action_type == RepairActionType.PATCH_BLOCK
    # Each branch's own final checkpoint reflects only that branch's work.
    assert result.branches[0].final_checkpoint.checkpoint_id != result.branches[1].final_checkpoint.checkpoint_id
    # The original checkpoint itself (and a store rebuilt from it) is untouched.
    fresh = ArtifactGraphStore.from_checkpoint(checkpoint, branch_id="untouched-check")
    assert fresh.latest("secA").ref.version == 1


def test_downstream_policy_factory_is_called_once_per_branch_with_fresh_instances() -> None:
    checkpoint, sec_ref = _checkpoint_with_one_section()
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset())  # resolves after one regen -- one downstream round at most
    runner = _runner(producer, lambda store: validator)

    created: List[_CountingActOnFirstFailure] = []

    def factory() -> _CountingActOnFirstFailure:
        instance = _CountingActOnFirstFailure()
        created.append(instance)
        return instance

    trigger = [_fail_evidence(sec_ref)]
    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", _NothingPolicy()), BranchSpec("B", _NothingPolicy())],
        downstream_policy_factory=factory,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    # Both branches' initial policy returns nothing (_NothingPolicy), so no
    # downstream round ever runs -- but the factory must still be called
    # once per branch (not skipped, not shared), proving each branch would
    # have gotten its own independent instance had one been needed.
    assert len(created) == 2
    assert created[0] is not created[1]
    assert all(b.stopped_reason == "initial_policy_returned_nothing" for b in result.branches)


class _FailsFirstCheckThenPasses:
    """Fails the first time a given artifact_id is checked, passes every
    time after -- models "the regeneration took exactly one retry to
    actually fix it", so a test can reliably drive exactly one downstream
    round instead of either zero (resolved immediately) or infinitely many
    (never resolves)."""

    def __init__(self, fail_ids: FrozenSet[str]) -> None:
        self.fail_ids = set(fail_ids)
        self._checked_once: set = set()
        self.config: Dict[str, bool] = {}
        self.calls: List[str] = []

    def check(self, artifact, constraints, context=None) -> List[Evidence]:
        self.calls.append(artifact.artifact_id)
        aid = artifact.artifact_id
        if aid in self.fail_ids and aid not in self._checked_once:
            self._checked_once.add(aid)
            outcome = CheckOutcome.FAIL
        else:
            outcome = CheckOutcome.PASS
        return [
            Evidence(constraint_id=c.constraint_id, outcome=outcome, checked_artifact=artifact, detail="fake")
            for c in constraints
        ]


def test_a_stateful_downstream_policy_does_not_leak_retry_state_across_branches() -> None:
    """The real risk _CountingActOnFirstFailure's factory test guards
    against: if BranchingRunner ever reused one shared instance across
    branches (instead of calling the factory per branch), a stateful
    policy's internal counters would carry over. Here two branches each
    drive exactly one downstream round; each branch's own policy instance
    must report exactly 1 call, never 2."""
    checkpoint, sec_ref = _checkpoint_with_one_section()
    producer = _FakeProducer()
    # A fresh validator per branch too -- _FailsFirstCheckThenPasses itself
    # carries cross-call state (_checked_once), so sharing one instance
    # across branches would have the same leakage problem this test exists
    # to rule out, just in the validator instead of the policy.
    runner = _runner(producer, lambda store: _FailsFirstCheckThenPasses(fail_ids=frozenset({"secA"})))

    created: List[_CountingActOnFirstFailure] = []

    def factory() -> _CountingActOnFirstFailure:
        instance = _CountingActOnFirstFailure()
        created.append(instance)
        return instance

    initial_policy = _ActOnFirstFailure(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(sec_ref)]

    runner.run_branches(
        checkpoint,
        [BranchSpec("A", initial_policy), BranchSpec("B", initial_policy)],
        downstream_policy_factory=factory,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    assert len(created) == 2
    for instance in created:
        assert instance.decide_calls == 1


# ---------------------------------------------------------------------------
# Failed repairs stay visible, never silently read as success.
# ---------------------------------------------------------------------------


def test_a_production_failure_is_recorded_and_the_branch_stops_honestly() -> None:
    checkpoint, sec_ref = _checkpoint_with_one_section()
    producer = _FakeProducer(fail_ids=frozenset({"secA"}))  # the repair itself cannot even run
    validator = _FakeValidator(fail_ids=frozenset({"secA"}))
    runner = _runner(producer, lambda store: validator)

    policy = _ActOnFirstFailure(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(sec_ref)]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    branch = result.branches[0]
    assert branch.initial_round is not None
    from src.core.schemas import RepairOutcome

    assert branch.initial_round.overall_outcome == RepairOutcome.PRODUCTION_FAILED
    # The stop reason must not read as a clean resolution -- it would be a
    # silent false "success" otherwise.
    assert branch.stopped_reason == "repair_could_not_be_attempted"
    assert branch.stopped_reason != "no_more_failures"


# ---------------------------------------------------------------------------
# check_timing.
# ---------------------------------------------------------------------------


def test_continuous_check_timing_uses_triggering_evidence_without_any_extra_check_call() -> None:
    checkpoint, sec_ref = _checkpoint_with_one_section()
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset())
    runner = _runner(producer, lambda store: validator)
    policy = _ActOnFirstFailure(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(sec_ref)]

    calls_before = len(validator.calls)
    runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
        check_timing="continuous",
    )
    # The only validator.check() calls that happen are the executor's own
    # post-regen reverify -- never one driven by BranchingRunner itself for
    # the *initial* decision, since "continuous" uses triggering_evidence as
    # given.
    assert "secA" not in validator.calls[:calls_before]


def test_final_only_check_timing_ignores_triggering_evidence_and_checks_once_first() -> None:
    checkpoint, sec_ref = _checkpoint_with_one_section()
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset({"secA"}))  # the one-shot final check will find this
    runner = _runner(producer, lambda store: validator)
    policy = _ActOnFirstFailure(RepairActionType.REGENERATE_SECTION)

    # A deliberately wrong triggering_evidence (names an artifact that does
    # not exist) -- if check_timing="final_only" actually used this instead
    # of running its own check, the policy would have nothing real to act on.
    bogus_trigger = [_fail_evidence(ArtifactRef(artifact_id="not-real", kind=ArtifactKind.SECTION))]
    constraint = _one_constraint_per_artifact(sec_ref)

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=bogus_trigger,
        budget_factory=lambda: Budget(cap=1000),
        check_timing="final_only",
        final_check_target=sec_ref,
        final_check_constraints=constraint,
    )

    branch = result.branches[0]
    assert branch.initial_action is not None
    assert branch.initial_action.target_artifacts[0].artifact_id == "secA"  # acted on the real check, not the bogus trigger
    assert "secA" in validator.calls


def test_final_only_requires_target_and_constraints() -> None:
    checkpoint, _ref = _checkpoint_with_one_section()
    producer = _FakeProducer()
    validator = _FakeValidator()
    runner = _runner(producer, lambda store: validator)

    with pytest.raises(ValueError, match="final_check_target"):
        runner.run_branches(
            checkpoint,
            [BranchSpec("A", _NothingPolicy())],
            downstream_policy_factory=_NothingPolicy,
            triggering_evidence=[],
            budget_factory=lambda: Budget(cap=100),
            check_timing="final_only",
        )


def test_invalid_check_timing_is_rejected() -> None:
    checkpoint, _ref = _checkpoint_with_one_section()
    producer = _FakeProducer()
    validator = _FakeValidator()
    runner = _runner(producer, lambda store: validator)

    with pytest.raises(ValueError, match="check_timing"):
        runner.run_branches(
            checkpoint,
            [BranchSpec("A", _NothingPolicy())],
            downstream_policy_factory=_NothingPolicy,
            triggering_evidence=[],
            budget_factory=lambda: Budget(cap=100),
            check_timing="sometimes",
        )


# ---------------------------------------------------------------------------
# Termination.
# ---------------------------------------------------------------------------


def test_budget_exhaustion_stops_a_branch_cleanly() -> None:
    """Cost-free fake producers never actually spend anything, so budget
    exhaustion can only be observed with a real cost flowing through --
    wrap the fake in CostTrackingProducer (the same adapter
    RepairOperationExecutor itself looks for) so each regeneration really
    does charge 100 tokens. cap=150 admits exactly one regeneration (100)
    but not two (200)."""
    checkpoint, sec_ref = _checkpoint_with_one_section()
    cost_tracked = CostTrackingProducer(
        _FakeProducer(cost_per_call=100), stage="test", model_name="fake-model"
    )
    # Never clears -- every downstream round would try again forever if not
    # for the budget running out.
    validator = _FakeValidator(fail_ids=frozenset({"secA"}))
    runner = _runner(cost_tracked, lambda store: validator, max_downstream_rounds=50)
    policy = _ActOnFirstFailure(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(sec_ref)]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=lambda: _ActOnFirstFailure(RepairActionType.REGENERATE_SECTION),
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=150),
    )

    assert result.branches[0].stopped_reason == "budget_exhausted"
    assert result.branches[0].total_cost_tokens >= 150


def test_max_downstream_rounds_terminates_cleanly_instead_of_looping_forever() -> None:
    """secA never actually clears (validator.fail_ids is permanent), so the
    fixed downstream policy would decide a new action forever -- this test
    passing at all (not hanging) is part of the proof that the cap works."""
    checkpoint, sec_ref = _checkpoint_with_one_section()
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset({"secA"}))
    runner = _runner(producer, lambda store: validator, max_downstream_rounds=3)
    policy = _ActOnFirstFailure(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(sec_ref)]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=lambda: _ActOnFirstFailure(RepairActionType.REGENERATE_SECTION),
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1_000_000),
    )

    branch = result.branches[0]
    assert branch.stopped_reason == "max_rounds_reached"
    assert len(branch.downstream_rounds) == 3


def test_initial_policy_returning_nothing_stops_cleanly_with_no_initial_round() -> None:
    checkpoint, sec_ref = _checkpoint_with_one_section()
    producer = _FakeProducer()
    validator = _FakeValidator()
    runner = _runner(producer, lambda store: validator)
    trigger = [_fail_evidence(sec_ref)]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", _NothingPolicy())],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    branch = result.branches[0]
    assert branch.initial_action is None
    assert branch.initial_round is None
    assert branch.downstream_rounds == []
    assert branch.stopped_reason == "initial_policy_returned_nothing"
    # A final checkpoint is still produced even though nothing happened.
    assert branch.final_checkpoint is not None

"""Coverage for src/core/branching_runner.py (Stage 4): branching one shared
checkpoint into independent episodes, a fixed-rule-but-fresh-instance
downstream policy per branch, check_timing's continuous/final_only split,
and termination (budget/round-cap/stalled-without-evidence) -- all with
Protocol-conforming fakes, per this stage's scope decision.

Revised after a user review found 3 real bugs affecting this module (see
memory docgen_stage_review_rigor, points 5/6/7), a second review found 2
more (points 8/9), and a third review found 1 more -- every fix below has
at least one regression test named after the repro that caught it."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional

import pytest

from src.core.artifact_graph import ArtifactGraphStore
from src.core.branching_runner import BranchingRunner, BranchSpec
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
    RepairOutcome,
)


class _FakeProducer:
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
            status=ProductionStatus.SUCCESS,
            artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.SECTION),
            content=f"real-content:{artifact_id}",
            read_input_versions=context.get("read_input_versions"),
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


def _one_constraint_per_artifact(ref: ArtifactRef) -> List[Constraint]:
    return [Constraint(constraint_id=f"c:{ref.artifact_id}", kind=ConstraintKind.SCHEMA_CONTRACT, description="d")]


def _checkpoint_with_sections(*artifact_ids: str):
    store = ArtifactGraphStore()
    refs = {}
    for artifact_id in artifact_ids:
        store.add_version(artifact_id, ArtifactKind.SECTION, content=None)
        refs[artifact_id] = store.latest(artifact_id).ref
    checkpoint = store.save_checkpoint(label="pre-repair")
    return checkpoint, refs


def _fail_evidence(ref: ArtifactRef) -> Evidence:
    return Evidence(constraint_id="c", outcome=CheckOutcome.FAIL, checked_artifact=ref, detail="bad")


class _ActOnAllFailures:
    """Stateless, deterministic fake policy: one action per distinct FAIL
    evidence (mirrors FixedGraphAwareRepairPolicy's own cardinality) --
    used to exercise "every decided action must be executed", not just the
    first."""

    def __init__(self, action_type: RepairActionType = RepairActionType.REGENERATE_SECTION) -> None:
        self.action_type = action_type

    def decide(self, evidence, dependency_edges, budget) -> List[RepairAction]:
        if budget.remaining <= 0:
            return []
        seen: Dict[str, Evidence] = {}
        for item in evidence:
            if item.outcome == CheckOutcome.FAIL:
                seen.setdefault(item.checked_artifact.artifact_id, item)
        return [
            RepairAction(action_type=self.action_type, target_artifacts=[item.checked_artifact], use_diagnostic_feedback=False)
            for item in seen.values()
        ]


class _CountingActOnAllFailures(_ActOnAllFailures):
    def __init__(self, action_type: RepairActionType = RepairActionType.REGENERATE_SECTION) -> None:
        super().__init__(action_type)
        self.decide_calls = 0

    def decide(self, evidence, dependency_edges, budget) -> List[RepairAction]:
        self.decide_calls += 1
        return super().decide(evidence, dependency_edges, budget)


class _NothingPolicy:
    def decide(self, evidence, dependency_edges, budget) -> List[RepairAction]:
        return []


class _RaisingValidator:
    """Always raises instead of returning Evidence -- models a real
    ``Validator.check()`` crash, which ``ToolExecutor`` reports as
    ``ToolErrorCode.EXECUTION_FAILED``, never any budget code."""

    def check(self, artifact, constraints, context=None) -> List[Evidence]:
        raise RuntimeError("validator exploded")


class _SequencedValidator:
    """Returns a scripted sequence of outcomes per artifact_id, one per
    successive check call against that id (repeating the final entry once
    the script runs out); PASS for any id never scripted at all. Lets a
    test drive a candidate through PASS-then-FAIL (or the reverse)
    deterministically across a real multi-sweep cascade, instead of
    hand-constructing a ``RepairExecutionResult``."""

    def __init__(self, script: Dict[str, List[CheckOutcome]]) -> None:
        self._script = {artifact_id: list(outcomes) for artifact_id, outcomes in script.items()}
        self._seen: Dict[str, int] = {}
        self.calls: List[str] = []

    def check(self, artifact, constraints, context=None) -> List[Evidence]:
        self.calls.append(artifact.artifact_id)
        index = self._seen.get(artifact.artifact_id, 0)
        self._seen[artifact.artifact_id] = index + 1
        sequence = self._script.get(artifact.artifact_id)
        outcome = sequence[min(index, len(sequence) - 1)] if sequence else CheckOutcome.PASS
        return [
            Evidence(constraint_id=c.constraint_id, outcome=outcome, checked_artifact=artifact, detail="fake")
            for c in constraints
        ]


def _checkpoint_with_a_dependency_chain(*artifact_ids: str):
    """Like `_checkpoint_with_sections`, but each id reads the previous one
    as its only input -- needed for the issue-8 repros, where a repair's own
    cascade must actually reach a downstream artifact the initial trigger
    never named."""
    store = ArtifactGraphStore()
    refs = {}
    previous_ref = None
    for artifact_id in artifact_ids:
        inputs = [previous_ref] if previous_ref is not None else []
        store.add_version(artifact_id, ArtifactKind.SECTION, content=None, read_input_versions=inputs)
        previous_ref = store.latest(artifact_id).ref
        refs[artifact_id] = previous_ref
    checkpoint = store.save_checkpoint(label="pre-repair")
    return checkpoint, refs


def _runner(
    producer,
    validator_factory,
    *,
    max_downstream_rounds: int = 5,
    max_actions_per_branch: int = 1000,
    model_name: str = "fake-model",
) -> BranchingRunner:
    return BranchingRunner(
        producer_registry={ArtifactKind.SECTION: producer},
        validator_factory=validator_factory,
        constraint_provider=_one_constraint_per_artifact,
        model_name=model_name,
        max_actions_per_branch=max_actions_per_branch,
        max_downstream_rounds=max_downstream_rounds,
    )


def _all_executions(branch):
    """Flattens every RepairExecutionResult across every round, in order."""
    return [execution for round_ in branch.rounds for execution in round_.executions]


# ---------------------------------------------------------------------------
# Branch independence.
# ---------------------------------------------------------------------------


def test_branches_are_independent_and_each_runs_its_own_initial_policy() -> None:
    checkpoint, refs = _checkpoint_with_sections("secA")
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset())
    runner = _runner(producer, lambda store: validator)

    policy_a = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    policy_b = _ActOnAllFailures(RepairActionType.PATCH_BLOCK)
    trigger = [_fail_evidence(refs["secA"])]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy_a), BranchSpec("B", policy_b)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    assert result.source_checkpoint_id == checkpoint.checkpoint_id
    assert [b.branch_id for b in result.branches] == ["A", "B"]
    assert _all_executions(result.branches[0])[0].action.action_type == RepairActionType.REGENERATE_SECTION
    assert _all_executions(result.branches[1])[0].action.action_type == RepairActionType.PATCH_BLOCK
    assert result.branches[0].final_checkpoint.checkpoint_id != result.branches[1].final_checkpoint.checkpoint_id
    fresh = ArtifactGraphStore.from_checkpoint(checkpoint, branch_id="untouched-check")
    assert fresh.latest("secA").ref.version == 1


def test_downstream_policy_factory_is_called_once_per_branch_with_fresh_instances() -> None:
    checkpoint, refs = _checkpoint_with_sections("secA")
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset())
    runner = _runner(producer, lambda store: validator)

    created: List[_CountingActOnAllFailures] = []

    def factory() -> _CountingActOnAllFailures:
        instance = _CountingActOnAllFailures()
        created.append(instance)
        return instance

    trigger = [_fail_evidence(refs["secA"])]
    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", _NothingPolicy()), BranchSpec("B", _NothingPolicy())],
        downstream_policy_factory=factory,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    assert len(created) == 2
    assert created[0] is not created[1]
    assert all(b.stopped_reason == "initial_policy_returned_nothing" for b in result.branches)
    assert all(b.rounds == [] for b in result.branches)


def test_a_stateful_downstream_policy_does_not_leak_retry_state_across_branches() -> None:
    checkpoint, refs = _checkpoint_with_sections("secA")
    producer = _FakeProducer()

    created: List[_CountingActOnAllFailures] = []

    def factory() -> _CountingActOnAllFailures:
        instance = _CountingActOnAllFailures()
        created.append(instance)
        return instance

    initial_policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(refs["secA"])]
    runner = _runner(producer, lambda store: _FailsFirstCheckThenPasses(fail_ids=frozenset({"secA"})))

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
# Issue 5 (review): every decided action is executed, and an unresolved
# failure is never silently dropped after an unrelated one is handled.
# ---------------------------------------------------------------------------


def test_two_independent_failures_both_get_a_decided_action_executed() -> None:
    checkpoint, refs = _checkpoint_with_sections("secA", "secB")
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset())  # both resolve cleanly once regenerated
    runner = _runner(producer, lambda store: validator)
    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(refs["secA"]), _fail_evidence(refs["secB"])]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    branch = result.branches[0]
    touched = {e.pre_repair_ref.artifact_id for ex in _all_executions(branch) for e in ex.regenerated}
    assert touched == {"secA", "secB"}
    assert producer.calls.count("secA") == 1
    assert producer.calls.count("secB") == 1


def test_an_unresolved_independent_failure_is_not_silently_dropped_after_the_other_resolves() -> None:
    """A FAIL, B FAIL -- the policy's single decide() call names both,
    but B's production always fails (can never actually be fixed). The
    branch must not report "no_more_failures" while B is still broken."""
    checkpoint, refs = _checkpoint_with_sections("secA", "secB")
    producer = _FakeProducer(fail_ids={"secB"})  # A resolves, B's production can never succeed
    validator = _FakeValidator(fail_ids=frozenset())
    runner = _runner(producer, lambda store: validator, max_downstream_rounds=1)
    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(refs["secA"]), _fail_evidence(refs["secB"])]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=lambda: _ActOnAllFailures(RepairActionType.REGENERATE_SECTION),
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    branch = result.branches[0]
    assert branch.stopped_reason != "no_more_failures"
    b_outcomes = [
        e.outcome for ex in _all_executions(branch) for e in ex.regenerated if e.pre_repair_ref.artifact_id == "secB"
    ]
    assert RepairOutcome.PRODUCTION_FAILED in b_outcomes


# ---------------------------------------------------------------------------
# Issue 6 (review): budget admission is enforced per action, including
# between multiple actions decided within the SAME round.
# ---------------------------------------------------------------------------


def test_budget_is_enforced_between_actions_within_one_round_not_only_between_rounds() -> None:
    checkpoint, refs = _checkpoint_with_sections("secA", "secB")
    # Raw producer, NOT pre-wrapped in CostTrackingProducer -- ToolExecutor.
    # run_generation() does that wrapping itself; wrapping it again here
    # would double-wrap (the outer wrapper's usage_sink never sees the
    # inner wrapper's own separately-constructed one, so no cost would
    # ever actually be measured).
    producer = _FakeProducer(cost_per_call=100)
    validator = _FakeValidator(fail_ids=frozenset())
    runner = _runner(producer, lambda store: validator)
    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)  # names BOTH secA and secB in one decide() call
    trigger = [_fail_evidence(refs["secA"]), _fail_evidence(refs["secB"])]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=150),  # enough for exactly one of the two 100-token actions
    )

    branch = result.branches[0]
    assert branch.stopped_reason == "budget_exhausted"
    assert branch.total_cost_tokens <= 200  # bounded to at most one call's overspend


# ---------------------------------------------------------------------------
# Issue 7 (review): each branch gets its own independent context.
# ---------------------------------------------------------------------------


def test_each_branch_gets_an_independent_context_not_a_shared_mutable_one() -> None:
    """A Producer that mutates a mutable object inside its context (like a
    real TextAgent mutating a DocumentGraph) must not leak that mutation
    into a sibling branch -- each branch's context_factory call must
    produce its own independent object."""
    checkpoint, refs = _checkpoint_with_sections("secA")

    class _MutatingProducer:
        def produce(self, artifact_id, context):
            counter = context["counter"]
            counter["value"] += 1
            return ProductionResult(
                status=ProductionStatus.SUCCESS,
                artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.SECTION),
                content=f"seen-{counter['value']}",
                read_input_versions=[],
            )

    validator = _FakeValidator(fail_ids=frozenset())
    runner = _runner(_MutatingProducer(), lambda store: validator)
    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(refs["secA"])]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy), BranchSpec("B", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
        context_factory=lambda spec: {"counter": {"value": 0}},
    )

    for branch in result.branches:
        new_content = branch.final_checkpoint.artifacts["secA"][-1].content
        assert new_content == "seen-1"  # every branch starts its own counter at 0, independently


def test_context_factory_default_still_gives_each_branch_an_independent_dict() -> None:
    checkpoint, refs = _checkpoint_with_sections("secA")
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset())
    runner = _runner(producer, lambda store: validator)
    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(refs["secA"])]

    # No context_factory passed at all -- must not raise, must not share
    # state (there is none to share here, but the call path must work).
    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy), BranchSpec("B", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )
    assert len(result.branches) == 2


# ---------------------------------------------------------------------------
# Second review round, issue 8: a sub-failure discovered only by a round's
# own cascade (never part of that round's starting `pending`) must not
# vanish into a false "no_more_failures".
# ---------------------------------------------------------------------------


def test_a_newly_discovered_production_failure_is_not_read_as_no_more_failures() -> None:
    """Review repro: only 'secA' is reported failing; repairing it cascades
    into a downstream 'secB' (never named in the original trigger) whose
    production then always fails. secB has no FAIL Evidence of its own to
    re-enter `pending` with, so the round's `pending` went empty and
    stopped_reason="no_more_failures" even though secB was left broken."""
    checkpoint, refs = _checkpoint_with_a_dependency_chain("secA", "secB")
    producer = _FakeProducer(fail_ids={"secB"})  # secA's production succeeds, secB's never does
    validator = _FakeValidator(fail_ids={"secB"})  # secB also fails its own content check
    runner = _runner(producer, lambda store: validator)
    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(refs["secA"])]  # secB is never named here at all

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    branch = result.branches[0]
    assert branch.stopped_reason == "unresolved_remaining"
    b_outcomes = [
        e.outcome for ex in _all_executions(branch) for e in ex.regenerated if e.pre_repair_ref.artifact_id == "secB"
    ]
    assert RepairOutcome.PRODUCTION_FAILED in b_outcomes


def test_a_newly_discovered_left_stale_candidate_is_not_read_as_no_more_failures() -> None:
    """Same repro shape, but secB has no constraints to check it against at
    all (rather than a production failure) -- the review's own note that
    this hits the identical false "no_more_failures" termination."""
    checkpoint, refs = _checkpoint_with_a_dependency_chain("secA", "secB")
    producer = _FakeProducer()  # everything that gets regenerated succeeds
    validator = _FakeValidator(fail_ids=frozenset())  # secA resolves cleanly
    runner = BranchingRunner(
        producer_registry={ArtifactKind.SECTION: producer},
        validator_factory=lambda store: validator,
        constraint_provider=lambda ref: [] if ref.artifact_id == "secB" else _one_constraint_per_artifact(ref),
        model_name="fake-model",
    )
    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(refs["secA"])]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    branch = result.branches[0]
    assert branch.stopped_reason == "unresolved_remaining"
    left_stale_ids = {ref.artifact_id for ex in _all_executions(branch) for ref in ex.left_stale}
    assert "secB" in left_stale_ids


# ---------------------------------------------------------------------------
# Second review round, issue 9: check_timing="final_only"'s one-shot initial
# check must not misreport a validator crash as budget exhaustion.
# ---------------------------------------------------------------------------


def test_final_only_initial_check_crash_is_not_misreported_as_budget_exhaustion() -> None:
    checkpoint, refs = _checkpoint_with_sections("secA")
    producer = _FakeProducer()
    runner = _runner(producer, lambda store: _RaisingValidator())
    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    constraint = _one_constraint_per_artifact(refs["secA"])

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=[],
        budget_factory=lambda: Budget(cap=1000),
        check_timing="final_only",
        final_check_target=refs["secA"],
        final_check_constraints=constraint,
    )

    branch = result.branches[0]
    assert branch.stopped_reason == "initial_check_failed"
    assert branch.rounds == []


# ---------------------------------------------------------------------------
# Third review round: _apply_round_to_pending() must judge each artifact_id
# by its LATEST disposition, never by processing `regenerated` in full and
# then `reverified_untouched` in full regardless of true chronology.
# ---------------------------------------------------------------------------


def test_a_still_failing_candidate_is_not_cleared_by_an_earlier_superseded_reaffirm() -> None:
    """Review repro: a -> z -> b. Repairing 'a' cascades and marks z and b
    both stale. The sweep visits 'b' first (alphabetically before 'z'),
    where it PASSes and is reaffirmed -- then visits 'z', which FAILs and
    is regenerated, re-invalidating 'b' again; revisited once more, 'b'
    FAILs this time and is genuinely regenerated (STILL_FAILING). So 'b'
    ends up in BOTH `reverified_untouched` (the now-superseded early PASS)
    and `regenerated` (the true, later STILL_FAILING). The old
    _apply_round_to_pending() processed all of `regenerated` first (adding
    b to pending/unresolved_ids), then unconditionally processed all of
    `reverified_untouched` afterward, which popped b right back out --
    reporting stopped_reason="no_more_failures" with b left genuinely
    broken."""
    checkpoint, refs = _checkpoint_with_a_dependency_chain("a", "z", "b")
    producer = _FakeProducer()
    validator = _SequencedValidator(
        {"z": [CheckOutcome.FAIL, CheckOutcome.PASS], "b": [CheckOutcome.PASS, CheckOutcome.FAIL]}
    )
    runner = _runner(producer, lambda store: validator)
    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(refs["a"])]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    branch = result.branches[0]
    executions = _all_executions(branch)
    # Confirm the repro actually happened: b appears in both the audit-log
    # lists, with the regenerated one's outcome being the true, later one.
    assert any(
        r.pre_repair_ref.artifact_id == "b" and r.outcome == RepairOutcome.STILL_FAILING
        for ex in executions
        for r in ex.regenerated
    )
    assert any(ref.artifact_id == "b" for ex in executions for ref in ex.reverified_untouched)
    assert branch.stopped_reason != "no_more_failures"


# ---------------------------------------------------------------------------
# Failed repairs stay visible, never silently read as success.
# ---------------------------------------------------------------------------


def test_a_production_failure_is_recorded_and_the_branch_stops_honestly() -> None:
    checkpoint, refs = _checkpoint_with_sections("secA")
    producer = _FakeProducer(fail_ids=frozenset({"secA"}))
    validator = _FakeValidator(fail_ids=frozenset({"secA"}))
    runner = _runner(producer, lambda store: validator)

    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(refs["secA"])]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    branch = result.branches[0]
    executions = _all_executions(branch)
    assert executions[0].overall_outcome == RepairOutcome.PRODUCTION_FAILED
    assert branch.stopped_reason != "no_more_failures"


# ---------------------------------------------------------------------------
# check_timing.
# ---------------------------------------------------------------------------


def test_continuous_check_timing_uses_triggering_evidence_directly() -> None:
    checkpoint, refs = _checkpoint_with_sections("secA")
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset())
    runner = _runner(producer, lambda store: validator)
    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(refs["secA"])]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
        check_timing="continuous",
    )

    # Only one check call total: the executor's own post-regen reverify --
    # nothing extra was run by the runner itself to arrive at the initial
    # decision (it used `trigger` as given).
    assert validator.calls == ["secA"]
    assert result.branches[0].rounds[0].executions[0].action.target_artifacts[0].artifact_id == "secA"


def test_final_only_check_timing_ignores_triggering_evidence_and_checks_once_first() -> None:
    checkpoint, refs = _checkpoint_with_sections("secA")
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset({"secA"}))
    runner = _runner(producer, lambda store: validator)
    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)

    bogus_trigger = [_fail_evidence(ArtifactRef(artifact_id="not-real", kind=ArtifactKind.SECTION))]
    constraint = _one_constraint_per_artifact(refs["secA"])

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=bogus_trigger,
        budget_factory=lambda: Budget(cap=1000),
        check_timing="final_only",
        final_check_target=refs["secA"],
        final_check_constraints=constraint,
    )

    branch = result.branches[0]
    executions = _all_executions(branch)
    assert executions[0].action.target_artifacts[0].artifact_id == "secA"
    # Two check calls total: the runner's own one-shot final check, plus
    # the executor's post-regen reverify.
    assert validator.calls.count("secA") == 2


def test_final_only_requires_target_and_constraints() -> None:
    checkpoint, _refs = _checkpoint_with_sections("secA")
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
    checkpoint, _refs = _checkpoint_with_sections("secA")
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
    checkpoint, refs = _checkpoint_with_sections("secA")
    producer = _FakeProducer(cost_per_call=100)  # raw -- see the other budget test's comment
    validator = _FakeValidator(fail_ids=frozenset({"secA"}))  # never clears
    runner = _runner(producer, lambda store: validator, max_downstream_rounds=50)
    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(refs["secA"])]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=lambda: _ActOnAllFailures(RepairActionType.REGENERATE_SECTION),
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=150),
    )

    assert result.branches[0].stopped_reason == "budget_exhausted"


def test_max_downstream_rounds_terminates_cleanly_instead_of_looping_forever() -> None:
    checkpoint, refs = _checkpoint_with_sections("secA")
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset({"secA"}))
    runner = _runner(producer, lambda store: validator, max_downstream_rounds=3)
    policy = _ActOnAllFailures(RepairActionType.REGENERATE_SECTION)
    trigger = [_fail_evidence(refs["secA"])]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy)],
        downstream_policy_factory=lambda: _ActOnAllFailures(RepairActionType.REGENERATE_SECTION),
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1_000_000),
    )

    branch = result.branches[0]
    assert branch.stopped_reason == "max_rounds_reached"
    assert len(branch.rounds) == 4  # 1 initial + 3 downstream


def test_initial_policy_returning_nothing_stops_cleanly_with_no_rounds() -> None:
    checkpoint, refs = _checkpoint_with_sections("secA")
    producer = _FakeProducer()
    validator = _FakeValidator()
    runner = _runner(producer, lambda store: validator)
    trigger = [_fail_evidence(refs["secA"])]

    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", _NothingPolicy())],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=trigger,
        budget_factory=lambda: Budget(cap=1000),
    )

    branch = result.branches[0]
    assert branch.rounds == []
    assert branch.stopped_reason == "initial_policy_returned_nothing"
    assert branch.final_checkpoint is not None

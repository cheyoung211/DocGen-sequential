"""Coverage for src/core/failure_dataset.py (Stage 6): dependency-graph
difficulty classification, and repair gain/damage/cost built on top of
Stage 4's already-reviewed BranchExecutionResult -- via a real
BranchingRunner + Protocol-conforming fakes for the end-to-end wiring, and
hand-built minimal schema objects for the merge/damage logic in isolation.

Revised after a user review found 4 real bugs (full detail in
docs/stage6_failure_dataset.md and memory docgen_stage_review_rigor) --
every fix below has at least one regression test named after the repro
that caught it.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Set

import pytest

from src.core.artifact_graph import ArtifactGraphStore, UnknownArtifactError
from src.core.branching_runner import BranchingRunner, BranchSpec
from src.core.fault_injection import FaultInjector
from src.core.failure_dataset import (
    ALREADY_HEALTHY_UNTOUCHED,
    LEFT_STALE_OR_UNRESOLVED,
    _dependency_depth,
    _merge_final_outcomes,
    build_candidate_action_record,
    build_failure_record,
    classify_failure_difficulty,
    compute_common_target_artifact_ids,
    compute_repair_gain_score,
    count_missing_relations_against_reference,
)
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    BranchComparisonResult,
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
    UnknownReason,
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


def test_classify_difficulty_historical_inferred_relations_count_is_cumulative() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a")
    store.add_version("b", ArtifactKind.CLAIM, content="b")
    a_ref = store.latest("a").ref
    b_ref = store.latest("b").ref

    assert classify_failure_difficulty(store, "a").historical_inferred_relations_count == 0
    store.add_inferred_edge(a_ref, b_ref, reason="b actually cites a's label")
    assert classify_failure_difficulty(store, "a").historical_inferred_relations_count == 1
    assert classify_failure_difficulty(store, "b").historical_inferred_relations_count == 1


def test_historical_inferred_relations_count_stays_set_after_the_relation_stops_applying() -> None:
    """Repro from the user's review: once b is regenerated to a version
    that no longer has any relation to a at all, the historical count must
    still reflect that an INFERRED relation was once discovered here --
    this field is explicitly a cumulative, historical count, never a
    current-state one (see its docstring)."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a")
    store.add_version("b", ArtifactKind.CLAIM, content="b")
    a_ref = store.latest("a").ref
    b_ref = store.latest("b").ref
    store.add_inferred_edge(a_ref, b_ref, reason="b actually cites a's label")
    assert classify_failure_difficulty(store, "b").historical_inferred_relations_count == 1

    # b is regenerated with NO relation to a declared this time.
    store.add_version("b", ArtifactKind.CLAIM, content="b, rewritten, no longer cites a", read_input_versions=[])

    # The old a->b@1 edge is no longer "currently applicable" (b's current
    # version is now 2) -- direct_dependencies must correctly show nothing.
    assert store.direct_dependencies("b") == set()
    # But the historical count stays at 1 -- it is a historical fact, not
    # a snapshot of current applicability.
    assert classify_failure_difficulty(store, "b").historical_inferred_relations_count == 1


def test_current_missing_relations_vs_reference_detects_a_dropped_dependency() -> None:
    """Repro, phrased as the user's suggested fix: comparing against a
    KNOWN reference dependency set is what correctly expresses "missing
    right now" -- 0 while the real dependency is still intact, >0 once a
    regeneration drops it."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.CLAIM, content="b uses a", read_input_versions=[a_ref])

    assert count_missing_relations_against_reference(store, "b", {"a"}) == 0
    profile = classify_failure_difficulty(store, "b", reference_dependency_ids={"a"})
    assert profile.current_missing_relations_vs_reference == 0

    # b is regenerated WITHOUT re-declaring its real dependency on a.
    store.add_version("b", ArtifactKind.CLAIM, content="b, rewritten", read_input_versions=[])

    assert count_missing_relations_against_reference(store, "b", {"a"}) == 1
    profile_after = classify_failure_difficulty(store, "b", reference_dependency_ids={"a"})
    assert profile_after.current_missing_relations_vs_reference == 1
    # The historical count is a different axis entirely and stays 0 here
    # (no INFERRED edge was ever involved in this scenario).
    assert profile_after.historical_inferred_relations_count == 0


def test_current_missing_relations_vs_reference_is_none_without_a_supplied_reference() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a")
    profile = classify_failure_difficulty(store, "a")
    assert profile.current_missing_relations_vs_reference is None


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
    assert record.difficulty.current_missing_relations_vs_reference is None  # no injected_fault reference


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


def test_build_failure_record_injected_derives_reference_from_the_injected_fault() -> None:
    """An injected fault's own pre-injection record of t's real
    dependencies is exactly the known reference this stage has on hand --
    build_failure_record must use it automatically."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a")
    a_ref = store.latest("a").ref
    store.add_version(
        "t", ArtifactKind.CLAIM, content="Assume X > 0. Conclusion.", read_input_versions=[a_ref]
    )
    t_ref = store.latest("t").ref
    fault = FaultInjector().inject_missing_assumption(store, t_ref, assumption_text="Assume X > 0. ")

    record = build_failure_record(
        store,
        failure_id="f1",
        origin=FailureOrigin.INJECTED,
        failing_artifact_id="t",
        evidence=[],
        source_checkpoint_id="cp1",
        injected_fault=fault,
    )

    # Right after injection, the real dependency on a is still intact
    # (fault_injection's own bug-1 fix) -- so nothing is missing yet.
    assert record.difficulty.current_missing_relations_vs_reference == 0


def test_build_failure_record_reference_baseline_includes_inferred_dependencies() -> None:
    """Repro from the user's review: deriving the reference set from
    injected_fault.pre_injection_refs[i].read_input_versions alone misses
    any INFERRED dependency the artifact had right before injection
    (read_input_versions can only ever hold EXPLICIT ones) --
    build_failure_record must read InjectedFault.
    pre_injection_dependency_ids_by_artifact instead, which captures both."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a")
    a_ref = store.latest("a").ref
    store.add_version("t", ArtifactKind.CLAIM, content="Assume X > 0. Conclusion.", produced_by="seed")
    t_ref = store.latest("t").ref
    store.add_inferred_edge(a_ref, t_ref, reason="t actually cites a's label")
    fault = FaultInjector().inject_missing_assumption(store, t_ref, assumption_text="Assume X > 0. ")

    record = build_failure_record(
        store,
        failure_id="f1",
        origin=FailureOrigin.INJECTED,
        failing_artifact_id="t",
        evidence=[],
        source_checkpoint_id="cp1",
        injected_fault=fault,
    )
    # The INFERRED dependency on a was carried forward by the injector's
    # own fix, so nothing is missing yet.
    assert record.difficulty.current_missing_relations_vs_reference == 0

    # Now drop it entirely (a regeneration that forgets to re-declare any
    # dependency at all) -- this must register as a real miss, proving the
    # baseline really did include "a" and was not just vacuously empty.
    store.add_version("t", ArtifactKind.CLAIM, content="Assume X > 0. Conclusion, rewritten.", read_input_versions=[])
    record_after_drop = build_failure_record(
        store,
        failure_id="f2",
        origin=FailureOrigin.INJECTED,
        failing_artifact_id="t",
        evidence=[],
        source_checkpoint_id="cp1",
        injected_fault=fault,
    )
    assert record_after_drop.difficulty.current_missing_relations_vs_reference == 1


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
    """``outcome_overrides`` maps ``artifact_id -> {constraint_id:
    outcome}`` for PER-CONSTRAINT control within a single artifact -- a
    fifth review's repro needed one artifact checked against two
    DIFFERENT constraints with two DIFFERENT outcomes at once (e.g.
    format=PASS + computation=UNKNOWN on the same artifact), which
    ``fail_ids``/``unknown_ids`` alone cannot express since they set one
    outcome for every constraint on a given artifact_id. Falls back to
    the fail_ids/unknown_ids/PASS default for any constraint not named in
    ``outcome_overrides``.

    Every UNKNOWN outcome -- default or overridden -- gets a real
    ``unknown_reason``. A previous version of this fake built
    ``Evidence(outcome=UNKNOWN, ...)`` with no ``unknown_reason`` at all,
    which ``Evidence.model_post_init`` actually rejects
    (``pydantic.ValidationError``) -- that exception was never visible in
    a failing test because ``ToolExecutor``/``repair_operations`` already
    catch a raising ``Validator.check()`` and fall back to empty
    evidence, which happened to still classify as INCONCLUSIVE either
    way. That made the old UNKNOWN path "pass" for the wrong reason (a
    masked crash, not a real UNKNOWN verdict) -- fixed here so a fake
    UNKNOWN verdict is a real, valid one, exactly like production
    Validators must produce."""

    def __init__(
        self,
        fail_ids: FrozenSet[str] = frozenset(),
        unknown_ids: FrozenSet[str] = frozenset(),
        outcome_overrides: Optional[Dict[str, Dict[str, CheckOutcome]]] = None,
    ) -> None:
        self.fail_ids = set(fail_ids)
        self.unknown_ids = set(unknown_ids)
        self.outcome_overrides = outcome_overrides or {}
        self.config: Dict[str, bool] = {}

    def check(
        self, artifact: ArtifactRef, constraints: List[Constraint], context: Optional[Dict[str, Any]] = None
    ) -> List[Evidence]:
        if artifact.artifact_id in self.fail_ids:
            default_outcome = CheckOutcome.FAIL
        elif artifact.artifact_id in self.unknown_ids:
            default_outcome = CheckOutcome.UNKNOWN
        else:
            default_outcome = CheckOutcome.PASS
        per_constraint = self.outcome_overrides.get(artifact.artifact_id, {})
        evidence: List[Evidence] = []
        for c in constraints:
            outcome = per_constraint.get(c.constraint_id, default_outcome)
            evidence.append(
                Evidence(
                    constraint_id=c.constraint_id,
                    outcome=outcome,
                    checked_artifact=artifact,
                    detail="fake",
                    unknown_reason=(UnknownReason.NOT_REPRESENTABLE if outcome == CheckOutcome.UNKNOWN else None),
                )
            )
        return evidence


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


def _format_and_computation_constraints_for_b(ref: ArtifactRef) -> List[Constraint]:
    """"b" gets TWO independent, REQUIRED constraints (format/computation)
    instead of the default one-per-artifact -- the fixture a fifth
    review's multi-constraint repro needs. Every other artifact still
    gets the default single constraint."""
    if ref.artifact_id != "b":
        return _one_constraint_per_artifact(ref)
    return [
        Constraint(constraint_id="format:b", kind=ConstraintKind.SCHEMA_CONTRACT, description="format check"),
        Constraint(constraint_id="computation:b", kind=ConstraintKind.RECOMPUTE, description="computation check"),
    ]


def _record_pass(store: ArtifactGraphStore, ref: ArtifactRef) -> None:
    """Every fixture artifact gets an explicit, CONFIRMED PASS at
    creation -- never left merely unchecked. A review caught that
    _is_confirmed_healthy requires an actual live PASS, not just "not
    stale, no FAIL": a freshly-created artifact with zero evidence is
    UNKNOWN/unverified, not healthy, so any fixture that wants an
    artifact to behave as a genuine pre-existing-healthy baseline must
    say so explicitly, exactly like a real pipeline would have checked it
    at least once before now."""
    store.record_evidence(
        Evidence(constraint_id=f"c:{ref.artifact_id}", outcome=CheckOutcome.PASS, checked_artifact=ref, detail="baseline")
    )


def _checkpoint_with_sections(*artifact_ids: str):
    store = ArtifactGraphStore()
    refs = {}
    for artifact_id in artifact_ids:
        store.add_version(artifact_id, ArtifactKind.SECTION, content=None)
        ref = store.latest(artifact_id).ref
        _record_pass(store, ref)
        refs[artifact_id] = ref
    return store.save_checkpoint(label="pre-repair"), refs


def _checkpoint_with_dependency(parent_id: str, child_id: str):
    store = ArtifactGraphStore()
    store.add_version(parent_id, ArtifactKind.SECTION, content=None)
    parent_ref = store.latest(parent_id).ref
    _record_pass(store, parent_ref)
    store.add_version(child_id, ArtifactKind.SECTION, content=None, read_input_versions=[parent_ref])
    child_ref = store.latest(child_id).ref
    _record_pass(store, child_ref)
    return store.save_checkpoint(label="pre-repair"), {parent_id: parent_ref, child_id: child_ref}


def _fail_evidence(ref: ArtifactRef) -> Evidence:
    return Evidence(constraint_id="c", outcome=CheckOutcome.FAIL, checked_artifact=ref, detail="bad")


class _AlsoRegenerate:
    """Fake policy for the bug-9 repro: reacts to a FAIL exactly like
    _ActOnAllFailures, but also throws in one extra action against an
    unrelated, not-yet-failing artifact -- a stand-in for a WIDE repair
    action (e.g. whole-document regeneration) that regenerates more than
    strictly necessary."""

    def __init__(self, extra_ref: ArtifactRef) -> None:
        self.extra_ref = extra_ref

    def decide(self, evidence, dependency_edges, budget) -> List[RepairAction]:
        if budget.remaining <= 0:
            return []
        fail_refs = [item.checked_artifact for item in evidence if item.outcome == CheckOutcome.FAIL]
        if not fail_refs:
            return []
        actions = [
            RepairAction(action_type=RepairActionType.REGENERATE_SECTION, target_artifacts=[ref], use_diagnostic_feedback=False)
            for ref in fail_refs
        ]
        actions.append(
            RepairAction(
                action_type=RepairActionType.REGENERATE_SECTION,
                target_artifacts=[self.extra_ref],
                use_diagnostic_feedback=False,
            )
        )
        return actions


def _run_branch(
    checkpoint,
    trigger_refs: List[ArtifactRef],
    *,
    producer,
    validator,
    constraint_provider=_one_constraint_per_artifact,
    budget=None,
    policy=None,
) -> BranchExecutionResult:
    runner = BranchingRunner(
        producer_registry={ArtifactKind.SECTION: producer},
        validator_factory=lambda store: validator,
        constraint_provider=constraint_provider,
        model_name="fake-model",
    )
    result = runner.run_branches(
        checkpoint,
        [BranchSpec("A", policy if policy is not None else _ActOnAllFailures())],
        downstream_policy_factory=_NothingPolicy,
        triggering_evidence=[_fail_evidence(ref) for ref in trigger_refs],
        budget_factory=(budget if budget is not None else (lambda: Budget(cap=1000))),
    )
    return result.branches[0]


def _run_single_branch(checkpoint, refs, *, producer, validator, budget=None) -> BranchExecutionResult:
    return _run_branch(checkpoint, [refs["a"]], producer=producer, validator=validator, budget=budget)


def test_repair_gain_score_resolved_case_has_full_gain_and_no_damage() -> None:
    checkpoint, refs = _checkpoint_with_sections("a", "b")
    branch = _run_single_branch(checkpoint, refs, producer=_FakeProducer(), validator=_FakeValidator())

    score = compute_repair_gain_score(
        checkpoint,
        branch,
        target_artifact_ids={"a"},
        triggering_evidence=[_fail_evidence(refs["a"])],
        constraint_provider=_one_constraint_per_artifact,
    )

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

    score = compute_repair_gain_score(
        checkpoint,
        branch,
        target_artifact_ids={"a"},
        triggering_evidence=[_fail_evidence(refs["a"])],
        constraint_provider=_one_constraint_per_artifact,
    )

    assert score.gain == 0.0
    assert score.outcome_counts == {RepairOutcome.STILL_FAILING.value: 1}


def test_repair_gain_score_empty_target_set_leaves_gain_none_with_reason() -> None:
    """The ONLY case gain is None: target_artifact_ids itself is empty --
    never a 0-sized internal denominator derived from the branch's own
    history (see the next test for that case, which is now a real 0.0)."""
    checkpoint, refs = _checkpoint_with_sections("a", "b")
    branch = _run_single_branch(checkpoint, refs, producer=_FakeProducer(), validator=_FakeValidator())

    score = compute_repair_gain_score(
        checkpoint,
        branch,
        target_artifact_ids=set(),
        triggering_evidence=[],
        constraint_provider=_one_constraint_per_artifact,
    )

    assert score.gain is None
    assert score.not_implemented_reason is not None
    assert score.outcome_counts == {}


def test_repair_gain_score_unresolved_target_is_a_real_zero_not_a_fabricated_none() -> None:
    """Repro adjacent to the main bug: with budget exhausted before
    anything ran, the target ("a") was never resolved -- gain must be a
    real 0.0 (0 of 1 target resolved), landing in the
    LEFT_STALE_OR_UNRESOLVED bucket, never silently reported as None just
    because the branch's own cascade never got around to it."""
    checkpoint, refs = _checkpoint_with_sections("a", "b")
    branch = _run_single_branch(
        checkpoint, refs, producer=_FakeProducer(), validator=_FakeValidator(), budget=lambda: Budget(cap=0)
    )

    score = compute_repair_gain_score(
        checkpoint,
        branch,
        target_artifact_ids={"a"},
        triggering_evidence=[_fail_evidence(refs["a"])],
        constraint_provider=_one_constraint_per_artifact,
    )

    assert score.gain == 0.0
    assert score.outcome_counts == {LEFT_STALE_OR_UNRESOLVED: 1}
    assert score.not_implemented_reason is None


def test_repair_gain_score_orphaned_dependent_counts_against_a_branch_that_ignored_it() -> None:
    """Main repro from the user's review: branch A resolves the original
    trigger ("a") but orphans its dependent ("b") as left_stale (no
    constraint registered for it at all); branch B resolves both. Using
    the COMMON target set (computed across both branches), branch A's
    gain must reflect the orphaned b, not report a misleadingly perfect
    1.0 just because its own cascade never discovered b was a problem."""
    checkpoint, refs = _checkpoint_with_dependency("a", "b")
    trigger = [refs["a"]]

    branch_a = _run_branch(
        checkpoint,
        trigger,
        producer=_FakeProducer(),
        validator=_FakeValidator(),
        constraint_provider=lambda ref: [] if ref.artifact_id == "b" else _one_constraint_per_artifact(ref),
    )
    branch_b = _run_branch(
        checkpoint,
        trigger,
        producer=_FakeProducer(),
        validator=_FakeValidator(),
        constraint_provider=_one_constraint_per_artifact,
    )

    # Sanity-check the premise: branch A really did leave b unresolved.
    assert "b" not in _merge_final_outcomes(branch_a)
    assert "b" in _merge_final_outcomes(branch_b)

    trigger_evidence = [_fail_evidence(refs["a"])]
    comparison = BranchComparisonResult(source_checkpoint_id=checkpoint.checkpoint_id, branches=[branch_a, branch_b])
    target_ids = compute_common_target_artifact_ids(comparison, triggering_evidence=trigger_evidence)
    assert target_ids == {"a", "b"}

    score_a = compute_repair_gain_score(
        checkpoint,
        branch_a,
        target_artifact_ids=target_ids,
        triggering_evidence=trigger_evidence,
        constraint_provider=_one_constraint_per_artifact,
    )
    score_b = compute_repair_gain_score(
        checkpoint,
        branch_b,
        target_artifact_ids=target_ids,
        triggering_evidence=trigger_evidence,
        constraint_provider=_one_constraint_per_artifact,
    )

    assert score_a.gain == 0.5
    assert score_a.outcome_counts == {RepairOutcome.RESOLVED.value: 1, LEFT_STALE_OR_UNRESOLVED: 1}
    assert score_b.gain == 1.0
    assert score_b.outcome_counts == {RepairOutcome.RESOLVED.value: 2}

    # Repro from the second review: "b" is also in target_ids (above), but
    # damage must be decided from the ORIGINAL failure ("a" alone), not
    # from target_ids -- so "b"'s real final state still gets evaluated
    # for both branches, regardless of its target_ids membership. Branch
    # A left "b" genuinely stale; branch B genuinely resolved it.
    assert score_a.damage.newly_broken_artifacts == ["b"]
    assert score_a.damage.preserved_fraction == 0.0
    assert score_b.damage.newly_broken_artifacts == []
    assert score_b.damage.preserved_fraction == 1.0


def test_damage_of_a_branch_is_not_hidden_by_a_siblings_collateral_sweep() -> None:
    """Main repro from the second review: "a" fails, "b" is completely
    unrelated and healthy. A narrow branch resolves only "a"; a wide
    sibling branch's own action ALSO regenerates "b" (unprompted) and
    breaks it. compute_common_target_artifact_ids correctly unions "b"
    into the COMMON target (since the wide branch's history names it) --
    but that common set must never be reused as damage's exclusion set:
    doing so let the wide branch's own collateral damage to "b" disappear
    from ITS OWN score, merely because "b" also collided with the common
    target computed across both branches."""
    checkpoint, refs = _checkpoint_with_sections("a", "b")
    trigger_evidence = [_fail_evidence(refs["a"])]

    branch_narrow = _run_branch(checkpoint, [refs["a"]], producer=_FakeProducer(), validator=_FakeValidator())
    branch_wide = _run_branch(
        checkpoint,
        [refs["a"]],
        producer=_FakeProducer(),
        validator=_FakeValidator(fail_ids=frozenset({"b"})),
        policy=_AlsoRegenerate(refs["b"]),
    )

    # Sanity-check the premise: "b" was never part of the original
    # failure, yet the wide branch's own history now names it.
    assert "b" not in _merge_final_outcomes(branch_narrow)
    assert "b" in _merge_final_outcomes(branch_wide)

    comparison = BranchComparisonResult(
        source_checkpoint_id=checkpoint.checkpoint_id, branches=[branch_narrow, branch_wide]
    )
    target_ids = compute_common_target_artifact_ids(comparison, triggering_evidence=trigger_evidence)
    assert target_ids == {"a", "b"}  # "b" is swept in purely via the wide branch's own history

    score_narrow = compute_repair_gain_score(
        checkpoint,
        branch_narrow,
        target_artifact_ids=target_ids,
        triggering_evidence=trigger_evidence,
        constraint_provider=_one_constraint_per_artifact,
    )
    score_wide = compute_repair_gain_score(
        checkpoint,
        branch_wide,
        target_artifact_ids=target_ids,
        triggering_evidence=trigger_evidence,
        constraint_provider=_one_constraint_per_artifact,
    )

    # Narrow never touched "b" -- it stayed healthy, and damage must say so.
    assert score_narrow.damage.newly_broken_artifacts == []
    assert score_narrow.damage.preserved_fraction == 1.0
    # Wide actually broke the previously-healthy "b" -- damage must catch
    # this even though "b" is also (correctly) inside target_artifact_ids.
    assert score_wide.damage.newly_broken_artifacts == ["b"]
    assert score_wide.damage.preserved_fraction == 0.0


def test_repair_gain_score_does_not_trust_a_resolved_verdict_the_final_state_has_since_invalidated() -> None:
    """Second repro from the second review: "b" is resolved in round 1,
    then something in a later round regenerates "a" again, which
    re-invalidates "b" via ordinary staleness propagation -- but nothing
    ever re-checks "b" afterward. _merge_final_outcomes's last-write-wins
    rule still reads back RESOLVED for "b" (round 2 never touched it at
    all), even though the branch's own final_checkpoint shows "b"
    genuinely stale again. Gain must trust the final checkpoint's ground
    truth over a now-superseded historical verdict, and must leave this
    as unresolved -- never promoted to FAIL (RepairOutcome has no real
    verdict for it; only the sentinel bucket is honest)."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.CLAIM, content="b1", read_input_versions=[a_ref])
    b_ref = store.latest("b").ref
    before_checkpoint = store.save_checkpoint(label="pre-repair")

    # Round 1: b is regenerated and passes -- genuinely RESOLVED at the time.
    store.add_version("b", ArtifactKind.CLAIM, content="b-fixed", read_input_versions=[a_ref])
    fixed_b_ref = store.latest("b").ref
    store.record_evidence(
        Evidence(constraint_id="c:b", outcome=CheckOutcome.PASS, checked_artifact=fixed_b_ref, detail="fine")
    )
    round_1 = RepairRound(
        round_index=0, policy_label="initial", executions=[_minimal_execution({"b": RepairOutcome.RESOLVED})]
    )

    # Something else, in a later round, regenerates "a" again -- this
    # re-invalidates "b" through ordinary staleness propagation, but
    # nothing re-checks "b" afterward.
    store.add_version("a", ArtifactKind.DEFINITION, content="a2", read_input_versions=[])
    assert store.is_stale("b") is True  # sanity-check the premise is real

    round_2 = RepairRound(
        round_index=1, policy_label="downstream", executions=[_minimal_execution({"a": RepairOutcome.RESOLVED})]
    )
    final_checkpoint = store.save_checkpoint(label="post-repair")
    branch = BranchExecutionResult(
        branch_id="A",
        source_checkpoint_id=before_checkpoint.checkpoint_id,
        policy_label="A",
        rounds=[round_1, round_2],
        stopped_reason="unresolved_remaining",
        final_checkpoint=final_checkpoint,
    )

    # Sanity-check the premise: the raw merge still says RESOLVED -- that
    # stale ledger entry is exactly what must not be trusted blindly.
    assert _merge_final_outcomes(branch)["b"] == RepairOutcome.RESOLVED

    score = compute_repair_gain_score(
        before_checkpoint,
        branch,
        target_artifact_ids={"b"},
        triggering_evidence=[_fail_evidence(b_ref)],
        constraint_provider=_one_constraint_per_artifact,
    )

    assert score.gain == 0.0
    assert score.outcome_counts == {LEFT_STALE_OR_UNRESOLVED: 1}


def test_repair_gain_score_already_healthy_untouched_counts_toward_gain() -> None:
    """Main repro from a third review: "a" fails, "b" already has a
    valid PASS recorded against it. A narrow branch correctly repairs
    only "a" and leaves "b" completely alone; a wide sibling branch's
    action redundantly regenerates BOTH "a" and "b" (both end up fine).
    compute_common_target_artifact_ids correctly pulls "b" into the
    COMMON target purely via the wide branch's own history -- but the
    narrow branch never touched "b" at all, and "b" was never actually a
    problem, so it must not be scored as if the narrow branch had ignored
    a real one."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.SECTION, content="orig-a")
    store.add_version("b", ArtifactKind.SECTION, content="orig-b")
    b_ref = store.latest("b").ref
    store.record_evidence(
        Evidence(constraint_id="c:b", outcome=CheckOutcome.PASS, checked_artifact=b_ref, detail="already fine")
    )
    checkpoint = store.save_checkpoint()
    refs = {"a": store.latest("a").ref, "b": b_ref}
    trigger_evidence = [_fail_evidence(refs["a"])]

    branch_narrow = _run_branch(checkpoint, [refs["a"]], producer=_FakeProducer(), validator=_FakeValidator())
    branch_wide = _run_branch(
        checkpoint,
        [refs["a"]],
        producer=_FakeProducer(),
        validator=_FakeValidator(),
        policy=_AlsoRegenerate(refs["b"]),
    )
    # Sanity-check the premise: the narrow branch truly never touched "b".
    assert "b" not in _merge_final_outcomes(branch_narrow)
    assert "b" in _merge_final_outcomes(branch_wide)

    comparison = BranchComparisonResult(
        source_checkpoint_id=checkpoint.checkpoint_id, branches=[branch_narrow, branch_wide]
    )
    target_ids = compute_common_target_artifact_ids(comparison, triggering_evidence=trigger_evidence)
    assert target_ids == {"a", "b"}  # "b" is swept in purely via the wide branch's own history

    score_narrow = compute_repair_gain_score(
        checkpoint,
        branch_narrow,
        target_artifact_ids=target_ids,
        triggering_evidence=trigger_evidence,
        constraint_provider=_one_constraint_per_artifact,
    )
    score_wide = compute_repair_gain_score(
        checkpoint,
        branch_wide,
        target_artifact_ids=target_ids,
        triggering_evidence=trigger_evidence,
        constraint_provider=_one_constraint_per_artifact,
    )

    assert score_narrow.outcome_counts == {RepairOutcome.RESOLVED.value: 1, ALREADY_HEALTHY_UNTOUCHED: 1}
    assert score_narrow.gain == 1.0
    assert score_wide.outcome_counts == {RepairOutcome.RESOLVED.value: 2}
    assert score_wide.gain == 1.0


def test_repair_gain_score_already_healthy_untouched_does_not_apply_to_a_genuinely_broken_target() -> None:
    """The guard that keeps the new sentinel from over-reaching: a target
    absent from this branch's own history that IS genuinely broken in
    this branch's final state (the original orphaned-dependent repro,
    not a redundant touch) must stay LEFT_STALE_OR_UNRESOLVED, never be
    reclassified as "already healthy" just because it was never directly
    attempted."""
    checkpoint, refs = _checkpoint_with_dependency("a", "b")
    trigger = [refs["a"]]
    def narrow_provider(ref: ArtifactRef) -> List[Constraint]:
        return [] if ref.artifact_id == "b" else _one_constraint_per_artifact(ref)

    branch = _run_branch(
        checkpoint,
        trigger,
        producer=_FakeProducer(),
        validator=_FakeValidator(),
        constraint_provider=narrow_provider,
    )
    assert "b" not in _merge_final_outcomes(branch)

    score = compute_repair_gain_score(
        checkpoint,
        branch,
        target_artifact_ids={"a", "b"},
        triggering_evidence=[_fail_evidence(refs["a"])],
        constraint_provider=narrow_provider,
    )

    assert score.outcome_counts == {RepairOutcome.RESOLVED.value: 1, LEFT_STALE_OR_UNRESOLVED: 1}
    assert score.gain == 0.5


def test_repair_gain_score_unknown_untouched_does_not_count_toward_gain() -> None:
    """Main repro from a fourth review: "a" fails, "b" starts on a live
    UNKNOWN (checked, but could not be evaluated either way -- never
    actually confirmed healthy). A narrow branch repairs only "a" and
    leaves "b" untouched (still UNKNOWN); a wide sibling branch
    re-checks "b" too, and the recheck is ALSO UNKNOWN -- an honest
    INCONCLUSIVE, since neither branch ever confirms "b" is fine.
    Absence of a FAIL is not confirmation of a PASS: the branch that did
    LESS (never re-checked "b" at all) must not score better than the
    one that honestly re-verified and still could not confirm it."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.SECTION, content="orig-a")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.SECTION, content="orig-b")
    b_ref = store.latest("b").ref
    store.record_evidence(
        Evidence(
            constraint_id="c:b",
            outcome=CheckOutcome.UNKNOWN,
            checked_artifact=b_ref,
            detail="could not evaluate",
            unknown_reason=UnknownReason.NOT_REPRESENTABLE,
        )
    )
    checkpoint = store.save_checkpoint()
    trigger_evidence = [_fail_evidence(a_ref)]

    branch_narrow = _run_branch(checkpoint, [a_ref], producer=_FakeProducer(), validator=_FakeValidator())
    branch_wide = _run_branch(
        checkpoint,
        [a_ref],
        producer=_FakeProducer(),
        validator=_FakeValidator(unknown_ids=frozenset({"b"})),
        policy=_AlsoRegenerate(b_ref),
    )
    # Sanity-check the premise: narrow truly never touched "b"; wide did,
    # and the recheck itself came back inconclusive, not a confirmed pass.
    assert "b" not in _merge_final_outcomes(branch_narrow)
    assert _merge_final_outcomes(branch_wide)["b"] == RepairOutcome.INCONCLUSIVE

    comparison = BranchComparisonResult(
        source_checkpoint_id=checkpoint.checkpoint_id, branches=[branch_narrow, branch_wide]
    )
    target_ids = compute_common_target_artifact_ids(comparison, triggering_evidence=trigger_evidence)
    assert target_ids == {"a", "b"}

    score_narrow = compute_repair_gain_score(
        checkpoint,
        branch_narrow,
        target_artifact_ids=target_ids,
        triggering_evidence=trigger_evidence,
        constraint_provider=_one_constraint_per_artifact,
    )
    score_wide = compute_repair_gain_score(
        checkpoint,
        branch_wide,
        target_artifact_ids=target_ids,
        triggering_evidence=trigger_evidence,
        constraint_provider=_one_constraint_per_artifact,
    )

    # "b" must land in the same kind of non-success bucket for both --
    # never ALREADY_HEALTHY_UNTOUCHED, since nobody ever confirmed it.
    assert score_narrow.outcome_counts == {RepairOutcome.RESOLVED.value: 1, LEFT_STALE_OR_UNRESOLVED: 1}
    assert score_narrow.gain == 0.5
    assert score_wide.outcome_counts == {RepairOutcome.RESOLVED.value: 1, RepairOutcome.INCONCLUSIVE.value: 1}
    assert score_wide.gain == 0.5


def test_damage_excludes_an_unconfirmed_artifact_from_the_healthy_before_baseline() -> None:
    """The damage-side twin of the gain repro above: "b" starts on a live
    UNKNOWN (never confirmed healthy) and is left completely untouched.
    It must not be checked for damage at all -- there is no confirmed
    healthy baseline to preserve -- so it must never appear in
    newly_broken_artifacts, and the checked denominator must not count
    it either."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.SECTION, content="orig-a")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.SECTION, content="orig-b")
    b_ref = store.latest("b").ref
    store.record_evidence(
        Evidence(
            constraint_id="c:b",
            outcome=CheckOutcome.UNKNOWN,
            checked_artifact=b_ref,
            detail="could not evaluate",
            unknown_reason=UnknownReason.NOT_REPRESENTABLE,
        )
    )
    checkpoint = store.save_checkpoint()

    branch = _run_branch(checkpoint, [a_ref], producer=_FakeProducer(), validator=_FakeValidator())
    score = compute_repair_gain_score(
        checkpoint,
        branch,
        target_artifact_ids={"a"},
        triggering_evidence=[_fail_evidence(a_ref)],
        constraint_provider=_one_constraint_per_artifact,
    )

    assert "b" not in score.damage.newly_broken_artifacts
    assert score.damage.preserved_fraction is None
    assert score.damage.not_implemented_reason is not None


def test_build_candidate_action_record_end_to_end() -> None:
    checkpoint, refs = _checkpoint_with_sections("a", "b")
    branch = _run_single_branch(checkpoint, refs, producer=_FakeProducer(), validator=_FakeValidator())

    record = build_candidate_action_record(
        record_id="rec1",
        failure_id="f1",
        before_checkpoint=checkpoint,
        branch=branch,
        target_artifact_ids={"a"},
        triggering_evidence=[_fail_evidence(refs["a"])],
        constraint_provider=_one_constraint_per_artifact,
    )

    assert record.failure_id == "f1"
    assert len(record.actions) == 1
    assert record.actions[0].action_type == RepairActionType.REGENERATE_SECTION
    assert record.outcome.gain == 1.0
    assert record.rounds_executed == len(branch.rounds)


def test_build_candidate_action_record_keeps_every_initiating_action_not_just_the_first() -> None:
    """Repro from the user's review: a and b both fail initially, and one
    decide() call returns 2 independent actions for round 0 -- the record
    must keep both, since outcome already reflects the combined effect of
    both (building the record from only actions[0] would misattribute b's
    share of the result to a's action alone)."""
    checkpoint, refs = _checkpoint_with_sections("a", "b")
    branch = _run_branch(checkpoint, [refs["a"], refs["b"]], producer=_FakeProducer(), validator=_FakeValidator())

    assert len(branch.rounds[0].executions) == 2  # sanity: both actions really did run in round 0

    record = build_candidate_action_record(
        record_id="rec1",
        failure_id="f1",
        before_checkpoint=checkpoint,
        branch=branch,
        target_artifact_ids={"a", "b"},
        triggering_evidence=[_fail_evidence(refs["a"]), _fail_evidence(refs["b"])],
        constraint_provider=_one_constraint_per_artifact,
    )

    assert len(record.actions) == 2
    assert {action.target_artifacts[0].artifact_id for action in record.actions} == {"a", "b"}
    assert record.outcome.gain == 1.0
    assert record.outcome.outcome_counts == {RepairOutcome.RESOLVED.value: 2}


def test_build_candidate_action_record_raises_when_branch_never_executed_anything() -> None:
    checkpoint, refs = _checkpoint_with_sections("a", "b")
    branch = _run_single_branch(
        checkpoint, refs, producer=_FakeProducer(), validator=_FakeValidator(), budget=lambda: Budget(cap=0)
    )
    with pytest.raises(ValueError):
        build_candidate_action_record(
            record_id="rec1",
            failure_id="f1",
            before_checkpoint=checkpoint,
            branch=branch,
            target_artifact_ids={"a"},
            triggering_evidence=[_fail_evidence(refs["a"])],
            constraint_provider=_one_constraint_per_artifact,
        )


# ---------------------------------------------------------------------------
# damage detection in isolation: confirmed FAIL/stale, never a bare content diff
# ---------------------------------------------------------------------------


def test_damage_ignores_a_harmless_content_change_with_no_real_error() -> None:
    """Repro from the user's review: a non-target artifact's content
    legitimately changing (e.g. a successful regeneration whose wording
    simply differs) must NOT count as damage on its own -- only a
    confirmed new FAIL/stale state does."""
    from src.core.failure_dataset import _compute_damage

    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.SECTION, content="orig-a")
    store.add_version("b", ArtifactKind.SECTION, content="orig-b")
    before = store.save_checkpoint()

    branch_store = ArtifactGraphStore.from_checkpoint(before, branch_id="branch-harmless")
    a_ref = branch_store.latest("a").ref
    branch_store.add_version("a", ArtifactKind.SECTION, content="fixed-a", read_input_versions=[a_ref])
    # b's content changes too, but it is checked and PASSES -- a harmless
    # refresh, not damage.
    branch_store.add_version("b", ArtifactKind.SECTION, content="refreshed-b-wording", read_input_versions=[])
    b_ref_v2 = branch_store.latest("b").ref
    branch_store.record_evidence(
        Evidence(constraint_id="c:b", outcome=CheckOutcome.PASS, checked_artifact=b_ref_v2, detail="fine")
    )
    after = branch_store.save_checkpoint()

    damage = _compute_damage(
        before, after, healthy_before_artifact_ids={"b"}, constraint_provider=_one_constraint_per_artifact
    )

    assert damage.newly_broken_artifacts == []
    assert damage.preserved_fraction == 1.0


def test_damage_flags_a_confirmed_new_failure_even_with_unchanged_content() -> None:
    """The flip side: content staying byte-identical does not excuse a
    non-target artifact that a fresh check now reports FAIL -- a bare
    content diff would have missed this entirely."""
    from src.core.failure_dataset import _compute_damage

    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.SECTION, content="orig-a")
    store.add_version("b", ArtifactKind.SECTION, content="orig-b")
    before = store.save_checkpoint()

    branch_store = ArtifactGraphStore.from_checkpoint(before, branch_id="branch-broken")
    b_ref = branch_store.latest("b").ref  # unchanged content, same version
    branch_store.record_evidence(
        Evidence(constraint_id="c:b", outcome=CheckOutcome.FAIL, checked_artifact=b_ref, detail="actually broken")
    )
    after = branch_store.save_checkpoint()

    damage = _compute_damage(
        before, after, healthy_before_artifact_ids={"b"}, constraint_provider=_one_constraint_per_artifact
    )

    assert damage.newly_broken_artifacts == ["b"]
    assert damage.preserved_fraction == 0.0


def test_damage_considers_every_non_target_artifact_even_ones_the_branch_touched() -> None:
    """The other half of the original bug: a non-target artifact that WAS
    swept into the repair's own cascade (and thus appears in
    final_outcome_by_artifact) must still be damage-checked -- "visited by
    this branch" and "was this artifact supposed to be touched at all" are
    different questions."""
    checkpoint, refs = _checkpoint_with_dependency("a", "b")
    # b starts out healthy; only a is triggering. The validator fails b
    # once it gets swept into a's cascade (a stand-in for "a wide repair
    # broke something that used to be fine").
    branch = _run_branch(
        checkpoint, [refs["a"]], producer=_FakeProducer(), validator=_FakeValidator(fail_ids=frozenset({"b"}))
    )

    score = compute_repair_gain_score(
        checkpoint,
        branch,
        target_artifact_ids={"a"},
        triggering_evidence=[_fail_evidence(refs["a"])],
        constraint_provider=_one_constraint_per_artifact,
    )

    assert "b" in score.damage.newly_broken_artifacts
    assert score.damage.preserved_fraction == 0.0


def test_damage_excludes_an_artifact_that_was_already_broken_before_the_repair() -> None:
    """Repro from a third review: "b" is already stale in the
    before_checkpoint, for a reason that predates this repair entirely
    (x, something b depends on, changed again after b was built from
    it) and has nothing to do with "a". Repairing only "a" leaves "b"
    completely untouched -- same content, same version, same stale
    status throughout -- so it must not be reported as "newly broken";
    it was never healthy to begin with."""
    store = ArtifactGraphStore()
    store.add_version("x", ArtifactKind.SECTION, content="x")
    x_ref = store.latest("x").ref
    store.add_version("b", ArtifactKind.SECTION, content="b depends on x", read_input_versions=[x_ref])
    store.add_version("a", ArtifactKind.SECTION, content="orig-a")
    # x changes again, after b was already built from it -- b is now
    # genuinely, pre-existingly stale, for a reason unrelated to a's
    # own repair.
    store.add_version("x", ArtifactKind.SECTION, content="x changed", read_input_versions=[])
    x_ref_2 = store.latest("x").ref
    _record_pass(store, x_ref_2)  # x itself is confirmed healthy -- only b is pre-broken
    assert store.is_stale("b") is True  # sanity-check the premise is real

    before_checkpoint = store.save_checkpoint()
    a_ref = store.latest("a").ref
    branch = _run_branch(before_checkpoint, [a_ref], producer=_FakeProducer(), validator=_FakeValidator())

    score = compute_repair_gain_score(
        before_checkpoint,
        branch,
        target_artifact_ids={"a"},
        triggering_evidence=[_fail_evidence(a_ref)],
        constraint_provider=_one_constraint_per_artifact,
    )

    assert score.damage.newly_broken_artifacts == []
    assert score.damage.preserved_fraction == 1.0


def test_repair_gain_score_already_healthy_untouched_requires_every_required_constraint_not_just_one() -> None:
    """Main repro from a fifth review: "b" has TWO required constraints
    (format/computation). format already has a valid PASS; computation
    is live UNKNOWN -- never actually confirmed either way. A narrow
    branch repairs only "a" and leaves "b" completely untouched (still
    format=PASS, computation=UNKNOWN); a wide sibling branch re-checks
    "b" too, and the recheck is ALSO UNKNOWN for computation --
    genuinely never confirmed by either branch.

    The old ``_is_confirmed_healthy``, ``any(outcome == PASS for ... in
    live_evidence(...))``, let the single passing "format" constraint
    stand in for the WHOLE artifact, so the narrow branch (which did
    LESS -- never even looked at computation again) scored a perfect
    gain=1.0 while the wide branch -- which re-checked and got an
    equally honest "still can't confirm computation" -- scored only 0.5.
    Neither branch ever confirmed ALL of "b"'s required constraints, so
    neither should get full credit for it; both must land on gain=0.5."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.SECTION, content="orig-a")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.SECTION, content="orig-b")
    b_ref = store.latest("b").ref
    store.record_evidence(
        Evidence(constraint_id="format:b", outcome=CheckOutcome.PASS, checked_artifact=b_ref, detail="format ok")
    )
    store.record_evidence(
        Evidence(
            constraint_id="computation:b",
            outcome=CheckOutcome.UNKNOWN,
            checked_artifact=b_ref,
            detail="could not evaluate",
            unknown_reason=UnknownReason.NOT_REPRESENTABLE,
        )
    )
    checkpoint = store.save_checkpoint()
    trigger_evidence = [_fail_evidence(a_ref)]

    branch_narrow = _run_branch(
        checkpoint,
        [a_ref],
        producer=_FakeProducer(),
        validator=_FakeValidator(),
        constraint_provider=_format_and_computation_constraints_for_b,
    )
    branch_wide = _run_branch(
        checkpoint,
        [a_ref],
        producer=_FakeProducer(),
        validator=_FakeValidator(outcome_overrides={"b": {"computation:b": CheckOutcome.UNKNOWN}}),
        constraint_provider=_format_and_computation_constraints_for_b,
        policy=_AlsoRegenerate(b_ref),
    )
    # Sanity-check the premise: narrow truly never touched "b"; wide did,
    # and the recheck itself came back inconclusive, not a confirmed pass.
    assert "b" not in _merge_final_outcomes(branch_narrow)
    assert _merge_final_outcomes(branch_wide)["b"] == RepairOutcome.INCONCLUSIVE

    comparison = BranchComparisonResult(
        source_checkpoint_id=checkpoint.checkpoint_id, branches=[branch_narrow, branch_wide]
    )
    target_ids = compute_common_target_artifact_ids(comparison, triggering_evidence=trigger_evidence)
    assert target_ids == {"a", "b"}

    score_narrow = compute_repair_gain_score(
        checkpoint,
        branch_narrow,
        target_artifact_ids=target_ids,
        triggering_evidence=trigger_evidence,
        constraint_provider=_format_and_computation_constraints_for_b,
    )
    score_wide = compute_repair_gain_score(
        checkpoint,
        branch_wide,
        target_artifact_ids=target_ids,
        triggering_evidence=trigger_evidence,
        constraint_provider=_format_and_computation_constraints_for_b,
    )

    # "b" must land in the same kind of non-success bucket for both --
    # never ALREADY_HEALTHY_UNTOUCHED, since neither branch ever
    # confirmed every one of its required constraints.
    assert score_narrow.outcome_counts == {RepairOutcome.RESOLVED.value: 1, LEFT_STALE_OR_UNRESOLVED: 1}
    assert score_narrow.gain == 0.5
    assert score_wide.outcome_counts == {RepairOutcome.RESOLVED.value: 1, RepairOutcome.INCONCLUSIVE.value: 1}
    assert score_wide.gain == 0.5


def test_repair_gain_score_already_healthy_untouched_does_not_apply_when_a_required_constraints_evidence_is_invalidated() -> None:
    """Second repro from the same fifth review: "b" has format=PASS (no
    real inputs of its own) and a separate, REQUIRED computation=PASS
    that depended on an unrelated "src" artifact's version
    (``Evidence.input_versions`` -- not a structural dependency of "b"
    itself via ``read_input_versions``). "src" later moves to a new
    version, which invalidates ONLY the computation Evidence -- it drops
    out of ``live_evidence()`` entirely, with nothing ever re-checking
    computation against "b" again -- while "b" itself never becomes
    ``is_stale`` (its own position in the dependency graph never moved).
    "b" is left completely untouched by the repair.

    ``_is_confirmed_healthy`` must not read "format is still a live PASS"
    as "b is fine": a REQUIRED constraint now has no live verdict at
    all, invalidated or simply never re-checked -- the same unconfirmed
    state as the UNKNOWN repro above, reached a different way."""
    store = ArtifactGraphStore()
    store.add_version("src", ArtifactKind.SECTION, content="src-v1")
    src_ref = store.latest("src").ref
    store.add_version("a", ArtifactKind.SECTION, content="orig-a")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.SECTION, content="orig-b")
    b_ref = store.latest("b").ref
    store.record_evidence(
        Evidence(constraint_id="format:b", outcome=CheckOutcome.PASS, checked_artifact=b_ref, detail="format ok")
    )
    store.record_evidence(
        Evidence(
            constraint_id="computation:b",
            outcome=CheckOutcome.PASS,
            checked_artifact=b_ref,
            detail="computed against src",
            input_versions=[src_ref],
        )
    )
    store.add_version("src", ArtifactKind.SECTION, content="src-v2", read_input_versions=[])
    assert store.is_stale("b") is False  # b's own graph position never moved
    assert [item.constraint_id for item in store.live_evidence("b")] == ["format:b"]  # computation dropped

    checkpoint = store.save_checkpoint()
    branch = _run_branch(checkpoint, [a_ref], producer=_FakeProducer(), validator=_FakeValidator())

    score = compute_repair_gain_score(
        checkpoint,
        branch,
        target_artifact_ids={"a", "b"},
        triggering_evidence=[_fail_evidence(a_ref)],
        constraint_provider=_format_and_computation_constraints_for_b,
    )
    assert score.outcome_counts == {RepairOutcome.RESOLVED.value: 1, LEFT_STALE_OR_UNRESOLVED: 1}
    assert score.gain == 0.5


def test_damage_excludes_a_target_whose_other_required_constraint_already_had_a_pre_existing_fail() -> None:
    """Third repro from the same fifth review, this time on the damage
    side: "b" has format=PASS and computation=FAIL, BOTH already live in
    the before_checkpoint, before the repair even starts -- "b" was never
    actually healthy, just partially passing. Repairing only "a" leaves
    "b" completely untouched -- same content, same version, same FAIL
    throughout. The old ``any(... PASS ...)`` still admitted "b" into the
    healthy-before baseline purely because of its format PASS, so
    leaving it exactly as broken as it always was got reported as this
    repair's own NEW damage."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.SECTION, content="orig-a")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.SECTION, content="orig-b")
    b_ref = store.latest("b").ref
    store.record_evidence(
        Evidence(constraint_id="format:b", outcome=CheckOutcome.PASS, checked_artifact=b_ref, detail="format ok")
    )
    store.record_evidence(
        Evidence(constraint_id="computation:b", outcome=CheckOutcome.FAIL, checked_artifact=b_ref, detail="bad calc")
    )
    before_checkpoint = store.save_checkpoint()

    branch = _run_branch(before_checkpoint, [a_ref], producer=_FakeProducer(), validator=_FakeValidator())
    score = compute_repair_gain_score(
        before_checkpoint,
        branch,
        target_artifact_ids={"a"},
        triggering_evidence=[_fail_evidence(a_ref)],
        constraint_provider=_format_and_computation_constraints_for_b,
    )

    assert "b" not in score.damage.newly_broken_artifacts
    assert score.damage.preserved_fraction is None
    assert score.damage.not_implemented_reason is not None


def test_damage_does_not_count_a_post_repair_unknown_as_preserved() -> None:
    """Fourth repro from the same fifth review: "b" starts CONFIRMED
    healthy (a single live PASS) and gets swept into a wide repair
    action alongside "a" -- but the recheck after regeneration comes
    back UNKNOWN (could not be evaluated), not a fresh PASS.
    ``_compute_damage`` previously counted everything NOT confirmed-
    broken as preserved by default, so this honestly-unverified
    post-repair state still scored ``preserved_fraction=1.0`` -- exactly
    as misleading as calling a never-confirmed artifact "healthy"
    elsewhere in this module. It must land in ``inconclusive_artifacts``
    instead, excluded from the preserved/broken ratio entirely."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.SECTION, content="orig-a")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.SECTION, content="orig-b")
    b_ref = store.latest("b").ref
    store.record_evidence(
        Evidence(constraint_id="c:b", outcome=CheckOutcome.PASS, checked_artifact=b_ref, detail="already fine")
    )
    before_checkpoint = store.save_checkpoint()

    branch = _run_branch(
        before_checkpoint,
        [a_ref],
        producer=_FakeProducer(),
        validator=_FakeValidator(unknown_ids=frozenset({"b"})),
        policy=_AlsoRegenerate(b_ref),
    )
    score = compute_repair_gain_score(
        before_checkpoint,
        branch,
        target_artifact_ids={"a"},
        triggering_evidence=[_fail_evidence(a_ref)],
        constraint_provider=_one_constraint_per_artifact,
    )

    assert "b" not in score.damage.newly_broken_artifacts
    assert "b" in score.damage.inconclusive_artifacts
    assert score.damage.preserved_fraction is None


def test_required_constraint_set_comes_from_the_spec_not_from_what_was_actually_checked() -> None:
    """Main repro from a sixth review: "b" has a format+computation spec
    (``_format_and_computation_constraints_for_b``). ``b@1`` had BOTH
    format and computation PASS; ``b`` is then regenerated to ``b@2`` and
    only format gets re-checked against it -- computation is simply never
    re-verified against the new version, not because the evaluation spec
    ever stopped requiring it.

    The previous round's own fix derived the required set from
    ``evidence_history`` filtered to the CURRENT version -- which reads as
    ``{format}`` alone for ``b@2``, since computation was only ever
    checked against the now-superseded ``b@1``. That silently let a
    branch leaving ``b`` in exactly this half-checked state score as if
    ``b`` were fully confirmed healthy. The required set must come from
    the spec (``constraint_provider``) instead, which still says
    ``{format, computation}`` regardless of what actually got
    (re-)checked -- so ``b`` must land in ``LEFT_STALE_OR_UNRESOLVED``,
    never ``ALREADY_HEALTHY_UNTOUCHED``."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.SECTION, content="orig-a")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.SECTION, content="b-v1")
    b_ref_v1 = store.latest("b").ref
    store.record_evidence(
        Evidence(constraint_id="format:b", outcome=CheckOutcome.PASS, checked_artifact=b_ref_v1, detail="ok")
    )
    store.record_evidence(
        Evidence(constraint_id="computation:b", outcome=CheckOutcome.PASS, checked_artifact=b_ref_v1, detail="ok")
    )
    # "b" moves to a new version, and only format gets re-checked against
    # it -- computation is simply never re-verified, not de-scoped.
    store.add_version("b", ArtifactKind.SECTION, content="b-v2", read_input_versions=[])
    b_ref_v2 = store.latest("b").ref
    store.record_evidence(
        Evidence(constraint_id="format:b", outcome=CheckOutcome.PASS, checked_artifact=b_ref_v2, detail="ok")
    )
    checkpoint = store.save_checkpoint()

    branch = _run_branch(
        checkpoint,
        [a_ref],
        producer=_FakeProducer(),
        validator=_FakeValidator(),
        constraint_provider=_format_and_computation_constraints_for_b,
    )
    assert "b" not in _merge_final_outcomes(branch)  # sanity: "b" was never touched by this branch

    score = compute_repair_gain_score(
        checkpoint,
        branch,
        target_artifact_ids={"a", "b"},
        triggering_evidence=[_fail_evidence(a_ref)],
        constraint_provider=_format_and_computation_constraints_for_b,
    )

    assert score.outcome_counts == {RepairOutcome.RESOLVED.value: 1, LEFT_STALE_OR_UNRESOLVED: 1}
    assert score.gain == 0.5


def test_resolved_verdict_is_downgraded_when_its_required_pass_is_invalidated_not_just_when_broken() -> None:
    """Second repro from the same sixth review: "b" is resolved in round
    1 (a fresh PASS, recorded with ``Evidence.input_versions=[x@1]`` --
    not a structural dependency of "b" itself). "x" is later updated to
    ``x@2``, which invalidates "b"'s PASS -- "b" is now neither stale nor
    FAILing (``_is_broken_in_final_state`` is ``False``, since "x" is not
    one of "b"'s structural dependencies), it is genuinely INCONCLUSIVE
    (``_is_confirmed_healthy`` is also ``False`` -- the required PASS has
    no live verdict at all). ``damage`` independently classifies "b" as
    inconclusive for this exact final state; ``gain`` must not keep
    trusting the stale ``RESOLVED`` ledger entry for the same artifact
    just because nothing currently FAILS -- that would have ``gain`` and
    ``damage`` contradicting each other about the identical artifact."""
    store = ArtifactGraphStore()
    store.add_version("x", ArtifactKind.SECTION, content="x-v1")
    x_ref = store.latest("x").ref
    store.add_version("b", ArtifactKind.CLAIM, content="b-orig")
    b_ref = store.latest("b").ref
    before_checkpoint = store.save_checkpoint(label="pre-repair")

    branch_store = ArtifactGraphStore.from_checkpoint(before_checkpoint, branch_id="branch")
    branch_store.add_version("b", ArtifactKind.CLAIM, content="b-fixed", read_input_versions=[])
    fixed_b_ref = branch_store.latest("b").ref
    branch_store.record_evidence(
        Evidence(
            constraint_id="c:b",
            outcome=CheckOutcome.PASS,
            checked_artifact=fixed_b_ref,
            detail="computed against x",
            input_versions=[x_ref],
        )
    )
    round_1 = RepairRound(
        round_index=0, policy_label="initial", executions=[_minimal_execution({"b": RepairOutcome.RESOLVED})]
    )

    # x moves on afterward -- this invalidates b's PASS (it depended on
    # x@1), with nothing ever re-checking b afterward. b's own position
    # in the dependency graph never moved, so it never becomes is_stale.
    branch_store.add_version("x", ArtifactKind.SECTION, content="x-v2", read_input_versions=[])
    assert branch_store.is_stale("b") is False  # sanity: not stale
    assert branch_store.live_evidence("b") == []  # sanity: the PASS is gone, invalidated

    final_checkpoint = branch_store.save_checkpoint(label="post-repair")
    branch = BranchExecutionResult(
        branch_id="A",
        source_checkpoint_id=before_checkpoint.checkpoint_id,
        policy_label="A",
        rounds=[round_1],
        stopped_reason="unresolved_remaining",
        final_checkpoint=final_checkpoint,
    )
    # Sanity-check the premise: the raw merge still says RESOLVED.
    assert _merge_final_outcomes(branch)["b"] == RepairOutcome.RESOLVED

    score = compute_repair_gain_score(
        before_checkpoint,
        branch,
        target_artifact_ids={"b"},
        triggering_evidence=[_fail_evidence(b_ref)],
        constraint_provider=_one_constraint_per_artifact,
    )

    assert score.gain == 0.0
    assert score.outcome_counts == {LEFT_STALE_OR_UNRESOLVED: 1}

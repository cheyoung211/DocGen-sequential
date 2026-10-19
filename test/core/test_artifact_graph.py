"""Coverage for src/core/artifact_graph.py and src/core/run_log.py (Stage 2
of the "Revision 2026.10.15" plan): versioned artifact graph, typed edges,
change-impact computation, staleness/evidence invalidation, the
action/evidence/cost log, and checkpoint save/restore."""

from __future__ import annotations

import pytest

from src.core.artifact_graph import ArtifactGraphStore, UnknownArtifactError
from src.core.run_log import RunLog
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStatus,
    CheckOutcome,
    Checkpoint,
    CostRecord,
    DependencyRelationKind,
    Evidence,
    RepairAction,
    RepairActionType,
)


def _ref(artifact_id: str, version: int, kind: ArtifactKind) -> ArtifactRef:
    return ArtifactRef(artifact_id=artifact_id, version=version, kind=kind)


# ---------------------------------------------------------------------------
# Versioning, typed edges, provenance
# ---------------------------------------------------------------------------


def test_add_version_assigns_sequential_versions_and_explicit_edges() -> None:
    store = ArtifactGraphStore()
    store.add_version(
        "def_1", ArtifactKind.DEFINITION, content="Let X be ...", produced_by="seed"
    )
    def_ref = store.latest("def_1").ref
    assert def_ref.version == 1

    result = store.add_version(
        "thm_1",
        ArtifactKind.CLAIM,
        content="Then Y holds.",
        read_input_versions=[def_ref],
        produced_by="TextAgentProducer",
    )
    assert result.ref.version == 1
    assert result.used_outdated_inputs == []
    assert result.newly_stale_artifact_ids == []

    [edge] = store.edges
    assert edge.relation == DependencyRelationKind.EXPLICIT
    assert edge.from_artifact.artifact_id == "def_1"
    assert edge.to_artifact.artifact_id == "thm_1"

    second = store.add_version("thm_1", ArtifactKind.CLAIM, content="Then Y holds, v2.")
    assert second.ref.version == 2
    assert store.latest("thm_1").content == "Then Y holds, v2."


def test_latest_and_edges_return_copies_mutation_does_not_corrupt_the_store() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="original")
    def_ref = store.latest("def_1").ref
    store.add_version("thm_1", ArtifactKind.CLAIM, content="...", read_input_versions=[def_ref])

    stored = store.latest("def_1")
    stored.content = "tampered"
    stored.ref.version = 999
    assert store.latest("def_1").content == "original"
    assert store.latest("def_1").ref.version == 1

    edges = store.edges
    edges.clear()
    edges_again = store.edges
    edges_again[0].relation = DependencyRelationKind.INFERRED
    assert store.edges[0].relation == DependencyRelationKind.EXPLICIT


def test_add_version_rejects_a_kind_change_for_an_existing_artifact() -> None:
    store = ArtifactGraphStore()
    store.add_version("x", ArtifactKind.DEFINITION, content="...")
    with pytest.raises(ValueError):
        store.add_version("x", ArtifactKind.TABLE, content="...")


def test_add_version_raises_on_an_input_that_was_never_stored() -> None:
    store = ArtifactGraphStore()
    phantom = _ref("ghost", 1, ArtifactKind.DEFINITION)
    with pytest.raises(UnknownArtifactError):
        store.add_version("thm_1", ArtifactKind.CLAIM, read_input_versions=[phantom])


def test_add_version_surfaces_but_does_not_reject_an_outdated_input() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v1")
    old_ref = store.latest("def_1").ref
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v2")  # def_1 is now v2

    result = store.add_version(
        "thm_1", ArtifactKind.CLAIM, content="...", read_input_versions=[old_ref]
    )
    assert result.used_outdated_inputs == [old_ref]


# ---------------------------------------------------------------------------
# Inferred edges
# ---------------------------------------------------------------------------


def test_add_inferred_edge_requires_both_artifacts_to_already_exist() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="...")
    def_ref = store.latest("def_1").ref
    phantom = _ref("ghost", 1, ArtifactKind.CLAIM)

    with pytest.raises(UnknownArtifactError):
        store.add_inferred_edge(def_ref, phantom, reason="cites label")

    with pytest.raises(UnknownArtifactError):
        store.add_inferred_edge(phantom, def_ref, reason="cites label")


def test_add_inferred_edge_does_not_retroactively_mark_the_target_stale() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="...")
    store.add_version("ex_1", ArtifactKind.CLAIM, content="...")
    def_ref = store.latest("def_1").ref
    ex_ref = store.latest("ex_1").ref

    store.add_inferred_edge(def_ref, ex_ref, reason="ex_1 cites def_1's label")
    assert not store.is_stale("ex_1")
    assert "ex_1" in store.impact_candidates("def_1")


# ---------------------------------------------------------------------------
# Change impact / cycle safety
# ---------------------------------------------------------------------------


def test_impact_candidates_is_a_pure_query() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v1")
    def_ref = store.latest("def_1").ref
    store.add_version("thm_1", ArtifactKind.CLAIM, read_input_versions=[def_ref])

    before_stale = set(store.stale_artifacts)
    before_edges = len(store.edges)
    candidates = None
    for _ in range(3):
        candidates = store.impact_candidates("def_1")
    assert candidates == {"def_1", "thm_1"}
    assert set(store.stale_artifacts) == before_stale
    assert len(store.edges) == before_edges


def test_impact_candidates_terminates_on_a_cycle() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.DEFINITION, content="b1", read_input_versions=[a_ref])
    b_ref = store.latest("b").ref
    # "a" now also reads from "b" -- an a <-> b cycle (permitted: revision
    # direction 1 explicitly does not force this graph to be acyclic).
    store.add_version("a", ArtifactKind.DEFINITION, content="a2", read_input_versions=[b_ref])

    assert store.impact_candidates("a") == {"a", "b"}
    assert store.impact_candidates("b") == {"a", "b"}


# ---------------------------------------------------------------------------
# Staleness propagation + evidence invalidation
# ---------------------------------------------------------------------------


def test_changing_an_artifact_marks_its_descendants_stale_not_itself() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v1")
    def_ref = store.latest("def_1").ref
    store.add_version("thm_1", ArtifactKind.CLAIM, read_input_versions=[def_ref])

    result = store.add_version("def_1", ArtifactKind.DEFINITION, content="v2")
    assert result.newly_stale_artifact_ids == ["thm_1"]
    assert store.is_stale("thm_1")
    assert not store.is_stale("def_1")  # def_1 is the freshly-produced artifact

    [record] = [r for r in store.invalidation_history if r.affected_artifact_id == "thm_1"]
    assert record.changed_artifact.artifact_id == "def_1"
    assert record.changed_artifact.version == 2


def test_live_evidence_excludes_evidence_whose_artifact_became_stale() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v1")
    def_ref = store.latest("def_1").ref
    thm_result = store.add_version(
        "thm_1", ArtifactKind.CLAIM, content="uses def_1", read_input_versions=[def_ref]
    )

    passing_evidence = Evidence(
        constraint_id="c1", outcome=CheckOutcome.PASS, checked_artifact=thm_result.ref
    )
    store.record_evidence(passing_evidence)
    assert store.live_evidence("thm_1") == [passing_evidence]

    # def_1 changes -- thm_1's own version is untouched, but it is now stale.
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v2")
    assert store.is_stale("thm_1")
    assert store.live_evidence("thm_1") == []  # same version, no longer trustworthy

    # Regenerating thm_1 against the new def_1 clears staleness. The old
    # Evidence still doesn't count (wrong version), and nothing new has
    # been recorded yet against the new version.
    new_def_ref = store.latest("def_1").ref
    new_thm = store.add_version(
        "thm_1", ArtifactKind.CLAIM, content="uses def_1 v2", read_input_versions=[new_def_ref]
    )
    assert not store.is_stale("thm_1")
    assert store.live_evidence("thm_1") == []

    fresh_evidence = Evidence(
        constraint_id="c1", outcome=CheckOutcome.PASS, checked_artifact=new_thm.ref
    )
    store.record_evidence(fresh_evidence)
    assert store.live_evidence("thm_1") == [fresh_evidence]


# ---------------------------------------------------------------------------
# Stage 3 review: live_evidence must also invalidate on a change to one of
# an Evidence's *other* declared inputs (Evidence.input_versions), not only
# on checked_artifact itself changing/going stale -- and must return only
# the single current verdict per constraint_id, not every historical entry
# ever logged against the same (artifact_id, version).
# ---------------------------------------------------------------------------


def test_live_evidence_excludes_evidence_whose_declared_input_moved_to_a_newer_version() -> None:
    """Regression for a user-caught repro: a RECOMPUTE check's source
    artifact moves v1->v2, but the checked artifact's own version never
    changes (and is never marked stale, since recompute inputs aren't
    wired into the dependency-edge/staleness machinery at all) -- without
    binding the source's version onto the Evidence itself, the old PASS
    kept reading as live forever."""
    store = ArtifactGraphStore()
    store.add_version("checked", ArtifactKind.BLOCK, content="x")
    store.add_version("source_1", ArtifactKind.SOURCE_DATA, content="[1, 2, 3]")
    checked_ref = store.latest("checked").ref
    source_v1 = store.latest("source_1").ref

    evidence = Evidence(
        constraint_id="sum_check",
        outcome=CheckOutcome.PASS,
        checked_artifact=checked_ref,
        input_versions=[source_v1],
    )
    store.record_evidence(evidence)
    assert store.live_evidence("checked") == [evidence]

    # checked_ref's own version is untouched; only the declared input moved.
    store.add_version("source_1", ArtifactKind.SOURCE_DATA, content="[1, 2, 3, 4]")
    assert not store.is_stale("checked")  # the checked artifact itself is unaffected
    assert store.live_evidence("checked") == []  # but its evidence depended on source_1@v1
    assert store.evidence_history("checked") == [evidence]  # full history is untouched


def test_live_evidence_returns_only_the_latest_entry_per_constraint_id() -> None:
    """Regression for a user-caught repro: a required table is missing
    (FAIL), the table is then added, and the same constraint is re-checked
    (PASS) against the *same* checked-artifact version -- live_evidence
    must return only the PASS, not both. evidence_history keeps both."""
    store = ArtifactGraphStore()
    store.add_version("checked", ArtifactKind.BLOCK, content="x")
    checked_ref = store.latest("checked").ref

    failing = Evidence(
        constraint_id="req_table", outcome=CheckOutcome.FAIL, checked_artifact=checked_ref, detail="missing"
    )
    store.record_evidence(failing)
    assert store.live_evidence("checked") == [failing]

    passing = Evidence(
        constraint_id="req_table", outcome=CheckOutcome.PASS, checked_artifact=checked_ref, detail="now present"
    )
    store.record_evidence(passing)

    live = store.live_evidence("checked")
    assert live == [passing]  # only the most recent verdict for this constraint_id
    assert store.evidence_history("checked") == [failing, passing]  # both kept, in order


def test_live_evidence_keeps_distinct_constraint_ids_independent() -> None:
    """Deduping by constraint_id must not collapse unrelated constraints
    checked against the same artifact into one."""
    store = ArtifactGraphStore()
    store.add_version("checked", ArtifactKind.BLOCK, content="x")
    checked_ref = store.latest("checked").ref
    ev_a = Evidence(constraint_id="a", outcome=CheckOutcome.PASS, checked_artifact=checked_ref)
    ev_b = Evidence(constraint_id="b", outcome=CheckOutcome.FAIL, checked_artifact=checked_ref)
    store.record_evidence(ev_a)
    store.record_evidence(ev_b)
    live = store.live_evidence("checked")
    assert {e.constraint_id for e in live} == {"a", "b"}
    assert len(live) == 2


# ---------------------------------------------------------------------------
# Second Stage 3 review: live_evidence must treat "confirmed absent at
# check time" (Evidence.missing_inputs) as its own liveness condition --
# distinct from, and the mirror image of, an input_versions ref needing to
# still exist -- and must pick the latest entry per constraint_id BEFORE
# filtering for validity, never after.
# ---------------------------------------------------------------------------


def test_live_evidence_returns_a_fail_that_correctly_found_its_declared_input_missing() -> None:
    """Regression for a user-caught repro: a required table does not exist
    yet, so a check correctly FAILs. That FAIL must itself count as live
    evidence -- "this is missing" is still true, and is exactly what the
    verdict asserts -- not be excluded just because the thing it names
    does not (yet) exist."""
    store = ArtifactGraphStore()
    store.add_version("checked", ArtifactKind.BLOCK, content="x")
    checked_ref = store.latest("checked").ref

    failing = Evidence(
        constraint_id="req_table",
        outcome=CheckOutcome.FAIL,
        checked_artifact=checked_ref,
        detail="required table is missing",
        missing_inputs=["table_foo"],
    )
    store.record_evidence(failing)
    assert store.live_evidence("checked") == [failing]


def test_live_evidence_invalidates_a_missing_input_fail_once_that_input_starts_existing() -> None:
    """Regression for a user-caught repro: after the FAIL above, the
    missing table is added but the constraint is not yet rechecked.
    live_evidence must NOT keep returning the now-stale FAIL -- it must
    return nothing, signalling "needs a fresh check", not fall back to a
    verdict that no longer reflects reality."""
    store = ArtifactGraphStore()
    store.add_version("checked", ArtifactKind.BLOCK, content="x")
    checked_ref = store.latest("checked").ref
    failing = Evidence(
        constraint_id="req_table",
        outcome=CheckOutcome.FAIL,
        checked_artifact=checked_ref,
        detail="required table is missing",
        missing_inputs=["table_foo"],
    )
    store.record_evidence(failing)
    assert store.live_evidence("checked") == [failing]

    store.add_version("table_foo", ArtifactKind.TABLE, content="now present")
    assert store.live_evidence("checked") == []  # not the stale FAIL
    assert store.evidence_history("checked") == [failing]  # history is untouched


def test_live_evidence_does_not_fall_back_to_an_earlier_verdict_when_the_latest_one_for_the_same_constraint_is_invalidated() -> None:
    """Regression for a user-caught repro: constraint_id "r1" first PASSes
    (depending on source_1@v1), then is re-checked and FAILs (depending on
    source_1@v2, current at that moment). source_1 then moves to v3,
    invalidating the FAIL. live_evidence must return nothing for "r1" --
    never fall back to the earlier PASS, which the old filter-then-pick-
    latest order would have resurrected."""
    store = ArtifactGraphStore()
    store.add_version("checked", ArtifactKind.BLOCK, content="x")
    store.add_version("source_1", ArtifactKind.SOURCE_DATA, content="v1")
    checked_ref = store.latest("checked").ref
    source_v1 = store.latest("source_1").ref

    passing = Evidence(
        constraint_id="r1", outcome=CheckOutcome.PASS, checked_artifact=checked_ref, input_versions=[source_v1]
    )
    store.record_evidence(passing)
    assert store.live_evidence("checked") == [passing]

    store.add_version("source_1", ArtifactKind.SOURCE_DATA, content="v2")
    source_v2 = store.latest("source_1").ref
    failing = Evidence(
        constraint_id="r1", outcome=CheckOutcome.FAIL, checked_artifact=checked_ref, input_versions=[source_v2]
    )
    store.record_evidence(failing)
    assert store.live_evidence("checked") == [failing]  # the new latest verdict, still valid

    store.add_version("source_1", ArtifactKind.SOURCE_DATA, content="v3")  # invalidates failing's v2 binding
    assert store.live_evidence("checked") == []  # needs recheck -- not [passing]
    assert store.evidence_history("checked") == [passing, failing]  # both kept, in order


# ---------------------------------------------------------------------------
# Stage 2 checklist item 4: a shared-definition change must selectively
# invalidate its real dependents while leaving an unrelated artifact alone
# -- mirrors revision direction 1's own worked example verbatim: "정의의
# 가정이 잘못되어 정리와 예제까지 영향을 받았다면, 정의를 수정하고 관련
# 정리·예제를 재검사한다. 무관한 비교 표는 그대로 유지한다."
# ---------------------------------------------------------------------------


def test_shared_definition_change_invalidates_dependents_but_preserves_unrelated_artifact() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="Let X be a compact set.")
    def_ref = store.latest("def_1").ref

    # Theorem explicitly declares reading the definition.
    store.add_version(
        "thm_1",
        ArtifactKind.CLAIM,
        content="Then X is bounded.",
        read_input_versions=[def_ref],
    )
    # Example doesn't declare def_1 as a production input, but its content
    # cites def_1's label -- an INFERRED relation, added separately (not an
    # automatic NLP scan; Stage 2 only provides the distinction mechanism).
    store.add_version("ex_1", ArtifactKind.CLAIM, content="E.g. X = [0, 1] (see def:def_1).")
    ex_ref = store.latest("ex_1").ref
    store.add_inferred_edge(def_ref, ex_ref, reason="ex_1 cites def:def_1")

    # An unrelated comparison table has no relation to def_1 at all.
    store.add_version("tbl_1", ArtifactKind.TABLE, content="method | accuracy")
    original_table = store.latest("tbl_1")

    assert store.impact_candidates("def_1") == {"def_1", "thm_1", "ex_1"}

    result = store.add_version(
        "def_1", ArtifactKind.DEFINITION, content="Let X be a compact, convex set."
    )

    assert set(result.newly_stale_artifact_ids) == {"thm_1", "ex_1"}
    assert store.is_stale("thm_1")
    assert store.is_stale("ex_1")

    # The unrelated table is untouched: not stale, content/version
    # unchanged, and never mentioned in the invalidation audit trail.
    assert not store.is_stale("tbl_1")
    assert store.latest("tbl_1") == original_table
    assert all(r.affected_artifact_id != "tbl_1" for r in store.invalidation_history)


# ---------------------------------------------------------------------------
# Checkpoint save / restore / branch independence
# ---------------------------------------------------------------------------


def test_restore_checkpoint_reflects_the_saved_state_not_later_mutations() -> None:
    store = ArtifactGraphStore(branch_id="main")
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v1")
    checkpoint = store.save_checkpoint(label="before_repair")

    # Mutate the original branch after the checkpoint was taken.
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v2")
    store.record_cost(CostRecord(stage="text_agent", total_tokens=100))

    restored = store.restore_checkpoint(checkpoint.checkpoint_id, branch_id="branch_A")
    assert restored.branch_id == "branch_A"
    assert restored.latest("def_1").ref.version == 1
    assert restored.latest("def_1").content == "v1"
    assert restored.log.costs == []


def test_restoring_a_checkpoint_does_not_let_either_branch_affect_the_other() -> None:
    store = ArtifactGraphStore(branch_id="main")
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v1")
    checkpoint = store.save_checkpoint()

    branch_a = store.restore_checkpoint(checkpoint.checkpoint_id, branch_id="branch_A")
    branch_b = store.restore_checkpoint(checkpoint.checkpoint_id, branch_id="branch_B")

    branch_a.add_version("def_1", ArtifactKind.DEFINITION, content="branch A's regeneration")
    branch_b.add_version("def_1", ArtifactKind.DEFINITION, content="branch B's regeneration")

    assert branch_a.latest("def_1").content == "branch A's regeneration"
    assert branch_b.latest("def_1").content == "branch B's regeneration"
    assert store.latest("def_1").content == "v1"  # the source branch is untouched
    assert branch_a.latest("def_1").branch_id == "branch_A"
    assert branch_b.latest("def_1").branch_id == "branch_B"


def test_restored_artifacts_keep_their_original_branch_provenance() -> None:
    store = ArtifactGraphStore(branch_id="main")
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v1")
    checkpoint = store.save_checkpoint()
    restored = store.restore_checkpoint(checkpoint.checkpoint_id, branch_id="branch_A")

    # The checkpointed artifact still says it was produced on "main" --
    # restoring does not rewrite history, only where *new* writes land.
    assert restored.latest("def_1").branch_id == "main"


def test_from_checkpoint_rebuilds_from_serialized_json_alone() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v1")
    def_ref = store.latest("def_1").ref
    store.add_version("thm_1", ArtifactKind.CLAIM, read_input_versions=[def_ref])
    checkpoint = store.save_checkpoint()

    serialized = checkpoint.model_dump_json()
    rebuilt_checkpoint = Checkpoint.model_validate_json(serialized)

    rebuilt_store = ArtifactGraphStore.from_checkpoint(rebuilt_checkpoint, branch_id="replay")
    assert rebuilt_store.impact_candidates("def_1") == {"def_1", "thm_1"}
    assert rebuilt_store.latest("thm_1").ref.read_input_versions[0].artifact_id == "def_1"


def test_restore_checkpoint_raises_for_an_unknown_checkpoint_id() -> None:
    store = ArtifactGraphStore()
    with pytest.raises(KeyError):
        store.restore_checkpoint("does-not-exist", branch_id="branch_A")


# ---------------------------------------------------------------------------
# Action / cost logging
# ---------------------------------------------------------------------------


def test_record_action_logs_under_the_stores_own_branch_id() -> None:
    store = ArtifactGraphStore(branch_id="branch_A")
    action = RepairAction(
        action_type=RepairActionType.PATCH_BLOCK,
        target_artifacts=[_ref("blk_1", 1, ArtifactKind.BLOCK)],
        use_diagnostic_feedback=True,
        feedback_text="compile error at blk_1",
    )
    evidence = Evidence(
        constraint_id="c1",
        outcome=CheckOutcome.FAIL,
        checked_artifact=_ref("blk_1", 1, ArtifactKind.BLOCK),
    )
    entry = store.record_action(action, triggering_evidence=[evidence])
    assert entry.branch_id == "branch_A"
    assert entry.triggering_evidence == [evidence]
    assert store.log.actions == [entry]


def test_run_log_total_cost_tokens_includes_failed_attempts() -> None:
    log = RunLog()
    log.log_cost(CostRecord(stage="text_agent", total_tokens=50, succeeded=True))
    log.log_cost(CostRecord(stage="text_agent", total_tokens=30, succeeded=False))
    assert log.total_cost_tokens == 80


# ---------------------------------------------------------------------------
# Tool-call logging (Stage 3 review): execute()'s call+observation pair was
# previously not persisted anywhere -- ArtifactGraphStore.record_tool_call /
# RunLog.log_tool_call / Checkpoint.tool_calls close that gap.
# ---------------------------------------------------------------------------


def test_record_tool_call_logs_and_round_trips_through_checkpoint() -> None:
    from src.core.schemas import (
        FinishCall,
        FinishToolResult,
        ToolActionKind,
        ToolObservation,
    )

    store = ArtifactGraphStore()
    call = FinishCall(reason="budget exhausted", unresolved_artifact_ids=["a", "b"])
    observation = ToolObservation(
        action=ToolActionKind.FINISH, ok=True, output=FinishToolResult(acknowledged=True)
    )
    store.record_tool_call(call, observation)

    assert len(store.log.tool_calls) == 1
    logged = store.log.tool_calls[0]
    assert logged.call.reason == "budget exhausted"
    assert logged.call.unresolved_artifact_ids == ["a", "b"]

    checkpoint = store.save_checkpoint()
    assert len(checkpoint.tool_calls) == 1
    assert checkpoint.tool_calls[0].call.reason == "budget exhausted"

    # Round-trips through serialized JSON alone, like every other
    # Checkpoint field (test_from_checkpoint_rebuilds_from_serialized_json_alone).
    rebuilt = Checkpoint.model_validate_json(checkpoint.model_dump_json())
    restored_store = ArtifactGraphStore.from_checkpoint(rebuilt, branch_id="branch_B")
    assert len(restored_store.log.tool_calls) == 1
    assert restored_store.log.tool_calls[0].call.reason == "budget exhausted"


def test_mutating_a_logged_tool_call_entry_does_not_corrupt_the_log() -> None:
    from src.core.schemas import FinishCall, FinishToolResult, ToolActionKind, ToolObservation

    store = ArtifactGraphStore()
    call = FinishCall(reason="original")
    observation = ToolObservation(action=ToolActionKind.FINISH, ok=True, output=FinishToolResult())
    store.record_tool_call(call, observation)

    entry = store.log.tool_calls[0]
    entry.call.reason = "corrupted"
    assert store.log.tool_calls[0].call.reason == "original"


# ---------------------------------------------------------------------------
# Regression tests for a user-caught review (4 issues + a minor one). Named
# after the exact repro each one came from.
# ---------------------------------------------------------------------------

# -- Issue 1: every store boundary must deep-copy, not just latest()/edges --


def test_mutating_add_version_result_ref_does_not_corrupt_the_store() -> None:
    store = ArtifactGraphStore()
    result = store.add_version("def_1", ArtifactKind.DEFINITION, content="v1")
    result.ref.version = 99
    assert store.latest("def_1").ref.version == 1


def test_mutating_a_returned_checkpoint_does_not_affect_a_later_restore() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v1")
    checkpoint = store.save_checkpoint()
    checkpoint.artifacts["def_1"][0].content = "tampered"
    checkpoint.label = "tampered label"

    restored = store.restore_checkpoint(checkpoint.checkpoint_id, branch_id="branch_A")
    assert restored.latest("def_1").content == "v1"


def test_mutating_a_recorded_cost_record_after_the_call_does_not_affect_the_log() -> None:
    store = ArtifactGraphStore()
    cost = CostRecord(stage="text_agent", total_tokens=100)
    store.record_cost(cost)
    cost.total_tokens = 0
    assert store.log.total_cost_tokens == 100


def test_mutating_live_evidence_result_does_not_corrupt_the_stored_evidence() -> None:
    store = ArtifactGraphStore()
    result = store.add_version("thm_1", ArtifactKind.CLAIM, content="...")
    evidence = Evidence(constraint_id="c1", outcome=CheckOutcome.FAIL, checked_artifact=result.ref)
    store.record_evidence(evidence)

    live = store.live_evidence("thm_1")
    live[0].outcome = CheckOutcome.PASS
    assert store.live_evidence("thm_1")[0].outcome == CheckOutcome.FAIL


def test_mutating_an_invalidation_record_does_not_corrupt_the_audit_trail() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v1")
    def_ref = store.latest("def_1").ref
    store.add_version("thm_1", ArtifactKind.CLAIM, read_input_versions=[def_ref])
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v2")

    records = store.invalidation_history
    records[0].affected_artifact_id = "tampered"
    assert all(r.affected_artifact_id != "tampered" for r in store.invalidation_history)


def test_mutating_a_logged_action_entry_does_not_corrupt_the_log() -> None:
    store = ArtifactGraphStore()
    action = RepairAction(
        action_type=RepairActionType.PATCH_BLOCK,
        target_artifacts=[_ref("blk_1", 1, ArtifactKind.BLOCK)],
        use_diagnostic_feedback=False,
    )
    entry = store.record_action(action)
    entry.branch_id = "tampered"
    assert store.log.actions[0].branch_id != "tampered"


# -- Issue 2: input *staleness* (not just version-matching) must be
# checked, and status must have one source of truth. --


def test_creating_an_artifact_from_a_stale_input_is_itself_born_stale() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.DEFINITION, content="b1", read_input_versions=[a_ref])
    store.add_version("a", ArtifactKind.DEFINITION, content="a2")  # b becomes stale
    assert store.is_stale("b")

    b_ref = store.latest("b").ref  # still v1, still stale, version unchanged
    result = store.add_version(
        "c", ArtifactKind.DEFINITION, content="c1", read_input_versions=[b_ref]
    )

    assert result.born_stale is True
    assert result.used_stale_inputs == [b_ref]
    assert store.is_stale("c")

    store.record_evidence(
        Evidence(constraint_id="x", outcome=CheckOutcome.PASS, checked_artifact=result.ref)
    )
    assert store.live_evidence("c") == []  # built on stale ground -- never trustworthy


def test_regenerating_with_an_outdated_input_does_not_clear_staleness() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    old_a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.DEFINITION, content="b1", read_input_versions=[old_a_ref])
    store.add_version("a", ArtifactKind.DEFINITION, content="a2")
    assert store.is_stale("b")

    # "Regenerate" b, but still reading the old a@1 -- nothing was actually fixed.
    result = store.add_version(
        "b", ArtifactKind.DEFINITION, content="b1, again", read_input_versions=[old_a_ref]
    )
    assert result.born_stale is True
    assert result.used_outdated_inputs == [old_a_ref]
    assert store.is_stale("b")  # staleness must NOT have been cleared


def test_is_stale_and_latest_status_never_disagree() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.DEFINITION, content="b1", read_input_versions=[a_ref])
    store.add_version("a", ArtifactKind.DEFINITION, content="a2")

    assert store.is_stale("b") is True
    assert store.latest("b").status == ArtifactStatus.STALE
    assert store.get_version("b", 1).status == ArtifactStatus.STALE  # v1 IS the latest here

    new_a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.DEFINITION, content="b2", read_input_versions=[new_a_ref])

    assert store.is_stale("b") is False
    assert store.latest("b").status == ArtifactStatus.FRESH
    # v1 was genuinely stale right up until it was superseded by v2 -- that
    # historical fact is frozen permanently, not reset to FRESH just
    # because v1 is no longer current (see test_historical_version_status_*
    # below for why this must also match what a checkpoint records).
    assert store.get_version("b", 1).status == ArtifactStatus.STALE


# -- Issue 3: a declared input ref must point at something real. --


def test_add_version_rejects_a_nonexistent_input_version() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    phantom_version = _ref("a", 999, ArtifactKind.DEFINITION)
    with pytest.raises(UnknownArtifactError):
        store.add_version("b", ArtifactKind.DEFINITION, read_input_versions=[phantom_version])


def test_add_version_rejects_a_mismatched_kind_in_a_declared_input() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    wrong_kind_ref = _ref("a", 1, ArtifactKind.TABLE)
    with pytest.raises(ValueError):
        store.add_version("b", ArtifactKind.DEFINITION, read_input_versions=[wrong_kind_ref])


def test_add_inferred_edge_rejects_a_nonexistent_version_or_mismatched_kind() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    store.add_version("b", ArtifactKind.DEFINITION, content="b1")
    b_ref = store.latest("b").ref

    with pytest.raises(UnknownArtifactError):
        store.add_inferred_edge(_ref("a", 999, ArtifactKind.DEFINITION), b_ref, reason="x")

    with pytest.raises(ValueError):
        store.add_inferred_edge(_ref("a", 1, ArtifactKind.TABLE), b_ref, reason="x")


# -- Issue 4: impact computation must use only CURRENT-version relations. --


def test_regenerating_without_an_input_drops_it_from_current_impact() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.DEFINITION, content="b1", read_input_versions=[a_ref])
    assert "b" in store.impact_candidates("a")

    # b is regenerated with content unrelated to a -- no declared inputs.
    store.add_version("b", ArtifactKind.DEFINITION, content="b2, unrelated to a")
    assert store.impact_candidates("a") == {"a"}  # b is no longer a current dependent

    store.add_version("a", ArtifactKind.DEFINITION, content="a3")
    assert not store.is_stale("b")  # a's change no longer propagates to the current b@2

    # The old a@1 -> b@1 edge is still in the audit trail, just not live.
    assert any(
        e.from_artifact.artifact_id == "a" and e.to_artifact.version == 1 for e in store.edges
    )


# -- Minor: add_inferred_edge's reason must actually be stored. --


def test_add_inferred_edge_persists_its_reason() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="...")
    store.add_version("ex_1", ArtifactKind.CLAIM, content="...")
    def_ref = store.latest("def_1").ref
    ex_ref = store.latest("ex_1").ref

    store.add_inferred_edge(def_ref, ex_ref, reason="ex_1 cites def:def_1")
    [edge] = [e for e in store.edges if e.relation == DependencyRelationKind.INFERRED]
    assert edge.reason == "ex_1 cites def:def_1"


# ---------------------------------------------------------------------------
# Round 3: a second user review, after round 2's fixes, caught that
# add_inferred_edge never retroactively re-evaluates status, and that a
# superseded version's status disagreed between get_version() and a
# checkpoint. Named after the exact repro each one came from.
# ---------------------------------------------------------------------------

# -- A missed relation, filled in later via add_inferred_edge, must
# retroactively invalidate a *current* target if its revealed source is
# outdated or stale -- not leave the target looking fresh forever. --


def test_add_inferred_edge_retroactively_invalidates_target_when_source_is_outdated() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    a_ref_v1 = store.latest("a").ref

    # b actually used a@1, but that relation was never declared at the time.
    store.add_version("b", ArtifactKind.DEFINITION, content="b1")
    b_ref = store.latest("b").ref
    store.add_version("c", ArtifactKind.DEFINITION, content="c1", read_input_versions=[b_ref])

    passing_evidence = Evidence(constraint_id="x", outcome=CheckOutcome.PASS, checked_artifact=b_ref)
    store.record_evidence(passing_evidence)
    assert store.live_evidence("b") == [passing_evidence]

    # a moves on; b looks completely unaffected, since the relation is unknown.
    store.add_version("a", ArtifactKind.DEFINITION, content="a2")
    assert not store.is_stale("b")

    # The missing relation is discovered after the fact.
    store.add_inferred_edge(a_ref_v1, b_ref, reason="b1 actually used a@1, undeclared at the time")

    assert store.is_stale("b")
    assert store.is_stale("c")  # descendant of b, also affected
    assert store.live_evidence("b") == []  # the old PASS no longer counts as live


def test_add_inferred_edge_retroactively_invalidates_target_when_source_is_already_stale() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    a_ref = store.latest("a").ref
    store.add_version("x", ArtifactKind.DEFINITION, content="x1", read_input_versions=[a_ref])
    store.add_version("a", ArtifactKind.DEFINITION, content="a2")  # x becomes stale
    assert store.is_stale("x")

    store.add_version("b", ArtifactKind.DEFINITION, content="b1")  # relation to x undeclared
    b_ref = store.latest("b").ref
    x_ref = store.latest("x").ref  # x's version didn't change -- still stale

    store.add_inferred_edge(x_ref, b_ref, reason="b1 actually used x, which was already stale")
    assert store.is_stale("b")


def test_add_inferred_edge_does_not_invalidate_when_source_is_fresh_and_current() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    store.add_version("b", ArtifactKind.DEFINITION, content="b1")
    a_ref = store.latest("a").ref
    b_ref = store.latest("b").ref

    store.add_inferred_edge(a_ref, b_ref, reason="a healthy relation between two current artifacts")
    assert not store.is_stale("b")


def test_add_inferred_edge_does_not_invalidate_a_historical_target_version() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    a_ref_v1 = store.latest("a").ref
    store.add_version("b", ArtifactKind.DEFINITION, content="b1")
    b_ref_v1 = store.latest("b").ref
    store.add_version("b", ArtifactKind.DEFINITION, content="b2")  # b moves on, unrelated to a
    store.add_version("a", ArtifactKind.DEFINITION, content="a2")  # a@1 is now outdated too

    # This edge is purely a historical annotation about b's *old* v1 --
    # b's current v2 must be entirely unaffected.
    store.add_inferred_edge(a_ref_v1, b_ref_v1, reason="historical annotation about b@1 only")
    assert not store.is_stale("b")


# -- A superseded version's status must mean the same thing, and read the
# same way, everywhere it's exposed. --


def test_historical_version_status_is_frozen_at_supersession_and_matches_checkpoint() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.DEFINITION, content="b1", read_input_versions=[a_ref])
    store.add_version("a", ArtifactKind.DEFINITION, content="a2")  # b becomes stale
    assert store.is_stale("b")

    new_a_ref = store.latest("a").ref
    store.add_version("b", ArtifactKind.DEFINITION, content="b2", read_input_versions=[new_a_ref])
    assert not store.is_stale("b")  # b is healed as of v2

    # get_version() and a checkpoint taken afterward must agree that b@1
    # really was stale right up until it was superseded.
    assert store.get_version("b", 1).status == ArtifactStatus.STALE
    checkpoint = store.save_checkpoint()
    assert checkpoint.artifacts["b"][0].status == ArtifactStatus.STALE
    assert checkpoint.artifacts["b"][1].status == ArtifactStatus.FRESH


def test_historical_version_status_freezes_fresh_when_never_stale() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    store.add_version("a", ArtifactKind.DEFINITION, content="a2")
    assert store.get_version("a", 1).status == ArtifactStatus.FRESH

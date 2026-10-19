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

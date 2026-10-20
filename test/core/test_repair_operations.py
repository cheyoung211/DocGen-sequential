"""Coverage for src/core/repair_operations.py (Stage 4): the cascading
regenerate-or-reaffirm executor for the selective action types, the
unconditional whole-document path, and the two fixed RepairPolicy
implementations. Uses Protocol-conforming fakes throughout -- no real LLM
call anywhere in this file, per this stage's explicit scope decision (see
repair_operations.py's module docstring).

Revised after a user review found 8 real bugs in the first pass (see
memory docgen_stage_review_rigor) -- every fix below has at least one
regression test named after the repro that caught it."""

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
from src.core.tool_executor import ToolExecutor


class _FakeProducer:
    """Always SUCCEEDS unless the artifact_id is in `fail_ids`. Records
    every call's context so a test can assert what it was actually handed
    (``read_input_versions``, ``error_message``). ``content_for`` lets a
    test control exactly what "real content" this producer reports, to
    verify the executor persists it rather than fabricating ``None``."""

    def __init__(
        self,
        fail_ids: FrozenSet[str] = frozenset(),
        content_for: Optional[Dict[str, str]] = None,
    ) -> None:
        self.fail_ids = set(fail_ids)
        self.content_for = content_for or {}
        self.calls: List[Dict[str, Any]] = []

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ProductionResult:
        self.calls.append({"artifact_id": artifact_id, "context": dict(context)})
        if artifact_id in self.fail_ids:
            return ProductionResult(status=ProductionStatus.FAILED, error="fake production failure")
        return ProductionResult(
            status=ProductionStatus.SUCCESS,
            artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.BLOCK),
            content=self.content_for.get(artifact_id, f"real-content:{artifact_id}"),
            read_input_versions=context.get("read_input_versions"),
        )


class _FakeValidator:
    """outcome depends on membership in `fail_ids`/`unknown_ids` (mutable --
    a test can edit either mid-run)."""

    def __init__(self, fail_ids: FrozenSet[str] = frozenset(), unknown_ids: FrozenSet[str] = frozenset()) -> None:
        self.fail_ids = set(fail_ids)
        self.unknown_ids = set(unknown_ids)
        self.config: Dict[str, bool] = {}
        self.calls: List[str] = []

    def check(
        self, artifact: ArtifactRef, constraints: List[Constraint], context: Optional[Dict[str, Any]] = None
    ) -> List[Evidence]:
        self.calls.append(artifact.artifact_id)
        if artifact.artifact_id in self.unknown_ids:
            from src.core.schemas import UnknownReason

            return [
                Evidence(
                    constraint_id=c.constraint_id,
                    outcome=CheckOutcome.UNKNOWN,
                    checked_artifact=artifact,
                    unknown_reason=UnknownReason.NOT_REPRESENTABLE,
                )
                for c in constraints
            ]
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
    store: ArtifactGraphStore,
    producers: Dict[ArtifactKind, Any],
    validator: _FakeValidator,
    *,
    no_constraints_for: FrozenSet[str] = frozenset(),
    cap: float = 1_000_000.0,
    max_actions: int = 1_000_000,
) -> RepairOperationExecutor:
    tool_executor = ToolExecutor(store, validator, budget=Budget(cap=cap), max_actions=max_actions)
    return RepairOperationExecutor(
        store,
        tool_executor,
        producers,
        _one_constraint_per_artifact(no_constraints_for),
        base_context={},
        model_name="fake-model",
    )


def _action(action_type: RepairActionType, targets: List[ArtifactRef], feedback: bool = True) -> RepairAction:
    return RepairAction(action_type=action_type, target_artifacts=targets, use_diagnostic_feedback=feedback)


# ---------------------------------------------------------------------------
# RepairOperationExecutor -- the selective cascade.
# ---------------------------------------------------------------------------


def test_regenerates_only_the_target_when_nothing_downstream_exists() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("blkA", ArtifactKind.BLOCK))
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset())
    executor = _executor(store, {ArtifactKind.BLOCK: producer}, validator)

    result = executor.execute(_action(RepairActionType.PATCH_BLOCK, [store.latest("blkA").ref]))

    assert len(result.regenerated) == 1
    assert result.regenerated[0].outcome == RepairOutcome.RESOLVED
    assert result.reverified_untouched == []
    assert result.left_stale == []
    assert result.overall_outcome == RepairOutcome.RESOLVED
    assert [c["artifact_id"] for c in producer.calls] == ["blkA"]


def test_cascades_and_reaffirms_a_passing_descendant_leaving_it_fresh() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("secA", ArtifactKind.SECTION), ("secB", ArtifactKind.SECTION))
    assert store.is_stale("secB") is False

    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids={"secA"})
    executor = _executor(store, {ArtifactKind.SECTION: producer}, validator)

    result = executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("secA").ref]))

    regenerated_ids = {r.pre_repair_ref.artifact_id for r in result.regenerated}
    assert regenerated_ids == {"secA"}
    reverified_ids = {r.artifact_id for r in result.reverified_untouched}
    assert reverified_ids == {"secB"}
    assert [c["artifact_id"] for c in producer.calls] == ["secA"]
    assert store.is_stale("secB") is False
    assert store.latest("secB").ref.version == 2


def test_a_descendant_that_still_fails_after_reverify_gets_regenerated_and_reported() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("secA", ArtifactKind.SECTION), ("secB", ArtifactKind.SECTION))

    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids={"secA", "secB"})
    executor = _executor(store, {ArtifactKind.SECTION: producer}, validator)

    result = executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("secA").ref]))

    regenerated_ids = {r.pre_repair_ref.artifact_id for r in result.regenerated}
    assert regenerated_ids == {"secA", "secB"}
    secB_record = next(r for r in result.regenerated if r.pre_repair_ref.artifact_id == "secB")
    assert secB_record.outcome == RepairOutcome.STILL_FAILING
    assert result.overall_outcome == RepairOutcome.STILL_FAILING
    assert "secB" in [c["artifact_id"] for c in producer.calls]


def test_production_failure_does_not_crash_and_does_not_cascade() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("secA", ArtifactKind.SECTION), ("secB", ArtifactKind.SECTION))

    producer = _FakeProducer(fail_ids={"secA"})
    validator = _FakeValidator(fail_ids={"secA"})
    executor = _executor(store, {ArtifactKind.SECTION: producer}, validator)

    result = executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("secA").ref]))

    assert len(result.regenerated) == 1
    assert result.regenerated[0].outcome == RepairOutcome.PRODUCTION_FAILED
    assert result.regenerated[0].new_ref is None
    assert result.reverified_untouched == []
    assert result.overall_outcome == RepairOutcome.PRODUCTION_FAILED
    assert store.latest("secA").ref.version == 1


def test_no_producer_registered_for_a_kind_is_reported_honestly() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("def1", ArtifactKind.DEFINITION))
    validator = _FakeValidator(fail_ids={"def1"})
    executor = _executor(store, {}, validator)

    result = executor.execute(_action(RepairActionType.REPLAN_SUBGRAPH, [store.latest("def1").ref]))

    assert len(result.regenerated) == 1
    assert result.regenerated[0].outcome == RepairOutcome.NO_PRODUCER
    assert result.regenerated[0].new_ref is None
    assert result.overall_outcome == RepairOutcome.NO_PRODUCER


def test_a_candidate_with_no_constraints_is_left_stale_not_fabricated_as_resolved() -> None:
    """Earlier behavior silently dropped this candidate from every result
    list (invisible). A review's issue 3 asked that "검사 없음·누락도
    해결 완료로 집계하면 안 된다" -- it must show up as genuinely unresolved,
    not vanish."""
    store = ArtifactGraphStore()
    _store_with_chain(
        store,
        ("secA", ArtifactKind.SECTION),
        ("secB", ArtifactKind.SECTION),
        ("secC", ArtifactKind.SECTION),
    )
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids={"secA", "secC"})
    executor = _executor(store, {ArtifactKind.SECTION: producer}, validator, no_constraints_for=frozenset({"secB"}))

    result = executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("secA").ref]))

    assert "secB" not in [c["artifact_id"] for c in producer.calls]
    assert "secB" not in validator.calls
    assert {r.artifact_id for r in result.left_stale} == {"secB"}
    assert "secC" in {r.pre_repair_ref.artifact_id for r in result.regenerated}
    assert store.is_stale("secB") is True  # genuinely still stale, never resolved


def test_execute_logs_the_action_exactly_once_regardless_of_cascade_size() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("secA", ArtifactKind.SECTION), ("secB", ArtifactKind.SECTION))
    executor = _executor(store, {ArtifactKind.SECTION: _FakeProducer()}, _FakeValidator(fail_ids={"secA", "secB"}))

    action = _action(RepairActionType.REGENERATE_SECTION, [store.latest("secA").ref])
    trigger = [Evidence(constraint_id="x", outcome=CheckOutcome.FAIL, checked_artifact=store.latest("secA").ref)]
    executor.execute(action, triggering_evidence=trigger)

    assert len(store.log.actions) == 1
    assert store.log.actions[0].action.action_type == RepairActionType.REGENERATE_SECTION


# ---------------------------------------------------------------------------
# Issue 1 (review): real content and real input versions are persisted.
# ---------------------------------------------------------------------------


def test_regeneration_persists_the_producers_real_content_not_none() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("a", ArtifactKind.SECTION))
    producer = _FakeProducer(content_for={"a": "the real regenerated body"})
    validator = _FakeValidator(fail_ids=frozenset())
    executor = _executor(store, {ArtifactKind.SECTION: producer}, validator)

    result = executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("a").ref]))

    assert result.regenerated[0].outcome == RepairOutcome.RESOLVED
    assert store.latest("a").ref.version == 2
    assert store.latest("a").content == "the real regenerated body"


def test_regeneration_records_what_the_producer_actually_reports_using_not_a_blind_repin() -> None:
    """The Producer is told its "intended" inputs via
    context["read_input_versions"], but a conforming Producer that knows
    better can report something DIFFERENT as what it really used -- the
    executor must persist THAT, not silently override it with its own
    repinned guess."""
    store = ArtifactGraphStore()
    store.add_version("x", ArtifactKind.SOURCE_DATA, content=None)
    store.add_version("y", ArtifactKind.SOURCE_DATA, content=None)
    store.add_version("a", ArtifactKind.SECTION, content=None, read_input_versions=[store.latest("x").ref])

    class _ReportsDifferentInputs:
        def produce(self, artifact_id, context):
            # Ignores the intended ["x"] and reports it actually used "y".
            return ProductionResult(
                status=ProductionStatus.SUCCESS,
                artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.SECTION),
                content="body",
                read_input_versions=[store.latest("y").ref],
            )

    validator = _FakeValidator(fail_ids=frozenset())
    executor = _executor(store, {ArtifactKind.SECTION: _ReportsDifferentInputs()}, validator)

    executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("a").ref]))

    new_a = store.latest("a")
    assert [i.artifact_id for i in new_a.ref.read_input_versions] == ["y"]


def test_producer_that_reports_nothing_falls_back_to_the_intended_repinned_inputs() -> None:
    """A Producer with no finer-grained tracking of its own (read_input_versions
    left at its default None, not an empty list) is not penalized -- the
    executor's own stated intention is used, which is still real/non-
    fabricated: it is exactly what the caller told the Producer to use."""
    store = ArtifactGraphStore()
    store.add_version("x", ArtifactKind.SOURCE_DATA, content=None)
    store.add_version("a", ArtifactKind.SECTION, content=None, read_input_versions=[store.latest("x").ref])
    producer = _FakeProducer()  # echoes context["read_input_versions"] back -- see _FakeProducer
    validator = _FakeValidator(fail_ids=frozenset())
    executor = _executor(store, {ArtifactKind.SECTION: producer}, validator)

    executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("a").ref]))

    new_a = store.latest("a")
    assert [i.artifact_id for i in new_a.ref.read_input_versions] == ["x"]


# ---------------------------------------------------------------------------
# Issue 2 (review): processing-order staleness -- a→z→b, b must be revisited
# after z updates later in the same sweep.
# ---------------------------------------------------------------------------


def test_a_candidate_resolved_before_its_own_upstream_is_revisited_once_the_upstream_changes() -> None:
    """a -> z -> b. Repairing 'a' marks both z and b stale. Sorted
    processing order visits 'b' (alphabetically first) before 'z' -- a
    single visit-once pass would incorrectly leave 'b' settled against
    z's *old* version and never notice when z itself changes afterward.
    The real repro's reported bad outcome was RESOLVED with
    stale_artifacts={'b'} still set at the end; this must not happen."""
    store = ArtifactGraphStore()
    _store_with_chain(store, ("a", ArtifactKind.SECTION), ("z", ArtifactKind.SECTION), ("b", ArtifactKind.SECTION))

    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset())  # everything passes its own content check
    executor = _executor(store, {ArtifactKind.SECTION: producer}, validator)

    result = executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("a").ref]))

    assert store.is_stale("b") is False
    assert store.is_stale("z") is False
    assert "b" not in {r.artifact_id for r in result.left_stale}
    # b must have been settled against z's FINAL version, not an intermediate one.
    final_b = store.latest("b")
    final_z = store.latest("z")
    assert final_b.ref.read_input_versions[0].version == final_z.ref.version


# ---------------------------------------------------------------------------
# Issue 3 (review): UNKNOWN-only evidence must not be read as resolved.
# ---------------------------------------------------------------------------


def test_unknown_only_reverify_after_regeneration_is_inconclusive_not_resolved() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("a", ArtifactKind.SECTION))
    producer = _FakeProducer()
    validator = _FakeValidator(unknown_ids={"a"})
    executor = _executor(store, {ArtifactKind.SECTION: producer}, validator)

    result = executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("a").ref]))

    assert result.regenerated[0].outcome == RepairOutcome.INCONCLUSIVE
    assert result.overall_outcome == RepairOutcome.INCONCLUSIVE


def test_unknown_only_cascade_candidate_stays_stale_and_is_reported_left_stale() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("a", ArtifactKind.SECTION), ("b", ArtifactKind.SECTION))
    producer = _FakeProducer()
    validator = _FakeValidator(unknown_ids={"b"})
    executor = _executor(store, {ArtifactKind.SECTION: producer}, validator)

    result = executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("a").ref]))

    assert "b" not in [c["artifact_id"] for c in producer.calls]  # never reaffirmed, never regenerated
    assert {r.artifact_id for r in result.left_stale} == {"b"}
    assert store.is_stale("b") is True


# ---------------------------------------------------------------------------
# Issue 4 (review): INFERRED edges survive reaffirm; reaffirmed evidence is
# rebound to the new version.
# ---------------------------------------------------------------------------


def test_reaffirm_carries_forward_an_inferred_edge_onto_the_new_version() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("a", ArtifactKind.SECTION))
    store.add_version("b", ArtifactKind.SECTION, content=None)  # no EXPLICIT relation to a at all
    store.add_inferred_edge(store.latest("a").ref, store.latest("b").ref, reason="b cites a's label")

    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids={"a"})
    executor = _executor(store, {ArtifactKind.SECTION: producer}, validator)

    executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("a").ref]))

    # b must have been swept in via the (carried-forward) inferred edge and
    # reaffirmed -- not left behind just because the relation was INFERRED
    # rather than declared through read_input_versions.
    assert "b" not in [c["artifact_id"] for c in producer.calls]
    assert store.is_stale("b") is False
    new_b_version = store.latest("b").ref.version
    assert new_b_version > 1

    # The edge itself still names b -- now at its current version -- so a
    # LATER change to 'a' continues to invalidate 'b' going forward.
    inferred_edges_to_b = [
        e
        for e in store.edges
        if e.relation == DependencyRelationKind.INFERRED and e.to_artifact.artifact_id == "b"
    ]
    assert any(e.to_artifact.version == new_b_version for e in inferred_edges_to_b)


def test_reaffirmed_evidence_is_rebound_to_the_new_version_so_live_evidence_is_not_empty() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("a", ArtifactKind.SECTION), ("b", ArtifactKind.SECTION))
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids={"a"})
    executor = _executor(store, {ArtifactKind.SECTION: producer}, validator)

    executor.execute(_action(RepairActionType.REGENERATE_SECTION, [store.latest("a").ref]))

    live = store.live_evidence("b")
    assert len(live) == 1
    assert live[0].checked_artifact.version == store.latest("b").ref.version


# ---------------------------------------------------------------------------
# Additional review note: use_diagnostic_feedback=False must not leak a
# stale/previous error_message into the Producer's context.
# ---------------------------------------------------------------------------


def test_blind_action_clears_a_stale_error_message_from_base_context() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("secA", ArtifactKind.SECTION))
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset())
    tool_executor = ToolExecutor(store, validator, budget=Budget(cap=1_000_000.0), max_actions=1_000_000)
    executor = RepairOperationExecutor(
        store,
        tool_executor,
        {ArtifactKind.SECTION: producer},
        _one_constraint_per_artifact(),
        base_context={"error_message": "a leftover message from some earlier round"},
        model_name="fake-model",
    )

    action = _action(RepairActionType.REGENERATE_SECTION, [store.latest("secA").ref], feedback=False)
    executor.execute(action)

    assert producer.calls[0]["context"].get("error_message") is None


def test_blind_patch_block_gets_a_neutral_placeholder_not_a_stale_message() -> None:
    store = ArtifactGraphStore()
    _store_with_chain(store, ("blkA", ArtifactKind.BLOCK))
    producer = _FakeProducer()
    validator = _FakeValidator(fail_ids=frozenset())
    tool_executor = ToolExecutor(store, validator, budget=Budget(cap=1_000_000.0), max_actions=1_000_000)
    executor = RepairOperationExecutor(
        store,
        tool_executor,
        {ArtifactKind.BLOCK: producer},
        _one_constraint_per_artifact(),
        base_context={"error_message": "stale text that must not leak through"},
        model_name="fake-model",
    )

    action = _action(RepairActionType.PATCH_BLOCK, [store.latest("blkA").ref], feedback=False)
    executor.execute(action)

    assert producer.calls[0]["context"]["error_message"] == "retry"


# ---------------------------------------------------------------------------
# Issue 6 (review): budget admission is checked between individual
# generation/check calls, not only once per round.
# ---------------------------------------------------------------------------


def test_budget_is_checked_between_individual_generation_calls_within_one_execute() -> None:
    """Two independent seeds, 100 tokens each, cap=150 -- the second call
    must never be charged on top of an already-exhausted budget."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.SECTION, content=None)
    store.add_version("b", ArtifactKind.SECTION, content=None)

    from src.core.cost_tracking import CostTrackingProducer

    class _CostlyProducer:
        def produce(self, artifact_id, context):
            sink = context.get("usage_sink")
            if sink is not None:
                sink.append({"input_tokens": 0, "output_tokens": 100, "total_tokens": 100})
            return ProductionResult(
                status=ProductionStatus.SUCCESS,
                artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.SECTION),
                content="x",
                read_input_versions=[],
            )

    producer = _CostlyProducer()
    validator = _FakeValidator(fail_ids=frozenset())
    tool_executor = ToolExecutor(store, validator, budget=Budget(cap=150.0), max_actions=1000)
    executor = RepairOperationExecutor(
        store, tool_executor, {ArtifactKind.SECTION: producer}, _one_constraint_per_artifact(), {}, model_name="qwen-fake"
    )

    action = _action(RepairActionType.REPLAN_SUBGRAPH, [store.latest("a").ref, store.latest("b").ref])
    result = executor.execute(action)

    assert tool_executor.budget.spent <= 200  # bounded to at most one call's overspend, per ToolExecutor's own contract
    assert result.budget_exhausted is True
    # The second target must not have been silently resolved -- it is
    # either genuinely regenerated-and-charged, or left_stale/blocked, never
    # invisible.
    touched_ids = {r.pre_repair_ref.artifact_id for r in result.regenerated}
    left_ids = {r.artifact_id for r in result.left_stale}
    assert {"a", "b"} <= (touched_ids | left_ids)


# ---------------------------------------------------------------------------
# Issue 8 (review): whole-document regeneration is unconditional -- every
# artifact in the document's impact set is force-regenerated, no selective
# skip even for one that would otherwise pass.
# ---------------------------------------------------------------------------


def test_regenerate_document_forces_every_section_even_ones_that_would_pass() -> None:
    store = ArtifactGraphStore()
    doc_result = store.add_version("doc", ArtifactKind.DOCUMENT, content=None)
    store.add_version("secA", ArtifactKind.SECTION, content=None, read_input_versions=[doc_result.ref])
    store.add_version("secB", ArtifactKind.SECTION, content=None, read_input_versions=[doc_result.ref])

    doc_producer = _FakeProducer()
    sec_producer = _FakeProducer()
    # Nothing is failing at all -- a selective cascade would never touch
    # either section. The whole-document baseline must regenerate them
    # anyway.
    validator = _FakeValidator(fail_ids=frozenset())
    executor = _executor(
        store, {ArtifactKind.DOCUMENT: doc_producer, ArtifactKind.SECTION: sec_producer}, validator
    )

    action = _action(RepairActionType.REGENERATE_DOCUMENT, [store.latest("doc").ref], feedback=False)
    result = executor.execute(action)

    touched_section_ids = {c["artifact_id"] for c in sec_producer.calls}
    assert touched_section_ids == {"secA", "secB"}
    assert [c["artifact_id"] for c in doc_producer.calls] == ["doc"]
    assert {r.pre_repair_ref.artifact_id for r in result.regenerated} == {"doc", "secA", "secB"}
    assert result.reverified_untouched == []  # unconditional -- nothing is ever merely "reaffirmed"


def test_regenerate_document_with_no_document_kind_target_reports_no_producer_without_crashing() -> None:
    store = ArtifactGraphStore()
    store.add_version("secA", ArtifactKind.SECTION, content=None)
    validator = _FakeValidator(fail_ids=frozenset())
    executor = _executor(store, {ArtifactKind.SECTION: _FakeProducer()}, validator)

    action = _action(RepairActionType.REGENERATE_DOCUMENT, [store.latest("secA").ref])
    result = executor.execute(action)

    assert result.overall_outcome == RepairOutcome.NO_PRODUCER
    assert {r.artifact_id for r in result.left_stale} == {"secA"}


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

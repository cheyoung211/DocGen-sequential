"""Coverage for src/core/controller.py (Stage 5): the LangGraph controller
loop (ControllerRuntime), both policy variants (RuleBasedController/
LLMController), anti-infinite-loop/budget-overrun guards, durable
resume-from-checkpoint, and compute_ready_artifacts. Uses
Protocol-conforming fakes throughout -- no real LLM call and no real
local Qwen3.6 server dependency anywhere in this file, per this stage's
confirmed scope decision (see controller.py's module docstring): a real
pilot run is an explicitly deferred follow-up.

Revised after a user review found 7 real bugs in the first pass (see
controller.py's own module docstring for the full list, and memory
docgen_stage_review_rigor) -- every fix below has at least one regression
test named after the repro that caught it."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.artifact_graph import ArtifactGraphStore
from src.core.controller import (
    ControllerRuntime,
    LLMController,
    RuleBasedController,
    _compute_unresolved,
    _describe_observation,
    _merge_evidence,
    _prune_invalidated_evidence,
    compute_ready_artifacts,
)
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    CheckCall,
    CheckOutcome,
    Constraint,
    ConstraintKind,
    ControllerState,
    ControllerStepRecord,
    ControllerStopReason,
    CompileCall,
    Evidence,
    PatchCall,
    ProductionResult,
    ProductionStatus,
    ToolActionKind,
    UnknownReason,
)
from src.core.tool_executor import ToolExecutor


def _constraint_provider(ref: ArtifactRef) -> List[Constraint]:
    return [Constraint(constraint_id="c0", kind=ConstraintKind.SCHEMA_CONTRACT, description="x", applies_to=[ref])]


class _FakeProducer:
    """Always SUCCEEDS with ``f"fixed:{artifact_id}"`` unless ``artifact_id``
    is in ``fail_ids``."""

    def __init__(self, fail_ids: frozenset = frozenset()) -> None:
        self.fail_ids = set(fail_ids)
        self.calls: List[str] = []

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ProductionResult:
        self.calls.append(artifact_id)
        if artifact_id in self.fail_ids:
            return ProductionResult(status=ProductionStatus.FAILED, error="fake failure")
        return ProductionResult(
            status=ProductionStatus.SUCCESS,
            artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.BLOCK),
            content=f"fixed:{artifact_id}",
            read_input_versions=context.get("read_input_versions"),
        )


class _AlwaysFailingProducer:
    """Always reports ``ProductionStatus.FAILED`` -- never writes a new
    version, so the target artifact's staleness is never cleared (bug 2a's
    exact repro)."""

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ProductionResult:
        return ProductionResult(status=ProductionStatus.FAILED, error="production always fails")


class _VersionThresholdValidator:
    """FAILs any artifact whose *current* version is <= ``fail_below_version``,
    PASSes otherwise -- simulates "the first version is broken, a
    regenerated version resolves it" so a test can observe the controller
    actually resolve something instead of looping until a guard fires."""

    def __init__(self, fail_below_version: int = 0) -> None:
        self.fail_below_version = fail_below_version
        self.config: Dict[str, bool] = {}
        self.calls: List[str] = []

    def check(
        self, artifact: ArtifactRef, constraints: List[Constraint], context: Optional[Dict[str, Any]] = None
    ) -> List[Evidence]:
        self.calls.append(f"{artifact.artifact_id}@{artifact.version}")
        outcome = CheckOutcome.FAIL if artifact.version <= self.fail_below_version else CheckOutcome.PASS
        detail = "fail" if outcome == CheckOutcome.FAIL else ""
        return [
            Evidence(constraint_id=c.constraint_id, outcome=outcome, checked_artifact=artifact, detail=detail)
            for c in constraints
        ] or [Evidence(constraint_id="c0", outcome=outcome, checked_artifact=artifact, detail=detail)]


class _AlwaysFailValidator:
    def __init__(self) -> None:
        self.config: Dict[str, bool] = {}

    def check(
        self, artifact: ArtifactRef, constraints: List[Constraint], context: Optional[Dict[str, Any]] = None
    ) -> List[Evidence]:
        return [
            Evidence(constraint_id=c.constraint_id, outcome=CheckOutcome.FAIL, checked_artifact=artifact, detail="always fails")
            for c in constraints
        ] or [Evidence(constraint_id="c0", outcome=CheckOutcome.FAIL, checked_artifact=artifact, detail="always fails")]


class _FakeLLMGenerator:
    """Returns a scripted sequence of raw text responses, one per call --
    satisfies ``controller.LLMGenerator`` without any network/local-server
    dependency. ``raises`` optionally scripts an exception for a given
    call index instead of a normal response (``None`` means "don't raise
    this time"); ``tokens_per_call`` optionally reports real usage via
    ``usage_sink`` so a test can confirm cost is actually charged."""

    def __init__(
        self,
        responses: List[str],
        *,
        raises: Optional[List[Optional[Exception]]] = None,
        tokens_per_call: int = 0,
    ) -> None:
        self.responses = list(responses)
        self.raises = list(raises) if raises is not None else []
        self.tokens_per_call = tokens_per_call
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        self.calls += 1
        if self.raises:
            exc = self.raises.pop(0)
            if exc is not None:
                raise exc
        usage_sink = kwargs.get("usage_sink")
        if usage_sink is not None and self.tokens_per_call:
            usage_sink.append(
                {"input_tokens": self.tokens_per_call, "output_tokens": 0, "total_tokens": self.tokens_per_call}
            )
        return self.responses.pop(0)

    def extract_json_block(self, text: str) -> str:
        return text


# -- RuleBasedController + the full controller->tool->controller loop -------


def test_rule_based_controller_resolves_a_failing_document_then_compiles_then_finishes(tmp_path) -> None:
    """The exact trace the Stage 5 completion criterion asks for: a
    different observation (FAIL -> PASS -> compiled) leads to a different
    next action (CHECK -> REPLAN -> COMPILE -> FINISH). No separate
    re-CHECK after the repair -- the repair's own fresh reverify evidence
    (produced by the same validator, inside the cascade) is trusted
    directly, avoiding a redundant duplicate check. Uses a real minimal
    main.tex (same fixture pattern as test_tool_executor.py's own
    "dispatches to the real latex compiler" test) so COMPILE genuinely
    succeeds, exercising the true happy path through to a real
    reason="resolved" finish."""
    request_id = "req1"
    project_dir = tmp_path / request_id
    project_dir.mkdir()
    (project_dir / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}Hello\\end{document}\n", encoding="utf-8"
    )

    store = ArtifactGraphStore(branch_id="episode-1")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    producer = _FakeProducer()
    validator = _VersionThresholdValidator(fail_below_version=1)
    policy = RuleBasedController(doc_ref, _constraint_provider, request_id=request_id, output_dir=str(tmp_path))
    runtime = ControllerRuntime(
        store,
        policy,
        validator,
        {ArtifactKind.DOCUMENT: producer, ArtifactKind.BLOCK: producer},
        _constraint_provider,
        document_ref=doc_ref,
        budget=Budget(cap=1000.0),
        max_steps=20,
    )

    final_state = runtime.run(thread_id="t1")

    actions = [record.call_summary for record in final_state.history]
    assert actions[0] == "check:doc@1"
    assert actions[1] == "replan:doc@1"  # the only repair-scope action a DOCUMENT-kind target can map to
    assert actions[2] == f"compile:{request_id}"
    assert final_state.history[2].ok is True
    assert final_state.compiled is True
    assert final_state.history[-1].action.value == "finish"
    assert final_state.history[-1].call_summary == "finish:resolved"
    assert store.latest("doc").ref.version == 2
    assert store.latest("doc").content == "fixed:doc"


def test_rule_based_controller_never_calls_decide_on_a_done_state() -> None:
    """``_controller_node`` must short-circuit once ``done`` is already
    True -- a policy's own ``decide()`` should never be asked for a next
    action once the episode has already ended (there would be nothing
    meaningful to decide)."""
    store = ArtifactGraphStore(branch_id="episode-done")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref

    class _TripwirePolicy:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, state, store, tool_executor) -> Any:  # noqa: ANN001 - Protocol signature
            self.calls += 1
            from src.core.schemas import FinishCall

            return FinishCall(reason="resolved", unresolved_artifact_ids=[])

    policy = _TripwirePolicy()
    runtime = ControllerRuntime(
        store,
        policy,
        _AlwaysFailValidator(),
        {ArtifactKind.DOCUMENT: _FakeProducer()},
        _constraint_provider,
        document_ref=doc_ref,
        budget=Budget(cap=1000.0),
        max_steps=20,
    )
    final_state = runtime.run(thread_id="t-done")
    assert final_state.done is True
    assert policy.calls == 1  # FINISH ends the episode on the very first decision


# -- Bug 1: evidence merging must be per (artifact, constraint), and must --
# -- stay synchronized with the store's own invalidation rules -------------


def test_merge_evidence_keys_by_artifact_and_constraint_not_artifact_alone() -> None:
    """Repro: checking two DIFFERENT constraints on the same artifact (one
    FAIL, one PASS) must keep BOTH verdicts. The first Stage 5 pass keyed
    only by artifact_id, so whichever came last in the list silently won,
    discarding the other regardless of outcome."""
    ref = ArtifactRef(artifact_id="doc", version=1, kind=ArtifactKind.DOCUMENT)
    fresh = [
        Evidence(constraint_id="c_fail", outcome=CheckOutcome.FAIL, checked_artifact=ref),
        Evidence(constraint_id="c_pass", outcome=CheckOutcome.PASS, checked_artifact=ref),
    ]
    merged = _merge_evidence([], fresh)
    keys = {(item.checked_artifact.artifact_id, item.constraint_id, item.outcome) for item in merged}
    assert keys == {("doc", "c_fail", CheckOutcome.FAIL), ("doc", "c_pass", CheckOutcome.PASS)}


def test_merge_evidence_replaces_only_the_matching_constraint_on_a_later_check() -> None:
    ref_v1 = ArtifactRef(artifact_id="doc", version=1, kind=ArtifactKind.DOCUMENT)
    ref_v2 = ArtifactRef(artifact_id="doc", version=2, kind=ArtifactKind.DOCUMENT)
    existing = [
        Evidence(constraint_id="c_fail", outcome=CheckOutcome.FAIL, checked_artifact=ref_v1),
        Evidence(constraint_id="c_pass", outcome=CheckOutcome.PASS, checked_artifact=ref_v1),
    ]
    fresh = [Evidence(constraint_id="c_fail", outcome=CheckOutcome.PASS, checked_artifact=ref_v2)]
    merged = _merge_evidence(existing, fresh)
    by_constraint = {item.constraint_id: item for item in merged}
    assert by_constraint["c_fail"].outcome == CheckOutcome.PASS
    assert by_constraint["c_fail"].checked_artifact.version == 2
    assert by_constraint["c_pass"].outcome == CheckOutcome.PASS  # untouched


def test_prune_invalidated_evidence_drops_entries_whose_input_moved() -> None:
    """Repro: a PASS on "b" depending on "a" must stop reading as current
    once "a" moves to a new version -- the controller's own cache must
    stay synchronized with exactly the invalidation rule
    ArtifactGraphStore.live_evidence() already applies internally."""
    store = ArtifactGraphStore()
    a_ref = store.add_version("a", ArtifactKind.DEFINITION, content="a1").ref
    b_ref = store.add_version("b", ArtifactKind.CLAIM, content="b1", read_input_versions=[a_ref]).ref
    evidence = Evidence(constraint_id="c0", outcome=CheckOutcome.PASS, checked_artifact=b_ref, input_versions=[a_ref])
    assert store.is_evidence_still_valid(evidence) is True
    assert _prune_invalidated_evidence(store, [evidence]) == [evidence]

    store.add_version("a", ArtifactKind.DEFINITION, content="a2")  # "a" moves -> invalidates the PASS about "b"
    assert store.is_evidence_still_valid(evidence) is False
    assert _prune_invalidated_evidence(store, [evidence]) == []


# -- Bug 2: a failed/partial repair must not silently look resolved --------


def test_production_failure_does_not_let_the_controller_proceed_to_compile() -> None:
    """Repro: a repair whose Producer always reports FAILED never writes a
    new version, so the target stays is_stale() forever -- the first
    Stage 5 pass dropped the evidence for it without replacing it with
    anything, so the controller eventually read "nothing failing" and
    proceeded straight to COMPILE despite the artifact never actually
    having been fixed."""
    store = ArtifactGraphStore(branch_id="episode-prod-fail")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    policy = RuleBasedController(doc_ref, _constraint_provider, request_id="req1", output_dir="/tmp/x")
    runtime = ControllerRuntime(
        store,
        policy,
        _AlwaysFailValidator(),
        {ArtifactKind.DOCUMENT: _AlwaysFailingProducer()},
        _constraint_provider,
        document_ref=doc_ref,
        budget=Budget(cap=1000.0),
        max_steps=10,
        repeat_threshold=3,
    )
    final_state = runtime.run(thread_id="t-prod-fail")
    assert final_state.done is True
    assert not any(record.action == ToolActionKind.COMPILE for record in final_state.history)
    assert final_state.stop_reason == ControllerStopReason.MAX_STEPS_REACHED
    assert "doc" in final_state.tracked_artifact_ids
    assert store.latest("doc").ref.version == 1  # production never actually succeeded, ever


def test_production_failure_on_a_never_stale_artifact_is_not_hidden_by_an_unrelated_pass() -> None:
    """The precise repro behind bug 2a: "doc" is failing its own content
    check (not a dependency issue), so it is NEVER is_stale()==True at
    any point -- is_stale() alone cannot represent "this production
    attempt failed" for an artifact like this. Mixed in with an already-
    tracked, already-PASSING artifact ("other"): once "doc"'s dropped
    evidence is not replaced with anything, classify() reads PASS from
    "other" alone and the controller must NOT be fooled into compiling."""
    store = ArtifactGraphStore(branch_id="episode-prod-fail-mixed")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    other_ref = store.add_version("other", ArtifactKind.BLOCK, content="fine").ref
    assert store.is_stale("doc") is False  # never a dependency issue -- just a bad content check

    policy = RuleBasedController(doc_ref, _constraint_provider, request_id="req1", output_dir="/tmp/x")
    state = ControllerState(
        document_ref=doc_ref,
        checkpoint=store.save_checkpoint(),
        budget=Budget(cap=1000.0),
        current_evidence=[
            Evidence(constraint_id="c0", outcome=CheckOutcome.FAIL, checked_artifact=doc_ref),
            Evidence(constraint_id="c0", outcome=CheckOutcome.PASS, checked_artifact=other_ref),
        ],
        tracked_artifact_ids={"doc", "other"},
        compiled=False,
    )
    tool_executor = ToolExecutor(
        store,
        _AlwaysFailValidator(),
        budget=Budget(cap=1000.0),
        max_actions=10,
        producer_registry={ArtifactKind.DOCUMENT: _AlwaysFailingProducer()},
        constraint_provider=_constraint_provider,
    )
    # Step 1: FAIL on "doc" drives a repair attempt.
    call = policy.decide(state, store, tool_executor)
    assert call.action.value in ("patch", "replan")
    observation = tool_executor.execute(call)
    assert observation.ok is True
    result = observation.output.result
    assert result.final_outcome_by_artifact["doc"].value == "production_failed"
    assert store.is_stale("doc") is False  # still never flagged stale -- this is the whole point

    # Fold the observation back exactly as _tool_node would, to get the
    # next state to decide from.
    from src.core.controller import _drop_evidence_for, _update_blocked

    touched = set(result.final_outcome_by_artifact.keys())
    pruned = _drop_evidence_for(state.current_evidence, touched)
    next_state = state.model_copy(
        update={
            "current_evidence": pruned,  # "doc"'s FAIL is gone; "other"'s PASS remains
            "tracked_artifact_ids": state.tracked_artifact_ids | touched,
            "blocked_artifact_ids": _update_blocked(state.blocked_artifact_ids, result),
        }
    )
    assert next_state.blocked_artifact_ids == {"doc"}

    # Step 2: must NOT silently compile just because "other" still PASSes.
    next_call = policy.decide(next_state, store, tool_executor)
    assert next_call.action.value != "compile"


def test_tool_node_keeps_a_cascade_discovered_no_producer_failure_tracked_as_unresolved() -> None:
    """Repro: a->b, a->c (two direct dependents forces REPLAN_SUBGRAPH).
    "b" has no registered producer at all (NO_PRODUCER) while "a"/"c"
    succeed. The first Stage 5 pass only tracked artifacts touched by
    final_outcome_by_artifact for DROPPING old evidence, with nothing
    keeping "b" visible as still-broken once its (nonexistent) evidence
    was gone."""
    store = ArtifactGraphStore(branch_id="episode-cascade")
    a_ref = store.add_version("a", ArtifactKind.DEFINITION, content="a-bad").ref
    b_ref = store.add_version("b", ArtifactKind.CLAIM, content="b1", read_input_versions=[a_ref]).ref
    c_ref = store.add_version("c", ArtifactKind.TABLE, content="c1", read_input_versions=[a_ref]).ref

    class _FailOnlyOldA:
        def __init__(self) -> None:
            self.config: Dict[str, bool] = {}

        def check(self, artifact, constraints, context=None):  # noqa: ANN001
            outcome = CheckOutcome.FAIL if (artifact.artifact_id == "a" and artifact.version == 1) else CheckOutcome.PASS
            return [Evidence(constraint_id=c.constraint_id, outcome=outcome, checked_artifact=artifact) for c in constraints]

    tool_executor = ToolExecutor(
        store,
        _FailOnlyOldA(),
        budget=Budget(cap=1000.0),
        max_actions=100,
        producer_registry={ArtifactKind.DEFINITION: _FakeProducer(), ArtifactKind.TABLE: _FakeProducer()},
        constraint_provider=_constraint_provider,
    )
    from src.core.schemas import ReplanCall

    observation = tool_executor.execute(ReplanCall(targets=[a_ref, b_ref, c_ref]))
    assert observation.ok is True
    result = observation.output.result
    assert result.final_outcome_by_artifact["a"].value == "resolved"
    assert result.final_outcome_by_artifact["c"].value == "resolved"
    assert result.final_outcome_by_artifact["b"].value == "no_producer"
    assert store.is_stale("b") is True

    unresolved = _compute_unresolved(store, {"a", "b", "c"}, set())
    assert unresolved == {"b"}


def test_rule_based_controller_checks_an_unresolved_tracked_artifact_before_compiling() -> None:
    """A tracked artifact that is stale but has never been checked must be
    investigated (CHECK) before the controller even considers compiling,
    even though the document itself already has clean PASS evidence."""
    store = ArtifactGraphStore()
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="doc1").ref
    a_ref = store.add_version("a", ArtifactKind.DEFINITION, content="a1").ref
    b_ref = store.add_version("b", ArtifactKind.CLAIM, content="b1", read_input_versions=[a_ref]).ref
    store.add_version("a", ArtifactKind.DEFINITION, content="a2")  # marks "b" stale
    assert store.is_stale("b") is True

    policy = RuleBasedController(doc_ref, _constraint_provider, request_id="req1", output_dir="/tmp/x")
    state = ControllerState(
        document_ref=doc_ref,
        checkpoint=store.save_checkpoint(),
        budget=Budget(cap=1000.0),
        current_evidence=[Evidence(constraint_id="c0", outcome=CheckOutcome.PASS, checked_artifact=doc_ref)],
        tracked_artifact_ids={"doc", "b"},
        compiled=False,
    )
    tool_executor = ToolExecutor(store, _AlwaysFailValidator(), budget=Budget(cap=1000.0), max_actions=10)
    call = policy.decide(state, store, tool_executor)
    assert call.action.value == "check"
    assert call.artifact.artifact_id == "b"


def test_rule_based_controller_finishes_unresolved_rather_than_compiling() -> None:
    """Once "b" HAS been checked (so there is no more fresh evidence to
    gather) but is still is_stale()-true with nothing more this policy can
    try (no FAIL evidence names it), the controller must stop honestly
    rather than silently proceeding to COMPILE."""
    store = ArtifactGraphStore()
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="doc1").ref
    a_ref = store.add_version("a", ArtifactKind.DEFINITION, content="a1").ref
    b_ref = store.add_version("b", ArtifactKind.CLAIM, content="b1", read_input_versions=[a_ref]).ref
    store.add_version("a", ArtifactKind.DEFINITION, content="a2")  # marks "b" stale
    store.record_evidence(Evidence(constraint_id="c0", outcome=CheckOutcome.PASS, checked_artifact=store.latest("b").ref))
    assert store.is_stale("b") is True  # checking alone never clears staleness -- only a new version does

    policy = RuleBasedController(doc_ref, _constraint_provider, request_id="req1", output_dir="/tmp/x")
    state = ControllerState(
        document_ref=doc_ref,
        checkpoint=store.save_checkpoint(),
        budget=Budget(cap=1000.0),
        current_evidence=[
            Evidence(constraint_id="c0", outcome=CheckOutcome.PASS, checked_artifact=doc_ref),
            Evidence(constraint_id="c0", outcome=CheckOutcome.PASS, checked_artifact=store.latest("b").ref),
        ],
        tracked_artifact_ids={"doc", "b"},
        compiled=False,
    )
    tool_executor = ToolExecutor(store, _AlwaysFailValidator(), budget=Budget(cap=1000.0), max_actions=10)
    call = policy.decide(state, store, tool_executor)
    assert call.action.value == "finish"
    assert call.reason == "unresolved_remaining"
    assert call.unresolved_artifact_ids == ["b"]


# -- Bug 3: UNKNOWN must never read as resolved; compiled must be reset ----
# -- the moment anything is repaired again ---------------------------------


def test_rule_based_controller_does_not_report_resolved_on_unknown_evidence() -> None:
    store = ArtifactGraphStore()
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    policy = RuleBasedController(doc_ref, _constraint_provider, request_id="req1", output_dir="/tmp/x")
    state = ControllerState(
        document_ref=doc_ref,
        checkpoint=store.save_checkpoint(),
        budget=Budget(cap=1000.0),
        current_evidence=[
            Evidence(
                constraint_id="c0",
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=doc_ref,
                unknown_reason=UnknownReason.NOT_REPRESENTABLE,
            )
        ],
        compiled=True,
    )
    tool_executor = ToolExecutor(store, _AlwaysFailValidator(), budget=Budget(cap=1000.0), max_actions=10)
    call = policy.decide(state, store, tool_executor)
    assert call.action.value == "finish"
    assert call.reason == "evidence_inconclusive"
    assert call.unresolved_artifact_ids == ["doc"]


def test_tool_node_resets_compiled_state_after_any_successful_repair_dispatch() -> None:
    store = ArtifactGraphStore()
    ref = store.add_version("a", ArtifactKind.BLOCK, content="old").ref
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="docv1").ref
    policy = RuleBasedController(doc_ref, lambda r: [], request_id="req1", output_dir="/tmp/x")
    runtime = ControllerRuntime(
        store,
        policy,
        _AlwaysFailValidator(),
        {ArtifactKind.BLOCK: _FakeProducer()},
        lambda r: [],
        document_ref=doc_ref,
        budget=Budget(cap=1000.0),
    )
    state = ControllerState(
        document_ref=doc_ref,
        checkpoint=store.save_checkpoint(),
        budget=runtime._tool_executor.budget,
        compiled=True,
        last_compile_failed=True,
        pending_tool_call=PatchCall(target=ref),
    )
    updates = runtime._tool_node(state)
    assert updates["compiled"] is False
    assert updates["last_compile_failed"] is False


def test_rule_based_controller_stops_cleanly_after_a_failed_compile_instead_of_retrying() -> None:
    """A rule-based policy has no way to interpret a LaTeX compiler
    diagnostic -- once COMPILE has already failed once, it must stop
    honestly instead of blindly re-issuing an identical COMPILE call
    until the repeated-action guard eventually intervenes."""
    store = ArtifactGraphStore()
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    policy = RuleBasedController(doc_ref, _constraint_provider, request_id="req1", output_dir="/tmp/x")
    state = ControllerState(
        document_ref=doc_ref,
        checkpoint=store.save_checkpoint(),
        budget=Budget(cap=1000.0),
        current_evidence=[Evidence(constraint_id="c0", outcome=CheckOutcome.PASS, checked_artifact=doc_ref)],
        compiled=False,
        last_compile_failed=True,
    )
    tool_executor = ToolExecutor(store, _AlwaysFailValidator(), budget=Budget(cap=1000.0), max_actions=10)
    call = policy.decide(state, store, tool_executor)
    assert call.action.value == "finish"
    assert call.reason == "compile_failed"


# -- Bug 4: the next decision must see the real observation, not just ok ---


def test_describe_observation_includes_check_evidence_detail() -> None:
    store = ArtifactGraphStore()
    ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    executor = ToolExecutor(store, _AlwaysFailValidator(), budget=Budget(cap=1000.0), max_actions=10)
    call = CheckCall(artifact=ref, constraints=_constraint_provider(ref), context={})
    observation = executor.execute(call)
    detail = _describe_observation(call, observation)
    assert "c0=fail" in detail
    assert "always fails" in detail


def test_describe_observation_includes_compile_diagnostics_on_failure() -> None:
    store = ArtifactGraphStore()
    executor = ToolExecutor(store, _AlwaysFailValidator(), budget=Budget(cap=1000.0), max_actions=10)
    call = CompileCall(request_id="missing-main-tex", output_dir="/tmp/definitely-does-not-exist-xyz")
    observation = executor.execute(call)
    detail = _describe_observation(call, observation)
    assert "compile missing-main-tex: success=False" in detail


def test_llm_controller_user_prompt_includes_the_last_steps_observed_detail() -> None:
    store = ArtifactGraphStore()
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    state = ControllerState(
        document_ref=doc_ref,
        checkpoint=store.save_checkpoint(),
        budget=Budget(cap=100.0),
        history=[
            ControllerStepRecord(
                step_index=0,
                action=ToolActionKind.CHECK,
                call_summary="check:doc@1",
                ok=True,
                detail="check doc@1: c0=fail(boom)",
            )
        ],
    )
    policy = LLMController(doc_ref, _FakeLLMGenerator([]), request_id="req1")
    prompt = policy._user_prompt(state, store)
    assert "c0=fail(boom)" in prompt


# -- Bug 5: durable resume must carry over controller + executor state -----


def test_resume_from_checkpoint_carries_over_history_step_count_and_admission_state() -> None:
    store = ArtifactGraphStore(branch_id="original-branch")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    checkpoint = store.save_checkpoint(label="before-interruption")
    prior_history = [
        ControllerStepRecord(step_index=i, action=ToolActionKind.CHECK, call_summary="check:doc@1", ok=True)
        for i in range(7)
    ]
    prior_state = ControllerState(
        document_ref=doc_ref,
        checkpoint=checkpoint,
        budget=Budget(cap=1000.0, spent=40.0),
        step_count=7,
        actions_taken=7,
        suspended=False,
        history=prior_history,
        current_evidence=[Evidence(constraint_id="c0", outcome=CheckOutcome.FAIL, checked_artifact=doc_ref)],
        tracked_artifact_ids={"doc"},
        compiled=False,
    )

    built_with: List[ArtifactGraphStore] = []

    def checker_factory(resumed_store: ArtifactGraphStore) -> _AlwaysFailValidator:
        built_with.append(resumed_store)
        return _AlwaysFailValidator()

    policy = RuleBasedController(doc_ref, _constraint_provider, request_id="req1", output_dir="/tmp/x")
    resumed_runtime = ControllerRuntime.resume_from_checkpoint(
        prior_state,
        policy,
        checker_factory,
        {ArtifactKind.DOCUMENT: _FakeProducer()},
        _constraint_provider,
        new_branch_id="resumed-branch",
        max_steps=20,
        repeat_threshold=3,
    )

    assert built_with == [resumed_runtime.store]  # the Validator was bound to THIS store, not the original
    assert resumed_runtime.store is not store  # a fully independent instance
    assert resumed_runtime._tool_executor.actions_taken == 7  # admission state carried over, not reset
    assert resumed_runtime._tool_executor.budget.spent == 40.0

    final_state = resumed_runtime.run(thread_id="t-resumed")

    assert len(final_state.history) > 7
    assert [record.call_summary for record in final_state.history[:7]] == ["check:doc@1"] * 7
    assert final_state.step_count > 7
    assert final_state.budget.spent >= 40.0
    resumed_checkpoint = resumed_runtime.store.save_checkpoint()
    assert resumed_checkpoint.source_branch_id == "resumed-branch"
    # the original store/branch must be completely untouched by the resumed run
    assert store.latest("doc").ref.version == 1


# -- Bug 6/7: the LLM controller's own decision call must be budget-gated --
# -- and must never raise past decide() -------------------------------------


def test_llm_controller_own_decision_call_is_rejected_when_budget_already_exhausted() -> None:
    store = ArtifactGraphStore()
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    generator = _FakeLLMGenerator(["should never be consumed"])
    policy = LLMController(doc_ref, generator, request_id="req1")
    tool_executor = ToolExecutor(store, _AlwaysFailValidator(), budget=Budget(cap=0.0), max_actions=10)
    state = ControllerState(document_ref=doc_ref, checkpoint=store.save_checkpoint(), budget=Budget(cap=0.0))
    call = policy.decide(state, store, tool_executor)
    assert call.action.value == "finish"
    assert call.reason.startswith("llm_controller_call_rejected_")
    assert generator.calls == 0  # the model itself was never actually invoked


def test_llm_controller_own_decision_call_charges_the_shared_budget() -> None:
    store = ArtifactGraphStore()
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    decision_json = '{"call": {"action": "finish", "reason": "done", "unresolved_artifact_ids": []}, "rationale": ""}'
    generator = _FakeLLMGenerator([decision_json], tokens_per_call=42)
    policy = LLMController(doc_ref, generator, request_id="req1")
    tool_executor = ToolExecutor(store, _AlwaysFailValidator(), budget=Budget(cap=1000.0), max_actions=10)
    state = ControllerState(document_ref=doc_ref, checkpoint=store.save_checkpoint(), budget=Budget(cap=1000.0))
    before = tool_executor.budget.spent
    call = policy.decide(state, store, tool_executor)
    assert call.action.value == "finish"
    assert tool_executor.budget.spent == before + 42


def test_llm_controller_retries_then_safely_finishes_when_generate_keeps_raising() -> None:
    """Repro: generator.generate() itself raises (e.g. a timeout) on every
    attempt. The first Stage 5 pass only caught a JSON-parse failure --
    this exception propagated straight out of decide() uncaught."""
    store = ArtifactGraphStore()
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    generator = _FakeLLMGenerator([], raises=[TimeoutError("boom"), TimeoutError("boom"), TimeoutError("boom")])
    policy = LLMController(doc_ref, generator, request_id="req1", max_attempts=3)
    tool_executor = ToolExecutor(store, _AlwaysFailValidator(), budget=Budget(cap=1000.0), max_actions=10)
    state = ControllerState(document_ref=doc_ref, checkpoint=store.save_checkpoint(), budget=Budget(cap=1000.0))
    call = policy.decide(state, store, tool_executor)  # must not raise
    assert call.action.value == "finish"
    assert call.reason == "llm_controller_exhausted_retries"
    assert generator.calls == 3


def test_llm_controller_recovers_after_a_transient_generate_exception() -> None:
    """A limited, bounded retry must actually succeed once the transient
    failure stops happening -- confirms the retry loop recovers, not just
    that it eventually gives up."""
    store = ArtifactGraphStore()
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    decision_json = '{"call": {"action": "finish", "reason": "done", "unresolved_artifact_ids": []}, "rationale": ""}'
    generator = _FakeLLMGenerator([decision_json], raises=[ConnectionError("flaky"), None])
    policy = LLMController(doc_ref, generator, request_id="req1", max_attempts=3)
    tool_executor = ToolExecutor(store, _AlwaysFailValidator(), budget=Budget(cap=1000.0), max_actions=10)
    state = ControllerState(document_ref=doc_ref, checkpoint=store.save_checkpoint(), budget=Budget(cap=1000.0))
    call = policy.decide(state, store, tool_executor)
    assert call.action.value == "finish"
    assert call.reason == "done"
    assert generator.calls == 2


def test_llm_controller_executes_a_valid_structured_decision() -> None:
    store = ArtifactGraphStore(branch_id="episode-llm")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    decision_json = (
        '{"call": {"action": "finish", "reason": "llm says done", "unresolved_artifact_ids": []}, '
        '"rationale": "nothing to do"}'
    )
    generator = _FakeLLMGenerator([decision_json])
    policy = LLMController(doc_ref, generator, request_id="req1")
    runtime = ControllerRuntime(
        store,
        policy,
        _AlwaysFailValidator(),
        {ArtifactKind.DOCUMENT: _FakeProducer()},
        _constraint_provider,
        document_ref=doc_ref,
        budget=Budget(cap=100.0),
        max_steps=10,
    )
    final_state = runtime.run(thread_id="t-llm")
    assert final_state.done is True
    assert final_state.stop_reason == ControllerStopReason.FINISHED
    assert generator.calls == 1
    assert final_state.history[0].call_summary == "finish:llm says done"


# -- Anti-infinite-loop / repeated-fix guard ---------------------------------


def test_repeated_identical_successful_action_triggers_the_repeat_guard() -> None:
    store = ArtifactGraphStore(branch_id="episode-loop")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref

    class _AlwaysInspectPolicy:
        def decide(self, state, store, tool_executor) -> Any:  # noqa: ANN001 - Protocol signature
            from src.core.schemas import InspectArtifactCall

            return InspectArtifactCall(artifact_id="doc")

    runtime = ControllerRuntime(
        store,
        _AlwaysInspectPolicy(),
        _AlwaysFailValidator(),
        {ArtifactKind.DOCUMENT: _FakeProducer()},
        _constraint_provider,
        document_ref=doc_ref,
        budget=Budget(cap=1000.0),
        max_steps=50,
        repeat_threshold=3,
    )
    final_state = runtime.run(thread_id="t-loop")
    assert final_state.done is True
    assert final_state.stop_reason == ControllerStopReason.REPEATED_ACTION_DETECTED
    assert final_state.step_count == 4  # 3 identical inspects + the forced FINISH dispatch
    assert final_state.history[-1].action.value == "finish"


# -- Budget-overrun guard -----------------------------------------------------


def test_budget_exhaustion_stops_the_episode_even_mid_decision() -> None:
    store = ArtifactGraphStore(branch_id="episode-budget")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    policy = RuleBasedController(doc_ref, _constraint_provider, request_id="req1", output_dir="/tmp/x")
    runtime = ControllerRuntime(
        store,
        policy,
        _AlwaysFailValidator(),
        {ArtifactKind.DOCUMENT: _FakeProducer()},
        _constraint_provider,
        document_ref=doc_ref,
        budget=Budget(cap=0.0),  # already exhausted before the first call
        max_steps=50,
        repeat_threshold=100,
    )
    final_state = runtime.run(thread_id="t-budget")
    assert final_state.done is True
    assert final_state.stop_reason == ControllerStopReason.BUDGET_EXHAUSTED
    assert final_state.step_count == 1
    assert final_state.history[0].ok is False
    assert final_state.history[0].error_code.value == "budget_exceeded"


# -- compute_ready_artifacts --------------------------------------------------


def test_compute_ready_artifacts_requires_staleness_and_fresh_inputs() -> None:
    store = ArtifactGraphStore()
    a_ref = store.add_version("a", ArtifactKind.DEFINITION, content="a1").ref
    store.add_version("b", ArtifactKind.CLAIM, content="b1", read_input_versions=[a_ref])
    assert store.is_stale("b") is False  # nothing has changed yet

    store.add_version("a", ArtifactKind.DEFINITION, content="a2")  # marks "b" stale
    assert store.is_stale("b") is True
    assert store.is_stale("a") is False

    ready = compute_ready_artifacts(store, ["a", "b"])
    assert ready == {"b"}  # "a" isn't stale (nothing to do); "b" is stale AND its input "a" is fresh


def test_compute_ready_artifacts_excludes_a_stale_artifact_with_a_stale_input() -> None:
    store = ArtifactGraphStore()
    a_ref = store.add_version("a", ArtifactKind.DEFINITION, content="a1").ref
    store.add_version("b", ArtifactKind.CLAIM, content="b1", read_input_versions=[a_ref])
    store.add_version("a", ArtifactKind.DEFINITION, content="a2")  # "b" now stale
    b_latest = store.latest("b")
    assert b_latest is not None
    store.add_version("c", ArtifactKind.CLAIM, content="c1", read_input_versions=[b_latest.ref])

    ready = compute_ready_artifacts(store, ["a", "b", "c"])
    # "b" is ready (stale, input "a" is fresh); "c" is stale too but its
    # input "b" is itself stale -- not ready yet.
    assert ready == {"b"}


def test_compute_ready_artifacts_ignores_non_stale_candidates_entirely() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    assert compute_ready_artifacts(store, ["a", "does-not-exist"]) == set()

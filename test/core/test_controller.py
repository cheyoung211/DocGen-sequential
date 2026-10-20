"""Coverage for src/core/controller.py (Stage 5): the LangGraph controller
loop (ControllerRuntime), both policy variants (RuleBasedController/
LLMController), anti-infinite-loop/budget-overrun guards, durable
resume-from-checkpoint, and compute_ready_artifacts. Uses
Protocol-conforming fakes throughout -- no real LLM call and no real
local Qwen3.6 server dependency anywhere in this file, per this stage's
confirmed scope decision (see controller.py's module docstring): a real
pilot run is an explicitly deferred follow-up."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.artifact_graph import ArtifactGraphStore
from src.core.controller import (
    ControllerRuntime,
    LLMController,
    RuleBasedController,
    compute_ready_artifacts,
)
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    CheckOutcome,
    Constraint,
    ConstraintKind,
    ControllerState,
    ControllerStopReason,
    Evidence,
    ProductionResult,
    ProductionStatus,
)


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
        if artifact.version <= self.fail_below_version:
            return [
                Evidence(constraint_id=c.constraint_id, outcome=CheckOutcome.FAIL, checked_artifact=artifact, detail="fail")
                for c in constraints
            ] or [Evidence(constraint_id="c0", outcome=CheckOutcome.FAIL, checked_artifact=artifact, detail="fail")]
        return [
            Evidence(constraint_id=c.constraint_id, outcome=CheckOutcome.PASS, checked_artifact=artifact)
            for c in constraints
        ] or [Evidence(constraint_id="c0", outcome=CheckOutcome.PASS, checked_artifact=artifact)]


class _AlwaysFailValidator:
    def __init__(self) -> None:
        self.config: Dict[str, bool] = {}

    def check(
        self, artifact: ArtifactRef, constraints: List[Constraint], context: Optional[Dict[str, Any]] = None
    ) -> List[Evidence]:
        return [Evidence(constraint_id="c0", outcome=CheckOutcome.FAIL, checked_artifact=artifact, detail="always fails")]


class _FakeLLMGenerator:
    """Returns a scripted sequence of raw text responses, one per call --
    satisfies ``controller.LLMGenerator`` without any network/local-server
    dependency."""

    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        self.calls += 1
        return self.responses.pop(0)

    def extract_json_block(self, text: str) -> str:
        return text


# -- RuleBasedController + the full controller->tool->controller loop -------


def test_rule_based_controller_resolves_a_failing_document_then_compiles_then_finishes() -> None:
    """The exact trace the Stage 5 completion criterion asks for: a
    different observation (FAIL -> PASS -> compiled) leads to a different
    next action (CHECK -> REPLAN -> CHECK -> COMPILE -> FINISH)."""
    store = ArtifactGraphStore(branch_id="episode-1")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    producer = _FakeProducer()
    validator = _VersionThresholdValidator(fail_below_version=1)
    policy = RuleBasedController(doc_ref, _constraint_provider, request_id="req1", output_dir="/tmp/nonexistent")
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
    assert actions[2] == "check:doc@2"
    assert "compile:req1" in actions[3]
    assert final_state.history[-1].action.value == "finish"
    assert producer.calls == ["doc"]
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

        def decide(self, state, store) -> Any:  # noqa: ANN001 - Protocol signature
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


# -- Anti-infinite-loop / repeated-fix guard ---------------------------------


def test_repeated_identical_successful_action_triggers_the_repeat_guard() -> None:
    """A policy that keeps choosing the identical action, each dispatch
    succeeding (ok=True) but never actually resolving anything (no FAIL
    evidence to react to, nothing new to compile toward), must be stopped
    by the repeated-action guard -- not merely by the step cap -- so this
    sets repeat_threshold well below max_steps and checks the guard fires
    first."""
    store = ArtifactGraphStore(branch_id="episode-loop")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref

    class _AlwaysInspectPolicy:
        def decide(self, state, store) -> Any:  # noqa: ANN001 - Protocol signature
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


def test_a_run_of_still_failing_repairs_does_not_trip_the_repeat_guard() -> None:
    """The repeat guard is specifically about the SAME action succeeding
    without progress -- a run of genuinely still-failing repairs on the
    same target (ok=True at the dispatch level, but the underlying problem
    persists) must not be confused with a loop by this guard; it is
    expected to run until the step cap, not the repeat-detector, stops
    it."""
    store = ArtifactGraphStore(branch_id="episode-stuck")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    policy = RuleBasedController(doc_ref, _constraint_provider, request_id="req1", output_dir="/tmp/nonexistent")
    runtime = ControllerRuntime(
        store,
        policy,
        _AlwaysFailValidator(),  # never resolves, no matter how many times "doc" is regenerated
        {ArtifactKind.DOCUMENT: _FakeProducer()},
        _constraint_provider,
        document_ref=doc_ref,
        budget=Budget(cap=1000.0),
        max_steps=6,
        repeat_threshold=3,
    )
    final_state = runtime.run(thread_id="t-stuck")
    assert final_state.done is True
    assert final_state.stop_reason == ControllerStopReason.MAX_STEPS_REACHED
    # every regenerate->recheck pair alternates "replan:doc@N" / "check:doc@N+1" --
    # never 3 *identical* call_summary strings in a row, so the repeat guard
    # (which requires an exact-string repeat) correctly never fires here.
    summaries = [record.call_summary for record in final_state.history]
    assert len(set(summaries[-3:])) > 1


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


# -- Durable resume from a Stage 2 Checkpoint --------------------------------


def test_resume_from_checkpoint_reconstructs_the_store_under_a_new_branch_id() -> None:
    """A durable resume must create a genuinely new branch (Stage 2's own
    rule: a restored, LLM-calling continuation is logged as a separate
    run), not silently continue writing into the original branch_id."""
    store = ArtifactGraphStore(branch_id="original-branch")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    checkpoint = store.save_checkpoint(label="before-interruption")

    built_with: List[ArtifactGraphStore] = []

    def checker_factory(resumed_store: ArtifactGraphStore) -> _AlwaysFailValidator:
        built_with.append(resumed_store)
        return _AlwaysFailValidator()

    policy = RuleBasedController(doc_ref, _constraint_provider, request_id="req1", output_dir="/tmp/x")
    resumed_runtime = ControllerRuntime.resume_from_checkpoint(
        checkpoint,
        policy,
        checker_factory,
        {ArtifactKind.DOCUMENT: _FakeProducer()},
        _constraint_provider,
        document_ref=doc_ref,
        new_branch_id="resumed-branch",
        budget=Budget(cap=5.0),
        max_steps=10,
        repeat_threshold=3,
    )

    assert built_with == [resumed_runtime.store]  # the Validator was bound to THIS store, not the original
    assert resumed_runtime.store is not store  # a fully independent instance
    final_state = resumed_runtime.run(thread_id="t-resumed")
    assert final_state.done is True
    # the resumed store's own history must reflect the new branch, never the original
    resumed_checkpoint = resumed_runtime.store.save_checkpoint()
    assert resumed_checkpoint.source_branch_id == "resumed-branch"
    # the original store/branch must be completely untouched by the resumed run
    assert store.latest("doc").ref.version == 1


# -- LLMController -------------------------------------------------------------


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


def test_llm_controller_falls_back_to_finish_after_repeated_unparseable_output() -> None:
    """A controller must always be able to end cleanly -- the same
    reasoning ``schemas.FinishCall`` being exempt from suspension already
    reflects one layer down. This must never raise past ``decide()``."""
    store = ArtifactGraphStore(branch_id="episode-llm-bad")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    generator = _FakeLLMGenerator(["not json at all", "{\"nope\": true}", "still not it"])
    policy = LLMController(doc_ref, generator, request_id="req1", max_attempts=3)
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
    final_state = runtime.run(thread_id="t-llm-bad")
    assert final_state.done is True
    assert generator.calls == 3
    assert final_state.history[0].call_summary == "finish:llm_controller_could_not_parse_a_valid_decision"


def test_llm_controller_decision_targeting_an_unseen_artifact_still_round_trips() -> None:
    """``LLMControllerDecision`` validates whatever the model names --
    this controller does not cross-check the artifact_id against the live
    store itself (that is ``ToolExecutor``'s/the real dispatch's job, not
    this parsing step's); confirms a CHECK decision for an arbitrary
    artifact_id parses and is returned as-is."""
    store = ArtifactGraphStore(branch_id="episode-llm-check")
    doc_ref = store.add_version("doc", ArtifactKind.DOCUMENT, content="v1").ref
    decision_json = (
        '{"call": {"action": "check", '
        '"artifact": {"artifact_id": "doc", "version": 1, "kind": "document"}, '
        '"constraints": [], "context": {}}, "rationale": "check it"}'
    )
    generator = _FakeLLMGenerator([decision_json])
    policy = LLMController(doc_ref, generator, request_id="req1")
    decision_call = policy.decide(
        ControllerState(document_ref=doc_ref, checkpoint=store.save_checkpoint(), budget=Budget(cap=100.0)),
        store,
    )
    assert decision_call.action.value == "check"
    assert decision_call.artifact.artifact_id == "doc"


# -- compute_ready_artifacts --------------------------------------------------


def test_compute_ready_artifacts_requires_staleness_and_fresh_inputs() -> None:
    store = ArtifactGraphStore()
    a_ref = store.add_version("a", ArtifactKind.DEFINITION, content="a1").ref
    b_result = store.add_version("b", ArtifactKind.CLAIM, content="b1", read_input_versions=[a_ref])
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
    # now also change "b" itself in a way that keeps it stale (simulate "c"
    # depending on the still-stale "b")
    b_latest = store.latest("b")
    assert b_latest is not None
    c_ref_input = b_latest.ref
    store.add_version("c", ArtifactKind.CLAIM, content="c1", read_input_versions=[c_ref_input])

    ready = compute_ready_artifacts(store, ["a", "b", "c"])
    # "b" is ready (stale, input "a" is fresh); "c" is stale too but its
    # input "b" is itself stale -- not ready yet.
    assert ready == {"b"}


def test_compute_ready_artifacts_ignores_non_stale_candidates_entirely() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.DEFINITION, content="a1")
    assert compute_ready_artifacts(store, ["a", "does-not-exist"]) == set()

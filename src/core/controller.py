"""LangGraph-based autonomous controller MVP (Stage 5 of the "Revision
2026.10.15" plan: "5단계. LangGraph 기반 자율 controller MVP").

Builds on every earlier stage without modifying any of it except
``tool_executor.py`` (see below): Stage 2's ``ArtifactGraphStore``
(versioned artifacts, checkpoint/branch), Stage 3's ``ToolExecutor``
(budget/action-count/suspend enforcement), and Stage 4's
``RepairOperationExecutor``/``FixedGraphAwareRepairPolicy`` (the
graph-aware repair-scope logic ``RuleBasedController`` reuses rather than
reimplementing).

Checklist items and where each is satisfied:

- "typed state, conditional routing, checkpoint/resume을 사용해 controller->
  tool/worker->observation->controller 루프를 구성한다" -> ``ControllerRuntime``:
  a ``langgraph.graph.StateGraph`` over ``schemas.ControllerState``, two
  nodes (``controller``/``tool``), one conditional edge routing back to
  ``controller`` or to ``END``.
- "남은 예산과 검증 결과를 읽는 고정 규칙 controller와 LLM controller를 각각
  연결한다" -> ``RuleBasedController``/``LLMController``, both implementing
  ``ControllerPolicy``.
- "finish·재계획·반복 검사 조건을 구현하고 무한 loop, 동일 수정 반복, 예산
  초과를 막는다" -> ``ControllerRuntime._controller_node``'s own guards
  (max-step cap, repeated-action detector), checked *before* ever calling
  ``ControllerPolicy.decide()`` and never inside it -- the same separation
  Stage 1 keeps between a deciding method (side-effect-free) and whatever
  enforces limits around it (``interfaces.RepairPolicy``'s docstring,
  ``tool_executor.ToolExecutor``'s own budget/action-count enforcement).
- "의존성이 충족된 독립 artifact만 병렬 처리하도록 readiness를 정의한다. 병렬
  실행은 버전 충돌 처리까지 준비된 뒤 적용한다" -> ``compute_ready_artifacts``:
  defined and tested, never called from ``ControllerRuntime``'s own loop --
  the controller stays strictly sequential this stage, as the checklist
  itself asks for.

**Three scope decisions confirmed with the user before writing this**
(recorded in Notion Progress.ver2, the 2026-10-20 Stage 5 entry):

(a) Mechanism + Protocol-conforming-fake tests only this round -- the same
    discipline every earlier stage used for anything that would otherwise
    require a real, expensive LLM call in a test. A real small-scale pilot
    against the local Qwen3.6 pipeline (``LLMController`` included) is an
    explicitly deferred follow-up, not part of this stage's own code.
(b) Checkpoint/resume goes beyond LangGraph's own in-process checkpointer
    (``langgraph.checkpoint.memory.InMemorySaver``, used by default --
    in-process replay/resume only): ``ControllerRuntime.resume_from_checkpoint``
    additionally reconstructs the artifact store from a durable Stage 2
    ``Checkpoint`` under a brand-new ``branch_id``, so a resumed execution
    survives an actual process restart and is still logged as a separate
    run, never folded into the interrupted branch's own history.
(c) ``tool_executor.ToolExecutor`` itself (not a bypassing layer in this
    module) now dispatches PATCH/REPLAN for real when constructed with
    ``producer_registry``/``constraint_provider`` -- see that module's own
    updated docstring. Every pre-Stage-5 construction (every existing
    call site and test) omits those two arguments and keeps the exact
    Stage 3 ``NOT_IMPLEMENTED`` behavior unchanged; this module is simply
    the first caller that supplies them.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Set, runtime_checkable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.core.artifact_graph import ArtifactGraphStore
from src.core.interfaces import Validator
from src.core.repair_operations import ConstraintProvider, FixedGraphAwareRepairPolicy, ProducerRegistry
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    BUDGET_ERROR_CODES,
    CheckCall,
    CheckOutcome,
    Checkpoint,
    CompileCall,
    ControllerState,
    ControllerStepRecord,
    ControllerStopReason,
    Evidence,
    FinishCall,
    InspectArtifactCall,
    LLMControllerDecision,
    PatchCall,
    QueryDependenciesCall,
    ReplanCall,
    RepairAction,
    ToolCall,
)
from src.core.tool_executor import ToolExecutor


# ---------------------------------------------------------------------------
# Shared helpers (used by both controller variants and ControllerRuntime)
# ---------------------------------------------------------------------------


def _classify(evidence: List[Evidence]) -> str:
    """The same FAIL/PASS/UNKNOWN three-way split Stage 4 established
    (``repair_operations._classify_evidence``), reimplemented here rather
    than importing that module's private helper so this module does not
    reach into another module's underscored internals -- both copies must
    independently honor the identical rule: PASS requires at least one
    PASS and zero FAIL; empty, or UNKNOWN-only, evidence is UNKNOWN, never
    silently promoted to PASS (the exact distinction
    ``RepairOutcome.INCONCLUSIVE`` exists to make visible one layer down)."""
    if any(item.outcome == CheckOutcome.FAIL for item in evidence):
        return "FAIL"
    if evidence and all(item.outcome == CheckOutcome.PASS for item in evidence):
        return "PASS"
    return "UNKNOWN"


def _merge_evidence(existing: List[Evidence], fresh: List[Evidence]) -> List[Evidence]:
    """A fresh CHECK's verdict for a given artifact always replaces
    whatever evidence this controller held for that artifact before --
    never accumulates side by side with it. Same latest-wins-per-key
    principle Stage 3's own ``ArtifactGraphStore.live_evidence()`` already
    applies one layer down, at the store level; this is the controller's
    own in-memory mirror of the same rule for the subset of evidence it is
    currently reasoning from."""
    by_id: Dict[str, Evidence] = {item.checked_artifact.artifact_id: item for item in existing}
    for item in fresh:
        by_id[item.checked_artifact.artifact_id] = item
    return list(by_id.values())


def _drop_evidence_for(existing: List[Evidence], artifact_ids: Set[str]) -> List[Evidence]:
    """After a repair touches a set of artifacts, whatever evidence this
    controller held for them is no longer trustworthy (the content just
    changed) -- drop it rather than let a stale verdict masquerade as
    current until the next CHECK. The next step's own ``_classify()``/
    policy logic then honestly reads those artifacts as UNKNOWN (absent),
    never as a fabricated PASS or FAIL."""
    return [item for item in existing if item.checked_artifact.artifact_id not in artifact_ids]


def _repair_action_to_tool_call(action: RepairAction) -> ToolCall:
    """Inverse of ``tool_executor.ToolExecutor``'s own ``PatchCall``/
    ``ReplanCall`` -> ``RepairActionType`` mapping (see that module's
    ``_dispatch_patch``/``_dispatch_replan``) -- picked so that converting
    a ``RepairAction`` to a ``ToolCall`` and back through that executor
    reproduces the *exact* ``action_type`` the policy chose. This matters
    most for REGENERATE_DOCUMENT, whose execution mechanism -- not just its
    label -- genuinely differs from the other three
    (``repair_operations.RepairOperationExecutor``'s own docstring): a
    single DOCUMENT-kind target must route through ``ReplanCall`` (->
    REGENERATE_DOCUMENT), never through ``PatchCall`` (whose own mapping
    for a non-BLOCK/non-SECTION single target is REPLAN_SUBGRAPH, the
    selective cascade -- silently downgrading "regenerate everything
    unconditionally" to "regenerate only what still fails")."""
    targets = action.target_artifacts
    if len(targets) == 1 and targets[0].kind != ArtifactKind.DOCUMENT:
        return PatchCall(target=targets[0], feedback_text=action.feedback_text)
    return ReplanCall(targets=targets, reason=action.feedback_text)


def _summarize_call(call: ToolCall) -> str:
    """A short, deterministic one-line identity for ``call`` -- used both
    as the human-readable execution trace (``ControllerStepRecord.
    call_summary``) and, by plain equality, as this stage's "동일 수정 반복"
    (repeated-fix) detector (``_repeated_action``). Two calls that would
    have the same real-world effect always produce the same string; two
    calls targeting different artifacts, or the same artifact at a
    different version, never do."""
    if isinstance(call, InspectArtifactCall):
        version = call.version if call.version is not None else "latest"
        return f"inspect:{call.artifact_id}@{version}"
    if isinstance(call, QueryDependenciesCall):
        return f"query:{call.artifact_id}"
    if isinstance(call, CheckCall):
        return f"check:{call.artifact.artifact_id}@{call.artifact.version}"
    if isinstance(call, PatchCall):
        return f"patch:{call.target.artifact_id}@{call.target.version}"
    if isinstance(call, ReplanCall):
        ids = ",".join(sorted(f"{t.artifact_id}@{t.version}" for t in call.targets))
        return f"replan:{ids}"
    if isinstance(call, CompileCall):
        return f"compile:{call.request_id}"
    if isinstance(call, FinishCall):
        return f"finish:{call.reason}"
    return call.action.value  # pragma: no cover - exhaustive over today's ToolCall union


def _repeated_action(history: List[ControllerStepRecord], *, threshold: int) -> bool:
    """True once the identical action signature was both chosen *and*
    reported ``ok=True`` for ``threshold`` steps in a row, with the
    controller never moving on in between -- the Notion checklist's "동일
    수정 반복" guard, deliberately distinct from the step/budget caps: a run
    of outright *failures* on the same target is not this kind of loop (it
    is simply retrying through a genuinely unresolved problem, already
    bounded by the step cap and by budget/max-actions one layer down) --
    this guard exists for the case a pure cost/step cap would not catch on
    its own: the same cheap, individually-successful action repeating
    without the controller ever actually progressing past it."""
    if threshold <= 0 or len(history) < threshold:
        return False
    tail = history[-threshold:]
    if not all(record.ok for record in tail):
        return False
    return len({record.call_summary for record in tail}) == 1


def compute_ready_artifacts(store: ArtifactGraphStore, candidate_ids: Iterable[str]) -> Set[str]:
    """Which of ``candidate_ids`` are *ready*: currently stale (there is
    real work to do) **and** every one of its own live inputs (the
    ``from_artifact`` side of every edge whose ``to_artifact`` names this
    artifact's current version) is itself not stale. Such an artifact
    could in principle be processed independently of every other ready
    artifact right now -- the Stage 5 checklist's "의존성이 충족된 독립
    artifact".

    Deliberately **not** called anywhere in ``ControllerRuntime``'s own
    loop, and deliberately not wired to any concurrent dispatch -- the
    checklist is explicit that "병렬 실행은 버전 충돌 처리까지 준비된 뒤
    적용한다" (parallel execution only once version-conflict handling
    exists), which this stage does not build. This function exists so
    "ready" is an inspectable, tested concept *now* -- a later stage that
    does add real concurrency has a definition to build against instead of
    inventing one under schedule pressure at the same time it is first
    wiring concurrent execution."""
    ready: Set[str] = set()
    for artifact_id in candidate_ids:
        if not store.is_stale(artifact_id):
            continue
        current = store.latest(artifact_id)
        if current is None:
            continue
        deps_fresh = True
        for edge in store.edges:
            if edge.to_artifact.artifact_id != artifact_id:
                continue
            if edge.to_artifact.version != current.ref.version:
                continue  # a superseded edge -- not a live relation
            if store.is_stale(edge.from_artifact.artifact_id):
                deps_fresh = False
                break
        if deps_fresh:
            ready.add(artifact_id)
    return ready


# ---------------------------------------------------------------------------
# Controller policies
# ---------------------------------------------------------------------------


@runtime_checkable
class ControllerPolicy(Protocol):
    """Decides the controller's *next single action*. Mirrors
    ``interfaces.RepairPolicy``'s side-effect-free discipline one layer up:
    ``decide()`` must not mutate ``store``, must not advance any retry/step
    counter of its own, and may be called without the returned ``ToolCall``
    necessarily ever being executed. Anti-loop/budget-overrun guards are
    deliberately **not** this Protocol's responsibility --
    ``ControllerRuntime._controller_node`` checks those *before* ever
    calling ``decide()`` and short-circuits with a synthetic ``FinishCall``
    when one trips, the same separation Stage 3's ``ToolExecutor`` keeps
    between a deciding method and the executor that enforces limits around
    it."""

    def decide(self, state: ControllerState, store: ArtifactGraphStore) -> ToolCall: ...


class RuleBasedController:
    """Deterministic policy reading remaining budget and the latest CHECK
    evidence: no evidence yet -> CHECK the document; a FAIL present ->
    delegate scope selection to ``repair_operations.FixedGraphAwareRepairPolicy``
    (Stage 4, unmodified) and execute the *first* (lowest target
    artifact_id) action it names this step -- any other independent
    failure that same ``decide()`` call also named is picked up on a later
    step once a fresh CHECK re-surfaces it, since this controller executes
    exactly one action per step (unlike ``branching_runner.BranchingRunner``,
    which executes every action a round's ``decide()`` returns within one
    round -- a deliberate difference: that class drives one *round* of
    several branches, this one drives one *step* of a single sequential
    loop). Nothing failing and not yet compiled -> COMPILE. Compiled ->
    FINISH."""

    def __init__(
        self,
        document_ref: ArtifactRef,
        constraint_provider: ConstraintProvider,
        *,
        request_id: str,
        output_dir: str = "outputs/generations",
    ) -> None:
        self._document_ref = document_ref
        self._constraints_for = constraint_provider
        self._request_id = request_id
        self._output_dir = output_dir
        self._repair_policy = FixedGraphAwareRepairPolicy(document_ref)

    def decide(self, state: ControllerState, store: ArtifactGraphStore) -> ToolCall:
        current = store.latest(self._document_ref.artifact_id)
        live_document_ref = current.ref if current is not None else self._document_ref

        if not state.current_evidence:
            return CheckCall(
                artifact=live_document_ref,
                constraints=self._constraints_for(live_document_ref),
                context={},
            )

        classification = _classify(state.current_evidence)
        if classification == "FAIL":
            failing = [item for item in state.current_evidence if item.outcome == CheckOutcome.FAIL]
            actions = self._repair_policy.decide(failing, store.edges, state.budget)
            if not actions:
                return FinishCall(
                    reason="repair_policy_returned_nothing",
                    unresolved_artifact_ids=[item.checked_artifact.artifact_id for item in failing],
                )
            chosen = sorted(actions, key=lambda action: action.target_artifacts[0].artifact_id)[0]
            return _repair_action_to_tool_call(chosen)

        if not state.compiled:
            return CompileCall(request_id=self._request_id, output_dir=self._output_dir)

        return FinishCall(reason="resolved", unresolved_artifact_ids=[])


@runtime_checkable
class LLMGenerator(Protocol):
    """The slice of ``scripts.llm_client.LLMClient`` ``LLMController``
    actually needs -- a Protocol, not the concrete class, so tests supply
    a fake without any network call or local-server dependency (this
    stage's own confirmed scope: mechanism + fakes now, a real pilot
    against the local Qwen3.6 server is an explicitly deferred follow-up).
    A real ``LLMClient`` instance already satisfies this Protocol as-is --
    ``generate()`` works for every provider including ``"local"`` (unlike
    ``generate_structured()``, which ``LLMClient.supports_structured_output()``
    reports as unavailable for the local/vLLM provider), and
    ``extract_json_block`` is an existing method ``TextAgent``/
    ``PlannerAgent`` already call the same way."""

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str: ...

    def extract_json_block(self, text: str) -> str: ...


class LLMController:
    """Asks an LLM to pick the next action directly, reading a compact text
    summary of remaining budget, current evidence, and recent history.
    Falls back to a safe ``FinishCall`` -- never raises past this boundary,
    never leaves the controller without a decision -- if the model's
    output fails to parse after ``max_attempts`` tries: a controller must
    always be able to end cleanly, the same reasoning ``schemas.FinishCall``
    being exempt from suspension already reflects one layer down."""

    def __init__(
        self,
        document_ref: ArtifactRef,
        generator: LLMGenerator,
        *,
        request_id: str,
        output_dir: str = "outputs/generations",
        max_attempts: int = 3,
    ) -> None:
        self._document_ref = document_ref
        self._generator = generator
        self._request_id = request_id
        self._output_dir = output_dir
        self._max_attempts = max_attempts

    def decide(self, state: ControllerState, store: ArtifactGraphStore) -> ToolCall:
        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(state)
        for _ in range(self._max_attempts):
            raw = self._generator.generate(system_prompt, user_prompt)
            try:
                payload = self._generator.extract_json_block(raw)
                decision = LLMControllerDecision.model_validate_json(payload)
            except Exception:
                continue
            return decision.call
        return FinishCall(
            reason="llm_controller_could_not_parse_a_valid_decision",
            unresolved_artifact_ids=[self._document_ref.artifact_id],
        )

    def _system_prompt(self) -> str:
        return (
            "You are an autonomous repair controller for a LaTeX document "
            "generation pipeline. Given the current state below, choose "
            "exactly one next action from: inspect, query, check, patch, "
            "replan, compile, finish. Respond with exactly one JSON object "
            "matching the provided schema -- the 'call' field's 'action' "
            "value picks which of that object's other fields apply. Never "
            "invent an artifact_id that was not shown to you in the state."
        )

    def _user_prompt(self, state: ControllerState) -> str:
        evidence_lines = [
            f"- {item.checked_artifact.artifact_id}@{item.checked_artifact.version}: "
            f"{item.outcome.value} ({item.constraint_id}) {item.detail}".strip()
            for item in state.current_evidence
        ] or ["(no evidence yet)"]
        history_lines = [
            f"- step {record.step_index}: {record.call_summary} -> "
            + ("ok" if record.ok else f"error={record.error_code.value if record.error_code else '?'}")
            for record in state.history[-5:]
        ] or ["(no steps yet)"]
        return (
            f"Document artifact: {self._document_ref.artifact_id}\n"
            f"Budget remaining: {state.budget.remaining} / cap {state.budget.cap}\n"
            f"Compiled this episode: {state.compiled}\n"
            "Current evidence:\n" + "\n".join(evidence_lines) + "\n"
            "Recent history:\n" + "\n".join(history_lines) + "\n\n"
            "Decision JSON schema:\n" + json.dumps(LLMControllerDecision.model_json_schema())
        )


# ---------------------------------------------------------------------------
# ControllerRuntime -- the LangGraph graph itself
# ---------------------------------------------------------------------------


class ControllerRuntime:
    """Drives one controller episode: a LangGraph ``StateGraph`` over
    ``schemas.ControllerState`` with two nodes (``controller``/``tool``)
    and one conditional edge. Holds the *live* ``ArtifactGraphStore`` and
    its bound ``ToolExecutor`` as instance attributes across node calls
    within one episode (not reconstructed from ``state.checkpoint`` on
    every step) -- a normal step therefore stays on one continuous
    ``branch_id`` for the whole episode; ``state.checkpoint`` is refreshed
    after every tool dispatch purely as a durable snapshot to resume from,
    never as the live source of truth for execution. See
    ``resume_from_checkpoint`` for the one path that *does* create a new
    ``branch_id`` -- an actual interruption/resume, matching Stage 2's own
    rule that a restored, LLM-calling continuation is logged as a separate
    run.
    """

    def __init__(
        self,
        store: ArtifactGraphStore,
        policy: ControllerPolicy,
        checker: Validator,
        producer_registry: ProducerRegistry,
        constraint_provider: ConstraintProvider,
        *,
        document_ref: ArtifactRef,
        budget: Budget,
        max_actions: int = 1000,
        max_steps: int = 50,
        repeat_threshold: int = 3,
        model_name: str = "unknown",
        repair_context: Optional[Dict[str, Any]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> None:
        self._store = store
        self._policy = policy
        self._document_ref = document_ref
        self._max_steps = max_steps
        self._repeat_threshold = repeat_threshold
        self._tool_executor = ToolExecutor(
            store,
            checker,
            budget=budget,
            max_actions=max_actions,
            producer_registry=producer_registry,
            constraint_provider=constraint_provider,
            repair_context=repair_context,
            model_name=model_name,
        )
        self._checkpointer = checkpointer or InMemorySaver()
        self._compiled_graph = self._build_graph().compile(checkpointer=self._checkpointer)

    @classmethod
    def resume_from_checkpoint(
        cls,
        checkpoint: Checkpoint,
        policy: ControllerPolicy,
        checker_factory: Callable[[ArtifactGraphStore], Validator],
        producer_registry: ProducerRegistry,
        constraint_provider: ConstraintProvider,
        *,
        document_ref: ArtifactRef,
        new_branch_id: str,
        budget: Budget,
        **kwargs: Any,
    ) -> "ControllerRuntime":
        """Durable resume: reconstructs the artifact store from ``checkpoint``
        under a brand-new ``new_branch_id`` (``ArtifactGraphStore.
        from_checkpoint`` always requires a fresh one, by Stage 2's own
        design) -- so this survives an actual process restart, not only an
        in-process LangGraph replay, and the resumed continuation is
        recorded as its own run rather than folded into the interrupted
        branch's history. ``checker_factory`` is a callable, not a bare
        ``Validator`` -- the same reason ``branching_runner.BranchingRunner``
        takes one instead of a shared instance: a ``Validator`` like
        ``constraint_checks.BasicConstraintChecker`` binds to one specific
        store at construction time, so the resumed episode needs its own
        fresh instance bound to the *newly restored* store, never the one
        bound to whatever store existed before the interruption.

        ``budget`` is supplied fresh here (not read back out of the
        checkpoint) because ``Checkpoint`` does not carry a ``Budget`` at
        all -- Stage 2's checkpoint is scoped to the artifact graph alone;
        a caller resuming a real episode is expected to pass the
        last-known-remaining budget it tracked independently (e.g. from
        the last ``ControllerState.budget`` it persisted alongside this
        checkpoint)."""
        store = ArtifactGraphStore.from_checkpoint(checkpoint, branch_id=new_branch_id)
        checker = checker_factory(store)
        return cls(
            store,
            policy,
            checker,
            producer_registry,
            constraint_provider,
            document_ref=document_ref,
            budget=budget,
            **kwargs,
        )

    @property
    def store(self) -> ArtifactGraphStore:
        """Exposed read-only so a caller can inspect the live store (e.g.
        ``compute_ready_artifacts``) between runs without this class
        needing its own parallel accessor for every ``ArtifactGraphStore``
        query -- mirrors ``repair_operations.RepairOperationExecutor``'s
        own ``tool_executor`` property for the identical reason."""
        return self._store

    def run(self, *, thread_id: str) -> ControllerState:
        """Runs a fresh episode to completion (FINISH, or a guard trip) and
        returns the final typed state. ``thread_id`` is LangGraph's own key
        for in-process checkpointer replay -- a later ``self._compiled_graph.
        get_state({"configurable": {"thread_id": thread_id}})`` call can
        inspect the full in-process history of this run as long as this
        ``ControllerRuntime`` instance (and its checkpointer) is still
        alive."""
        initial_state = ControllerState(
            document_ref=self._document_ref,
            checkpoint=self._store.save_checkpoint(label="episode-start"),
            budget=self._tool_executor.budget,
        )
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(100, self._max_steps * 4),
        }
        result = self._compiled_graph.invoke(initial_state, config=config)
        return ControllerState.model_validate(result)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ControllerState)
        graph.add_node("controller", self._controller_node)
        graph.add_node("tool", self._tool_node)
        graph.add_edge(START, "controller")
        graph.add_edge("controller", "tool")
        graph.add_conditional_edges("tool", self._route_after_tool, {"controller": "controller", END: END})
        return graph

    @staticmethod
    def _route_after_tool(state: ControllerState) -> str:
        return END if state.done else "controller"

    def _controller_node(self, state: ControllerState) -> Dict[str, Any]:
        """Decides ``pending_tool_call`` for the next ``tool`` node visit.
        Anti-infinite-loop/budget-overrun guards are checked *here*, before
        ``self._policy.decide()`` is ever called -- never inside a
        ``ControllerPolicy`` implementation itself, per this module's own
        docstring. A guard trip produces a synthetic ``FinishCall`` with a
        reason string identifying exactly which guard fired; the ``tool``
        node (not this one) is what actually executes it and marks
        ``done=True`` -- deciding and executing never happen in the same
        node (``schemas.ControllerState``'s own docstring)."""
        if state.done:
            return {}
        if state.step_count >= self._max_steps:
            call: ToolCall = FinishCall(
                reason="max_steps_reached",
                unresolved_artifact_ids=sorted(self._store.stale_artifacts),
            )
            return {"pending_tool_call": call}
        if _repeated_action(state.history, threshold=self._repeat_threshold):
            call = FinishCall(
                reason="repeated_action_detected",
                unresolved_artifact_ids=sorted(self._store.stale_artifacts),
            )
            return {"pending_tool_call": call}
        call = self._policy.decide(state, self._store)
        return {"pending_tool_call": call}

    def _tool_node(self, state: ControllerState) -> Dict[str, Any]:
        """Executes ``state.pending_tool_call`` through this runtime's one
        ``ToolExecutor`` (shared with any ``RepairOperationExecutor`` it
        lazily built -- one store, one budget ledger, the Stage 4
        discipline this module inherits unchanged), and folds the
        observation back into the typed state: fresh CHECK evidence
        replaces old (``_merge_evidence``), a successful PATCH/REPLAN drops
        now-stale evidence for whatever it touched (``_drop_evidence_for``)
        rather than leaving a pre-repair verdict looking current, and a
        successful COMPILE sets ``compiled``. ``done``/``stop_reason`` are
        set here -- never by ``_controller_node`` -- whenever ``call`` is a
        ``FinishCall``, or whenever a dispatch was rejected for a
        budget-related reason (``schemas.BUDGET_ERROR_CODES``) even though
        the controller itself did not choose to stop."""
        call = state.pending_tool_call
        assert call is not None, "_tool_node reached with no pending_tool_call set"

        observation = self._tool_executor.execute(call)
        step = ControllerStepRecord(
            step_index=state.step_count,
            action=call.action,
            call_summary=_summarize_call(call),
            ok=observation.ok,
            error_code=observation.error.code if observation.error is not None else None,
        )
        updates: Dict[str, Any] = {
            "step_count": state.step_count + 1,
            "history": state.history + [step],
            "pending_tool_call": None,
            "checkpoint": self._store.save_checkpoint(label=f"step-{state.step_count}"),
            "budget": self._tool_executor.budget,
        }

        if observation.ok:
            assert observation.output is not None
            if isinstance(call, CheckCall):
                updates["current_evidence"] = _merge_evidence(state.current_evidence, observation.output.evidence)
            elif isinstance(call, (PatchCall, ReplanCall)):
                touched_ids = {record.pre_repair_ref.artifact_id for record in observation.output.result.regenerated}
                touched_ids |= {ref.artifact_id for ref in observation.output.result.reverified_untouched}
                updates["current_evidence"] = _drop_evidence_for(state.current_evidence, touched_ids)
            elif isinstance(call, CompileCall):
                updates["compiled"] = observation.output.compile_success
            elif isinstance(call, FinishCall):
                updates["done"] = True
                updates["stop_reason"] = _FINISH_REASON_MAP.get(call.reason, ControllerStopReason.FINISHED)
        elif observation.error is not None and observation.error.code in BUDGET_ERROR_CODES:
            updates["done"] = True
            updates["stop_reason"] = ControllerStopReason.BUDGET_EXHAUSTED

        return updates


_FINISH_REASON_MAP: Dict[str, ControllerStopReason] = {
    "max_steps_reached": ControllerStopReason.MAX_STEPS_REACHED,
    "repeated_action_detected": ControllerStopReason.REPEATED_ACTION_DETECTED,
}
"""Maps a ``FinishCall.reason`` string back to the ``ControllerStopReason``
enum ``_tool_node`` records -- only the two guard-produced reasons get
their own distinct value; every other reason (a policy's own voluntary
"resolved"/"nothing_left_to_do"/... string) reads as the generic
``FINISHED``, since those are a policy's free-text choice, not a fixed
vocabulary this module tracks on their behalf."""


__all__ = [
    "ControllerPolicy",
    "RuleBasedController",
    "LLMGenerator",
    "LLMController",
    "ControllerRuntime",
    "compute_ready_artifacts",
]

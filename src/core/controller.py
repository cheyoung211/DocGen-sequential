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

**Three scope decisions confirmed with the user before writing the first
pass** (recorded in Notion Progress.ver2, the 2026-10-20 Stage 5 entry):

(a) Mechanism + Protocol-conforming-fake tests only this round -- a real
    small-scale pilot against the local Qwen3.6 pipeline is an explicitly
    deferred follow-up.
(b) Checkpoint/resume goes beyond LangGraph's own in-process checkpointer:
    ``ControllerRuntime.resume_from_checkpoint`` additionally reconstructs
    the artifact store from a durable Stage 2 ``Checkpoint`` under a
    brand-new ``branch_id``.
(c) ``tool_executor.ToolExecutor`` itself (not a bypassing layer in this
    module) dispatches PATCH/REPLAN for real when constructed with
    ``producer_registry``/``constraint_provider``.

**Revised after a user review found 7 real bugs in the first pass**
(full pattern detail in memory ``docgen_stage_review_rigor``):

1. ``_merge_evidence`` keyed only by ``artifact_id`` -- two constraints on
   the same artifact (one FAIL, one PASS) collapsed into whichever
   happened to be last in the list, silently discarding the other. Fixed
   by keying on ``(artifact_id, constraint_id)``. Separately, a cached
   PASS could keep reading as current even after some OTHER artifact it
   depended on (``Evidence.input_versions``) moved -- fixed with a new
   public ``ArtifactGraphStore.is_evidence_still_valid()`` (exposes the
   same per-input check ``live_evidence()`` already applies internally,
   *without* that method's additional "the artifact itself must not be
   stale" gate -- see point 2) and ``_prune_invalidated_evidence()``,
   applied every time evidence is merged.
2. A failed/partial repair's own evidence update dropped whatever old
   verdict existed for the touched artifact(s) without replacing it with
   anything representing the real outcome -- a ``PRODUCTION_FAILED``
   regenerate, or a cascade-discovered downstream failure with no fresh
   Evidence of its own, both simply vanished, letting the controller
   proceed to COMPILE/FINISH as if nothing were wrong. Two distinct root
   causes, both needing their own tracking: (a) ``ArtifactGraphStore.
   live_evidence()`` returns ``[]`` for ANY stale artifact by design (a
   stale artifact definitionally has no trustworthy "current settled
   state" verdict) -- but a budget-cut-short cascade leaves the artifact
   stale forever (no new version is ever written), so Evidence-based
   tracking alone can never represent "this is still broken" for exactly
   that case; (b) a ``PRODUCTION_FAILED``/``NO_PRODUCER`` repair attempt
   against an artifact that was never stale to begin with (it is failing
   its own *content* check, not a dependency that moved) leaves **no**
   trace in ``is_stale()`` at all, not even transiently -- a narrower,
   second gap a first fix for (a) alone did not cover. Fixed with two new
   sets on ``ControllerState``: ``tracked_artifact_ids`` (every artifact
   this episode has ever dispatched a call against, or that a PATCH/
   REPLAN's own cascade touched or discovered broken -- checked against
   ``store.is_stale()`` for case (a)) and ``blocked_artifact_ids``
   (artifacts whose most recent repair attempt could not even produce new
   content -- the only thing that still catches case (b)).
   ``_compute_unresolved()`` unions both. ``RuleBasedController.decide()``
   now checks this before ever considering COMPILE/FINISH.
3. "Resolved" required only "no FAIL evidence", not "evidence is actually
   PASS" -- UNKNOWN-only evidence with ``compiled=True`` still returned
   ``FinishCall(reason="resolved", ...)``. Also, ``compiled`` was never
   invalidated by a later successful repair, so a document that changed
   after its last successful compile could still finish without
   recompiling. Fixed: ``RuleBasedController`` now requires strict PASS
   classification (not merely "not FAIL") before compiling/resolving, and
   ``_tool_node`` resets both ``compiled`` and the new ``last_compile_failed``
   to ``False`` after every successful PATCH/REPLAN dispatch, regardless
   of its outcome.
4. ``LLMController``'s prompt carried only a bare ok/error-code summary of
   each past step, never the real payload an INSPECT/QUERY/CHECK/COMPILE
   actually returned -- so asking the LLM to inspect or query something
   gave it no way to ever see the answer, and a failed compile's real
   diagnostic (engine/return code/fatal error count/first error type)
   never reached the next decision either. ``RuleBasedController`` had
   the same underlying gap, visible as "keeps repeating an identical
   COMPILE until the repeat-guard forces a stop" -- not wrong, but not a
   constructive response either. Fixed with a new ``ControllerStepRecord.
   detail`` field (``_describe_observation()``, included in every
   history entry and thus in the LLM prompt) and, for the rule-based
   policy, the new ``last_compile_failed`` flag (point 3) letting it stop
   cleanly with ``reason="compile_failed"`` instead of blindly retrying.
5. Durable resume reconstructed the artifact *store* from a checkpoint,
   but ``run()`` always built a brand-new ``ControllerState`` (history/
   step_count/current_evidence/compiled all reset) and a brand-new
   ``ToolExecutor`` (action-count/suspension reset) -- a "resume" was
   actually a fresh episode seeded only from the artifact graph, with
   every prior decision and every prior admission limit silently
   discarded. Fixed: ``ControllerState`` gained ``actions_taken``/
   ``suspended``/``suspend_reason`` (mirroring ``ToolExecutor``'s own
   admission state exactly); ``ToolExecutor.__init__`` gained matching
   optional constructor parameters (Stage 3, backward compatible --
   every pre-Stage-5 call site omits them and gets the old fresh-start
   behavior unchanged); ``resume_from_checkpoint`` now takes the full
   prior ``ControllerState`` (not a bare ``Checkpoint``+``Budget``) and
   seeds both the new ``ToolExecutor`` and ``run()``'s initial state from
   it.
6. ``LLMController.decide()`` called ``generator.generate()`` directly,
   with no admission check beforehand and no cost charged afterward --
   the controller's own reasoning cost silently bypassed the budget cap
   entirely (a ``cap=0.0`` budget still let an LLM call through). Fixed
   by routing it through ``ToolExecutor.run_generation()`` via a new
   ``_LLMDecisionProducer`` adapter -- admission is checked, and cost is
   charged, through the exact same ledger every other generation call in
   this codebase already uses, rather than inventing a second one.
7. A raised exception from ``generator.generate()`` itself (timeout,
   connection error) propagated straight out of ``decide()`` uncaught --
   only a JSON-parse failure was ever handled. Fixed inside
   ``_LLMDecisionProducer.produce()``: both failure modes are now caught
   in the same bounded retry loop, cost is always recorded (point 6), and
   the method itself never raises -- ``Producer``'s own contract
   (``interfaces.py``: never raise past this boundary for an expected,
   routine failure).

See ``docs/stage5_langgraph_controller.md`` for the full write-up.
"""

from __future__ import annotations

import json
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

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
    ProductionResult,
    ProductionStatus,
    QueryDependenciesCall,
    ReplanCall,
    RepairAction,
    RepairExecutionResult,
    RepairOutcome,
    ToolCall,
    ToolObservation,
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
    """A fresh verdict for a given ``(artifact_id, constraint_id)`` pair
    always replaces whatever this controller held for that exact pair
    before -- never accumulates side by side with it, and never collapses
    two distinct constraints checked on the SAME artifact into one entry.

    A user review of the first Stage 5 pass found the original version
    keyed only by ``artifact_id``: checking two constraints on the same
    artifact (one FAIL, one PASS) silently kept only whichever happened
    to be last in the returned list, discarding the other regardless of
    its outcome -- which could make a genuinely failing artifact read as
    clean the moment any OTHER constraint on it passed."""
    by_key: Dict[Tuple[str, str], Evidence] = {
        (item.checked_artifact.artifact_id, item.constraint_id): item for item in existing
    }
    for item in fresh:
        by_key[(item.checked_artifact.artifact_id, item.constraint_id)] = item
    return list(by_key.values())


def _prune_invalidated_evidence(store: ArtifactGraphStore, evidence: List[Evidence]) -> List[Evidence]:
    """Drops any entry whose own recorded inputs have since moved or
    appeared (``ArtifactGraphStore.is_evidence_still_valid`` -- the same
    per-input check ``live_evidence()`` applies internally to each
    candidate, exposed publicly for exactly this use) -- independent of
    whether ``checked_artifact`` ITSELF is currently stale, which
    ``live_evidence()``'s own additional gate would otherwise also apply
    (see this module's docstring, point 2, for why that gate is wrong for
    this controller's own tracking purpose).

    A user review of the first Stage 5 pass found the controller's own
    evidence cache could keep serving a PASS that depended on some OTHER
    artifact's now-superseded version -- exactly the drift
    ``live_evidence()`` already prevents at the store layer, but this
    controller-side cache was never synchronized against."""
    return [item for item in evidence if store.is_evidence_still_valid(item)]


def _drop_evidence_for(existing: List[Evidence], artifact_ids: Set[str]) -> List[Evidence]:
    """Drops every entry about any of ``artifact_ids`` -- used right
    before re-merging a repair's own fresh reverify evidence (if any) for
    the same ids, so a pre-repair verdict never masquerades as current
    once that content has actually changed."""
    return [item for item in existing if item.checked_artifact.artifact_id not in artifact_ids]


def _compute_unresolved(store: ArtifactGraphStore, tracked_ids: Set[str], blocked_ids: Set[str]) -> Set[str]:
    """Which of ``tracked_ids`` still have unresolved work right now:
    either flagged stale by the store (a dependency moved and nothing has
    re-settled it yet -- covers a budget-cut-short cascade's own
    ``left_stale``, which is itself defined as "whatever is still
    ``is_stale()``-true" in ``repair_operations.py``), OR in
    ``blocked_ids`` (this module's docstring, point 2): an artifact whose
    most recent repair attempt could not even produce new content
    (``RepairOutcome.PRODUCTION_FAILED``/``NO_PRODUCER``). The two sets
    are not redundant -- an artifact failing its own *content* check (not
    a dependency that moved) is never marked stale in the first place, so
    a production failure against it leaves no trace in ``is_stale()`` at
    all; ``blocked_ids`` is the only thing that still catches it."""
    return {aid for aid in tracked_ids if store.is_stale(aid)} | (blocked_ids & tracked_ids)


def _update_blocked(existing: Set[str], result: RepairExecutionResult) -> Set[str]:
    """Tracks artifacts whose most recent repair attempt could not even
    produce new content -- see ``_compute_unresolved``'s docstring for
    why this is not redundant with ``is_stale()``. Cleared the moment a
    LATER attempt actually produces something, regardless of whether
    that content then passes, fails, or is unknown (those cases DO get a
    fresh Evidence entry for the controller to react to on their own)."""
    updated = set(existing)
    for artifact_id, outcome in result.final_outcome_by_artifact.items():
        if outcome in (RepairOutcome.PRODUCTION_FAILED, RepairOutcome.NO_PRODUCER):
            updated.add(artifact_id)
        else:
            updated.discard(artifact_id)
    return updated


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
    different version, never do. Deliberately a short identity key only --
    the full observed payload is a separate field, see
    ``_describe_observation``."""
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


def _describe_observation(call: ToolCall, observation: ToolObservation) -> str:
    """The full human-readable detail of one dispatch -- what a user
    review found missing from the first Stage 5 pass (this module's
    docstring, point 4): only a bare ok/error-code summary ever reached
    the next decision, so an INSPECT/QUERY/COMPILE's real payload never
    flowed anywhere past the one step that produced it. Stored on
    ``ControllerStepRecord.detail``, which both controller variants'
    prompts/decisions can read from the recent history."""
    if not observation.ok:
        assert observation.error is not None
        return f"{call.action.value} rejected: {observation.error.code.value} -- {observation.error.message}"
    output = observation.output
    assert output is not None
    if isinstance(call, InspectArtifactCall):
        if not output.found or output.stored is None:
            return f"inspect {call.artifact_id}: not found"
        preview = (output.stored.content or "")[:200]
        return (
            f"inspect {call.artifact_id}@{output.stored.ref.version}: "
            f"status={output.stored.status.value}, produced_by={output.stored.produced_by!r}, "
            f"content preview={preview!r}"
        )
    if isinstance(call, QueryDependenciesCall):
        edges_desc = (
            ", ".join(
                f"{edge.from_artifact.artifact_id}@{edge.from_artifact.version}->"
                f"{edge.to_artifact.artifact_id}@{edge.to_artifact.version}({edge.relation.value})"
                for edge in output.edges
            )
            or "(none)"
        )
        return f"query {call.artifact_id}: impact_candidates={sorted(output.impact_candidates)}, edges=[{edges_desc}]"
    if isinstance(call, CheckCall):
        verdicts = (
            "; ".join(f"{item.constraint_id}={item.outcome.value}({item.detail})" for item in output.evidence)
            or "(no evidence)"
        )
        return f"check {call.artifact.artifact_id}@{call.artifact.version}: {verdicts}"
    if isinstance(call, CompileCall):
        result = output.compile_result
        extra = ""
        if result is not None:
            extra = (
                f" engine={result.engine} return_code={result.return_code} "
                f"fatal_errors={result.fatal_error_count} warnings={result.warning_count} "
                f"first_error_type={result.first_error_type} pdf_exists={result.pdf_exists}"
            )
        return f"compile {call.request_id}: success={output.compile_success}{extra}"
    if isinstance(call, (PatchCall, ReplanCall)):
        result = output.result
        per_artifact = (
            ", ".join(
                f"{artifact_id}={outcome.value}" for artifact_id, outcome in sorted(result.final_outcome_by_artifact.items())
            )
            or "(none)"
        )
        left_stale = ", ".join(sorted(ref.artifact_id for ref in result.left_stale)) or "(none)"
        return (
            f"{call.action.value}: overall={result.overall_outcome.value}, per_artifact=[{per_artifact}], "
            f"left_stale=[{left_stale}], budget_exhausted={result.budget_exhausted}"
        )
    if isinstance(call, FinishCall):
        return f"finish: reason={call.reason!r}, unresolved={call.unresolved_artifact_ids}"
    return _summarize_call(call)  # pragma: no cover - exhaustive over today's ToolCall union


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
    ``interfaces.RepairPolicy``'s side-effect-free discipline one layer up
    for ``store``/``state`` -- must not mutate either, and may be called
    without the returned ``ToolCall`` necessarily ever being executed.

    ``tool_executor`` is handed through so a policy whose OWN
    decision-making has a real cost (``LLMController``, which must call an
    LLM to decide) can route that call through the executor's own
    admission/charge pipeline -- the one deliberate, narrow exception to
    "no side effects" this Protocol allows, safe only because
    ``ControllerRuntime`` calls ``decide()`` exactly once per real step and
    always acts on the result (never speculatively, unlike how a
    comparison harness might call ``RepairPolicy.decide()``). A policy
    with no such cost of its own (``RuleBasedController``) simply never
    touches this argument. Anti-loop/budget-overrun guards remain entirely
    outside this Protocol's responsibility -- ``ControllerRuntime.
    _controller_node`` checks those *before* ever calling ``decide()`` and
    short-circuits with a synthetic ``FinishCall`` when one trips."""

    def decide(self, state: ControllerState, store: ArtifactGraphStore, tool_executor: ToolExecutor) -> ToolCall: ...


class RuleBasedController:
    """Deterministic policy reading remaining budget, the latest CHECK
    evidence, and (point 2 of this module's docstring) the store's own
    staleness on every artifact this episode has touched:

    1. Nothing checked and nothing unresolved yet -> CHECK the document.
    2. A FAIL present -> delegate scope selection to
       ``repair_operations.FixedGraphAwareRepairPolicy`` (Stage 4,
       unmodified) and execute the *first* (lowest target artifact_id)
       action it names this step -- any other independent failure that
       same ``decide()`` call also named is picked up on a later step
       once a fresh CHECK re-surfaces it, since this controller executes
       exactly one action per step (unlike ``branching_runner.
       BranchingRunner``, which executes every action a round's
       ``decide()`` returns within one round -- a deliberate difference:
       that class drives one *round* of several branches, this one drives
       one *step* of a single sequential loop).
    3. Something is ``is_stale()``-unresolved but has no fresh Evidence of
       its own yet -> CHECK it (try to get a real verdict before giving
       up on it).
    4. Something is still ``is_stale()``-unresolved even after being
       checked (a production failure, a missing producer, or anything a
       budget cutoff left mid-cascade) -> FINISH honestly reporting it,
       rather than silently proceeding.
    5. Evidence is UNKNOWN (not FAIL, not clean PASS) -> FINISH reporting
       it as inconclusive, never silently treated as resolved.
    6. Clean PASS, nothing unresolved, not yet compiled -> COMPILE, unless
       the previous COMPILE already failed (``last_compile_failed``), in
       which case FINISH -- this policy has no further action it knows
       how to try against a compiler diagnostic it cannot interpret, so it
       stops cleanly instead of blindly repeating an identical COMPILE
       until the repeated-action guard eventually intervenes.
    7. Compiled, clean, nothing unresolved -> FINISH(resolved).
    """

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

    def decide(self, state: ControllerState, store: ArtifactGraphStore, tool_executor: ToolExecutor) -> ToolCall:
        current = store.latest(self._document_ref.artifact_id)
        live_document_ref = current.ref if current is not None else self._document_ref

        unresolved = _compute_unresolved(store, state.tracked_artifact_ids, state.blocked_artifact_ids)
        checked_ids = {item.checked_artifact.artifact_id for item in state.current_evidence}

        if not state.current_evidence and not unresolved:
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
                    unresolved_artifact_ids=sorted({item.checked_artifact.artifact_id for item in failing}),
                )
            chosen = sorted(actions, key=lambda action: action.target_artifacts[0].artifact_id)[0]
            return _repair_action_to_tool_call(chosen)

        not_yet_checked = sorted(unresolved - checked_ids)
        if not_yet_checked:
            target = store.latest(not_yet_checked[0])
            assert target is not None  # staleness only ever applies to an artifact that exists
            return CheckCall(artifact=target.ref, constraints=self._constraints_for(target.ref), context={})

        if unresolved:
            return FinishCall(reason="unresolved_remaining", unresolved_artifact_ids=sorted(unresolved))

        if classification == "UNKNOWN":
            unknown_ids = sorted(
                {item.checked_artifact.artifact_id for item in state.current_evidence if item.outcome == CheckOutcome.UNKNOWN}
            )
            return FinishCall(reason="evidence_inconclusive", unresolved_artifact_ids=unknown_ids)

        if not state.compiled:
            if state.last_compile_failed:
                return FinishCall(reason="compile_failed", unresolved_artifact_ids=[live_document_ref.artifact_id])
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
    reports as unavailable for the local/vLLM provider), accepts
    ``usage_sink`` as a keyword argument already, and ``extract_json_block``
    is an existing method ``TextAgent``/``PlannerAgent`` already call the
    same way."""

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str: ...

    def extract_json_block(self, text: str) -> str: ...


class _LLMDecisionProducer:
    """Adapts one "ask the LLM for the next decision" call into the
    ``Producer`` Protocol, purely so ``LLMController.decide()`` can route
    it through ``ToolExecutor.run_generation()`` -- the same admission-
    then-charge pipeline every other generation call in this codebase
    already uses (Stage 3), rather than inventing a second, parallel one
    just for the controller's own reasoning cost (this module's
    docstring, points 6-7).

    Never raises: catches both a parse failure AND an exception from
    ``generator.generate()`` itself (e.g. a timeout/connection error)
    inside its own bounded retry loop, always returning a
    ``ProductionResult`` either way -- ``Producer``'s own contract
    (``interfaces.py``: never raise past this boundary for an expected,
    routine failure a controller needs to see and act on, not catch as an
    exception).

    The artifact_id/kind on a successful ``ProductionResult`` are never
    persisted anywhere (``LLMController`` only reads ``.content``/
    ``.status``/``.error``) -- this is a deliberate, narrow reuse of the
    Producer/``run_generation`` machinery for something that is not really
    a document artifact."""

    def __init__(self, generator: LLMGenerator, system_prompt: str, user_prompt: str, max_attempts: int) -> None:
        self._generator = generator
        self._system_prompt = system_prompt
        self._user_prompt = user_prompt
        self._max_attempts = max(1, max_attempts)

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ProductionResult:
        usage_sink = context.get("usage_sink")
        last_error = "no attempts made"
        for _ in range(self._max_attempts):
            try:
                raw = self._generator.generate(self._system_prompt, self._user_prompt, usage_sink=usage_sink)
                payload = self._generator.extract_json_block(raw)
                LLMControllerDecision.model_validate_json(payload)  # validate before committing to it
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: a flaky generate() call
                # and a malformed/unparseable response are both routine, retry-worthy failures
                # here, never something this adapter lets propagate past its own boundary.
                last_error = str(exc)
                continue
            return ProductionResult(
                status=ProductionStatus.SUCCESS,
                artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.BLOCK),
                content=payload,
            )
        return ProductionResult(
            status=ProductionStatus.FAILED,
            error=f"LLM controller decision failed after {self._max_attempts} attempt(s): {last_error}",
        )


class LLMController:
    """Asks an LLM to pick the next action directly, reading a compact text
    summary of remaining budget, current evidence, unresolved artifacts,
    and recent history (now including each step's full observed detail --
    this module's docstring, point 4). Falls back to a safe ``FinishCall``
    -- never raises past this boundary, never leaves the controller
    without a decision -- if the model's output fails to parse, or the
    call itself raises, after ``max_attempts`` tries: a controller must
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
        model_name: str = "unknown",
    ) -> None:
        self._document_ref = document_ref
        self._generator = generator
        self._request_id = request_id
        self._output_dir = output_dir
        self._max_attempts = max_attempts
        self._model_name = model_name

    def decide(self, state: ControllerState, store: ArtifactGraphStore, tool_executor: ToolExecutor) -> ToolCall:
        """The one deliberate exception to ``ControllerPolicy``'s
        side-effect-free rule (see that Protocol's own docstring): asking
        an LLM to decide is itself a real, billable call, so it is routed
        through ``tool_executor.run_generation()`` -- checked for
        admission *before* the generator is ever touched (a ``cap=0.0``
        budget never reaches the model), and charged afterward through
        the exact same ledger every other generation call in this episode
        uses."""
        adapter = _LLMDecisionProducer(
            self._generator, self._system_prompt(), self._user_prompt(state, store), self._max_attempts
        )
        observation = tool_executor.run_generation(
            adapter, "controller_decision", {}, model_name=self._model_name, stage="llm_controller_decision"
        )
        if not observation.ok:
            assert observation.error is not None
            return FinishCall(
                reason=f"llm_controller_call_rejected_{observation.error.code.value}",
                unresolved_artifact_ids=[self._document_ref.artifact_id],
            )
        assert observation.output is not None
        production = observation.output.production
        if production.status == ProductionStatus.FAILED or production.content is None:
            return FinishCall(
                reason="llm_controller_exhausted_retries",
                unresolved_artifact_ids=[self._document_ref.artifact_id],
            )
        decision = LLMControllerDecision.model_validate_json(production.content)
        return decision.call

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

    def _user_prompt(self, state: ControllerState, store: ArtifactGraphStore) -> str:
        unresolved = _compute_unresolved(store, state.tracked_artifact_ids, state.blocked_artifact_ids)
        evidence_lines = [
            f"- {item.checked_artifact.artifact_id}@{item.checked_artifact.version}: "
            f"{item.outcome.value} ({item.constraint_id}) {item.detail}".strip()
            for item in state.current_evidence
        ] or ["(no evidence yet)"]
        history_lines = [
            f"- step {record.step_index}: {record.call_summary} -> "
            + ("ok" if record.ok else f"error={record.error_code.value if record.error_code else '?'}")
            + (f" | observed: {record.detail}" if record.detail else "")
            for record in state.history[-5:]
        ] or ["(no steps yet)"]
        return (
            f"Document artifact: {self._document_ref.artifact_id}\n"
            f"Budget remaining: {state.budget.remaining} / cap {state.budget.cap}\n"
            f"Compiled this episode: {state.compiled} (last compile failed: {state.last_compile_failed})\n"
            f"Tracked artifacts with unresolved work but no fresh pass/fail evidence of their own "
            f"(e.g. a production failure, a missing producer, or left stale by a budget cutoff): "
            f"{sorted(unresolved)}\n"
            "Current evidence (pass/fail/unknown per constraint, on currently non-stale artifacts):\n"
            + "\n".join(evidence_lines)
            + "\n"
            "Recent history (with the full observed detail of each step):\n" + "\n".join(history_lines) + "\n\n"
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
        actions_taken: int = 0,
        suspended: bool = False,
        suspend_reason: Optional[str] = None,
        resume_state: Optional[ControllerState] = None,
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
            actions_taken=actions_taken,
            suspended=suspended,
            suspend_reason=suspend_reason,
        )
        self._checkpointer = checkpointer or InMemorySaver()
        self._compiled_graph = self._build_graph().compile(checkpointer=self._checkpointer)
        self._resume_state = resume_state

    @classmethod
    def resume_from_checkpoint(
        cls,
        prior_state: ControllerState,
        policy: ControllerPolicy,
        checker_factory: Callable[[ArtifactGraphStore], Validator],
        producer_registry: ProducerRegistry,
        constraint_provider: ConstraintProvider,
        *,
        new_branch_id: str,
        **kwargs: Any,
    ) -> "ControllerRuntime":
        """Durable resume: reconstructs the artifact store from
        ``prior_state.checkpoint`` under a brand-new ``new_branch_id``
        (``ArtifactGraphStore.from_checkpoint`` always requires a fresh
        one, by Stage 2's own design), AND continues the full controller
        execution state (``step_count``/``history``/``current_evidence``/
        ``tracked_artifact_ids``/``blocked_artifact_ids``/``compiled``/
        ``last_compile_failed``/``actions_taken``/``suspended``/
        ``suspend_reason``) from
        ``prior_state`` -- a user review of the first Stage 5 pass found
        the original implementation reconstructed only the artifact store
        and otherwise always started a brand-new ``ControllerState`` and a
        brand-new ``ToolExecutor``, silently discarding every prior
        decision and reopening the full action-count/budget allowance.
        That was not a resume, it was a new episode seeded only from the
        artifact graph.

        ``checker_factory`` is a callable, not a bare ``Validator`` -- the
        same reason ``branching_runner.BranchingRunner`` takes one instead
        of a shared instance: a ``Validator`` like ``constraint_checks.
        BasicConstraintChecker`` binds to one specific store at
        construction time, so the resumed episode needs its own fresh
        instance bound to the *newly restored* store, never the one bound
        to whatever store existed before the interruption."""
        store = ArtifactGraphStore.from_checkpoint(prior_state.checkpoint, branch_id=new_branch_id)
        checker = checker_factory(store)
        return cls(
            store,
            policy,
            checker,
            producer_registry,
            constraint_provider,
            document_ref=prior_state.document_ref,
            budget=prior_state.budget,
            actions_taken=prior_state.actions_taken,
            suspended=prior_state.suspended,
            suspend_reason=prior_state.suspend_reason,
            resume_state=prior_state,
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
        """Runs an episode to completion (FINISH, or a guard trip) and
        returns the final typed state. If this instance was built via
        ``resume_from_checkpoint``, the run continues from that prior
        state (history/step_count/evidence/admission-state included,
        only ``checkpoint`` is refreshed to this run's own starting
        snapshot) rather than starting fresh. ``thread_id`` is LangGraph's
        own key for in-process checkpointer replay -- a later
        ``self._compiled_graph.get_state({"configurable": {"thread_id":
        thread_id}})`` call can inspect the full in-process history of
        this run as long as this ``ControllerRuntime`` instance (and its
        checkpointer) is still alive."""
        if self._resume_state is not None:
            initial_state = self._resume_state.model_copy(
                update={"checkpoint": self._store.save_checkpoint(label="resume-start")}
            )
        else:
            initial_state = ControllerState(
                document_ref=self._document_ref,
                checkpoint=self._store.save_checkpoint(label="episode-start"),
                budget=self._tool_executor.budget,
                tracked_artifact_ids={self._document_ref.artifact_id},
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
        call = self._policy.decide(state, self._store, self._tool_executor)
        return {"pending_tool_call": call}

    def _tool_node(self, state: ControllerState) -> Dict[str, Any]:
        """Executes ``state.pending_tool_call`` through this runtime's one
        ``ToolExecutor`` (shared with any ``RepairOperationExecutor`` it
        lazily built -- one store, one budget ledger, the Stage 4
        discipline this module inherits unchanged), and folds the full
        observation back into the typed state:

        - CHECK/INSPECT/QUERY grow ``tracked_artifact_ids`` to include
          whatever artifact they named; CHECK additionally merges its
          fresh evidence per ``(artifact_id, constraint_id)``
          (``_merge_evidence``) and prunes anything now invalidated by a
          changed input (``_prune_invalidated_evidence``).
        - PATCH/REPLAN grow ``tracked_artifact_ids`` to include every
          artifact the repair's own cascade touched or discovered broken
          (``RepairExecutionResult.final_outcome_by_artifact``/
          ``.left_stale``), drop stale evidence for all of them, then
          re-merge whatever fresh reverify evidence the repair itself
          produced -- an artifact left ``is_stale()``-true with no fresh
          evidence of its own (a production failure, a missing producer,
          a budget-cut-short cascade) is still tracked and will surface
          through ``_compute_unresolved()`` even with no Evidence to show
          for it. Any successful dispatch also invalidates ``compiled``/
          ``last_compile_failed`` (point 3 of this module's docstring),
          and an exhausted cascade (``result.budget_exhausted``) ends the
          episode immediately rather than waiting for the next call to be
          rejected.
        - COMPILE sets ``compiled``/``last_compile_failed`` from the real
          ``compile_success`` value.
        - Every dispatch records ``_describe_observation()``'s full detail
          on the new ``ControllerStepRecord`` (point 4), and refreshes
          ``actions_taken``/``suspended``/``suspend_reason`` from this
          runtime's own ``ToolExecutor`` (point 5 -- so a later durable
          resume from this exact state restores the same admission state,
          not a fresh one).

        ``done``/``stop_reason`` are set here -- never by
        ``_controller_node`` -- whenever ``call`` is a ``FinishCall``, or
        whenever a dispatch was rejected for a budget-related reason
        (``schemas.BUDGET_ERROR_CODES``) even though the controller itself
        did not choose to stop."""
        call = state.pending_tool_call
        assert call is not None, "_tool_node reached with no pending_tool_call set"

        observation = self._tool_executor.execute(call)
        detail = _describe_observation(call, observation)
        step = ControllerStepRecord(
            step_index=state.step_count,
            action=call.action,
            call_summary=_summarize_call(call),
            ok=observation.ok,
            error_code=observation.error.code if observation.error is not None else None,
            detail=detail,
        )
        updates: Dict[str, Any] = {
            "step_count": state.step_count + 1,
            "history": state.history + [step],
            "pending_tool_call": None,
            "checkpoint": self._store.save_checkpoint(label=f"step-{state.step_count}"),
            "budget": self._tool_executor.budget,
            "actions_taken": self._tool_executor.actions_taken,
            "suspended": self._tool_executor.is_suspended,
            "suspend_reason": self._tool_executor.suspend_reason,
        }

        if observation.ok:
            assert observation.output is not None
            if isinstance(call, CheckCall):
                merged = _merge_evidence(state.current_evidence, observation.output.evidence)
                updates["current_evidence"] = _prune_invalidated_evidence(self._store, merged)
                updates["tracked_artifact_ids"] = state.tracked_artifact_ids | {call.artifact.artifact_id}
            elif isinstance(call, InspectArtifactCall):
                updates["tracked_artifact_ids"] = state.tracked_artifact_ids | {call.artifact_id}
            elif isinstance(call, QueryDependenciesCall):
                updates["tracked_artifact_ids"] = state.tracked_artifact_ids | {call.artifact_id}
            elif isinstance(call, (PatchCall, ReplanCall)):
                result = observation.output.result
                touched_ids = set(result.final_outcome_by_artifact.keys())
                touched_ids |= {ref.artifact_id for ref in result.left_stale}
                pruned = _drop_evidence_for(state.current_evidence, touched_ids)
                for record in result.regenerated:
                    if record.reverify_evidence:
                        pruned = _merge_evidence(pruned, record.reverify_evidence)
                updates["current_evidence"] = _prune_invalidated_evidence(self._store, pruned)
                updates["tracked_artifact_ids"] = state.tracked_artifact_ids | touched_ids
                updates["blocked_artifact_ids"] = _update_blocked(state.blocked_artifact_ids, result)
                updates["compiled"] = False
                updates["last_compile_failed"] = False
                if result.budget_exhausted:
                    updates["done"] = True
                    updates["stop_reason"] = ControllerStopReason.BUDGET_EXHAUSTED
            elif isinstance(call, CompileCall):
                updates["compiled"] = observation.output.compile_success
                updates["last_compile_failed"] = not observation.output.compile_success
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
"resolved"/"unresolved_remaining"/"compile_failed"/"evidence_inconclusive"/
... string) reads as the generic ``FINISHED``, since those are a policy's
free-text choice, not a fixed vocabulary this module tracks on their
behalf -- the full reason string is still visible on the last
``ControllerStepRecord``/``FinishCall`` either way."""


__all__ = [
    "ControllerPolicy",
    "RuleBasedController",
    "LLMGenerator",
    "LLMController",
    "ControllerRuntime",
    "compute_ready_artifacts",
]

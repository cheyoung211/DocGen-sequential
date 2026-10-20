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

**Revised a second time after a user review found 5 more real bugs**
(full pattern detail in memory ``docgen_stage_review_rigor``):

8. A later, truer PASS could still lose to an earlier, superseded FAIL.
   Repro: ``a -> b``, a cascade processes targets ``[b, a]`` in that
   order -- ``b`` is regenerated first and still fails (``a`` is not
   fixed yet), recorded as ``STILL_FAILING`` against ``b@2``; only once
   ``a`` is fixed afterward does the cascade's own sweep re-verify ``b``
   again, this time confirming it clean with no further content change
   (``_reaffirm``, which still writes a new version, ``b@3``, with the
   fresh PASS rebound to it). ``final_outcome_by_artifact["b"]`` correctly
   reads ``RESOLVED`` -- but ``_tool_node`` only ever scanned
   ``regenerated[*].reverify_evidence`` (the STILL_FAILING record from the
   first sweep), never ``reverified_untouched`` (bare ``ArtifactRef``s
   with no evidence attached at all) or the store's own log, so it kept
   re-merging the stale FAIL and the controller chose ``patch:b@2`` again
   on the very next step despite the store itself agreeing nothing was
   still wrong. Two independent fixes, because this had two independent
   causes: (a) ``_tool_node``'s PATCH/REPLAN branch now re-pulls
   ``ArtifactGraphStore.live_evidence()`` fresh for every touched id
   instead of scanning ``regenerated``/``reverified_untouched`` at all --
   exactly what ``RepairExecutionResult``'s own docstring already warns
   against, for the identical reason; (b) ``ArtifactGraphStore.
   _evidence_is_live`` (and therefore ``is_evidence_still_valid``) now
   also rejects any entry whose ``checked_artifact`` is no longer its own
   artifact_id's current version -- a verdict about a superseded version
   is never live regardless of what its own inputs did, which a cached
   evidence entry spanning a version bump (as opposed to
   ``live_evidence()``'s own call site, which only ever hands this check
   entries already pre-filtered to the current version) could not
   otherwise ever detect.
9. An invalidated constraint could be deleted and never rechecked. Repro:
   one document with constraint A=FAIL and constraint B=PASS; A's (not
   B's) input changes through a relation recorded only on ``Evidence.
   input_versions`` -- with no matching graph edge, so the document
   itself never becomes ``is_stale()``. The next merge correctly dropped
   exactly A's now-invalid FAIL (``is_evidence_still_valid`` turned False
   for it) and kept B's still-valid PASS -- but nothing recorded that A
   was now *owed* a recheck, so ``_classify()`` read the one remaining
   entry (B, PASS) as a clean "PASS" and proceeded straight to COMPILE/
   FINISH without ever re-examining A. Fixed with a new ``ControllerState.
   needs_recheck_keys`` (``(artifact_id, constraint_id)`` pairs -- see
   point 13 below for why per-constraint, not per-artifact, turned out to
   matter): any pair losing its entry to invalidation (``_invalidated_keys``)
   joins this set in both the CHECK and PATCH/REPLAN branches of
   ``_tool_node``, unioned into ``_compute_unresolved()`` so
   ``RuleBasedController.decide()`` treats that artifact_id as still
   needing a fresh CHECK regardless of what other, still-valid evidence
   happens to survive for the same artifact_id -- cleared only once a
   fresh CHECK actually produces new, currently-valid evidence for that
   *exact* pair again (originally described here as "a fresh CHECK
   re-examines every constraint, so clearing by artifact_id is enough" --
   point 13 below found that claim false in general and narrowed the key
   accordingly).
10. Resume landing in the window right after an ``LLMController`` decision
    was made (and paid for) but before it was executed lost both the
    decision and its own cost. ``_controller_node`` previously returned
    only ``pending_tool_call`` -- by the time ``self._policy.decide()``
    returns, ``LLMController`` may already have called ``tool_executor.
    run_generation()`` (points 6-7 above) and mutated that executor's real
    ledger, but nothing synced that ledger into the typed state this node
    returns. A real repro showed the executor having actually charged 7
    tokens/1 action while the serialized ``ControllerState`` immediately
    after this node still read 0/0; resuming from exactly that
    checkpoint silently reopened budget already spent, AND ``run()``
    always re-entered at ``controller`` regardless, which re-decided from
    scratch and overwrote the still-pending, already-paid-for decision
    with a different one. Fixed two ways: ``_controller_node`` now folds
    ``budget``/``actions_taken``/``suspended``/``suspend_reason``/a fresh
    ``checkpoint`` into every return path (mirroring what ``_tool_node``
    already did), and ``_build_graph`` now routes ``START`` through a new
    conditional edge (``_route_at_start``) that goes straight to ``tool``
    whenever the state it is given already carries a ``pending_tool_call``
    -- executing the one decision that already happened instead of asking
    the policy again.
11. A guard-forced termination (``max_steps_reached``/
    ``repeated_action_detected``) could report an empty
    ``unresolved_artifact_ids`` while real problems existed.
    ``_controller_node``'s own guard trips reported only
    ``sorted(self._store.stale_artifacts)`` -- which a FAIL on an
    artifact's own *content* (never a dependency issue, so never
    ``is_stale()``-true -- the same root cause as point 2 above) or a
    ``blocked_artifact_ids`` entry both leave completely silent. A real
    repro: ``doc`` was FAILing its own content check and sitting in
    ``blocked_artifact_ids``, yet ``max_steps_reached`` still reported
    ``unresolved_artifact_ids=[]``. Fixed with one new ``_aggregate_unresolved()``
    -- structurally unresolved ids (``_compute_unresolved``, now also
    covering point 9's ``needs_recheck_keys``) plus every id with current
    FAIL/UNKNOWN evidence plus the document itself if the last compile
    failed. This fix's own claim at the time -- "used by EVERY
    ``FinishCall`` this module ever constructs" -- turned out to still be
    false in one case; see point 14 below for the fix that actually made
    it true.
12. ``LLMController`` never told the model, or enforced on its own output,
    the one correct ``request_id``/``output_dir`` for a ``compile`` it
    might choose -- both were known to the constructor the whole time but
    never reached the prompt or the dispatched ``CompileCall``, so a model
    deciding to compile ``doc`` under request ``req1`` had no way to know
    that was the right pair, and nothing corrected it if it guessed wrong.
    Fixed both ways: the prompt now states the exact values that will be
    used, and ``decide()`` unconditionally overwrites
    ``request_id``/``output_dir`` on any ``CompileCall`` the model returns
    -- the model can still reason about *whether*/*when* to compile, but
    never decides *which* request/output path that compile actually
    targets.

**Revised a third time after a user review found 2 more real bugs**
(full pattern detail in memory ``docgen_stage_review_rigor``):

13. Checking only SOME of an artifact's constraints cleared ALL of that
    artifact's pending recheck obligations. ``needs_recheck_keys``
    (point 9) was keyed by artifact_id alone when it was first added, and
    ``_tool_node``'s CHECK branch removed ``call.artifact.artifact_id``
    from it unconditionally -- but ``CheckCall.constraints`` is never
    guaranteed to be the FULL set a ``ConstraintProvider`` would return
    for that artifact; nothing stops a caller (``LLMController`` in
    particular, whose ``CheckCall`` comes straight from the model's own
    structured output) from naming only some of them. Repro: ``doc`` has
    constraint A (invalidated, owed a recheck) and B; a CHECK naming only
    B comes back PASS -- the artifact-keyed version cleared A's
    obligation too, even though A was never re-examined, letting a
    now-stale FAIL on A go completely unrepresented the next time
    ``_classify()`` ran. Fixed by re-keying ``needs_recheck_keys`` to
    ``(artifact_id, constraint_id)`` pairs (``_invalidated_keys``,
    matching ``_merge_evidence``'s own per-constraint key from point 1) --
    both the CHECK and PATCH/REPLAN branches of ``_tool_node`` now clear
    only the exact pairs that a fresh, currently-valid piece of evidence
    was actually produced for this step, never every pair sharing that
    artifact_id.
14. Point 11's ``_aggregate_unresolved()`` fix was only ever called from
    the two guard branches and from ``LLMController``'s own *failure*
    fallbacks (admission-rejected, retries-exhausted) -- a model that
    successfully parsed a ``FinishCall`` of its OWN choosing (e.g.
    ``reason="cannot_repair"``) had that call returned completely
    unmodified, bypassing the aggregate entirely. Repro: ``doc`` FAILing
    and sitting in ``blocked_artifact_ids``, with the model returning
    ``FinishCall(reason="cannot_repair", unresolved_artifact_ids=[])`` --
    recorded verbatim, with the empty list the model happened to supply.
    Fixed by moving the enforcement out of every individual call site and
    into exactly one place: ``_controller_node`` now unions
    ``_aggregate_unresolved()`` into ANY ``FinishCall``'s
    ``unresolved_artifact_ids`` right after ``call`` is decided --
    whichever of the three paths (a guard, ``RuleBasedController``, or
    ``LLMController``) produced it -- rather than trusting each path (and
    every future ``ControllerPolicy`` written later) to remember to call
    it correctly on its own. The two guards now pass an empty list and
    rely entirely on this one union.

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


def _compute_unresolved(
    store: ArtifactGraphStore,
    tracked_ids: Set[str],
    blocked_ids: Set[str],
    needs_recheck_keys: Set[Tuple[str, str]],
) -> Set[str]:
    """Which of ``tracked_ids`` still have unresolved work right now:
    flagged stale by the store (a dependency moved and nothing has
    re-settled it yet -- covers a budget-cut-short cascade's own
    ``left_stale``, which is itself defined as "whatever is still
    ``is_stale()``-true" in ``repair_operations.py``), OR in
    ``blocked_ids`` (this module's docstring, point 2): an artifact whose
    most recent repair attempt could not even produce new content
    (``RepairOutcome.PRODUCTION_FAILED``/``NO_PRODUCER``), OR named by at
    least one pair in ``needs_recheck_keys`` (point 2b below): an artifact
    with at least one ``constraint_id`` whose cached evidence was just
    invalidated and has not been freshly re-checked since. None of the
    three is redundant with either other: an artifact failing its own
    *content* check (not a dependency that moved) is never marked stale in
    the first place, so a production failure against it leaves no trace in
    ``is_stale()`` at all (``blocked_ids`` is the only thing that still
    catches it); and an invalidated verdict whose dependency was recorded
    only on ``Evidence.input_versions``, with no matching graph edge, never
    marks the checked artifact stale either (``needs_recheck_keys`` is the
    only thing that still catches that one)."""
    return (
        {aid for aid in tracked_ids if store.is_stale(aid)}
        | (blocked_ids & tracked_ids)
        | {aid for aid, _constraint_id in needs_recheck_keys if aid in tracked_ids}
    )


def _invalidated_keys(store: ArtifactGraphStore, evidence: List[Evidence]) -> Set[Tuple[str, str]]:
    """Which ``(artifact_id, constraint_id)`` pairs have an entry in
    ``evidence`` that ``store.is_evidence_still_valid`` now rejects -- the
    exact pairs ``_prune_invalidated_evidence`` is about to drop something
    for. Used to feed ``ControllerState.needs_recheck_keys`` (point 2b,
    this module's docstring): dropping an invalidated verdict must never
    be silent -- that *specific constraint* is now owed a fresh CHECK, not
    simply forgotten. Keyed per-constraint, not per-artifact, for the same
    reason ``_merge_evidence`` is (point 1): one artifact can have several
    independent constraints, and only some of them may actually be
    invalidated."""
    return {
        (item.checked_artifact.artifact_id, item.constraint_id)
        for item in evidence
        if not store.is_evidence_still_valid(item)
    }


def _aggregate_unresolved(state: ControllerState, store: ArtifactGraphStore) -> Set[str]:
    """The complete "what is still wrong, right now" picture: every
    structurally unresolved id (``_compute_unresolved``) plus every
    artifact_id ``state.current_evidence`` currently reports FAIL or
    UNKNOWN for, plus the document itself if the last COMPILE failed.
    Added after a second user review found every termination path
    (``ControllerRuntime._controller_node``'s own guard trips, in
    particular) reporting its ``FinishCall.unresolved_artifact_ids`` from
    a DIFFERENT, each ad hoc and each incomplete, subset of this same
    picture -- a real repro: a guard tripped while ``doc`` was FAILing its
    own content check AND sitting in ``blocked_artifact_ids``, with
    neither fact reflected because the guard's own ``FinishCall`` only
    ever looked at ``store.stale_artifacts`` (empty here -- ``doc`` was
    never stale to begin with, same root cause as this module's docstring,
    point 2).

    A third user review found that even this was not the end of the
    story: calling this function correctly is still a *per-caller*
    obligation, and ``LLMController.decide()``'s own successfully-parsed
    ``FinishCall`` (the model's OWN choice, not a parse/admission failure)
    never called it at all -- a model returning
    ``reason="cannot_repair", unresolved_artifact_ids=[]`` against a
    document that was, in fact, still FAILing and ``blocked`` had that
    empty list recorded verbatim. See
    ``ControllerRuntime._controller_node``, which now applies this
    function to *every* ``FinishCall`` right after it is decided --
    whichever of the three paths (a guard, ``RuleBasedController``, or
    ``LLMController``) produced it -- rather than trusting each path to
    have called it correctly on its own; this function itself is
    unchanged, it is simply no longer optional for any caller to remember."""
    unresolved = _compute_unresolved(store, state.tracked_artifact_ids, state.blocked_artifact_ids, state.needs_recheck_keys)
    unresolved |= {
        item.checked_artifact.artifact_id
        for item in state.current_evidence
        if item.outcome in (CheckOutcome.FAIL, CheckOutcome.UNKNOWN)
    }
    if state.last_compile_failed:
        unresolved.add(state.document_ref.artifact_id)
    return unresolved


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

        unresolved = _compute_unresolved(store, state.tracked_artifact_ids, state.blocked_artifact_ids, state.needs_recheck_keys)
        # An id owed a recheck on AT LEAST ONE constraint (``needs_recheck_keys``)
        # must never read as "already checked" just because some OTHER,
        # still-valid entry for it survives -- see this module's
        # docstring, point 2b: dropping only the invalidated constraint
        # while leaving an unrelated PASS in place must not let that id
        # look fully covered.
        pending_recheck_ids = {aid for aid, _constraint_id in state.needs_recheck_keys}
        checked_ids = {item.checked_artifact.artifact_id for item in state.current_evidence} - pending_recheck_ids

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
                    unresolved_artifact_ids=sorted(_aggregate_unresolved(state, store)),
                )
            chosen = sorted(actions, key=lambda action: action.target_artifacts[0].artifact_id)[0]
            return _repair_action_to_tool_call(chosen)

        not_yet_checked = sorted(unresolved - checked_ids)
        if not_yet_checked:
            target = store.latest(not_yet_checked[0])
            assert target is not None  # staleness only ever applies to an artifact that exists
            return CheckCall(artifact=target.ref, constraints=self._constraints_for(target.ref), context={})

        if unresolved:
            return FinishCall(reason="unresolved_remaining", unresolved_artifact_ids=sorted(_aggregate_unresolved(state, store)))

        if classification == "UNKNOWN":
            return FinishCall(
                reason="evidence_inconclusive", unresolved_artifact_ids=sorted(_aggregate_unresolved(state, store))
            )

        if not state.compiled:
            if state.last_compile_failed:
                return FinishCall(
                    reason="compile_failed", unresolved_artifact_ids=sorted(_aggregate_unresolved(state, store))
                )
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
                unresolved_artifact_ids=sorted(_aggregate_unresolved(state, store)),
            )
        assert observation.output is not None
        production = observation.output.production
        if production.status == ProductionStatus.FAILED or production.content is None:
            return FinishCall(
                reason="llm_controller_exhausted_retries",
                unresolved_artifact_ids=sorted(_aggregate_unresolved(state, store)),
            )
        decision = LLMControllerDecision.model_validate_json(production.content)
        call = decision.call
        if isinstance(call, CompileCall):
            # Never trust the model's own echo of request_id/output_dir --
            # this controller already knows the one correct execution
            # config for this episode (constructor-only, never shown to
            # the model as something it could legitimately choose
            # differently); a second user review found neither value ever
            # reached the model OR the dispatched call, so a model with no
            # way to know "which request_id is this" could silently send
            # COMPILE down the wrong path. Telling it the right values in
            # the prompt (below) still lets it reason about WHEN to
            # compile; this makes WHICH compile correct regardless.
            call = call.model_copy(update={"request_id": self._request_id, "output_dir": self._output_dir})
        return call

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
        unresolved = _compute_unresolved(store, state.tracked_artifact_ids, state.blocked_artifact_ids, state.needs_recheck_keys)
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
            f"If you choose 'compile', the request_id/output_dir will always be forced to "
            f"{self._request_id!r}/{self._output_dir!r} regardless of what you put in those fields -- "
            f"only use them to help you decide, never invent different ones.\n"
            f"Tracked artifacts with unresolved work but no fresh pass/fail evidence of their own "
            f"(e.g. a production failure, a missing producer, a dropped-and-not-yet-rechecked "
            f"verdict, or left stale by a budget cutoff): {sorted(unresolved)}\n"
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
        graph.add_conditional_edges(START, self._route_at_start, {"controller": "controller", "tool": "tool"})
        graph.add_edge("controller", "tool")
        graph.add_conditional_edges("tool", self._route_after_tool, {"controller": "controller", END: END})
        return graph

    @staticmethod
    def _route_at_start(state: ControllerState) -> str:
        """A fresh episode's initial state never carries a
        ``pending_tool_call`` (``run()``'s own construction), so this
        always routes to ``controller`` for a normal start -- but a
        RESUMED episode may be restarted from a state that was
        checkpointed in the narrow window right *after* ``controller``
        already decided (and, for ``LLMController``, already spent real
        budget deciding) but *before* ``tool`` ever executed that
        decision. A third user review found ``run()`` always entering at
        ``controller`` regardless silently re-decided from scratch in
        that case -- discarding the pending decision entirely (along with
        whatever it cost to produce) and asking the policy again, which
        could pick a *different* action the second time. Routing straight
        to ``tool`` whenever a pending call already survived into the
        resumed state executes exactly the one decision that was already
        made and already paid for, instead of silently replacing it."""
        return "tool" if state.pending_tool_call is not None else "controller"

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
        node (``schemas.ControllerState``'s own docstring).

        The executor/store snapshot (``checkpoint``/``budget``/
        ``actions_taken``/``suspended``/``suspend_reason``) is taken ONCE,
        right before returning -- deliberately *after* ``call`` is fully
        decided, never before. A third user review found this node
        previously returned only ``pending_tool_call``, and this method's
        own first fix for that mistakenly read the snapshot too early (as
        a local ``base_updates`` built before ``self._policy.decide()``
        ever ran) -- harmless for the two guard trips, which never touch
        the executor, but wrong for the real decision path:
        ``self._policy.decide()`` may itself have already spent real
        budget/action-count by the time it returns (``LLMController``'s
        own call to ``tool_executor.run_generation()``, this module's
        docstring points 6-7), and a snapshot taken before that call ran
        is exactly the stale 0/0 this fix exists to stop. A real repro
        showed the executor having actually charged 7 tokens and 1 action
        while the serialized ``ControllerState`` right after this node
        still read 0/0, so a checkpoint/resume landing in that exact
        window silently reopened budget that had, in reality, already
        been spent.

        Whatever ``call`` turns out to be -- a guard trip, or whatever
        ``self._policy.decide()`` returns -- a ``FinishCall`` among them
        always gets ``_aggregate_unresolved()`` unioned into its own
        ``unresolved_artifact_ids`` right here, unconditionally, before
        ``pending_tool_call`` is ever set. A third user review found this
        was still only true for the two guard trips and for
        ``LLMController``'s own *failure* fallbacks (admission-rejected,
        parse-exhausted) -- a model that successfully parsed into a
        ``FinishCall`` of its OWN choosing (e.g.
        ``reason="cannot_repair"``) returned it completely unmodified,
        with whatever (possibly empty, possibly wrong) list the model
        itself supplied, bypassing the aggregate entirely. Enforcing this
        once, here, after every path that can produce a ``FinishCall``,
        rather than inside each one individually, is what makes this a
        guarantee instead of a convention every current and future
        ``ControllerPolicy`` has to remember to uphold on its own -- the
        guard branches below now pass an empty list for exactly this
        reason, relying entirely on this one union to fill it in."""
        if state.done:
            return {}
        if state.step_count >= self._max_steps:
            call: ToolCall = FinishCall(reason="max_steps_reached", unresolved_artifact_ids=[])
        elif _repeated_action(state.history, threshold=self._repeat_threshold):
            call = FinishCall(reason="repeated_action_detected", unresolved_artifact_ids=[])
        else:
            call = self._policy.decide(state, self._store, self._tool_executor)
        if isinstance(call, FinishCall):
            call = call.model_copy(
                update={
                    "unresolved_artifact_ids": sorted(
                        set(call.unresolved_artifact_ids) | _aggregate_unresolved(state, self._store)
                    )
                }
            )
        return {
            "pending_tool_call": call,
            "checkpoint": self._store.save_checkpoint(label=f"controller-{state.step_count}"),
            "budget": self._tool_executor.budget,
            "actions_taken": self._tool_executor.actions_taken,
            "suspended": self._tool_executor.is_suspended,
            "suspend_reason": self._tool_executor.suspend_reason,
        }

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
          changed input (``_prune_invalidated_evidence``) -- any pair that
          loses its entry this way (``_invalidated_keys``) joins
          ``needs_recheck_keys`` (this module's docstring, point 2b/13)
          rather than simply vanishing. Only the EXACT pairs this call
          actually produced fresh, currently-valid evidence for are
          cleared from that set -- never every pending pair sharing the
          checked artifact_id, since ``CheckCall.constraints`` is never
          guaranteed to be the full set an artifact could be checked
          against (point 13: a caller, ``LLMController`` in particular,
          can legitimately name only some of them).
        - PATCH/REPLAN grow ``tracked_artifact_ids`` to include every
          artifact the repair's own cascade touched or discovered broken
          (``RepairExecutionResult.final_outcome_by_artifact``/
          ``.left_stale``), drop stale evidence for all of them, then pull
          each one's own ``ArtifactGraphStore.live_evidence()`` fresh --
          never the ``regenerated``/``reverified_untouched`` audit lists
          directly, which ``RepairExecutionResult``'s own docstring already
          warns against scanning for "current truth" (the same artifact_id
          can legitimately appear more than once across a multi-sweep
          cascade, superseded entries included). A second user review
          found the original version merged ONLY ``regenerated[*].
          reverify_evidence``, which both (a) always skipped
          ``reverified_untouched`` entirely -- an artifact re-confirmed
          PASS without being regenerated has no ``RegeneratedArtifactRecord``
          at all, so its fresh verdict (recorded by ``_reaffirm``, rebound
          to the artifact's new version) never reached the controller --
          and (b) could resurrect a STALE, already-superseded record from
          an early sweep pass for an id later reaffirmed clean, exactly
          the failure mode ``final_outcome_by_artifact``'s own docstring
          warns about. ``live_evidence()`` sidesteps both: it always
          reflects the artifact's truly current version, regardless of
          which mechanism (regenerate vs. reaffirm) last settled it.
          Whatever still gets dropped after that re-pull (an id left
          genuinely stale, or with no live verdict at all) joins
          ``needs_recheck_keys`` precisely like the CHECK branch, pair by
          pair; only the pairs ``live_evidence()`` actually returned
          something fresh for are cleared -- a touched id left with no
          constraints registered, or genuinely still stale/blocked,
          contributes nothing to clear, so any of its own pending pairs
          stay pending rather than being wiped just for having been
          "touched." An artifact left ``is_stale()``-true with no fresh
          evidence of its own (a
          production failure, a missing producer, a budget-cut-short
          cascade) is still tracked and will surface through
          ``_compute_unresolved()`` even with no Evidence to show for it.
          Any successful dispatch also invalidates ``compiled``/
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
                invalidated = _invalidated_keys(self._store, merged)
                updates["current_evidence"] = _prune_invalidated_evidence(self._store, merged)
                updates["tracked_artifact_ids"] = state.tracked_artifact_ids | {call.artifact.artifact_id}
                # Only the EXACT (artifact_id, constraint_id) pairs this
                # call actually produced currently-valid evidence for are
                # cleared -- never every pending pair for call.artifact's
                # id. ``CheckCall.constraints`` is never guaranteed to
                # cover everything that artifact could be checked
                # against (a third user review: the LLM controller in
                # particular can name an arbitrary subset) -- clearing by
                # artifact_id alone let a CHECK that only re-examined
                # constraint B silently mark constraint A's own, still
                # outstanding, recheck obligation as satisfied too.
                resolved_keys = {
                    (item.checked_artifact.artifact_id, item.constraint_id)
                    for item in observation.output.evidence
                    if self._store.is_evidence_still_valid(item)
                }
                updates["needs_recheck_keys"] = (state.needs_recheck_keys | invalidated) - resolved_keys
            elif isinstance(call, InspectArtifactCall):
                updates["tracked_artifact_ids"] = state.tracked_artifact_ids | {call.artifact_id}
            elif isinstance(call, QueryDependenciesCall):
                updates["tracked_artifact_ids"] = state.tracked_artifact_ids | {call.artifact_id}
            elif isinstance(call, (PatchCall, ReplanCall)):
                result = observation.output.result
                touched_ids = set(result.final_outcome_by_artifact.keys())
                touched_ids |= {ref.artifact_id for ref in result.left_stale}
                pruned = _drop_evidence_for(state.current_evidence, touched_ids)
                fresh_live: List[Evidence] = []
                for artifact_id in sorted(touched_ids):
                    fresh_live.extend(self._store.live_evidence(artifact_id))
                pruned = _merge_evidence(pruned, fresh_live)
                invalidated = _invalidated_keys(self._store, pruned)
                updates["current_evidence"] = _prune_invalidated_evidence(self._store, pruned)
                updates["tracked_artifact_ids"] = state.tracked_artifact_ids | touched_ids
                updates["blocked_artifact_ids"] = _update_blocked(state.blocked_artifact_ids, result)
                # Same discipline as the CHECK branch above: a touched id
                # with no constraints registered for it, or one left
                # genuinely stale/blocked, produces no fresh_live entries
                # at all -- its own pending recheck pairs (if any) must
                # stay pending, not be cleared just because the id was
                # "touched" by this cascade.
                resolved_keys = {(item.checked_artifact.artifact_id, item.constraint_id) for item in fresh_live}
                updates["needs_recheck_keys"] = (state.needs_recheck_keys | invalidated) - resolved_keys
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

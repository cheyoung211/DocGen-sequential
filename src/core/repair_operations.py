"""Repair operations and fixed repair policies (Stage 4 of the "Revision
2026.10.15" plan: "4단계. 고정 복구 정책과 분기 재실행").

Builds on Stage 1's ``Producer``/``RepairPolicy`` Protocols, Stage 2's
``ArtifactGraphStore`` (versioning, ``impact_candidates``, staleness,
INFERRED edges), and Stage 3's ``ToolExecutor``/``Constraint``/``Validator``
machinery. Nothing here is wired into ``scripts/run_full_pipeline.py`` or
any agent -- per the user's explicit scope decision for this stage, this
module is proven correct with Protocol-conforming fakes/stubs
(``test/core/test_repair_operations.py``), the same discipline Stages 1-3
already used for anything that would otherwise require a real, expensive
LLM call in a test. Actually running this against the real pipeline with
the local Qwen3.6 model (producing a genuine small natural-failure/
injected-failure pilot) is an explicitly deferred follow-up, not part of
this stage's own code.

**Revised after a user review found 8 real bugs in the first pass** (full
detail in memory ``docgen_stage_review_rigor``) -- this revision:

1. Persists the Producer's *real* output and *real* used inputs
   (``ProductionResult.content``/``.read_input_versions``) instead of
   ``content=None`` and a guessed re-pin of stale bookkeeping.
2. Re-checks staleness fresh on every sweep of the cascade (a fixed-point
   loop, not a single visit-once pass) so a candidate resolved before its
   own upstream settles gets revisited once that upstream actually changes.
3. Classifies a fresh check's evidence into FAIL / PASS / UNKNOWN three
   ways (``_classify_evidence``) -- UNKNOWN-only (or no evidence at all)
   is ``RepairOutcome.INCONCLUSIVE``, never silently read as resolved.
4. Carries forward INFERRED edges onto a new version
   (``_carry_forward_inferred_edges``) and re-binds reaffirmed evidence to
   the new version it now actually describes, so neither the dependency
   graph nor ``live_evidence()`` goes blind the moment a reaffirmed
   artifact's version number moves.
5. (fixed in ``branching_runner.py``, not here) executes every action a
   policy's ``decide()`` returns, not just the first.
6. Routes every real generation/check call through Stage 3's
   ``ToolExecutor`` (``run_generation()``/``execute(CheckCall(...))``)
   instead of calling ``Producer``/``Validator`` directly -- budget
   admission and cost logging are enforced exactly once, in the place
   Stage 3 already built and reviewed for it, not re-implemented here.
7. (fixed in ``branching_runner.py``, not here) gives each branch its own
   independent context/graph rather than sharing one mutable object.
8. Gives "whole-document regeneration" its own unconditional execution
   path (``_execute_unconditional``) that force-regenerates every artifact
   in the document's impact set with no selective skip -- the earlier
   version routed REGENERATE_DOCUMENT through the same selective cascade
   as every other action type, which made it behave identically to
   "fixed graph-aware"'s rare fallback case instead of the deliberately
   blind baseline ``docs/stage1_experiment_design.md`` defines.

**Revised a second time after a user review found 2 more real bugs in
this file** (full detail in memory ``docgen_stage_review_rigor``):

9. ``_execute_selective``'s final ``left_stale`` computation (and the
   same-shaped one in ``_execute_unconditional``) no longer trusts
   "received a ``RegeneratedArtifactRecord``/reaffirmation at some point
   during this call" as proof of a *final* resolved state. Two concrete
   repros: (a) explicit targets given out of dependency order (``[b, a]``
   where ``b`` depends on ``a``) -- regenerating ``b`` first, then ``a``,
   leaves ``b`` stale again with its only record dated *before* ``a`` ever
   changed; (b) ``_execute_unconditional``'s plain id-sort processing
   order (``doc``, ``b``, ``z`` for a ``doc -> z -> b`` chain) means
   ``doc`` gets regenerated from ``z``'s *old* version, and nothing ever
   revisits it once ``z`` changes afterward. Both previously reported
   ``RepairOutcome.RESOLVED`` with the store's own ``is_stale()`` still
   ``True`` for the affected artifact -- the two disagreeing is exactly
   what this module's own completion criterion forbids. Fixed by (i) no
   longer excluding explicitly-named targets from the selective cascade's
   own re-sweep pool, (ii) giving the unconditional path the same
   fixed-point sweep ``_execute_selective`` already had plus a best-effort
   ``_dependency_order`` for its first pass, and (iii) computing the
   final ``left_stale`` in both from a fresh ``is_stale()`` check across
   the whole relevant id set, excluding only ids a
   ``PRODUCTION_FAILED``/``NO_PRODUCER`` record already fully explains --
   never excluding on "was touched at some point" alone.
10. ``_check()`` no longer treats every non-budget ``CheckCall`` rejection
    as budget exhaustion. A real repro: the wrapped ``Validator.check()``
    raises, ``ToolExecutor`` correctly reports
    ``ToolErrorCode.EXECUTION_FAILED`` (10,000 budget still remaining,
    executor not suspended) -- the previous version still returned
    ``budget_rejected=True`` for *any* ``not observation.ok``, which
    wrongly aborted the whole cascade as out of room. Now only
    ``schemas.BUDGET_ERROR_CODES`` membership sets that flag; anything
    else (``EXECUTION_FAILED``/``TIMEOUT``) comes back as empty evidence,
    which the existing UNKNOWN/``left_stale`` path already handles
    honestly with no new return state needed.

Two components:

- ``RepairOperationExecutor`` carries out one ``RepairAction`` -- the
  Stage 4 checklist's "block patch, section regeneration, subgraph replan,
  whole-document regeneration 연산을 구현한다" -- and, for the first three
  action types, cascades through everything the repair could plausibly
  affect, re-verifying each one fresh and regenerating only what still
  fails ("수정 이후 영향받는 artifact를 재검사하고, 필요한 항목만 재생성하는
  고정 graph-aware 정책을 구현한다"). REGENERATE_DOCUMENT instead forces
  every reachable artifact unconditionally -- see point 8 above.

- Two concrete ``RepairPolicy`` implementations fill in named rows from
  ``docs/stage1_experiment_design.md``'s 7-comparison-group table:
  ``FixedGraphAwareRepairPolicy`` ("fixed graph-aware") and
  ``AlwaysRegenerateDocumentPolicy`` ("whole-document regeneration" -- and,
  combined with ``branching_runner.BranchingRunner``'s ``check_timing``
  option, also "final-only verification", the same scope rule under a
  different trigger-timing condition, not a different policy).

**Deliberate scope boundary: no producer for direction 1's finer-grained
artifact kinds.** The Notion plan's artifact vocabulary includes
DEFINITION/SYMBOL/EQUATION/TABLE/CLAIM/FIGURE/SOURCE_DATA/CALCULATION
alongside DOCUMENT/SECTION/BLOCK (``schemas.ArtifactKind``) -- but today's
real pipeline only ever produces a whole plan (``PlannerAgent.generate_plan``)
or a section's blocks (``TextAgent.process_node``/``regenerate_blocks``).
There is no agent entry point that regenerates "just this one equation"
independent of the block containing it. A ``RepairOperationExecutor`` asked
to regenerate a target of one of those kinds therefore cannot look one up
in its producer registry -- this is reported honestly as
``RepairOutcome.NO_PRODUCER``, not papered over with an invented producer or
silently skipped. See ``RepairOutcome``'s own docstring in ``schemas.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.core.artifact_graph import ArtifactGraphStore
from src.core.interfaces import Producer
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    BUDGET_ERROR_CODES,
    CheckCall,
    CheckOutcome,
    Constraint,
    CostRecord,
    DependencyRelationKind,
    Evidence,
    ProductionResult,
    ProductionStatus,
    RegeneratedArtifactRecord,
    RepairAction,
    RepairActionType,
    RepairExecutionResult,
    RepairOutcome,
)
from src.core.tool_executor import ToolExecutor

ProducerRegistry = Dict[ArtifactKind, Producer]
"""Which real (or fake, in tests) ``Producer`` regenerates an artifact of a
given kind. Deliberately a plain dict, not a class -- a missing entry *is*
the ``RepairOutcome.NO_PRODUCER`` signal, so there is nothing for a wrapper
class to add beyond what ``dict.get`` already gives for free. Holds *raw*
producers, never pre-wrapped in ``cost_tracking.CostTrackingProducer`` --
``ToolExecutor.run_generation()`` does that wrapping itself; wrapping here
too would double-wrap and double-measure."""

ConstraintProvider = Callable[[ArtifactRef], List[Constraint]]
"""Given a live artifact ref, which ``Constraint``s should be checked
against it right now. A plain callable, supplied by the caller -- neither
``ArtifactGraphStore`` nor ``Validator`` tracks "which constraints apply to
which artifact" as its own state (consistent with Stage 3's
``CheckCall.constraints`` always being caller-supplied, never looked up
internally). Returning an empty list means "nothing to check this artifact
against": a cascade candidate in that state is left exactly as-is
(reported via ``RepairExecutionResult.left_stale``, never fabricated a
verdict either way)."""

def _classify_evidence(evidence: List[Evidence]) -> str:
    """One of ``"FAIL"``/``"PASS"``/``"UNKNOWN"`` -- the three-way split a
    review found missing (point 3 above). ``"PASS"`` requires at least one
    PASS and zero FAIL; empty evidence or any UNKNOWN-with-no-FAIL both
    fall through to ``"UNKNOWN"`` -- "nothing failed" is not the same claim
    as "something passed", and a caller must never treat them as
    interchangeable (``RepairOutcome.INCONCLUSIVE`` is exactly this
    distinction made explicit downstream)."""
    if any(item.outcome == CheckOutcome.FAIL for item in evidence):
        return "FAIL"
    if evidence and all(item.outcome == CheckOutcome.PASS for item in evidence):
        return "PASS"
    return "UNKNOWN"


class RepairOperationExecutor:
    """Carries out one ``RepairAction`` against a live ``ArtifactGraphStore``,
    through a ``ToolExecutor`` bound to that same store (one instance, one
    store, one budget -- matching ``ToolExecutor``'s own "one instance
    targets exactly one store" design, and the reason ``BranchingRunner``
    builds a fresh pair of both per branch).

    PATCH_BLOCK/REGENERATE_SECTION/REPLAN_SUBGRAPH share one selective
    cascade (``_execute_selective``): every directly-named target is
    regenerated; from each one actually written, ``impact_candidates`` is
    swept **repeatedly, re-checking staleness fresh each sweep**, until a
    full sweep makes no further progress (bounded by
    ``len(affected_ids) + 1`` sweeps) -- not a single visit-once pass,
    which a review found could permanently miss a candidate re-invalidated
    by a sibling processed *later* in the same pass (point 2 above).
    REGENERATE_DOCUMENT instead force-regenerates everything in the
    document's impact set unconditionally (``_execute_unconditional``,
    point 8 above) -- deliberately a different, blunter operation, not a
    wider instance of the same selective cascade.

    Every real generation/check call goes through ``ToolExecutor`` (point 6
    above): a rejection due to budget/max-actions/suspension stops the
    cascade -- whatever was not yet reached is reported via
    ``RepairExecutionResult.left_stale``/``.budget_exhausted``, never
    silently treated as resolved.
    """

    def __init__(
        self,
        store: ArtifactGraphStore,
        tool_executor: ToolExecutor,
        producer_registry: ProducerRegistry,
        constraint_provider: ConstraintProvider,
        base_context: Dict[str, Any],
        *,
        model_name: str = "unknown",
    ) -> None:
        self._store = store
        self._tool_executor = tool_executor
        self._producers = producer_registry
        self._constraints_for = constraint_provider
        self._base_context = base_context
        self._model_name = model_name

    @property
    def tool_executor(self) -> ToolExecutor:
        """Exposed so a caller (``BranchingRunner``) can read
        ``.budget``/``.is_suspended`` after the fact without this class
        needing its own parallel bookkeeping of the same facts."""
        return self._tool_executor

    def execute(
        self, action: RepairAction, *, triggering_evidence: Optional[List[Evidence]] = None
    ) -> RepairExecutionResult:
        """Carry out ``action`` and return exactly what happened. Logs the
        action itself via ``ArtifactGraphStore.record_action`` exactly once,
        at the start, regardless of how many artifacts its cascade goes on
        to touch -- the thing that "began executing" is the one
        ``RepairAction``, not each individual regeneration inside it.
        """
        self._store.record_action(action, triggering_evidence)
        if action.action_type == RepairActionType.REGENERATE_DOCUMENT:
            return self._execute_unconditional(action)
        return self._execute_selective(action)

    # -- selective cascade (PATCH_BLOCK / REGENERATE_SECTION / REPLAN_SUBGRAPH) --

    def _execute_selective(self, action: RepairAction) -> RepairExecutionResult:
        regenerated: List[RegeneratedArtifactRecord] = []
        reverified_untouched: List[ArtifactRef] = []
        left_stale: List[ArtifactRef] = []
        costs: List[CostRecord] = []
        budget_exhausted = False
        blocked_ids: Set[str] = set()
        # The most recent thing that actually happened to each artifact_id,
        # updated in true chronological execution order (never
        # reconstructed after the fact from `regenerated`/
        # `reverified_untouched`, since the same id can appear in both --
        # regenerated, then later reaffirmed, or vice versa -- across
        # different sweep passes, and only the *last* one is still true).
        # Drives the final ``left_stale`` decision below: only
        # ``RESOLVED``/reaffirmed-PASS can be *wrong* about whether this id
        # is still stale right now; every other outcome already reports
        # its own current unresolved state honestly.
        last_outcome: Dict[str, RepairOutcome] = {}

        seed_refs: List[ArtifactRef] = []
        seen_ids: Set[str] = set()
        for ref in action.target_artifacts:
            current = self._store.latest(ref.artifact_id)
            if current is None or current.ref.artifact_id in seen_ids:
                continue
            seen_ids.add(current.ref.artifact_id)
            seed_refs.append(current.ref)

        affected_ids: Set[str] = set()
        remaining_seeds = list(seed_refs)
        while remaining_seeds:
            live_ref = remaining_seeds.pop(0)
            record, cost, exhausted = self._regenerate(live_ref, action)
            regenerated.append(record)
            last_outcome[live_ref.artifact_id] = record.outcome
            if cost is not None:
                costs.append(cost)
            if exhausted:
                budget_exhausted = True
                break
            if record.outcome in (RepairOutcome.PRODUCTION_FAILED, RepairOutcome.NO_PRODUCER):
                blocked_ids.add(live_ref.artifact_id)
            else:
                affected_ids |= self._store.impact_candidates(live_ref.artifact_id)
        left_stale_ids: Set[str] = set()
        for ref in remaining_seeds:  # never even attempted -- budget ran out first
            left_stale.append(ref)
            left_stale_ids.add(ref.artifact_id)

        if not budget_exhausted:
            max_sweeps = len(affected_ids) + 1
            for _ in range(max_sweeps):
                if budget_exhausted:
                    break
                made_progress = False
                for candidate_id in sorted(affected_ids):
                    if budget_exhausted:
                        break
                    if candidate_id in blocked_ids:
                        continue
                    if not self._store.is_stale(candidate_id):
                        continue  # settled (or never affected) -- nothing to do
                    current = self._store.latest(candidate_id)
                    if current is None:
                        continue
                    live_ref = current.ref
                    constraints = self._constraints_for(live_ref)
                    if not constraints:
                        continue  # nothing to check -- left_stale, collected below
                    evidence, rejected = self._check(live_ref, constraints)
                    if rejected:
                        budget_exhausted = True
                        break
                    made_progress = True
                    classification = _classify_evidence(evidence)
                    if classification == "FAIL":
                        record, cost, exhausted = self._regenerate(live_ref, action)
                        regenerated.append(record)
                        last_outcome[candidate_id] = record.outcome
                        if cost is not None:
                            costs.append(cost)
                        if exhausted:
                            budget_exhausted = True
                            break
                        if record.outcome in (RepairOutcome.PRODUCTION_FAILED, RepairOutcome.NO_PRODUCER):
                            blocked_ids.add(candidate_id)
                        else:
                            assert record.new_ref is not None
                            affected_ids |= self._store.impact_candidates(
                                record.new_ref.artifact_id
                            ) - {candidate_id}
                    elif classification == "PASS":
                        new_ref = self._reaffirm(live_ref, evidence)
                        reverified_untouched.append(new_ref)
                        last_outcome[candidate_id] = RepairOutcome.RESOLVED
                        affected_ids |= self._store.impact_candidates(new_ref.artifact_id) - {candidate_id}
                    # "UNKNOWN": leave exactly as-is (still stale) -- collected
                    # into left_stale below if it remains so at the end.
                if not made_progress:
                    break

        # Regardless of processing history, every explicitly-named target AND
        # every impact candidate that is CURRENTLY still stale belongs in
        # left_stale -- unless its own `last_outcome` already honestly
        # reports it as unresolved (STILL_FAILING/INCONCLUSIVE/
        # PRODUCTION_FAILED/NO_PRODUCER via `regenerated`), or it was
        # already listed above as never even attempted. Only RESOLVED (a
        # fresh regenerate's own check passing, or a reaffirm) claims "this
        # is fine now" -- and a review found that claim can go stale within
        # this same call, when a *sibling* processed afterward invalidates
        # it (e.g. explicit targets given as ``[b, a]`` where ``b`` depends
        # on ``a``: regenerating ``b`` first, then ``a``, leaves ``b``
        # stale again with its only record dated from *before* ``a`` ever
        # changed). ``is_stale()`` is this store's one source of truth
        # (``artifact_graph.py``'s own module docstring) -- an earlier
        # RESOLVED claim is simply not proof of a *later* final state, so a
        # candidate whose last known outcome is RESOLVED (or never touched
        # at all) still gets a fresh live check here rather than being
        # trusted on the strength of that one past record.
        for candidate_id in sorted(seen_ids | affected_ids):
            if candidate_id in left_stale_ids:
                continue
            outcome = last_outcome.get(candidate_id)
            if outcome is not None and outcome != RepairOutcome.RESOLVED:
                continue  # already honestly reported as unresolved via `regenerated`
            if self._store.is_stale(candidate_id):
                current = self._store.latest(candidate_id)
                if current is not None:
                    left_stale.append(current.ref)
                    left_stale_ids.add(candidate_id)

        return RepairExecutionResult(
            action=action,
            regenerated=regenerated,
            reverified_untouched=reverified_untouched,
            left_stale=left_stale,
            overall_outcome=self._combine_outcomes(regenerated, left_stale, budget_exhausted),
            cost=costs,
            budget_exhausted=budget_exhausted,
        )

    # -- unconditional whole-document regeneration (REGENERATE_DOCUMENT) --

    def _execute_unconditional(self, action: RepairAction) -> RepairExecutionResult:
        """"whole-document regeneration" baseline's own execution semantic:
        force-regenerate the document and *every* artifact in its impact
        set, unconditionally -- no check-first, no skip-if-still-passing.
        That selectivity is exactly what distinguishes "fixed graph-aware"
        from this deliberately blind baseline (``docs/
        stage1_experiment_design.md`` §3); routing REGENERATE_DOCUMENT
        through the same selective cascade as the other three action types
        (the first-pass bug) made the two baselines behave identically
        once either one escalated to document scope.

        A later review found that forcing everything exactly once, in
        plain id-sort order, is not the same as everything actually being
        fresh afterward: if an artifact processed early is itself an input
        to one processed later (e.g. ``doc`` reads ``z`` reads ``b``,
        id-sorted order visits ``doc``, ``b``, ``z``), the early one is
        regenerated from what was, at that moment, still the *old* version
        of its own input, and nothing ever revisited it once that input
        changed. Two things fix this, mirroring ``_execute_selective``'s
        own established pattern rather than inventing a new one: a
        best-effort dependency order for the first pass
        (``_dependency_order`` -- a pure ordering optimization, not a
        correctness requirement, since the dependency graph is explicitly
        allowed to contain cycles that no order can fully satisfy, per
        ``artifact_graph.py``'s own module docstring), followed by a
        bounded fixed-point sweep that keeps re-regenerating whatever is
        still (or newly) stale until a full pass makes no further
        progress. The final ``left_stale`` is then computed from a fresh
        ``is_stale()`` check across the *whole* impact set, regardless of
        processing history -- the same fix ``_execute_selective`` needed
        for the identical reason."""
        regenerated: List[RegeneratedArtifactRecord] = []
        left_stale: List[ArtifactRef] = []
        costs: List[CostRecord] = []
        budget_exhausted = False
        blocked_ids: Set[str] = set()
        # Same role as `_execute_selective`'s own `last_outcome`: the most
        # recent thing that actually happened to each artifact_id, updated
        # in true chronological order -- only a RESOLVED outcome can be
        # *wrong* about whether this id is still stale right now.
        last_outcome: Dict[str, RepairOutcome] = {}

        document_refs = [ref for ref in action.target_artifacts if ref.kind == ArtifactKind.DOCUMENT]
        if not document_refs:
            for ref in action.target_artifacts:
                current = self._store.latest(ref.artifact_id)
                if current is not None:
                    left_stale.append(current.ref)
            return RepairExecutionResult(
                action=action,
                regenerated=[],
                reverified_untouched=[],
                left_stale=left_stale,
                overall_outcome=RepairOutcome.NO_PRODUCER,
                cost=[],
            )

        current = self._store.latest(document_refs[0].artifact_id)
        document_ref = current.ref if current is not None else document_refs[0]
        all_ids = self._store.impact_candidates(document_ref.artifact_id)
        order = self._dependency_order(all_ids)

        processed_ids: Set[str] = set()
        for artifact_id in order:
            if budget_exhausted:
                break
            current = self._store.latest(artifact_id)
            if current is None:
                continue
            record, cost, exhausted = self._regenerate(current.ref, action)
            regenerated.append(record)
            last_outcome[artifact_id] = record.outcome
            processed_ids.add(artifact_id)
            if cost is not None:
                costs.append(cost)
            if exhausted:
                budget_exhausted = True
            elif record.outcome in (RepairOutcome.PRODUCTION_FAILED, RepairOutcome.NO_PRODUCER):
                blocked_ids.add(artifact_id)

        if budget_exhausted:
            for artifact_id in order:  # never even attempted -- budget ran out first
                if artifact_id not in processed_ids:
                    current = self._store.latest(artifact_id)
                    if current is not None:
                        left_stale.append(current.ref)
        else:
            remaining = set(order) - blocked_ids
            max_sweeps = len(remaining) + 1
            for _ in range(max_sweeps):
                if budget_exhausted:
                    break
                made_progress = False
                for artifact_id in sorted(remaining):
                    if budget_exhausted:
                        break
                    if artifact_id in blocked_ids or not self._store.is_stale(artifact_id):
                        continue
                    current = self._store.latest(artifact_id)
                    if current is None:
                        continue
                    record, cost, exhausted = self._regenerate(current.ref, action)
                    regenerated.append(record)
                    last_outcome[artifact_id] = record.outcome
                    made_progress = True
                    if cost is not None:
                        costs.append(cost)
                    if exhausted:
                        budget_exhausted = True
                        break
                    if record.outcome in (RepairOutcome.PRODUCTION_FAILED, RepairOutcome.NO_PRODUCER):
                        blocked_ids.add(artifact_id)
                if not made_progress:
                    break

            # Regardless of processing history -- forced once, swept again
            # any number of times, or never reached a sweep pass at all --
            # every artifact in this document's impact set that is
            # CURRENTLY still stale belongs in left_stale, unless its own
            # `last_outcome` already honestly reports it as unresolved
            # (STILL_FAILING/INCONCLUSIVE/PRODUCTION_FAILED/NO_PRODUCER).
            # Only RESOLVED (or never having been reached at all, e.g. the
            # sweep bound ran out) leaves room for the live check below to
            # actually matter -- see `_execute_selective`'s identical
            # reasoning.
            for artifact_id in sorted(all_ids):
                outcome = last_outcome.get(artifact_id)
                if outcome is not None and outcome != RepairOutcome.RESOLVED:
                    continue
                if self._store.is_stale(artifact_id):
                    current = self._store.latest(artifact_id)
                    if current is not None:
                        left_stale.append(current.ref)

        return RepairExecutionResult(
            action=action,
            regenerated=regenerated,
            reverified_untouched=[],
            left_stale=left_stale,
            overall_outcome=self._combine_outcomes(regenerated, left_stale, budget_exhausted),
            cost=costs,
            budget_exhausted=budget_exhausted,
        )

    def _dependency_order(self, ids: Set[str]) -> List[str]:
        """Best-effort order, restricted to ``ids``, with each artifact
        ahead of anything that currently declares depending on it --
        Kahn's algorithm over the same "currently applicable edge" notion
        ``ArtifactGraphStore._current_dependents`` uses (an edge whose
        ``to_artifact`` is no longer that artifact's current version is
        ignored, exactly like that method). The dependency graph is
        **not** assumed acyclic (``artifact_graph.py``'s own module
        docstring says so explicitly) -- any ``ids`` left over once no
        further zero-dependency node remains (a genuine cycle within this
        set) are appended in plain id-sort order instead of raising.

        This ordering only ever *reduces how often*
        ``_execute_unconditional``'s own fixed-point sweep has to do
        further work after the first pass -- it is never what makes the
        final result correct on its own (a cycle, or an input declared
        only after this call started, can still leave something stale
        after even a perfect topological pass); the sweep, and the final
        live ``is_stale()`` check after it, are."""
        successors: Dict[str, Set[str]] = {artifact_id: set() for artifact_id in ids}
        in_degree: Dict[str, int] = {artifact_id: 0 for artifact_id in ids}
        for edge in self._store.edges:
            from_id = edge.from_artifact.artifact_id
            to_id = edge.to_artifact.artifact_id
            if from_id not in ids or to_id not in ids or from_id == to_id or to_id in successors[from_id]:
                continue
            current = self._store.latest(to_id)
            if current is None or current.ref.version != edge.to_artifact.version:
                continue  # superseded edge -- no longer a live relation
            successors[from_id].add(to_id)
            in_degree[to_id] += 1

        order: List[str] = []
        seen: Set[str] = set()
        ready = sorted(artifact_id for artifact_id in ids if in_degree[artifact_id] == 0)
        while ready:
            node = ready.pop(0)
            seen.add(node)
            order.append(node)
            newly_ready = set()
            for dependent in successors[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    newly_ready.add(dependent)
            if newly_ready:
                ready = sorted(set(ready) | newly_ready)
        order.extend(sorted(ids - seen))  # a genuine cycle -- the sweep fixes correctness, not this order
        return order

    # -- ToolExecutor-routed primitives --

    def _call_producer(
        self, producer: Producer, artifact_id: str, context: Dict[str, Any], *, stage: str
    ) -> Tuple[ProductionResult, Optional[CostRecord], bool]:
        """Returns ``(production, cost, admission_rejected)``. Routes
        through ``ToolExecutor.run_generation()`` -- admission checked
        before the call, cost charged against this executor's own budget
        either way (point 6 above). ``admission_rejected=True`` means the
        call was never dispatched at all (budget/max-actions/suspended);
        the ``ProductionResult`` is a synthetic FAILED placeholder in that
        case, not a real attempt by the wrapped Producer."""
        observation = self._tool_executor.run_generation(
            producer, artifact_id, context, model_name=self._model_name, stage=stage
        )
        if not observation.ok:
            assert observation.error is not None
            rejected = observation.error.code in BUDGET_ERROR_CODES
            return (
                ProductionResult(status=ProductionStatus.FAILED, error=observation.error.message),
                observation.cost,
                rejected,
            )
        assert observation.output is not None
        return observation.output.production, observation.cost, False

    def _check(self, ref: ArtifactRef, constraints: List[Constraint]) -> Tuple[List[Evidence], bool]:
        """Returns ``(evidence, budget_rejected)``. Routes through
        ``ToolExecutor.execute(CheckCall(...))`` -- which already records
        the resulting ``Evidence`` to the store itself
        (``ToolExecutor._dispatch_check``), so this method must not record
        it again.

        A rejected ``CheckCall`` does **not** only mean budget/max-actions/
        suspension -- a review found a real repro where the wrapped
        ``Validator.check()`` itself raised (e.g. a ``RuntimeError``),
        which ``ToolExecutor._execute_inner`` correctly reports as
        ``ToolErrorCode.EXECUTION_FAILED``, not any budget code, with
        plenty of budget still remaining. The earlier version of this
        method treated *every* ``not observation.ok`` as a budget
        rejection, which made a single flaky check call wrongly abort the
        entire cascade/branch as "budget exhausted". ``budget_rejected`` is
        now only ``True`` for ``schemas.BUDGET_ERROR_CODES`` membership;
        any other failure (``EXECUTION_FAILED``/``TIMEOUT``) comes back as
        ``([], False)`` -- empty evidence, same as a genuinely unknown
        verdict, which the existing ``_classify_evidence``/``UNKNOWN``
        path already handles honestly (left stale, never fabricated)
        without this method needing a third return state."""
        call = CheckCall(artifact=ref, constraints=constraints, context=dict(self._base_context))
        observation = self._tool_executor.execute(call)
        if not observation.ok:
            assert observation.error is not None
            return [], observation.error.code in BUDGET_ERROR_CODES
        assert observation.output is not None
        return observation.output.evidence, False

    # -- regeneration / reaffirmation --

    def _regenerate(
        self, ref: ArtifactRef, action: RepairAction
    ) -> Tuple[RegeneratedArtifactRecord, Optional[CostRecord], bool]:
        """Returns ``(record, cost, budget_exhausted)``."""
        producer = self._producers.get(ref.kind)
        if producer is None:
            return (
                RegeneratedArtifactRecord(
                    pre_repair_ref=ref,
                    production=ProductionResult(
                        status=ProductionStatus.FAILED,
                        error=f"no producer registered for kind {ref.kind.value!r}",
                    ),
                    outcome=RepairOutcome.NO_PRODUCER,
                ),
                None,
                False,
            )

        stored = self._store.get_version(ref.artifact_id, ref.version)
        previous_inputs = stored.ref.read_input_versions if stored is not None else []
        intended_inputs = [self._repin(i) for i in previous_inputs]

        context = dict(self._base_context)
        # What this call is being ASKED to use -- a conforming Producer with
        # no finer-grained tracking of its own may honestly echo this back
        # on its ProductionResult (see that schema's docstring); this
        # executor never fabricates `read_input_versions` on the Producer's
        # behalf beyond handing it this explicit intention.
        context["read_input_versions"] = intended_inputs
        if action.use_diagnostic_feedback and action.feedback_text:
            context["error_message"] = action.feedback_text
        else:
            # Explicitly clear/neutralize -- a stale error_message left over
            # in base_context from an earlier round must never leak into a
            # "no diagnostic feedback" comparison condition. PATCH_BLOCK's
            # real producer (BlockPatchProducer) has no way to omit this
            # parameter at all (regenerate_blocks requires it) -- a neutral,
            # content-free placeholder is used instead of silently keeping
            # whatever text happened to already be in context.
            context.pop("error_message", None)
            if ref.kind == ArtifactKind.BLOCK:
                context["error_message"] = "retry"

        production, cost, rejected = self._call_producer(
            producer, ref.artifact_id, context, stage=f"repair:{action.action_type.value}"
        )
        if rejected:
            return (
                RegeneratedArtifactRecord(pre_repair_ref=ref, production=production, outcome=RepairOutcome.PRODUCTION_FAILED),
                cost,
                True,
            )
        if production.status == ProductionStatus.FAILED:
            return (
                RegeneratedArtifactRecord(pre_repair_ref=ref, production=production, outcome=RepairOutcome.PRODUCTION_FAILED),
                cost,
                False,
            )

        # The real inputs this call actually used, per the Producer itself
        # -- falling back to our own stated intention only when the
        # Producer did not report anything at all (``None``, not ``[]``;
        # see ProductionResult's docstring for why those two are distinct).
        real_inputs = production.read_input_versions if production.read_input_versions is not None else intended_inputs
        add_result = self._store.add_version(
            ref.artifact_id,
            ref.kind,
            content=production.content,
            read_input_versions=real_inputs,
            produced_by=f"repair:{action.action_type.value}",
        )
        self._carry_forward_inferred_edges(ref, add_result.ref)

        reverify_evidence: List[Evidence] = []
        budget_exhausted = False
        constraints = self._constraints_for(add_result.ref)
        if constraints:
            reverify_evidence, rejected = self._check(add_result.ref, constraints)
            budget_exhausted = rejected

        classification = _classify_evidence(reverify_evidence)
        outcome = {
            "FAIL": RepairOutcome.STILL_FAILING,
            "PASS": RepairOutcome.RESOLVED,
            "UNKNOWN": RepairOutcome.INCONCLUSIVE,
        }[classification]

        return (
            RegeneratedArtifactRecord(
                pre_repair_ref=ref,
                production=production,
                outcome=outcome,
                new_ref=add_result.ref,
                reverify_evidence=reverify_evidence,
            ),
            cost,
            budget_exhausted,
        )

    def _reaffirm(self, ref: ArtifactRef, evidence: List[Evidence]) -> ArtifactRef:
        """Re-confirm ``ref`` is still valid without changing its content --
        a same-content ``add_version`` call with every originally-declared
        input re-pinned to its current latest version, which Stage 2's
        existing outdated/stale-input logic then correctly reads as
        genuinely fresh (or correctly still-stale, if some upstream is
        itself unresolved) -- no new store primitive needed.

        Two things a review found missing from the first pass, both fixed
        here: (a) any INFERRED edge that pointed at the OLD version is
        carried forward onto the new one (``_carry_forward_inferred_edges``)
        -- otherwise the dependency graph silently forgets a relationship
        the moment a reaffirmed artifact's version number moves; (b) the
        PASS evidence that justified this reaffirmation is re-recorded
        bound to the NEW ref, not left attached only to the old, now-
        superseded version -- otherwise ``live_evidence()`` on the new
        version comes back empty even though the (unchanged) content was
        just confirmed valid.
        """
        stored = self._store.get_version(ref.artifact_id, ref.version)
        previous_inputs = stored.ref.read_input_versions if stored is not None else []
        repinned_inputs = [self._repin(i) for i in previous_inputs]
        add_result = self._store.add_version(
            ref.artifact_id,
            ref.kind,
            content=stored.content if stored is not None else None,
            read_input_versions=repinned_inputs,
            produced_by="reaffirmed_after_reverify",
        )
        self._carry_forward_inferred_edges(ref, add_result.ref)
        for item in evidence:
            rebound = item.model_copy(update={"checked_artifact": add_result.ref})
            self._store.record_evidence(rebound)
        return add_result.ref

    def _carry_forward_inferred_edges(self, old_ref: ArtifactRef, new_ref: ArtifactRef) -> None:
        """Every INFERRED edge whose ``to_artifact`` named exactly
        ``old_ref`` is re-declared pointing at ``new_ref`` instead --
        ``add_version`` only ever auto-creates EXPLICIT edges from
        ``read_input_versions``, so an INFERRED relationship (added
        out-of-band via ``add_inferred_edge``) would otherwise simply stop
        applying the moment the artifact it points at gets superseded by a
        new version, for *any* reason (reaffirm here, or a real
        regeneration). ``from_artifact`` is re-pinned to its own current
        version first, matching ``add_inferred_edge``'s own validation
        (a real, current ref) and its retroactive-invalidation semantics
        (if that upstream is itself outdated/stale, carrying the edge
        forward correctly re-flags the new version too -- the same
        behavior a fresh declaration of this relationship would have)."""
        for edge in self._store.edges:
            if edge.relation != DependencyRelationKind.INFERRED:
                continue
            if edge.to_artifact.artifact_id != old_ref.artifact_id or edge.to_artifact.version != old_ref.version:
                continue
            from_current = self._store.latest(edge.from_artifact.artifact_id)
            from_ref = from_current.ref if from_current is not None else edge.from_artifact
            self._store.add_inferred_edge(
                from_ref, new_ref, reason=f"carried forward from v{old_ref.version}: {edge.reason}"
            )

    def _repin(self, ref: ArtifactRef) -> ArtifactRef:
        """``ref``'s artifact_id, re-pointed at its CURRENT latest version."""
        current = self._store.latest(ref.artifact_id)
        return current.ref if current is not None else ref

    @staticmethod
    def _combine_outcomes(
        records: List[RegeneratedArtifactRecord], left_stale: List[ArtifactRef], budget_exhausted: bool
    ) -> RepairOutcome:
        """Worst-first. ``budget_exhausted``/non-empty ``left_stale`` both
        mean "this execution did not reach a clean final state" and
        outrank everything else -- a caller reading only
        ``overall_outcome`` must never see RESOLVED when something affected
        was left unresolved for any reason, including simply running out of
        room to look at it."""
        if budget_exhausted or left_stale:
            return RepairOutcome.INCONCLUSIVE
        if any(r.outcome == RepairOutcome.STILL_FAILING for r in records):
            return RepairOutcome.STILL_FAILING
        if any(r.outcome == RepairOutcome.PRODUCTION_FAILED for r in records):
            return RepairOutcome.PRODUCTION_FAILED
        if any(r.outcome == RepairOutcome.NO_PRODUCER for r in records):
            return RepairOutcome.NO_PRODUCER
        if any(r.outcome == RepairOutcome.INCONCLUSIVE for r in records):
            return RepairOutcome.INCONCLUSIVE
        return RepairOutcome.RESOLVED


class FixedGraphAwareRepairPolicy:
    """The "fixed graph-aware" comparison group
    (``docs/stage1_experiment_design.md`` §3) -- picks the narrowest scope
    structurally justified by a failing artifact's own kind and immediate
    fan-out in ``dependency_edges``, rather than either a fixed single-block
    rule that never looks at the dependency graph at all
    (``adapters.TodayFixedRepairPolicyAdapter``) or always escalating to the
    whole document (``AlwaysRegenerateDocumentPolicy``).

    **Decision rule**, applied independently per distinct failing artifact
    -- one ``RepairAction`` per failing artifact_id in a single ``decide()``
    call, not capped at one total. (Unlike ``TodayFixedRepairPolicyAdapter``,
    which deliberately mirrors a real pipeline method that only ever acts on
    one location per round; this policy has no such real method to mirror,
    so nothing requires it to artificially under-batch. A caller must
    execute *every* returned action -- see ``branching_runner.py``'s own
    review-fixed history for why executing only the first silently drops
    independent failures.)

    1. ``kind == BLOCK`` -> ``PATCH_BLOCK``, target = the failing block
       alone.
    2. ``kind == SECTION`` -> ``REGENERATE_SECTION``, target = the failing
       section alone.
    3. Anything else, with **more than one** direct dependent found in
       ``dependency_edges`` (an edge whose ``from_artifact.artifact_id``
       equals this failing id) -> ``REPLAN_SUBGRAPH``, target = the failing
       artifact plus every direct dependent named in ``dependency_edges`` --
       a *shallow*, one-hop seed set (direction 1's own example: "정의를
       수정하고 관련 정리·예제를 재검사한다"). Deeper propagation beyond this
       seed is not this policy's job -- that is
       ``RepairOperationExecutor``'s cascade, computed against the live
       graph rather than this possibly-partial edge list.
    4. Anything else, with at most one direct dependent -> ``REGENERATE_
       DOCUMENT`` targeting ``document_ref``, as the bounded fallback for a
       failing artifact with no narrower scope this policy can structurally
       justify (no per-kind producer, no useful local fan-out signal). Note
       that once chosen, this scope is carried out by the executor's
       *unconditional* document-regeneration path, same as
       ``AlwaysRegenerateDocumentPolicy``'s own choice -- the two baselines
       differ in how *often* they reach for this scope, not in what
       choosing it does once chosen.

    ``use_diagnostic_feedback`` is always ``True`` (``feedback_text`` set to
    the triggering ``Evidence.detail``) -- this policy's comparison purpose
    is isolating *scope*; ``adapters.BlindRetryRepairPolicy`` wraps any
    policy, this one included, to isolate feedback separately.

    ``document_ref`` (constructor-only) names the artifact used for case 4's
    ``REGENERATE_DOCUMENT`` target -- this policy has no way to discover
    "which artifact is the document" from ``evidence``/``dependency_edges``
    alone, since a failing leaf artifact's own data says nothing about the
    document root above it.
    """

    def __init__(self, document_ref: ArtifactRef) -> None:
        self._document_ref = document_ref

    def decide(
        self, evidence: List[Evidence], dependency_edges: List[Any], budget: Budget
    ) -> List[RepairAction]:
        if budget.remaining <= 0:
            return []

        failing_by_artifact: Dict[str, Evidence] = {}
        for item in evidence:
            if item.outcome != CheckOutcome.FAIL:
                continue
            failing_by_artifact.setdefault(item.checked_artifact.artifact_id, item)

        actions: List[RepairAction] = []
        for item in failing_by_artifact.values():
            failing_ref = item.checked_artifact
            action_type, targets = self._scope_for(failing_ref, dependency_edges)
            actions.append(
                RepairAction(
                    action_type=action_type,
                    target_artifacts=targets,
                    use_diagnostic_feedback=True,
                    feedback_text=item.detail,
                    triggering_constraint_id=item.constraint_id,
                )
            )
        return actions

    def _scope_for(
        self, failing_ref: ArtifactRef, dependency_edges: List[Any]
    ) -> Tuple[RepairActionType, List[ArtifactRef]]:
        if failing_ref.kind == ArtifactKind.BLOCK:
            return RepairActionType.PATCH_BLOCK, [failing_ref]
        if failing_ref.kind == ArtifactKind.SECTION:
            return RepairActionType.REGENERATE_SECTION, [failing_ref]

        direct_dependents: Dict[str, ArtifactRef] = {}
        for edge in dependency_edges:
            from_artifact = getattr(edge, "from_artifact", None)
            to_artifact = getattr(edge, "to_artifact", None)
            if from_artifact is None or to_artifact is None:
                continue
            if from_artifact.artifact_id == failing_ref.artifact_id:
                direct_dependents[to_artifact.artifact_id] = to_artifact

        if len(direct_dependents) > 1:
            return RepairActionType.REPLAN_SUBGRAPH, [failing_ref, *direct_dependents.values()]
        return RepairActionType.REGENERATE_DOCUMENT, [self._document_ref]


class AlwaysRegenerateDocumentPolicy:
    """The "whole-document regeneration" comparison group
    (``docs/stage1_experiment_design.md`` §3): any FAIL at all, anywhere,
    always escalates to ``REGENERATE_DOCUMENT`` targeting ``document_ref`` --
    deliberately blind to kind, fan-out, or how many distinct artifacts
    actually failed. Carried out by
    ``RepairOperationExecutor._execute_unconditional`` -- every artifact in
    the document's impact set, force-regenerated, no selective skip.

    This is also the scope half of "final-only verification"
    (``branching_runner.BranchingRunner``'s ``check_timing`` parameter
    supplies the other half): the design doc's comparison table lists these
    as two separate rows sharing the identical "항상 REGENERATE_DOCUMENT" /
    "whole-document" scope rule, differing only in *when*
    ``Validator.check()`` is ever called -- continuously, vs. once on the
    finished document. That timing decision belongs to whatever drives the
    episode (the branching runner here; a controller from Stage 5 on), not
    to this policy, which has no notion of "when" at all -- it only ever
    reacts to whatever ``evidence`` it is handed.
    """

    def __init__(self, document_ref: ArtifactRef) -> None:
        self._document_ref = document_ref

    def decide(
        self, evidence: List[Evidence], dependency_edges: List[Any], budget: Budget
    ) -> List[RepairAction]:
        if budget.remaining <= 0:
            return []
        if not any(item.outcome == CheckOutcome.FAIL for item in evidence):
            return []
        return [
            RepairAction(
                action_type=RepairActionType.REGENERATE_DOCUMENT,
                target_artifacts=[self._document_ref],
                use_diagnostic_feedback=False,
                feedback_text=None,
            )
        ]


__all__ = [
    "ProducerRegistry",
    "ConstraintProvider",
    "RepairOperationExecutor",
    "FixedGraphAwareRepairPolicy",
    "AlwaysRegenerateDocumentPolicy",
]

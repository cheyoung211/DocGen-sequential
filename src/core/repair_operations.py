"""Repair operations and fixed repair policies (Stage 4 of the "Revision
2026.10.15" plan: "4단계. 고정 복구 정책과 분기 재실행").

Builds on Stage 1's ``Producer``/``RepairPolicy`` Protocols, Stage 2's
``ArtifactGraphStore`` (versioning, ``impact_candidates``, staleness), and
Stage 3's ``Constraint``/``Validator`` machinery. Nothing here is wired into
``scripts/run_full_pipeline.py`` or any agent -- per the user's explicit
scope decision for this stage, this module is proven correct with
Protocol-conforming fakes/stubs (``test/core/test_repair_operations.py``),
the same discipline Stages 1-3 already used for anything that would
otherwise require a real, expensive LLM call in a test. Actually running
this against the real pipeline with the local Qwen3.6 model (producing a
genuine small natural-failure/injected-failure pilot) is an explicitly
deferred follow-up, not part of this stage's own code.

Two components:

- ``RepairOperationExecutor`` carries out one ``RepairAction`` -- the
  Stage 4 checklist's "block patch, section regeneration, subgraph replan,
  whole-document regeneration 연산을 구현한다" -- and, in the same pass,
  cascades through everything the repair could plausibly affect, re-
  verifying each one fresh and regenerating only what still fails
  ("수정 이후 영향받는 artifact를 재검사하고, 필요한 항목만 재생성하는 고정
  graph-aware 정책을 구현한다"). Both checklist bullets are one mechanism
  here, not two: the *scope* of what gets directly targeted is a
  ``RepairPolicy``'s decision (below); *how far re-verification/selective
  regeneration then propagates from that scope* is this executor's job,
  uniformly, regardless of which of the four action types was chosen.

- Two concrete ``RepairPolicy`` implementations fill in named rows from
  ``docs/stage1_experiment_design.md``'s 7-comparison-group table that were
  explicitly assigned to this stage: ``FixedGraphAwareRepairPolicy``
  ("fixed graph-aware") and ``AlwaysRegenerateDocumentPolicy``
  ("whole-document regeneration" -- and, combined with
  ``branching_runner.BranchingRunner``'s ``check_timing`` option, also
  "final-only verification", which that table defines as the *same* scope
  rule under a different trigger-timing condition, not a different
  policy).

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

from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from src.core.artifact_graph import ArtifactGraphStore
from src.core.cost_tracking import CostTrackingProducer
from src.core.interfaces import Producer, Validator
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    CheckOutcome,
    Constraint,
    CostRecord,
    Evidence,
    ProductionResult,
    ProductionStatus,
    RegeneratedArtifactRecord,
    RepairAction,
    RepairActionType,
    RepairExecutionResult,
    RepairOutcome,
)

ProducerRegistry = Dict[ArtifactKind, Producer]
"""Which real (or fake, in tests) ``Producer`` regenerates an artifact of a
given kind. Deliberately a plain dict, not a class -- a missing entry *is*
the ``RepairOutcome.NO_PRODUCER`` signal, so there is nothing for a wrapper
class to add beyond what ``dict.get`` already gives for free."""

ConstraintProvider = Callable[[ArtifactRef], List[Constraint]]
"""Given a live artifact ref, which ``Constraint``s should be checked
against it right now. A plain callable, supplied by the caller -- neither
``ArtifactGraphStore`` nor ``Validator`` tracks "which constraints apply to
which artifact" as its own state (consistent with Stage 3's
``CheckCall.constraints`` always being caller-supplied, never looked up
internally). Returning an empty list means "nothing to check this artifact
against": ``RepairOperationExecutor`` leaves such an artifact untouched --
neither regenerated nor reaffirmed -- rather than fabricating a verdict for
it either way."""


class RepairOperationExecutor:
    """Carries out one ``RepairAction`` against a live ``ArtifactGraphStore``.

    All four ``RepairActionType`` values go through exactly the same
    mechanism; they differ only in which artifact(s)
    ``action.target_artifacts`` names up front (one block, one section, an
    explicit multi-artifact subgraph seed, or the document). This executor
    does not know or care which policy produced ``action`` -- see
    ``FixedGraphAwareRepairPolicy``/``AlwaysRegenerateDocumentPolicy`` for
    where that choice is actually made.

    **The cascade.** Every directly-named target is regenerated (its real
    ``Producer`` is called; on success, ``ArtifactGraphStore.add_version``
    persists the result). From each artifact actually written (regenerated
    *or* reaffirmed, see below), ``ArtifactGraphStore.impact_candidates`` is
    consulted for the conservative ``I(R) = R ∪ Desc_G(R)`` candidate set
    Stage 2 already computes. Each candidate is **freshly re-checked**
    (``Validator.check()`` -- never a stale/cached verdict): a FAIL queues
    it for regeneration exactly like a directly-named target (expanding
    further in turn); a clean check instead **reaffirms** it, content
    unchanged (see ``_reaffirm``). Every artifact_id is resolved at most
    once per ``execute()`` call (``processed``), so this always terminates
    on a finite graph regardless of cycles.

    **Why reaffirm, not just invalidate-or-skip.** Direction 1's own worked
    example ("무관한 비교 표는 그대로 유지한다") implies a *stable* preserved
    state, not "not touched this round, but still flagged dirty forever."
    ``ArtifactGraphStore`` as built in Stage 2 only ever *sets*
    ``_stale_ids`` (``add_version``'s forward propagation,
    ``add_inferred_edge``'s retroactive invalidation) -- it has no
    operation that *clears* staleness without writing a new version.
    Rather than adding a new staleness-clearing primitive to already
    multiply-reviewed Stage 2 code (which would need its own full round of
    the same invariant scrutiny every other mechanism in that module has
    already been through -- see memory ``docgen_stage_review_rigor``),
    ``_reaffirm`` reuses ``add_version`` exactly as built: a new version
    with the *same* content, but with every originally-declared input
    **re-pinned to that input's current latest version** (``_repin``). Fed
    through ``add_version``'s existing outdated/stale-input logic, a
    candidate whose real inputs have now all caught up correctly comes back
    ``born_stale=False`` -- genuinely fresh, not merely asserted so -- while
    one whose *own* upstream is itself still unresolved correctly stays
    stale, exactly mirroring what a real regeneration's own ``add_version``
    call would conclude. No new store method, no second mechanism for the
    same fact.

    ``execute()`` performs at most one regeneration attempt per artifact --
    it never retries internally. A ``RegeneratedArtifactRecord`` whose fresh
    re-check still shows FAIL is reported as ``RepairOutcome.STILL_FAILING``,
    not silently retried; another attempt is a later round's business,
    driven by a caller re-invoking a ``RepairPolicy`` with fresh evidence
    (``branching_runner.BranchingRunner``), the same layering every earlier
    stage already uses (``RepairPolicy.decide()`` itself never loops either).
    """

    def __init__(
        self,
        producer_registry: ProducerRegistry,
        validator: Validator,
        constraint_provider: ConstraintProvider,
        base_context: Dict[str, Any],
    ) -> None:
        self._producers = producer_registry
        self._validator = validator
        self._constraints_for = constraint_provider
        self._base_context = base_context

    def execute(
        self,
        action: RepairAction,
        store: ArtifactGraphStore,
        *,
        triggering_evidence: Optional[List[Evidence]] = None,
    ) -> RepairExecutionResult:
        """Carry out ``action`` against ``store`` and return exactly what
        happened -- see this class's and ``RepairExecutionResult``'s
        docstrings. Logs the action itself via
        ``ArtifactGraphStore.record_action`` exactly once, at the start,
        regardless of how many artifacts its cascade goes on to touch: the
        thing that "began executing" is the one ``RepairAction``, not each
        individual regeneration inside its cascade.
        """
        store.record_action(action, triggering_evidence)

        regenerated: List[RegeneratedArtifactRecord] = []
        reverified_untouched: List[ArtifactRef] = []
        costs: List[CostRecord] = []
        processed: Set[str] = set()
        # (ref, already_known_failing) -- directly-named targets are
        # already known failing (the policy that produced `action` saw
        # FAIL evidence for them); anything discovered via cascade starts
        # unknown and must be freshly checked before either branch below.
        queue: Deque[Tuple[ArtifactRef, bool]] = deque(
            (ref, True) for ref in action.target_artifacts
        )

        while queue:
            seed_ref, known_failing = queue.popleft()
            if seed_ref.artifact_id in processed:
                continue
            current = store.latest(seed_ref.artifact_id)
            if current is None:
                continue  # defensive: a ref naming an artifact never stored
            live_ref = current.ref
            processed.add(live_ref.artifact_id)

            if known_failing:
                failing = True
            else:
                constraints = self._constraints_for(live_ref)
                if not constraints:
                    # Nothing asserts this candidate is broken, and nothing
                    # confirms it either -- leave it exactly as-is. This
                    # does not cut off reachability past it: impact_candidates
                    # below already returns the *full* transitive descendant
                    # set from each resolved artifact, not just one hop, so
                    # skipping this one node loses no downstream coverage.
                    continue
                fresh_evidence = self._validator.check(live_ref, constraints, self._base_context)
                for item in fresh_evidence:
                    store.record_evidence(item)
                failing = any(item.outcome == CheckOutcome.FAIL for item in fresh_evidence)

            if failing:
                record, cost = self._regenerate(live_ref, action, store)
                regenerated.append(record)
                if cost is not None:
                    costs.append(cost)
                if record.outcome in (RepairOutcome.PRODUCTION_FAILED, RepairOutcome.NO_PRODUCER):
                    continue  # nothing new was written -- nothing propagates from here
                resolved_ref = record.new_ref
                assert resolved_ref is not None  # guaranteed by RegeneratedArtifactRecord's own validation
            else:
                resolved_ref = self._reaffirm(live_ref, store)
                reverified_untouched.append(resolved_ref)

            for candidate_id in sorted(
                store.impact_candidates(resolved_ref.artifact_id) - {resolved_ref.artifact_id}
            ):
                if candidate_id in processed:
                    continue
                candidate_current = store.latest(candidate_id)
                if candidate_current is not None:
                    queue.append((candidate_current.ref, False))

        return RepairExecutionResult(
            action=action,
            regenerated=regenerated,
            reverified_untouched=reverified_untouched,
            overall_outcome=self._combine_outcomes(regenerated),
            cost=costs,
        )

    def _regenerate(
        self, ref: ArtifactRef, action: RepairAction, store: ArtifactGraphStore
    ) -> Tuple[RegeneratedArtifactRecord, Optional[CostRecord]]:
        """Returns the regeneration record plus the ``CostRecord`` actually
        incurred, if any -- populated only when ``producer_registry[ref.kind]``
        happens to be a ``CostTrackingProducer`` (the "common local/API
        model adapter" Stage 3 already built). A plain, untracked
        ``Producer`` (e.g. a test fake) simply yields no cost, exactly like
        calling an untracked ``Producer`` anywhere else in this codebase.
        Any cost found here is also persisted via
        ``ArtifactGraphStore.record_cost`` immediately -- mirroring
        ``ToolExecutor._charge``'s "every dispatched call is logged, here
        and nowhere else" discipline -- and handed back to the caller so
        ``branching_runner.BranchingRunner`` can actually charge its own
        ``Budget`` (a *measurement* that nothing charges against a budget
        is not an *enforcement* -- the exact Stage 3 review finding this
        mirrors, see ``docgen_stage_review_rigor`` pattern #15).
        """
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
            )

        context = dict(self._base_context)
        if action.use_diagnostic_feedback and action.feedback_text:
            context["error_message"] = action.feedback_text

        production = producer.produce(ref.artifact_id, context)
        cost = producer.last_cost if isinstance(producer, CostTrackingProducer) else None
        if cost is not None:
            store.record_cost(cost)

        if production.status == ProductionStatus.FAILED:
            return (
                RegeneratedArtifactRecord(
                    pre_repair_ref=ref, production=production, outcome=RepairOutcome.PRODUCTION_FAILED
                ),
                cost,
            )

        stored = store.get_version(ref.artifact_id, ref.version)
        previous_inputs = stored.ref.read_input_versions if stored is not None else []
        repinned_inputs = [self._repin(i, store) for i in previous_inputs]
        add_result = store.add_version(
            ref.artifact_id,
            ref.kind,
            content=None,
            read_input_versions=repinned_inputs,
            produced_by=f"repair:{action.action_type.value}",
        )

        reverify_evidence: List[Evidence] = []
        constraints = self._constraints_for(add_result.ref)
        if constraints:
            reverify_evidence = self._validator.check(add_result.ref, constraints, self._base_context)
            for item in reverify_evidence:
                store.record_evidence(item)
        still_failing = any(item.outcome == CheckOutcome.FAIL for item in reverify_evidence)

        return (
            RegeneratedArtifactRecord(
                pre_repair_ref=ref,
                production=production,
                outcome=RepairOutcome.STILL_FAILING if still_failing else RepairOutcome.RESOLVED,
                new_ref=add_result.ref,
                reverify_evidence=reverify_evidence,
            ),
            cost,
        )

    def _reaffirm(self, ref: ArtifactRef, store: ArtifactGraphStore) -> ArtifactRef:
        """Re-confirm ``ref`` is still valid without changing its content --
        see this class's own docstring, "why reaffirm" paragraph."""
        stored = store.get_version(ref.artifact_id, ref.version)
        previous_inputs = stored.ref.read_input_versions if stored is not None else []
        repinned_inputs = [self._repin(i, store) for i in previous_inputs]
        add_result = store.add_version(
            ref.artifact_id,
            ref.kind,
            content=stored.content if stored is not None else None,
            read_input_versions=repinned_inputs,
            produced_by="reaffirmed_after_reverify",
        )
        return add_result.ref

    def _repin(self, ref: ArtifactRef, store: ArtifactGraphStore) -> ArtifactRef:
        """``ref``'s artifact_id, re-pointed at its CURRENT latest version --
        the mechanism that lets a reaffirmation ``add_version`` call
        correctly come back fresh once its own inputs have caught up."""
        current = store.latest(ref.artifact_id)
        return current.ref if current is not None else ref

    @staticmethod
    def _combine_outcomes(records: List[RegeneratedArtifactRecord]) -> RepairOutcome:
        """Worst-first: any artifact this action touched that is still
        failing (or could not even be attempted) makes the whole action's
        outcome something other than a clean ``RESOLVED`` -- a caller
        reading only ``overall_outcome`` must never see RESOLVED if any
        individual regeneration inside the cascade did not actually
        resolve."""
        if any(r.outcome == RepairOutcome.STILL_FAILING for r in records):
            return RepairOutcome.STILL_FAILING
        if any(r.outcome == RepairOutcome.PRODUCTION_FAILED for r in records):
            return RepairOutcome.PRODUCTION_FAILED
        if any(r.outcome == RepairOutcome.NO_PRODUCER for r in records):
            return RepairOutcome.NO_PRODUCER
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
    so nothing requires it to artificially under-batch.)

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
       justify (no per-kind producer, no useful local fan-out signal). The
       executor's own cascade still re-verifies before regenerating
       anything further out from there -- this fallback is not a return to
       "blindly rebuild everything", the way ``AlwaysRegenerateDocumentPolicy``
       is by design.

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
    actually failed.

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

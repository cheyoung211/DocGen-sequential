"""Checkpoint-branch comparison harness (Stage 4 of the "Revision
2026.10.15" plan: "같은 checkpoint에서 여러 repair action을 실행하고 후속
정책을 고정해 비교하는 분기 runner를 만든다").

Operationalizes ``docs/stage1_experiment_design.md`` §5's checkpoint-branch
design, fixed back in Stage 1 and executed here for the first time:

  1. 복구 전 상태를 versioned checkpoint로 저장한다 -- the caller already has
     this (Stage 2's ``ArtifactGraphStore.save_checkpoint``); ``BranchingRunner``
     takes the ``Checkpoint`` as given, it does not create one.
  2. 동일 checkpoint에서 서로 다른 RepairPolicy로 분기한다 -- each
     ``BranchSpec`` gets its own independent store via
     ``ArtifactGraphStore.from_checkpoint`` with its own ``branch_id``.
     Nothing here lets one branch's execution read or write another
     branch's store.
  3. 각 분기의 후속 실행은 동일한 후속 정책으로 고정한다 -- every branch gets its
     own fresh instance of the *same rule* (``downstream_policy_factory()``,
     called once per branch), used identically from round 2 onward.
  4. 효과는 해당 checkpoint·행동·후속 정책 조건에만 한정해서 해석한다 -- a
     ``BranchExecutionResult`` keeps its own ``source_checkpoint_id`` and
     every round it actually ran.

**Revised after a user review found 3 real bugs affecting this module**
(full detail in memory ``docgen_stage_review_rigor``):

5. **Every action a round's ``decide()`` call returns is now executed, not
   just the first.** The original version took ``decisions[0]`` and
   silently discarded the rest -- two independent failures named by the
   same ``decide()`` call (A and B, no dependency between them) collapsed
   into "only A was ever attempted, B's failure vanished from the record,"
   and a round where one action's cascade hit PRODUCTION_FAILED while
   another's passed could misreport ``stopped_reason="no_more_failures"``.
   Fixed with an explicit ``pending`` map of still-failing artifact_ids,
   updated (never wholesale-replaced) after every executed action.
6. **Every real generation/check call is now routed through one
   ``ToolExecutor`` per branch**, including the ``check_timing="final_only"``
   one-shot check this class itself performs -- admission is checked
   before each call and cost is charged against the branch's own budget
   exactly once, in Stage 3's already-reviewed enforcement point, instead
   of this module summing costs after the fact (which let a whole round's
   worth of generation calls run uncapped between budget checks).
7. **Each branch gets its own independent context**
   (``context_factory(spec)``, called once per branch, mirroring
   ``budget_factory``/``downstream_policy_factory``) -- the original
   version handed every branch the *same* ``context`` dict (and
   ``RepairOperationExecutor`` only ever shallow-copied it), so a mutable
   object inside it (e.g. a real ``DocumentGraph`` a ``Producer`` mutates
   as a side effect) was silently shared across branches, breaking the
   "same initial state" guarantee the whole point of branching depends on.

**Revised a second time after a user review found 2 more real bugs
affecting this module** (full detail in memory ``docgen_stage_review_rigor``):

8. **A sub-failure a round's cascade discovers for the first time (never
   part of that round's starting ``pending``) no longer vanishes.** Real
   repro: only ``a`` is reported failing; repairing ``a`` cascades into a
   downstream ``b`` whose production then fails. ``b`` has no FAIL
   Evidence to re-enter ``pending`` with (production failures/left-stale
   candidates carry none), so the round ended with an empty ``pending``
   and ``stopped_reason="no_more_failures"`` even though ``b`` was left
   genuinely broken -- the same misreport for ``b`` classified UNKNOWN or
   with no constraints to check it against at all. Fixed with a second,
   persistent ``unresolved_ids`` set (``_apply_round_to_pending``) that
   accumulates every artifact_id any round ever reports
   PRODUCTION_FAILED/NO_PRODUCER/left_stale and has not since reported
   RESOLVED or reaffirmed PASS -- purely bookkeeping for
   ``_finalize_stopped_reason`` to pick an honest label with, never
   Evidence fed into ``decide()`` (UNKNOWN is still never promoted to
   FAIL, per the review's own explicit note).
9. ``check_timing="final_only"``'s one-shot initial check no longer
   treats every rejection as budget exhaustion -- the same
   ``schemas.BUDGET_ERROR_CODES`` distinction ``repair_operations.
   RepairOperationExecutor._check()`` needed for the identical reason
   (see that module's own docstring, point 10): a ``Validator.check()``
   that raises is ``EXECUTION_FAILED``, not a budget code, even with
   plenty of budget left.

Nothing here is wired into ``scripts/run_full_pipeline.py`` -- same scope
decision as ``repair_operations.py`` (proven with Protocol-conforming
fakes this stage; a real small-scale pilot against the local Qwen3.6
pipeline is a deferred follow-up, not part of this stage's own code).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set

from src.core.artifact_graph import ArtifactGraphStore
from src.core.interfaces import RepairPolicy, Validator
from src.core.repair_operations import ConstraintProvider, ProducerRegistry, RepairOperationExecutor
from src.core.schemas import (
    ArtifactRef,
    Budget,
    BranchComparisonResult,
    BranchExecutionResult,
    BUDGET_ERROR_CODES,
    CheckCall,
    CheckOutcome,
    Checkpoint,
    Constraint,
    Evidence,
    RegeneratedArtifactRecord,
    RepairExecutionResult,
    RepairOutcome,
    RepairRound,
)
from src.core.tool_executor import ToolExecutor


class BranchSpec(NamedTuple):
    """One branch's distinguishing choice: a label (becomes the restored
    store's ``branch_id`` and the result's ``policy_label``) and the
    ``RepairPolicy`` that decides *only* this branch's initial round --
    every round after that uses the one shared downstream policy."""

    label: str
    policy: RepairPolicy


class BranchingRunner:
    """Branches one shared checkpoint into several independent episodes,
    one per initial ``RepairPolicy`` (``BranchSpec``), each then continuing
    under one shared fixed downstream policy. See this module's docstring
    for the §5 design points this operationalizes and the 3 bugs a review
    found and fixed.

    ``check_timing`` controls only how each branch's **initial** round is
    decided:

    - ``"continuous"``: the initial ``decide()`` call for every branch uses
      ``triggering_evidence`` exactly as given.
    - ``"final_only"``: ``triggering_evidence`` is ignored; this runner
      performs exactly one fresh check (routed through that branch's own
      ``ToolExecutor``, same as every other check -- point 6 above) against
      ``final_check_target``/``final_check_constraints`` and decides from
      that instead.

    Both a fresh ``Validator`` (``validator_factory(store)``) and a fresh
    ``ToolExecutor``/``RepairOperationExecutor`` pair are built per branch,
    bound to that branch's own restored store -- a ``Validator`` like
    ``constraint_checks.BasicConstraintChecker`` binds to one specific
    store at construction time, so sharing one across branches would read/
    write the *original* store regardless of which branch's restored store
    everything else was correctly operating on.
    """

    def __init__(
        self,
        producer_registry: ProducerRegistry,
        validator_factory: Callable[[ArtifactGraphStore], Validator],
        constraint_provider: ConstraintProvider,
        *,
        model_name: str = "unknown",
        max_actions_per_branch: int = 1000,
        max_downstream_rounds: int = 5,
    ) -> None:
        self._producers = producer_registry
        self._validator_factory = validator_factory
        self._constraints_for = constraint_provider
        self._model_name = model_name
        self._max_actions_per_branch = max_actions_per_branch
        self._max_downstream_rounds = max_downstream_rounds

    def run_branches(
        self,
        checkpoint: Checkpoint,
        branch_specs: List[BranchSpec],
        downstream_policy_factory: Callable[[], RepairPolicy],
        *,
        triggering_evidence: List[Evidence],
        budget_factory: Callable[[], Budget],
        context_factory: Callable[[BranchSpec], Dict[str, Any]] = lambda spec: {},
        check_timing: str = "continuous",
        final_check_target: Optional[ArtifactRef] = None,
        final_check_constraints: Optional[List[Constraint]] = None,
    ) -> BranchComparisonResult:
        """Run every ``BranchSpec`` from the same ``checkpoint`` and return
        all of their results together.

        ``downstream_policy_factory``/``budget_factory``/``context_factory``
        are each called once *per branch*, never reused across branches --
        see this class's own docstring, points 3 and 7, for why a shared
        instance of any of them would let one branch's state leak into
        another's.
        """
        if check_timing not in ("continuous", "final_only"):
            raise ValueError(f"check_timing must be 'continuous' or 'final_only', got {check_timing!r}")
        if check_timing == "final_only" and (final_check_target is None or final_check_constraints is None):
            raise ValueError(
                "check_timing='final_only' requires both final_check_target and final_check_constraints"
            )

        branches = [
            self._run_one_branch(
                checkpoint,
                spec,
                downstream_policy_factory(),
                triggering_evidence=triggering_evidence,
                budget=budget_factory(),
                context=context_factory(spec),
                check_timing=check_timing,
                final_check_target=final_check_target,
                final_check_constraints=final_check_constraints,
            )
            for spec in branch_specs
        ]
        return BranchComparisonResult(source_checkpoint_id=checkpoint.checkpoint_id, branches=branches)

    def _run_one_branch(
        self,
        checkpoint: Checkpoint,
        spec: BranchSpec,
        downstream_policy: RepairPolicy,
        *,
        triggering_evidence: List[Evidence],
        budget: Budget,
        context: Dict[str, Any],
        check_timing: str,
        final_check_target: Optional[ArtifactRef],
        final_check_constraints: Optional[List[Constraint]],
    ) -> BranchExecutionResult:
        store = ArtifactGraphStore.from_checkpoint(checkpoint, branch_id=spec.label)
        validator = self._validator_factory(store)
        tool_executor = ToolExecutor(store, validator, budget=budget, max_actions=self._max_actions_per_branch)
        executor = RepairOperationExecutor(
            store, tool_executor, self._producers, self._constraints_for, context, model_name=self._model_name
        )

        rounds: List[RepairRound] = []
        stopped_reason = "no_more_failures"

        if check_timing == "continuous":
            current_evidence = list(triggering_evidence)
        else:
            assert final_check_target is not None and final_check_constraints is not None
            current = store.latest(final_check_target.artifact_id)
            live_ref = current.ref if current is not None else final_check_target
            observation = tool_executor.execute(
                CheckCall(artifact=live_ref, constraints=final_check_constraints, context=dict(context))
            )
            if observation.ok:
                assert observation.output is not None
                current_evidence = observation.output.evidence
            else:
                # Not every rejection here means budget/max-actions/
                # suspended -- the wrapped Validator.check() can itself
                # raise (surfaced as ToolErrorCode.EXECUTION_FAILED), which
                # a review found this branch previously misreported as
                # "budget_exhausted" even with plenty of budget left and
                # the executor not suspended. Only schemas.BUDGET_ERROR_CODES
                # membership earns that label; anything else is its own,
                # honestly distinct reason -- still no round ever starts
                # either way, since there is nothing to decide from.
                assert observation.error is not None
                reason = "budget_exhausted" if observation.error.code in BUDGET_ERROR_CODES else "initial_check_failed"
                return BranchExecutionResult(
                    branch_id=spec.label,
                    source_checkpoint_id=checkpoint.checkpoint_id,
                    policy_label=spec.label,
                    rounds=[],
                    stopped_reason=reason,
                    final_checkpoint=store.save_checkpoint(label=f"{spec.label}-final"),
                    total_cost_tokens=tool_executor.budget.spent,
                )

        # Artifact_ids this branch has at some point seen reported
        # PRODUCTION_FAILED/NO_PRODUCER/left_stale and not since reported
        # RESOLVED or reaffirmed PASS -- see `_apply_round_to_pending`.
        # Purely a bookkeeping set for choosing an honest `stopped_reason`
        # below; never fed into `decide()` as fabricated Evidence (a
        # review's explicit note: UNKNOWN must never be promoted to FAIL).
        unresolved_ids: Set[str] = set()

        for round_index in range(self._max_downstream_rounds + 1):
            if tool_executor.is_suspended or tool_executor.budget.remaining <= 0:
                stopped_reason = "budget_exhausted"
                break

            policy = spec.policy if round_index == 0 else downstream_policy
            decisions = policy.decide(current_evidence, store.edges, tool_executor.budget)
            if not decisions:
                clean_reason = (
                    "initial_policy_returned_nothing" if round_index == 0 else "downstream_policy_returned_nothing"
                )
                stopped_reason = self._finalize_stopped_reason(clean_reason, unresolved_ids)
                break

            executions: List[RepairExecutionResult] = []
            pending: Dict[str, Evidence] = {
                item.checked_artifact.artifact_id: item
                for item in current_evidence
                if item.outcome == CheckOutcome.FAIL
            }
            round_budget_exhausted = False
            for action in decisions:
                if tool_executor.is_suspended or tool_executor.budget.remaining <= 0:
                    round_budget_exhausted = True
                    break
                result = executor.execute(action, triggering_evidence=current_evidence)
                executions.append(result)
                self._apply_round_to_pending(pending, unresolved_ids, result)
                if result.budget_exhausted:
                    round_budget_exhausted = True
                    break

            rounds.append(
                RepairRound(
                    round_index=round_index,
                    policy_label=spec.label if round_index == 0 else "downstream",
                    executions=executions,
                )
            )

            if round_budget_exhausted or tool_executor.is_suspended or tool_executor.budget.remaining <= 0:
                stopped_reason = "budget_exhausted"
                break

            current_evidence = list(pending.values())
            if not any(item.outcome == CheckOutcome.FAIL for item in current_evidence):
                stopped_reason = self._finalize_stopped_reason("no_more_failures", unresolved_ids)
                break
        else:
            stopped_reason = "max_rounds_reached"

        return BranchExecutionResult(
            branch_id=spec.label,
            source_checkpoint_id=checkpoint.checkpoint_id,
            policy_label=spec.label,
            rounds=rounds,
            stopped_reason=stopped_reason,
            final_checkpoint=store.save_checkpoint(label=f"{spec.label}-final"),
            total_cost_tokens=tool_executor.budget.spent,
        )

    @staticmethod
    def _apply_round_to_pending(
        pending: Dict[str, Evidence], unresolved_ids: Set[str], result: RepairExecutionResult
    ) -> None:
        """Updates ``pending`` (artifact_id -> its current FAIL Evidence,
        fed back into the next round's ``decide()`` call) and
        ``unresolved_ids`` (artifact_id -> "not confirmed resolved",
        bookkeeping only for ``stopped_reason``, never fed into
        ``decide()``) from one executed action's result -- never
        wholesale-replaces either. This is the point 5 fix (``pending``)
        plus a later review's fix (``unresolved_ids``): an artifact nobody's
        executed action touched this round simply stays as it was in both,
        and a sub-failure DISCOVERED only by this action's own cascade
        (never part of this round's starting ``pending`` -- e.g. a
        downstream artifact a repair's cascade newly finds broken, with no
        FAIL Evidence available to feed back for a production failure or a
        left-stale/UNKNOWN candidate) still has to leave some trace,
        exactly what ``unresolved_ids`` exists for.

        Reads ``result.final_outcome_by_artifact`` -- never
        ``result.regenerated``/``.reverified_untouched`` directly. A
        review found this method used to process ``regenerated`` in full,
        then unconditionally process ``reverified_untouched`` afterward,
        regardless of which one actually happened LAST for a given
        artifact_id within a multi-sweep execution -- so an early PASS
        reaffirm could pop a later, truer FAIL back out of ``pending``
        (and vice versa: an early FAIL regeneration record could outlive a
        later PASS). ``final_outcome_by_artifact`` is already resolved to
        the one current outcome per artifact_id by the executor itself, in
        true execution order, so this method no longer needs -- or is
        allowed -- to re-derive its own chronology from the two raw lists.
        """
        latest_record_by_id: Dict[str, RegeneratedArtifactRecord] = {}
        for record in result.regenerated:
            latest_record_by_id[record.pre_repair_ref.artifact_id] = record

        for artifact_id, outcome in result.final_outcome_by_artifact.items():
            if outcome == RepairOutcome.RESOLVED:
                pending.pop(artifact_id, None)
                unresolved_ids.discard(artifact_id)
            elif outcome in (RepairOutcome.STILL_FAILING, RepairOutcome.INCONCLUSIVE):
                unresolved_ids.add(artifact_id)
                record = latest_record_by_id.get(artifact_id)
                fail_evidence = (
                    next((item for item in record.reverify_evidence if item.outcome == CheckOutcome.FAIL), None)
                    if record is not None
                    else None
                )
                if fail_evidence is not None:
                    pending[artifact_id] = fail_evidence
                # else: INCONCLUSIVE with no FAIL evidence at all (or this
                # artifact_id's current outcome came from a reaffirm, which
                # has no record at all) -- leave whatever was already
                # pending for it (if anything) unchanged; there is nothing
                # new, honest to replace it with. Marking it unresolved
                # above is bookkeeping, not an Evidence fabrication -- it
                # never reaches `decide()`.
            else:  # PRODUCTION_FAILED / NO_PRODUCER
                unresolved_ids.add(artifact_id)
                # `pending` itself deliberately left untouched -- the
                # underlying problem was never actually re-examined, so
                # whatever was already pending for it (the original
                # trigger, if any) remains exactly as it was.
        for ref in result.left_stale:
            unresolved_ids.add(ref.artifact_id)
            # No Evidence to add to `pending` -- these remain visible via
            # RepairExecutionResult.left_stale itself, which every round's
            # `executions` keeps, rather than being fabricated into a FAIL
            # this method never actually observed.

    @staticmethod
    def _finalize_stopped_reason(clean_reason: str, unresolved_ids: Set[str]) -> str:
        """``clean_reason`` ("no_more_failures"/"*_policy_returned_nothing")
        is only actually true if nothing was left unresolved along the
        way. A review found both reasons reported even when a sub-failure
        a cascade discovered mid-run was never part of ``pending`` at all
        (a production failure, or a left-stale/UNKNOWN candidate) -- such a
        failure simply has no FAIL Evidence to re-enter ``pending`` with,
        so ``current_evidence`` goes empty and these reasons fire as if
        everything were actually fine. This never promotes UNKNOWN into a
        FAIL fed back into ``decide()`` (an explicit review note) -- it
        only changes which *label* describes a stop that was already about
        to happen, based on ``unresolved_ids`` (see
        ``_apply_round_to_pending``)."""
        return "unresolved_remaining" if unresolved_ids else clean_reason


__all__ = ["BranchSpec", "BranchingRunner"]

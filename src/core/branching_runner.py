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
     called once per branch), used identically from round 2 onward, so only
     round 1 (what each ``BranchSpec`` actually varies) differs between
     branches. A factory, not one shared instance: a stateful policy (e.g.
     ``adapters.BlindRetryRepairPolicy``, which tracks a per-artifact retry
     count across calls) would otherwise leak state across branches that
     are supposed to be completely independent -- round 2 of branch A
     incrementing a retry counter that round 2 of branch B then
     unexpectedly inherits. ``ArtifactGraphStore.from_checkpoint`` already
     guarantees this independence for store state; the policy and the
     budget (see ``budget_factory`` below) need the same guarantee
     explicitly, since neither lives inside the checkpoint.
  4. 효과는 해당 checkpoint·행동·후속 정책 조건에만 한정해서 해석한다 -- a
     ``BranchExecutionResult`` keeps its own ``source_checkpoint_id`` and
     every round it actually ran, so a reader is never tempted to read a
     result past the specific (checkpoint, initial action, downstream
     policy) combination it came from.

Nothing here is wired into ``scripts/run_full_pipeline.py`` -- same scope
decision as ``repair_operations.py`` (proven with Protocol-conforming
fakes this stage; a real small-scale pilot against the local Qwen3.6
pipeline is a deferred follow-up, not part of this stage's own code).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Optional

from src.core.artifact_graph import ArtifactGraphStore
from src.core.interfaces import RepairPolicy, Validator
from src.core.repair_operations import ConstraintProvider, ProducerRegistry, RepairOperationExecutor
from src.core.schemas import (
    ArtifactRef,
    Budget,
    BranchComparisonResult,
    BranchExecutionResult,
    CheckOutcome,
    Checkpoint,
    Constraint,
    Evidence,
    RepairAction,
    RepairExecutionResult,
    RepairOutcome,
)


class BranchSpec(NamedTuple):
    """One branch's distinguishing choice: a label (becomes the restored
    store's ``branch_id`` and the result's ``policy_label``) and the
    ``RepairPolicy`` that decides *only* this branch's initial round --
    every round after that uses the one shared ``downstream_policy`` passed
    to ``BranchingRunner.run_branches``, not this policy again."""

    label: str
    policy: RepairPolicy


class BranchingRunner:
    """Branches one shared checkpoint into several independent episodes,
    one per initial ``RepairPolicy`` (``BranchSpec``), each then continuing
    under one shared fixed downstream policy. See this module's docstring
    for the §5 design points this operationalizes.

    ``check_timing`` controls only how each branch's **initial** round is
    decided -- the one axis with a Stage-4-representable "when" distinction
    absent a real controller/orchestration loop (Stage 5):

    - ``"continuous"``: the initial ``decide()`` call for every branch uses
      ``triggering_evidence`` exactly as given -- whatever was already
      discovered and is sitting on the checkpoint, acted on as soon as it
      is available (``docs/stage1_experiment_design.md``'s "whole-document
      regeneration" row: trigger=on).
    - ``"final_only"``: ``triggering_evidence`` is ignored for the initial
      decision; instead this runner performs exactly one fresh
      ``Validator.check()`` against ``final_check_target``/
      ``final_check_constraints`` and decides from *that* instead (the
      "final-only verification" row: trigger=off mid-generation, on only at
      a one-shot final check).

      Downstream rounds are unaffected by ``check_timing`` either way --
      each one is driven by the fresh evidence
      ``RepairOperationExecutor``'s own cascade already produces while
      executing the previous round (``RegeneratedArtifactRecord.
      reverify_evidence``), which has no real analogue of "defer checking"
      without inventing one; operationalizing *that* needs a real
      controller deciding when to look, which is explicitly Stage 5's job,
      not this harness's.

    **Why ``validator_factory``, not a single ``Validator`` instance.**
    ``Validator`` implementations are not all "pure" with respect to which
    store they read: ``constraint_checks.BasicConstraintChecker`` binds to
    one specific ``ArtifactGraphStore`` *at construction time*
    (``self._store = store``), mirroring ``tool_executor.ToolExecutor``'s
    own documented "one instance targets exactly one store" design. If
    ``BranchingRunner`` shared one such validator across every branch, it
    would keep reading/writing against whichever single store it was
    originally bound to -- silently ignoring each branch's own
    independently-restored store entirely, even though
    ``RepairOperationExecutor.execute()``'s *other* operations
    (``store.add_version``, ``store.latest``, ...) would correctly operate
    on the right branch. Building both the validator and the executor fresh
    per branch, from this branch's own restored store, is what keeps every
    part of one branch's execution looking at the same, correct store.
    ``producer_registry``/``constraint_provider`` do not have this problem
    (today's real producers bind to an agent, not a store; a constraint
    provider is a pure callable) and so stay shared across branches.
    """

    def __init__(
        self,
        producer_registry: ProducerRegistry,
        validator_factory: Callable[[ArtifactGraphStore], Validator],
        constraint_provider: ConstraintProvider,
        *,
        max_downstream_rounds: int = 5,
    ) -> None:
        self._producers = producer_registry
        self._validator_factory = validator_factory
        self._constraints_for = constraint_provider
        self._max_downstream_rounds = max_downstream_rounds

    def run_branches(
        self,
        checkpoint: Checkpoint,
        branch_specs: List[BranchSpec],
        downstream_policy_factory: Callable[[], RepairPolicy],
        *,
        triggering_evidence: List[Evidence],
        budget_factory: Callable[[], Budget],
        check_timing: str = "continuous",
        final_check_target: Optional[ArtifactRef] = None,
        final_check_constraints: Optional[List[Constraint]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> BranchComparisonResult:
        """Run every ``BranchSpec`` from the same ``checkpoint`` and return
        all of their results together.

        ``downstream_policy_factory``/``budget_factory`` are each called
        once *per branch* (never once total, never reused across branches)
        -- see this class's own docstring, point 3, for why a shared
        instance of either would let one branch's state leak into
        another's.
        """
        if check_timing not in ("continuous", "final_only"):
            raise ValueError(f"check_timing must be 'continuous' or 'final_only', got {check_timing!r}")
        if check_timing == "final_only" and (final_check_target is None or final_check_constraints is None):
            raise ValueError(
                "check_timing='final_only' requires both final_check_target and final_check_constraints"
            )

        resolved_context = context or {}
        branches = [
            self._run_one_branch(
                checkpoint,
                spec,
                downstream_policy_factory(),
                triggering_evidence=triggering_evidence,
                budget=budget_factory(),
                check_timing=check_timing,
                final_check_target=final_check_target,
                final_check_constraints=final_check_constraints,
                context=resolved_context,
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
        check_timing: str,
        final_check_target: Optional[ArtifactRef],
        final_check_constraints: Optional[List[Constraint]],
        context: Dict[str, Any],
    ) -> BranchExecutionResult:
        store = ArtifactGraphStore.from_checkpoint(checkpoint, branch_id=spec.label)
        # Both built fresh, bound to THIS branch's own store -- see this
        # class's docstring, "why validator_factory" paragraph.
        validator = self._validator_factory(store)
        executor = RepairOperationExecutor(self._producers, validator, self._constraints_for, context)

        if check_timing == "continuous":
            initial_evidence = list(triggering_evidence)
        else:
            assert final_check_target is not None and final_check_constraints is not None
            current = store.latest(final_check_target.artifact_id)
            live_ref = current.ref if current is not None else final_check_target
            initial_evidence = validator.check(live_ref, final_check_constraints, context)
            for item in initial_evidence:
                store.record_evidence(item)

        initial_action: Optional[RepairAction] = None
        initial_round: Optional[RepairExecutionResult] = None
        stopped_reason = "no_more_failures"

        initial_decisions = spec.policy.decide(initial_evidence, store.edges, budget)
        if not initial_decisions:
            stopped_reason = "initial_policy_returned_nothing"
        else:
            initial_action = initial_decisions[0]
            initial_round = executor.execute(
                initial_action, store, triggering_evidence=initial_evidence
            )
            for cost in initial_round.cost:
                budget = budget.charge(cost)

        downstream_rounds: List[RepairExecutionResult] = []
        if initial_round is not None:
            current_evidence = self._collect_round_evidence(initial_round)
            if not current_evidence and self._stalled_without_evidence(initial_round):
                # Empty evidence here does NOT mean "resolved" -- the
                # initial round could not even be attempted (every target
                # came back PRODUCTION_FAILED/NO_PRODUCER, so no fresh
                # Validator.check() ever ran to produce reverify_evidence
                # either way). Reporting this as "no_more_failures" would
                # misread a failed repair as a clean one -- exactly the
                # "실패한 복구도 남는다" completion criterion this guards.
                # The failure itself is never lost either way: it is
                # already visible on `initial_round.overall_outcome`.
                stopped_reason = "repair_could_not_be_attempted"
            else:
                rounds_run = 0
                while rounds_run < self._max_downstream_rounds:
                    if budget.remaining <= 0:
                        stopped_reason = "budget_exhausted"
                        break
                    if not any(item.outcome == CheckOutcome.FAIL for item in current_evidence):
                        stopped_reason = "no_more_failures"
                        break
                    decisions = downstream_policy.decide(current_evidence, store.edges, budget)
                    if not decisions:
                        stopped_reason = "downstream_policy_returned_nothing"
                        break
                    round_result = executor.execute(
                        decisions[0], store, triggering_evidence=current_evidence
                    )
                    downstream_rounds.append(round_result)
                    for cost in round_result.cost:
                        budget = budget.charge(cost)
                    current_evidence = self._collect_round_evidence(round_result)
                    if not current_evidence and self._stalled_without_evidence(round_result):
                        stopped_reason = "repair_could_not_be_attempted"
                        break
                    rounds_run += 1
                else:
                    # Loop exited by exhausting `_max_downstream_rounds`
                    # without ever `break`-ing -- a real cap being hit, not
                    # one of the explicit stop conditions above.
                    stopped_reason = "max_rounds_reached"

        return BranchExecutionResult(
            branch_id=spec.label,
            source_checkpoint_id=checkpoint.checkpoint_id,
            policy_label=spec.label,
            initial_action=initial_action,
            initial_round=initial_round,
            downstream_rounds=downstream_rounds,
            stopped_reason=stopped_reason,
            final_checkpoint=store.save_checkpoint(label=f"{spec.label}-final"),
            total_cost_tokens=budget.spent,
        )

    @staticmethod
    def _stalled_without_evidence(round_result: RepairExecutionResult) -> bool:
        """True if this round produced no reverify evidence specifically
        *because* it could not even be attempted (every regenerated target
        came back ``PRODUCTION_FAILED``/``NO_PRODUCER``) -- as opposed to
        every target genuinely passing its fresh re-check, which also
        yields no FAIL evidence but means the opposite thing. Only
        meaningful together with an already-empty
        ``_collect_round_evidence`` result -- see the two call sites."""
        return any(
            record.outcome in (RepairOutcome.PRODUCTION_FAILED, RepairOutcome.NO_PRODUCER)
            for record in round_result.regenerated
        )

    @staticmethod
    def _collect_round_evidence(round_result: RepairExecutionResult) -> List[Evidence]:
        """The fresh evidence a round's own cascade already produced while
        re-verifying regenerated artifacts -- fed into the *next* round's
        ``decide()`` call as its input. Deliberately excludes
        ``reverified_untouched`` (nothing failed for those, so there is
        nothing for the next round to act on) and excludes whatever
        evidence an un-regenerated candidate's skip-for-no-constraints path
        might have produced (there is none -- see
        ``RepairOperationExecutor.execute()``'s own docstring)."""
        collected: List[Evidence] = []
        for record in round_result.regenerated:
            collected.extend(record.reverify_evidence)
        return collected


__all__ = ["BranchSpec", "BranchingRunner"]

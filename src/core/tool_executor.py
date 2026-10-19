"""Tool executor: dispatches one structured ``ToolCall`` at a time and
enforces budget cap / max action count / suspend-resume (Stage 3 of the
"Revision 2026.10.15" plan, checklist items 1 and 4).

Deliberately a plain class, not a ``RepairPolicy`` and not a controller:
it never decides *which* action to take next (that is Stage 5's
controller, reading ``Evidence``/``Budget`` and calling a ``RepairPolicy``
-- ``interfaces.py``). This class only carries out whichever single
``ToolCall`` it is handed, and is the one place -- "controller 외부의
executor" (Stage 3 checklist item 4) -- where budget/action-count/
suspension are actually enforced, deliberately kept out of any
decision-making method so that a ``RepairPolicy.decide()``-style method
never needs its own copy of this bookkeeping (see ``interfaces.py``'s
``RepairPolicy`` docstring and ``docgen_stage_review_rigor`` pattern #1:
anything that "decides" must stay side-effect-free; only an actual
execution step may advance counters).

PATCH/REPLAN are schema-only this stage (see ``schemas.ToolActionKind``'s
docstring) -- calling them always returns ``ToolErrorCode.NOT_IMPLEMENTED``
without touching budget/action-count at all, since nothing was actually
run. INSPECT/QUERY/CHECK/COMPILE are fully dispatched to, respectively,
``ArtifactGraphStore``, the same, a supplied ``Validator``
(``constraint_checks.BasicConstraintChecker`` by default), and the real
``LatexCompiler.compile_pdf``. FINISH is always admissible regardless of
suspension or exhausted budget/action-count (see ``schemas.FinishCall``'s
docstring).
"""

from __future__ import annotations

import time
from typing import Optional

from src.agents.latex_compiler import LatexCompiler
from src.core.artifact_graph import ArtifactGraphStore
from src.core.interfaces import Validator
from src.core.schemas import (
    Budget,
    CheckCall,
    CheckResult,
    CompileCall,
    CompileToolResult,
    CostRecord,
    FinishCall,
    FinishToolResult,
    InspectArtifactCall,
    InspectArtifactResult,
    QueryDependenciesCall,
    QueryDependenciesResult,
    ToolActionKind,
    ToolCall,
    ToolErrorCode,
    ToolExecutionError,
    ToolObservation,
)
from src.evaluation.event_logger import EventLogger


class ToolExecutor:
    """One branch's tool-call dispatcher. One instance targets exactly one
    ``ArtifactGraphStore`` (and therefore one branch), mirroring that
    store's own one-instance-one-branch design (``artifact_graph.py``).

    ``budget``/``max_actions`` are this executor's own admission limits --
    distinct from, and layered on top of, any per-artifact retry cap a
    ``RepairPolicy`` like ``adapters.BlindRetryRepairPolicy`` tracks
    internally; those are about *which artifact* gets another attempt,
    this is about *the run as a whole* running out of room to do anything
    further.
    """

    def __init__(
        self,
        store: ArtifactGraphStore,
        checker: Validator,
        *,
        budget: Budget,
        max_actions: int,
    ) -> None:
        self._store = store
        self._checker = checker
        self._budget = budget
        self._max_actions = max_actions
        self._actions_taken = 0
        self._suspended = False
        self._suspend_reason: Optional[str] = None

    @property
    def budget(self) -> Budget:
        return self._budget.model_copy(deep=True)

    @property
    def actions_taken(self) -> int:
        return self._actions_taken

    @property
    def is_suspended(self) -> bool:
        return self._suspended

    @property
    def suspend_reason(self) -> Optional[str]:
        return self._suspend_reason

    def suspend(self, reason: str) -> None:
        self._suspended = True
        self._suspend_reason = reason

    def resume(self, *, extra_budget: float = 0.0, extra_actions: int = 0) -> None:
        """Lifts suspension, optionally raising the cap/action-count that
        caused it. ``extra_budget``/``extra_actions`` default to 0: a
        caller resuming purely because an external condition changed
        (e.g. a human raised the cap through some other channel first)
        does not need to pass anything through here."""
        if extra_budget:
            self._budget = self._budget.model_copy(update={"cap": self._budget.cap + extra_budget})
        self._max_actions += extra_actions
        self._suspended = False
        self._suspend_reason = None

    def execute(self, call: ToolCall) -> ToolObservation:
        if call.action != ToolActionKind.FINISH:
            admission_error = self._check_admission(call.action)
            if admission_error is not None:
                return ToolObservation(action=call.action, ok=False, error=admission_error, cost=None)

        if call.action == ToolActionKind.PATCH or call.action == ToolActionKind.REPLAN:
            # Schema-only this stage -- never charged, never counted as an
            # action taken, since nothing was actually run. See this
            # module's and ``ToolActionKind``'s docstrings.
            return ToolObservation(
                action=call.action,
                ok=False,
                error=ToolExecutionError(
                    code=ToolErrorCode.NOT_IMPLEMENTED,
                    action=call.action,
                    message=(
                        f"{call.action.value} is schema-only in Stage 3 -- actual repair "
                        "operations are Stage 4's scope (see docs/stage2_artifact_graph.md, "
                        "\"하지 않은 것\")."
                    ),
                ),
                cost=None,
            )

        start = time.monotonic()
        try:
            output = self._dispatch(call)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
            # dispatch failure must still be charged and reported, not
            # propagated past this boundary uncaught (mirrors
            # ``Producer.produce()``'s own "never raise past this boundary
            # for an expected, routine failure" contract in
            # ``interfaces.py``).
            cost = CostRecord(
                stage=f"tool_executor:{call.action.value}",
                latency_s=time.monotonic() - start,
                succeeded=False,
            )
            self._charge(cost)
            return ToolObservation(
                action=call.action,
                ok=False,
                error=ToolExecutionError(
                    code=ToolErrorCode.EXECUTION_FAILED, action=call.action, message=str(exc)
                ),
                cost=cost,
            )

        cost = CostRecord(
            stage=f"tool_executor:{call.action.value}",
            latency_s=time.monotonic() - start,
            succeeded=True,
        )
        self._charge(cost)
        return ToolObservation(action=call.action, ok=True, output=output, error=None, cost=cost)

    def _check_admission(self, action: ToolActionKind) -> Optional[ToolExecutionError]:
        if self._suspended:
            return ToolExecutionError(
                code=ToolErrorCode.SUSPENDED,
                action=action,
                message=self._suspend_reason or "executor is suspended",
            )
        if self._budget.remaining <= 0:
            self.suspend(f"budget exhausted (cap={self._budget.cap}, spent={self._budget.spent})")
            return ToolExecutionError(
                code=ToolErrorCode.BUDGET_EXCEEDED,
                action=action,
                message=f"no budget remaining (cap={self._budget.cap}, spent={self._budget.spent})",
            )
        if self._actions_taken >= self._max_actions:
            self.suspend(f"max action count reached ({self._max_actions})")
            return ToolExecutionError(
                code=ToolErrorCode.MAX_ACTIONS_EXCEEDED,
                action=action,
                message=f"max_actions={self._max_actions} reached (actions_taken={self._actions_taken})",
            )
        return None

    def _charge(self, cost: CostRecord) -> None:
        """Only reached for a call that was actually dispatched (see
        ``execute()``: an admission-rejected or not-implemented call never
        reaches here) -- so "every executed call is charged, every
        rejected call is not" holds by construction, not by remembering to
        check at each call site."""
        self._budget = self._budget.charge(cost)
        self._store.record_cost(cost)
        self._actions_taken += 1
        if self._budget.remaining <= 0:
            self.suspend(f"budget exhausted after this call (cap={self._budget.cap}, spent={self._budget.spent})")
        elif self._actions_taken >= self._max_actions:
            self.suspend(f"max action count reached after this call ({self._max_actions})")

    def _dispatch(self, call: ToolCall):
        if isinstance(call, InspectArtifactCall):
            return self._dispatch_inspect(call)
        if isinstance(call, QueryDependenciesCall):
            return self._dispatch_query(call)
        if isinstance(call, CheckCall):
            return self._dispatch_check(call)
        if isinstance(call, CompileCall):
            return self._dispatch_compile(call)
        if isinstance(call, FinishCall):
            return self._dispatch_finish(call)
        raise TypeError(f"Unhandled ToolCall type: {type(call)!r}")

    def _dispatch_inspect(self, call: InspectArtifactCall) -> InspectArtifactResult:
        stored = (
            self._store.get_version(call.artifact_id, call.version)
            if call.version is not None
            else self._store.latest(call.artifact_id)
        )
        return InspectArtifactResult(found=stored is not None, stored=stored)

    def _dispatch_query(self, call: QueryDependenciesCall) -> QueryDependenciesResult:
        candidates = sorted(self._store.impact_candidates(call.artifact_id))
        relevant_edges = [
            edge
            for edge in self._store.edges
            if edge.from_artifact.artifact_id == call.artifact_id
            or edge.to_artifact.artifact_id == call.artifact_id
        ]
        return QueryDependenciesResult(impact_candidates=candidates, edges=relevant_edges)

    def _dispatch_check(self, call: CheckCall) -> CheckResult:
        # Always a fresh call to the checker -- never a ``live_evidence()``
        # shortcut. See ``CheckCall``'s docstring: a stale verification
        # result must never stand in for actually re-checking.
        evidence = self._checker.check(call.artifact, call.constraints, call.context)
        for item in evidence:
            self._store.record_evidence(item)
        return CheckResult(evidence=evidence)

    def _dispatch_compile(self, call: CompileCall) -> CompileToolResult:
        event_logger = EventLogger(run_id=call.request_id)
        compiler = LatexCompiler()
        compile_success = compiler.compile_pdf(
            call.request_id, output_dir=call.output_dir, event_logger=event_logger
        )
        return CompileToolResult(compile_success=compile_success, compile_result=event_logger.compile_result)

    def _dispatch_finish(self, call: FinishCall) -> FinishToolResult:
        return FinishToolResult(acknowledged=True)


__all__ = ["ToolExecutor"]

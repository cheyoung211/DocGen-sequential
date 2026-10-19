"""Append-only action/evidence/cost log (Stage 2 of the "Revision
2026.10.15" plan, 2단계 세 번째 bullet: "action/evidence/cost를 추가
기록하는 로그").

Deliberately separate from ``src/evaluation/event_logger.py::EventLogger``:
that class accumulates today's low-level, per-agent-call
``VerificationEvent``/``RecoveryEvent`` records -- an existing, unrelated
mechanism this revision does not touch (see ``src/core/schemas.py``'s
module docstring on why these are new, additive concepts). ``RunLog``
accumulates the *new* Stage 1/2 concepts (``RepairAction``, ``Evidence``,
``CostRecord``) that a controller-level executor produces once one exists
(Stage 3+); nothing here reads or writes ``EventLogger``'s state, and
nothing in ``EventLogger`` reads or writes this.

Mirrors ``EventLogger``'s accumulate-in-lists shape on purpose -- same
append-then-read contract, same "a logger never decides anything, only
records what already happened" rule. In particular, nothing on this class
may be called from inside a ``RepairPolicy.decide()`` (see that Protocol's
docstring in ``interfaces.py``): ``log_action`` records that an action
actually *started* executing, which only an executor, never a policy, can
know.
"""

from __future__ import annotations

from typing import List, Optional

from src.core.schemas import ActionLogEntry, CostRecord, Evidence, RepairAction


class RunLog:
    """Accumulates ``ActionLogEntry``/``Evidence``/``CostRecord`` for one
    ``ArtifactGraphStore`` branch.

    Holds no reference to the store itself and tracks no branch of its own
    -- ``branch_id`` is passed in explicitly on every ``log_action`` call,
    so one ``RunLog`` instance can never silently mislabel which branch an
    entry belongs to (the store, not this class, decides what its own
    ``branch_id`` is).
    """

    def __init__(self) -> None:
        self.actions: List[ActionLogEntry] = []
        self.evidence: List[Evidence] = []
        self.costs: List[CostRecord] = []

    def log_action(
        self,
        action: RepairAction,
        branch_id: str,
        triggering_evidence: Optional[List[Evidence]] = None,
    ) -> ActionLogEntry:
        entry = ActionLogEntry(
            action=action,
            branch_id=branch_id,
            triggering_evidence=list(triggering_evidence or []),
        )
        self.actions.append(entry)
        return entry

    def log_evidence(self, evidence: Evidence) -> None:
        self.evidence.append(evidence)

    def log_cost(self, cost: CostRecord) -> None:
        self.costs.append(cost)

    @property
    def total_cost_tokens(self) -> int:
        """Sum of every charged ``CostRecord.total_tokens``, including
        failed attempts -- ``CostRecord.succeeded=False`` records are not
        excluded (revision direction 2: "비용에는 ... 실패한 시도 ...를
        포함한다")."""
        return sum(cost.total_tokens for cost in self.costs)


__all__ = ["RunLog"]

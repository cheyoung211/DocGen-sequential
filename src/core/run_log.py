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

Every list here is private, and every boundary -- ingestion (``log_*``)
and read (the ``actions``/``evidence``/``costs`` properties) -- deep-copies.
A caller handing in a ``CostRecord`` and then mutating their own local
object afterward, or mutating a list this class handed back, must never be
able to retroactively rewrite history that has already been logged.
"""

from __future__ import annotations

import copy
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
        self._actions: List[ActionLogEntry] = []
        self._evidence: List[Evidence] = []
        self._costs: List[CostRecord] = []

    @property
    def actions(self) -> List[ActionLogEntry]:
        """A deep copy -- mutating the returned list or any entry in it
        has no effect on this log."""
        return copy.deepcopy(self._actions)

    @property
    def evidence(self) -> List[Evidence]:
        """A deep copy -- same non-mutating guarantee as ``actions``."""
        return copy.deepcopy(self._evidence)

    @property
    def costs(self) -> List[CostRecord]:
        """A deep copy -- same non-mutating guarantee as ``actions``."""
        return copy.deepcopy(self._costs)

    def log_action(
        self,
        action: RepairAction,
        branch_id: str,
        triggering_evidence: Optional[List[Evidence]] = None,
    ) -> ActionLogEntry:
        """Deep-copies ``action``/``triggering_evidence`` before storing --
        the caller's own objects are never retained by reference -- and
        returns a further independent deep copy, so mutating the returned
        ``ActionLogEntry`` afterward cannot corrupt what was just logged
        either."""
        entry = ActionLogEntry(
            action=action.model_copy(deep=True),
            branch_id=branch_id,
            triggering_evidence=[e.model_copy(deep=True) for e in (triggering_evidence or [])],
        )
        self._actions.append(entry)
        return entry.model_copy(deep=True)

    def log_evidence(self, evidence: Evidence) -> None:
        self._evidence.append(evidence.model_copy(deep=True))

    def log_cost(self, cost: CostRecord) -> None:
        self._costs.append(cost.model_copy(deep=True))

    @property
    def total_cost_tokens(self) -> int:
        """Sum of every charged ``CostRecord.total_tokens``, including
        failed attempts -- ``CostRecord.succeeded=False`` records are not
        excluded (revision direction 2: "비용에는 ... 실패한 시도 ...를
        포함한다")."""
        return sum(cost.total_tokens for cost in self._costs)

    def replace_all(
        self,
        actions: List[ActionLogEntry],
        evidence: List[Evidence],
        costs: List[CostRecord],
    ) -> None:
        """Wholesale-replace this log's contents. Used only by
        ``ArtifactGraphStore.from_checkpoint`` to load a checkpoint's
        already-independent history into a freshly constructed
        ``RunLog`` -- deep-copies anyway, so the caller's own
        ``checkpoint.actions``/``.evidence``/``.costs`` lists are never
        retained by reference either."""
        self._actions = copy.deepcopy(actions)
        self._evidence = copy.deepcopy(evidence)
        self._costs = copy.deepcopy(costs)


__all__ = ["RunLog"]

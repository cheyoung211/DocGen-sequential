"""Per-run event collection for the Foundation evaluation layer.

``EventLogger`` is the single place a pipeline run accumulates
``VerificationEvent``/``RecoveryEvent`` records. Every agent call site that
takes an ``event_logger`` parameter defaults it to ``None`` and guards its
use (``if event_logger is not None: ...``), so passing nothing reproduces
today's behavior exactly.

``AttemptTracker`` pairs one agent call's own internal retry-with-feedback
loop (Planner/TextAgent: ``for attempt in range(1, max_attempts+1): try:
...; except: ...``) into RecoveryEvents, without duplicating that pairing
logic in each agent.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from src.evaluation.schemas import CompileResult, RecoveryEvent, Stage, VerificationEvent


class EventLogger:
    """Accumulates verification/recovery events for one pipeline run (``run_id``)."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.verification_events: List[VerificationEvent] = []
        self.recovery_events: List[RecoveryEvent] = []
        self.compile_result: Optional[CompileResult] = None
        self._failure_counter = 0

    def new_failure_id(self) -> str:
        """Mint a fresh join key for a failure that has a recovery mechanism.

        Only called by a caller that knows this stage *can* be recovered
        (see ``AttemptTracker``) -- a failure with no recovery path (e.g.
        image_agent, compile) is logged with ``failure_id=None`` instead, so
        it's automatically excluded from the recovery-success denominator
        and counted as "detected failure without recovery path".

        Prefixed with ``run_id``: an evaluation run over a batch of many
        requests pools every request's VerificationEvents/RecoveryEvents
        together (see ``src/evaluation/aggregate.py``), and each request
        gets its own fresh ``EventLogger`` starting back at
        ``failure_0001`` -- without the prefix, two different requests'
        first failure would collide onto the same join key once pooled,
        silently overwriting one recovery outcome with the other's.
        """
        self._failure_counter += 1
        return f"{self.run_id}:failure_{self._failure_counter:04d}"

    def log_verification(
        self,
        *,
        stage: Stage,
        verifier: str,
        result: str,
        attempt: int,
        artifact_id: Optional[str] = None,
        section_id: Optional[str] = None,
        block_id: Optional[str] = None,
        signal_type: Optional[str] = None,
        message: Optional[str] = None,
        producer_model: Optional[str] = None,
        failure_id: Optional[str] = None,
    ) -> VerificationEvent:
        event = VerificationEvent(
            run_id=self.run_id,
            stage=stage,
            verifier=verifier,
            artifact_id=artifact_id,
            section_id=section_id,
            block_id=block_id,
            attempt=attempt,
            result=result,
            signal_type=signal_type,
            message=message,
            producer_model=producer_model,
            failure_id=failure_id,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        self.verification_events.append(event)
        return event

    def log_recovery(
        self,
        *,
        failure_id: str,
        stage: Stage,
        recovery_action: str,
        attempt: int,
        success: bool,
        artifact_id: Optional[str] = None,
        resulting_artifact_id: Optional[str] = None,
    ) -> RecoveryEvent:
        event = RecoveryEvent(
            run_id=self.run_id,
            failure_id=failure_id,
            stage=stage,
            artifact_id=artifact_id,
            recovery_action=recovery_action,
            attempt=attempt,
            success=success,
            resulting_artifact_id=resulting_artifact_id,
        )
        self.recovery_events.append(event)
        return event

    @property
    def total_generation_attempts(self) -> int:
        from src.evaluation.schemas import GENERATION_STAGES

        return sum(1 for e in self.verification_events if e.stage in GENERATION_STAGES)

    @property
    def total_recovery_attempts(self) -> int:
        return len(self.recovery_events)


class AttemptTracker:
    """Pairs one agent call's pass/fail attempts into RecoveryEvents.

    Scoped to a single ``generate_plan()``/``process_node()`` invocation --
    each such call constructs its own tracker before its retry loop starts.
    Deliberately NOT chained across the *outer* Phase-2 iteration re-dispatch
    in ``run_full_pipeline.py`` (a second, coarser retry mechanism sitting on
    top of each agent's own inner retry) -- that would need a second,
    node-id-keyed tracker living at the ``DocGenPipeline.run()`` level, which
    is out of scope for this pass (see the plan's design decisions).

    Pairing semantics: a ``RecoveryEvent`` represents one real retry
    attempt's outcome relative to the failure that triggered it. When a
    failure is immediately followed by another attempt (pass or fail), that
    next attempt's outcome resolves it. The *last* failure before the loop
    exhausts never gets a ``RecoveryEvent`` -- nothing was ever retried in
    response to it, so there is nothing to report as "recovery attempted and
    failed" for it specifically (it still exists as a raw fail event, just
    unpaired).
    """

    def __init__(
        self,
        event_logger: Optional[EventLogger],
        *,
        stage: Stage,
        verifier: str,
        recovery_action: str,
        artifact_id: Optional[str] = None,
        section_id: Optional[str] = None,
        block_id: Optional[str] = None,
        producer_model: Optional[str] = None,
    ) -> None:
        self._logger = event_logger
        self._stage = stage
        self._verifier = verifier
        self._recovery_action = recovery_action
        self._artifact_id = artifact_id
        self._section_id = section_id
        self._block_id = block_id
        self._producer_model = producer_model
        self._pending_failure_id: Optional[str] = None

    def record_failure(self, attempt: int, signal_type: str, message: str) -> None:
        if self._logger is None:
            return
        failure_id = self._logger.new_failure_id()
        self._logger.log_verification(
            stage=self._stage,
            verifier=self._verifier,
            artifact_id=self._artifact_id,
            section_id=self._section_id,
            block_id=self._block_id,
            attempt=attempt,
            result="fail",
            signal_type=signal_type,
            message=message,
            producer_model=self._producer_model,
            failure_id=failure_id,
        )
        if self._pending_failure_id is not None:
            self._logger.log_recovery(
                failure_id=self._pending_failure_id,
                stage=self._stage,
                recovery_action=self._recovery_action,
                attempt=attempt,
                success=False,
                artifact_id=self._artifact_id,
            )
        self._pending_failure_id = failure_id

    def record_success(
        self, attempt: int, resulting_artifact_id: Optional[str] = None
    ) -> None:
        if self._logger is None:
            return
        self._logger.log_verification(
            stage=self._stage,
            verifier=self._verifier,
            artifact_id=self._artifact_id,
            section_id=self._section_id,
            block_id=self._block_id,
            attempt=attempt,
            result="pass",
            producer_model=self._producer_model,
        )
        if self._pending_failure_id is not None:
            self._logger.log_recovery(
                failure_id=self._pending_failure_id,
                stage=self._stage,
                recovery_action=self._recovery_action,
                attempt=attempt,
                success=True,
                artifact_id=self._artifact_id,
                resulting_artifact_id=resulting_artifact_id,
            )
        self._pending_failure_id = None

    def record_exhausted(self) -> None:
        """Mark the call as having exhausted every attempt with no recovery.

        The final failure already has its own fail-event (from the last
        ``record_failure`` call); there is no further attempt to pair it
        with, so nothing new is logged here -- this only clears tracker
        state.
        """
        self._pending_failure_id = None

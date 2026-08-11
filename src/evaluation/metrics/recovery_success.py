"""Metric 2: Recovery Success Rate.

recovery_success_rate = successfully_recovered_failures / failures_with_recovery_attempt.
"""

from __future__ import annotations

from typing import Dict, Sequence

from src.evaluation.schemas import RecoveryEvent, VerificationEvent


def compute_recovery_success_rate(
    verification_events: Sequence[VerificationEvent],
    recovery_events: Sequence[RecoveryEvent],
) -> dict:
    """A detected failure with no RecoveryEvent falls into one of two
    distinct buckets (kept separate for the spec's "recovery gap" analysis):

    - ``detected_failures_without_recovery_path``: the stage has no recovery
      mechanism at all (``failure_id`` is ``None`` -- e.g. image_agent,
      compile in this Foundation pass).
    - ``detected_failures_without_recovery_attempt``: the mechanism exists
      (``failure_id`` is set) but this particular failure never got a next
      attempt before the retry budget ran out (the last failure before
      exhaustion -- see ``AttemptTracker``).
    """
    failures = [e for e in verification_events if e.result == "fail"]
    recovery_by_failure_id: Dict[str, RecoveryEvent] = {
        recovery.failure_id: recovery for recovery in recovery_events
    }

    without_recovery_path = [f for f in failures if f.failure_id is None]
    with_failure_id = [f for f in failures if f.failure_id is not None]
    with_recovery_attempt = [f for f in with_failure_id if f.failure_id in recovery_by_failure_id]
    without_recovery_attempt = [
        f for f in with_failure_id if f.failure_id not in recovery_by_failure_id
    ]
    successfully_recovered = [
        f for f in with_recovery_attempt if recovery_by_failure_id[f.failure_id].success
    ]

    denominator = len(with_recovery_attempt)

    by_stage: Dict[str, dict] = {}
    for failure in with_recovery_attempt:
        stage = failure.stage.value if hasattr(failure.stage, "value") else str(failure.stage)
        bucket = by_stage.setdefault(
            stage, {"failures_with_recovery_attempt": 0, "successfully_recovered": 0}
        )
        bucket["failures_with_recovery_attempt"] += 1
        if recovery_by_failure_id[failure.failure_id].success:
            bucket["successfully_recovered"] += 1
    for bucket in by_stage.values():
        bucket["recovery_success_rate"] = (
            bucket["successfully_recovered"] / bucket["failures_with_recovery_attempt"]
        )

    return {
        "recovery_success_rate": (len(successfully_recovered) / denominator) if denominator else None,
        "successfully_recovered": len(successfully_recovered),
        "failures_with_recovery_attempt": denominator,
        "detected_failures_without_recovery_path": len(without_recovery_path),
        "detected_failures_without_recovery_attempt": len(without_recovery_attempt),
        "by_stage": by_stage,
    }

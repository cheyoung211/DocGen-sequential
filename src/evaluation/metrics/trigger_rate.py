"""Metric 1: Verification Trigger Rate.

trigger_rate = triggered_count / checked_count, aggregated overall / by
stage / by verifier / by signal_type / by producer model, computed both at
the raw event level (every retry attempt) and restricted to the first
attempt only, per the spec's explicit request to keep both distinguishable.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

from src.evaluation.schemas import VerificationEvent


def _stage_value(event: VerificationEvent) -> str:
    stage = event.stage
    return stage.value if hasattr(stage, "value") else str(stage)


def _overall(events: Sequence[VerificationEvent]) -> dict:
    checked = len(events)
    triggered = sum(1 for e in events if e.result == "fail")
    return {
        "checked_count": checked,
        "triggered_count": triggered,
        "trigger_rate": (triggered / checked) if checked else None,
    }


def _rate_buckets(
    events: Sequence[VerificationEvent], key_fn: Callable[[VerificationEvent], Optional[str]]
) -> Dict[str, dict]:
    buckets: Dict[str, Dict[str, int]] = {}
    for event in events:
        key = key_fn(event)
        if key is None:
            continue
        bucket = buckets.setdefault(key, {"checked_count": 0, "triggered_count": 0})
        bucket["checked_count"] += 1
        if event.result == "fail":
            bucket["triggered_count"] += 1
    return {
        key: {**bucket, "trigger_rate": bucket["triggered_count"] / bucket["checked_count"]}
        for key, bucket in buckets.items()
    }


def _signal_type_distribution(events: Sequence[VerificationEvent]) -> Dict[str, dict]:
    """Not a "rate": signal_type is only ever set on a failed event, so a
    checked/triggered rate restricted to one signal_type would always be
    1.0. Reports a distribution over the failures themselves instead -- how
    often each signal_type occurs, as a share of all failures in this set.
    """
    failures = [e for e in events if e.result == "fail" and e.signal_type]
    total = len(failures)
    counts: Dict[str, int] = {}
    for event in failures:
        counts[event.signal_type] = counts.get(event.signal_type, 0) + 1
    return {
        signal_type: {"count": count, "share_of_failures": count / total}
        for signal_type, count in counts.items()
    }


def _one_view(events: Sequence[VerificationEvent]) -> dict:
    return {
        "overall": _overall(events),
        "by_stage": _rate_buckets(events, _stage_value),
        "by_verifier": _rate_buckets(events, lambda e: e.verifier),
        "by_producer_model": _rate_buckets(events, lambda e: e.producer_model),
        "by_signal_type": _signal_type_distribution(events),
    }


def compute_trigger_rate(events: Sequence[VerificationEvent]) -> dict:
    initial_attempt_events = [e for e in events if e.attempt == 1]
    return {
        "event_level": _one_view(events),
        "initial_attempt_only": _one_view(initial_attempt_events),
    }

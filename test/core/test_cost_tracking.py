"""Coverage for src/core/cost_tracking.py (Stage 3): GpuUtilizationSampler
and CostTrackingProducer, the "로컬/API 공통 모델 adapter"."""

from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.core.cost_tracking import CostTrackingProducer, GpuUtilizationSampler, _is_local_model
from src.core.schemas import ArtifactKind, ArtifactRef, ProductionResult, ProductionStatus


class _FakeProducer:
    """A minimal conforming Producer: appends usage_sink entries like a
    real agent would, then returns a ProductionResult."""

    def __init__(self, usage_entries: List[dict], *, sleep_s: float = 0.0, status: ProductionStatus = ProductionStatus.SUCCESS):
        self._usage_entries = usage_entries
        self._sleep_s = sleep_s
        self._status = status
        self.received_context: Dict[str, Any] = {}

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ProductionResult:
        self.received_context = context
        if self._sleep_s:
            time.sleep(self._sleep_s)
        sink = context.get("usage_sink")
        if sink is not None:
            for entry in self._usage_entries:
                sink.append(entry)
        if self._status == ProductionStatus.SUCCESS:
            return ProductionResult(
                status=ProductionStatus.SUCCESS,
                artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.SECTION),
            )
        return ProductionResult(status=ProductionStatus.FAILED, error="synthetic failure")


class _RaisingProducer:
    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ProductionResult:
        context.get("usage_sink", []).append({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
        raise RuntimeError("out-of-contract raise")


def test_is_local_model_matches_llm_client_infer_provider_rule() -> None:
    assert _is_local_model("qwen3.6-35b-a3b") is True
    assert _is_local_model("QWEN-anything") is True
    assert _is_local_model("gpt-4o-mini") is False
    assert _is_local_model("gemini-3-flash-preview") is False


def test_sums_multiple_usage_sink_entries_from_one_produce_call() -> None:
    """A Producer's inner retry loop can append more than one usage entry
    before returning once -- every entry must be summed into the one
    CostRecord for this produce() call, failed inner attempts included."""
    fake = _FakeProducer([
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    ])
    wrapped = CostTrackingProducer(fake, stage="text_agent", model_name="gpt-4o-mini")
    result = wrapped.produce("sec_1", {})
    assert result.status == ProductionStatus.SUCCESS
    assert wrapped.last_cost.input_tokens == 13
    assert wrapped.last_cost.output_tokens == 7
    assert wrapped.last_cost.total_tokens == 20
    assert wrapped.last_cost.succeeded is True


def test_no_usage_sink_entries_is_zero_tokens_not_an_error() -> None:
    fake = _FakeProducer([])
    wrapped = CostTrackingProducer(fake, stage="planner", model_name="gpt-4o-mini")
    wrapped.produce("p1", {})
    assert wrapped.last_cost.total_tokens == 0


def test_failed_production_result_is_still_recorded_as_a_cost_with_succeeded_false() -> None:
    fake = _FakeProducer([], status=ProductionStatus.FAILED)
    wrapped = CostTrackingProducer(fake, stage="text_agent", model_name="gpt-4o-mini")
    result = wrapped.produce("sec_1", {})
    assert result.status == ProductionStatus.FAILED
    assert wrapped.last_cost is not None
    assert wrapped.last_cost.succeeded is False


def test_an_unexpected_raise_from_the_wrapped_producer_still_propagates_and_still_records_cost() -> None:
    """Out-of-contract for a conforming Producer, but if it happens anyway
    the latency genuinely elapsed must not be lost -- and the original
    exception must not be swallowed."""
    wrapped = CostTrackingProducer(_RaisingProducer(), stage="planner", model_name="gpt-4o-mini")
    with pytest.raises(RuntimeError, match="out-of-contract raise"):
        wrapped.produce("p1", {})
    assert wrapped.last_cost is not None
    assert wrapped.last_cost.succeeded is False
    assert wrapped.last_cost.total_tokens == 2  # the one entry appended before the raise


def test_does_not_mutate_the_callers_own_context_dict() -> None:
    """The wrapper injects its own usage_sink without retroactively
    changing what the caller's own context dict looks like afterward --
    same "copy on ingestion" discipline Stage 2 applies throughout
    artifact_graph.py."""
    fake = _FakeProducer([])
    wrapped = CostTrackingProducer(fake, stage="planner", model_name="gpt-4o-mini")
    caller_context = {"request_id": "r1"}
    wrapped.produce("p1", caller_context)
    assert "usage_sink" not in caller_context
    assert fake.received_context["usage_sink"] == []
    assert fake.received_context["request_id"] == "r1"


def test_api_model_never_runs_a_gpu_sampler() -> None:
    fake = _FakeProducer([])
    wrapped = CostTrackingProducer(fake, stage="planner", model_name="gpt-4o-mini")
    with patch("src.core.cost_tracking.GpuUtilizationSampler") as mock_sampler_cls:
        wrapped.produce("p1", {})
    mock_sampler_cls.assert_not_called()
    assert wrapped.last_cost.gpu_seconds is None


def test_local_model_runs_a_gpu_sampler_around_the_call() -> None:
    fake = _FakeProducer([])
    wrapped = CostTrackingProducer(fake, stage="planner", model_name="qwen3.6-35b-a3b")
    with patch("src.core.cost_tracking.GpuUtilizationSampler") as mock_sampler_cls:
        mock_sampler_cls.return_value.stop.return_value = 0.42
        wrapped.produce("p1", {})
    mock_sampler_cls.return_value.start.assert_called_once()
    mock_sampler_cls.return_value.stop.assert_called_once()
    assert wrapped.last_cost.gpu_seconds == 0.42


class TestGpuUtilizationSampler:
    def test_unavailable_nvidia_smi_returns_none_not_zero(self) -> None:
        with patch("src.core.cost_tracking.shutil.which", return_value=None):
            sampler = GpuUtilizationSampler(device_index=0, interval_s=0.05)
            sampler.start()
            time.sleep(0.1)
            assert sampler.stop() is None

    def test_a_failing_sample_is_skipped_not_raised(self) -> None:
        with patch("src.core.cost_tracking.shutil.which", return_value="/usr/bin/nvidia-smi"):
            sampler = GpuUtilizationSampler(device_index=0, interval_s=0.9)
            with patch.object(sampler, "_sample_once", return_value=None):
                sampler.start()
                time.sleep(0.05)
                # _sample_once is patched to always fail -> stop() takes one
                # best-effort final reading via the same (still-patched,
                # still-None) method, so the overall result stays None.
                assert sampler.stop() is None

    def test_a_call_shorter_than_one_interval_still_gets_one_best_effort_sample(self) -> None:
        with patch("src.core.cost_tracking.shutil.which", return_value="/usr/bin/nvidia-smi"):
            sampler = GpuUtilizationSampler(device_index=0, interval_s=5.0)
            with patch.object(sampler, "_sample_once", return_value=0.5):
                sampler.start()
                result = sampler.stop()
        assert result is not None
        assert result >= 0.0

    def test_estimate_is_mean_utilization_times_elapsed_wall_clock(self) -> None:
        with patch("src.core.cost_tracking.shutil.which", return_value="/usr/bin/nvidia-smi"):
            sampler = GpuUtilizationSampler(device_index=0, interval_s=1000.0)  # never fires in time
            with patch.object(sampler, "_sample_once", return_value=1.0):  # pretend-100% utilization
                sampler.start()
                time.sleep(0.2)
                result = sampler.stop()
        # one best-effort sample at stop() time (1.0 == 100%), so the
        # estimate should equal ~elapsed seconds, not some other scaling.
        assert result is not None
        assert 0.15 <= result <= 0.6

    def test_real_nvidia_smi_subprocess_failure_is_handled_without_raising(self) -> None:
        """Exercise the real (unpatched) _sample_once path with a
        nonexistent binary name, proving the try/except around the actual
        subprocess call -- not just the mocked unit above -- never lets a
        sampling failure escape."""
        sampler = GpuUtilizationSampler(device_index=0, interval_s=0.05)
        sampler._available = True  # force the real subprocess path
        with patch("src.core.cost_tracking.subprocess.run", side_effect=FileNotFoundError("no nvidia-smi")):
            assert sampler._sample_once() is None

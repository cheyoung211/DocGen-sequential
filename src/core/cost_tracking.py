"""Cost/latency measurement for model calls (Stage 3 of the "Revision
2026.10.15" plan, checklist item 4: "초기 생성과 모든 후속 호출의 비용·
지연을 기록하고 ... 필요 시 로컬/API 공통 모델 adapter를 둔다").

``CostTrackingProducer`` is that common local/API adapter. It wraps any
Stage 1 ``Producer`` (``src/core/interfaces.py``) -- today, concretely,
``adapters.PlannerAgentProducer``/``adapters.TextAgentProducer``, the only
two real LLM-calling surfaces that exist in this codebase so far ("초기
생성", direction 2) -- and produces one ``CostRecord`` per ``produce()``
call, with real measured wall-clock ``latency_s`` and, for a local model,
a measured ``gpu_seconds`` via ``GpuUtilizationSampler``. It does not call
any agent or LLM client directly; it only supplies the ``usage_sink`` list
every agent method already accepts (``src/agents/planner.py``,
``src/agents/text_agent.py``, ``scripts/llm_client.py``) and reads it back
after the call, so wrapping it changes nothing about what the wrapped
``Producer`` actually does.

Deliberately stays additive, same as every other Stage 1-3 class: nothing
here is wired into ``scripts/run_full_pipeline.py``. Wiring real pipeline
runs through cost-tracked producers is Stage 4/5's job, matching every
other "do the plumbing now, connect it to the real run later" boundary
already drawn in ``docs/stage1_experiment_design.md``/
``docs/stage2_artifact_graph.md``.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

from src.core.interfaces import Producer
from src.core.schemas import CostRecord, ProductionResult, ProductionStatus


def _is_local_model(model_name: str) -> bool:
    """Mirrors ``scripts/llm_client.py::_infer_provider``'s "qwen" ->
    "local" rule exactly, without importing that module's private
    function across a package boundary. Only ``gpu_seconds`` measurement
    depends on this: getting it wrong for a model name this project does
    not actually use yet has no consequence beyond that model's own
    ``CostRecord.gpu_seconds`` being ``None`` instead of measured."""
    return model_name.lower().startswith("qwen")


class GpuUtilizationSampler:
    """Coarse, nvidia-smi-based estimate of GPU time consumed on one
    device during a wall-clock window, with zero new dependencies (no
    ``pynvml``/``nvidia-ml-py`` -- ``nvidia-smi`` is already present on
    this host; see the Stage 3 design discussion).

    **This is an approximation, not per-process GPU accounting.**
    ``nvidia-smi``'s ``utilization.gpu`` reports whole-device utilization
    -- if anything else is also using the same device concurrently (e.g.
    another run's vLLM traffic sharing the same GPU), its activity is
    indistinguishable from this call's own and gets counted into the same
    estimate. ``gpu_seconds`` is computed as
    ``mean(sampled utilization fraction) * elapsed wall-clock seconds``,
    i.e. "how much of one full GPU-second this window looks like it used,
    going by whole-device utilization" -- not an exact figure, and
    documented as such everywhere it is produced
    (``CostRecord.gpu_seconds``'s own docstring already says API and local
    cost must not be assumed equal; this is the other half of that: local
    cost itself is only ever an estimate here).

    Returns ``None`` (never ``0.0``) when no real measurement was taken at
    all -- ``nvidia-smi`` missing from ``PATH``, or every sample attempt
    failing -- so a caller can tell "measured zero utilization" apart from
    "could not measure".
    """

    def __init__(self, device_index: int, interval_s: float = 0.2) -> None:
        self.device_index = device_index
        self.interval_s = interval_s
        self._available = shutil.which("nvidia-smi") is not None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[float] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def _sample_once(self) -> Optional[float]:
        """One ``utilization.gpu`` reading as a 0..1 fraction, or ``None``
        if the command fails/times out/returns something unparsable --
        never raises, since a sampling hiccup must not take down whatever
        call is being measured."""
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_index}",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if completed.returncode != 0:
                return None
            return float(completed.stdout.strip()) / 100.0
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample_once()
            if sample is not None:
                self._samples.append(sample)
            self._stop_event.wait(self.interval_s)

    def start(self) -> None:
        """No-op (but still marks a start time) if ``nvidia-smi`` was never
        available -- ``stop()`` then returns ``None``."""
        self._samples = []
        self._stop_event.clear()
        self._start_time = time.monotonic()
        if not self._available:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[float]:
        self._end_time = time.monotonic()
        if not self._available:
            return None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2 + 1.0)
        if not self._samples:
            # The window was shorter than one sampling interval -- take one
            # best-effort reading now rather than reporting "unmeasured"
            # for every call too fast for the background thread to ever
            # fire.
            sample = self._sample_once()
            if sample is None:
                return None
            self._samples = [sample]
        elapsed = (self._end_time or 0.0) - (self._start_time or 0.0)
        mean_utilization = sum(self._samples) / len(self._samples)
        return max(0.0, elapsed) * mean_utilization


def _sum_usage(usage_sink: List[dict]) -> Dict[str, int]:
    """Sums every entry ``usage_sink`` accumulated during one ``produce()``
    call into one token total. A ``Producer``'s inner retry loop can make
    several real LLM calls before returning once (``TextAgent.process_node``'s
    own ``_run_generation_loop``, up to its configured attempt cap) --
    each successful or failed inner attempt that actually reached an LLM
    appends its own entry (``scripts/llm_client.py``'s
    ``_record_*_usage``), so summing here is what makes one external
    ``CostRecord`` per ``produce()`` call already include every inner
    attempt's cost, failed ones included (Stage 3 completion criterion:
    "실패한 시도 ... 도 비용에서 누락되지 않는다") -- without needing to
    intercept each inner attempt separately."""
    return {
        "input_tokens": sum(entry.get("input_tokens", 0) for entry in usage_sink),
        "output_tokens": sum(entry.get("output_tokens", 0) for entry in usage_sink),
        "total_tokens": sum(entry.get("total_tokens", 0) for entry in usage_sink),
    }


class CostTrackingProducer:
    """Wraps a ``Producer`` to measure and attach a ``CostRecord`` to every
    ``produce()`` call -- the "common local/API model adapter". ``stage``
    labels the resulting ``CostRecord.stage`` (e.g. ``"planner"``,
    ``"text_agent"``); ``model_name`` decides only whether a
    ``GpuUtilizationSampler`` runs (see ``_is_local_model``) and is never
    forwarded to the wrapped ``Producer`` -- that ``Producer`` already
    knows its own model internally (set at the wrapped agent's own
    construction), this class does not change or need to know it beyond
    this one branch.

    ``last_cost`` holds the most recent call's ``CostRecord`` (mirroring
    ``PlannerAgentProducer.last_output``/``TextAgentProducer.last_output``'s
    existing "stash the rich thing on self, Protocol return type stays
    narrow" pattern) -- a caller logs it via
    ``ArtifactGraphStore.record_cost(wrapped.last_cost)`` itself; this
    class does not hold or require a store reference, so it is usable
    standalone (e.g. in tests) without one.

    Every ``produce()`` call gets exactly one ``CostRecord``, whether the
    wrapped ``Producer`` reports ``SUCCESS`` or ``FAILED`` --
    ``Producer.produce()`` never raises for an expected, routine failure
    (``interfaces.py``'s own contract), so there is no exception path to
    special-case here: ``succeeded`` simply mirrors
    ``result.status == ProductionStatus.SUCCESS``.
    """

    def __init__(
        self,
        wrapped: Producer,
        *,
        stage: str,
        model_name: str,
        gpu_device_index: int = 3,
        gpu_sample_interval_s: float = 0.2,
    ) -> None:
        self._wrapped = wrapped
        self._stage = stage
        self._model_name = model_name
        self._gpu_device_index = gpu_device_index
        self._gpu_sample_interval_s = gpu_sample_interval_s
        self.last_cost: Optional[CostRecord] = None

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ProductionResult:
        usage_sink: List[dict] = []
        call_context = dict(context)
        call_context["usage_sink"] = usage_sink

        sampler = (
            GpuUtilizationSampler(self._gpu_device_index, self._gpu_sample_interval_s)
            if _is_local_model(self._model_name)
            else None
        )

        start = time.monotonic()
        if sampler is not None:
            sampler.start()
        try:
            result = self._wrapped.produce(artifact_id, call_context)
        except Exception:
            # Out-of-contract for a conforming ``Producer`` (``interfaces.py``
            # says it must never raise for an expected, routine failure) --
            # but if it does anyway, the latency/GPU time were still
            # genuinely spent, so they are still recorded (succeeded=False,
            # zero tokens: ``usage_sink`` may be empty or partial) before
            # the original exception propagates unchanged. This wrapper
            # does not swallow or convert the exception into a
            # ``ProductionResult`` -- only ``ArtifactGraphStore.record_cost``
            # exists to make sure the attempt is not missing from the cost
            # ledger either way.
            latency_s = time.monotonic() - start
            gpu_seconds = sampler.stop() if sampler is not None else None
            totals = _sum_usage(usage_sink)
            self.last_cost = CostRecord(
                stage=self._stage,
                input_tokens=totals["input_tokens"],
                output_tokens=totals["output_tokens"],
                total_tokens=totals["total_tokens"],
                latency_s=latency_s,
                gpu_seconds=gpu_seconds,
                succeeded=False,
            )
            raise

        latency_s = time.monotonic() - start
        gpu_seconds = sampler.stop() if sampler is not None else None
        totals = _sum_usage(usage_sink)
        self.last_cost = CostRecord(
            stage=self._stage,
            input_tokens=totals["input_tokens"],
            output_tokens=totals["output_tokens"],
            total_tokens=totals["total_tokens"],
            latency_s=latency_s,
            gpu_seconds=gpu_seconds,
            succeeded=(result.status == ProductionStatus.SUCCESS),
        )
        return result


__all__ = ["GpuUtilizationSampler", "CostTrackingProducer"]

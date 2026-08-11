"""Coverage for aggregate_batch()'s orchestration: merging real run_result.json
data with the fallback path for a sample that predates this instrumentation.
Every metric aggregation function this glues together already has its own
dedicated unit tests -- this only covers the wiring between them."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.aggregate import aggregate_batch
from src.evaluation.schemas import CompileResult, RunResult, VerificationEvent


class AggregateBatchTest(unittest.TestCase):
    def test_merges_instrumented_and_pre_instrumentation_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)

            # Sample A: fully instrumented (has run_result.json).
            sample_a = base_dir / "item_a" / "sample_000"
            sample_a.mkdir(parents=True)
            (sample_a / "plan.json").write_text("{}", encoding="utf-8")
            run_result = RunResult(
                run_id="item_a/sample_000",
                producer_model="gpt-4o-mini",
                completed=True,
                compile_success=True,
                compile_result=CompileResult(compile_success=True, engine="tectonic", pdf_exists=True),
                verification_events=[
                    VerificationEvent(
                        run_id="item_a/sample_000", stage="planner",
                        verifier="planner_blueprint_validator", attempt=1, result="pass",
                    )
                ],
            )
            (sample_a / "run_result.json").write_text(run_result.model_dump_json(), encoding="utf-8")

            # Sample B: predates instrumentation -- no run_result.json at
            # all, only the files the pipeline has always written.
            sample_b = base_dir / "item_b" / "sample_000"
            sample_b.mkdir(parents=True)
            (sample_b / "plan.json").write_text("{}", encoding="utf-8")
            (sample_b / "compile.log").write_text(
                "note: Running TeX ...\nnote: Writing `main.pdf` (1.0 KiB)\n", encoding="utf-8"
            )
            (sample_b / "main.pdf").write_bytes(b"%PDF-1.5 fake")

            summary = aggregate_batch(base_dir)

        self.assertEqual(summary["sample_count"], 2)
        # Only sample A contributed real events.
        self.assertEqual(summary["verification_trigger_rate"]["event_level"]["overall"]["checked_count"], 1)
        # Both samples contributed a compile result (one real, one fallback).
        self.assertEqual(summary["compile_success_rate"]["generation_runs"], 2)
        self.assertEqual(summary["compile_success_rate"]["successfully_compiled"], 2)

        by_rel_path = {row["rel_path"]: row for row in summary["per_sample"]}
        self.assertEqual(len(by_rel_path), 2)
        self.assertTrue(by_rel_path["item_a/sample_000"]["has_run_result"])
        self.assertFalse(by_rel_path["item_b/sample_000"]["has_run_result"])
        self.assertTrue(by_rel_path["item_b/sample_000"]["compile_success"])

    def test_empty_directory_produces_zero_sample_summary_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = aggregate_batch(Path(tmp))
        self.assertEqual(summary["sample_count"], 0)
        self.assertIsNone(summary["compile_success_rate"]["compile_success_rate"])


if __name__ == "__main__":
    unittest.main()

"""Round-trip coverage for every new evaluation-layer schema, including the
degrade-gracefully case (no benchmark item / no compile) that -q/--query
single-run mode hits."""

from __future__ import annotations

import unittest

from src.evaluation.schemas import (
    CompileResult,
    ContractCategoryDetail,
    ContractResult,
    RecoveryEvent,
    RunResult,
    TokenUsage,
    VerificationConfigSnapshot,
    VerificationEvent,
)


def _round_trip(model):
    cls = type(model)
    return cls.model_validate_json(model.model_dump_json())


class SchemaRoundTripTest(unittest.TestCase):
    def test_verification_event(self) -> None:
        event = VerificationEvent(
            run_id="run_001", stage="text_agent", verifier="text_semantic_block_validator",
            artifact_id="sec_intro", attempt=1, result="fail", signal_type="invalid_math_delimiter",
            message="boom", producer_model="gpt-4o-mini", failure_id="failure_0001",
        )
        restored = _round_trip(event)
        self.assertEqual(restored, event)

    def test_recovery_event(self) -> None:
        event = RecoveryEvent(
            run_id="run_001", failure_id="failure_0001", stage="planner",
            recovery_action="retry_with_feedback", attempt=2, success=True,
        )
        self.assertEqual(_round_trip(event), event)

    def test_compile_result(self) -> None:
        result = CompileResult(
            compile_success=False, engine="tectonic", return_code=1,
            fatal_error_count=1, warning_count=4, first_error_type="missing_math_delimiter",
            pdf_exists=False,
        )
        self.assertEqual(_round_trip(result), result)

    def test_contract_result_with_details(self) -> None:
        result = ContractResult(
            overall=0.83, required_sections=1.0, required_components=1.0,
            terminology=0.8, notation=0.75,
            details={"required_sections": [ContractCategoryDetail(item_id="intro", status="satisfied")]},
        )
        restored = _round_trip(result)
        self.assertEqual(restored, result)
        self.assertEqual(restored.not_implemented, ["content_coverage", "cross_section_dependency"])

    def test_token_usage(self) -> None:
        usage = TokenUsage(total_input_tokens=100, total_output_tokens=50, total_tokens=150, call_count=3)
        self.assertEqual(_round_trip(usage), usage)

    def test_verification_config_snapshot_defaults_match_current_reality(self) -> None:
        snapshot = VerificationConfigSnapshot()
        # No real toggle system exists yet -- these two are the only
        # mechanisms that genuinely don't exist in the pipeline today.
        self.assertFalse(snapshot.compiler_recovery)
        self.assertFalse(snapshot.image_retry_feedback)

    def test_run_result_degrades_gracefully_with_no_benchmark_item_or_compile(self) -> None:
        result = RunResult(
            run_id="req_20260805_114105",
            producer_model="gpt-4o-mini",
            completed=True,
            compile_success=False,
        )
        restored = _round_trip(result)
        self.assertIsNone(restored.benchmark_id)
        self.assertIsNone(restored.contract_result)
        self.assertIsNone(restored.compile_result)
        self.assertIsNone(restored.token_usage)
        self.assertIsNone(restored.elapsed_time)
        self.assertEqual(restored.verification_events, [])
        self.assertEqual(restored.config_name, "always_on_v1")

    def test_run_result_full_shape(self) -> None:
        result = RunResult(
            run_id="math_real_analysis_0001/sample_000",
            benchmark_id="math_real_analysis_0001",
            producer_model="gpt-4o-mini",
            producer_models={"planner": "gpt-4o-mini", "text": "gpt-4o-mini"},
            completed=True,
            compile_success=True,
            compile_result=CompileResult(compile_success=True, engine="tectonic", pdf_exists=True),
            verification_events=[
                VerificationEvent(
                    run_id="math_real_analysis_0001/sample_000", stage="planner",
                    verifier="planner_blueprint_validator", attempt=1, result="pass",
                )
            ],
            contract_result=ContractResult(overall=0.9),
            token_usage=TokenUsage(call_count=5),
            elapsed_time=42.5,
        )
        self.assertEqual(_round_trip(result), result)


if __name__ == "__main__":
    unittest.main()

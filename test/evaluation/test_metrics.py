"""Coverage for the pure-aggregation metrics: Trigger Rate, Recovery Success
Rate, Compile Success Rate."""

from __future__ import annotations

import unittest

from src.evaluation.metrics.compile_success import compute_compile_success_rate
from src.evaluation.metrics.recovery_success import compute_recovery_success_rate
from src.evaluation.metrics.trigger_rate import compute_trigger_rate
from src.evaluation.schemas import CompileResult, RecoveryEvent, VerificationEvent


def _event(stage, verifier, result, attempt=1, signal_type=None, producer_model=None, failure_id=None):
    return VerificationEvent(
        run_id="run_001", stage=stage, verifier=verifier, attempt=attempt, result=result,
        signal_type=signal_type, producer_model=producer_model, failure_id=failure_id,
    )


class TriggerRateTest(unittest.TestCase):
    def test_overall_and_by_stage_and_signal_type(self) -> None:
        events = [
            _event("planner", "planner_blueprint_validator", "pass"),
            _event("text_agent", "text_semantic_block_validator", "fail", signal_type="json_parse_error"),
            _event("text_agent", "text_semantic_block_validator", "fail", attempt=2, signal_type="content_validation_error"),
            _event("text_agent", "text_semantic_block_validator", "pass", attempt=3),
        ]
        result = compute_trigger_rate(events)["event_level"]
        self.assertEqual(result["overall"], {"checked_count": 4, "triggered_count": 2, "trigger_rate": 0.5})
        self.assertAlmostEqual(result["by_stage"]["text_agent"]["trigger_rate"], 2 / 3)
        self.assertEqual(result["by_stage"]["planner"]["trigger_rate"], 0.0)
        self.assertEqual(
            set(result["by_signal_type"]), {"json_parse_error", "content_validation_error"}
        )
        self.assertEqual(result["by_signal_type"]["json_parse_error"]["share_of_failures"], 0.5)

    def test_initial_attempt_only_excludes_later_retries(self) -> None:
        events = [
            _event("text_agent", "v", "fail", attempt=1),
            _event("text_agent", "v", "fail", attempt=2),
            _event("text_agent", "v", "pass", attempt=3),
        ]
        result = compute_trigger_rate(events)
        self.assertEqual(result["initial_attempt_only"]["overall"]["checked_count"], 1)
        self.assertEqual(result["initial_attempt_only"]["overall"]["trigger_rate"], 1.0)
        self.assertEqual(result["event_level"]["overall"]["checked_count"], 3)

    def test_empty_events_is_none_not_a_crash(self) -> None:
        result = compute_trigger_rate([])
        self.assertIsNone(result["event_level"]["overall"]["trigger_rate"])


class RecoverySuccessRateTest(unittest.TestCase):
    def test_recovered_and_unrecovered_and_no_path_buckets(self) -> None:
        verification_events = [
            _event("text_agent", "v", "fail", failure_id="f1"),
            _event("text_agent", "v", "fail", failure_id="f2"),
            _event("image_agent", "image_asset_validator", "fail", failure_id=None),
        ]
        recovery_events = [
            RecoveryEvent(run_id="run_001", failure_id="f1", stage="text_agent",
                           recovery_action="retry_with_feedback", attempt=2, success=True),
            RecoveryEvent(run_id="run_001", failure_id="f2", stage="text_agent",
                           recovery_action="retry_with_feedback", attempt=2, success=False),
        ]
        result = compute_recovery_success_rate(verification_events, recovery_events)
        self.assertEqual(result["failures_with_recovery_attempt"], 2)
        self.assertEqual(result["successfully_recovered"], 1)
        self.assertEqual(result["recovery_success_rate"], 0.5)
        self.assertEqual(result["detected_failures_without_recovery_path"], 1)
        self.assertEqual(result["detected_failures_without_recovery_attempt"], 0)

    def test_failure_id_with_no_matching_recovery_event_counts_as_without_attempt(self) -> None:
        verification_events = [_event("planner", "v", "fail", failure_id="f1")]
        result = compute_recovery_success_rate(verification_events, [])
        self.assertEqual(result["failures_with_recovery_attempt"], 0)
        self.assertEqual(result["detected_failures_without_recovery_attempt"], 1)
        self.assertIsNone(result["recovery_success_rate"])


class CompileSuccessRateTest(unittest.TestCase):
    def test_success_and_failure_counted_independent_of_warnings(self) -> None:
        results = [
            CompileResult(compile_success=True, engine="tectonic", warning_count=3, pdf_exists=True),
            CompileResult(compile_success=False, engine="tectonic", fatal_error_count=1,
                           first_error_type="missing_math_delimiter"),
            None,  # a run that never reached compile (content_generation_error)
        ]
        result = compute_compile_success_rate(results)
        self.assertEqual(result["generation_runs"], 3)
        self.assertEqual(result["runs_that_reached_compile"], 2)
        self.assertEqual(result["successfully_compiled"], 1)
        self.assertAlmostEqual(result["compile_success_rate"], 1 / 3)
        self.assertEqual(result["runs_with_warnings_only"], 1)
        self.assertEqual(result["runs_with_fatal_errors"], 1)
        self.assertEqual(result["by_first_error_type"], {"missing_math_delimiter": 1})


if __name__ == "__main__":
    unittest.main()

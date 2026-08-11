"""Unit coverage for AttemptTracker's failure/recovery pairing logic, isolated
from any agent or LLM call."""

from __future__ import annotations

import unittest

from src.evaluation.event_logger import AttemptTracker, EventLogger


class EventLoggerTest(unittest.TestCase):
    def test_new_failure_id_is_unique_and_ordered(self) -> None:
        logger = EventLogger(run_id="run_001")
        first = logger.new_failure_id()
        second = logger.new_failure_id()
        self.assertNotEqual(first, second)
        self.assertEqual(first, "run_001:failure_0001")
        self.assertEqual(second, "run_001:failure_0002")

    def test_failure_id_is_globally_unique_across_independent_runs(self) -> None:
        """The bug this guards against: two different requests' EventLoggers
        each mint their own "failure_0001" -- once pooled by aggregate.py
        across a batch, an unprefixed id would collide and one run's
        recovery outcome would silently overwrite the other's."""
        logger_a = EventLogger(run_id="item_a/sample_000")
        logger_b = EventLogger(run_id="item_b/sample_000")
        self.assertNotEqual(logger_a.new_failure_id(), logger_b.new_failure_id())

    def test_total_generation_attempts_excludes_composer_and_compile(self) -> None:
        logger = EventLogger(run_id="run_001")
        logger.log_verification(stage="planner", verifier="v", result="pass", attempt=1)
        logger.log_verification(stage="text_agent", verifier="v", result="fail", attempt=1)
        logger.log_verification(stage="composer", verifier="v", result="pass", attempt=1)
        logger.log_verification(stage="compile", verifier="v", result="pass", attempt=1)
        self.assertEqual(logger.total_generation_attempts, 2)

    def test_total_recovery_attempts_counts_recovery_events(self) -> None:
        logger = EventLogger(run_id="run_001")
        logger.log_recovery(
            failure_id="failure_0001", stage="planner", recovery_action="retry_with_feedback",
            attempt=2, success=True,
        )
        self.assertEqual(logger.total_recovery_attempts, 1)

    def test_sequence_id_is_globally_monotonic_across_verification_and_recovery_events(self) -> None:
        """Guards the outer-loop provenance gap: attempt alone resets to 1 on
        every re-dispatch, so (stage, verifier, artifact_id, attempt) can
        legitimately repeat within one run -- sequence_id must not."""
        logger = EventLogger(run_id="run_001")
        v1 = logger.log_verification(stage="text_agent", verifier="v", result="fail", attempt=1)
        r1 = logger.log_recovery(
            failure_id="f1", stage="text_agent", recovery_action="retry_with_feedback",
            attempt=1, success=False,
        )
        v2 = logger.log_verification(stage="text_agent", verifier="v", result="pass", attempt=1)
        ids = [v1.sequence_id, r1.sequence_id, v2.sequence_id]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(set(ids)), 3)

    def test_outer_iteration_distinguishes_events_that_share_the_same_attempt_number(self) -> None:
        """Simulates run_full_pipeline.py's outer Phase-2 loop re-dispatching
        the same node: a fresh AttemptTracker per outer iteration means
        attempt resets to 1 each time, so outer_iteration is what keeps the
        raw event stream unambiguous."""
        logger = EventLogger(run_id="run_001")
        for outer_iter in (1, 2):
            tracker = AttemptTracker(
                logger, stage="text_agent", verifier="text_semantic_block_validator",
                recovery_action="retry_with_feedback", artifact_id="sec_prerequisites",
                outer_iteration=outer_iter,
            )
            tracker.record_failure(1, "content_validation_error", "bad")
            tracker.record_exhausted()

        events = [e for e in logger.verification_events if e.artifact_id == "sec_prerequisites"]
        self.assertEqual([e.attempt for e in events], [1, 1])
        self.assertEqual([e.outer_iteration for e in events], [1, 2])
        self.assertEqual(len({e.sequence_id for e in events}), 2)


class AttemptTrackerTest(unittest.TestCase):
    def _tracker(self, logger: EventLogger) -> AttemptTracker:
        return AttemptTracker(
            logger,
            stage="text_agent",
            verifier="text_semantic_block_validator",
            recovery_action="retry_with_feedback",
            artifact_id="sec_intro",
            section_id="sec_intro",
        )

    def test_fail_then_fail_then_success_pairs_each_failure_with_its_next_attempt(self) -> None:
        logger = EventLogger(run_id="run_001")
        tracker = self._tracker(logger)

        tracker.record_failure(1, "invalid_math_delimiter", "bad delimiter")
        tracker.record_failure(2, "invalid_math_delimiter", "still bad")
        tracker.record_success(3, resulting_artifact_id="sec_intro.blocks.json")

        events = logger.verification_events
        self.assertEqual([e.result for e in events], ["fail", "fail", "pass"])
        self.assertEqual([e.attempt for e in events], [1, 2, 3])
        fail1_id, fail2_id = events[0].failure_id, events[1].failure_id
        self.assertIsNotNone(fail1_id)
        self.assertIsNotNone(fail2_id)
        self.assertNotEqual(fail1_id, fail2_id)
        self.assertIsNone(events[2].failure_id)

        recoveries = logger.recovery_events
        self.assertEqual(len(recoveries), 2)
        # The first failure is resolved by attempt 2's (failed) outcome.
        self.assertEqual(recoveries[0].failure_id, fail1_id)
        self.assertEqual(recoveries[0].attempt, 2)
        self.assertFalse(recoveries[0].success)
        # The second failure is resolved by attempt 3's (successful) outcome.
        self.assertEqual(recoveries[1].failure_id, fail2_id)
        self.assertEqual(recoveries[1].attempt, 3)
        self.assertTrue(recoveries[1].success)
        self.assertEqual(recoveries[1].resulting_artifact_id, "sec_intro.blocks.json")

    def test_all_attempts_failing_leaves_the_last_failure_unpaired(self) -> None:
        logger = EventLogger(run_id="run_001")
        tracker = self._tracker(logger)

        tracker.record_failure(1, "schema_validation_error", "e1")
        tracker.record_failure(2, "schema_validation_error", "e2")
        tracker.record_failure(3, "schema_validation_error", "e3")
        tracker.record_exhausted()

        events = logger.verification_events
        self.assertEqual([e.result for e in events], ["fail", "fail", "fail"])
        failure_ids = [e.failure_id for e in events]
        self.assertEqual(len(set(failure_ids)), 3)

        recoveries = logger.recovery_events
        self.assertEqual(len(recoveries), 2)
        self.assertTrue(all(not r.success for r in recoveries))
        recovered_failure_ids = {r.failure_id for r in recoveries}
        # The first two failures each got a next-attempt outcome; the third
        # (last) failure never did, since nothing was retried after it.
        self.assertEqual(recovered_failure_ids, set(failure_ids[:2]))
        self.assertNotIn(failure_ids[2], recovered_failure_ids)

    def test_first_attempt_success_produces_no_recovery_event(self) -> None:
        logger = EventLogger(run_id="run_001")
        tracker = self._tracker(logger)

        tracker.record_success(1)

        self.assertEqual(len(logger.verification_events), 1)
        self.assertEqual(logger.verification_events[0].result, "pass")
        self.assertEqual(logger.recovery_events, [])

    def test_none_logger_is_safe_to_use(self) -> None:
        tracker = self._tracker(None)  # type: ignore[arg-type]
        tracker.record_failure(1, "x", "msg")
        tracker.record_success(2)
        tracker.record_exhausted()  # must not raise


if __name__ == "__main__":
    unittest.main()

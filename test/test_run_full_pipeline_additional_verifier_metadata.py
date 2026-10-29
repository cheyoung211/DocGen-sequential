"""Coverage for scripts/run_full_pipeline.py's AdditionalVerifierRunMetadata
wiring (Revision (Additional verifiers), Stage 0 / P1).

Importing run_full_pipeline.py pulls in its real top-level dependencies
(LLMClient, SDXLGenerator, FluxGenerator) -- this is the one module that
genuinely needs them at import time, so this test is slow (tens of
seconds) rather than light. See test_pipeline_dependency_ordering.py for
a separate, pre-existing, unrelated import issue in the same module
(stale reference to a renamed/removed `_dependency_ordered` helper) --
not touched here, out of this stage's scope.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import run_full_pipeline as rfp  # noqa: E402  (sys.path setup must come first)

from src.evaluation.event_logger import EventLogger  # noqa: E402


class CurrentCodeCommitTest(unittest.TestCase):
    def test_matches_real_git_head(self) -> None:
        expected = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(rfp._current_code_commit(), expected)

    def test_is_cached_across_calls(self) -> None:
        # @lru_cache(maxsize=1): confirm this is actually the same cached
        # result object, not just two subprocess calls that happen to
        # agree -- the cache exists so a batch run over many benchmark
        # items doesn't shell out to git once per item.
        self.assertIs(rfp._current_code_commit(), rfp._current_code_commit())


class GitIsDirtyTest(unittest.TestCase):
    def test_matches_real_git_status(self) -> None:
        expected = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        self.assertEqual(rfp._git_is_dirty(), expected)


class TextAgentPromptHashTest(unittest.TestCase):
    def test_matches_a_direct_hash_of_the_live_system_prompt(self) -> None:
        from src.agents.text_agent import SYSTEM_PROMPT

        expected = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
        self.assertEqual(rfp._text_agent_prompt_hash(), expected)


class PlannerPromptHashTest(unittest.TestCase):
    def test_matches_a_direct_hash_of_the_live_planner_templates(self) -> None:
        from src.agents.planner import (
            PLANNER_LEGACY_OUTPUT_FORMAT_PROMPT,
            PLANNER_NO_IMAGES_PROMPT,
            PLANNER_REQUIRED_SECTIONS_PROMPT,
            PLANNER_STRATEGY_PROMPT,
        )

        combined = (
            PLANNER_STRATEGY_PROMPT
            + PLANNER_NO_IMAGES_PROMPT
            + PLANNER_REQUIRED_SECTIONS_PROMPT
            + PLANNER_LEGACY_OUTPUT_FORMAT_PROMPT
        )
        expected = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
        self.assertEqual(rfp._planner_prompt_hash(), expected)

    def test_differs_from_the_text_agent_prompt_hash(self) -> None:
        # Sanity check against the trivial bug of accidentally hashing the
        # same string for both agents.
        self.assertNotEqual(rfp._planner_prompt_hash(), rfp._text_agent_prompt_hash())


class BuildRunResultAdditionalVerifierMetadataTest(unittest.TestCase):
    def test_populates_metadata_with_real_commit_and_honest_disabled_flags(self) -> None:
        event_logger = EventLogger(run_id="test_run_stage0")
        result = rfp._build_run_result(
            request_id="test_run_stage0",
            status="success",
            error=None,
            event_logger=event_logger,
            start_time=None,
            token_usage_records=None,
            benchmark_item=None,
            producer_models={"text": "gpt-4o-mini"},
            project_dir=REPO_ROOT,
            text_agent_enabled_validators={"forbidden_environment": True, "table_columns": False},
            planner_enabled_validators={"plain_title": True},
            assembler_enabled_validators={"layout_policy": False},
        )
        metadata = result.additional_verifier_metadata
        self.assertEqual(metadata.code_commit, rfp._current_code_commit())
        self.assertEqual(metadata.git_dirty, rfp._git_is_dirty())
        self.assertEqual(metadata.text_agent_prompt_hash, rfp._text_agent_prompt_hash())
        self.assertEqual(metadata.planner_prompt_hash, rfp._planner_prompt_hash())
        self.assertEqual(
            metadata.text_agent_enabled_validators_snapshot,
            {"forbidden_environment": True, "table_columns": False},
        )
        self.assertEqual(metadata.planner_enabled_validators_snapshot, {"plain_title": True})
        self.assertEqual(metadata.assembler_enabled_validators_snapshot, {"layout_policy": False})
        # None of Fix A/B/C exist yet (Stage 0) -- every flag must stay
        # honestly False, never a placeholder True for an unbuilt verifier.
        self.assertFalse(metadata.markdown_emphasis_enabled)
        self.assertFalse(metadata.math_verifier_enabled)
        self.assertFalse(metadata.dependency_verifier_enabled)

    def test_defaults_all_three_snapshots_to_empty_when_not_supplied(self) -> None:
        # Every pre-Stage-0 call shape (none exist yet, but the
        # parameters are optional) must not crash -- empty snapshots, not
        # an error.
        event_logger = EventLogger(run_id="test_run_stage0_no_validators")
        result = rfp._build_run_result(
            request_id="test_run_stage0_no_validators",
            status="success",
            error=None,
            event_logger=event_logger,
            start_time=None,
            token_usage_records=None,
            benchmark_item=None,
            producer_models={"text": "gpt-4o-mini"},
            project_dir=REPO_ROOT,
        )
        metadata = result.additional_verifier_metadata
        self.assertEqual(metadata.text_agent_enabled_validators_snapshot, {})
        self.assertEqual(metadata.planner_enabled_validators_snapshot, {})
        self.assertEqual(metadata.assembler_enabled_validators_snapshot, {})


if __name__ == "__main__":
    unittest.main()

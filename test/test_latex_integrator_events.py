"""Coverage that LatexIntegratorAgent.integrate()/compile_pdf() actually emit
VerificationEvents at each of their 4 gates plus the compile stage, when an
EventLogger is supplied. test_semantic_integrator.py already covers the
composer's rendering/validation *behavior* unchanged -- this only covers the
new, additive event-logging layer on top of it."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.latex_integrator import LatexIntegratorAgent
from src.common.schemas import ContentBlock, SemanticBlockType, TextAgentOutput
from src.common.state import (
    DocumentBlueprint,
    DocumentGraph,
    DocumentKind,
    DocumentNode,
    DocumentSpec,
    LayoutBlock,
    LayoutBlockKind,
    NodeType,
    SectionLayout,
)
from src.evaluation.event_logger import EventLogger


def _graph() -> DocumentGraph:
    blocks = [LayoutBlock(id="intro", kind=LayoutBlockKind.PARAGRAPH)]
    blueprint = DocumentBlueprint(
        document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Event logging test"),
        document_class="article",
        sections=[SectionLayout(section_id="overview", title="Overview", blocks=blocks)],
    )
    graph = DocumentGraph()
    graph.add_node(DocumentNode(id="overview", type=NodeType.SECTION, title="Overview"))
    graph.set_blueprint(blueprint)
    return graph


def _write_blocks_json(sections_dir: Path) -> None:
    output = TextAgentOutput(
        request_id="events-test",
        section_id="overview",
        blocks=[ContentBlock(block_id="intro", type=SemanticBlockType.PARAGRAPH, content="Hello.")],
    )
    (sections_dir / "overview.blocks.json").write_text(output.model_dump_json(indent=2), encoding="utf-8")


class ComposerEventLoggingTest(unittest.TestCase):
    def test_successful_integrate_logs_a_pass_event_per_gate_and_a_compile_event(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                graph = _graph()
                sections_dir = Path("outputs/generations/events-test/sections")
                sections_dir.mkdir(parents=True)
                _write_blocks_json(sections_dir)

                event_logger = EventLogger(run_id="events-test")
                # Force "no engine resolvable" deterministically, regardless
                # of whether tectonic/pdflatex actually happen to be
                # installed on the machine running this test (unlike
                # test_semantic_integrator.py's engine="" trick, which only
                # produces "no engine" incidentally, when nothing is
                # installed -- not a safe assumption to build a test on).
                with patch("shutil.which", return_value=None), \
                     patch.dict(os.environ, {"PDFLATEX_BIN": ""}):
                    success = LatexIntegratorAgent(engine="").integrate(
                        "events-test", graph, event_logger=event_logger
                    )
                self.assertFalse(success)  # no engine resolvable -> compile can't succeed

                verifiers = {e.verifier: e.result for e in event_logger.verification_events}
                self.assertEqual(
                    verifiers,
                    {
                        "layout_policy_validator": "pass",
                        "figure_asset_completeness_validator": "pass",
                        "composer_semantic_block_validator": "pass",
                        "figure_single_use_validator": "pass",
                        "reference_validator": "pass",
                        "document_frame_validator": "pass",
                        "latex_compiler": "fail",
                    },
                )
                compile_event = next(
                    e for e in event_logger.verification_events if e.verifier == "latex_compiler"
                )
                self.assertEqual(compile_event.signal_type, "no_latex_engine_found")
                self.assertEqual(compile_event.stage, "compile")

                self.assertIsNotNone(event_logger.compile_result)
                self.assertFalse(event_logger.compile_result.compile_success)
                self.assertEqual(event_logger.compile_result.first_error_type, "no_latex_engine_found")

                compile_result_path = Path("outputs/generations/events-test/compile_result.json")
                self.assertTrue(compile_result_path.is_file())
            finally:
                os.chdir(original_cwd)

    def test_missing_blocks_json_logs_a_fail_event_before_reraising(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                graph = _graph()
                event_logger = EventLogger(run_id="events-test")

                with self.assertRaises(FileNotFoundError):
                    LatexIntegratorAgent(engine="").integrate(
                        "events-test", graph, event_logger=event_logger
                    )

                composer_events = [
                    e for e in event_logger.verification_events
                    if e.verifier == "composer_semantic_block_validator"
                ]
                self.assertEqual(len(composer_events), 1)
                self.assertEqual(composer_events[0].result, "fail")
                self.assertEqual(composer_events[0].signal_type, "missing_semantic_artifact")
                # No inner retry loop exists in the composer -- this failure
                # has no recovery path, so it must not carry a failure_id.
                self.assertIsNone(composer_events[0].failure_id)
            finally:
                os.chdir(original_cwd)

    def test_no_event_logger_is_fully_backward_compatible(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                graph = _graph()
                sections_dir = Path("outputs/generations/events-test/sections")
                sections_dir.mkdir(parents=True)
                _write_blocks_json(sections_dir)

                LatexIntegratorAgent(engine="").integrate("events-test", graph)  # no event_logger

                self.assertFalse(
                    Path("outputs/generations/events-test/compile_result.json").exists()
                )
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()

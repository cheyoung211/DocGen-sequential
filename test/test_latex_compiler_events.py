"""Coverage that LatexCompiler.compile_pdf() actually emits a VerificationEvent
plus a classified CompileResult when an EventLogger is supplied."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.latex_compiler import LatexCompiler
from src.agents.latex_assembler import LatexAssembler
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
from src.evaluation.schemas import RecoveryAction


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


class CompilerEventLoggingTest(unittest.TestCase):
    def test_no_engine_failure_logs_event_and_classifies_recovery_action(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                graph = _graph()
                sections_dir = Path("outputs/generations/events-test/sections")
                sections_dir.mkdir(parents=True)
                _write_blocks_json(sections_dir)
                LatexAssembler().assemble("events-test", graph)

                event_logger = EventLogger(run_id="events-test")
                # Force "no engine resolvable" deterministically, regardless
                # of whether tectonic/pdflatex actually happen to be
                # installed on the machine running this test.
                with patch("shutil.which", return_value=None), \
                     patch.dict(os.environ, {"PDFLATEX_BIN": ""}):
                    success = LatexCompiler(engine="").compile_pdf(
                        "events-test", event_logger=event_logger
                    )
                self.assertFalse(success)

                compile_event = next(
                    e for e in event_logger.verification_events if e.verifier == "latex_compiler"
                )
                self.assertEqual(compile_event.result, "fail")
                self.assertEqual(compile_event.signal_type, "no_latex_engine_found")
                self.assertEqual(compile_event.stage, "compile")

                self.assertIsNotNone(event_logger.compile_result)
                self.assertFalse(event_logger.compile_result.compile_success)
                self.assertEqual(event_logger.compile_result.first_error_type, "no_latex_engine_found")
                self.assertEqual(
                    event_logger.compile_result.recovery_action,
                    RecoveryAction.INFRASTRUCTURE_FAILURE,
                )

                compile_result_path = Path("outputs/generations/events-test/compile_result.json")
                self.assertTrue(compile_result_path.is_file())
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
                LatexAssembler().assemble("events-test", graph)

                LatexCompiler(engine="").compile_pdf("events-test")  # no event_logger

                self.assertFalse(
                    Path("outputs/generations/events-test/compile_result.json").exists()
                )
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()

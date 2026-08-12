"""Coverage that LatexAssembler.assemble() actually emits VerificationEvents at
each of its gates when an EventLogger is supplied. test_latex_assembler.py
already covers the composer's rendering/validation *behavior* unchanged --
this only covers the additive event-logging layer on top of it."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

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
    def test_successful_assemble_logs_a_pass_event_per_gate(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                graph = _graph()
                sections_dir = Path("outputs/generations/events-test/sections")
                sections_dir.mkdir(parents=True)
                _write_blocks_json(sections_dir)

                event_logger = EventLogger(run_id="events-test")
                LatexAssembler().assemble("events-test", graph, event_logger=event_logger)

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
                    },
                )
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
                    LatexAssembler().assemble("events-test", graph, event_logger=event_logger)

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

    def test_disabled_gate_validators_skip_both_the_check_and_its_event(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                graph = _graph()
                sections_dir = Path("outputs/generations/events-test/sections")
                sections_dir.mkdir(parents=True)
                _write_blocks_json(sections_dir)

                event_logger = EventLogger(run_id="events-test")
                assembler = LatexAssembler(
                    enabled_validators={
                        "layout_policy": False,
                        "figure_asset_completeness": False,
                        "figure_single_use": False,
                        "reference": False,
                        "document_frame": False,
                    }
                )
                assembler.assemble("events-test", graph, event_logger=event_logger)

                verifiers = {e.verifier for e in event_logger.verification_events}
                # Only the always-on structural check remains -- every
                # disabled gate neither runs nor logs an event for itself.
                self.assertEqual(verifiers, {"composer_semantic_block_validator"})
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()

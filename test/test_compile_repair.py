"""End-to-end coverage for the compile-triggered block regeneration path.

Unlike the other tests in this directory, this one runs the real ``tectonic``
engine (no mocking of LatexCompiler) against a deliberately broken block, so
it proves the whole chain works against tectonic's actual log format, not
just against a hand-written log fixture:

    LatexAssembler renders `% <layout-anchor:...>` comments
    -> a real tectonic compile fails with a real, located error
    -> compile_log_parser.extract_error_locations parses that real log
    -> LatexAssembler.locate_block_at_line resolves the error to one block
    -> TextAgent.regenerate_blocks fixes only that block, leaving its
       sibling block's content untouched
    -> re-assemble + recompile succeeds

Skipped automatically if tectonic isn't installed on the machine running
the suite (see LatexCompiler._resolve_engine).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import List

from src.agents.latex_assembler import LatexAssembler
from src.agents.latex_compiler import LatexCompiler
from src.agents.text_agent import TextAgent
from src.common.schemas import ContentBlock, SemanticBlockType, TextAgentOutput
from src.common.state import (
    DocumentBlueprint,
    DocumentGraph,
    DocumentKind,
    DocumentNode,
    DocumentSpec,
    LayoutBlock,
    LayoutBlockKind,
    NodeStatus,
    NodeType,
    SectionLayout,
)
from src.evaluation.compile_log_parser import classify_recovery_action, extract_error_locations
from src.evaluation.event_logger import AttemptTracker, EventLogger
from src.evaluation.schemas import RecoveryAction, Stage
from scripts.llm_client import LLMClient


class ScriptedLLMClient:
    """Duck-typed against the subset of LLMClient TextAgent uses -- same
    helper test_text_agent.py already introduces for this purpose."""

    def __init__(self, responses: List[str], model_name: str = "fake-model") -> None:
        self._responses = responses
        self.model_name = model_name
        self.calls = 0

    def generate(self, system_prompt, user_prompt, temperature=0.3, max_new_tokens=None, usage_sink=None):
        response = self._responses[self.calls]
        self.calls += 1
        return response

    @staticmethod
    def extract_json_block(text: str) -> str:
        return LLMClient.extract_json_block(text)


def _graph() -> DocumentGraph:
    blocks = [
        LayoutBlock(id="intro", kind=LayoutBlockKind.PARAGRAPH),
        LayoutBlock(id="broken", kind=LayoutBlockKind.PARAGRAPH),
    ]
    blueprint = DocumentBlueprint(
        document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Compile repair test"),
        document_class="article",
        sections=[SectionLayout(section_id="sec_derivation", title="Derivation", blocks=blocks)],
    )
    graph = DocumentGraph()
    graph.add_node(DocumentNode(id="sec_derivation", type=NodeType.SECTION, title="Derivation"))
    graph.set_blueprint(blueprint)
    return graph


def _write_blocks_json(sections_dir: Path) -> None:
    output = TextAgentOutput(
        request_id="repair-test",
        section_id="sec_derivation",
        blocks=[
            ContentBlock(block_id="intro", type=SemanticBlockType.PARAGRAPH, content="This part is fine."),
            # A bare '\times' outside '$...$' -- this is a genuine failure mode
            # observed in this repo's own past output (outputs/batch_test_07281829/
            # basic/sample_003), predating the bare_math_notation TextAgent
            # validator: it passes every *other* content check but tectonic
            # rejects it outright with "Missing $ inserted", exactly the
            # "content looks fine but isn't valid LaTeX" gap
            # RecoveryAction.REGENERATE_CONTENT exists for.
            ContentBlock(
                block_id="broken", type=SemanticBlockType.PARAGRAPH,
                content="The scale factor is 3\\times3 overall.",
            ),
        ],
    )
    (sections_dir / "sec_derivation.blocks.json").write_text(output.model_dump_json(indent=2), encoding="utf-8")


@unittest.skipUnless(shutil.which("tectonic"), "tectonic is not installed")
class CompileRepairTest(unittest.TestCase):
    def test_block_level_regeneration_fixes_a_real_tectonic_failure(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                graph = _graph()
                sections_dir = Path("outputs/generations/repair-test/sections")
                sections_dir.mkdir(parents=True)
                _write_blocks_json(sections_dir)

                assembler = LatexAssembler()
                assembler.assemble("repair-test", graph)

                rendered = (sections_dir / "sec_derivation.tex").read_text(encoding="utf-8")
                self.assertIn("% <layout-anchor:intro>", rendered)
                self.assertIn("% <layout-anchor:broken>", rendered)

                event_logger = EventLogger(run_id="repair-test")
                compiler = LatexCompiler()
                first_success = compiler.compile_pdf(
                    "repair-test", output_dir="outputs/generations", event_logger=event_logger
                )
                self.assertFalse(first_success, "the bare '\\times' should make the first compile fail")

                log_text = Path("outputs/generations/repair-test/compile.log").read_text(encoding="utf-8")
                engine_name = event_logger.compile_result.engine
                locations = extract_error_locations(log_text, engine_name)
                self.assertEqual(len(locations), 1)
                location = locations[0]
                self.assertEqual(location.section_id, "sec_derivation")
                self.assertEqual(
                    classify_recovery_action(location.error_type), RecoveryAction.REGENERATE_CONTENT
                )

                section_tex_path = Path("outputs/generations/repair-test/sections/sec_derivation.tex")
                block_id = LatexAssembler.locate_block_at_line(section_tex_path, location.line)
                self.assertEqual(block_id, "broken")

                tracker = AttemptTracker(
                    event_logger, stage=Stage.COMPILE, verifier="latex_compiler",
                    recovery_action=RecoveryAction.REGENERATE_CONTENT.value,
                    artifact_id="repair-test", section_id=location.section_id,
                )
                tracker.record_failure(1, location.error_type, location.message)

                llm = ScriptedLLMClient([
                    json.dumps({
                        "content_blocks": [
                            {
                                "block_id": "broken", "type": "paragraph",
                                "content": "The scale factor is $3\\times3$ overall.",
                            }
                        ],
                        "tool_requests": [],
                    })
                ])
                text_agent = TextAgent(llm_client=llm)
                output = text_agent.regenerate_blocks(
                    location.section_id, block_id, location.message, graph, "repair-test",
                    output_dir="outputs/generations", event_logger=event_logger,
                )
                self.assertIsNotNone(output)
                self.assertEqual(llm.calls, 1)
                self.assertEqual(graph.nodes["sec_derivation"].status, NodeStatus.DRAFTED)

                # The untouched sibling block must survive regeneration verbatim --
                # this is the whole point of block-level repair over redrafting the
                # entire section.
                fixed_blocks = {b.block_id: b.content for b in output.blocks}
                self.assertEqual(fixed_blocks["intro"], "This part is fine.")
                self.assertEqual(fixed_blocks["broken"], "The scale factor is $3\\times3$ overall.")

                assembler.assemble("repair-test", graph)
                second_success = compiler.compile_pdf(
                    "repair-test", output_dir="outputs/generations", event_logger=event_logger
                )
                self.assertTrue(second_success, "recompiling after the block-level fix should now succeed")
                tracker.record_success(2)

                recoveries = event_logger.recovery_events
                self.assertEqual(len(recoveries), 1)
                self.assertTrue(recoveries[0].success)
                self.assertEqual(recoveries[0].stage, Stage.COMPILE)
                self.assertEqual(recoveries[0].recovery_action, RecoveryAction.REGENERATE_CONTENT.value)
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()

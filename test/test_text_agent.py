"""End-to-end coverage that TextAgent.process_node's event_logger wiring
actually pairs a real failure with its real recovery -- test_event_logger.py
covers AttemptTracker's pairing logic in isolation; this is the one test
that proves text_agent.py's own retry loop calls it correctly.

No existing convention in this repo mocks LLMClient (see planning notes),
so this introduces a small duck-typed ScriptedLLMClient matching the one
injection point TextAgent already exposes (``llm_client=``).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional

from src.agents.text_agent import TextAgent
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
from src.evaluation.event_logger import EventLogger
from scripts.llm_client import LLMClient


class ScriptedLLMClient:
    """Returns one canned response per call, in order. Duck-typed against the
    subset of LLMClient that TextAgent actually uses."""

    def __init__(self, responses: List[str], model_name: str = "fake-model") -> None:
        self._responses = responses
        self.model_name = model_name
        self.calls = 0

    def generate(self, system_prompt, user_prompt, temperature=0.3, max_new_tokens=None, usage_sink=None):
        response = self._responses[self.calls]
        self.calls += 1
        if usage_sink is not None:
            usage_sink.append({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        return response

    @staticmethod
    def extract_json_block(text: str) -> str:
        return LLMClient.extract_json_block(text)


def _single_block_graph() -> DocumentGraph:
    block = LayoutBlock(id="p1", kind=LayoutBlockKind.PARAGRAPH)
    blueprint = DocumentBlueprint(
        document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Test document"),
        document_class="article",
        sections=[SectionLayout(section_id="sec_intro", title="Introduction", blocks=[block])],
    )
    graph = DocumentGraph()
    graph.add_node(DocumentNode(id="sec_intro", type=NodeType.SECTION, title="Introduction"))
    graph.set_blueprint(blueprint)
    return graph


class TextAgentRetryRecoveryTest(unittest.TestCase):
    def test_failure_then_success_emits_matching_verification_and_recovery_events(self) -> None:
        graph = _single_block_graph()
        llm = ScriptedLLMClient([
            "{}",  # attempt 1: no content_blocks key -> content_validation_error
            json.dumps({
                "content_blocks": [
                    {"block_id": "p1", "type": "paragraph", "content": "Hello world."}
                ],
                "tool_requests": [],
            }),
        ])
        agent = TextAgent(llm_client=llm)
        event_logger = EventLogger(run_id="run_test")
        usage_sink: list = []

        with tempfile.TemporaryDirectory() as tmp:
            output = agent.process_node(
                "sec_intro",
                graph,
                "req_test",
                output_dir=tmp,
                event_logger=event_logger,
                usage_sink=usage_sink,
            )

        self.assertIsNotNone(output)
        self.assertEqual(llm.calls, 2)
        self.assertEqual(len(usage_sink), 2)

        events = event_logger.verification_events
        self.assertEqual([e.result for e in events], ["fail", "pass"])
        self.assertEqual(events[0].stage, "text_agent")
        self.assertEqual(events[0].verifier, "text_semantic_block_validator")
        self.assertEqual(events[0].signal_type, "content_validation_error")
        self.assertEqual(events[0].artifact_id, "sec_intro")
        self.assertEqual(events[0].producer_model, "fake-model")
        self.assertIsNotNone(events[0].failure_id)
        self.assertIsNone(events[1].failure_id)

        recoveries = event_logger.recovery_events
        self.assertEqual(len(recoveries), 1)
        self.assertEqual(recoveries[0].failure_id, events[0].failure_id)
        self.assertTrue(recoveries[0].success)
        self.assertEqual(recoveries[0].attempt, 2)
        self.assertEqual(recoveries[0].resulting_artifact_id, "sec_intro.blocks.json")

        self.assertEqual(graph.nodes["sec_intro"].status, NodeStatus.DRAFTED)

    def test_exhausting_every_attempt_marks_node_error_and_logs_every_failure(self) -> None:
        graph = _single_block_graph()
        llm = ScriptedLLMClient(["{}", "{}", "{}"])
        agent = TextAgent(llm_client=llm)
        event_logger = EventLogger(run_id="run_test")

        with tempfile.TemporaryDirectory() as tmp:
            output = agent.process_node(
                "sec_intro",
                graph,
                "req_test",
                output_dir=tmp,
                max_attempts=3,
                event_logger=event_logger,
            )

        self.assertIsNone(output)
        self.assertEqual(llm.calls, 3)
        self.assertEqual(
            [e.result for e in event_logger.verification_events], ["fail", "fail", "fail"]
        )
        # Only the first two failures got a subsequent attempt to pair with.
        self.assertEqual(len(event_logger.recovery_events), 2)
        self.assertTrue(all(not r.success for r in event_logger.recovery_events))
        self.assertEqual(graph.nodes["sec_intro"].status, NodeStatus.ERROR)


class TextAgentPreRepairSnapshotTest(unittest.TestCase):
    """Revision (Additional verifiers) Stage 0 / P2 repro: draft.tex alone
    does not preserve pre-repair content -- it is a single, fixed path
    that the next successful _run_generation_loop pass overwrites in
    place. _snapshot_before_overwrite closes that gap locally, without
    needing ArtifactGraphStore.save_checkpoint() wired into this pipeline
    (it isn't -- see docs/av_stage0_scope_and_contracts.md)."""

    @staticmethod
    def _response_for(content: str) -> str:
        return json.dumps({
            "content_blocks": [{"block_id": "p1", "type": "paragraph", "content": content}],
            "tool_requests": [],
        })

    def test_first_draft_creates_no_pre_repair_snapshot(self) -> None:
        graph = _single_block_graph()
        llm = ScriptedLLMClient([self._response_for("Hello world.")])
        agent = TextAgent(llm_client=llm)

        with tempfile.TemporaryDirectory() as tmp:
            agent.process_node("sec_intro", graph, "req_test", output_dir=tmp)
            sections_dir = Path(tmp) / "req_test" / "sections"
            self.assertEqual(list(sections_dir.glob("sec_intro.pre_repair_*")), [])

    def test_process_node_called_again_preserves_the_prior_draft(self) -> None:
        """Simulates run_full_pipeline.py's section-level compile-repair
        branch, which calls process_node a second time on an already-
        drafted node (see this module's DocGenPipeline._repair_and_
        recompile -- distinct from the block-level regenerate_blocks
        branch covered below)."""
        graph = _single_block_graph()
        llm = ScriptedLLMClient([
            self._response_for("Hello world."),
            self._response_for("Repaired hello world."),
        ])
        agent = TextAgent(llm_client=llm)

        with tempfile.TemporaryDirectory() as tmp:
            agent.process_node("sec_intro", graph, "req_test", output_dir=tmp)
            agent.process_node("sec_intro", graph, "req_test", output_dir=tmp)

            sections_dir = Path(tmp) / "req_test" / "sections"
            snapshots = sorted(sections_dir.glob("sec_intro.pre_repair_*.blocks.json"))
            self.assertEqual(len(snapshots), 1)
            preserved = json.loads(snapshots[0].read_text(encoding="utf-8"))
            self.assertEqual(preserved["blocks"][0]["content"], "Hello world.")

            current = json.loads((sections_dir / "sec_intro.blocks.json").read_text(encoding="utf-8"))
            self.assertEqual(current["blocks"][0]["content"], "Repaired hello world.")

            preserved_draft = (sections_dir / "sec_intro.pre_repair_1.draft.tex").read_text(encoding="utf-8")
            self.assertIn("Hello world.", preserved_draft)

    def test_regenerate_blocks_preserves_the_pre_repair_content(self) -> None:
        graph = _single_block_graph()
        llm = ScriptedLLMClient([
            self._response_for("Hello world."),
            self._response_for("Fixed hello world."),
        ])
        agent = TextAgent(llm_client=llm)

        with tempfile.TemporaryDirectory() as tmp:
            agent.process_node("sec_intro", graph, "req_test", output_dir=tmp)
            agent.regenerate_blocks(
                "sec_intro", "p1", "compiler said: boom", graph, "req_test", output_dir=tmp
            )

            sections_dir = Path(tmp) / "req_test" / "sections"
            snapshots = sorted(sections_dir.glob("sec_intro.pre_repair_*.blocks.json"))
            self.assertEqual(len(snapshots), 1)
            preserved = json.loads(snapshots[0].read_text(encoding="utf-8"))
            self.assertEqual(preserved["blocks"][0]["content"], "Hello world.")

            current = json.loads((sections_dir / "sec_intro.blocks.json").read_text(encoding="utf-8"))
            self.assertEqual(current["blocks"][0]["content"], "Fixed hello world.")

    def test_two_repair_rounds_create_two_sequentially_numbered_snapshots(self) -> None:
        graph = _single_block_graph()
        llm = ScriptedLLMClient([
            self._response_for("v1"),
            self._response_for("v2"),
            self._response_for("v3"),
        ])
        agent = TextAgent(llm_client=llm)

        with tempfile.TemporaryDirectory() as tmp:
            agent.process_node("sec_intro", graph, "req_test", output_dir=tmp)
            agent.regenerate_blocks("sec_intro", "p1", "err1", graph, "req_test", output_dir=tmp)
            agent.regenerate_blocks("sec_intro", "p1", "err2", graph, "req_test", output_dir=tmp)

            sections_dir = Path(tmp) / "req_test" / "sections"
            snap1 = json.loads(
                (sections_dir / "sec_intro.pre_repair_1.blocks.json").read_text(encoding="utf-8")
            )
            snap2 = json.loads(
                (sections_dir / "sec_intro.pre_repair_2.blocks.json").read_text(encoding="utf-8")
            )
            self.assertEqual(snap1["blocks"][0]["content"], "v1")
            self.assertEqual(snap2["blocks"][0]["content"], "v2")
            current = json.loads((sections_dir / "sec_intro.blocks.json").read_text(encoding="utf-8"))
            self.assertEqual(current["blocks"][0]["content"], "v3")


if __name__ == "__main__":
    unittest.main()

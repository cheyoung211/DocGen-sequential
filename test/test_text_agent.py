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


if __name__ == "__main__":
    unittest.main()

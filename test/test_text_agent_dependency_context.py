"""Regression tests for TextAgent._build_prompt_from_node's dependency
context: a node whose DocumentNode.dependencies were wired by
PlannerAgent._validate_blueprint (see test_planner_cross_section_dependencies.py)
must see a condensed excerpt + key_points + relation/description for each
already-drafted dependency, and silently see nothing for one that isn't
drafted yet."""

from __future__ import annotations

import unittest

from src.agents.text_agent import TextAgent
from src.common.state import (
    DocumentBlueprint,
    DocumentGraph,
    DocumentKind,
    DocumentNode,
    DocumentSpec,
    NodeStatus,
    NodeType,
)

# _build_prompt_from_node never calls the LLM client.
_STUB_LLM_CLIENT = object()


def _graph_with(intro_status: NodeStatus) -> DocumentGraph:
    graph = DocumentGraph()
    graph.add_node(
        DocumentNode(
            id="sec_intro",
            type=NodeType.SECTION,
            title="Introduction",
            status=intro_status,
            content="We classify volatility regimes into three classes: low, medium and crisis.",
            spec={"key_points": ["Define 3 volatility regimes", "Justify thresholds"]},
        )
    )
    graph.add_node(
        DocumentNode(
            id="sec_results",
            type=NodeType.SECTION,
            title="Results",
            status=NodeStatus.PENDING,
            dependencies=["sec_intro"],
            spec={
                "dependency_notes": [
                    {
                        "depends_on": "sec_intro",
                        "relation": "must_reuse",
                        "description": "Results must reuse the exact taxonomy defined in the introduction.",
                    }
                ]
            },
        )
    )
    graph.set_blueprint(
        DocumentBlueprint(
            document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Dependency Context Test"),
            document_class="article",
        )
    )
    return graph


class DependencyContextPromptTest(unittest.TestCase):
    def test_drafted_dependency_is_surfaced(self) -> None:
        graph = _graph_with(intro_status=NodeStatus.DRAFTED)
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        prompt = agent._build_prompt_from_node(graph.nodes["sec_results"], graph, layout_blocks=[])

        self.assertIn("MUST reuse/reference/stay consistent with", prompt)
        self.assertIn("sec_intro", prompt)
        self.assertIn("Introduction", prompt)
        self.assertIn("must_reuse", prompt)
        self.assertIn("Results must reuse the exact taxonomy", prompt)
        self.assertIn("Define 3 volatility regimes", prompt)
        self.assertIn("We classify volatility regimes", prompt)

    def test_undrafted_dependency_is_silently_omitted(self) -> None:
        graph = _graph_with(intro_status=NodeStatus.PENDING)
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        prompt = agent._build_prompt_from_node(graph.nodes["sec_results"], graph, layout_blocks=[])

        self.assertIn("(none)", prompt)
        self.assertNotIn("We classify volatility regimes", prompt)

    def test_excerpt_is_condensed_for_long_content(self) -> None:
        graph = _graph_with(intro_status=NodeStatus.DRAFTED)
        graph.nodes["sec_intro"].content = "word " * 400  # far longer than the 500-char cap
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        prompt = agent._build_prompt_from_node(graph.nodes["sec_results"], graph, layout_blocks=[])

        self.assertIn("[…]", prompt)
        self.assertNotIn("word " * 400, prompt)


if __name__ == "__main__":
    unittest.main()

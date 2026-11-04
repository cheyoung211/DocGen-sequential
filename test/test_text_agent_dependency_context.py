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

    def test_drafted_dependency_records_the_input_version_actually_used(self) -> None:
        graph = _graph_with(intro_status=NodeStatus.DRAFTED)
        graph.nodes["sec_intro"].attempt_count = 2
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        agent._build_prompt_from_node(graph.nodes["sec_results"], graph, layout_blocks=[])

        versions_used = graph.nodes["sec_results"].spec["dependency_versions_used"]
        self.assertEqual(len(versions_used), 1)
        self.assertEqual(versions_used[0]["depends_on"], "sec_intro")
        self.assertEqual(versions_used[0]["attempt_count"], 2)
        self.assertEqual(
            versions_used[0]["last_updated"], graph.nodes["sec_intro"].last_updated.isoformat()
        )

    def test_undrafted_dependency_records_no_version_used(self) -> None:
        graph = _graph_with(intro_status=NodeStatus.PENDING)
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        agent._build_prompt_from_node(graph.nodes["sec_results"], graph, layout_blocks=[])

        self.assertEqual(graph.nodes["sec_results"].spec["dependency_versions_used"], [])

    def test_excerpt_is_condensed_for_long_content(self) -> None:
        graph = _graph_with(intro_status=NodeStatus.DRAFTED)
        graph.nodes["sec_intro"].content = "word " * 400  # far longer than the 500-char cap
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        prompt = agent._build_prompt_from_node(graph.nodes["sec_results"], graph, layout_blocks=[])

        self.assertIn("[…]", prompt)
        self.assertNotIn("word " * 400, prompt)

    def test_excerpt_selects_the_relevant_paragraph_even_when_it_is_not_near_the_start(self) -> None:
        # A review found that always taking the first 500 chars could drop
        # a required value placed later in a long source section entirely
        # -- not even truncated, just absent. The excerpt selection must
        # prefer the paragraph that actually matches the dependency's own
        # description over an earlier, unrelated one.
        graph = _graph_with(intro_status=NodeStatus.DRAFTED)
        filler = "Unrelated filler text about something else entirely. " * 10  # > 500 chars alone
        graph.nodes["sec_intro"].content = (
            filler + "\n\n" +
            "The crisis threshold constant is set to 0.42 standard deviations above the mean."
        )
        graph.nodes["sec_results"].spec["dependency_notes"][0]["description"] = (
            "Results must reuse the exact crisis threshold constant defined in the introduction."
        )
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        prompt = agent._build_prompt_from_node(graph.nodes["sec_results"], graph, layout_blocks=[])

        self.assertIn("0.42 standard deviations", prompt)

    def test_excerpt_falls_back_to_the_prefix_when_description_has_no_keyword_overlap(self) -> None:
        # No shared keywords at all -> ties on score 0 everywhere -> the
        # tie-break (original order) reproduces the old "take the prefix"
        # behavior exactly, not an arbitrary reshuffling.
        graph = _graph_with(intro_status=NodeStatus.DRAFTED)
        graph.nodes["sec_intro"].content = "Alpha paragraph here.\n\nBeta paragraph here."
        graph.nodes["sec_results"].spec["dependency_notes"][0]["description"] = "xyz123 nonmatching"
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        prompt = agent._build_prompt_from_node(graph.nodes["sec_results"], graph, layout_blocks=[])

        self.assertIn("Alpha paragraph here.", prompt)

    def test_must_reference_is_surfaced_even_when_source_is_undrafted(self) -> None:
        # must_reference only needs the source's label (valid before
        # drafting) -- a legitimate forward reference must not be dropped
        # just because the source hasn't been drafted yet.
        graph = _graph_with(intro_status=NodeStatus.PENDING)
        graph.nodes["sec_results"].spec["dependency_notes"] = [
            {
                "depends_on": "sec_intro",
                "relation": "must_reference",
                "description": "Results must cite the introduction's key result.",
            }
        ]
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        prompt = agent._build_prompt_from_node(graph.nodes["sec_results"], graph, layout_blocks=[])

        self.assertIn("must_reference", prompt)
        self.assertIn("Results must cite the introduction's key result.", prompt)
        self.assertNotIn("We classify volatility regimes", prompt)  # no content excerpt, source undrafted
        # must_reference never needs drafted content, so it must not count
        # as an input version actually read.
        self.assertEqual(graph.nodes["sec_results"].spec["dependency_versions_used"], [])

    def test_must_reference_excerpt_omitted_even_when_source_is_drafted(self) -> None:
        # Deliberately no content excerpt for must_reference even once the
        # source IS drafted -- the obligation is "cite the label", and the
        # label itself is already listed under "Available cross-reference
        # labels"; a content excerpt here would blur this relation with
        # must_reuse/must_be_consistent_with's genuinely different need.
        graph = _graph_with(intro_status=NodeStatus.DRAFTED)
        graph.nodes["sec_results"].spec["dependency_notes"] = [
            {
                "depends_on": "sec_intro",
                "relation": "must_reference",
                "description": "Results must cite the introduction's key result.",
            }
        ]
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        prompt = agent._build_prompt_from_node(graph.nodes["sec_results"], graph, layout_blocks=[])

        self.assertIn("must_reference", prompt)
        self.assertNotIn("We classify volatility regimes", prompt)
        self.assertEqual(graph.nodes["sec_results"].spec["dependency_versions_used"], [])


if __name__ == "__main__":
    unittest.main()

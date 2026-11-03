"""Regression tests for Revision (Additional verifiers) Stage 1 / Fix A --
Markdown-emphasis (``**bold**``/``__bold__``) regulation.

Covers, in order: the shared detector/converter (``find_markdown_emphasis_
matches``/``convert_markdown_emphasis``) in isolation; TextAgent's
``_validate_one_block`` wiring (generation-time convert-or-reject, plus the
plain-text `title`/`caption` rejection); LatexAssembler's composer-side
residual check (``_validate_fragment``); and one real end-to-end compile
through tectonic, confirming the forbidden syntax actually disappears from
the rendered output while normal prose/table/list structure survives.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import List

from src.agents.latex_assembler import (
    LatexAssembler,
    convert_markdown_emphasis,
    find_markdown_emphasis_matches,
)
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
from scripts.llm_client import LLMClient

_STUB_LLM_CLIENT = object()


class FindMarkdownEmphasisMatchesTest(unittest.TestCase):
    def test_simple_bold_is_found_and_safe(self) -> None:
        matches = find_markdown_emphasis_matches("This is **important** context.")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].marker, "**")
        self.assertEqual(matches[0].inner, "important")
        self.assertFalse(matches[0].ambiguous)

    def test_double_underscore_bold_is_found_and_safe(self) -> None:
        matches = find_markdown_emphasis_matches("This is __also bold__ text.")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].marker, "__")
        self.assertEqual(matches[0].inner, "also bold")
        self.assertFalse(matches[0].ambiguous)

    def test_multiple_spans_on_one_line_are_all_found(self) -> None:
        matches = find_markdown_emphasis_matches("**One** and **two** and __three__.")
        self.assertEqual([m.inner for m in matches], ["One", "two", "three"])
        self.assertTrue(all(not m.ambiguous for m in matches))

    def test_multiple_spans_across_lines_are_all_found(self) -> None:
        matches = find_markdown_emphasis_matches("**One**\nplain\n**Two**")
        self.assertEqual([m.inner for m in matches], ["One", "Two"])
        self.assertEqual([m.line_number for m in matches], [1, 3])

    def test_unterminated_marker_is_ambiguous(self) -> None:
        matches = find_markdown_emphasis_matches("This has an **unterminated marker problem.")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].ambiguous)
        self.assertIn("unterminated", matches[0].reason)

    def test_existing_textbf_is_not_touched_at_all(self) -> None:
        matches = find_markdown_emphasis_matches(r"This is \textbf{already bold} text.")
        self.assertEqual(matches, [])

    def test_code_shaped_exponent_is_ignored_not_flagged(self) -> None:
        matches = find_markdown_emphasis_matches("x**2 and y**3 are both exponents.")
        self.assertEqual(matches, [])

    def test_single_exponent_occurrence_is_ignored_not_flagged(self) -> None:
        matches = find_markdown_emphasis_matches("The term x**2 appears once.")
        self.assertEqual(matches, [])

    def test_double_underscore_identifier_converts_like_ordinary_bold(self) -> None:
        # "__init__" and genuine "__bold__" are the same syntax with no
        # reliable way to tell them apart in this project's content schema
        # (no inline-code/literal span exists) -- see
        # find_markdown_emphasis_matches's docstring for why this is an
        # accepted tradeoff rather than a special-cased identifier guard.
        matches = find_markdown_emphasis_matches("Python calls __init__ automatically.")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].inner, "init")
        self.assertFalse(matches[0].ambiguous)

    def test_emphasis_inside_dollar_math_is_ignored(self) -> None:
        matches = find_markdown_emphasis_matches(r"Note that $a**b$ is not emphasis.")
        self.assertEqual(matches, [])

    def test_emphasis_inside_bracket_math_is_ignored(self) -> None:
        matches = find_markdown_emphasis_matches(r"See \[ a**b \] above.")
        self.assertEqual(matches, [])

    def test_bold_table_cell_is_found_and_safe(self) -> None:
        table = "| **Metric** | Value |\n| --- | --- |\n| Accuracy | 0.95 |"
        matches = find_markdown_emphasis_matches(table)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].inner, "Metric")
        self.assertFalse(matches[0].ambiguous)

    def test_span_crossing_a_table_cell_boundary_is_ambiguous(self) -> None:
        # The author forgot to close bold in the first cell; a later '**'
        # in a different cell must not be read as its closing marker.
        table = "| **Metric | 5** |\n| --- | --- |"
        matches = find_markdown_emphasis_matches(table)
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].ambiguous)
        self.assertIn("table cell boundary", matches[0].reason)

    def test_bold_list_item_is_found_and_safe(self) -> None:
        matches = find_markdown_emphasis_matches("- **First** item\n- Second item")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].inner, "First")
        self.assertFalse(matches[0].ambiguous)

    def test_overlapping_different_markers_are_both_ambiguous(self) -> None:
        matches = find_markdown_emphasis_matches("**foo __bar** baz__")
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(m.ambiguous for m in matches))
        self.assertTrue(all("overlaps" in m.reason for m in matches))

    def test_properly_nested_different_markers_are_both_safe(self) -> None:
        matches = find_markdown_emphasis_matches("**foo __bar__ baz**")
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(not m.ambiguous for m in matches))

    def test_adjacent_glued_same_marker_spans_are_ignored_not_guessed(self) -> None:
        # "**foo **bar** baz**": the same delimiter run at position 6 would
        # have to simultaneously close the first span and open the second,
        # with no space to separate the two roles -- genuinely ambiguous,
        # not nesting (CommonMark can't nest '**' in '**' either). Each
        # candidate pair fails the alphanumeric-flanking test (one's close
        # is glued to the next span's 'b', the other's open is glued to the
        # previous span's 'r'), so both are silently ignored rather than
        # guessed at -- the same conservative outcome as "x**2".
        matches = find_markdown_emphasis_matches("**foo **bar** baz**")
        self.assertEqual(matches, [])


class ConvertMarkdownEmphasisTest(unittest.TestCase):
    def test_safe_span_is_converted_to_textbf(self) -> None:
        converted, violations = convert_markdown_emphasis("This is **important** context.")
        self.assertEqual(converted, r"This is \textbf{important} context.")
        self.assertEqual(violations, [])

    def test_double_underscore_is_also_converted(self) -> None:
        converted, violations = convert_markdown_emphasis("This is __also bold__ text.")
        self.assertEqual(converted, r"This is \textbf{also bold} text.")
        self.assertEqual(violations, [])

    def test_multiple_safe_spans_all_convert_correctly(self) -> None:
        converted, violations = convert_markdown_emphasis("**One** and **two** and __three__.")
        self.assertEqual(converted, r"\textbf{One} and \textbf{two} and \textbf{three}.")
        self.assertEqual(violations, [])

    def test_ambiguous_span_is_left_untouched_and_reported(self) -> None:
        original = "This has an **unterminated marker problem."
        converted, violations = convert_markdown_emphasis(original, block_id="para1")
        self.assertEqual(converted, original)
        self.assertEqual(len(violations), 1)
        self.assertIn("para1", violations[0])
        self.assertIn("unterminated", violations[0])

    def test_safe_and_ambiguous_spans_coexist_independently(self) -> None:
        text = "**Safe bold** then an **unterminated one."
        converted, violations = convert_markdown_emphasis(text)
        self.assertEqual(converted, r"\textbf{Safe bold} then an **unterminated one.")
        self.assertEqual(len(violations), 1)

    def test_code_and_math_content_is_never_modified(self) -> None:
        text = "x**2 and y**3, and $a**b$ too."
        converted, violations = convert_markdown_emphasis(text)
        self.assertEqual(converted, text)
        self.assertEqual(violations, [])

    def test_existing_textbf_survives_unchanged(self) -> None:
        text = r"Already \textbf{bold} stays that way."
        converted, violations = convert_markdown_emphasis(text)
        self.assertEqual(converted, text)
        self.assertEqual(violations, [])

    def test_table_cell_bold_converts_without_disturbing_pipes(self) -> None:
        table = "| **Metric** | Value |\n| --- | --- |\n| Accuracy | 0.95 |"
        converted, violations = convert_markdown_emphasis(table)
        self.assertEqual(
            converted,
            "| \\textbf{Metric} | Value |\n| --- | --- |\n| Accuracy | 0.95 |",
        )
        self.assertEqual(violations, [])

    def test_conversion_is_idempotent(self) -> None:
        text = "**One** and __two__ and an **unterminated marker."
        once, violations_once = convert_markdown_emphasis(text)
        twice, violations_twice = convert_markdown_emphasis(once)
        self.assertEqual(once, twice)
        self.assertEqual(violations_once, violations_twice)


class TextAgentMarkdownEmphasisValidatorTest(unittest.TestCase):
    """Integration with TextAgent._validate_one_block -- the generation-time
    convert-or-reject wiring, plus the plain-text title/caption rejection."""

    @staticmethod
    def _layout_block(kind: LayoutBlockKind = LayoutBlockKind.PARAGRAPH) -> LayoutBlock:
        return LayoutBlock(id="b1", kind=kind)

    def test_safe_emphasis_is_silently_converted_with_no_error(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block = self._layout_block()
        content_block = ContentBlock(
            block_id="b1", type=SemanticBlockType.PARAGRAPH, content="This is **important**."
        )
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertNotIn("markdown_emphasis", codes)
        self.assertEqual(content_block.content, r"This is \textbf{important}.")

    def test_ambiguous_emphasis_raises_markdown_emphasis_error(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block = self._layout_block()
        content_block = ContentBlock(
            block_id="b1", type=SemanticBlockType.PARAGRAPH,
            content="This has an **unterminated problem.",
        )
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertIn("markdown_emphasis", codes)

    def test_disabled_validator_skips_conversion_and_check(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT, enabled_validators={"markdown_emphasis": False})
        layout_block = self._layout_block()
        content_block = ContentBlock(
            block_id="b1", type=SemanticBlockType.PARAGRAPH, content="This is **important**."
        )
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertNotIn("markdown_emphasis", codes)
        # No conversion either -- the raw Markdown survives untouched.
        self.assertEqual(content_block.content, "This is **important**.")

    def test_title_with_markdown_emphasis_is_rejected(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block = self._layout_block(LayoutBlockKind.KEY_TAKEAWAY)
        content_block = ContentBlock(
            block_id="b1", type=SemanticBlockType.KEY_TAKEAWAY,
            content="Plain content.", title="**Bold Title**",
        )
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertIn("plain_title", codes)

    def test_caption_with_markdown_emphasis_is_rejected(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block = self._layout_block(LayoutBlockKind.TABLE)
        content_block = ContentBlock(
            block_id="b1", type=SemanticBlockType.TABLE,
            content="| A | B |\n| --- | --- |\n| 1 | 2 |",
            caption="__Bold Caption__",
        )
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertIn("plain_caption", codes)

    def test_plain_title_and_caption_without_emphasis_still_pass(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block = self._layout_block(LayoutBlockKind.TABLE)
        content_block = ContentBlock(
            block_id="b1", type=SemanticBlockType.TABLE,
            content="| A | B |\n| --- | --- |\n| 1 | 2 |",
            title="Plain title", caption="Plain caption",
        )
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertNotIn("plain_title", codes)
        self.assertNotIn("plain_caption", codes)


class LatexAssemblerMarkdownEmphasisResidualCheckTest(unittest.TestCase):
    """LatexAssembler._validate_fragment must refuse a block that still
    contains Markdown emphasis -- it must never silently convert it right
    before rendering (that would desync the stored artifact from the PDF)."""

    def test_unconverted_emphasis_reaching_the_composer_is_rejected(self) -> None:
        assembler = LatexAssembler()
        block = ContentBlock(
            block_id="p1", type=SemanticBlockType.PARAGRAPH, content="This is **important**."
        )
        with self.assertRaises(ValueError):
            assembler._validate_fragment(block)

    def test_already_converted_textbf_passes(self) -> None:
        assembler = LatexAssembler()
        block = ContentBlock(
            block_id="p1", type=SemanticBlockType.PARAGRAPH,
            content=r"This is \textbf{important}.",
        )
        assembler._validate_fragment(block)  # must not raise

    def test_disabled_validator_lets_residual_emphasis_through(self) -> None:
        assembler = LatexAssembler(enabled_validators={"markdown_emphasis": False})
        block = ContentBlock(
            block_id="p1", type=SemanticBlockType.PARAGRAPH, content="This is **important**."
        )
        assembler._validate_fragment(block)  # must not raise -- gate is off


class ScriptedLLMClient:
    """Duck-typed against the subset of LLMClient TextAgent uses -- same
    helper test_compile_repair.py already introduces for this purpose."""

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
        LayoutBlock(id="table", kind=LayoutBlockKind.TABLE),
        LayoutBlock(id="list", kind=LayoutBlockKind.LIST),
        LayoutBlock(id="broken", kind=LayoutBlockKind.PARAGRAPH),
    ]
    blueprint = DocumentBlueprint(
        document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Markdown emphasis test"),
        document_class="article",
        sections=[SectionLayout(section_id="sec_main", title="Main", blocks=blocks)],
    )
    graph = DocumentGraph()
    graph.add_node(DocumentNode(id="sec_main", type=NodeType.SECTION, title="Main"))
    graph.set_blueprint(blueprint)
    return graph


@unittest.skipUnless(shutil.which("tectonic"), "tectonic is not installed")
class MarkdownEmphasisEndToEndTest(unittest.TestCase):
    """Drives the real TextAgent validate/convert/retry loop and a real
    tectonic compile, confirming forbidden emphasis actually disappears
    from the rendered output while prose/table/list structure, and a
    genuinely ambiguous span's block-repair round-trip, both survive."""

    def test_markdown_emphasis_disappears_from_the_compiled_document(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                graph = _graph()
                first_attempt = json.dumps(
                    {
                        "content_blocks": [
                            {
                                "block_id": "intro", "type": "paragraph",
                                "content": "This is **important** and __also bold__ context.",
                            },
                            {
                                "block_id": "table", "type": "table",
                                "content": "| **Metric** | Value |\n| --- | --- |\n| Accuracy | 0.95 |",
                            },
                            {
                                "block_id": "list", "type": "list",
                                "content": "- **First** item\n- Second item",
                            },
                            {
                                "block_id": "broken", "type": "paragraph",
                                "content": "This has an **unterminated marker problem.",
                            },
                        ],
                        "tool_requests": [],
                    }
                )
                second_attempt = json.dumps(
                    {
                        "content_blocks": [
                            {
                                "block_id": "broken", "type": "paragraph",
                                "content": "This has a **fixed** marker now.",
                            }
                        ],
                        "tool_requests": [],
                    }
                )
                llm = ScriptedLLMClient([first_attempt, second_attempt])
                text_agent = TextAgent(llm_client=llm)
                output = text_agent.process_node(
                    "sec_main", graph, "md-emphasis-test",
                    output_dir="outputs/generations", max_attempts=2,
                )
                self.assertIsNotNone(output)
                self.assertEqual(llm.calls, 2)
                self.assertEqual(graph.nodes["sec_main"].status, NodeStatus.DRAFTED)

                fixed_blocks = {b.block_id: b.content for b in output.blocks}
                self.assertEqual(
                    fixed_blocks["intro"], r"This is \textbf{important} and \textbf{also bold} context."
                )
                self.assertIn(r"\textbf{Metric}", fixed_blocks["table"])
                self.assertIn(r"\textbf{First}", fixed_blocks["list"])
                self.assertEqual(fixed_blocks["broken"], r"This has a \textbf{fixed} marker now.")

                assembler = LatexAssembler()
                assembler.assemble("md-emphasis-test", graph)

                rendered = Path(
                    "outputs/generations/md-emphasis-test/sections/sec_main.tex"
                ).read_text(encoding="utf-8")
                self.assertNotIn("**", rendered)
                self.assertNotIn("__", rendered)
                self.assertIn(r"\textbf{important}", rendered)
                self.assertIn(r"\textbf{Metric}", rendered)
                self.assertIn(r"\textbf{First}", rendered)
                self.assertIn(r"\textbf{fixed}", rendered)

                compiler = LatexCompiler()
                success = compiler.compile_pdf("md-emphasis-test", output_dir="outputs/generations")
                self.assertTrue(success, "the fully-converted document should compile cleanly")
                pdf_path = Path("outputs/generations/md-emphasis-test/main.pdf")
                self.assertTrue(pdf_path.is_file())
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()

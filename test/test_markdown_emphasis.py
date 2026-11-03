"""Regression tests for Revision (Additional verifiers) Stage 1 / Fix A --
Markdown-emphasis (``**bold**``/``__bold__``) regulation.

Covers, in order: the shared detector/converter (``find_markdown_emphasis_
matches``/``convert_markdown_emphasis``) in isolation, including the round-1
review fixes (nested-span corruption, code-disrupts-later-bold, Korean
particles glued to a closing marker, ``\\verb``/``\\texttt{}``/multi-line
math preservation); TextAgent's pure (non-mutating) ``_validate_one_block``
check plus the explicit, separately-provenanced conversion step in
``_validate_blocks_against_layout``; LatexAssembler's composer-side residual
check (content AND title/caption, plus a figure's own caption); and one real
end-to-end compile through tectonic.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import List

from src.common.state import (
    DocumentBlueprint,
    DocumentGraph,
    DocumentKind,
    DocumentNode,
    DocumentSpec,
    FigureLayout,
    FigurePlacement,
    LayoutBlock,
    LayoutBlockKind,
    NodeStatus,
    NodeType,
    SectionLayout,
)

from src.agents.latex_assembler import (
    LatexAssembler,
    convert_markdown_emphasis,
    find_markdown_emphasis_matches,
)
from src.agents.latex_compiler import LatexCompiler
from src.agents.text_agent import TextAgent
from src.common.schemas import ContentBlock, SemanticBlockType, TextAgentOutput
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

    def test_code_glued_marker_does_not_disrupt_a_later_real_bold(self) -> None:
        # Round-1 review finding: blind sequential pairing used to pair
        # x**2's own "**" with **important**'s OPENING "**", scrambling
        # the real bold's own pairing. Excluding a code-glued occurrence
        # before any pairing happens (see _find_matches_for_marker) fixes
        # this: "x**2" contributes nothing, "**important**" still pairs
        # with itself correctly.
        matches = find_markdown_emphasis_matches("x**2 and **important**.")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].inner, "important")
        self.assertFalse(matches[0].ambiguous)

    def test_double_underscore_identifier_converts_like_ordinary_bold(self) -> None:
        # "__init__" and genuine "__bold__" are the same syntax with no
        # reliable way to tell them apart in this project's content schema
        # outside an explicit literal span (\texttt{}/\verb, see below) --
        # see find_markdown_emphasis_matches's docstring.
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

    def test_emphasis_inside_multiline_bracket_math_is_ignored(self) -> None:
        # Round-1 review finding: math state used to reset every line
        # (each line ran its own independent _find_math_spans), so a
        # \[ ... \] that spans more than one line left its middle lines
        # completely unprotected. _literal_or_math_mask tracks math state
        # across the whole text instead.
        text = "\\[\n  a**b\n\\]"
        matches = find_markdown_emphasis_matches(text)
        self.assertEqual(matches, [])

    def test_emphasis_inside_verb_is_ignored(self) -> None:
        # Round-1 review finding: \verb's argument renders byte-for-byte;
        # markers inside it are the literal text being quoted, not
        # emphasis.
        matches = find_markdown_emphasis_matches(r"Use \verb|**bold**| in code.")
        self.assertEqual(matches, [])

    def test_emphasis_inside_texttt_is_ignored(self) -> None:
        # Round-1 review finding: \texttt{} is this project's own
        # convention for code/identifier-like text, so content inside it
        # (including a doubled-underscore identifier) must never be read
        # as Markdown bold.
        matches = find_markdown_emphasis_matches(r"Call \texttt{__init__} once.")
        self.assertEqual(matches, [])

    def test_korean_particle_glued_to_a_closing_marker_still_converts(self) -> None:
        # Round-1 review finding: the old design rejected ANY pair whose
        # closing marker was followed by an alphanumeric character, which
        # incorrectly excluded completely normal Korean (and other
        # agglutinative-language) text that attaches a particle/verb
        # ending directly after a bolded word with no space. Only gluing
        # on the OPENING side is checked now (see
        # _find_matches_for_marker's docstring).
        matches = find_markdown_emphasis_matches("**중요**합니다.")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].inner, "중요")
        self.assertFalse(matches[0].ambiguous)

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

    def test_properly_nested_different_markers_are_both_ambiguous(self) -> None:
        # Round-1 review finding: this used to be classified "safe" (both
        # spans independently convertible), but converting the inner span
        # first and then splicing the outer span's stale inner/end indices
        # on top corrupts the text ("**foo __bar__ baz**" ->
        # "\\textbf{foo __bar__ baz}baz**"). Nesting is never
        # independently convertible now, regardless of marker type -- see
        # _flag_cross_marker_overlaps.
        matches = find_markdown_emphasis_matches("**foo __bar__ baz**")
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(m.ambiguous for m in matches))

    def test_adjacent_glued_same_marker_spans_are_ambiguous_not_silent(self) -> None:
        # Round-1 review finding: this used to silently produce zero
        # matches at all (both candidate pairs failed an overly broad
        # before-OR-after alphanumeric check), so the whole input passed
        # TextAgent/the assembler with no error. The 1-deep stack now
        # pairs (0,6) as "foo " -- empty/whitespace-padded inner -- which
        # is ambiguous, not silently dropped; the block is correctly
        # routed to repair instead of silently passing.
        matches = find_markdown_emphasis_matches("**foo **bar** baz**")
        self.assertTrue(any(m.ambiguous for m in matches))


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

    def test_code_glued_marker_does_not_block_a_later_real_bold_from_converting(self) -> None:
        converted, violations = convert_markdown_emphasis("x**2 and **important**.")
        self.assertEqual(converted, r"x**2 and \textbf{important}.")
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

    def test_nested_different_markers_are_never_corrupted(self) -> None:
        # The exact round-1 repro: converting the inner span and then the
        # outer span used to produce "\\textbf{foo __bar__ baz}baz**" --
        # duplicated trailing text, broken LaTeX. Both spans are ambiguous
        # now, so the content is returned byte-for-byte unchanged and both
        # violations are reported instead.
        original = "**foo __bar__ baz**"
        converted, violations = convert_markdown_emphasis(original, block_id="b1")
        self.assertEqual(converted, original)
        self.assertEqual(len(violations), 2)

    def test_korean_particle_after_closing_marker_converts_correctly(self) -> None:
        converted, violations = convert_markdown_emphasis("**중요**합니다.")
        self.assertEqual(converted, r"\textbf{중요}합니다.")
        self.assertEqual(violations, [])

    def test_verb_content_is_preserved_byte_for_byte(self) -> None:
        text = r"Use \verb|**bold**| in code."
        converted, violations = convert_markdown_emphasis(text)
        self.assertEqual(converted, text)
        self.assertEqual(violations, [])

    def test_texttt_content_is_preserved_byte_for_byte(self) -> None:
        text = r"Call \texttt{__init__} once."
        converted, violations = convert_markdown_emphasis(text)
        self.assertEqual(converted, text)
        self.assertEqual(violations, [])

    def test_multiline_math_is_preserved_across_every_line_it_spans(self) -> None:
        text = "\\[\n  a**b\n\\]"
        converted, violations = convert_markdown_emphasis(text)
        self.assertEqual(converted, text)
        self.assertEqual(violations, [])

    def test_conversion_is_idempotent(self) -> None:
        text = "**One** and __two__ and an **unterminated marker."
        once, violations_once = convert_markdown_emphasis(text)
        twice, violations_twice = convert_markdown_emphasis(once)
        self.assertEqual(once, twice)
        self.assertEqual(violations_once, violations_twice)


class TextAgentMarkdownEmphasisValidatorTest(unittest.TestCase):
    """Integration with TextAgent._validate_one_block -- now a PURE,
    read-only check (round-1 review finding: it used to mutate
    content_block.content as a side effect, which corrupted
    TextAgentValidatorAdapter's verdicts -- see test/core/test_adapters.py's
    adapter-purity test for that half of the fix). The actual conversion is
    tested separately below, against _validate_blocks_against_layout, the
    real generation path's own explicit normalization step."""

    @staticmethod
    def _layout_block(kind: LayoutBlockKind = LayoutBlockKind.PARAGRAPH) -> LayoutBlock:
        return LayoutBlock(id="b1", kind=kind)

    def test_safe_emphasis_passes_without_mutating_content(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block = self._layout_block()
        content_block = ContentBlock(
            block_id="b1", type=SemanticBlockType.PARAGRAPH, content="This is **important**."
        )
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertNotIn("markdown_emphasis", codes)
        self.assertEqual(content_block.content, "This is **important**.")

    def test_ambiguous_emphasis_raises_markdown_emphasis_error_without_mutating(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block = self._layout_block()
        original = "This has an **unterminated problem."
        content_block = ContentBlock(block_id="b1", type=SemanticBlockType.PARAGRAPH, content=original)
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertIn("markdown_emphasis", codes)
        self.assertEqual(content_block.content, original)

    def test_disabled_validator_skips_the_check(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT, enabled_validators={"markdown_emphasis": False})
        layout_block = self._layout_block()
        content_block = ContentBlock(
            block_id="b1", type=SemanticBlockType.PARAGRAPH,
            content="This has an **unterminated problem.",
        )
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertNotIn("markdown_emphasis", codes)

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


class ValidateBlocksAgainstLayoutConversionTest(unittest.TestCase):
    """The real generation path's explicit, separately-provenanced
    conversion step (Revision (Additional verifiers) Stage 1, round-1 review:
    conversion must not live inside the pure per-block check -- see
    TextAgentMarkdownEmphasisValidatorTest above)."""

    @staticmethod
    def _layout_blocks() -> List[LayoutBlock]:
        return [LayoutBlock(id="b1", kind=LayoutBlockKind.PARAGRAPH)]

    def test_conversion_is_applied_and_provenance_is_persisted(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        with tempfile.TemporaryDirectory() as tmp:
            sections_dir = Path(tmp)
            raw_blocks = [
                {"block_id": "b1", "type": "paragraph", "content": "This is **important**."}
            ]
            blocks = agent._validate_blocks_against_layout(
                raw_blocks, self._layout_blocks(), sections_dir, "sec1"
            )
            self.assertEqual(blocks[0].content, r"This is \textbf{important}.")

            provenance_files = list(sections_dir.glob("sec1.markdown_emphasis_conversion_*.json"))
            self.assertEqual(len(provenance_files), 1)
            record = json.loads(provenance_files[0].read_text(encoding="utf-8"))
            self.assertEqual(record["block_id"], "b1")
            self.assertEqual(record["original_content"], "This is **important**.")
            self.assertEqual(record["converted_content"], r"This is \textbf{important}.")
            self.assertIn("textbf", record["diff"])
            self.assertGreaterEqual(record["transform_latency_s"], 0.0)

    def test_no_provenance_file_when_nothing_was_converted(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        with tempfile.TemporaryDirectory() as tmp:
            sections_dir = Path(tmp)
            raw_blocks = [{"block_id": "b1", "type": "paragraph", "content": "Plain content."}]
            agent._validate_blocks_against_layout(raw_blocks, self._layout_blocks(), sections_dir, "sec1")
            self.assertEqual(list(sections_dir.glob("sec1.markdown_emphasis_conversion_*.json")), [])


class LatexAssemblerMarkdownEmphasisResidualCheckTest(unittest.TestCase):
    """LatexAssembler._validate_fragment must refuse a block that still
    contains Markdown emphasis in content, title, OR caption -- it must
    never silently convert any of them right before rendering (that would
    desync the stored artifact from the PDF)."""

    def test_unconverted_content_emphasis_reaching_the_composer_is_rejected(self) -> None:
        assembler = LatexAssembler()
        block = ContentBlock(
            block_id="p1", type=SemanticBlockType.PARAGRAPH, content="This is **important**."
        )
        with self.assertRaises(ValueError):
            assembler._validate_fragment(block)

    def test_title_emphasis_reaching_the_composer_is_rejected(self) -> None:
        # Round-1 review finding: the residual check only ever looked at
        # `content`; a stale/ablated artifact with bad `title` or
        # `caption` -- the fields _render_semantic_block/_render_table
        # actually render -- passed through untouched.
        assembler = LatexAssembler()
        block = ContentBlock(
            block_id="p1", type=SemanticBlockType.KEY_TAKEAWAY,
            content="Plain content.", title="**Bold Title**",
        )
        with self.assertRaises(ValueError):
            assembler._validate_fragment(block)

    def test_caption_emphasis_reaching_the_composer_is_rejected(self) -> None:
        assembler = LatexAssembler()
        block = ContentBlock(
            block_id="p1", type=SemanticBlockType.TABLE,
            content="| A | B |\n| --- | --- |\n| 1 | 2 |",
            caption="**Data summary**",
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


class FigureCaptionMarkdownEmphasisTest(unittest.TestCase):
    """A figure's caption is planner-authored (FigureRegistryEntry/
    FigurePlacement), not TextAgent-authored -- round-1 review finding:
    ContentBlock.caption's new check never covers this completely separate
    field, even though _render_figure renders it through the exact same
    plain-text-only _escape_latex."""

    @staticmethod
    def _graph_with_figure(caption: str) -> tuple:
        blueprint = DocumentBlueprint(
            document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Figure caption test"),
            document_class="article",
            sections=[SectionLayout(section_id="sec1", title="Sec", blocks=[])],
            figure_slots={
                "fig1": FigurePlacement(
                    figure_id="fig1", owner_section="sec1", layout=FigureLayout.FLOAT, caption=caption
                )
            },
        )
        graph = DocumentGraph()
        graph.add_node(DocumentNode(id="sec1", type=NodeType.SECTION, title="Sec"))
        graph.register_figure_asset("fig1", "figs/fig1.png")
        graph.set_blueprint(blueprint)
        return graph, blueprint

    def test_figure_caption_with_markdown_emphasis_is_rejected(self) -> None:
        graph, blueprint = self._graph_with_figure("**Bold caption**")
        assembler = LatexAssembler()
        assembler._rendered_labels = set()  # normally set up by assemble()
        with self.assertRaises(ValueError):
            assembler._render_figure("fig1", "sec1", blueprint, graph)

    def test_figure_caption_without_emphasis_renders_fine(self) -> None:
        graph, blueprint = self._graph_with_figure("Plain caption")
        assembler = LatexAssembler()
        assembler._rendered_labels = set()  # normally set up by assemble()
        rendered = assembler._render_figure("fig1", "sec1", blueprint, graph)
        self.assertIn("Plain caption", rendered)


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
    from the rendered output while prose/table/list structure, a Korean
    particle glued to a closing marker, and a genuinely ambiguous span's
    block-repair round-trip, all survive."""

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
                                "content": (
                                    "This is **important** and __also bold__ context, "
                                    "and **중요**합니다."
                                ),
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
                    fixed_blocks["intro"],
                    r"This is \textbf{important} and \textbf{also bold} context, "
                    r"and \textbf{중요}합니다.",
                )
                self.assertIn(r"\textbf{Metric}", fixed_blocks["table"])
                self.assertIn(r"\textbf{First}", fixed_blocks["list"])
                self.assertEqual(fixed_blocks["broken"], r"This has a \textbf{fixed} marker now.")

                sections_dir = Path("outputs/generations/md-emphasis-test/sections")
                provenance_files = list(sections_dir.glob("sec_main.markdown_emphasis_conversion_*.json"))
                self.assertGreaterEqual(len(provenance_files), 3)

                assembler = LatexAssembler()
                assembler.assemble("md-emphasis-test", graph)

                rendered = (sections_dir / "sec_main.tex").read_text(encoding="utf-8")
                self.assertNotIn("**", rendered)
                self.assertNotIn("__", rendered)
                self.assertIn(r"\textbf{important}", rendered)
                self.assertIn(r"\textbf{Metric}", rendered)
                self.assertIn(r"\textbf{First}", rendered)
                self.assertIn(r"\textbf{fixed}", rendered)
                self.assertIn(r"\textbf{중요}합니다", rendered)

                compiler = LatexCompiler()
                success = compiler.compile_pdf("md-emphasis-test", output_dir="outputs/generations")
                self.assertTrue(success, "the fully-converted document should compile cleanly")
                pdf_path = Path("outputs/generations/md-emphasis-test/main.pdf")
                self.assertTrue(pdf_path.is_file())
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()

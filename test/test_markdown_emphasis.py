"""Regression tests for Revision (Additional verifiers) Stage 1 / Fix A --
Markdown-emphasis (``**bold**``/``__bold__``) regulation.

Covers, in order: the shared detector/converter (``find_markdown_emphasis_
matches``/``convert_markdown_emphasis``) in isolation, including the round-1
review fixes (nested-span corruption, code-disrupts-later-bold, Korean
particles glued to a closing marker, ``\\verb``/``\\texttt{}``/multi-line
math preservation) AND the round-2 review fixes (escape-aware literal/math
mask, plain title/caption fields skipping the math mask entirely, compound
marker runs, unbalanced-brace safety); TextAgent's pure (non-mutating)
``_validate_one_block`` check -- which now fails on ANY remaining match, not
only ambiguous ones -- plus the explicit, always-provenanced conversion step
in ``_validate_blocks_against_layout``; LatexAssembler's composer-side
residual check (content AND title/caption, plus a figure's own caption),
now also provenanced; and one real end-to-end compile through tectonic.
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
    MARKDOWN_EMPHASIS_RULE_VERSION,
    LatexAssembler,
    convert_markdown_emphasis,
    find_markdown_emphasis_matches,
    validate_plain_caption,
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

    def test_korean_word_glued_to_an_opening_marker_still_converts(self) -> None:
        # Round-3 review finding: the old rule excluded ANY occurrence
        # preceded by an alphanumeric character from opening at all, which
        # silently dropped completely ordinary Korean prose that glues a
        # bold span directly onto a preceding word with no space ("우" in
        # "매우" is alphanumeric, same as the 'x' in "x**2" -- but this is
        # not a code pattern at all). Only an ASCII-alnum-before +
        # ASCII-digit-after combination (see _looks_like_code_exponent_open)
        # is excluded now.
        matches = find_markdown_emphasis_matches("매우**중요**합니다.")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].inner, "중요")
        self.assertFalse(matches[0].ambiguous)

    def test_word_glued_marker_separating_two_spans_both_convert(self) -> None:
        # Round-3 review finding: "**first**and**second**" used to convert
        # only the first span -- the second span's opening "**" is glued
        # to "and" (alphanumeric), which the old rule treated exactly like
        # "x**2" and silently excluded from ever opening at all.
        matches = find_markdown_emphasis_matches("**first**and**second**")
        self.assertEqual([m.inner for m in matches], ["first", "second"])
        self.assertTrue(all(not m.ambiguous for m in matches))

    def test_code_exponent_followed_by_a_letter_is_still_ignored(self) -> None:
        # The code-exponent exclusion requires BOTH signals (ASCII-alnum
        # before, ASCII-digit after) -- confirm the "after" side alone
        # isn't sufficient to flag a real word as code-shaped.
        matches = find_markdown_emphasis_matches("x**2 and y**3 are both exponents.")
        self.assertEqual(matches, [])

    def test_digit_emphasis_glued_to_a_word_at_end_of_line_is_not_silently_dropped(self) -> None:
        # Round-4 review finding: `"" in "0123456789"` is True in Python,
        # so _looks_like_code_exponent_open read "nothing follows the
        # marker" (end of line) as "a digit follows," wrongly excluding
        # the marker from opening at all -- with nothing left to classify
        # it as ambiguous either, "**2**" at the very end of
        # "**first**and**2**" vanished completely: zero matches, no
        # conversion, no error. It must be detected now (reported
        # ambiguous -- unterminated, since nothing closes it once the
        # preceding "d"+"2" pairing is correctly excluded as code-shaped),
        # never silently passed through.
        matches = find_markdown_emphasis_matches("**first**and**2**")
        self.assertEqual([m.inner for m in matches if not m.ambiguous], ["first"])
        self.assertTrue(any(m.ambiguous for m in matches))

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

    def test_emphasis_inside_align_environment_is_ignored(self) -> None:
        # Round-3 review finding: "\begin{align} V^{**} &= V \end{align}"
        # false-flagged the bare "**" as Markdown emphasis -- the mask
        # only recognized $/\(/\[ delimiters, never that the align
        # environment itself already establishes math mode.
        text = r"\begin{align} V^{**} &= V \end{align}"
        matches = find_markdown_emphasis_matches(text)
        self.assertEqual(matches, [])

    def test_emphasis_inside_pmatrix_environment_is_ignored(self) -> None:
        text = r"$\begin{pmatrix} a & ** \\ b & c \end{pmatrix}$"
        matches = find_markdown_emphasis_matches(text)
        self.assertEqual(matches, [])

    def test_emphasis_outside_a_math_environment_on_the_same_text_still_converts(self) -> None:
        text = r"\begin{align} a &= b \end{align} and **important**."
        matches = find_markdown_emphasis_matches(text)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].inner, "important")

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

    def test_escaped_dollar_does_not_start_a_math_exclusion(self) -> None:
        # Round-2 review finding: _literal_or_math_mask used to treat a
        # backslash-escaped dollar sign ("\$") the same as a bare,
        # math-opening "$" -- so this input's "$" toggled math on with no
        # closing "$" ever found, excluding "**important**" from being
        # scanned at all.
        matches = find_markdown_emphasis_matches("The price is \\$5 and **important**.")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].inner, "important")
        self.assertFalse(matches[0].ambiguous)

    def test_escaped_brace_inside_texttt_does_not_extend_the_literal_span(self) -> None:
        # Round-2 review finding: the \texttt{} brace-depth counter used to
        # treat an escaped "\{" the same as a real, unescaped group-open,
        # so "\texttt{\{}" (whose real closing brace is the very next
        # character) never found a second real "}" and masked the rest of
        # the line, including the following "**bold**", as if it were
        # still inside the \texttt{} argument.
        matches = find_markdown_emphasis_matches(r"\texttt{\{} then **bold**.")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].inner, "bold")
        self.assertFalse(matches[0].ambiguous)

    def test_compound_triple_marker_run_is_ambiguous_not_misread(self) -> None:
        # Round-2 review finding: a naive 2-char-window scan read
        # "***bold***" as "**" + "*bold" + "**" + "*", which _classify_pair
        # called SAFE (no leading/trailing whitespace, no pipe) and
        # convert_markdown_emphasis then turned into "\textbf{*bold}*" --
        # a stray asterisk left over from a marker run this project's
        # Markdown subset does not support at all. A run of 3+ of the same
        # marker character is never treated as an ordinary 2-char marker.
        matches = find_markdown_emphasis_matches("***bold***")
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(m.ambiguous for m in matches))
        self.assertTrue(all("compound" in m.reason for m in matches))

    def test_quadruple_marker_run_is_also_ambiguous(self) -> None:
        matches = find_markdown_emphasis_matches("****very bold****")
        self.assertTrue(all(m.ambiguous for m in matches))

    def test_unbalanced_brace_inside_a_pair_is_ambiguous(self) -> None:
        # Round-2 review finding: "**text { brace**" has inner text
        # "text { brace" -- one naked, unmatched "{". Converting it to
        # "\textbf{text { brace}" leaves two opens and one close, a brace
        # imbalance the composer only caught much later, far from the
        # actual cause.
        matches = find_markdown_emphasis_matches("**text { brace**")
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].ambiguous)
        self.assertIn("unbalanced brace", matches[0].reason)

    def test_balanced_brace_inside_a_pair_still_converts(self) -> None:
        matches = find_markdown_emphasis_matches("**a set {x, y}**")
        self.assertEqual(len(matches), 1)
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

    def test_korean_word_glued_marker_converts_correctly(self) -> None:
        converted, violations = convert_markdown_emphasis("매우**중요**합니다.")
        self.assertEqual(converted, "매우\\textbf{중요}합니다.")
        self.assertEqual(violations, [])

    def test_word_glued_separator_both_spans_convert(self) -> None:
        converted, violations = convert_markdown_emphasis("**first**and**second**")
        self.assertEqual(converted, r"\textbf{first}and\textbf{second}")
        self.assertEqual(violations, [])

    def test_digit_emphasis_glued_to_a_word_at_end_of_line_is_reported_not_silently_passed(self) -> None:
        original = "**first**and**2**"
        converted, violations = convert_markdown_emphasis(original, block_id="b1")
        self.assertEqual(converted, r"\textbf{first}and**2**")
        self.assertEqual(len(violations), 1)
        self.assertIn("unterminated", violations[0])

    def test_align_environment_content_is_preserved_byte_for_byte(self) -> None:
        text = r"\begin{align} V^{**} &= V \end{align}"
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

    def test_escaped_dollar_does_not_block_later_conversion(self) -> None:
        text = "The price is \\$5 and **important**."
        converted, violations = convert_markdown_emphasis(text)
        self.assertEqual(converted, "The price is \\$5 and \\textbf{important}.")
        self.assertEqual(violations, [])

    def test_escaped_brace_in_texttt_does_not_block_later_conversion(self) -> None:
        text = r"\texttt{\{} then **bold**."
        converted, violations = convert_markdown_emphasis(text)
        self.assertEqual(converted, r"\texttt{\{} then \textbf{bold}.")
        self.assertEqual(violations, [])

    def test_compound_marker_run_is_never_auto_converted(self) -> None:
        original = "***bold***"
        converted, violations = convert_markdown_emphasis(original, block_id="b1")
        self.assertEqual(converted, original)
        self.assertEqual(len(violations), 2)

    def test_unbalanced_brace_span_is_never_auto_converted(self) -> None:
        original = "**text { brace**"
        converted, violations = convert_markdown_emphasis(original, block_id="b1")
        self.assertEqual(converted, original)
        self.assertEqual(len(violations), 1)
        self.assertIn("unbalanced brace", violations[0])

    def test_violation_message_shows_the_exact_source_substring(self) -> None:
        # The violation string must reconstruct the real source text, not
        # double-wrap a compound run's already-complete `inner` in
        # `marker` on both sides again (which used to garble "***bold***"'s
        # own violation message into something like "'*******'").
        _, violations = convert_markdown_emphasis("***bold***", block_id="b1")
        self.assertTrue(any("'***'" in v for v in violations))

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
    adapter-purity test for that half of the fix) that fails on ANY
    remaining match, safe or ambiguous (round-2 review finding: it used to
    only fail on AMBIGUOUS matches, so a direct caller that skips the real
    generation path's conversion step entirely -- exactly what the adapter
    does -- got a PASS for a stored artifact that still literally
    contained "**important**"; being theoretically convertible is not the
    same as the artifact, as it actually exists right now, satisfying the
    rule). The actual conversion is tested separately below, against
    _validate_blocks_against_layout, the real generation path's own
    explicit normalization step."""

    @staticmethod
    def _layout_block(kind: LayoutBlockKind = LayoutBlockKind.PARAGRAPH) -> LayoutBlock:
        return LayoutBlock(id="b1", kind=kind)

    def test_unconverted_safe_emphasis_still_fails_the_pure_check(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block = self._layout_block()
        original = "This is **important**."
        content_block = ContentBlock(block_id="b1", type=SemanticBlockType.PARAGRAPH, content=original)
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertIn("markdown_emphasis", codes)
        self.assertEqual(content_block.content, original)

    def test_already_converted_content_passes(self) -> None:
        # The flip side of the case above: this must not become an
        # unconditional fail. Content with no remaining "**"/"__"
        # delimiter at all -- already converted, or always plain -- still
        # passes.
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block = self._layout_block()
        content_block = ContentBlock(
            block_id="b1", type=SemanticBlockType.PARAGRAPH, content=r"This is \textbf{important}."
        )
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertNotIn("markdown_emphasis", codes)

    def test_digit_emphasis_glued_to_a_word_at_end_of_line_fails_the_check(self) -> None:
        # Round-4 review finding: the empty-string membership bug made
        # this exact input invisible to both the generation-time check
        # and the composer's residual check -- it passed both silently.
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block = self._layout_block()
        content_block = ContentBlock(
            block_id="b1", type=SemanticBlockType.PARAGRAPH, content="**first**and**2**"
        )
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertIn("markdown_emphasis", codes)

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

    def test_caption_with_bare_dollar_and_markdown_emphasis_is_still_rejected(self) -> None:
        # Round-2 review finding: validate_plain_caption used to apply the
        # math-exclusion mask, so an unterminated bare "$" in a caption
        # (never real math -- caption is rendered through _escape_latex
        # only) wrongly "protected" everything after it, including a real
        # "**discount**" violation, letting the whole caption through.
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block = self._layout_block(LayoutBlockKind.TABLE)
        content_block = ContentBlock(
            block_id="b1", type=SemanticBlockType.TABLE,
            content="| A | B |\n| --- | --- |\n| 1 | 2 |",
            caption="Cost $5 **discount**",
        )
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertIn("plain_caption", codes)

    def test_equation_block_bare_formula_is_exempt(self) -> None:
        # Round-3 review finding: an EQUATION block's whole content IS a
        # math body, with no delimiter required at all -- "V^{**} = V" is
        # valid, correct LaTeX with zero Markdown intent, but the detector
        # has no way to know that without being told the block's kind.
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block = self._layout_block(LayoutBlockKind.EQUATION)
        content_block = ContentBlock(
            block_id="b1", type=SemanticBlockType.EQUATION, content="V^{**} = V"
        )
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertNotIn("markdown_emphasis", codes)

    def test_equation_block_align_environment_is_exempt(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block = self._layout_block(LayoutBlockKind.EQUATION)
        content_block = ContentBlock(
            block_id="b1", type=SemanticBlockType.EQUATION,
            content=r"\begin{align} V^{**} &= V \end{align}",
        )
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertNotIn("markdown_emphasis", codes)


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
            self.assertEqual(record["phase"], "generation")
            self.assertEqual(record["rule_version"], MARKDOWN_EMPHASIS_RULE_VERSION)
            self.assertEqual(record["original_content"], "This is **important**.")
            self.assertEqual(record["converted_content"], r"This is \textbf{important}.")
            self.assertIn("textbf", record["diff"])
            self.assertEqual(record["safe_match_count"], 1)
            self.assertEqual(record["ambiguous_match_count"], 0)
            self.assertGreaterEqual(record["transform_latency_s"], 0.0)

    def test_provenance_is_persisted_even_when_nothing_needed_converting(self) -> None:
        # Round-2 review finding: a record used to exist only when
        # converted != original, so a block with no Markdown emphasis at
        # all -- the common case -- left no record that its check had
        # ever run: no latency, no rule_version, nothing. The check/convert
        # pass is persisted every time now; converted_content stays None
        # when there was nothing to change.
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        with tempfile.TemporaryDirectory() as tmp:
            sections_dir = Path(tmp)
            raw_blocks = [{"block_id": "b1", "type": "paragraph", "content": "Plain content."}]
            agent._validate_blocks_against_layout(raw_blocks, self._layout_blocks(), sections_dir, "sec1")

            provenance_files = list(sections_dir.glob("sec1.markdown_emphasis_conversion_*.json"))
            self.assertEqual(len(provenance_files), 1)
            record = json.loads(provenance_files[0].read_text(encoding="utf-8"))
            self.assertIsNone(record["converted_content"])
            self.assertEqual(record["safe_match_count"], 0)
            self.assertEqual(record["ambiguous_match_count"], 0)
            self.assertGreaterEqual(record["transform_latency_s"], 0.0)

    def test_provenance_is_persisted_for_an_all_ambiguous_block_even_though_it_then_fails(self) -> None:
        # The other previously-unrecorded gap: a block whose Markdown
        # emphasis is entirely ambiguous never changes (converted_content
        # stays None), but the check still ran, still cost real latency,
        # and the block still fails validation right after -- all three
        # facts must be recoverable from the persisted record.
        from src.agents.text_agent import SemanticBlockErrors

        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        with tempfile.TemporaryDirectory() as tmp:
            sections_dir = Path(tmp)
            raw_blocks = [
                {"block_id": "b1", "type": "paragraph", "content": "This has an **unterminated marker."}
            ]
            with self.assertRaises(SemanticBlockErrors):
                agent._validate_blocks_against_layout(raw_blocks, self._layout_blocks(), sections_dir, "sec1")

            provenance_files = list(sections_dir.glob("sec1.markdown_emphasis_conversion_*.json"))
            self.assertEqual(len(provenance_files), 1)
            record = json.loads(provenance_files[0].read_text(encoding="utf-8"))
            self.assertIsNone(record["converted_content"])
            self.assertEqual(record["ambiguous_match_count"], 1)

    def test_equation_block_is_skipped_entirely_and_not_flagged(self) -> None:
        # Round-3 review finding: without this exemption, a correct bare
        # formula like "V^{**} = V" would fail validation right after
        # this conversion step (nothing safely converts it, since it's
        # not Markdown at all) and get regenerated forever.
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_blocks = [LayoutBlock(id="eq1", kind=LayoutBlockKind.EQUATION)]
        with tempfile.TemporaryDirectory() as tmp:
            sections_dir = Path(tmp)
            raw_blocks = [{"block_id": "eq1", "type": "equation", "content": "V^{**} = V"}]
            blocks = agent._validate_blocks_against_layout(raw_blocks, layout_blocks, sections_dir, "sec1")
            self.assertEqual(blocks[0].content, "V^{**} = V")
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

    def test_equation_block_bare_formula_is_exempt_from_the_residual_check(self) -> None:
        # Round-3 review finding: without this exemption, every correct
        # EQUATION artifact using "**" as a genuine LaTeX superscript (not
        # Markdown at all) would fail right before rendering.
        assembler = LatexAssembler()
        block = ContentBlock(
            block_id="eq1", type=SemanticBlockType.EQUATION, content="V^{**} = V"
        )
        assembler._validate_fragment(block)  # must not raise

    def test_equation_block_title_emphasis_is_still_rejected(self) -> None:
        # The EQUATION exemption covers `content` only -- title/caption
        # are separate plain-text fields with no such exemption.
        assembler = LatexAssembler()
        block = ContentBlock(
            block_id="eq1", type=SemanticBlockType.EQUATION,
            content="V^{**} = V", title="**Bold Title**",
        )
        with self.assertRaises(ValueError):
            assembler._validate_fragment(block)

    def test_residual_check_cost_is_persisted_when_sections_dir_is_given(self) -> None:
        # Round-2 review finding: this check's own cost went completely
        # unrecorded before -- unlike TextAgent's generation-time
        # conversion pass. sections_dir/node_id are optional (default
        # None, same convention as event_logger elsewhere in this class)
        # so every other test in this class above, which calls
        # _validate_fragment(block) with neither, is unaffected; the real
        # _validate_semantic_blocks call site always supplies both.
        assembler = LatexAssembler()
        block = ContentBlock(
            block_id="p1", type=SemanticBlockType.PARAGRAPH,
            content=r"This is \textbf{important}.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            sections_dir = Path(tmp)
            assembler._validate_fragment(block, sections_dir, "sec1")
            provenance_files = list(sections_dir.glob("sec1.markdown_emphasis_conversion_*.json"))
            self.assertEqual(len(provenance_files), 1)
            record = json.loads(provenance_files[0].read_text(encoding="utf-8"))
            self.assertEqual(record["phase"], "assembler_residual_check")
            self.assertEqual(record["block_id"], "p1")
            self.assertIsNone(record["converted_content"])
            self.assertGreaterEqual(record["transform_latency_s"], 0.0)

    def test_residual_check_cost_is_persisted_even_when_the_check_then_fails(self) -> None:
        assembler = LatexAssembler()
        block = ContentBlock(
            block_id="p1", type=SemanticBlockType.PARAGRAPH, content="This is **important**."
        )
        with tempfile.TemporaryDirectory() as tmp:
            sections_dir = Path(tmp)
            with self.assertRaises(ValueError):
                assembler._validate_fragment(block, sections_dir, "sec1")
            provenance_files = list(sections_dir.glob("sec1.markdown_emphasis_conversion_*.json"))
            self.assertEqual(len(provenance_files), 1)
            record = json.loads(provenance_files[0].read_text(encoding="utf-8"))
            self.assertEqual(record["safe_match_count"], 1)


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

    def test_figure_caption_check_cost_is_persisted_when_sections_dir_is_given(self) -> None:
        graph, blueprint = self._graph_with_figure("Plain caption")
        assembler = LatexAssembler()
        assembler._rendered_labels = set()
        with tempfile.TemporaryDirectory() as tmp:
            sections_dir = Path(tmp)
            assembler._render_figure("fig1", "sec1", blueprint, graph, sections_dir)
            provenance_files = list(sections_dir.glob("sec1.markdown_emphasis_conversion_*.json"))
            self.assertEqual(len(provenance_files), 1)
            record = json.loads(provenance_files[0].read_text(encoding="utf-8"))
            self.assertEqual(record["phase"], "assembler_figure_caption_check")
            self.assertEqual(record["block_id"], "fig1")


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

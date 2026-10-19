"""Regression tests for TextAgent's table_shape/equation_shape checks: a
`table`/`notation_table` block whose content isn't actually table-shaped, or
an `equation` block whose content has no math notation at all, must be
rejected -- previously neither was checked anywhere and silently rendered as
a broken fake table / garbled prose-in-math-mode (see LatexAssembler.
_single_column_table / _render_equation)."""

from __future__ import annotations

import unittest

from src.agents.text_agent import TextAgent
from src.common.schemas import ContentBlock, SemanticBlockType
from src.common.state import LayoutBlock, LayoutBlockKind

# _validate_one_block never calls the LLM client -- avoid TextAgent's default
# `llm_client or LLMClient()` fallback, which would try to construct a real
# provider client and depend on credentials being configured.
_STUB_LLM_CLIENT = object()


class TableShapeValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        self.layout_block = LayoutBlock(id="tbl", kind=LayoutBlockKind.TABLE)

    def _errors_for(self, content: str) -> list:
        content_block = ContentBlock(block_id="tbl", type=SemanticBlockType.TABLE, content=content)
        return self.agent._validate_one_block(content_block, self.layout_block)

    def test_markdown_table_passes(self) -> None:
        content = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        codes = [e.code for e in self._errors_for(content)]
        self.assertNotIn("table_shape", codes)

    def test_tabularx_environment_passes(self) -> None:
        content = "\\begin{tabularx}{\\linewidth}{XX}\nA & B \\\\\n\\end{tabularx}"
        codes = [e.code for e in self._errors_for(content)]
        self.assertNotIn("table_shape", codes)

    def test_plain_prose_is_rejected(self) -> None:
        content = "The results improved significantly across every measured category."
        codes = [e.code for e in self._errors_for(content)]
        self.assertIn("table_shape", codes)

    def test_disabled_validator_skips_the_check(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT, enabled_validators={"table_shape": False})
        content_block = ContentBlock(
            block_id="tbl", type=SemanticBlockType.TABLE, content="Just a sentence, no table here."
        )
        codes = [e.code for e in agent._validate_one_block(content_block, self.layout_block)]
        self.assertNotIn("table_shape", codes)


class EquationShapeValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        self.layout_block = LayoutBlock(id="eq", kind=LayoutBlockKind.EQUATION)

    def _errors_for(self, content: str) -> list:
        content_block = ContentBlock(block_id="eq", type=SemanticBlockType.EQUATION, content=content)
        return self.agent._validate_one_block(content_block, self.layout_block)

    def test_bare_formula_with_no_dollar_signs_passes(self) -> None:
        # A bare equation body (no $...$ wrapper) is the expected shape for
        # this kind -- LatexAssembler._render_equation wraps it in
        # \begin{equation}...\end{equation} itself. This must NOT be rejected
        # just because it lacks '$'.
        codes = [e.code for e in self._errors_for("E = mc^2")]
        self.assertNotIn("equation_shape", codes)

    def test_dollar_wrapped_formula_passes(self) -> None:
        codes = [e.code for e in self._errors_for("$E = mc^2$")]
        self.assertNotIn("equation_shape", codes)

    def test_prose_with_no_math_notation_is_rejected(self) -> None:
        content = "Energy equals mass times the speed of light squared."
        codes = [e.code for e in self._errors_for(content)]
        self.assertIn("equation_shape", codes)

    def test_disabled_validator_skips_the_check(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT, enabled_validators={"equation_shape": False})
        content_block = ContentBlock(
            block_id="eq", type=SemanticBlockType.EQUATION, content="Energy equals mass times speed squared."
        )
        codes = [e.code for e in agent._validate_one_block(content_block, self.layout_block)]
        self.assertNotIn("equation_shape", codes)


if __name__ == "__main__":
    unittest.main()

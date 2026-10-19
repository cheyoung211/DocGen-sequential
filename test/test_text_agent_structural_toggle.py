"""Regression tests for TextAgent's type_mismatch/empty_content toggles.

These two used to always run regardless of ``enabled_validators`` (the
composer's single-sole-input contract depends on them) -- they were made
toggleable for full verifier-ablation runs on the understanding that
LatexAssembler._validate_semantic_blocks (the composer_semantic_block_validator
gate) independently re-checks both unconditionally, so disabling them here
doesn't let a defect through silently -- it just surfaces later, at the
composer, instead of at TextAgent. See _validate_one_block's docstring."""

from __future__ import annotations

import unittest

from src.agents.text_agent import TextAgent
from src.common.schemas import ContentBlock, SemanticBlockType
from src.common.state import LayoutBlock, LayoutBlockKind

# _validate_one_block never calls the LLM client -- avoid TextAgent's default
# `llm_client or LLMClient()` fallback, which would try to construct a real
# provider client and depend on credentials being configured.
_STUB_LLM_CLIENT = object()


class TypeMismatchToggleTest(unittest.TestCase):
    def _layout_and_block(self, content: str) -> tuple:
        layout_block = LayoutBlock(id="para", kind=LayoutBlockKind.PARAGRAPH)
        content_block = ContentBlock(block_id="para", type=SemanticBlockType.TABLE, content=content)
        return layout_block, content_block

    def test_mismatched_type_is_rejected_by_default(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block, content_block = self._layout_and_block("| A | B |\n| --- | --- |\n| 1 | 2 |")
        codes = [e.code for e in agent._validate_one_block(content_block, layout_block)]
        self.assertIn("type_mismatch", codes)

    def test_disabled_validator_skips_the_check_and_continues_validating(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT, enabled_validators={"type_mismatch": False})
        layout_block, content_block = self._layout_and_block("| A | B |\n| --- | --- |\n| 1 | 2 |")
        errors = agent._validate_one_block(content_block, layout_block)
        codes = [e.code for e in errors]
        self.assertNotIn("type_mismatch", codes)
        # Falls through instead of short-circuiting -- other checks still ran.
        self.assertNotIn("empty_content", codes)


class EmptyContentToggleTest(unittest.TestCase):
    def _layout_and_block(self) -> tuple:
        layout_block = LayoutBlock(id="para", kind=LayoutBlockKind.PARAGRAPH)
        content_block = ContentBlock(block_id="para", type=SemanticBlockType.PARAGRAPH, content="   ")
        return layout_block, content_block

    def test_empty_content_is_rejected_by_default(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT)
        layout_block, content_block = self._layout_and_block()
        codes = [e.code for e in agent._validate_one_block(content_block, layout_block)]
        self.assertIn("empty_content", codes)

    def test_disabled_validator_skips_the_check(self) -> None:
        agent = TextAgent(llm_client=_STUB_LLM_CLIENT, enabled_validators={"empty_content": False})
        layout_block, content_block = self._layout_and_block()
        codes = [e.code for e in agent._validate_one_block(content_block, layout_block)]
        self.assertNotIn("empty_content", codes)


if __name__ == "__main__":
    unittest.main()

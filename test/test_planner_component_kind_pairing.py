"""Regression tests for PlannerAgent._validate_blueprint's component_kind_pairing
gate: a LayoutBlock's `component` must be one KIND_ALLOWED_COMPONENTS actually
allows for its `kind` -- e.g. `kind: equation, component: tabularx` (or the
`kind: figure, component: figure` confusion the prompt already warns about
in prose) is otherwise silently accepted and simply ignored by every renderer."""

from __future__ import annotations

import unittest

from src.agents.planner import PlannerAgent
from src.common.schemas import PlannerOutput
from src.common.state import (
    DocumentBlueprint,
    DocumentKind,
    DocumentSpec,
    LatexComponent,
    LayoutBlock,
    LayoutBlockKind,
    SectionLayout,
)

# _validate_blueprint never calls the LLM client -- avoid PlannerAgent's
# default `llm_client or LLMClient()` fallback, which would try to construct
# a real provider client and depend on credentials being configured.
_STUB_LLM_CLIENT = object()


def _plan_with_block(block: LayoutBlock) -> PlannerOutput:
    blueprint = DocumentBlueprint(
        document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Component Pairing Test"),
        document_class="article",
        sections=[SectionLayout(section_id="overview", title="Overview", blocks=[block])],
    )
    return PlannerOutput(
        nodes_to_create=[{"id": "overview", "type": "SECTION"}],
        hierarchy_edges=[],
        global_context="Test document.",
        blueprint=blueprint,
    )


class ComponentKindPairingValidationTest(unittest.TestCase):
    def test_matching_pairing_passes(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        block = LayoutBlock(id="tbl", kind=LayoutBlockKind.TABLE, component=LatexComponent.TABULARX)
        agent._validate_blueprint(_plan_with_block(block))  # no raise

    def test_plain_is_always_allowed(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        block = LayoutBlock(id="eq", kind=LayoutBlockKind.EQUATION, component=LatexComponent.PLAIN)
        agent._validate_blueprint(_plan_with_block(block))  # no raise

    def test_table_component_on_an_equation_block_is_rejected(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        block = LayoutBlock(id="eq", kind=LayoutBlockKind.EQUATION, component=LatexComponent.TABULARX)
        with self.assertRaisesRegex(ValueError, "component_kind_pairing|component"):
            agent._validate_blueprint(_plan_with_block(block))

    def test_figure_component_on_a_figure_kind_block_is_rejected(self) -> None:
        # The exact confusion PLANNER_STRATEGY_PROMPT already warns against
        # in prose: a `figure`-kind LayoutBlock is prose introducing the
        # image, not the image itself -- FIGURE_FLOAT belongs to the
        # separate FigurePlacement.layout field, never to LayoutBlock.component.
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        block = LayoutBlock(
            id="fig_ref",
            kind=LayoutBlockKind.FIGURE,
            component=LatexComponent.FIGURE_FLOAT,
            asset_id="system_diagram",
        )
        with self.assertRaises(ValueError):
            agent._validate_blueprint(_plan_with_block(block))

    def test_disabled_validator_skips_the_check(self) -> None:
        agent = PlannerAgent(
            llm_client=_STUB_LLM_CLIENT, enabled_validators={"component_kind_pairing": False}
        )
        block = LayoutBlock(id="eq", kind=LayoutBlockKind.EQUATION, component=LatexComponent.TABULARX)
        agent._validate_blueprint(_plan_with_block(block))  # no raise


if __name__ == "__main__":
    unittest.main()

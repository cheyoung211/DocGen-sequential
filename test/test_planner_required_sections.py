"""Regression tests for PlannerAgent._validate_blueprint's required_sections
gate: a blueprint that doesn't cover every RequiredSection from the request
must be rejected (retryable, same as every other _validate_blueprint check),
unless the check is explicitly disabled via enabled_validators."""

from __future__ import annotations

import unittest

from src.agents.planner import PlannerAgent
from src.common.schemas import PlannerOutput
from src.common.state import (
    DocumentBlueprint,
    DocumentKind,
    DocumentSpec,
    LayoutBlock,
    LayoutBlockKind,
    SectionLayout,
)
from src.dataset.schemas import RequiredSection

# _validate_blueprint never calls the LLM client -- avoid PlannerAgent's
# default `llm_client or LLMClient()` fallback, which would try to construct
# a real provider client and depend on credentials being configured.
_STUB_LLM_CLIENT = object()


def _plan(section_ids: list[str]) -> PlannerOutput:
    sections = [
        SectionLayout(
            section_id=section_id,
            title=section_id.replace("_", " ").title(),
            blocks=[LayoutBlock(id=f"{section_id}_intro", kind=LayoutBlockKind.PARAGRAPH)],
        )
        for section_id in section_ids
    ]
    blueprint = DocumentBlueprint(
        document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Required Sections Test"),
        document_class="article",
        sections=sections,
    )
    return PlannerOutput(
        nodes_to_create=[{"id": section_id, "type": "SECTION"} for section_id in section_ids],
        hierarchy_edges=[],
        global_context="Test document.",
        blueprint=blueprint,
    )


class RequiredSectionsValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.required = [
            RequiredSection(
                section_id="methodology",
                title="Methodology",
                purpose="Explain the approach taken.",
                required_points=["data collection", "analysis method"],
                target_words=200,
            )
        ]

    def test_covered_required_section_passes(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        plan = _plan(["overview", "methodology"])
        agent._validate_blueprint(plan, required_sections=self.required)  # no raise

    def test_missing_required_section_raises(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        plan = _plan(["overview", "conclusion"])
        with self.assertRaisesRegex(ValueError, "missing required section"):
            agent._validate_blueprint(plan, required_sections=self.required)

    def test_disabled_validator_skips_the_check(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT, enabled_validators={"required_sections": False})
        plan = _plan(["overview", "conclusion"])
        agent._validate_blueprint(plan, required_sections=self.required)  # no raise

    def test_title_match_covers_a_section_id_that_was_renamed(self) -> None:
        # The planner is free to reuse the exact section_id, but if it
        # doesn't, an exact normalized-title match should still count --
        # this is the same tolerance evaluate_contract's post-hoc scoring
        # already gives a differently-id'd section with a matching title.
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        sections = [
            SectionLayout(
                section_id="sec_methods",
                title="Methodology",
                blocks=[LayoutBlock(id="sec_methods_intro", kind=LayoutBlockKind.PARAGRAPH)],
            )
        ]
        blueprint = DocumentBlueprint(
            document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Required Sections Test"),
            document_class="article",
            sections=sections,
        )
        plan = PlannerOutput(
            nodes_to_create=[{"id": "sec_methods", "type": "SECTION"}],
            hierarchy_edges=[],
            global_context="Test document.",
            blueprint=blueprint,
        )
        agent._validate_blueprint(plan, required_sections=self.required)  # no raise


if __name__ == "__main__":
    unittest.main()

"""Regression tests for PlannerAgent._validate_blueprint's cross-section
dependency wiring: a benchmark CrossSectionDependency (whose source_section/
target_sections reference RequiredSection.section_id, not the Planner's own
generated node ids) must be resolved onto the actual generated section ids
and written into the target node's `dependencies` + `spec["dependency_notes"]`
-- the same id-matching PlannerAgent already does for required_sections
coverage (see _match_required_section), reused here rather than
reimplemented."""

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
from src.dataset.schemas import CrossSectionDependency, DependencyRelation, RequiredSection

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
        document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Cross-Section Dependency Test"),
        document_class="article",
        sections=sections,
    )
    return PlannerOutput(
        nodes_to_create=[{"id": section_id, "type": "SECTION"} for section_id in section_ids],
        hierarchy_edges=[],
        global_context="Test document.",
        blueprint=blueprint,
    )


class CrossSectionDependencyWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.required = [
            RequiredSection(
                section_id="intro",
                title="Introduction",
                purpose="Establish the taxonomy.",
                target_words=200,
            ),
            RequiredSection(
                section_id="results",
                title="Results",
                purpose="Report findings using the taxonomy.",
                target_words=200,
            ),
        ]
        self.dependency = CrossSectionDependency(
            dependency_id="dep_1",
            source_section="intro",
            target_sections=["results"],
            relation=DependencyRelation.MUST_REUSE,
            description="Results must reuse the exact taxonomy defined in the introduction.",
        )

    def _nodes_by_id(self, plan: PlannerOutput) -> dict:
        return {node["id"]: node for node in plan.nodes_to_create}

    def test_dependency_is_wired_onto_the_matched_target_node(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        plan = _plan(["intro", "results"])
        agent._validate_blueprint(
            plan,
            required_sections=self.required,
            cross_section_dependencies=[self.dependency],
        )
        nodes_by_id = self._nodes_by_id(plan)
        self.assertEqual(nodes_by_id["results"]["dependencies"], ["intro"])
        self.assertNotIn("dependencies", nodes_by_id["intro"])

        notes = nodes_by_id["results"]["spec"]["dependency_notes"]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["depends_on"], "intro")
        self.assertEqual(notes[0]["relation"], "must_reuse")
        self.assertEqual(notes[0]["description"], self.dependency.description)

    def test_dependency_survives_section_id_renaming_via_title_match(self) -> None:
        # Same tolerance _match_required_section already gives required_sections
        # coverage: the Planner is free to invent its own ids as long as the
        # title matches.
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        plan = _plan(["sec_a", "sec_b"])
        # Retitle to match by normalized title instead of section_id.
        plan.blueprint.sections[0].title = "Introduction"
        plan.blueprint.sections[1].title = "Results"
        agent._validate_blueprint(
            plan,
            required_sections=self.required,
            cross_section_dependencies=[self.dependency],
        )
        nodes_by_id = self._nodes_by_id(plan)
        self.assertEqual(nodes_by_id["sec_b"]["dependencies"], ["sec_a"])

    def test_unresolvable_dependency_is_skipped_silently(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        # Only "intro" is covered -- "results" never matches, so the
        # dependency's target endpoint can't be resolved.
        plan = _plan(["intro"])
        agent._validate_blueprint(
            plan,
            required_sections=[self.required[0]],
            cross_section_dependencies=[self.dependency],
        )  # no raise
        nodes_by_id = self._nodes_by_id(plan)
        self.assertNotIn("dependencies", nodes_by_id["intro"])

    def test_wiring_is_independent_of_the_required_sections_qa_toggle(self) -> None:
        # Disabling the required_sections *coverage check* must not disable
        # dependency wiring -- it's data enrichment, not a QA gate.
        agent = PlannerAgent(
            llm_client=_STUB_LLM_CLIENT, enabled_validators={"required_sections": False}
        )
        plan = _plan(["intro", "conclusion"])  # "results" is missing/uncovered
        agent._validate_blueprint(
            plan,
            required_sections=self.required,
            cross_section_dependencies=[self.dependency],
        )  # no raise, even though "results" is missing
        nodes_by_id = self._nodes_by_id(plan)
        self.assertNotIn("dependencies", nodes_by_id["conclusion"])


if __name__ == "__main__":
    unittest.main()

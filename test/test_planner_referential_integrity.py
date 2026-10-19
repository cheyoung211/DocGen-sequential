"""Regression tests for PlannerAgent._validate_blueprint's five referential-
integrity gates: duplicate_ids, section_graph_consistency, non_empty_sections,
figure_reference_integrity, document_title_required.

These used to be unconditional (no self.enabled_validators.get(...) gate at
all) -- they were made toggleable so a full verifier-ablation run can turn
every disable-able check off, per the same reasoning as the rest of this
file's checks. Unlike TextAgent's type_mismatch/empty_content, most of these
have no redundant always-on re-check downstream: disabling one means a
defect surfaces later as a less specific (possibly uncaught) exception in
TextAgent or the composer, rather than this method's clean early ValueError.
See _validate_blueprint's docstring."""

from __future__ import annotations

import unittest

from src.agents.planner import PlannerAgent
from src.common.schemas import PlannerOutput
from src.common.state import (
    DocumentBlueprint,
    DocumentKind,
    DocumentSpec,
    FigurePlacement,
    LayoutBlock,
    LayoutBlockKind,
    SectionLayout,
)

# _validate_blueprint never calls the LLM client -- avoid PlannerAgent's
# default `llm_client or LLMClient()` fallback, which would try to construct
# a real provider client and depend on credentials being configured.
_STUB_LLM_CLIENT = object()


def _plan(nodes, sections, figure_slots=None, title="Referential Integrity Test") -> PlannerOutput:
    blueprint = DocumentBlueprint(
        document=DocumentSpec(kind=DocumentKind.ARTICLE, title=title),
        document_class="article",
        sections=sections,
        figure_slots=figure_slots or {},
    )
    return PlannerOutput(
        nodes_to_create=nodes,
        hierarchy_edges=[],
        global_context="Test document.",
        blueprint=blueprint,
    )


def _section(section_id: str, block_id: str = "p1") -> SectionLayout:
    return SectionLayout(
        section_id=section_id,
        title="Overview",
        blocks=[LayoutBlock(id=block_id, kind=LayoutBlockKind.PARAGRAPH)],
    )


class DuplicateIdsToggleTest(unittest.TestCase):
    def test_duplicate_node_id_rejected_by_default(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        nodes = [{"id": "overview", "type": "SECTION"}, {"id": "overview", "type": "SECTION"}]
        plan = _plan(nodes, [_section("overview")])
        with self.assertRaises(ValueError):
            agent._validate_blueprint(plan)

    def test_disabled_validator_skips_duplicate_node_id_check(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT, enabled_validators={"duplicate_ids": False})
        nodes = [{"id": "overview", "type": "SECTION"}, {"id": "overview", "type": "SECTION"}]
        plan = _plan(nodes, [_section("overview")])
        agent._validate_blueprint(plan)  # no raise

    def test_disabled_validator_skips_duplicate_block_id_check(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT, enabled_validators={"duplicate_ids": False})
        section = SectionLayout(
            section_id="overview",
            title="Overview",
            blocks=[
                LayoutBlock(id="p1", kind=LayoutBlockKind.PARAGRAPH),
                LayoutBlock(id="p1", kind=LayoutBlockKind.PARAGRAPH),
            ],
        )
        plan = _plan([{"id": "overview", "type": "SECTION"}], [section])
        agent._validate_blueprint(plan)  # no raise


class SectionGraphConsistencyToggleTest(unittest.TestCase):
    def test_mismatched_sections_rejected_by_default(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        nodes = [{"id": "overview", "type": "SECTION"}]
        plan = _plan(nodes, [_section("different_id")])
        with self.assertRaises(ValueError):
            agent._validate_blueprint(plan)

    def test_disabled_validator_skips_the_check(self) -> None:
        agent = PlannerAgent(
            llm_client=_STUB_LLM_CLIENT, enabled_validators={"section_graph_consistency": False}
        )
        nodes = [{"id": "overview", "type": "SECTION"}]
        plan = _plan(nodes, [_section("different_id")])
        agent._validate_blueprint(plan)  # no raise


class NonEmptySectionsToggleTest(unittest.TestCase):
    def test_empty_section_rejected_by_default(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        empty_section = SectionLayout(section_id="overview", title="Overview", blocks=[])
        plan = _plan([{"id": "overview", "type": "SECTION"}], [empty_section])
        with self.assertRaises(ValueError):
            agent._validate_blueprint(plan)

    def test_disabled_validator_skips_the_check(self) -> None:
        agent = PlannerAgent(
            llm_client=_STUB_LLM_CLIENT, enabled_validators={"non_empty_sections": False}
        )
        empty_section = SectionLayout(section_id="overview", title="Overview", blocks=[])
        plan = _plan([{"id": "overview", "type": "SECTION"}], [empty_section])
        agent._validate_blueprint(plan)  # no raise


class FigureReferenceIntegrityToggleTest(unittest.TestCase):
    def test_orphan_figure_rejected_by_default(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        nodes = [
            {"id": "overview", "type": "SECTION"},
            {"id": "fig_a", "type": "FIGURE"},
        ]
        figure_slots = {"fig_a": FigurePlacement(figure_id="fig_a", owner_section="overview")}
        plan = _plan(nodes, [_section("overview")], figure_slots=figure_slots)
        with self.assertRaises(ValueError):
            agent._validate_blueprint(plan)

    def test_disabled_validator_skips_orphan_figure_check(self) -> None:
        agent = PlannerAgent(
            llm_client=_STUB_LLM_CLIENT, enabled_validators={"figure_reference_integrity": False}
        )
        nodes = [
            {"id": "overview", "type": "SECTION"},
            {"id": "fig_a", "type": "FIGURE"},
        ]
        figure_slots = {"fig_a": FigurePlacement(figure_id="fig_a", owner_section="overview")}
        plan = _plan(nodes, [_section("overview")], figure_slots=figure_slots)
        agent._validate_blueprint(plan)  # no raise

    def test_disabled_validator_skips_missing_figure_slot_check(self) -> None:
        agent = PlannerAgent(
            llm_client=_STUB_LLM_CLIENT, enabled_validators={"figure_reference_integrity": False}
        )
        nodes = [
            {"id": "overview", "type": "SECTION"},
            {"id": "fig_a", "type": "FIGURE"},
        ]
        # FIGURE node with no matching figure_slots entry at all.
        plan = _plan(nodes, [_section("overview")])
        agent._validate_blueprint(plan)  # no raise


class DocumentTitleRequiredToggleTest(unittest.TestCase):
    def test_missing_title_rejected_by_default(self) -> None:
        agent = PlannerAgent(llm_client=_STUB_LLM_CLIENT)
        plan = _plan([{"id": "overview", "type": "SECTION"}], [_section("overview")], title="")
        with self.assertRaises(ValueError):
            agent._validate_blueprint(plan)

    def test_disabled_validator_skips_the_check(self) -> None:
        agent = PlannerAgent(
            llm_client=_STUB_LLM_CLIENT, enabled_validators={"document_title_required": False}
        )
        plan = _plan([{"id": "overview", "type": "SECTION"}], [_section("overview")], title="")
        agent._validate_blueprint(plan)  # no raise


if __name__ == "__main__":
    unittest.main()

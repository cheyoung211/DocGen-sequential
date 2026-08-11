"""Coverage for evaluate_contract() against a synthetic BenchmarkItem +
synthetic generated output (plan.json / sections/*.blocks.json), written
into a temp dir -- mirroring test_semantic_integrator.py's own
fixture-building style."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.common.schemas import ContentBlock, SemanticBlockType, TextAgentOutput
from src.common.state import (
    DocumentBlueprint,
    DocumentKind,
    DocumentSpec,
    LayoutBlock,
    LayoutBlockKind,
    SectionLayout,
)
from src.dataset.schemas import (
    BenchmarkItem,
    ComponentType,
    ConstraintType,
    CrossSectionDependency,
    Difficulty,
    Domain,
    DocumentType,
    DependencyRelation,
    EvaluationContract,
    HardConstraint,
    LayoutRequirements,
    NotationConstraint,
    RequiredComponent,
    RequiredSection,
    SourceProvenanceEntry,
    TargetLength,
    TerminologyConstraint,
)
from src.evaluation.metrics.contract_satisfaction import evaluate_contract


def _benchmark_item(forbidden_variant_in_text: bool = False) -> BenchmarkItem:
    return BenchmarkItem(
        sample_id="test_item_0001",
        domain=Domain.MATHEMATICS,
        subdomain="real_analysis",
        document_type=DocumentType.LECTURE_NOTE,
        difficulty=Difficulty.BASIC,
        title="Convergence",
        overview="A short note.",
        audience="students",
        document_purpose="teach",
        target_length=TargetLength(target_pages=1, min_words=100, max_words=500, min_sections=1, max_sections=3),
        required_sections=[
            RequiredSection(section_id="introduction", title="Introduction", purpose="intro", target_words=50),
            RequiredSection(section_id="conclusion", title="Conclusion", purpose="wrap up", target_words=50),
        ],
        required_components=[
            RequiredComponent(
                component_id="definition_core", type=ComponentType.DEFINITION, placement="introduction",
                description="Define convergence.",
            ),
            RequiredComponent(
                component_id="prop_1", type=ComponentType.PROPOSITION, placement="introduction",
                description="State a proposition.",
            ),
        ],
        hard_constraints=[
            HardConstraint(constraint_id="terminology_consistency", type=ConstraintType.TERMINOLOGY_CONSISTENCY, description="d"),
        ],
        cross_section_dependencies=[
            CrossSectionDependency(
                dependency_id="dep_1", source_section="introduction", target_sections=["conclusion"],
                relation=DependencyRelation.MUST_REUSE, description="d",
            ),
        ],
        terminology_constraints=[
            TerminologyConstraint(
                canonical_term="convergence",
                forbidden_variants=["divergence"] if forbidden_variant_in_text else ["nonexistent_forbidden_term"],
            ),
        ],
        notation_constraints=[NotationConstraint(symbol="a_n", meaning="sequence term")],
        layout_requirements=LayoutRequirements(preferred_layout_profile="technical_lecture_note"),
        evaluation_contract=EvaluationContract(),
        natural_language_instruction="Write about convergence.",
        source_provenance=[SourceProvenanceEntry(source_id="standard_curriculum", usage="topic seed")],
    )


def _write_generated_output(project_dir: Path, block_content: str) -> None:
    blueprint = DocumentBlueprint(
        document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Convergence"),
        document_class="article",
        sections=[
            SectionLayout(
                section_id="sec_intro",
                title="Introduction",
                blocks=[LayoutBlock(id="def1", kind=LayoutBlockKind.DEFINITION)],
            ),
            # No "Conclusion" section at all -- simulates a required section
            # the planner never produced.
        ],
    )
    from src.common.schemas import PlannerOutput

    plan = PlannerOutput(
        request_id="test-run",
        nodes_to_create=[{"id": "sec_intro", "type": "SECTION", "title": "Introduction"}],
        hierarchy_edges=[],
        global_context="",
        blueprint=blueprint,
    )
    (project_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    sections_dir = project_dir / "sections"
    sections_dir.mkdir(parents=True)
    output = TextAgentOutput(
        request_id="test-run",
        section_id="sec_intro",
        blocks=[ContentBlock(block_id="def1", type=SemanticBlockType.DEFINITION, content=block_content)],
    )
    (sections_dir / "sec_intro.blocks.json").write_text(output.model_dump_json(indent=2), encoding="utf-8")


class EvaluateContractTest(unittest.TestCase):
    def test_missing_plan_json_returns_null_filled_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_contract(Path(tmp), _benchmark_item())
        self.assertIsNone(result.overall)
        self.assertIsNone(result.required_sections)
        self.assertEqual(result.not_implemented, ["content_coverage", "cross_section_dependency"])

    def test_partially_satisfied_contract_scores_each_category_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            _write_generated_output(
                project_dir,
                block_content="A sequence exhibits convergence as n grows, with a_n approaching the limit.",
            )
            result = evaluate_contract(project_dir, _benchmark_item())

        # 1 of 2 required sections drafted (Introduction yes, Conclusion no).
        self.assertAlmostEqual(result.required_sections, 0.5)
        section_statuses = {d.item_id: d.status for d in result.details["required_sections"]}
        self.assertEqual(section_statuses["introduction"], "satisfied")
        self.assertEqual(section_statuses["conclusion"], "missing")

        # definition_core is representable and present; prop_1 is not
        # representable at all and must be excluded from the denominator,
        # not counted as a failure -- so the score is 1/1, not 1/2.
        self.assertAlmostEqual(result.required_components, 1.0)
        component_statuses = {d.item_id: d.status for d in result.details["required_components"]}
        self.assertEqual(component_statuses["definition_core"], "satisfied")
        self.assertEqual(component_statuses["prop_1"], "not_representable")

        self.assertAlmostEqual(result.terminology, 1.0)
        self.assertAlmostEqual(result.notation, 1.0)

        self.assertIsNone(result.content_coverage)
        self.assertIsNone(result.cross_section_dependency)
        self.assertIsNotNone(result.overall)

    def test_forbidden_terminology_variant_present_fails_that_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            _write_generated_output(
                project_dir,
                block_content="A sequence shows convergence, in contrast to divergence elsewhere.",
            )
            result = evaluate_contract(project_dir, _benchmark_item(forbidden_variant_in_text=True))

        self.assertAlmostEqual(result.terminology, 0.0)
        detail = result.details["terminology"][0]
        self.assertEqual(detail.status, "missing")
        self.assertIn("forbidden", detail.reason)

    def test_notation_check_is_skipped_when_contract_flag_is_off(self) -> None:
        item = _benchmark_item()
        item.evaluation_contract.notation_consistency_check = False
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            _write_generated_output(project_dir, block_content="No notation mentioned here.")
            result = evaluate_contract(project_dir, item)
        self.assertIsNone(result.notation)
        self.assertEqual(result.details["notation"], [])


if __name__ == "__main__":
    unittest.main()

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


def _write_generated_output_with_section(
    project_dir: Path, *, generated_section_id: str, generated_title: str, block_content: str
) -> None:
    """Like ``_write_generated_output`` but with a caller-chosen generated
    section_id/title, so the three required_sections matching tiers
    (section_id / normalized title / keyword overlap) can be exercised
    independently."""
    blueprint = DocumentBlueprint(
        document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Convergence"),
        document_class="article",
        sections=[
            SectionLayout(
                section_id=generated_section_id,
                title=generated_title,
                blocks=[LayoutBlock(id="def1", kind=LayoutBlockKind.DEFINITION)],
            ),
        ],
    )
    from src.common.schemas import PlannerOutput

    plan = PlannerOutput(
        request_id="test-run",
        nodes_to_create=[{"id": generated_section_id, "type": "SECTION", "title": generated_title}],
        hierarchy_edges=[],
        global_context="",
        blueprint=blueprint,
    )
    (project_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    sections_dir = project_dir / "sections"
    sections_dir.mkdir(parents=True)
    output = TextAgentOutput(
        request_id="test-run",
        section_id=generated_section_id,
        blocks=[ContentBlock(block_id="def1", type=SemanticBlockType.DEFINITION, content=block_content)],
    )
    (sections_dir / f"{generated_section_id}.blocks.json").write_text(
        output.model_dump_json(indent=2), encoding="utf-8"
    )


def _write_generated_output_with_sections(
    project_dir: Path, sections: "list[tuple[str, str, str]]"
) -> None:
    """``sections`` is a list of (section_id, title, block_content) triples,
    all drafted. Used for topic-word-collision regression coverage, where a
    single generated section isn't enough to exercise cross-matching."""
    layouts = [
        SectionLayout(
            section_id=sid, title=title,
            blocks=[LayoutBlock(id=f"{sid}_def", kind=LayoutBlockKind.DEFINITION)],
        )
        for sid, title, _ in sections
    ]
    blueprint = DocumentBlueprint(
        document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Topic"),
        document_class="article",
        sections=layouts,
    )
    from src.common.schemas import PlannerOutput

    plan = PlannerOutput(
        request_id="test-run",
        nodes_to_create=[{"id": sid, "type": "SECTION", "title": title} for sid, title, _ in sections],
        hierarchy_edges=[],
        global_context="",
        blueprint=blueprint,
    )
    (project_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    sections_dir = project_dir / "sections"
    sections_dir.mkdir(parents=True)
    for sid, _, content in sections:
        output = TextAgentOutput(
            request_id="test-run", section_id=sid,
            blocks=[ContentBlock(block_id=f"{sid}_def", type=SemanticBlockType.DEFINITION, content=content)],
        )
        (sections_dir / f"{sid}.blocks.json").write_text(output.model_dump_json(indent=2), encoding="utf-8")


class RequiredSectionMatchingTest(unittest.TestCase):
    """Covers the section_id / title / keyword-overlap matching tiers and
    the required_points evidence scoring introduced to replace exact-title
    matching (which the benchmark schema never actually mandated)."""

    def _item_with_required_points(self) -> BenchmarkItem:
        item = _benchmark_item()
        item.required_sections = [
            RequiredSection(
                section_id="introduction", title="Introduction",
                purpose="Motivate convergence of real sequences and outline the document structure.",
                required_points=[
                    "State why convergence of real sequences matters.",
                    "Outline the structure of the remaining sections.",
                ],
                target_words=50,
            ),
        ]
        return item

    def test_section_id_tier_matches_even_with_a_different_title(self) -> None:
        item = self._item_with_required_points()
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            _write_generated_output_with_section(
                project_dir, generated_section_id="introduction",
                generated_title="Something Else Entirely",
                block_content="Convergence of real sequences matters because it underlies analysis. "
                              "This document's structure proceeds section by section.",
            )
            result = evaluate_contract(project_dir, item)
        detail = result.details["required_sections"][0]
        self.assertEqual(detail.matched_via, "section_id")
        self.assertEqual(detail.status, "satisfied")
        self.assertAlmostEqual(result.required_sections, 1.0)

    def test_keyword_overlap_tier_matches_an_elaborated_title(self) -> None:
        """Reproduces the live-validation case: the Planner elaborates
        "Introduction" into a longer title that shares no section_id and no
        normalized-title equality with the requirement, but does share real
        keywords with the requirement's own stated purpose."""
        item = self._item_with_required_points()
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            _write_generated_output_with_section(
                project_dir, generated_section_id="sec_0001",
                generated_title="Motivation and Structure of Convergence of Real Sequences",
                block_content="Convergence of real sequences matters because it underlies analysis. "
                              "This document's structure proceeds section by section.",
            )
            result = evaluate_contract(project_dir, item)
        detail = result.details["required_sections"][0]
        self.assertEqual(detail.matched_via, "keyword_overlap")
        self.assertEqual(detail.status, "satisfied")

    def test_partially_satisfied_when_only_some_required_points_are_evidenced(self) -> None:
        item = self._item_with_required_points()
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            _write_generated_output_with_section(
                project_dir, generated_section_id="introduction", generated_title="Introduction",
                # Covers the first required_point's keywords, not the second's.
                block_content="Convergence of real sequences matters a great deal in analysis.",
            )
            result = evaluate_contract(project_dir, item)
        detail = result.details["required_sections"][0]
        self.assertEqual(detail.status, "partially_satisfied")
        self.assertAlmostEqual(detail.required_points_coverage, 0.5)
        self.assertAlmostEqual(result.required_sections, 0.5)

    def test_shared_document_topic_words_do_not_cause_cross_matching(self) -> None:
        """Regression for a real bug hit during live validation: every
        required section's purpose restates the document's own topic (e.g.
        "... Basic Set Operations and Venn Diagrams"), and the Planner
        echoes that same topic phrase in every generated section's title
        too -- so unweighted keyword overlap let "Prerequisites and
        Notation" out-score-match onto a generated "Synthesis of ..."
        section instead of the real "Prerequisites and Notation for ..."
        one, purely on shared topic words. Topic words must be down-weighted
        enough that the genuinely distinguishing words win."""
        item = _benchmark_item()
        item.title = "Basic Set Operations and Venn Diagrams"
        item.required_sections = [
            RequiredSection(
                section_id="introduction", title="Introduction",
                purpose="Motivate Basic Set Operations and Venn Diagrams and outline the structure.",
                target_words=50,
            ),
            RequiredSection(
                section_id="prerequisites_and_notation", title="Prerequisites and Notation",
                purpose="Establish prerequisite background and fix the notation used throughout "
                        "the discussion of Basic Set Operations and Venn Diagrams.",
                target_words=50,
            ),
            RequiredSection(
                section_id="summary", title="Summary",
                purpose="Synthesize the document and restate the central conclusions about "
                        "Basic Set Operations and Venn Diagrams.",
                target_words=50,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            _write_generated_output_with_sections(project_dir, [
                ("sec_introduction", "Motivation for Basic Set Operations and Venn Diagrams", "..."),
                ("sec_prerequisites", "Prerequisites and Notation for Set Operations", "..."),
                ("sec_summary", "Synthesis of Basic Set Operations and Venn Diagrams", "..."),
            ])
            result = evaluate_contract(project_dir, item)
        matched = {d.item_id: d.reason for d in result.details["required_sections"]}
        self.assertIn("sec_introduction", matched["introduction"])
        self.assertIn("sec_prerequisites", matched["prerequisites_and_notation"])
        self.assertIn("sec_summary", matched["summary"])

    def test_two_required_sections_do_not_both_claim_the_same_generated_section(self) -> None:
        item = _benchmark_item()  # required_sections: introduction, conclusion (no required_points)
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            # Only one generated section exists, matchable by title to
            # "Introduction" -- "Conclusion" must not also claim it via the
            # keyword-overlap tier.
            _write_generated_output_with_section(
                project_dir, generated_section_id="sec_intro", generated_title="Introduction",
                block_content="Some content.",
            )
            result = evaluate_contract(project_dir, item)
        statuses = {d.item_id: d.status for d in result.details["required_sections"]}
        self.assertEqual(statuses["introduction"], "satisfied")
        self.assertEqual(statuses["conclusion"], "missing")


if __name__ == "__main__":
    unittest.main()

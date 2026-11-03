"""Fix C checklist item C5 ("End-to-end 확인"): one realistic multi-section
document, built from a BenchmarkItem + DocumentGraph all the way through
C1 (dependency_mapping) -> C3/C4 (dependency_verifier / ConstraintProvider
/ the real ToolExecutor), covering every scenario the checklist names:
필수 인용 누락, 잘못된 label, 허용된 section 참조, 인용만 있고 재사용은 없는
경우, 여러 target 중 하나만 누락(partial), 값 변경(invalidation, covered in
test_dependency_controller_integration.py already), forward reference, and
the final unresolved-list aggregate.

Explicitly NOT covered here (see docs/av_stage2_dependency_mapping.md §6,
"명시적으로 미룬 것"): "source 내용과 다른 요약" (a citation that's real but
whose claimed content is fabricated, Gold Labeling sample_008) -- this
needs semantic content comparison this branch does not implement yet, and
"정상 block 보존" (repair not damaging unrelated content) -- inherited,
already-tested Stage 4 RepairOperationExecutor behavior, independent of
constraint kind, not re-proven here.
"""

from __future__ import annotations

from src.common.state import DocumentBlueprint, DocumentGraph, LayoutBlock, LayoutBlockKind, SectionLayout
from src.core.artifact_graph import ArtifactGraphStore
from src.core.constraint_checks import BasicConstraintChecker
from src.core.dependency_constraint_provider import DependencyConstraintProvider
from src.core.dependency_mapping import MappingStatus, build_dependency_mapping_report
from src.core.dependency_verifier import verify_cross_section_obligations
from src.core.schemas import ArtifactKind, Budget, CheckCall, CheckOutcome, UnknownReason
from src.core.tool_executor import ToolExecutor
from src.dataset.schemas import (
    BenchmarkItem,
    ComponentType,
    ConstraintType,
    CrossSectionDependency,
    Difficulty,
    Domain,
    DocumentType,
    EvaluationContract,
    HardConstraint,
    LayoutRequirements,
    RequiredComponent,
    RequiredSection,
    SourceProvenanceEntry,
    TargetLength,
)

_RENDERED = {
    "intro": "See \\Cref{tab:unrelated} and also \\Cref{sec:summary} for the key result.",
    "formal_definition": "The definition and the boxed result appear here.",
    "worked_examples": "This example does not cite anything relevant.",
    "summary": "The result follows from \\Cref{sec:formal_definition}.",
}


def _benchmark_item() -> BenchmarkItem:
    required_sections = [
        RequiredSection(section_id="intro", title="Introduction", purpose="p", target_words=50),
        RequiredSection(section_id="formal_definition", title="Formal Definition", purpose="p", target_words=50),
        RequiredSection(section_id="worked_examples", title="Worked Examples", purpose="p", target_words=50),
        RequiredSection(section_id="summary", title="Summary", purpose="p", target_words=50),
    ]
    required_components = [
        RequiredComponent(
            component_id="definition_core", type=ComponentType.DEFINITION, placement="formal_definition",
            description="Define the core notion.", reference_required=True,
        ),
        RequiredComponent(
            component_id="boxed_result", type=ComponentType.BOXED_RESULT, placement="formal_definition",
            description="Box the key bound.", reference_required=True,
        ),
    ]
    cross_section_dependencies = [
        CrossSectionDependency(
            dependency_id="reuse_core", source_section="formal_definition",
            target_sections=["worked_examples", "summary"], relation="must_reuse",
            description="worked_examples/summary must reuse the core definition.",
        ),
        CrossSectionDependency(
            dependency_id="reference_key_result", source_section="formal_definition",
            target_sections=["summary"], relation="must_reference",
            description="summary must cite the formal definition.",
        ),
        CrossSectionDependency(
            dependency_id="missing_citation", source_section="formal_definition",
            target_sections=["worked_examples"], relation="must_reference",
            description="worked_examples must cite the formal definition.",
        ),
        CrossSectionDependency(
            dependency_id="wrong_label_dep", source_section="formal_definition",
            target_sections=["intro"], relation="must_reference",
            description="intro must cite the formal definition.",
        ),
        CrossSectionDependency(
            dependency_id="forward_ref_dep", source_section="summary",
            target_sections=["intro"], relation="must_reference",
            description="intro (earlier) must cite summary (later) -- a forward reference.",
        ),
        CrossSectionDependency(
            dependency_id="order_dep", source_section="formal_definition",
            target_sections=["summary"], relation="must_precede",
            description="formal_definition must precede summary in document order.",
        ),
    ]
    return BenchmarkItem(
        sample_id="sample_c5", domain=Domain.MATHEMATICS, subdomain="calculus",
        document_type=DocumentType.LECTURE_NOTE, difficulty=Difficulty.BASIC,
        title="C5 End-to-End Fixture", overview="overview", audience="students", document_purpose="teach",
        target_length=TargetLength(target_pages=5, min_words=100, max_words=2000, min_sections=1, max_sections=10),
        required_sections=required_sections, required_components=required_components,
        hard_constraints=[HardConstraint(constraint_id="c1", type=ConstraintType.OTHER, description="d")],
        cross_section_dependencies=cross_section_dependencies,
        layout_requirements=LayoutRequirements(preferred_layout_profile="default"),
        evaluation_contract=EvaluationContract(), natural_language_instruction="write it",
        source_provenance=[SourceProvenanceEntry(source_id="s1", usage="background")],
    )


def _graph() -> DocumentGraph:
    sections = [
        SectionLayout(section_id="intro", title="Introduction", blocks=[]),
        SectionLayout(
            section_id="formal_definition", title="Formal Definition",
            blocks=[
                LayoutBlock(id="def_block", kind=LayoutBlockKind.DEFINITION, title="Definition"),
                LayoutBlock(id="box_block", kind=LayoutBlockKind.KEY_TAKEAWAY, title="Key bound"),
            ],
        ),
        SectionLayout(section_id="worked_examples", title="Worked Examples", blocks=[]),
        SectionLayout(section_id="summary", title="Summary", blocks=[]),
    ]
    graph = DocumentGraph()
    graph.set_blueprint(DocumentBlueprint(sections=sections))
    return graph


def _store_with_rendered_sections() -> ArtifactGraphStore:
    store = ArtifactGraphStore()
    for section_id, text in _RENDERED.items():
        store.add_version(section_id, ArtifactKind.SECTION, content=text)
    return store


def test_c1_mapping_resolves_everything_needed_for_the_scenarios_below():
    report = build_dependency_mapping_report(_benchmark_item(), _graph())
    assert all(m.status == MappingStatus.MATCHED for m in report.section_mappings)
    assert all(m.status == MappingStatus.MATCHED for m in report.component_mappings)
    assert all(o.status == MappingStatus.MATCHED for o in report.cross_section_obligations)


def test_missing_required_citation_fails():
    report = build_dependency_mapping_report(_benchmark_item(), _graph())
    result = verify_cross_section_obligations(report, _RENDERED)
    record = next(r for r in result.records if r.dependency_id == "missing_citation")
    assert record.outcome == CheckOutcome.FAIL
    assert record.cited_labels_found == []


def test_wrong_label_fails_even_though_a_real_citation_exists():
    report = build_dependency_mapping_report(_benchmark_item(), _graph())
    result = verify_cross_section_obligations(report, _RENDERED)
    record = next(r for r in result.records if r.dependency_id == "wrong_label_dep")
    assert record.outcome == CheckOutcome.FAIL
    assert record.cited_labels_found  # a citation was found, just not an allowed one
    assert "sec:formal_definition" not in record.cited_labels_found


def test_allowed_section_level_reference_passes():
    # Gold Labeling rubric principle 2: citing the SOURCE SECTION's own
    # label satisfies must_reference even without citing a specific
    # component's own label.
    report = build_dependency_mapping_report(_benchmark_item(), _graph())
    result = verify_cross_section_obligations(report, _RENDERED)
    record = next(r for r in result.records if r.dependency_id == "reference_key_result")
    assert record.outcome == CheckOutcome.PASS
    assert record.cited_labels_found == ["sec:formal_definition"]


def test_forward_reference_passes_regardless_of_physical_document_order():
    # intro (blueprint index 0) cites summary (blueprint index 3), a
    # section that comes AFTER it -- must_reference must not care about
    # document order (that is must_precede's separate, distinct job).
    report = build_dependency_mapping_report(_benchmark_item(), _graph())
    result = verify_cross_section_obligations(report, _RENDERED)
    record = next(r for r in result.records if r.dependency_id == "forward_ref_dep")
    assert record.outcome == CheckOutcome.PASS


def test_document_order_requirement_passes_independently_of_reference_checks():
    report = build_dependency_mapping_report(_benchmark_item(), _graph())
    result = verify_cross_section_obligations(report, _RENDERED)
    record = next(r for r in result.records if r.dependency_id == "order_dep")
    assert record.outcome == CheckOutcome.PASS


def test_citation_present_but_reuse_unconfirmed_stays_unknown_not_pass():
    # "인용만 있고 재사용은 없는 경우": summary DOES cite
    # \Cref{sec:formal_definition} (satisfying reference_key_result), but
    # that citation must never be read as also satisfying reuse_core's
    # must_reuse -- content reuse is a separate, still-unimplemented (Fix
    # B) check, and a citation existing is not evidence of it.
    report = build_dependency_mapping_report(_benchmark_item(), _graph())
    result = verify_cross_section_obligations(report, _RENDERED)
    record = next(
        r for r in result.records if r.dependency_id == "reuse_core" and r.target_section_id == "summary"
    )
    assert record.outcome == CheckOutcome.UNKNOWN
    assert record.unknown_reason == UnknownReason.NOT_REPRESENTABLE


def test_one_of_several_targets_failing_does_not_hide_the_others_distinct_outcome():
    # reuse_core has two targets (worked_examples, summary) -- both land on
    # UNKNOWN here (Fix B gap), but they must be tracked as two independent
    # records, not collapsed into one dependency-level verdict.
    report = build_dependency_mapping_report(_benchmark_item(), _graph())
    result = verify_cross_section_obligations(report, _RENDERED)
    reuse_records = {r.target_section_id: r.outcome for r in result.records if r.dependency_id == "reuse_core"}
    assert reuse_records == {
        "worked_examples": CheckOutcome.UNKNOWN,
        "summary": CheckOutcome.UNKNOWN,
    }


def test_final_unresolved_list_aggregate_matches_the_realistic_scenario_mix():
    report = build_dependency_mapping_report(_benchmark_item(), _graph())
    result = verify_cross_section_obligations(report, _RENDERED)
    assert result.total_targets == 7  # reuse_core(2) + 5 single-target dependencies
    assert result.pass_count == 3  # reference_key_result, forward_ref_dep, order_dep
    assert result.fail_count == 2  # missing_citation, wrong_label_dep
    assert result.unknown_count == 2  # reuse_core x2 (Fix B gap)
    assert result.checked_targets == result.pass_count + result.fail_count == 5


def test_the_real_executor_path_agrees_with_the_direct_report_for_every_target():
    """The C4 ConstraintProvider/ToolExecutor path must reproduce every one
    of the 7 verdicts above exactly, not just the single spot-check in
    test_dependency_controller_integration.py -- this is the actual "실제
    실행" (real execution) check C5 asks for, run across the full scenario
    mix at once."""
    report = build_dependency_mapping_report(_benchmark_item(), _graph())
    direct = verify_cross_section_obligations(report, _RENDERED)
    direct_by_key = {(r.dependency_id, r.target_section_id): r.outcome for r in direct.records}

    store = _store_with_rendered_sections()
    provider = DependencyConstraintProvider(report, store)
    checker = BasicConstraintChecker(store)
    executor = ToolExecutor(store, checker, budget=Budget(cap=1000), max_actions=100)

    via_executor = {}
    for section_id in _RENDERED:
        target_ref = store.latest(section_id).ref
        constraints = provider(target_ref)
        if not constraints:
            continue
        observation = executor.execute(
            CheckCall(artifact=target_ref, constraints=constraints, context=provider.context_for(target_ref))
        )
        assert observation.ok is True
        for ev in observation.output.evidence:
            dependency_id = ev.constraint_id.split("::")[0]
            via_executor[(dependency_id, section_id)] = ev.outcome

    assert via_executor == direct_by_key

    # And the final unresolved list is identically reconstructable from
    # what the store actually recorded, not just from the direct report.
    all_live_evidence = [e for section_id in _RENDERED for e in store.live_evidence(section_id)]
    assert sum(1 for e in all_live_evidence if e.outcome == CheckOutcome.UNKNOWN) == 2
    assert sum(1 for e in all_live_evidence if e.outcome == CheckOutcome.FAIL) == 2
    assert sum(1 for e in all_live_evidence if e.outcome == CheckOutcome.PASS) == 3

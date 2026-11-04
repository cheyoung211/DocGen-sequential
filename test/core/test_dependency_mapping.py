"""Coverage for src/core/dependency_mapping.py (Revision "Additional
verifiers" plan, Fix C checklist item C1) -- see
docs/av_stage2_dependency_mapping.md for the design this implements and the
Gold-Labeling-derived rubric principles each test traces back to."""

from __future__ import annotations

from typing import List

import pytest

from src.common.state import (
    DocumentBlueprint,
    DocumentGraph,
    LayoutBlock,
    LayoutBlockKind,
    SectionLayout,
)
from src.core.dependency_mapping import (
    MappingStatus,
    build_dependency_mapping_report,
    build_section_mappings,
)
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


def _benchmark_item(
    *,
    required_sections: List[RequiredSection],
    required_components: List[RequiredComponent],
    cross_section_dependencies: List[CrossSectionDependency],
) -> BenchmarkItem:
    return BenchmarkItem(
        sample_id="sample_test",
        domain=Domain.MATHEMATICS,
        subdomain="calculus",
        document_type=DocumentType.LECTURE_NOTE,
        difficulty=Difficulty.BASIC,
        title="Taylor Polynomials",
        overview="overview",
        audience="students",
        document_purpose="teach",
        target_length=TargetLength(target_pages=5, min_words=100, max_words=2000, min_sections=1, max_sections=10),
        required_sections=required_sections,
        required_components=required_components,
        hard_constraints=[
            HardConstraint(constraint_id="c1", type=ConstraintType.OTHER, description="d")
        ],
        cross_section_dependencies=cross_section_dependencies,
        layout_requirements=LayoutRequirements(preferred_layout_profile="default"),
        evaluation_contract=EvaluationContract(),
        natural_language_instruction="write it",
        source_provenance=[SourceProvenanceEntry(source_id="s1", usage="background")],
    )


def _graph(sections: List[SectionLayout]) -> DocumentGraph:
    graph = DocumentGraph()
    graph.set_blueprint(DocumentBlueprint(sections=sections))
    return graph


def _basic_fixture():
    """formal_definition (has a labeled DEFINITION block + an unlabeled
    boxed_result-shaped block) -> worked_examples, summary -- mirrors
    sample_000's reuse_core_definition/theorem_core shape closely enough to
    exercise direct-label, section-label, and no-label obligations at once."""
    required_sections = [
        RequiredSection(section_id="formal_definition", title="Formal Definition", purpose="p", target_words=100),
        RequiredSection(section_id="worked_examples", title="Worked Examples", purpose="p", target_words=100),
        RequiredSection(section_id="summary", title="Summary", purpose="p", target_words=100),
    ]
    required_components = [
        RequiredComponent(
            component_id="definition_core", type=ComponentType.DEFINITION, placement="formal_definition",
            description="Define the Taylor polynomial.", reference_required=True,
        ),
        RequiredComponent(
            component_id="boxed_result", type=ComponentType.BOXED_RESULT, placement="formal_definition",
            description="Box the key bound.", reference_required=True,
        ),
        RequiredComponent(
            component_id="worked_example_1", type=ComponentType.WORKED_EXAMPLE, placement="worked_examples",
            description="Work an example.", depends_on=["definition_core"],
        ),
    ]
    cross_section_dependencies = [
        CrossSectionDependency(
            dependency_id="reuse_core_definition", source_section="formal_definition",
            target_sections=["worked_examples", "summary"], relation="must_reuse", description="reuse",
        ),
        CrossSectionDependency(
            dependency_id="reference_key_result", source_section="formal_definition",
            target_sections=["summary"], relation="must_reference", description="reference",
        ),
    ]
    benchmark_item = _benchmark_item(
        required_sections=required_sections,
        required_components=required_components,
        cross_section_dependencies=cross_section_dependencies,
    )
    sections = [
        SectionLayout(
            section_id="formal_definition", title="Formal Definition",
            blocks=[
                LayoutBlock(id="def_block", kind=LayoutBlockKind.DEFINITION, title="Definition"),
                LayoutBlock(id="box_block", kind=LayoutBlockKind.KEY_TAKEAWAY, title="Key bound"),
            ],
        ),
        SectionLayout(
            section_id="worked_examples", title="Worked Examples",
            blocks=[LayoutBlock(id="we_block", kind=LayoutBlockKind.CASE_STUDY, title="Example")],
        ),
        SectionLayout(section_id="summary", title="Summary", blocks=[]),
    ]
    graph = _graph(sections)
    return benchmark_item, graph


def test_section_mapping_matches_by_section_id():
    benchmark_item, graph = _basic_fixture()
    mappings = build_section_mappings(benchmark_item, graph)
    assert {m.section_id: m.matched_section_id for m in mappings} == {
        "formal_definition": "formal_definition",
        "worked_examples": "worked_examples",
        "summary": "summary",
    }
    assert all(m.status == MappingStatus.MATCHED for m in mappings)


def test_component_mapping_matches_kind_only_and_assigns_default_label():
    benchmark_item, graph = _basic_fixture()
    report = build_dependency_mapping_report(benchmark_item, graph)
    by_id = {c.component_id: c for c in report.component_mappings}

    assert by_id["definition_core"].status == MappingStatus.MATCHED
    assert by_id["definition_core"].matched_block_id == "def_block"
    assert by_id["definition_core"].matched_label == "def:def_block"

    # boxed_result -> KEY_TAKEAWAY has no entry in _LABEL_PREFIX_BY_KIND, so
    # it matches a real block but gets no citable label -- exactly the case
    # Gold Labeling's "adjacent label" principle (3) is built for.
    assert by_id["boxed_result"].status == MappingStatus.MATCHED
    assert by_id["boxed_result"].matched_block_id == "box_block"
    assert by_id["boxed_result"].matched_label is None


def test_component_mapping_reports_not_representable_distinct_from_unmatched():
    required_sections = [RequiredSection(section_id="s1", title="S1", purpose="p", target_words=50)]
    required_components = [
        RequiredComponent(component_id="c1", type=ComponentType.EXERCISE, placement="s1", description="d")
    ]
    benchmark_item = _benchmark_item(
        required_sections=required_sections, required_components=required_components,
        cross_section_dependencies=[
            CrossSectionDependency(dependency_id="d1", source_section="s1", target_sections=["s1"], relation="must_precede", description="x")
        ],
    )
    graph = _graph([SectionLayout(section_id="s1", title="S1", blocks=[])])
    report = build_dependency_mapping_report(benchmark_item, graph)
    assert report.component_mappings[0].status == MappingStatus.NOT_REPRESENTABLE


def test_component_mapping_unmatched_when_no_candidate_block():
    required_sections = [RequiredSection(section_id="s1", title="S1", purpose="p", target_words=50)]
    required_components = [
        RequiredComponent(component_id="c1", type=ComponentType.THEOREM, placement="s1", description="d")
    ]
    benchmark_item = _benchmark_item(
        required_sections=required_sections, required_components=required_components,
        cross_section_dependencies=[
            CrossSectionDependency(dependency_id="d1", source_section="s1", target_sections=["s1"], relation="must_precede", description="x")
        ],
    )
    graph = _graph([SectionLayout(section_id="s1", title="S1", blocks=[])])
    report = build_dependency_mapping_report(benchmark_item, graph)
    assert report.component_mappings[0].status == MappingStatus.UNMATCHED


def test_component_mapping_is_ambiguous_on_an_unbroken_kind_tie():
    required_sections = [RequiredSection(section_id="s1", title="S1", purpose="p", target_words=50)]
    required_components = [
        RequiredComponent(component_id="c1", type=ComponentType.TABLE, placement="s1", description="totally generic")
    ]
    benchmark_item = _benchmark_item(
        required_sections=required_sections, required_components=required_components,
        cross_section_dependencies=[
            CrossSectionDependency(dependency_id="d1", source_section="s1", target_sections=["s1"], relation="must_precede", description="x")
        ],
    )
    graph = _graph([
        SectionLayout(
            section_id="s1", title="S1",
            blocks=[
                LayoutBlock(id="t1", kind=LayoutBlockKind.TABLE, title="unrelated alpha"),
                LayoutBlock(id="t2", kind=LayoutBlockKind.TABLE, title="unrelated beta"),
            ],
        )
    ])
    report = build_dependency_mapping_report(benchmark_item, graph)
    mapping = report.component_mappings[0]
    assert mapping.status == MappingStatus.AMBIGUOUS
    assert "t1" in mapping.reason and "t2" in mapping.reason


def test_component_mapping_breaks_kind_tie_via_keyword_overlap():
    required_sections = [RequiredSection(section_id="s1", title="S1", purpose="p", target_words=50)]
    required_components = [
        RequiredComponent(component_id="c1", type=ComponentType.TABLE, placement="s1", description="notation symbols table")
    ]
    benchmark_item = _benchmark_item(
        required_sections=required_sections, required_components=required_components,
        cross_section_dependencies=[
            CrossSectionDependency(dependency_id="d1", source_section="s1", target_sections=["s1"], relation="must_precede", description="x")
        ],
    )
    graph = _graph([
        SectionLayout(
            section_id="s1", title="S1",
            blocks=[
                LayoutBlock(id="t_notation", kind=LayoutBlockKind.TABLE, title="notation symbols reference"),
                LayoutBlock(id="t_unrelated", kind=LayoutBlockKind.TABLE, title="unrelated comparison"),
            ],
        )
    ])
    report = build_dependency_mapping_report(benchmark_item, graph)
    mapping = report.component_mappings[0]
    assert mapping.status == MappingStatus.MATCHED
    assert mapping.matched_block_id == "t_notation"
    assert mapping.matched_via == "kind_and_keyword"


def test_cross_section_obligation_allowed_labels_combine_section_and_direct_label():
    benchmark_item, graph = _basic_fixture()
    report = build_dependency_mapping_report(benchmark_item, graph)
    obligations = {
        (o.dependency_id, o.target_section_id): o for o in report.cross_section_obligations
    }
    reuse_to_summary = obligations[("reuse_core_definition", "summary")]
    assert reuse_to_summary.status == MappingStatus.MATCHED
    assert "sec:formal_definition" in reuse_to_summary.allowed_labels
    assert "def:def_block" in reuse_to_summary.allowed_labels
    # boxed_result has no citable label -- its presence in the source section
    # must set adjacent_label_possible rather than silently omitting it.
    assert reuse_to_summary.adjacent_label_possible is True
    assert reuse_to_summary.content_check_required is True  # must_reuse


def test_cross_section_obligation_must_reference_does_not_require_content_check():
    benchmark_item, graph = _basic_fixture()
    report = build_dependency_mapping_report(benchmark_item, graph)
    obligation = next(
        o for o in report.cross_section_obligations if o.dependency_id == "reference_key_result"
    )
    assert obligation.relation.value == "must_reference"
    assert obligation.content_check_required is False


def test_cross_section_obligation_creates_one_record_per_target_for_partial_failure():
    benchmark_item, graph = _basic_fixture()
    report = build_dependency_mapping_report(benchmark_item, graph)
    reuse_targets = {
        o.target_section_id for o in report.cross_section_obligations if o.dependency_id == "reuse_core_definition"
    }
    assert reuse_targets == {"worked_examples", "summary"}


def test_cross_section_obligation_order_hint_reflects_blueprint_order():
    benchmark_item, graph = _basic_fixture()
    report = build_dependency_mapping_report(benchmark_item, graph)
    obligation = next(
        o for o in report.cross_section_obligations
        if o.dependency_id == "reuse_core_definition" and o.target_section_id == "summary"
    )
    assert obligation.order_hint == (0, 2)  # formal_definition=0, summary=2 in blueprint.sections


def test_cross_section_obligation_propagates_unmatched_target_without_guessing():
    required_sections = [RequiredSection(section_id="s1", title="S1", purpose="p", target_words=50)]
    cross_section_dependencies = [
        CrossSectionDependency(
            dependency_id="d1", source_section="s1", target_sections=["ghost_target"],
            relation="must_reference", description="x",
        )
    ]
    benchmark_item = _benchmark_item(
        required_sections=required_sections,
        required_components=[RequiredComponent(component_id="c1", type=ComponentType.DEFINITION, placement="s1", description="d")],
        cross_section_dependencies=cross_section_dependencies,
    )
    graph = _graph([SectionLayout(section_id="s1", title="S1", blocks=[LayoutBlock(id="b1", kind=LayoutBlockKind.DEFINITION)])])
    report = build_dependency_mapping_report(benchmark_item, graph)
    obligation = report.cross_section_obligations[0]
    assert obligation.status == MappingStatus.UNMATCHED
    assert obligation.matched_source_section_id == "s1"  # source still reported even though target failed
    assert obligation.matched_target_section_id is None
    assert obligation.allowed_labels == []


def test_component_dependency_obligation_matched_and_unmatched():
    benchmark_item, graph = _basic_fixture()
    report = build_dependency_mapping_report(benchmark_item, graph)
    obligation = report.component_dependency_obligations[0]
    assert obligation.component_id == "worked_example_1"
    assert obligation.depends_on_component_id == "definition_core"
    assert obligation.status == MappingStatus.MATCHED
    assert obligation.matched_component_block_id == "we_block"
    assert obligation.matched_depends_on_block_id == "def_block"
    assert obligation.content_check_required is True


def test_duplicate_section_id_is_reported_and_excluded_from_matching():
    required_sections = [
        RequiredSection(section_id="s1", title="First", purpose="p", target_words=50),
        RequiredSection(section_id="s1", title="Duplicate", purpose="p", target_words=50),
    ]
    benchmark_item = _benchmark_item(
        required_sections=required_sections,
        required_components=[RequiredComponent(component_id="c1", type=ComponentType.DEFINITION, placement="s1", description="d")],
        cross_section_dependencies=[
            CrossSectionDependency(dependency_id="d1", source_section="s1", target_sections=["s1"], relation="must_precede", description="x")
        ],
    )
    graph = _graph([SectionLayout(section_id="s1", title="First", blocks=[])])
    report = build_dependency_mapping_report(benchmark_item, graph)
    assert any("s1" in issue and "declared 2 times" in issue for issue in report.data_quality_issues)
    assert all(m.status == MappingStatus.DUPLICATE_ID for m in report.section_mappings)
    # the component whose placement points at the duplicated id must not be
    # silently matched against whichever generated section happens to exist.
    assert report.component_mappings[0].status == MappingStatus.UNMATCHED


def test_duplicate_component_id_is_reported_and_excluded_from_matching():
    required_sections = [RequiredSection(section_id="s1", title="S1", purpose="p", target_words=50)]
    required_components = [
        RequiredComponent(component_id="c1", type=ComponentType.DEFINITION, placement="s1", description="d"),
        RequiredComponent(component_id="c1", type=ComponentType.THEOREM, placement="s1", description="d2"),
    ]
    benchmark_item = _benchmark_item(
        required_sections=required_sections, required_components=required_components,
        cross_section_dependencies=[
            CrossSectionDependency(dependency_id="d1", source_section="s1", target_sections=["s1"], relation="must_precede", description="x")
        ],
    )
    graph = _graph([SectionLayout(section_id="s1", title="S1", blocks=[LayoutBlock(id="b1", kind=LayoutBlockKind.DEFINITION)])])
    report = build_dependency_mapping_report(benchmark_item, graph)
    assert any("c1" in issue and "declared 2 times" in issue for issue in report.data_quality_issues)
    assert all(m.status == MappingStatus.DUPLICATE_ID for m in report.component_mappings)


def test_duplicate_component_id_still_reports_its_depends_on_edges():
    required_sections = [RequiredSection(section_id="s1", title="S1", purpose="p", target_words=50)]
    required_components = [
        RequiredComponent(component_id="base", type=ComponentType.DEFINITION, placement="s1", description="d"),
        RequiredComponent(component_id="c1", type=ComponentType.THEOREM, placement="s1", description="d", depends_on=["base"]),
        RequiredComponent(component_id="c1", type=ComponentType.TABLE, placement="s1", description="d2", depends_on=["base"]),
    ]
    benchmark_item = _benchmark_item(
        required_sections=required_sections, required_components=required_components,
        cross_section_dependencies=[
            CrossSectionDependency(dependency_id="d1", source_section="s1", target_sections=["s1"], relation="must_precede", description="x")
        ],
    )
    graph = _graph([SectionLayout(section_id="s1", title="S1", blocks=[LayoutBlock(id="b1", kind=LayoutBlockKind.DEFINITION)])])
    report = build_dependency_mapping_report(benchmark_item, graph)
    # c1 is duplicated, but it declared a depends_on=["base"] edge each time --
    # that declared requirement must still surface, not vanish silently.
    c1_obligations = [o for o in report.component_dependency_obligations if o.component_id == "c1"]
    assert len(c1_obligations) == 2
    assert all(o.status == MappingStatus.DUPLICATE_ID for o in c1_obligations)
    assert all(o.depends_on_component_id == "base" for o in c1_obligations)


def test_reference_required_is_preserved_on_component_mapping():
    """A review found that flipping ``reference_required`` True->False left
    the report byte-for-byte identical -- the obligation itself (not just
    whether C3/B have gotten around to checking it) must survive mapping."""
    def _report_for(reference_required: bool):
        required_sections = [RequiredSection(section_id="s1", title="S1", purpose="p", target_words=50)]
        required_components = [
            RequiredComponent(
                component_id="c1", type=ComponentType.DEFINITION, placement="s1",
                description="d", reference_required=reference_required,
            )
        ]
        benchmark_item = _benchmark_item(
            required_sections=required_sections, required_components=required_components,
            cross_section_dependencies=[
                CrossSectionDependency(dependency_id="d1", source_section="s1", target_sections=["s1"], relation="must_precede", description="x")
            ],
        )
        graph = _graph([SectionLayout(section_id="s1", title="S1", blocks=[LayoutBlock(id="b1", kind=LayoutBlockKind.DEFINITION)])])
        return build_dependency_mapping_report(benchmark_item, graph)

    report_true = _report_for(True)
    report_false = _report_for(False)
    assert report_true.component_mappings[0].reference_required is True
    assert report_false.component_mappings[0].reference_required is False
    assert report_true.model_dump() != report_false.model_dump()


def test_reference_required_is_preserved_even_when_component_is_unmatched():
    required_sections = [RequiredSection(section_id="s1", title="S1", purpose="p", target_words=50)]
    required_components = [
        RequiredComponent(
            component_id="c1", type=ComponentType.THEOREM, placement="s1",
            description="d", reference_required=True,
        )
    ]
    benchmark_item = _benchmark_item(
        required_sections=required_sections, required_components=required_components,
        cross_section_dependencies=[
            CrossSectionDependency(dependency_id="d1", source_section="s1", target_sections=["s1"], relation="must_precede", description="x")
        ],
    )
    graph = _graph([SectionLayout(section_id="s1", title="S1", blocks=[])])  # no candidate block -> UNMATCHED
    report = build_dependency_mapping_report(benchmark_item, graph)
    mapping = report.component_mappings[0]
    assert mapping.status == MappingStatus.UNMATCHED
    assert mapping.reference_required is True


def test_section_mapping_reports_ambiguous_on_title_tie_regardless_of_list_order():
    """Two unclaimed sections with the same normalized title must both be
    flagged AMBIGUOUS, not silently resolved to whichever happens to come
    first in ``blueprint.sections`` -- a review found that reversing the
    two sections' order changed which one a required section matched
    while both runs still reported MATCHED."""
    req = RequiredSection(section_id="req_sec", title="Worked Examples", purpose="p", target_words=50)
    benchmark_item = _benchmark_item(
        required_sections=[req],
        required_components=[RequiredComponent(component_id="c1", type=ComponentType.DEFINITION, placement="req_sec", description="d")],
        cross_section_dependencies=[
            CrossSectionDependency(dependency_id="d1", source_section="req_sec", target_sections=["req_sec"], relation="must_precede", description="x")
        ],
    )
    forward = [
        SectionLayout(section_id="secA", title="Worked Examples", blocks=[]),
        SectionLayout(section_id="secB", title="worked   examples", blocks=[]),
    ]
    for sections in (forward, list(reversed(forward))):
        graph = _graph(sections)
        mapping = build_section_mappings(benchmark_item, graph)[0]
        assert mapping.status == MappingStatus.AMBIGUOUS
        assert "secA" in mapping.reason and "secB" in mapping.reason
        assert mapping.matched_section_id is None


def test_section_mapping_reports_ambiguous_on_keyword_overlap_tie():
    req = RequiredSection(section_id="req_sec", title="Alpha Beta", purpose="", target_words=50)
    benchmark_item = _benchmark_item(
        required_sections=[req],
        required_components=[RequiredComponent(component_id="c1", type=ComponentType.DEFINITION, placement="req_sec", description="d")],
        cross_section_dependencies=[
            CrossSectionDependency(dependency_id="d1", source_section="req_sec", target_sections=["req_sec"], relation="must_precede", description="x")
        ],
    )
    graph = _graph([
        SectionLayout(section_id="secA", title="Gamma Alpha", blocks=[]),
        SectionLayout(section_id="secB", title="Alpha Gamma", blocks=[]),
    ])
    mapping = build_section_mappings(benchmark_item, graph)[0]
    assert mapping.status == MappingStatus.AMBIGUOUS
    assert "secA" in mapping.reason and "secB" in mapping.reason


def test_section_mapping_exact_id_match_is_never_ambiguous():
    """``matched_via == "section_id"`` names exactly one node by
    construction -- the tie-check must not second-guess it even when other,
    unrelated sections happen to share a title with each other."""
    req = RequiredSection(section_id="s1", title="Whatever", purpose="p", target_words=50)
    benchmark_item = _benchmark_item(
        required_sections=[req],
        required_components=[RequiredComponent(component_id="c1", type=ComponentType.DEFINITION, placement="s1", description="d")],
        cross_section_dependencies=[
            CrossSectionDependency(dependency_id="d1", source_section="s1", target_sections=["s1"], relation="must_precede", description="x")
        ],
    )
    graph = _graph([
        SectionLayout(section_id="s1", title="Same Title", blocks=[]),
        SectionLayout(section_id="s2", title="Same Title", blocks=[]),
    ])
    mapping = build_section_mappings(benchmark_item, graph)[0]
    assert mapping.status == MappingStatus.MATCHED
    assert mapping.matched_section_id == "s1"


def test_duplicate_dependency_id_is_reported_for_every_target():
    required_sections = [
        RequiredSection(section_id="s1", title="S1", purpose="p", target_words=50),
        RequiredSection(section_id="s2", title="S2", purpose="p", target_words=50),
    ]
    cross_section_dependencies = [
        CrossSectionDependency(dependency_id="d1", source_section="s1", target_sections=["s2"], relation="must_reference", description="a"),
        CrossSectionDependency(dependency_id="d1", source_section="s1", target_sections=["s2"], relation="must_reuse", description="b"),
    ]
    benchmark_item = _benchmark_item(
        required_sections=required_sections,
        required_components=[RequiredComponent(component_id="c1", type=ComponentType.DEFINITION, placement="s1", description="d")],
        cross_section_dependencies=cross_section_dependencies,
    )
    graph = _graph([
        SectionLayout(section_id="s1", title="S1", blocks=[]),
        SectionLayout(section_id="s2", title="S2", blocks=[]),
    ])
    report = build_dependency_mapping_report(benchmark_item, graph)
    assert any("d1" in issue and "declared 2 times" in issue for issue in report.data_quality_issues)
    assert all(o.status == MappingStatus.DUPLICATE_ID for o in report.cross_section_obligations)
    assert len(report.cross_section_obligations) == 2  # one per declared dependency, not deduplicated away

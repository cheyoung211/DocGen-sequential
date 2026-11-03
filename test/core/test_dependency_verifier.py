"""Coverage for src/core/dependency_verifier.py (Revision "Additional
verifiers" plan, Fix C checklist item C3's first bullet: required-reference
and document-order checks) -- see docs/av_stage2_dependency_mapping.md for
the design and the rubric principles each test traces back to."""

from __future__ import annotations

from src.core.dependency_mapping import CrossSectionObligation, DependencyMappingReport, MappingStatus
from src.core.dependency_verifier import verify_cross_section_obligations
from src.core.schemas import CheckOutcome, UnknownReason
from src.dataset.schemas import DependencyRelation


def _report(obligations):
    return DependencyMappingReport(
        section_mappings=[], component_mappings=[],
        cross_section_obligations=obligations, component_dependency_obligations=[],
    )


def test_must_reference_passes_on_a_real_citation_command():
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_REFERENCE, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
        allowed_labels=["sec:intro", "def:intro_def"],
    )
    rendered = {"summary": "As shown earlier, \\Cref{def:intro_def} gives the key bound."}
    result = verify_cross_section_obligations(_report([obligation]), rendered)
    assert result.records[0].outcome == CheckOutcome.PASS
    assert result.records[0].cited_labels_found == ["def:intro_def"]
    assert result.records[0].matched_at_line == 1
    assert result.pass_count == 1 and result.fail_count == 0 and result.unknown_count == 0


def test_must_reference_fails_on_prose_mention_without_a_real_command():
    # Gold Labeling sample_005: "the analysis section" in prose, no \Cref.
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_REFERENCE, description="x",
        source_section_id="analysis", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="analysis", matched_target_section_id="summary",
        allowed_labels=["sec:analysis"],
    )
    rendered = {"summary": "This follows the analysis section's reasoning."}
    result = verify_cross_section_obligations(_report([obligation]), rendered)
    assert result.records[0].outcome == CheckOutcome.FAIL
    assert result.fail_count == 1


def test_must_reference_fails_when_citing_an_unrelated_label():
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_REFERENCE, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
        allowed_labels=["sec:intro"],
    )
    rendered = {"summary": "See \\Cref{tab:unrelated} for details."}
    result = verify_cross_section_obligations(_report([obligation]), rendered)
    assert result.records[0].outcome == CheckOutcome.FAIL
    assert result.records[0].cited_labels_found == ["tab:unrelated"]


def test_must_reference_handles_a_multi_label_command():
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_REFERENCE, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
        allowed_labels=["eq:b"],
    )
    rendered = {"summary": "By \\Cref{eq:a,eq:b} the result follows."}
    result = verify_cross_section_obligations(_report([obligation]), rendered)
    assert result.records[0].outcome == CheckOutcome.PASS
    assert result.records[0].cited_labels_found == ["eq:a", "eq:b"]


def test_must_precede_passes_when_order_hint_is_increasing():
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_PRECEDE, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
        order_hint=(0, 3),
    )
    result = verify_cross_section_obligations(_report([obligation]), {})
    assert result.records[0].outcome == CheckOutcome.PASS


def test_must_precede_fails_when_target_does_not_come_after_source():
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_PRECEDE, description="x",
        source_section_id="summary", target_section_id="intro", status=MappingStatus.MATCHED,
        matched_source_section_id="summary", matched_target_section_id="intro",
        order_hint=(3, 0),
    )
    result = verify_cross_section_obligations(_report([obligation]), {})
    assert result.records[0].outcome == CheckOutcome.FAIL


def test_must_precede_is_unknown_without_an_order_hint():
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_PRECEDE, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
        order_hint=None,
    )
    result = verify_cross_section_obligations(_report([obligation]), {})
    assert result.records[0].outcome == CheckOutcome.UNKNOWN
    assert result.records[0].unknown_reason == UnknownReason.NOT_CHECKED


def test_unmatched_obligation_is_unknown_not_silently_dropped_or_passed():
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_REFERENCE, description="x",
        source_section_id="intro", target_section_id="ghost", status=MappingStatus.UNMATCHED,
        reason="target_section 'ghost' did not resolve.",
    )
    result = verify_cross_section_obligations(_report([obligation]), {})
    assert result.total_targets == 1
    assert result.records[0].outcome == CheckOutcome.UNKNOWN
    assert result.records[0].unknown_reason == UnknownReason.MISSING_DEPENDENCY


def test_must_reuse_and_must_be_consistent_with_are_unknown_not_passed():
    # Fix B's registry doesn't exist yet on this branch -- these must never
    # silently read as PASS just because no checker is wired up yet.
    reuse = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_REUSE, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
    )
    consistent = CrossSectionObligation(
        dependency_id="d2", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
    )
    result = verify_cross_section_obligations(_report([reuse, consistent]), {"summary": "anything"})
    assert all(r.outcome == CheckOutcome.UNKNOWN for r in result.records)
    assert all(r.unknown_reason == UnknownReason.NOT_REPRESENTABLE for r in result.records)


def test_partial_target_success_does_not_hide_the_other_targets_failure():
    # Gold Labeling sample_002/006/008: one target of a multi-target
    # dependency can pass while another fails -- aggregate counts must
    # reflect both, never collapse to one verdict for the dependency_id.
    passing = CrossSectionObligation(
        dependency_id="reuse_core", relation=DependencyRelation.MUST_REFERENCE, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
        allowed_labels=["sec:intro"],
    )
    failing = CrossSectionObligation(
        dependency_id="reuse_core", relation=DependencyRelation.MUST_REFERENCE, description="x",
        source_section_id="intro", target_section_id="worked_examples", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="worked_examples",
        allowed_labels=["sec:intro"],
    )
    rendered = {"summary": "\\Cref{sec:intro}", "worked_examples": "no citation here"}
    result = verify_cross_section_obligations(_report([passing, failing]), rendered)
    assert result.total_targets == 2
    assert result.pass_count == 1
    assert result.fail_count == 1


def test_coverage_counts_separate_pass_fail_unknown_and_checked_vs_total():
    obligations = [
        CrossSectionObligation(
            dependency_id="d1", relation=DependencyRelation.MUST_REFERENCE, description="x",
            source_section_id="a", target_section_id="b", status=MappingStatus.MATCHED,
            matched_source_section_id="a", matched_target_section_id="b", allowed_labels=["lbl"],
        ),
        CrossSectionObligation(
            dependency_id="d2", relation=DependencyRelation.MUST_REUSE, description="x",
            source_section_id="a", target_section_id="b", status=MappingStatus.MATCHED,
            matched_source_section_id="a", matched_target_section_id="b",
        ),
        CrossSectionObligation(
            dependency_id="d3", relation=DependencyRelation.MUST_REFERENCE, description="x",
            source_section_id="a", target_section_id="ghost", status=MappingStatus.UNMATCHED,
        ),
    ]
    result = verify_cross_section_obligations(_report(obligations), {"b": "\\Cref{lbl}"})
    assert result.total_targets == 3
    assert result.checked_targets == 1  # only the PASS was actually checked
    assert result.pass_count == 1
    assert result.unknown_count == 2

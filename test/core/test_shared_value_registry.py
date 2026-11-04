"""Coverage for src/core/shared_value_registry.py (Revision "Additional
verifiers" plan, Fix B checklist item B3: "공유 값·계수 일관성 검사")."""

from __future__ import annotations

from src.core.schemas import CheckOutcome, UnknownReason
from src.core.shared_value_registry import (
    SharedValueCheckConfig,
    check_shared_value_consistency,
    find_symbol_value_occurrences,
)
from src.dataset.schemas import DependencyRelation


def test_finds_a_simple_symbol_value_occurrence():
    occurrences = find_symbol_value_occurrences("We fix $n = 10$ for this document.", "n")
    assert len(occurrences) == 1
    assert occurrences[0].value == 10.0
    assert occurrences[0].line == 1


def test_does_not_match_a_symbol_that_is_a_prefix_of_a_longer_identifier():
    occurrences = find_symbol_value_occurrences("n_0 = 5 is the baseline.", "n")
    assert occurrences == []


def test_matches_a_subscripted_symbol_exactly():
    occurrences = find_symbol_value_occurrences("n_0 = 5 is the baseline.", "n_0")
    assert len(occurrences) == 1
    assert occurrences[0].value == 5.0


def test_matches_a_backslash_command_symbol():
    occurrences = find_symbol_value_occurrences("Let $\\alpha = 0.05$ be the significance level.", "\\alpha")
    assert len(occurrences) == 1
    assert occurrences[0].value == 0.05


def test_backslash_symbol_does_not_match_a_longer_command():
    occurrences = find_symbol_value_occurrences("\\alphabet = 1", "\\alpha")
    assert occurrences == []


def test_finds_multiple_occurrences_with_correct_line_numbers():
    text = "line one\nn = 1\nsomething else\nn = 2"
    occurrences = find_symbol_value_occurrences(text, "n")
    assert [occ.line for occ in occurrences] == [2, 4]
    assert [occ.value for occ in occurrences] == [1.0, 2.0]


def _config(**kwargs) -> SharedValueCheckConfig:
    return SharedValueCheckConfig(**kwargs)


def test_consistency_check_is_unknown_with_no_relevant_symbols():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 1", target_text="n = 1", config=_config(),
    )
    assert record.outcome == CheckOutcome.UNKNOWN
    assert record.unknown_reason == UnknownReason.NOT_REPRESENTABLE


def test_consistency_check_passes_when_restated_value_matches():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10", target_text="using n = 10 here",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.PASS
    assert record.checked_semantic_ids == ["n"]


def test_consistency_check_fails_on_a_restated_mismatch():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="k = 25", target_text="k = 10",
        config=_config(relevant_symbols=["k"]),
    )
    assert record.outcome == CheckOutcome.FAIL
    assert len(record.mismatches) == 1


def test_consistency_check_is_unknown_when_target_never_restates_it():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10", target_text="nothing numeric here",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN
    assert record.unknown_reason == UnknownReason.NOT_CHECKED


def test_consistency_check_never_vacuously_passes_from_zero_comparisons():
    # Two relevant symbols, neither of which the target restates --
    # "no FAIL" must not become "PASS" (artifact-graph review pattern #31).
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_REUSE,
        source_section_id="a", target_section_id="b",
        source_text="n = 10, m = 5", target_text="neither symbol appears here",
        config=_config(relevant_symbols=["n", "m"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_ambiguous_source_definition_is_not_used_as_a_basis_for_comparison():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10 ... later n = 20", target_text="n = 10",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN
    assert "inconsistent" in record.detail


def test_aliases_let_differently_spelled_symbols_be_compared():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="u = 42", target_text="U = 42",
        config=_config(relevant_symbols=["energy"], symbol_aliases={"energy": ["u", "U"]}),
    )
    assert record.outcome == CheckOutcome.PASS


def test_different_scope_values_are_not_compared_unless_named_relevant():
    # The config simply never names "n" as relevant for this obligation --
    # the two different example-local values are never even scanned for.
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="example_a", target_section_id="example_b",
        source_text="n = 5 for this example", target_text="n = 12 for a different example",
        config=_config(relevant_symbols=["alpha"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_one_symbol_can_pass_while_another_fails_in_the_same_obligation():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10, k = 25", target_text="n = 10, k = 10",
        config=_config(relevant_symbols=["n", "k"]),
    )
    # Overall outcome is FAIL (k mismatches), but n's own comparison is
    # recorded as having been checked, not silently dropped.
    assert record.outcome == CheckOutcome.FAIL
    assert "n" in record.checked_semantic_ids
    assert "k" in record.checked_semantic_ids
    assert not any(m.startswith("n:") for m in record.mismatches)
    assert any(m.startswith("k:") for m in record.mismatches)

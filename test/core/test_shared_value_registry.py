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


def test_one_resolved_symbol_does_not_let_an_unresolved_sibling_hide_behind_a_pass():
    # Regression: n matches cleanly, but k has no source definition at
    # all -- the obligation must not read as PASS just because n did.
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10", target_text="n = 10, nothing about k",
        config=_config(relevant_symbols=["n", "k"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN
    assert "n" in record.checked_semantic_ids
    assert "k" not in record.checked_semantic_ids


def test_an_ambiguous_symbol_is_not_hidden_by_a_different_symbols_success():
    # Regression: n matches cleanly, but k's source definition is itself
    # ambiguous -- must not be buried by n's success into an overall PASS.
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10, k = 25 ... later k = 99", target_text="n = 10, k = 25",
        config=_config(relevant_symbols=["n", "k"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN
    assert "inconsistent" in record.detail
    assert "k" in record.detail


def test_not_applicable_symbol_is_excluded_from_the_decision_and_reported_separately():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10", target_text="n = 10",
        config=_config(relevant_symbols=["n", "k"], not_applicable_symbols=["k"]),
    )
    assert record.outcome == CheckOutcome.PASS
    assert record.not_applicable_semantic_ids == ["k"]
    assert "k" not in record.checked_semantic_ids


def test_all_symbols_not_applicable_is_unknown_not_a_vacuous_pass():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10", target_text="n = 10",
        config=_config(relevant_symbols=["n"], not_applicable_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_must_reuse_never_passes_from_a_numeric_match_alone():
    # A restated matching number is necessary but not sufficient evidence
    # that the source's definition/formula was actually applied.
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_REUSE,
        source_section_id="a", target_section_id="b",
        source_text="n = 10", target_text="using n = 10 here",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN
    assert "n" in record.checked_semantic_ids  # it WAS compared, just insufficient to confirm reuse


def test_must_reuse_still_fails_on_a_genuine_mismatch():
    # A wrong restated number is still real, sufficient negative evidence
    # for must_reuse, even though a correct match alone can't confirm it.
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_REUSE,
        source_section_id="a", target_section_id="b",
        source_text="n = 10", target_text="using n = 15 here",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.FAIL


def test_declared_unit_mismatch_between_source_and_target_is_a_fail():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="L = 5", target_text="L = 5",
        config=_config(relevant_symbols=["L"], source_unit="meters", target_unit="feet"),
    )
    assert record.outcome == CheckOutcome.FAIL
    assert "unit" in record.detail


def test_unit_declared_on_only_one_side_is_unknown():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="L = 5", target_text="L = 5",
        config=_config(relevant_symbols=["L"], source_unit="meters"),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_declared_assumptions_mismatch_is_a_fail_even_when_values_match():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10", target_text="n = 10",
        config=_config(
            relevant_symbols=["n"],
            source_assumptions=["n is prime"], target_assumptions=["n is even"],
        ),
    )
    assert record.outcome == CheckOutcome.FAIL
    assert "assumptions" in record.detail


def test_matching_declared_assumptions_still_allow_pass():
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10", target_text="n = 10",
        config=_config(
            relevant_symbols=["n"],
            source_assumptions=["n is prime"], target_assumptions=["n is prime"],
        ),
    )
    assert record.outcome == CheckOutcome.PASS


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


def test_a_fraction_right_hand_side_is_not_truncated_to_its_numerator():
    # Regression: the old regex matched only the leading digit(s) of the
    # right-hand side and used a lookahead that only rejected a following
    # DIGIT, so "n = 1/2" silently read as n=1 and "n = 1/3" as n=1 --
    # both "matching" and PASSing despite being genuinely different
    # fractions. A fraction is not a supported restatement shape, so
    # neither side should produce an occurrence at all.
    occurrences = find_symbol_value_occurrences("n = 1/2", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 1/2", target_text="n = 1/3",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN
    assert record.outcome != CheckOutcome.PASS


def test_a_scientific_notation_right_hand_side_is_not_truncated_to_its_leading_digit():
    # Same bug class as the fraction case: "n = 1e3" and "n = 1e6" both
    # used to read as n=1 (the lookahead only rejected a following
    # digit, not a following "e"/"E").
    occurrences = find_symbol_value_occurrences("n = 1e3", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 1e3", target_text="n = 1e6",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN
    assert record.outcome != CheckOutcome.PASS


def test_a_sentence_ending_period_right_after_the_number_still_matches():
    # The fix must reject a continuation of the numeric literal itself
    # (digit, "/", "e"/"E"+digit, a second "."+digit) without becoming
    # so strict that an ordinary sentence-ending period right after the
    # number (no space) stops matching.
    occurrences = find_symbol_value_occurrences("n = 10.", "n")
    assert len(occurrences) == 1
    assert occurrences[0].value == 10.0


def test_a_double_decimal_point_right_hand_side_is_rejected():
    occurrences = find_symbol_value_occurrences("n = 10.5.3", "n")
    assert occurrences == []


def test_a_spaced_fraction_right_hand_side_is_not_truncated_to_its_numerator():
    # Regression: the round-1 fix only rejected an operator with NO space
    # before it ("1/2"); "n = 1 / 2" still silently read as n=1, since the
    # character immediately after "1" was a space, not "/".
    occurrences = find_symbol_value_occurrences("n = 1 / 2", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 1 / 2", target_text="n = 1 / 3",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_a_spaced_product_right_hand_side_is_not_truncated_to_its_first_factor():
    occurrences = find_symbol_value_occurrences("n = 10 * 2", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10 * 2", target_text="n = 10 * 3",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_a_following_markdown_bullet_on_the_next_line_is_not_mistaken_for_a_continuation():
    # The spaced-operator rejection must not cross a newline -- a "-"
    # bullet starting the NEXT line is not a subtraction continuation of
    # THIS line's restated number.
    occurrences = find_symbol_value_occurrences("n = 10\n- some bullet point", "n")
    assert len(occurrences) == 1
    assert occurrences[0].value == 10.0


def test_a_latex_times_right_hand_side_is_not_truncated_to_its_first_factor():
    # Regression: round 3's spaced-operator fix only covered ASCII
    # operators (/ * + - ^); "n = 10 \times 2" still silently read as
    # n=10 because \times starts with a backslash, not one of those.
    occurrences = find_symbol_value_occurrences(r"n = 10 \times 2", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text=r"n = 10 \times 2", target_text=r"n = 10 \times 3",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_a_latex_cdot_right_hand_side_is_not_truncated_to_its_first_factor():
    occurrences = find_symbol_value_occurrences(r"n = 10 \cdot 2", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text=r"n = 10 \cdot 2", target_text=r"n = 10 \cdot 3",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN

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


def test_a_glued_parenthetical_right_hand_side_is_not_truncated_to_its_first_token():
    # Regression: "n = 2(3)" (implied multiplication, no space before the
    # "(") used to read as n=2, discarding "(3)" entirely.
    occurrences = find_symbol_value_occurrences("n = 2(3)", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 2(3)", target_text="n = 2(4)",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_a_letter_glued_directly_to_the_number_is_not_truncated_to_the_number():
    # Regression: "n = 10x" (no space -- an exponent/variable shorthand or
    # a glued unit, genuinely ambiguous) used to read as n=10, discarding
    # the "x". A letter SEPARATED by a space is still accepted prose --
    # see test_finds_a_simple_symbol_value_occurrence and others above,
    # which already rely on exactly that ("n = 10 here", "n = 5 for this
    # example").
    occurrences = find_symbol_value_occurrences("n = 10x", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10x", target_text="n = 10y",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_a_unicode_multiplication_sign_right_hand_side_is_not_truncated_to_its_first_factor():
    # Regression: "n = 10 × 2" (Unicode multiplication sign, not the
    # ASCII "*" already rejected) used to read as n=10. Caught by a
    # CATEGORICAL check (Unicode category "Sm"), not by naming "×"
    # specifically -- this is the fix for the whack-a-mole pattern #102/
    # #105/#107 already named three times over for this exact function.
    occurrences = find_symbol_value_occurrences("n = 10 × 2", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 10 × 2", target_text="n = 10 × 3",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_a_previously_untested_unicode_math_symbol_is_also_rejected_categorically():
    # The categorical fix's whole point: a Unicode math operator this
    # test file never explicitly names before now (division sign) must
    # still be rejected, because it is caught by category, not by name.
    occurrences = find_symbol_value_occurrences("n = 10 ÷ 2", "n")
    assert occurrences == []


def test_a_space_separated_letter_inside_math_delimiters_is_algebraic_multiplication():
    # Regression: "$n = 2 x$" used to read as n=2 -- the space-separated-
    # letter exception (correct for ordinary prose like "n = 5 for this
    # example") does not hold INSIDE a LaTeX formula, where a space
    # between a number and a letter means multiplication, not prose.
    occurrences = find_symbol_value_occurrences("$n = 2 x$", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="$n = 2 x$", target_text="$n = 2 y$",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_the_same_space_separated_letter_outside_math_delimiters_still_passes():
    # The math-mode gate must not become too aggressive: the identical
    # shape OUTSIDE any $...$/\(...\)/\[...\] span is still ordinary prose.
    occurrences = find_symbol_value_occurrences("n = 2 x is a separate variable", "n")
    assert len(occurrences) == 1
    assert occurrences[0].value == 2.0


def test_a_glued_square_bracket_right_hand_side_is_not_truncated_to_its_first_token():
    # Regression: "n = 2[3]" used to read as n=2, discarding "[3]"
    # entirely -- the same bug class as round 5's "(" fix, for "[".
    occurrences = find_symbol_value_occurrences("n = 2[3]", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 2[3]", target_text="n = 2[4]",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_a_close_bracket_right_after_the_number_is_not_treated_as_a_continuation():
    # Mirrors round 5's ")" decision: a CLOSE bracket right after a
    # number ends a parenthetical remark, it doesn't continue an
    # expression -- only the OPEN bracket "[" is rejected.
    occurrences = find_symbol_value_occurrences("see item [n = 10] above", "n")
    assert len(occurrences) == 1
    assert occurrences[0].value == 10.0


def test_an_escaped_dollar_sign_is_not_mistaken_for_an_opening_math_delimiter():
    # _math_spans must not treat "\$" as toggling math mode -- otherwise
    # ordinary prose after a literal dollar amount would be wrongly
    # treated as being inside a formula.
    occurrences = find_symbol_value_occurrences(r"The price is \$5. n = 2 x is a separate variable", "n")
    assert len(occurrences) == 1
    assert occurrences[0].value == 2.0


def test_display_math_dollar_dollar_is_recognized_as_one_span_not_two_empty_ones():
    # Regression: "$$n = 2 x$$" used to read as two EMPTY math spans (one
    # "$$" pair treated as its own span, then the content "n = 2 x"
    # treated as ordinary, non-math prose, then the closing "$$" treated
    # as a second empty span) -- the real content sat entirely outside
    # any detected math span, so the space-separated "x" wrongly read as
    # prose and the restatement wrongly PASSed.
    occurrences = find_symbol_value_occurrences("$$n = 2 x$$", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="$$n = 2 x$$", target_text="$$n = 2 y$$",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_a_math_environment_is_recognized_as_math_mode():
    # Regression: \begin{equation}...\end{equation} was never recognized
    # as a math region at all, so "n = 2 x" inside it read as ordinary
    # prose and wrongly PASSed.
    text = r"\begin{equation} n = 2 x \end{equation}"
    occurrences = find_symbol_value_occurrences(text, "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text=text, target_text=r"\begin{equation} n = 2 y \end{equation}",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_an_unrecognized_latex_environment_is_not_treated_as_math():
    # The environment check must not become too aggressive: a \begin{...}
    # whose name isn't in the known math set should not change behavior
    # at all (the content inside is still checked under ordinary rules --
    # note the restatement here is NOT immediately followed by "\end{...}"
    # with only a space in between, since ANY backslash right after a
    # number is already rejected outright by round 4's unrelated blanket
    # rule, regardless of math-environment recognition).
    text = r"\begin{itemize} n = 10 is fine \end{itemize}"
    occurrences = find_symbol_value_occurrences(text, "n")
    assert len(occurrences) == 1
    assert occurrences[0].value == 10.0


def test_a_glued_curly_brace_right_hand_side_is_not_truncated_to_its_first_token():
    # Regression: "n = 2{3}" used to read as n=2, discarding "{3}"
    # entirely -- the same bug class as "(" (round 5) and "[" (round 6).
    occurrences = find_symbol_value_occurrences("n = 2{3}", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="n = 2{3}", target_text="n = 2{4}",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_an_unterminated_math_delimiter_discards_the_restatement_entirely():
    # Regression: an opening "$" with no matching close used to be
    # silently skipped, letting whatever followed it (here, "n = 2 x")
    # fall back to ordinary non-math rules -- where the space-separated
    # "x" read as harmless prose and the restatement wrongly PASSed.
    # Once a delimiter's boundary can't be resolved at all, nothing
    # after it should be trusted as a confirmed value either.
    occurrences = find_symbol_value_occurrences("$n = 2 x", "n")
    assert occurrences == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text="$n = 2 x", target_text="$n = 2 y",
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_an_unterminated_math_delimiter_discards_a_restatement_even_with_nothing_trailing():
    # The unparseable-span rejection is unconditional -- unlike the
    # ordinary in_math letter rule, it must reject a bare number with
    # NOTHING after it too, since the reason for distrust is "we don't
    # know where this formula ends," not "something specific follows."
    occurrences = find_symbol_value_occurrences("$n = 2", "n")
    assert occurrences == []


def test_a_newline_inside_a_math_environment_does_not_hide_a_continuation():
    # Regression: inside a confirmed math span, a newline is ordinary
    # LaTeX whitespace -- "n = 2\n+ 3" inside \begin{equation}...
    # \end{equation} means "n = 2 + 3", not "n = 2" followed by an
    # unrelated new line. The old code only ever skipped spaces/tabs
    # (never a newline, anywhere) before checking what followed the
    # number, so the "+ 3" on the NEXT line was invisible to that check
    # and "n = 2" read as a clean, standalone restatement.
    src = "\\begin{equation}\nn = 2\n+ 3\n\\end{equation}"
    tgt = "\\begin{equation}\nn = 2\n+ 4\n\\end{equation}"
    assert find_symbol_value_occurrences(src, "n") == []
    assert find_symbol_value_occurrences(tgt, "n") == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text=src, target_text=tgt,
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_a_newline_inside_math_mode_with_nothing_problematic_after_it_still_matches():
    # The newline-as-whitespace rule must not become "any multi-line math
    # is automatically rejected" -- a restatement immediately followed by
    # a newline and then the formula's own closing delimiter (nothing
    # arithmetic in between) still reads as a clean value.
    occurrences = find_symbol_value_occurrences("$n = 2\n$", "n")
    assert len(occurrences) == 1
    assert occurrences[0].value == 2.0


def test_a_newline_outside_math_mode_is_still_never_skipped():
    # The in_math gate must not leak into ordinary prose: a markdown "-"
    # bullet on the NEXT line, outside any math delimiters, is still not
    # mistaken for a same-formula continuation (round 3's own case,
    # re-confirmed here specifically against the newline-skip change).
    occurrences = find_symbol_value_occurrences("n = 10\n- some bullet point", "n")
    assert len(occurrences) == 1
    assert occurrences[0].value == 10.0


def test_a_crlf_line_ending_inside_math_mode_does_not_hide_a_continuation():
    # Regression: round 8's whitespace-skip only recognized a bare "\n",
    # not "\r" (or CRLF's "\r\n" pair) -- a CRLF-terminated line inside a
    # confirmed math span stopped the skip at the un-recognized "\r"
    # before ever reaching the "+ 3"/"+ 4" on the next line.
    src = "\\begin{equation}\r\nn = 2\r\n+ 3\r\n\\end{equation}"
    tgt = "\\begin{equation}\r\nn = 2\r\n+ 4\r\n\\end{equation}"
    assert find_symbol_value_occurrences(src, "n") == []
    assert find_symbol_value_occurrences(tgt, "n") == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text=src, target_text=tgt,
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_a_non_breaking_space_inside_math_mode_does_not_hide_a_continuation():
    # Regression: the same enumeration mistake as CRLF, for NBSP -- a
    # non-breaking space (U+00A0) between the number and an operator was
    # never in the old hand-picked skip set either.
    src = "$n = 2\u00A0+\u00A03$"
    tgt = "$n = 2\u00A0+\u00A04$"
    assert find_symbol_value_occurrences(src, "n") == []
    assert find_symbol_value_occurrences(tgt, "n") == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text=src, target_text=tgt,
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_a_latex_comment_after_the_number_is_rejected_not_parsed():
    # Regression: "n = 2 % comment\n+ 3" hid the real continuation behind
    # a LaTeX comment -- "%" was never in any rejection rule, so the scan
    # simply stopped there without realizing the rest of the line (and
    # thus whatever logically continues the formula past it) was hidden.
    # Rejected unconditionally, the same "can't verify it, don't guess"
    # response as the existing backslash rule -- never an attempt to
    # actually skip past the comment to see what's on the next line.
    src = "\\begin{equation}\nn = 2 % comment\n+ 3\n\\end{equation}"
    tgt = "\\begin{equation}\nn = 2 % comment\n+ 4\n\\end{equation}"
    assert find_symbol_value_occurrences(src, "n") == []
    assert find_symbol_value_occurrences(tgt, "n") == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text=src, target_text=tgt,
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN


def test_an_escaped_dollar_inside_a_formula_does_not_close_its_span_early():
    # Regression: "_math_spans"'s CLOSING-delimiter search used a plain
    # str.find, which has no escape-awareness -- an escaped "\$" INSIDE a
    # real formula could be mistaken for that formula's own closing "$",
    # ending the span early and leaving the rest of the real formula
    # (here, "n = 2 x"/"n = 2 y") to be read as ordinary prose, where the
    # space-separated letter wrongly passed as harmless narration.
    src = r"$\text{price} = \$5, n = 2 x$"
    tgt = r"$\text{price} = \$5, n = 2 y$"
    assert find_symbol_value_occurrences(src, "n") == []
    assert find_symbol_value_occurrences(tgt, "n") == []
    record = check_shared_value_consistency(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH,
        source_section_id="a", target_section_id="b",
        source_text=src, target_text=tgt,
        config=_config(relevant_symbols=["n"]),
    )
    assert record.outcome == CheckOutcome.UNKNOWN

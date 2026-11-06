"""Shared value/coefficient consistency checking for the "Revision
(Additional verifiers)" plan's Fix B, checklist item B3 ("공유 값·계수
일관성 검사") -- see ``docs/av_stage3_math_verifier.md`` for the full design
and the Gold Labeling repros this traces back to (sample_005's bending-stress/
deflection 10x/25x errors; sample_003/007/008's silent notation swaps).

Deliberately narrow, deterministic, and explicit rather than inferential:

- **No NLP/LLM guessing of "which symbols are the same referent."** A
  caller must explicitly name the symbol(s) to check per obligation
  (``SharedValueCheckConfig.relevant_symbols``) and, if two differently-
  spelled symbols denote the same thing, explicitly declare that via
  ``symbol_aliases`` -- the plan's own text: "기호 문자열만 비교하지 않고
  의미 ID... 사용한다... 명시적으로 정의한 표기 alias도 허용할 수 있다."
  An empty ``relevant_symbols`` list is UNKNOWN, never a guess -- this is
  what keeps a legitimately-independent worked-example's own ``n`` (Gold
  Labeling: "다른 예제의 n 값은 다를 수 있고") from ever being
  misdetected as an inconsistency: nothing is compared unless a caller
  explicitly said it should be.
- **Literal restatement comparison only**, not formula/derivation
  checking. If the target section never restates a relevant symbol's
  value at all, that is UNKNOWN (cannot confirm OR deny), never a silent
  PASS -- a target can legitimately reuse a definition/formula without
  literally repeating its number (the plan's "인용 한 개가 있다는 사실만
  으로 자동 PASS하지 않는다" cuts the other way too: no restatement found
  must not read as "quietly fine" either).
- **Never promotes an ambiguous source to a guessed answer.** If the
  source text itself states two different values for the same symbol,
  this is reported distinctly from a genuine source-vs-target mismatch --
  the plan explicitly warns against assuming the upstream value is
  correct ("원자료·요구사항·가정으로 수정 대상을 판단한다", not this
  module's job to decide).
- **Every relevant symbol must individually resolve before the whole
  obligation can PASS.** One symbol's successful comparison must never
  silently carry a dependency to PASS while a different relevant symbol
  was ambiguous, undefined, or never restated -- that other symbol's
  unresolved state would otherwise disappear from the result entirely. A
  symbol genuinely out of scope for THIS specific obligation (as opposed
  to merely unresolved) is tracked as its own, separate "N/A" category
  (``SharedValueCheckConfig.not_applicable_symbols``) -- never silently
  dropped from the result, and never conflated with "could not check".
- **A numeric match alone never confirms ``must_reuse``.** Restating the
  same number is necessary but not sufficient evidence that a target
  actually applied the source's definition/formula -- a numeric
  MISMATCH is still real, sufficient evidence AGAINST either
  ``must_reuse`` or ``must_be_consistent_with`` (a wrong restated number
  is a real problem either way), but a numeric match can only confirm
  ``must_be_consistent_with`` (literally the question being asked);
  ``must_reuse`` stays UNKNOWN even on a full match, honestly naming that
  this checker cannot confirm actual application from restatement alone.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.core.schemas import CheckOutcome, UnknownReason
from src.dataset.schemas import DependencyRelation

#: Shared with math_claims.py -- a single named tolerance for "two restated
#: numbers count as the same value," so a per-usage MathClaim's own outcome
#: (recomputed independently against the source definition) can never
#: disagree with this module's own mismatch detection over a float-precision
#: edge case.
NUMERIC_MATCH_TOLERANCE = 1e-9


class SharedValueOccurrence(BaseModel):
    """One ``symbol = number`` restatement actually found in rendered
    text, at a specific 1-indexed line."""

    value: float
    raw_text: str
    line: int


#: Punctuation that signals an arithmetic/grouping continuation but is NOT
#: classified under Unicode category "Sm" (Math Symbol) -- "/" and "*" are
#: "Po" (Other Punctuation), "(" is "Ps" (Open Punctuation), "-" (ASCII
#: hyphen-minus) is "Pd" (Dash Punctuation), "[" and "{" are "Ps" too, and
#: "·" (middle dot, a common plain-text stand-in for \cdot) is "Po". "^" is
#: NOT listed here -- it is already category "Sk" (Modifier Symbol), so
#: the categorical check below covers it without enumeration. Verified
#: empirically via ``unicodedata.category()``, not assumed. The matching
#: CLOSE brackets (")", "]", "}") are deliberately NOT included -- a close
#: bracket right after a number reads as ending a parenthetical remark
#: (``"(as shown, n = 10)"``), not continuing an expression, the same way
#: a trailing comma or period is not rejected either.
_NON_SM_CONTINUATION_CHARS = frozenset("/*([{-·")

#: LaTeX environment names (``\begin{name}...\end{name}``) this module
#: recognizes as math mode -- the common display-math environments, plus
#: their starred variants. Deliberately a short, explicit list rather than
#: an attempt to recognize every environment LaTeX ships (e.g. "array",
#: "matrix" are intentionally excluded: those commonly appear inside an
#: already-recognized outer math region, e.g. ``$\begin{matrix}...
#: \end{matrix}$``, so this module does not need to recognize them on
#: their own to get the outer span right).
_MATH_ENVIRONMENTS = frozenset({
    "equation", "equation*",
    "align", "align*",
    "alignat", "alignat*",
    "gather", "gather*",
    "multline", "multline*",
    "eqnarray", "eqnarray*",
    "math", "displaymath",
})


def _is_arithmetic_continuation_char(ch: str) -> bool:
    """True if ``ch`` is a character that, found right after a restated
    number, means the right-hand side continues into a larger expression
    rather than ending at that number. A CATEGORICAL check
    (``unicodedata.category(ch) in ("Sm", "Sk")``) rather than a
    per-operator blacklist -- "Sm"/"Sk" cover ``+``, ``=``, ``^``, and
    every common Unicode math operator this function has been asked to
    reject one at a time across several rounds (``×``, ``÷``, ``±``, the
    Unicode minus sign ``−``, the dot operator ``⋅``), plus any FUTURE
    math symbol never explicitly tested for -- the whole point after three
    straight rounds of "add one more operator" (bare ASCII, then spaced
    ASCII, then LaTeX commands, then Unicode symbols) is to stop
    enumerating members of an open-ended set. ``_NON_SM_CONTINUATION_CHARS``
    only exists because a handful of ordinary ASCII arithmetic/grouping
    characters are, empirically, NOT classified as math symbols by
    Unicode itself (see that set's own comment) -- this is the one place
    enumeration is unavoidable, kept as small as verified necessary."""
    return ch in _NON_SM_CONTINUATION_CHARS or unicodedata.category(ch) in ("Sm", "Sk")


def _math_spans(text: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Scans ``text`` for LaTeX math delimiters -- ``$...$``, ``$$...$$``,
    ``\\(...\\)``, ``\\[...\\]``, and a known math environment's
    ``\\begin{name}...\\end{name}`` (see ``_MATH_ENVIRONMENTS``) -- and
    returns ``(confirmed, unparseable)``, two separate lists of
    ``(start, end)`` spans:

    - ``confirmed``: both the opening AND a matching closing delimiter
      were actually found. A ``symbol = number`` restatement inside one
      of these is checked under math-mode grammar -- a space-separated
      letter right after the number is algebraic multiplication
      (``$n = 2 x$`` means ``n = 2 * x``, not "n is 2, for some reason
      named x"), not the ordinary-prose shape that identical text would
      be outside any math delimiter (``n = 10 kg``); see
      ``_is_continuation_at``'s ``in_math`` parameter.
    - ``unparseable``: an opening delimiter was found with NO matching
      close anywhere in the rest of ``text``. An EARLIER version of this
      function silently skipped an unmatched opener and let whatever
      followed it fall back to ordinary, non-math rules -- which is its
      own bug, not a safe default: an opening delimiter this function
      cannot even find the end of means we genuinely do not know what the
      rest of that formula says, so a restatement found anywhere inside
      an ``unparseable`` span must be discarded outright, unconditionally,
      regardless of what character follows it (see ``find_symbol_value_
      occurrences``) -- never read as a confirmed plain value just
      because the specific trailing character it happens to be followed
      by isn't one this module's letter/operator rules already reject.
      Each unparseable span extends from its opening delimiter to the END
      of ``text`` (not just the current line): once a delimiter cannot be
      resolved, nothing after it in this string can be trusted to be
      outside that same unresolved formula either, so scanning stops
      there entirely.

    A ``\\$`` is never treated as an opening/closing delimiter (an escaped
    dollar sign is literal text, the same escape-awareness ``src/agents/
    latex_assembler.py``'s own math-span handling already established was
    necessary -- re-implemented locally here, deliberately NOT imported
    from that module, since ``src/core`` does not depend on ``src/agents``
    and this module only needs this one narrow fact, not that module's
    full literal/texttt masking). Not a LaTeX parser -- a best-effort
    delimiter scan, not a validator of what's actually well-formed LaTeX
    inside a span it does confirm."""
    confirmed: List[Tuple[int, int]] = []
    unparseable: List[Tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and text.startswith("begin{", i + 1):
            name_start = i + len("\\begin{")
            name_end = text.find("}", name_start)
            if name_end != -1 and text[name_start:name_end] in _MATH_ENVIRONMENTS:
                name = text[name_start:name_end]
                close_tag = "\\end{" + name + "}"
                close_at = text.find(close_tag, name_end + 1)
                if close_at == -1:
                    unparseable.append((i, n))
                    break
                confirmed.append((i, close_at + len(close_tag)))
                i = close_at + len(close_tag)
                continue
            # Not a recognized math environment (or a malformed
            # "\begin{" with no closing brace for the name) -- falls
            # through to the generic backslash-escape handling below, so
            # this "\" is simply consumed as an ordinary character.
        if ch == "\\" and i + 1 < n and text[i + 1] in "([":
            closer = ")" if text[i + 1] == "(" else "]"
            end = text.find("\\" + closer, i + 2)
            if end == -1:
                unparseable.append((i, n))
                break
            confirmed.append((i, end + 2))
            i = end + 2
            continue
        if ch == "\\":
            # Any other backslash escape, including "\$" -- skip as a
            # two-character unit so an escaped dollar sign is never
            # misread as an opening/closing "$" below.
            i += 2
            continue
        if ch == "$":
            if i + 1 < n and text[i + 1] == "$":
                end = text.find("$$", i + 2)
                if end == -1:
                    unparseable.append((i, n))
                    break
                confirmed.append((i, end + 2))
                i = end + 2
                continue
            end = text.find("$", i + 1)
            if end == -1:
                unparseable.append((i, n))
                break
            confirmed.append((i, end + 1))
            i = end + 1
            continue
        i += 1
    return confirmed, unparseable


def _is_within_math_span(pos: int, spans: List[Tuple[int, int]]) -> bool:
    """``pos`` (a ``match.end()``) counts as inside ``(start, end)`` when
    ``start <= pos <= end`` -- inclusive on the right, unlike a normal
    half-open slice range. An ``unparseable`` span's ``end`` is exactly
    ``len(text)``, and a restated number can legitimately be the very
    last characters of ``text`` (``"$n = 2"``, no closing delimiter and
    nothing trailing at all) -- with a strict ``<``, ``match.end() ==
    len(text) == end`` would read as NOT inside the span, silently
    exempting exactly the "nothing follows, so there's nothing to reject"
    case an unparseable span exists to catch unconditionally. A real
    CONFIRMED span's own ``end`` is always one past an actual delimiter
    character, which a number match can never advance into, so this
    inclusive check never wrongly pulls in a different, later match."""
    return any(start <= pos <= end for start, end in spans)


def _is_continuation_at(text: str, pos: int, in_math: bool) -> bool:
    """True if the right-hand side number ending at ``pos`` does NOT stand
    alone -- i.e. the ENTIRE right-hand side is not a plain literal, so
    this restatement must not be read as a clean value at all (this
    function's caller then discards the occurrence instead of reporting
    a truncated-prefix value, e.g. reading ``n = 2(3)`` as ``n=2`` or
    ``n = 10x`` as ``n=10``).

    Skips a run of plain spaces/tabs (never a newline -- a markdown ``-``
    bullet starting the NEXT line must never be misread as a continuation
    of THIS line's number) starting at ``pos``, then inspects the next
    character, if any:

    - a digit, or a ``.`` immediately followed by a digit (the number
      itself was truncated -- an already-consumed-prefix problem);
    - a backslash (ANY LaTeX command -- blanket rejection, not specific
      operator names; see ``_occurrence_pattern``'s own history for why);
    - an arithmetic/grouping symbol or operator, by category (see
      ``_is_arithmetic_continuation_char``);
    - a letter immediately GLUED to the number with NO space at all
      (``10x``, ``10kg``) -- genuinely ambiguous (a variable-multiplication
      shorthand, an exponent, or a unit with no delimiter), never guessed
      at;
    - a letter separated from the number by a space, but ONLY when
      ``in_math`` is true (``$n = 2 x$``) -- inside a LaTeX formula there
      is no such thing as a space-separated "prose word," so the SAME
      shape that is safely ordinary prose outside math (``n = 10 kg``,
      ``n = 5 for this example``) is algebraic multiplication inside it,
      and must not be read as a clean, standalone value either.

    End of string/line, a comma, a sentence-ending period with nothing
    (or a non-digit) after it, or an ordinary space-separated word OUTSIDE
    math mode are all "not a continuation" -- the number is the whole
    right-hand side."""
    i = pos
    while i < len(text) and text[i] in " \t":
        i += 1
    glued = i == pos
    if i >= len(text):
        return False
    ch = text[i]
    if ch.isdigit():
        return True
    if ch == "." and i + 1 < len(text) and text[i + 1].isdigit():
        return True
    if ch == "\\":
        return True
    if _is_arithmetic_continuation_char(ch):
        return True
    if ch.isalpha() and (glued or in_math):
        return True
    return False


def _occurrence_pattern(symbol: str) -> "re.Pattern[str]":
    """A regex anchored to this exact literal symbol spelling -- never a
    fuzzy/substring match. A backslash-led LaTeX command (``\\alpha``) only
    needs a lookahead boundary (the leading backslash is itself already a
    left boundary); a plain identifier (``n``, ``n_0``, ``R_g``) needs both
    boundaries, so ``n`` never matches inside ``n_0`` or ``xn``.

    Captures a plain (optionally signed, optionally decimal) number
    greedily -- whether the ENTIRE right-hand side is actually just that
    bare literal (vs. continuing into a larger expression) is decided
    separately, by ``_is_continuation_at`` on the matched text, not by
    this pattern itself; see that function for why, and ``find_symbol_
    value_occurrences`` for how the two combine."""
    escaped = re.escape(symbol)
    if symbol.startswith("\\"):
        left_boundary = ""
        right_boundary = r"(?![A-Za-z])"
    else:
        left_boundary = r"(?<![A-Za-z0-9_\\])"
        right_boundary = r"(?![A-Za-z0-9_])"
    return re.compile(
        left_boundary + escaped + right_boundary
        + r"\s*=\s*(?P<value>-?\d+(?:\.\d+)?)"
    )


def find_symbol_value_occurrences(text: str, symbol: str) -> List[SharedValueOccurrence]:
    """Every literal ``symbol = <number>`` restatement in ``text``, each
    with its own 1-indexed line and raw snippet. Deterministic regex scan
    over rendered LaTeX/content text -- not a general expression parser;
    a right-hand side that isn't a plain (optionally signed, optionally
    decimal) literal number is simply not matched, never guessed at (a
    match whose right-hand side continues into a larger expression --
    checked via ``_is_continuation_at`` immediately after the matched
    number -- is discarded here rather than reported as a truncated-prefix
    value; no backtracking/retry with fewer digits is attempted, so e.g.
    ``n = 10.5.3`` yields zero occurrences, never a fallback ``10``). A
    match found inside an ``unparseable`` math span (an opening ``$``/
    ``\\(``/``\\[``/math-environment delimiter this module could not find
    a matching close for) is discarded unconditionally, regardless of
    what follows it -- see ``_math_spans``."""
    pattern = _occurrence_pattern(symbol)
    confirmed_spans, unparseable_spans = _math_spans(text)
    occurrences: List[SharedValueOccurrence] = []
    for match in pattern.finditer(text):
        if _is_within_math_span(match.end(), unparseable_spans):
            continue
        in_math = _is_within_math_span(match.end(), confirmed_spans)
        if _is_continuation_at(text, match.end(), in_math):
            continue
        line = text.count("\n", 0, match.start()) + 1
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        occurrences.append(
            SharedValueOccurrence(
                value=float(match.group("value")),
                raw_text=text[line_start:line_end].strip(),
                line=line,
            )
        )
    return occurrences


class SharedValueCheckConfig(BaseModel):
    """What a caller explicitly declares for ONE ``CrossSectionObligation``
    before this module will check anything -- see this module's own
    docstring for why every field here is deliberately opt-in, never
    inferred.

    ``source_scope``/``target_scope`` default to the same value
    (``"global"``) so two obligations that never mention scope at all
    never trigger a scope mismatch -- only an EXPLICIT difference is a
    finding. ``source_unit``/``target_unit`` and ``source_assumptions``/
    ``target_assumptions`` default to ``None`` (not declared); a mismatch
    is only reported when BOTH sides are explicitly given and genuinely
    differ, and it is reported as UNKNOWN (not skipped, not assumed
    equal) when only ONE side is given -- "지원하지 않는 항목", this
    checker has no basis to confirm or deny the undeclared side.
    """

    relevant_symbols: List[str] = Field(default_factory=list)
    #: semantic_id -> the literal symbol spellings that denote it. A
    #: ``relevant_symbols`` entry with no key here is its own sole alias
    #: (i.e. compared against itself, literally).
    symbol_aliases: Dict[str, List[str]] = Field(default_factory=dict)
    #: Symbols explicitly declared out of scope for THIS specific
    #: obligation -- excluded from the PASS/FAIL/UNKNOWN decision
    #: entirely and reported separately as N/A, never silently dropped.
    not_applicable_symbols: List[str] = Field(default_factory=list)
    source_scope: str = "global"
    target_scope: str = "global"
    source_unit: Optional[str] = None
    target_unit: Optional[str] = None
    source_assumptions: Optional[List[str]] = None
    target_assumptions: Optional[List[str]] = None

    def aliases_for(self, semantic_id: str) -> List[str]:
        return self.symbol_aliases.get(semantic_id) or [semantic_id]


class SharedValueDefinition(BaseModel):
    """One source-side binding actually found in rendered text: 'this
    semantic_id means this number, in this scope, as of this source
    section.'"""

    semantic_id: str
    source_section_id: str
    scope: str
    unit: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)
    value: float
    raw_text: str
    matched_at_line: Optional[int] = None


class SharedValueUsage(BaseModel):
    """One target-side restatement actually found in rendered text."""

    semantic_id: str
    target_section_id: str
    value: float
    raw_text: str
    matched_at_line: Optional[int] = None


class SharedValueConsistencyRecord(BaseModel):
    """Per-target verdict for one ``must_reuse``/``must_be_consistent_with``
    obligation's numeric-restatement component specifically -- never a
    claim about general semantic reuse (formula/classification reuse
    without a literal number is explicitly out of this module's scope,
    deferred to B6's logic review).

    ``checked_semantic_ids`` lists every symbol where BOTH a source
    definition AND a target restatement were actually found -- regardless
    of whether they agreed (a mismatched symbol is still "checked", just
    with its own entry in ``mismatches`` too). ``not_applicable_semantic_ids``
    is the separate, explicit "this symbol does not apply to this
    obligation" category (never conflated with "could not be resolved").
    Any ``relevant_symbols`` entry missing from BOTH lists is in
    ``unresolved_semantic_ids`` instead (no source definition, an
    ambiguous source definition, or no target restatement) -- that
    unresolved state is what forces the record's overall outcome to
    UNKNOWN rather than letting some OTHER symbol's success silently
    stand in for it, and is also what lets a MathClaim be built per
    unresolved symbol instead of that symbol's UNKNOWN fact disappearing
    during claim conversion (``unresolved_reason_by_id``/
    ``unresolved_detail_by_id`` carry the specific reason/detail text for
    each one). ``context_outcome``/``context_detail`` are the separate,
    OBLIGATION-level (not per-symbol) verdict from comparing the caller's
    declared scope/unit/assumptions (``None`` when nothing was declared
    on either side, the common case) -- kept distinct from the per-symbol
    numeric ``mismatches`` so a MathClaim can represent "the numbers
    agree but the declared assumptions don't" as its own claim rather
    than letting a clean numeric match silently absorb it.
    """

    dependency_id: str
    relation: DependencyRelation
    target_section_id: str
    outcome: CheckOutcome
    unknown_reason: Optional[UnknownReason] = None
    detail: str = ""
    checked_semantic_ids: List[str] = Field(default_factory=list)
    not_applicable_semantic_ids: List[str] = Field(default_factory=list)
    unresolved_semantic_ids: List[str] = Field(default_factory=list)
    unresolved_reason_by_id: Dict[str, UnknownReason] = Field(default_factory=dict)
    unresolved_detail_by_id: Dict[str, str] = Field(default_factory=dict)
    context_outcome: Optional[CheckOutcome] = None
    context_detail: str = ""
    mismatches: List[str] = Field(default_factory=list)
    source_definitions: List[SharedValueDefinition] = Field(default_factory=list)
    target_usages: List[SharedValueUsage] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if self.outcome == CheckOutcome.UNKNOWN and self.unknown_reason is None:
            raise ValueError("SharedValueConsistencyRecord with outcome=UNKNOWN requires unknown_reason.")


def _compare_declared_context(
    config: SharedValueCheckConfig,
) -> Tuple[Optional[CheckOutcome], str]:
    """Compares the OBLIGATION-level (not per-symbol) scope/unit/assumptions
    a caller explicitly declared for the source vs. the target side.
    Returns ``(None, "")`` when there is nothing to report (the common
    case: neither side declared anything beyond the default scope).
    FAIL outranks UNKNOWN -- a confirmed difference is reported even if
    another field on the same call couldn't be compared."""
    fail_parts: List[str] = []
    unknown_parts: List[str] = []

    if config.source_scope != config.target_scope:
        fail_parts.append(
            f"scope differs: source={config.source_scope!r} vs. target={config.target_scope!r}."
        )

    if config.source_unit is not None and config.target_unit is not None:
        if config.source_unit != config.target_unit:
            fail_parts.append(
                f"unit differs: source={config.source_unit!r} vs. target={config.target_unit!r}."
            )
    elif config.source_unit is not None or config.target_unit is not None:
        unknown_parts.append(
            "unit declared on only one side -- cannot confirm the other side's unit matches."
        )

    if config.source_assumptions is not None and config.target_assumptions is not None:
        if set(config.source_assumptions) != set(config.target_assumptions):
            fail_parts.append(
                f"assumptions differ: source={sorted(config.source_assumptions)} vs. "
                f"target={sorted(config.target_assumptions)}."
            )
    elif config.source_assumptions is not None or config.target_assumptions is not None:
        unknown_parts.append(
            "assumptions declared on only one side -- cannot confirm the other side's assumptions match."
        )

    if fail_parts:
        return CheckOutcome.FAIL, " ".join(fail_parts)
    if unknown_parts:
        return CheckOutcome.UNKNOWN, " ".join(unknown_parts)
    return None, ""


def check_shared_value_consistency(
    *,
    dependency_id: str,
    relation: DependencyRelation,
    source_section_id: str,
    target_section_id: str,
    source_text: str,
    target_text: str,
    config: SharedValueCheckConfig,
) -> SharedValueConsistencyRecord:
    """The real check: for every ``config.relevant_symbols`` entry (minus
    any ``not_applicable_symbols``), does the target restate the same
    value the source defines (via any declared alias)? Every one of them
    must individually resolve -- a mismatch anywhere is FAIL; an
    unresolved symbol anywhere (no source definition, an ambiguous source
    definition, or no target restatement) is UNKNOWN, regardless of
    whether some OTHER symbol matched cleanly; PASS requires every
    relevant, applicable symbol to have been actually compared and agree.
    ``must_reuse`` additionally never reaches PASS from this numeric
    check alone (see this module's own docstring)."""
    if not config.relevant_symbols:
        return SharedValueConsistencyRecord(
            dependency_id=dependency_id,
            relation=relation,
            target_section_id=target_section_id,
            outcome=CheckOutcome.UNKNOWN,
            unknown_reason=UnknownReason.NOT_REPRESENTABLE,
            detail=(
                "no relevant_symbols configured for this obligation -- a shared-value check "
                "never guesses which symbols a dependency shares, to avoid flagging "
                "independently-scoped values (e.g. a different worked example's own n) as "
                "inconsistent."
            ),
        )

    checked_ids: List[str] = []
    not_applicable_ids: List[str] = []
    mismatches: List[str] = []
    unresolved_ids: List[str] = []
    unresolved_reason_by_id: Dict[str, UnknownReason] = {}
    unresolved_detail_by_id: Dict[str, str] = {}
    unresolved_details: List[str] = []
    unresolved_reasons: List[UnknownReason] = []
    source_definitions: List[SharedValueDefinition] = []
    target_usages: List[SharedValueUsage] = []

    def _mark_unresolved(semantic_id: str, reason: UnknownReason, detail: str) -> None:
        unresolved_ids.append(semantic_id)
        unresolved_reason_by_id[semantic_id] = reason
        unresolved_detail_by_id[semantic_id] = detail
        unresolved_reasons.append(reason)
        unresolved_details.append(detail)

    for semantic_id in config.relevant_symbols:
        if semantic_id in config.not_applicable_symbols:
            not_applicable_ids.append(semantic_id)
            continue

        aliases = config.aliases_for(semantic_id)

        source_hits: List[Tuple[str, SharedValueOccurrence]] = []
        for alias in aliases:
            source_hits.extend((alias, occ) for occ in find_symbol_value_occurrences(source_text, alias))
        if not source_hits:
            _mark_unresolved(
                semantic_id, UnknownReason.MISSING_DEPENDENCY,
                f"{semantic_id}: source section {source_section_id!r} never defines this symbol.",
            )
            continue

        distinct_source_values: Set[float] = {round(occ.value, 9) for _, occ in source_hits}
        if len(distinct_source_values) > 1:
            # An ambiguous SOURCE definition is a different fact than a
            # genuine source-vs-target mismatch -- it must never feed the
            # `mismatches` list that drives the FAIL branch below, or an
            # unresolvable source-side ambiguity would be misreported as
            # a confirmed target violation (the plan's own "정답 source를
            # 추측해 덮어쓰지 않는다" cuts against guessing in either
            # direction).
            values_detail = ", ".join(str(v) for v in sorted(distinct_source_values))
            _mark_unresolved(
                semantic_id, UnknownReason.NOT_REPRESENTABLE,
                f"{semantic_id}: source section {source_section_id!r} itself states inconsistent "
                f"values ({values_detail}) -- not used as a basis for comparison, the correct "
                "one is not guessed.",
            )
            continue

        _, source_occ = source_hits[0]
        target_hits: List[Tuple[str, SharedValueOccurrence]] = []
        for alias in aliases:
            target_hits.extend((alias, occ) for occ in find_symbol_value_occurrences(target_text, alias))
        if not target_hits:
            _mark_unresolved(
                semantic_id, UnknownReason.NOT_CHECKED,
                f"{semantic_id}: target section {target_section_id!r} never restates this symbol.",
            )
            continue

        # Both sides resolved -- this symbol is now actually "checked",
        # whether or not it turns out to match.
        checked_ids.append(semantic_id)
        source_definitions.append(
            SharedValueDefinition(
                semantic_id=semantic_id,
                source_section_id=source_section_id,
                scope=config.source_scope,
                unit=config.source_unit,
                assumptions=list(config.source_assumptions or []),
                value=source_occ.value,
                raw_text=source_occ.raw_text,
                matched_at_line=source_occ.line,
            )
        )
        for _, occ in target_hits:
            target_usages.append(
                SharedValueUsage(
                    semantic_id=semantic_id,
                    target_section_id=target_section_id,
                    value=occ.value,
                    raw_text=occ.raw_text,
                    matched_at_line=occ.line,
                )
            )
            if abs(occ.value - source_occ.value) > NUMERIC_MATCH_TOLERANCE:
                mismatches.append(
                    f"{semantic_id}: source {source_section_id!r}={source_occ.value} vs. "
                    f"target {target_section_id!r}={occ.value} (line {occ.line})."
                )

    context_outcome, context_detail = _compare_declared_context(config)

    if mismatches or context_outcome == CheckOutcome.FAIL:
        detail_parts = list(mismatches)
        if context_outcome == CheckOutcome.FAIL:
            detail_parts.append(context_detail)
        return SharedValueConsistencyRecord(
            dependency_id=dependency_id,
            relation=relation,
            target_section_id=target_section_id,
            outcome=CheckOutcome.FAIL,
            detail="; ".join(detail_parts),
            checked_semantic_ids=checked_ids,
            not_applicable_semantic_ids=not_applicable_ids,
            unresolved_semantic_ids=unresolved_ids,
            unresolved_reason_by_id=unresolved_reason_by_id,
            unresolved_detail_by_id=unresolved_detail_by_id,
            context_outcome=context_outcome,
            context_detail=context_detail,
            mismatches=mismatches,
            source_definitions=source_definitions,
            target_usages=target_usages,
        )

    if unresolved_details or context_outcome == CheckOutcome.UNKNOWN:
        detail_parts = list(unresolved_details)
        if context_outcome == CheckOutcome.UNKNOWN:
            detail_parts.append(context_detail)
        # One record carries one unknown_reason; an obligation checking
        # several symbols can have several different unresolved reasons
        # at once (one missing from source, another never restated) --
        # the first one encountered (relevant_symbols order) is reported,
        # same "don't average/guess, report the first real fact found"
        # discipline the rest of this module already applies.
        reason = unresolved_reasons[0] if unresolved_reasons else UnknownReason.NOT_REPRESENTABLE
        return SharedValueConsistencyRecord(
            dependency_id=dependency_id,
            relation=relation,
            target_section_id=target_section_id,
            outcome=CheckOutcome.UNKNOWN,
            unknown_reason=reason,
            detail="; ".join(detail_parts),
            checked_semantic_ids=checked_ids,
            not_applicable_semantic_ids=not_applicable_ids,
            unresolved_semantic_ids=unresolved_ids,
            unresolved_reason_by_id=unresolved_reason_by_id,
            unresolved_detail_by_id=unresolved_detail_by_id,
            context_outcome=context_outcome,
            context_detail=context_detail,
            source_definitions=source_definitions,
            target_usages=target_usages,
        )

    if not checked_ids:
        # Every relevant_symbols entry was N/A -- nothing was ever
        # actually confirmed, so this can never read as PASS (the same
        # "no FAIL is not PASS" principle applied to "no comparison
        # happened" rather than "no mismatch happened").
        return SharedValueConsistencyRecord(
            dependency_id=dependency_id,
            relation=relation,
            target_section_id=target_section_id,
            outcome=CheckOutcome.UNKNOWN,
            unknown_reason=UnknownReason.NOT_CHECKED,
            detail="every relevant_symbols entry was marked not-applicable to this obligation -- nothing checked.",
            not_applicable_semantic_ids=not_applicable_ids,
            context_outcome=context_outcome,
            context_detail=context_detail,
        )

    # Every relevant, applicable symbol resolved and agreed, and no
    # declared-context mismatch was found.
    if relation != DependencyRelation.MUST_BE_CONSISTENT_WITH:
        # must_reuse (and, defensively, anything else routed through this
        # numeric-only checker): a clean numeric match is necessary but
        # not sufficient evidence that the source's definition/formula
        # was actually applied -- see this module's own docstring. Stays
        # UNKNOWN, never PASS, from this check alone.
        return SharedValueConsistencyRecord(
            dependency_id=dependency_id,
            relation=relation,
            target_section_id=target_section_id,
            outcome=CheckOutcome.UNKNOWN,
            unknown_reason=UnknownReason.NOT_REPRESENTABLE,
            detail=(
                f"numeric restatement is consistent for {checked_ids}, but a numeric match alone "
                f"does not confirm {relation.value} was actually satisfied -- this checker only "
                "verifies the numeric-restatement component, never general semantic reuse."
            ),
            checked_semantic_ids=checked_ids,
            not_applicable_semantic_ids=not_applicable_ids,
            unresolved_semantic_ids=unresolved_ids,
            context_outcome=context_outcome,
            context_detail=context_detail,
            source_definitions=source_definitions,
            target_usages=target_usages,
        )

    return SharedValueConsistencyRecord(
        dependency_id=dependency_id,
        relation=relation,
        target_section_id=target_section_id,
        outcome=CheckOutcome.PASS,
        detail=f"consistent restated value(s) for {checked_ids}.",
        checked_semantic_ids=checked_ids,
        not_applicable_semantic_ids=not_applicable_ids,
        unresolved_semantic_ids=unresolved_ids,
        context_outcome=context_outcome,
        context_detail=context_detail,
        source_definitions=source_definitions,
        target_usages=target_usages,
    )


__all__ = [
    "NUMERIC_MATCH_TOLERANCE",
    "SharedValueOccurrence",
    "find_symbol_value_occurrences",
    "SharedValueCheckConfig",
    "SharedValueDefinition",
    "SharedValueUsage",
    "SharedValueConsistencyRecord",
    "check_shared_value_consistency",
]

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
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.core.schemas import CheckOutcome, UnknownReason
from src.dataset.schemas import DependencyRelation


class SharedValueOccurrence(BaseModel):
    """One ``symbol = number`` restatement actually found in rendered
    text, at a specific 1-indexed line."""

    value: float
    raw_text: str
    line: int


def _occurrence_pattern(symbol: str) -> "re.Pattern[str]":
    """A regex anchored to this exact literal symbol spelling -- never a
    fuzzy/substring match. A backslash-led LaTeX command (``\\alpha``) only
    needs a lookahead boundary (the leading backslash is itself already a
    left boundary); a plain identifier (``n``, ``n_0``, ``R_g``) needs both
    boundaries, so ``n`` never matches inside ``n_0`` or ``xn``."""
    escaped = re.escape(symbol)
    if symbol.startswith("\\"):
        left_boundary = ""
        right_boundary = r"(?![A-Za-z])"
    else:
        left_boundary = r"(?<![A-Za-z0-9_\\])"
        right_boundary = r"(?![A-Za-z0-9_])"
    return re.compile(
        left_boundary + escaped + right_boundary
        + r"\s*=\s*(?P<value>-?\d+(?:\.\d+)?)(?!\d)"
    )


def find_symbol_value_occurrences(text: str, symbol: str) -> List[SharedValueOccurrence]:
    """Every literal ``symbol = <number>`` restatement in ``text``, each
    with its own 1-indexed line and raw snippet. Deterministic regex scan
    over rendered LaTeX/content text -- not a general expression parser;
    a right-hand side that isn't a plain (optionally signed, optionally
    decimal) literal number is simply not matched, never guessed at."""
    pattern = _occurrence_pattern(symbol)
    occurrences: List[SharedValueOccurrence] = []
    for match in pattern.finditer(text):
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
    inferred."""

    relevant_symbols: List[str] = Field(default_factory=list)
    #: semantic_id -> the literal symbol spellings that denote it. A
    #: ``relevant_symbols`` entry with no key here is its own sole alias
    #: (i.e. compared against itself, literally).
    symbol_aliases: Dict[str, List[str]] = Field(default_factory=dict)
    scope: str = "global"
    unit: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)

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
    deferred to B6's logic review)."""

    dependency_id: str
    relation: DependencyRelation
    target_section_id: str
    outcome: CheckOutcome
    unknown_reason: Optional[UnknownReason] = None
    detail: str = ""
    checked_semantic_ids: List[str] = Field(default_factory=list)
    mismatches: List[str] = Field(default_factory=list)
    source_definitions: List[SharedValueDefinition] = Field(default_factory=list)
    target_usages: List[SharedValueUsage] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if self.outcome == CheckOutcome.UNKNOWN and self.unknown_reason is None:
            raise ValueError("SharedValueConsistencyRecord with outcome=UNKNOWN requires unknown_reason.")


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
    """The real check: for every ``config.relevant_symbols`` entry, does
    the target restate the same value the source defines (via any
    declared alias)? Never vacuously PASSes from zero actually-compared
    symbols (mirrors the artifact-graph review's "no FAIL is not PASS"
    discipline) -- PASS requires at least one symbol where BOTH a source
    definition and a target restatement were actually found and agree."""
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
    mismatches: List[str] = []
    ambiguous_source_details: List[str] = []
    source_definitions: List[SharedValueDefinition] = []
    target_usages: List[SharedValueUsage] = []
    any_target_usage_found = False

    for semantic_id in config.relevant_symbols:
        aliases = config.aliases_for(semantic_id)

        source_hits: List[Tuple[str, SharedValueOccurrence]] = []
        for alias in aliases:
            source_hits.extend((alias, occ) for occ in find_symbol_value_occurrences(source_text, alias))
        if not source_hits:
            continue  # source never defines this id -- nothing to compare against

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
            ambiguous_source_details.append(
                f"{semantic_id}: source section {source_section_id!r} itself states inconsistent "
                f"values ({values_detail}) -- not used as a basis for comparison, the correct "
                "one is not guessed."
            )
            continue

        _, source_occ = source_hits[0]
        target_hits: List[Tuple[str, SharedValueOccurrence]] = []
        for alias in aliases:
            target_hits.extend((alias, occ) for occ in find_symbol_value_occurrences(target_text, alias))
        if not target_hits:
            continue  # target never restates this id -- no evidence either way

        any_target_usage_found = True
        checked_ids.append(semantic_id)
        source_definitions.append(
            SharedValueDefinition(
                semantic_id=semantic_id,
                source_section_id=source_section_id,
                scope=config.scope,
                unit=config.unit,
                assumptions=list(config.assumptions),
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
            if abs(occ.value - source_occ.value) > 1e-9:
                mismatches.append(
                    f"{semantic_id}: source {source_section_id!r}={source_occ.value} vs. "
                    f"target {target_section_id!r}={occ.value} (line {occ.line})."
                )

    if mismatches:
        return SharedValueConsistencyRecord(
            dependency_id=dependency_id,
            relation=relation,
            target_section_id=target_section_id,
            outcome=CheckOutcome.FAIL,
            detail="; ".join(mismatches),
            checked_semantic_ids=checked_ids,
            mismatches=mismatches,
            source_definitions=source_definitions,
            target_usages=target_usages,
        )
    if any_target_usage_found:
        return SharedValueConsistencyRecord(
            dependency_id=dependency_id,
            relation=relation,
            target_section_id=target_section_id,
            outcome=CheckOutcome.PASS,
            detail=f"consistent restated value(s) for {checked_ids}.",
            checked_semantic_ids=checked_ids,
            source_definitions=source_definitions,
            target_usages=target_usages,
        )
    if ambiguous_source_details:
        return SharedValueConsistencyRecord(
            dependency_id=dependency_id,
            relation=relation,
            target_section_id=target_section_id,
            outcome=CheckOutcome.UNKNOWN,
            unknown_reason=UnknownReason.NOT_REPRESENTABLE,
            detail="; ".join(ambiguous_source_details),
        )
    return SharedValueConsistencyRecord(
        dependency_id=dependency_id,
        relation=relation,
        target_section_id=target_section_id,
        outcome=CheckOutcome.UNKNOWN,
        unknown_reason=UnknownReason.NOT_CHECKED,
        detail=(
            f"none of {config.relevant_symbols} were restated as a plain number in target section "
            f"{target_section_id!r} -- cannot confirm or deny numeric reuse/consistency from literal "
            "restatement alone (the target may still reuse the definition/formula without repeating "
            "its number, which this check cannot see)."
        ),
    )


__all__ = [
    "SharedValueOccurrence",
    "find_symbol_value_occurrences",
    "SharedValueCheckConfig",
    "SharedValueDefinition",
    "SharedValueUsage",
    "SharedValueConsistencyRecord",
    "check_shared_value_consistency",
]

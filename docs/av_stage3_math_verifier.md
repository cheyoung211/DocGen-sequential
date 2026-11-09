# Stage 3 (Revision "Additional verifiers") -- B1-B3 + C3's second bullet

Notion "Revision (Additional verifiers)" §4's own recommended build order puts this
round at **순서 3**: "B1-B3 및 C3의 재사용·일관성 검사", with its own stated
pass-criterion for moving to 순서 4: "source 기반 재계산, 같은 대상의 값·가정 비교"
(source-based recomputation works; comparison of the same referent's value/assumptions
works). Code baseline: branch `additional_verifier`, on top of commit `d8720df`
("stage2 revision2" -- C1/C2/C4's 2026-11-04 review-fix round, per
`docs/av_stage2_dependency_mapping.md` §7).

Explicitly **not** this round, per the plan's own §4 table: B4-B5 (wiring into the
real `ToolExecutor`/controller loop and a pilot run against a real model) and B6
(symbolic/logic review) are 순서 4 and 후속, respectively.

## 1. B1 -- scope and `MathClaim` schema

`src/core/math_claims.py`. Three `MathCriterionId` values, never collapsed into one
pass/fail per the plan's own text ("계산 정확성, 값 일관성, 논리 타당성에 서로 다른
criterion ID를 부여한다"):

- `CALCULATION_CORRECTNESS` -- B2's recompute-from-source checks.
- `VALUE_CONSISTENCY` -- B3's shared-value restatement checks.
- `LOGICAL_VALIDITY` -- declared now so a future B6 claim can reuse this same
  coverage shape, but **deliberately never produced by any checker on this
  branch** -- pinned down by
  `test_math_claims.py::test_logical_validity_criterion_is_declared_but_never_auto_produced`
  so an accidental future use doesn't go unnoticed (the Stage 3 review pattern #20
  "an error code nothing raises" applied in advance, rather than found later).

`MathClaim` carries exactly the fields the plan's text asks for: artifact id/version,
raw text + line, operation/symbol, observed/expected value, `source_ref`
(`None` means an externally-asserted fact with no computable source at all --
distinct from a *sourced* claim that still came out UNKNOWN for some other reason,
e.g. bad JSON or a missing artifact; see `MathClaimCoverageReport.
externally_sourced_count`), scope/unit/assumptions, tolerance, and the real
outcome/unknown_reason.

**Deliberately not done this round**: `MathClaim` is a reporting/coverage structure
built from a check's own real inputs (`math_claim_for_recompute`) or a
`SharedValueConsistencyRecord` (`math_claims_for_shared_value_record`) -- never
reverse-parsed from `Evidence.detail`'s free text, which would be guessing at
already-structured data the checker had. It is **not** wired into
`BasicConstraintChecker.check()`'s own return type (`List[Evidence]`, the
`Validator` Protocol's fixed contract) -- changing that shape would ripple into
every real caller (`tool_executor.py`, `repair_operations.py`, `controller.py`) for
a schema this round doesn't need wired that deep; actually populating a
`MathClaimCoverageReport` from a real pilot run is B5's job ("coverage·UNKNOWN·비용
보고"), once B4 has wired the checks into the real execution path at all.

## 2. B2 -- source-based recomputation, extended

`src/core/constraint_checks.py`'s existing `RECOMPUTE`-from-source mode (one
operation, `"sum"`) is extended per the plan's own table row ("사칙연산, 합계,
비율·백분율, 값 대입"):

| operation | arity | notes |
| --- | --- | --- |
| `sum`, `mean`, `product` | any (>=1) | reductions over the whole source array |
| `difference` | any (>=1) | `numbers[0] - sum(numbers[1:])` |
| `ratio`, `percentage` | exactly 2 | `a/b`, `(a/b)*100` |
| `percentage_change` | exactly 2, `[old, new]` | `((new-old)/old)*100` |
| `power` | exactly 2, `[base, exponent]` | added specifically for Gold Labeling
  sample_001's `0.998^999` miscalculation -- `test_power_operation_catches_gold_
  labeling_sample_001` reproduces that exact claim |

An arity mismatch is UNKNOWN/`NOT_REPRESENTABLE` ("입력 부족"), never padded,
truncated, or guessed at. Division by zero, a negative base with a fractional
exponent (`**` returning a non-representable complex result), or an overflow are
all UNKNOWN/`NOT_REPRESENTABLE`, never an unhandled exception.

**"값 대입" (value substitution)** is satisfied by these operations themselves
consuming already-substituted source values, not by adding a general expression
evaluator -- explicitly the same boundary the pre-existing `"sum"` mode already
drew (its own docstring: "not a general expression evaluator"). An arbitrary-formula
`eval()` path would be a real code-execution surface over source *content*, which
this checker does not open.

**반올림/절대·상대 오차**: `_recompute_verdict` gained `round_to` (rounds both
`expected`/`observed` to N decimal places before comparing) and `tolerance_mode`
(`"absolute"`, the pre-existing default, or `"relative"`, scaling the tolerance by
`abs(expected)`). An unrecognized `tolerance_mode` is UNKNOWN/`NOT_REPRESENTABLE`,
never silently treated as `"absolute"` -- a typo in this parameter must not quietly
change what tolerance was actually enforced (Stage 3 review pattern #25's "a
guarantee word is a testable claim" applies symmetrically to a *mode name*, not
just a guarantee word).

**분수 (fractions)**: a source JSON array element may now be a plain number, a
decimal string (`"3.5"`), or a simple `"a/b"` fraction string (`"3/4"`, `"-1/2"`),
via the new `_coerce_recompute_number`. A percentage *string* (`"12%"`) is
deliberately left unparsed -- whether that means `0.12` or `12` is genuinely
ambiguous without more context than a bare source array gives, so it is a parse
error (same as any other unsupported shape) rather than a guess; the
`"percentage"`/`"percentage_change"` *operations* exist for percentage
*computation* instead. JSON booleans are explicitly rejected even though
`isinstance(True, int)` is `True` in Python -- without this, a stray
`true`/`false` in a source array would silently become `1.0`/`0.0`.

**실행 시간 한도**: a hard cap (`_MAX_RECOMPUTE_INPUTS = 10_000`) on how many source
values one recompute call reads, reported as UNKNOWN/`SOLVER_TIMEOUT` when
exceeded. This is honestly a size bound, **not** a preemptive wall-clock timeout --
every operation above is a single linear pass with no recursion, so the real risk
is a pathologically large stored artifact, not a slow algorithm; the docstring says
exactly this, per this project's standing rule against claiming a stronger guarantee
than the literal code enforces. A useful side effect: this is the first real code
path that can ever produce `UnknownReason.SOLVER_TIMEOUT` -- Stage 3's own review
(pattern #20) had flagged that value as declared-but-unreachable dead schema.

**External-fact vs. calculable claims**: already structurally distinguished by the
pre-existing direct-comparison mode (`expected` supplied directly, no `source_ref`)
vs. recompute-from-source mode -- no new code needed here; `MathClaim.source_ref`
(B1) is what lets a coverage report count the two separately.

## 3. B3 -- shared value/coefficient consistency

`src/core/shared_value_registry.py` (new module). The genuinely new piece this
round, because no existing code does anything like it: `contract_satisfaction.
_score_notation` only checks whether a `NotationConstraint.symbol` string appears
*anywhere* in the whole document -- no per-section location, no value, no
consistency check at all (confirmed by reading it, not assumed).

### Design decision: never infer which symbols are "the same referent"

The plan's own text is explicit that two things must stay human/caller-declared,
never guessed by this module:

- **which symbols are shared** between a specific source/target pair
  (`SharedValueCheckConfig.relevant_symbols`) -- "다른 예제의 n 값은 다를 수
  있다" means a bare "scan both sections for any matching symbol name" design
  would misfire exactly on worked examples that happen to reuse a common letter
  for an unrelated, independently-scoped value. `relevant_symbols=[]` (the
  default) is UNKNOWN/`NOT_REPRESENTABLE`, never a guess.
- **which differently-spelled symbols denote the same thing**
  (`symbol_aliases`) -- "명시적으로 정의한 표기 alias도 허용할 수 있다". Without
  an explicit alias, a genuine notation swap (Gold Labeling sample_003/007/008's
  `u`/`U`, `Δ%`/`RG`) is **not** detected as an inconsistency -- it reads as
  UNKNOWN ("target never restates this id"), which is the correct, conservative
  answer: this module has no way to know `u` and `U` are meant to be the same
  symbol without being told, and guessing wrong in the other direction (treating
  every same-letter-different-case pair as automatically identical) would itself
  misfire on cases where they really are different quantities. Closing this gap
  for real would need either explicit authoring-time aliases (not generated by
  this round) or genuine semantic/NLP matching -- out of scope, and arguably B6
  territory (logic/semantic review), not B3's "값·계수" comparison.

### The actual check (`find_symbol_value_occurrences` + `check_shared_value_consistency`)

A deterministic regex scan for literal `symbol = <number>` restatements (boundary-
aware so `n` never matches inside `n_0`, and a backslash command like `\alpha`
never matches as a prefix of `\alphabet`) -- never a general expression parser.
For each `relevant_symbols` entry (each is its own alias group of 1+ literal
spellings):

1. If the source text never states it at all -- skip (nothing to compare against).
2. If the source itself states two different values for the same id -- that id is
   excluded from comparison, with an explicit, separately-tracked "ambiguous
   source" reason (never silently picks one as "the real" value, and never lets
   this feed the same `mismatches` list a genuine target disagreement would --
   confirmed by a regression test after a self-caught bug where it initially did,
   misreporting an ambiguous-source case as a confirmed FAIL).
3. If the target text never restates it -- no evidence either way (a target can
   legitimately reuse a definition/formula without literally repeating the
   number); UNKNOWN, never a vacuous PASS.
4. If the target restates it with a different value -- FAIL, with both sides'
   value/location in the detail.
5. If the target restates it with the same value -- that one id counts toward
   PASS.

Overall outcome: FAIL if ANY relevant symbol mismatched; else PASS if AT LEAST ONE
symbol was actually compared (both sides found, values agree); else UNKNOWN. Never
PASSes from zero actually-compared symbols -- the same "no FAIL is not PASS"
discipline the artifact-graph review settled (pattern #31), applied here to "no
comparison happened" rather than "no FAIL outcome happened". A dependency checking
two symbols where only one disagrees still reports the *other* symbol's own
comparison in `checked_semantic_ids`/`source_definitions`/`target_usages` -- not
collapsed into the aggregate FAIL (mirrored one level down in B1's
`math_claims_for_shared_value_record`, which emits one `MathClaim` per checked
symbol with its own PASS/FAIL, never the record's aggregate outcome).

## 4. C3's second bullet -- wiring `must_reuse`/`must_be_consistent_with`

Per `docs/av_stage2_dependency_mapping.md` §5.2's own forward note: "B가 실제
registry/recompute 검사를 들고 오면 `_check_dependency`의 해당 분기와
`dependency_verifier`의 해당 분기만 채우면 되고, 이 라운드의 provider/executor/
checkpoint 연결은 그대로 재사용된다." Followed almost exactly, with one addition
explained below.

- `dependency_verifier._verify_shared_value_relation` (new): adapts
  `shared_value_registry.check_shared_value_consistency`'s verdict into this
  module's `DependencyVerificationRecord` vocabulary -- the same "decide once,
  adapt elsewhere" pattern `_verify_must_reference`/`_verify_must_precede` already
  established, never a second implementation of the decision.
- `verify_cross_section_obligations(report, rendered_sections, shared_value_configs=None)`:
  the whole-document entry point gained an optional third parameter, keyed by
  `dependency_id`, each value a `SharedValueCheckConfig`. **Backward compatible by
  construction**: the pre-existing 2-arg call (used by
  `test_must_reuse_and_must_be_consistent_with_are_unknown_not_passed`, unchanged)
  still resolves to exactly the pre-B3 UNKNOWN/`NOT_REPRESENTABLE` for every
  `must_reuse`/`must_be_consistent_with` obligation.
- `constraint_checks.BasicConstraintChecker._check_dependency`: the branch that used
  to unconditionally return UNKNOWN for these two relations now checks
  `values.get("relevant_symbols")` first -- empty/absent keeps the exact prior
  behavior (pinned by
  `test_must_be_consistent_with_is_unknown_without_relevant_symbols_even_when_values_present`);
  non-empty fetches BOTH the source's and the target's stored content via
  `self._store` (the method already had target-content access for
  `must_reference`; source-content access is new, resolved from
  `constraint.applies_to[0]`, the same ref `_resolve_input_refs` already
  existence-checks) and delegates to `_verify_shared_value_relation`.

**One deviation from "provider/executor/checkpoint connection is reused as-is",
explained**: `DependencyConstraintProvider` gained one new optional constructor
parameter, `relevant_symbols_by_dependency: Dict[str, List[str]]`, threaded into
`context_for()`'s per-constraint dict as a new `"relevant_symbols"` key (alongside
the pre-existing `relation`/`allowed_labels`/`order_hint`). This was necessary, not
optional: B3's own design above requires an *explicit, per-dependency* symbol
allowlist to avoid false-positiving independently-scoped values, and nothing else
in the existing provider/report has that information to hand to `_check_dependency`
without it. The change is additive and default-empty -- a caller who does not pass
`relevant_symbols_by_dependency` gets `"relevant_symbols": []` for every
obligation, which `_check_dependency` already treats as "stay UNKNOWN", so every
existing caller's behavior is unchanged (confirmed:
`test_relevant_symbols_default_to_empty_without_opting_in`). The existing
`test_context_for_matches_call_one_to_one` was updated to include the new key in
its exact-equality assertion, since the dict's real shape legitimately changed.

Actually populating `relevant_symbols_by_dependency` from the real benchmark's
`NotationConstraint`s (or any other automatic source) for the live pipeline is
explicitly **not** done this round -- that is B4's "ConstraintProvider가 실제
검증 입력을 공급하도록 한다", and doing it without B4's own wiring/pilot
validation would be assuming a mapping heuristic this round never actually tested
end-to-end.

## 5. What this round deliberately does not solve

- **General semantic reuse** (a target applying a source's formula/classification
  without literally restating a number) -- B3 only ever confirms/denies the
  *numeric-restatement* component of `must_reuse`; this is the exact limitation
  the plan's own text warns about ("인용 한 개가 있다는 사실만으로 자동 PASS하지
  않는다" cuts the other way too -- zero restatement is not proof of non-reuse
  either, so it stays UNKNOWN, never FAIL).
- **Logical/proof validity** (Gold Labeling sample_000's `T_n(x)` substituted for
  `f(x)`) -- `MathCriterionId.LOGICAL_VALIDITY` is declared, never produced; B6.
- **General symbolic equivalence** (polynomial expansion/differentiation/equality
  under a stated domain) -- B6, needs SymPy, not a dependency of this project yet.
- **Automatic alias/scope inference** -- both are caller-declared only, as
  explained in §3. A real pilot (B5) may reveal this MVP boundary is too narrow in
  practice (e.g. too many real `must_reuse` obligations come back UNKNOWN because
  nobody supplied `relevant_symbols`) -- that is exactly what B5's "MVP 검증"
  exists to surface before any broader claim is made.
- **Wiring into the real execution path** (`run_full_pipeline.py`,
  `ToolExecutor`/controller loop) -- B4. This round's checks are real and
  independently tested (B1-B3 schemas/registry, C3's `dependency_verifier`/
  `constraint_checks` branches), but nothing yet calls them from a live generation
  run.

## 5.1. Review round 1 (2026-11-04, same day, uncommitted) -- 4 real bugs found

A user review against the real code (not just the passing tests) found 4 real
bugs on the first pass, each reproduced directly before fixing.

1. **A dependency's overall PASS could hide a sibling symbol that was never
   actually resolved.** `check_shared_value_consistency` originally `continue`d
   silently past a `relevant_symbols` entry with no source definition, an
   ambiguous source definition, or no target restatement -- so with
   `relevant_symbols=["n", "k"]`, "n" matching cleanly produced an overall PASS
   even when "k" had no source definition at all, or an ambiguous one. Reproduced
   directly (`check_shared_value_consistency(relevant_symbols=["n","k"], ...)`
   with only "n" resolvable -> `CheckOutcome.PASS`, confirmed wrong). Fixed: every
   relevant symbol is now tracked into exactly one of three buckets --
   `checked_semantic_ids` (both sides resolved, match or mismatch), a new
   `not_applicable_semantic_ids` (explicitly declared out-of-scope for this
   specific obligation via the new `SharedValueCheckConfig.not_applicable_symbols`,
   never silently inferred), or unresolved (tracked internally, forces the whole
   record to UNKNOWN). Priority is now: any mismatch -> FAIL; any unresolved
   symbol -> UNKNOWN; all relevant/applicable symbols resolved and agreed -> PASS
   (or the must_reuse exception below); all symbols N/A -> UNKNOWN, never a
   vacuous PASS. 4 new regression tests.
2. **`must_reuse`'s PASS never actually confirmed reuse.** The checker treated
   `must_reuse` and `must_be_consistent_with` identically -- a clean numeric
   restatement match produced PASS for both, but a target literally stating the
   same number as the source is not evidence that the source's definition/
   formula was actually *applied* (the plan's own "인용 한 개가 있다는 사실만으로
   자동 PASS하지 않는다" extends to a bare numeric match too). Fixed: a full,
   clean numeric match now yields PASS only for `must_be_consistent_with`;
   `must_reuse` stays UNKNOWN/`NOT_REPRESENTABLE` even on a full match, with a
   detail explicitly naming that this checker only verifies the numeric-
   restatement component, never general semantic reuse -- a genuine MISMATCH is
   still FAIL for `must_reuse` too (a wrong restated number is real negative
   evidence either way). Separately, `SharedValueCheckConfig` gained
   `source_scope`/`target_scope` (default `"global"`/`"global"`, so undeclared
   scope never triggers a mismatch), `source_unit`/`target_unit`, and
   `source_assumptions`/`target_assumptions` (all `Optional`, previously a single
   shared `scope`/`unit`/`assumptions` field was silently copied onto both sides,
   so a genuine source-vs-target assumption difference could never be detected at
   all) -- compared once per obligation by the new `_compare_declared_context`:
   a confirmed difference on either side is FAIL; a field declared on only one
   side is UNKNOWN ("지원하지 않는 항목"); both undeclared is simply not compared
   (zero behavior change for every existing caller that never sets these). 7 new
   regression tests (must_reuse-match-stays-UNKNOWN, must_reuse-mismatch-still-
   FAILs, unit mismatch/one-sided, assumptions mismatch/match).
   `_verify_shared_value_relation` and its one real caller
   (`constraint_checks._check_dependency`) were refactored to pass a whole
   `SharedValueCheckConfig` object through instead of unpacked scope/unit/
   assumptions kwargs, specifically so this module and its caller can never drift
   apart on the config's actual field set again.
3. **Recompute input parsing could crash instead of returning UNKNOWN, and
   NaN/Infinity were silently mis-classified.** `_check_recompute` called
   `float(values["observed"])`/`float(values["expected"])` with no exception
   handling at all -- a non-numeric `observed` (reproduced:
   `observed="not-a-number"`) raised an uncaught `ValueError` instead of
   producing Evidence. Separately, Python's `json.loads` accepts the bare tokens
   `NaN`/`Infinity` by default, and `float("nan")`/`float(nan)` parse without
   error -- comparing a NaN with `<=` is always `False` (never raises), so a NaN
   `observed` silently fell through to the FAIL branch instead of being
   recognized as "not comparable at all" (reproduced and confirmed). Fixed:
   `observed`/`tolerance`/`expected` are now parsed via the same
   `_coerce_recompute_number` the source array already used (reused, not
   duplicated), wrapped in `try/except` at each call site -> UNKNOWN/
   `NOT_REPRESENTABLE`, never an uncaught exception; `_coerce_recompute_number`
   itself now rejects any non-finite result (catches NaN/Infinity from BOTH a
   source array string token AND a directly-supplied bool/number); `_recompute_
   verdict` (the one funnel point both RECOMPUTE modes' `expected` passes
   through) additionally validates `expected`/`observed`/`tolerance` are finite
   right before comparing -- needed because a legitimate operation on
   all-finite inputs can still overflow to `inf` with no exception raised at all
   (`1e308 + 1e308` silently becomes `inf` in Python). `round_to` is now
   validated as a real (non-bool) `int` before use, UNKNOWN/`NOT_REPRESENTABLE`
   otherwise. 8 new regression tests, including one reproducing the exact crash
   and one for an `inf` arising from legitimate-looking direct-comparison-mode
   input.
4. **B1's shared-value `MathClaim`s fabricated their provenance.**
   `math_claims_for_shared_value_record` hardcoded `artifact_version=1`
   regardless of the real target section's actual version, and never attached a
   `source_ref` at all -- a claim about a since-regenerated artifact would
   silently misreport which version it was actually checking. Fixed:
   `MathClaim.artifact_version` is now `Optional[int] = None` (never a fabricated
   placeholder -- the same "honest not-reported" discipline `ProductionResult`
   already established); `math_claims_for_shared_value_record` gained optional
   `target_ref`/`source_ref: ArtifactRef` parameters, using the real version when
   supplied and honestly `None` when not. This created a second, correctness
   question: `MathClaimCoverageReport.externally_sourced_count` was defined as
   "any claim with `source_ref is None`" -- which would now wrongly count every
   SHARED_VALUE claim whose caller simply didn't pass `source_ref` as
   "externally sourced" (a CALCULATION-only concept: an externally-asserted fact
   with no computable source at all). Fixed by restricting that count to
   `MathClaimKind.CALCULATION` claims specifically -- a SHARED_VALUE claim always
   has a real source SECTION by construction (`check_shared_value_consistency`
   always takes one), so a missing ref there is a reporting gap, never the "no
   source exists" fact the count is for. 3 new regression tests.

All 4 fixes reproduced directly against the actual code before being fixed (not
assumed from reading alone). 22 new regression tests across the 3 affected test
files (all passing); full suite re-confirmed: same 50 pre-existing unrelated
failures byte-identical via `git stash`, 1187 passed (was 1165). `pyflakes`-clean.

## 5.2. Review round 2 (2026-11-06, same branch, uncommitted) -- 3 more real bugs found

A second review pass on the round-1 code caught 3 more concrete bugs, again each
reproduced directly against the real code before fixing:

1. **The value regex read only the number-looking prefix of the right-hand
   side, not the whole thing.** `_occurrence_pattern`'s trailing lookahead was
   `(?!\d)` -- it only rejected a FOLLOWING DIGIT, so `n = 1/2` and `n = 1/3`
   both silently read as `n=1` (the `/2`/`/3` simply never consumed), and
   `n = 1e3` / `n = 1e6` both read as `n=1` the same way (the `e3`/`e6` never
   consumed) -- all four cases then compared as a clean numeric match and
   PASSed, despite being genuinely different values in an unsupported
   notation. This directly contradicted the module's own docstring claim ("a
   right-hand side that isn't a plain... literal number is simply not
   matched, never guessed at") -- the claim was already written, the regex
   just didn't implement it. Fixed: the lookahead now also rejects a
   following `/` (fraction), a following `e`/`E` plus sign/digit (scientific
   notation), and a following `.` plus digit (a second decimal point) --
   `(?![\d/]|[eE][+-]?\d|\.\d)`. An ordinary sentence-ending period right
   after the number with no following digit (`n = 10.`) is deliberately NOT
   rejected, since that is not a continuation of the number at all (verified
   with a dedicated regression test, since this was the main risk of being
   too aggressive). 4 new regression tests (fraction, scientific notation,
   sentence-ending period still matches, double-decimal-point rejected).
2. **`math_claims_for_shared_value_record` dropped unresolved symbols and
   context mismatches during conversion.** The function iterated only
   `record.checked_semantic_ids` and derived each claim's outcome only from
   `record.mismatches` (purely numeric). Two concrete cases silently lost the
   record's real verdict: (a) `n` matches but a second required symbol `k`
   has no source definition at all -- record outcome UNKNOWN, but conversion
   produced a single PASS claim for `n` with `k`'s UNKNOWN nowhere in the
   claims; (b) the numbers agree but `source_assumptions`/`target_assumptions`
   declared and differ -- record outcome FAIL (from `_compare_declared_
   context`), but conversion produced a single PASS claim since the per-symbol
   `mismatches` list genuinely was empty. Fixed by giving
   `SharedValueConsistencyRecord` two new pieces of state that were already
   being computed internally but never exposed: `unresolved_semantic_ids`
   (+ `unresolved_reason_by_id`/`unresolved_detail_by_id`, populated at each
   of the 3 unresolved sites in the main loop) and `context_outcome`/
   `context_detail` (the `_compare_declared_context` result, previously only
   folded into the aggregate detail string). `math_claims_for_shared_value_
   record` now emits three independent kinds of claim -- one UNKNOWN claim
   per unresolved symbol, one PASS/FAIL claim per checked symbol (unchanged),
   and one separate claim for the context comparison when there was anything
   to compare -- so none of the three can hide behind another, and a
   single fallback claim covers the case where none of the three produced
   anything (e.g. no `relevant_symbols` configured, or all marked N/A). 2 new
   regression tests, each reproducing one of the two concrete cases above.
3. **A zero-denominator fraction string still crashed instead of resolving
   UNKNOWN.** `"1/0"` matches `_FRACTION_PATTERN`, so `_coerce_recompute_
   number` reached `Fraction("1/0")`, which raises `ZeroDivisionError` -- a
   type none of the `except (ValueError, TypeError)` clauses around its call
   sites in `_check_recompute` caught (round 1 only added `ZeroDivisionError`
   to the source-ARRAY parsing path's except clause, not these two). Fixed at
   the single conversion function itself (not each call site) by catching
   `ZeroDivisionError` around the `Fraction(...)` call and re-raising as
   `ValueError` -- every existing and future caller that already catches
   `(ValueError, TypeError)` around `_coerce_recompute_number` is covered
   without having to enumerate call sites. 2 new regression tests (`"1/0"` as
   `observed`, `"1/0"` as `expected`).

All 3 fixes reproduced directly against the actual code before being fixed. 8
new regression tests (4 in `test_shared_value_registry.py`, 2 in
`test_math_claims.py`, 2 in `test_constraint_checks.py`), 1195 passed (was
1187), same 50 pre-existing unrelated failures confirmed byte-identical via
`git stash`. `pyflakes`-clean. **Committed as `3a995ec` "stage3 revision2"**
(discovered after the fact -- same pattern as every prior stage's rounds: this
session never ran `git commit`; confirmed via `git log`/`git status` matching
this round's exact diff before starting round 3 below).

## 5.3. Review round 3 (2026-11-06, on top of `3a995ec`, uncommitted) -- 2 more real bugs found

A third review pass -- on the code AFTER round 2's fixes were committed -- found
round 2's own fraction/exponent fix was still incomplete, plus a second,
independent bug in how a FAIL claim's evidence gets attached:

1. **The fraction/product fix only caught an operator with NO space before
   it.** Round 2 fixed `n = 1/2` and `n = 1e3` (operator/exponent glued
   directly to the digit), but `n = 1 / 2` and `n = 10 * 2` (a space before
   the operator) still silently read as `n=1`/`n=10` -- the trailing
   lookahead only inspected the character immediately adjacent to the
   matched digits, and a space there isn't a digit, `/`, `e`/`E`, or a second
   `.`, so it passed through untouched. Confirmed through the full
   `BasicConstraintChecker`/dependency-check path too, not just the bare
   registry function. Fixed: added a second trailing lookahead,
   `(?![ \t]*[\d/*+^-])`, that skips plain spaces/tabs (deliberately NOT
   `\s*`, which also matches a newline -- a markdown `-` bullet starting the
   NEXT line must never be misread as a subtraction continuation of THIS
   line's number; covered by a dedicated regression test) and then checks
   for a following digit or one of `/ * + - ^`. 3 new regression tests
   (spaced fraction, spaced product, and the bullet-on-next-line negative
   case confirming the fix didn't become too aggressive).
2. **A FAIL claim's observed value/raw text/line could point at a DIFFERENT,
   matching usage instead of the one that actually mismatched.** When a
   symbol was restated more than once in the target (e.g. `n = 10` then
   later `n = 20`), `math_claims_for_shared_value_record` built exactly ONE
   claim per checked symbol from `usages[0]` alone, with its outcome decided
   by whether ANY usage appeared in `record.mismatches` -- so the claim could
   read `outcome=FAIL` while `observed_value=10`/`line=1` (the first,
   MATCHING usage), never surfacing the actual `n=20` mismatch at all. Fixed
   by generating one claim PER TARGET USAGE instead of per symbol, each
   outcome recomputed directly against the source definition's value using
   the same comparison and tolerance the registry itself uses to decide
   `mismatches` in the first place (`NUMERIC_MATCH_TOLERANCE`, newly exported
   from `shared_value_registry.py` so the two modules can never silently
   drift onto two different epsilons). 1 new regression test confirming a
   mixed PASS+FAIL pair of claims, each with its own correct
   observed_value/line/raw_text.

Both fixes reproduced directly against the actual code before being fixed
(including the dependency-check path for bug 1). 4 new regression tests (3 in
`test_shared_value_registry.py`, 1 in `test_math_claims.py`); full suite 1199
passed / 50 failed, same pre-existing failures confirmed byte-identical via
`git stash`. `pyflakes`-clean. **Committed as `4aec19f` "stage3 revision3"**
(discovered after the fact -- same pattern as every prior round; confirmed via
`git log`/`git status` before starting round 4 below).

## 5.4. Review round 4 (2026-11-06, on top of `4aec19f`, uncommitted) -- 2 more real bugs found

A fourth review pass found round 3's spaced-operator fix still didn't cover
LaTeX operator commands, plus a second claim_id collision round 3's own fix
for the first collision didn't fully close:

1. **The spaced-operator rejection only covered ASCII operators, not LaTeX
   operator commands.** `n = 10 \times 2` and `n = 10 \cdot 2` both still
   silently read as `n=10` -- `\times`/`\cdot` start with a backslash, which
   round 3's `[ \t]*[\d/*+^-]` class never included. Confirmed through the
   full `BasicConstraintChecker` dependency-check path for the `\times` case
   too. Rather than add `\times`/`\cdot` by name (the third round in a row
   of enumerating one more operator -- the exact whack-a-mole this function
   had already hit twice), the fix is a blanket one: the class now also
   rejects a bare backslash, so ANY LaTeX command immediately following the
   number (across plain spaces/tabs) reads as "not a bare literal," full
   stop -- never requiring this function to know every operator command
   LaTeX has. The accepted tradeoff, stated explicitly in the function's own
   docstring now: a unit annotation with no delimiter of its own
   (`n = 10 \text{m}`) also reads UNKNOWN rather than a confirmed `10` --
   intentional lost recall in exchange for never risking a false PASS on an
   actual operator, the same "never guess" discipline this module has
   applied everywhere else. 2 new regression tests (`\times`, `\cdot`).
2. **Two usages of the same symbol on the SAME line still got a colliding
   claim_id.** Round 3 fixed the *per-symbol* collision (one claim per
   symbol regardless of usage count) by moving to one claim per usage, but
   the new claim_id (`dependency::target::symbol::line<N>`) only
   disambiguated by line number -- two usages on the exact same line (one
   sentence restating a symbol twice, e.g. "n = 10, then n = 20") still
   produced two claims with the identical id. A caller storing claims keyed
   by claim_id (exactly the use case claim_ids exist for) would silently
   drop one of the two. Fixed by adding the usage's position within that
   symbol's usage list (`occ{occurrence_index}`) to the id, guaranteeing
   uniqueness regardless of whether usages share a line. 1 new regression
   test confirming two same-line usages get distinct ids.

Both fixes reproduced directly against the actual code before being fixed. 3
new regression tests (2 in `test_shared_value_registry.py`, 1 in
`test_math_claims.py`); full suite 1202 passed / 50 failed, same pre-existing
failures confirmed byte-identical via `git stash`. `pyflakes`-clean.
**Committed as `c3ae480` "stage3 revision4"** (discovered after the fact --
same pattern as every prior round).

## 5.5. Review round 5 (2026-11-06, on top of `c3ae480`, uncommitted) -- 3 more real bugs found, fixed with a structural rewrite instead of one more blacklist entry

A fifth review found the SAME underlying bug surfacing through 3 more
right-hand-side shapes: `n = 2(3)` (implied multiplication via a glued
parenthetical), `n = 10x` (a letter glued directly to the number, no space --
an exponent/variable shorthand), and `n = 10 × 2` (the Unicode multiplication
sign, distinct from the ASCII `*` round 3 already rejected). The user's own
fix direction named the root cause explicitly: "현재처럼 제외할 연산자를
하나씩 추가하는 방식은 같은 오류를 계속 남깁니다" ("adding excluded operators
one at a time, as done so far, keeps leaving the same error behind") -- after
four straight rounds of extending the same blacklist (bare ASCII operator,
spaced ASCII operator, LaTeX commands via a backslash blanket rule, now
Unicode operators), the right fix is structural, not one more character
class entry.

**Rewrite**: `_occurrence_pattern`'s regex no longer tries to encode "is the
whole right-hand side a bare literal" via trailing lookaheads at all -- it
now only captures the number itself (`-?\d+(?:\.\d+)?`), greedily, with no
suffix condition. A new `_is_continuation_at(text, pos)` runs AFTER each
regex match, in plain Python (not regex), and decides whether what follows
(skipping only plain spaces/tabs, never a newline) means the match is just a
truncated PREFIX of a larger expression rather than the whole right-hand
side:

- a digit, or a `.` immediately followed by a digit (the number itself was
  truncated);
- a backslash (any LaTeX command -- round 4's blanket rule, unchanged);
- an arithmetic/grouping symbol, decided CATEGORICALLY
  (`unicodedata.category(ch) in ("Sm", "Sk")`) rather than by name -- this
  one change is what makes `×`/`÷`/`±`/the Unicode minus sign/the dot
  operator all rejected together, including ones never explicitly tested
  for, closing the Unicode-operator whack-a-mole in one step instead of
  adding `×` and waiting for the next report to add `÷`. A small
  `_NON_SM_CONTINUATION_CHARS` set (`/ * ( - ^ ·`) covers the few ordinary
  ASCII arithmetic/grouping characters Unicode itself does NOT classify as
  "Sm" (verified empirically via `unicodedata.category()`, not assumed:
  `/` and `*` are "Po", `(` is "Ps", `-` is "Pd", and the middle dot `·`, a
  common plain-text stand-in for `\cdot`, is "Po" too -- `^` IS already
  "Sk" and so is covered by the categorical check without needing to be
  in this set) -- this is the one place enumeration remains, kept as
  small as verified necessary rather than growing indefinitely;
- a letter immediately GLUED to the number with zero whitespace (`10x`,
  `10kg`) -- ambiguous by construction (exponent shorthand, variable, or a
  unit with no delimiter), never guessed at. A letter separated from the
  number by at least one space (`n = 5 for this example`, `n = 10 kg`) is
  explicitly NOT a continuation -- this distinction (`glued` in the new
  function) is what keeps the project's existing "ordinary prose after a
  clean number" behavior intact.

No backtracking/retry is attempted when a continuation is detected -- the
match is discarded outright, so e.g. `n = 10.5.3` (round 2's own case) still
yields zero occurrences, never a fallback `10`. Re-verified every prior
round's fix (fraction, exponent, double-decimal, spaced operators, LaTeX
operators, sentence-ending period, the next-line markdown-bullet case, the
backslash-symbol test) still behaves identically under the rewrite -- none of
their existing tests needed changing.

Reproduced directly against the real code before fixing, including the
`\(2(3)\)` case through the full `BasicConstraintChecker` dependency-check
path. 5 new regression tests (paren, glued-letter, Unicode `×`, plus one
Unicode operator -- `÷` -- never previously named in this test file, added
specifically to prove the fix is categorical and not just "× works now").
Full suite 1206 passed / 50 failed, same pre-existing failures confirmed
byte-identical via `git stash`. `pyflakes`-clean. **Committed as `a793768`
"stage3 revision5"** (discovered after the fact -- same pattern as every
prior round).

## 5.6. Review round 6 (2026-11-06, on top of `a793768`, uncommitted) -- 2 more real bugs found

A sixth review found round 5's categorical rewrite still had two gaps: one
semantic (the "glued vs. spaced" letter distinction, correct for ordinary
prose, does not hold inside a LaTeX formula), one a plain missing bracket
type.

1. **A space-separated letter inside math delimiters is multiplication, not
   prose -- `_is_continuation_at` only ever checked whether the letter was
   GLUED (no space) to the number, treating any space-separated letter as
   ordinary prose regardless of context.** `$n = 2 x$` still read as `n=2`,
   discarding `x` entirely, because "2 x" has a space and the function's
   only letter-rejection rule required zero whitespace. Confirmed through
   the full `BasicConstraintChecker` path too. The fix adds LaTeX math-mode
   awareness: a new `_math_spans(text)` detects `$...$`/`\(...\)`/`\[...\]`
   delimited regions (with `\$`-escape awareness, so a literal dollar
   amount in prose never toggles math mode by accident), and
   `_is_continuation_at` gained an `in_math` parameter -- when true, EVERY
   letter following the number is a continuation, glued or not, since
   there is no such thing as a space-separated prose word inside a
   formula. Outside any detected math span, the existing glued-only rule
   is unchanged (`n = 10 kg`, `n = 5 for this example` still read as clean
   restatements). Deliberately NOT reused from `src/agents/latex_
   assembler.py`'s own, more complete `_literal_or_math_mask` (built for a
   different purpose -- Markdown-emphasis masking, with `\texttt{}`
   literal-span handling this module has no need for) -- `src/core` does
   not depend on `src/agents`, and importing across that boundary for one
   narrow fact would be a real layering violation, not a convenience.
2. **`n = 2[3]` (square bracket, glued) still truncated to `n=2`** -- round
   5's fix added `(` to the small ASCII exception set but not its `[`
   sibling (both category "Ps", confirmed via `unicodedata.category()`).
   Added `[` to `_NON_SM_CONTINUATION_CHARS`. The matching close brackets
   (`)`, `]`) are deliberately still NOT included, mirroring round 5's own
   reasoning for `)`: a close bracket right after a number ends a
   parenthetical remark rather than continuing an expression (verified
   with a dedicated regression test, `see item [n = 10] above`).

Both fixes reproduced directly against the actual code before being fixed
(including the dependency-check path for bug 1). 5 new regression tests in
`test_shared_value_registry.py` (the math-mode case, its non-math-mode
negative counterpart, the escaped-`\$` negative case, the glued bracket, and
the close-bracket-is-fine negative case) -- plus every prior round's existing
test re-confirmed unchanged under the new code. Full suite 1211 passed / 50
failed, same pre-existing failures confirmed byte-identical via `git stash`.
`pyflakes`-clean. **Committed as `de7036d` "stage3 revision6"** (discovered
after the fact -- same pattern as every prior round).

## 5.7. Review round 7 (2026-11-06, on top of `de7036d`, uncommitted) -- 4 more real bugs found, all variants of "the math boundary itself was never actually confirmed"

A seventh review found 4 more ways the same underlying class of defect
survives: `_math_spans`/`_is_continuation_at` correctly handle the math
delimiters they recognize, but a math boundary the detector gets WRONG --
mis-segmented, unrecognized, mis-parsed, or simply never resolved -- still
let the content inside fall back to being read as a confirmed plain value.
The user's own fix direction named this precisely: "수식의 우변 전체가 허용된
숫자 리터럴인지 확인하고, 수식 경계나 구문을 해석하지 못하면 UNKNOWN으로
처리해야 합니다... 파싱 실패한 수식이 본문 처리로 넘어가 PASS하는 경로도
막아야 합니다."

1. **`$$n = 2 x$$` (display math) was parsed as TWO empty math spans, not
   one.** `_math_spans` paired each `$` with the very NEXT `$` it found --
   for `$$...$$`, that is the adjacent second `$` of the SAME opening pair,
   producing an empty confirmed span covering just `$$`, then (after the
   real content) another empty span for the closing `$$` -- the actual
   `n = 2 x` content sat entirely OUTSIDE any detected span, so it was
   checked under ordinary (non-math) rules and the space-separated `x`
   read as harmless prose. Fixed: when a `$` is immediately followed by a
   second `$`, search for the next `$$` (not the next single `$`) to
   close it, treating the whole thing as one display-math span.
2. **`\begin{equation}...\end{equation}` (and other math environments)
   were never recognized as math regions at all.** `_math_spans` only
   ever looked for `$`/`\(`/`\[` -- a `\begin{equation}` block's content
   fell back to ordinary rules entirely, letting `n = 2 x` read as prose
   the same way the `$$` case did. Fixed: added environment-delimiter
   recognition (`\begin{name}...\end{name}`) for a short, explicit list of
   common math environments (`_MATH_ENVIRONMENTS`: equation, align,
   gather, multline, eqnarray, math, displaymath, and their starred
   variants) -- deliberately NOT every environment LaTeX has (e.g.
   `array`/`matrix` are excluded on purpose: those normally nest inside an
   already-recognized OUTER math region, so this module doesn't need to
   know them on their own). An unrecognized environment name changes
   nothing, confirmed by a dedicated test.
3. **`$n = 2{3}$` (a glued curly-brace group) still truncated to `n=2`.**
   The small ASCII-exception set already had `(` (round 5) and `[` (round
   6) for the identical reason -- `{` (same Unicode category "Ps") was
   simply the one bracket type not yet added. Added `{` to
   `_NON_SM_CONTINUATION_CHARS`; the matching close brace `}` stays
   unrejected, same reasoning as `)`/`]`.
4. **An unterminated math delimiter (`$n = 2 x` with no closing `$`) let
   its content fall through to ordinary, non-math rules -- exactly the
   "parse-failed formula falls through to body-text processing and
   PASSes" path the user named directly.** The PREVIOUS version of
   `_math_spans` treated a failed search for a closing delimiter as "not
   math after all" and simply moved on -- which is itself the bug, not a
   safe default: not being able to find where a formula ends is not the
   same fact as "this was never a formula." Fixed with a second category
   of span, `unparseable` (returned alongside the existing `confirmed`
   spans) -- an opening delimiter with no resolvable close creates a span
   from that opener to the END of the text (scanning stops there; nothing
   after an unresolved delimiter can be trusted to be outside it either),
   and `find_symbol_value_occurrences` discards ANY match found inside an
   `unparseable` span UNCONDITIONALLY, regardless of what character
   follows it -- unlike the ordinary `in_math` rule, this rejection does
   not depend on what trails the number, since the whole reason for
   distrust is "we don't know where this formula ends," not "something
   specific follows." Self-caught edge case while writing this fix's own
   test for "nothing trails the number at all" (`$n = 2`, no closing `$`
   and nothing after): `_is_within_math_span`'s original `start <= pos <
   end` check is a half-open range, so `match.end() == len(text) == end`
   (the number IS the last characters of the whole string) read as NOT
   inside the span -- fixed by making the right boundary inclusive
   (`pos <= end`), verified safe for `confirmed` spans too (whose `end` is
   always one past a real delimiter character a number match can never
   advance into).

All 4 fixes (plus the self-caught boundary edge case) reproduced directly
against the actual code before being fixed, including the `$$` case through
the full `BasicConstraintChecker` dependency-check path. 6 new regression
tests in `test_shared_value_registry.py`. Full suite 1217 passed / 50 failed,
same pre-existing failures confirmed byte-identical via `git stash`.
`pyflakes`-clean. **Committed as `9e20d7a` "stage3 revision7"** (discovered
after the fact -- same pattern as every prior round).

## 5.8. Review round 8 (2026-11-06, on top of `9e20d7a`, uncommitted) -- 1 more real bug found: a newline inside math mode was treated as the right-hand side's end

An eighth review found round 6's `in_math` check still had a gap: it decided
whether a letter counts as a continuation based on math-mode context, but
the WHITESPACE-SKIPPING step that runs before that decision (`_is_
continuation_at`'s leading loop) only ever skipped spaces/tabs -- never a
newline, inside math mode or not. Round 3 had deliberately made that choice
for a real reason (a markdown `-` bullet starting the next line, OUTSIDE any
math delimiters, must never read as a subtraction continuation of the
previous line's number) -- but that reasoning is specific to ORDINARY prose,
and round 6 already established that prose rules and math-mode rules are
not the same rules.

**Bug**: inside a confirmed math span, LaTeX treats a newline as ordinary
whitespace -- `n = 2\n+ 3` inside `\begin{equation}...\end{equation}` means
exactly `n = 2 + 3`, the same formula as if it were all on one line. But
`_is_continuation_at` stopped at the newline itself (not in `" \t"`), found
a character (`\n`) that matches none of its rejection rules, and returned
"not a continuation" -- so `n = 2` read as a clean, standalone value with the
following `+ 3`/`+ 4` on the next line invisible to the check entirely, and
source/target differing only in that second line's number still PASSed.
Confirmed through the full `BasicConstraintChecker` dependency-check path too.

**Fix**: the leading whitespace-skip in `_is_continuation_at` now skips
newlines too, but ONLY when `in_math` is true -- `whitespace_chars = " \t\n"
if in_math else " \t"`. Outside math mode, the skip set is unchanged, so
round 3's markdown-bullet case is unaffected (re-confirmed by its own
existing test, plus a new one specifically pinning this down against the
newline-skip change). A second new test confirms the fix isn't overly
aggressive: a restatement followed by a newline and then the formula's own
closing delimiter (nothing arithmetic in between, e.g. `$n = 2\n$`) still
reads as a clean value -- "newline is whitespace in math" does not mean "any
multi-line formula is automatically rejected." (A related, pre-existing,
already-accepted conservative behavior was re-confirmed rather than
"fixed": a one-line formula like `\begin{equation}\nn = 2\n\end{equation}`
with nothing between the number and the environment's own `\end{equation}`
tag reads UNKNOWN, not PASS -- because round 4's blanket "any backslash is a
continuation" rule fires on the closing tag itself once the newline before
it is skipped. This is the same recall-for-safety tradeoff round 4 already
named explicitly, not a new regression introduced here, and was not changed
-- adding a special case for "the span's own known closing delimiter"
specifically would be new, unrequested scope, not what this round's report
asked for.)

Reproduced directly against the actual code before being fixed, including
the dependency-check path. 3 new regression tests in `test_shared_value_
registry.py`. Full suite 1220 passed / 50 failed, same pre-existing failures
confirmed byte-identical via `git stash`. `pyflakes`-clean. **Committed as
`90362cc` "stage3 revision8"** (discovered after the fact -- same pattern as
every prior round).

## 5.9. Review round 9 (2026-11-09, on top of `90362cc`, uncommitted) -- 3 reported issues, 4 concrete bugs: two more instances of the SAME enumeration mistake round 8 already named, plus one escape-awareness gap

A ninth review found 3 reported issues resolving to 4 concrete code-level
bugs (the first report bundled CRLF and NBSP together as one row, but they
are two separate gaps in the same function): two were the exact
"hand-enumerated character set instead of a categorical check" mistake
pattern #113 already named for round 8's own newline fix (just for a
different whitespace character this time), and two were new escape-awareness
and unrecognized-syntax gaps.

1. **CRLF line endings inside math mode still hid a continuation.**
   Round 8's fix added `\n` to the math-mode whitespace-skip set, but not
   `\r` -- a CRLF-terminated line (`n = 2\r\n+ 3`) stopped the skip at the
   un-recognized `\r` before ever reaching the operator on the next line,
   the identical failure shape round 8 had just fixed for bare `\n`.
2. **A non-breaking space (NBSP, U+00A0) between the number and an
   operator had the same problem** -- `n = 2 + 3` inside a
   confirmed math span stopped at the NBSP, which was never in the
   hand-picked `" \t\n"` set either.

   Both fixed the SAME way, this time categorically rather than by naming
   `\r` and NBSP individually (stopping the enumeration pattern rather
   than extending it a third time, the lesson pattern #107/#109/#111
   already established for the operator-character side of this function,
   now applied to its whitespace-skipping side too): a new
   `_should_skip_whitespace(ch, in_math)` uses `str.isspace()` -- which is
   already True for CRLF, NBSP, and every other Unicode space separator --
   with newline characters (`\n`/`\r`, tracked separately via
   `_NEWLINE_CHARS` since `isspace()` does not distinguish a newline from
   ordinary spacing) still gated on `in_math`, preserving round 3's and
   round 8's own reasoning for why a newline specifically is treated
   differently outside math mode. Confirmed through the full
   `BasicConstraintChecker` path too.
3. **A LaTeX comment (`%`) right after the number let whatever followed
   it on the next line slip through undetected.** `n = 2 % comment\n+ 3`
   -- the `%` starts a comment that hides everything to the end of that
   line, so this deterministic character-scan genuinely cannot know what
   the formula continues into past it; the old code had no rule for `%`
   at all, so it simply read "2" as a clean value with the comment (and
   whatever real continuation hides behind it) invisible to the check.
   Fixed with a blanket rejection, the same "can't verify it, don't
   guess" response as the existing backslash rule -- deliberately NOT an
   attempt to actually parse/skip the comment to see what's on the next
   line (the user's own two offered options were "handle the comment, or
   just call it UNKNOWN"; the simpler, more conservative option was
   chosen, consistent with every other ambiguous-syntax decision in this
   function).
4. **An escaped `\$` INSIDE a real formula could prematurely close that
   formula's own math span.** `_math_spans`'s main scan loop already
   treated `\$` as literal text when deciding whether a bare `$` OPENS a
   new span (round 6's own escape-awareness) -- but the separate search
   for a span's CLOSING delimiter still used a plain `str.find("$", ...)`
   call, which has no escape-awareness at all, so an escaped `\$`
   appearing inside the formula (e.g. `$\text{price} = \$5, n = 2 x$`)
   could be mistaken for that formula's own closing `$`, ending the span
   right there and leaving the REAL remainder of the formula (`5, n = 2
   x$`) to be read as ordinary prose, where the space-separated `x`
   wrongly passed as narration. Fixed with a new `_find_unescaped(text,
   needle, start)` helper (walks forward exactly like the main scan loop
   already does, skipping `\X` two-character units) used for both the
   single-`$` and double-`$$` closing searches.

Reproduced directly against the actual code before being fixed, all four
confirmed through the full `BasicConstraintChecker` dependency-check path.
4 new regression tests in `test_shared_value_registry.py`. Full suite 1224
passed / 50 failed, same pre-existing failures confirmed byte-identical via
`git stash`. `pyflakes`-clean.

## 6. Regression

Pre-existing baseline (before any of B1-B3 existed): 1121 passed / 50 failed.
First pass: 44 new tests (`test_shared_value_registry.py` 14, `test_math_claims.py`
6, `test_dependency_verifier.py` +6, `test_dependency_constraint_provider.py` +2,
`test_constraint_checks.py` +11 B2 operations +3 DEPENDENCY must_reuse/
must_be_consistent_with wiring) -> 1165 passed. §5.1: +22 -> 66 new, 1187
passed. §5.2: +8 -> 74 new, 1195 passed. §5.3: +4 -> 78 new, 1199 passed.
§5.4: +3 -> 81 new, 1202 passed. §5.5: +4 (all in `test_shared_value_
registry.py`) -> 85 new, 1206 passed. §5.6: +5 (all in `test_shared_value_
registry.py`) -> 90 new, 1211 passed. §5.7: +6 (all in `test_shared_value_
registry.py`) -> 96 new, 1217 passed. §5.8: +3 (all in `test_shared_value_
registry.py`) -> 99 new, 1220 passed. §5.9: +4 (all in `test_shared_value_
registry.py`) -> 103 new tests total, 1224 passed, all passing. The exact
same 50 pre-existing unrelated failures confirmed byte-identical (not just
count) via `git stash` before/after comparison, across all nine rounds --
same discipline every prior stage in this revision has applied.
`pyflakes`-clean throughout.

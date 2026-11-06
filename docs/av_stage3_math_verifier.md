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
`git stash`. `pyflakes`-clean.

## 6. Regression

First pass: 44 new tests (`test_shared_value_registry.py` 14, `test_math_claims.py`
6, `test_dependency_verifier.py` +6, `test_dependency_constraint_provider.py` +2,
`test_constraint_checks.py` +11 B2 operations +3 DEPENDENCY must_reuse/
must_be_consistent_with wiring). After §5.1's review-fix round: 22 more (10 in
`test_shared_value_registry.py`, 3 in `test_math_claims.py`, 9 in
`test_constraint_checks.py`), 66 new tests. After §5.2's second review-fix
round: 8 more (4 in `test_shared_value_registry.py`, 2 in `test_math_claims.py`,
2 in `test_constraint_checks.py`), 74 new tests total, all passing. Full suite:
1195 passed / 50 failed, the exact same 50 pre-existing unrelated failures
confirmed byte-identical (not just count) via `git stash` before/after
comparison, across all three rounds -- same discipline every prior stage in
this revision has applied. `pyflakes`-clean throughout.

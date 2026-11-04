"""Basic constraint checking over the Stage 2 artifact graph (Stage 3 of the
"Revision 2026.10.15" plan, checklist item 2: "참조·필수 구성요소·숫자
재계산부터 구현하고, 필요한 좁은 제약군에만 solver를 적용한다").

``BasicConstraintChecker`` implements the Stage 1 ``Validator`` Protocol
(``src/core/interfaces.py``) -- it is not a new, parallel interface. It
operates purely over ``ArtifactGraphStore``/``ArtifactRef``/``Constraint``,
decoupled from the legacy ``ContentBlock``/``LayoutBlock`` domain that
``adapters.TextAgentValidatorAdapter`` wraps: the two ``Validator``
implementations check different things at different layers and neither
replaces the other.

Covers two of ``ConstraintKind``'s four families this stage:

- ``SCHEMA_CONTRACT``: both "참조 무결성" (reference integrity) and "필수
  구성요소" (required-component presence) collapse to the same mechanism
  here -- every ``ArtifactRef`` named in ``Constraint.applies_to`` must
  resolve to a real, stored ``(artifact_id, version)`` **of the exact
  ``kind`` it claims** -- a Stage 3 review found the first pass only
  checked id/version existence, so a constraint requiring a TABLE
  silently PASSed against a real, stored DEFINITION that merely happened
  to share the same id/version. Listing a required component's ref is
  indistinguishable, at this level, from listing a referenced target's
  ref; both are "this must exist, as this kind". **This cannot detect a
  reference that exists in an artifact's raw *content* but was never
  declared in ``applies_to``** -- that would be real reference-scanning,
  which Stage 2 already deferred explicitly
  (``docs/stage2_artifact_graph.md``, "하지 않은 것": "실제 레퍼런스 스캔
  기반 INFERRED edge 자동 탐지"). This checker only ever answers "are the
  artifacts this constraint says must exist actually there, as the kind
  claimed", not "does this artifact's content actually cite them".
- ``RECOMPUTE``: **two distinct modes, named precisely because a Stage 3
  review found the original single mode was accurately described only as
  numeric *comparison*, not *recomputation*:**

  1. **Direct comparison** (the original mode): caller supplies both
     ``expected``/``observed`` already computed, via
     ``context["recompute"][constraint_id] = {"expected": ..., "observed":
     ..., "tolerance": ..., "source_refs": [...]}``. This checker only
     compares the two numbers -- it does not compute ``expected`` itself,
     so describing this mode alone as "숫자 재계산" (numeric
     *recomputation*) would overstate what it does; "숫자 비교" (numeric
     comparison) is the accurate description.
  2. **Recompute from a structured source artifact** (added in the same
     review, to actually earn the "재계산" claim): caller supplies
     ``{"source_ref": ArtifactRef, "operation": "sum", "observed": ...,
     "tolerance": ...}``. This checker itself reads
     ``source_ref``'s stored content from the ``ArtifactGraphStore``,
     parses it as a JSON array of numbers, and computes ``expected`` via
     the named operation -- today only ``"sum"`` (deliberately one
     narrow, concrete case, not a general expression evaluator; any other
     operation name is UNKNOWN/NOT_REPRESENTABLE). ``expected`` is
     genuinely recomputed from a versioned artifact here, not merely
     handed in.

  Both modes resolve every artifact ref the verdict actually consulted --
  ``constraint.applies_to`` merged with whichever mode-specific ref(s) the
  caller additionally supplies (direct mode: ``source_refs``; recompute
  mode: ``source_ref`` itself) -- via ``_resolve_input_refs``, which
  existence-checks each one *right now* and splits the result: resolved
  refs go on ``Evidence.input_versions``, refs whose exact
  ``(artifact_id, version)`` does not resolve go on
  ``Evidence.missing_inputs``, as the full ref, never just an id. Three
  Stage 3 reviews shaped this: (1) merging ``source_refs`` only, not
  ``applies_to`` too, left a caller who declared the source solely via
  ``applies_to`` (a legitimate choice -- the one mechanism SCHEMA_CONTRACT
  itself uses) with an empty ``input_versions``, so a later version
  change on that source never invalidated the verdict; (2) routing every
  merged ref onto ``input_versions`` unconditionally, with no existence
  check, meant declaring the *same not-yet-created* source via both
  ``applies_to`` and ``source_ref`` produced a ref required to
  simultaneously exist and not exist -- self-contradictory, never live;
  (3) tracking a not-found ref by id alone (not the full ref, version
  included) let *any other* version of that id count as "no longer
  missing," even one that does nothing to satisfy what was actually
  required. ``Validator.check(..., context=...)`` is the same extension
  point Stage 1's ``TextAgentValidatorAdapter`` already uses to carry
  whatever a given checker needs beyond the bare ``Constraint`` list.

``CSP_SMT`` and ``SYMBOLIC`` are deliberately left unimplemented this
stage: no concrete constraint instance needing Z3/SymPy exists yet, and
neither library is a dependency of this project
(``requirements.txt``). Both kinds always resolve to
``CheckOutcome.UNKNOWN``/``UnknownReason.NOT_REPRESENTABLE`` with a detail
string naming this explicitly, rather than silently reporting PASS or
raising ImportError. Tracked as explicit follow-up work in Notion
Progress.ver2 under Stage 3, not forgotten.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from src.core.artifact_graph import ArtifactGraphStore
from src.core.dependency_mapping import CrossSectionObligation, MappingStatus
from src.core.dependency_verifier import (
    _verify_must_precede,
    _verify_must_reference,
    _verify_shared_value_relation,
)
from src.core.schemas import (
    ArtifactRef,
    CheckOutcome,
    Constraint,
    ConstraintKind,
    Evidence,
    UnknownReason,
)
from src.dataset.schemas import DependencyRelation

def _product(numbers: List[float]) -> float:
    result = 1.0
    for n in numbers:
        result *= n
    return result


def _mean(numbers: List[float]) -> float:
    return sum(numbers) / len(numbers)


def _difference(numbers: List[float]) -> float:
    return numbers[0] - sum(numbers[1:])


def _ratio(numbers: List[float]) -> float:
    return numbers[0] / numbers[1]


def _percentage(numbers: List[float]) -> float:
    return (numbers[0] / numbers[1]) * 100.0


def _percentage_change(numbers: List[float]) -> float:
    old, new = numbers
    return ((new - old) / old) * 100.0


def _power(numbers: List[float]) -> float:
    base, exponent = numbers
    return base ** exponent


# Revision (Additional verifiers) plan, Fix B checklist item B2: "기존 합계
# 재계산을 필요한 연산으로 확장" -- the plan's own 사칙연산/합계/비율·백분율
# table row, plus "power" for Gold Labeling's sample_001 (0.998^999
# miscalculated). Deliberately still one narrow, named set of concrete
# operations, never a general expression evaluator -- "값 대입" (value
# substitution) in the plan's text is satisfied by these operations
# themselves taking already-substituted source values, not by adding an
# arbitrary-formula eval path (which would be a real code-execution
# surface over source content, not a narrow deterministic tool). Any
# operation name not listed here is UNKNOWN/NOT_REPRESENTABLE, exactly
# like an unimplemented ConstraintKind.
_RECOMPUTE_OPERATIONS = {
    "sum": sum,
    "mean": _mean,
    "product": _product,
    "difference": _difference,
    "ratio": _ratio,
    "percentage": _percentage,
    "percentage_change": _percentage_change,
    "power": _power,
}

# None = any length >= 1 (sum/mean/product: a reduction over the whole
# array). An explicit int = exactly that many operands -- ratio/
# percentage/percentage_change/power read as binary operations on
# [a, b] (percentage_change and power are order-sensitive: [old, new]
# and [base, exponent] respectively), not well-defined for any other
# arity, so a mismatched count is UNKNOWN/NOT_REPRESENTABLE ("입력 부족"),
# never silently padded/truncated or crashed on.
_RECOMPUTE_OPERATION_ARITY: Dict[str, Optional[int]] = {
    "sum": None,
    "mean": None,
    "product": None,
    "difference": None,
    "ratio": 2,
    "percentage": 2,
    "percentage_change": 2,
    "power": 2,
}

# A hard bound on how many source numbers a single recompute call will
# read, not a preemptive wall-clock timeout -- every operation above is a
# single linear pass with no recursion, so the real risk this guards
# against is a pathologically large stored artifact, not a slow
# algorithm. Reported as UnknownReason.SOLVER_TIMEOUT (the plan's "실행
# 시간 한도") since that is the closest honest fit in the existing
# vocabulary; the docstring is deliberately exact about what is and isn't
# actually enforced, per this project's own standing rule against
# claiming a stronger guarantee than the literal code provides.
_MAX_RECOMPUTE_INPUTS = 10_000

_FRACTION_PATTERN = re.compile(r"^[+-]?\d+/\d+$")


def _coerce_recompute_number(raw: Any) -> float:
    """B2's "분수" handling: accepts a plain JSON number, a decimal
    string (``"3.5"``), or a simple ``"a/b"`` fraction string
    (``"3/4"``, ``"-1/2"``) -- never a percentage string (``"12%"``):
    whether that means ``0.12`` or ``12`` is genuinely ambiguous without
    more context than a source array gives, so it is deliberately left
    unparsed (a parse error, same as any other unsupported shape) rather
    than guessed; the ``"percentage"``/``"percentage_change"``
    *operations* exist instead for percentage computation. JSON
    booleans are explicitly rejected even though ``isinstance(True,
    int)`` is true in Python -- without this, a stray ``true``/``false``
    in the source array would silently become ``1.0``/``0.0``."""
    if isinstance(raw, bool):
        raise ValueError(f"boolean {raw!r} is not a recompute input")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if _FRACTION_PATTERN.match(stripped):
            return float(Fraction(stripped))
        return float(stripped)
    raise ValueError(f"{raw!r} is not a number, decimal string, or 'a/b' fraction string")


# The three constraint kinds this checker can actually evaluate; CSP_SMT
# and SYMBOLIC still always fall through to the final "not implemented"
# branch in ``check()``. Kept as an explicit set (not an if/elif chain
# spread across the method) so the one place that decides "do we have a
# narrow-enough mechanism for this kind at all" is unambiguous.
#
# DEPENDENCY itself only has a REAL mechanism for two of the four
# DependencyRelation values (must_reference/must_precede, see
# ``_check_dependency``) -- must_reuse/must_be_consistent_with resolve to
# UNKNOWN/NOT_REPRESENTABLE inside that method, the same "kind recognized,
# no checker for this specific case yet" pattern CSP_SMT/SYMBOLIC already
# use at the whole-kind level, just one level more granular (per relation,
# not per kind) because this one kind covers four genuinely different
# relations instead of one.
_IMPLEMENTED_KINDS = frozenset(
    {ConstraintKind.SCHEMA_CONTRACT, ConstraintKind.RECOMPUTE, ConstraintKind.DEPENDENCY}
)


class BasicConstraintChecker:
    """``Validator`` over ``SCHEMA_CONTRACT`` (reference/required-component
    existence) and ``RECOMPUTE`` (numeric comparison) constraints, reading
    one ``ArtifactGraphStore`` branch.

    ``config``/``set_enabled`` follow exactly the same read-only-snapshot-
    vs-live-mutator split as ``adapters.TextAgentValidatorAdapter`` (see
    that class's and ``Validator``'s own docstrings for why a bare
    ``.config`` must never be assumed mutable): mutating the dict
    ``config`` returns has no effect; only ``set_enabled()`` changes what
    the next ``check()`` call actually runs.
    """

    def __init__(self, store: ArtifactGraphStore) -> None:
        self._store = store
        self._enabled: Dict[str, bool] = {
            ConstraintKind.SCHEMA_CONTRACT.value: True,
            ConstraintKind.RECOMPUTE.value: True,
            ConstraintKind.DEPENDENCY.value: True,
        }

    @property
    def config(self) -> Dict[str, bool]:
        """Read-only snapshot -- mutating the returned dict does not affect
        which checks ``check()`` runs next; see ``Validator.config``'s own
        docstring in ``interfaces.py``."""
        return dict(self._enabled)

    def set_enabled(self, kind: ConstraintKind, enabled: bool) -> None:
        key = kind.value if isinstance(kind, ConstraintKind) else kind
        if key not in self._enabled:
            raise ValueError(
                f"Unknown/unimplemented constraint kind for this checker: {key!r}. "
                f"Togglable kinds: {sorted(self._enabled)}."
            )
        self._enabled[key] = enabled

    def check(
        self,
        artifact: ArtifactRef,
        constraints: List[Constraint],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Evidence]:
        """Answers for every ``constraint`` given, matching ``Validator``'s
        contract in ``interfaces.py`` -- never silently drops one, and
        never consults previously recorded ``Evidence`` as a shortcut (see
        this module's docstring: a stale result is never reused in place
        of actually re-checking)."""
        context = context or {}
        recompute_inputs: Dict[str, Dict[str, Any]] = context.get("recompute", {})
        dependency_inputs: Dict[str, Dict[str, Any]] = context.get("dependency", {})

        evidence: List[Evidence] = []
        for constraint in constraints:
            if constraint.kind not in _IMPLEMENTED_KINDS:
                evidence.append(
                    Evidence(
                        constraint_id=constraint.constraint_id,
                        outcome=CheckOutcome.UNKNOWN,
                        checked_artifact=artifact,
                        detail=(
                            f"constraint kind {constraint.kind.value!r} has no solver-backed "
                            "checker implemented yet (schema_contract, recompute, and dependency "
                            "are implemented; csp_smt/symbolic are tracked as follow-up work)"
                        ),
                        unknown_reason=UnknownReason.NOT_REPRESENTABLE,
                    )
                )
                continue

            if not self._enabled[constraint.kind.value]:
                evidence.append(
                    Evidence(
                        constraint_id=constraint.constraint_id,
                        outcome=CheckOutcome.UNKNOWN,
                        checked_artifact=artifact,
                        detail=f"trigger disabled for this run (kind={constraint.kind.value!r})",
                        unknown_reason=UnknownReason.NOT_CHECKED,
                    )
                )
                continue

            if constraint.kind == ConstraintKind.SCHEMA_CONTRACT:
                evidence.append(self._check_schema_contract(constraint, artifact))
            elif constraint.kind == ConstraintKind.RECOMPUTE:
                evidence.append(
                    self._check_recompute(constraint, artifact, recompute_inputs.get(constraint.constraint_id))
                )
            else:
                evidence.append(
                    self._check_dependency(constraint, artifact, dependency_inputs.get(constraint.constraint_id))
                )

        return evidence

    def _check_schema_contract(self, constraint: Constraint, artifact: ArtifactRef) -> Evidence:
        """PASS iff every ``ArtifactRef`` in ``constraint.applies_to``
        resolves to a real stored ``(artifact_id, version)`` *of the exact
        kind it claims* right now; FAIL lists exactly which ones do not,
        and why (not found vs. wrong kind) -- a Stage 3 review found the
        original version checked only id/version existence, so a
        constraint requiring kind=TABLE silently PASSed against a real,
        stored artifact of a *different* kind that merely shared the same
        id/version. An empty ``applies_to`` is vacuously PASS -- a
        schema-contract constraint naming nothing to check for existence
        has nothing it can fail on.

        A ref that resolved (found, right kind, *or* found with the wrong
        kind -- it still exists, just incorrectly) goes on
        ``Evidence.input_versions``: a later review found this check's own
        correctness also depends on that version staying the current one,
        PASS or FAIL alike. A ref whose exact ``(artifact_id, version)``
        was not found -- regardless of whether *other* versions of that
        same artifact_id exist -- goes on ``Evidence.missing_inputs``
        instead, as the full ``ArtifactRef`` (never just the id): a second
        review found that recording a not-found ref as if it were some
        trustworthy existing version inverted
        ``ArtifactGraphStore.live_evidence``'s invalidation direction for
        exactly this case, and a third review found that dropping the
        version number (id-only) wrongly let *any* other version of that
        id (e.g. ``table@1``, already present) count as "no longer
        missing" for a FAIL that specifically required ``table@2`` -- see
        ``Evidence.missing_inputs``'s own docstring.
        """
        unresolved_detail: List[str] = []
        live_refs: List[ArtifactRef] = []
        missing_refs: List[ArtifactRef] = []
        for ref in constraint.applies_to:
            stored = self._store.get_version(ref.artifact_id, ref.version)
            if stored is None:
                unresolved_detail.append(f"{ref.artifact_id}@v{ref.version} (not found)")
                missing_refs.append(ref)
            elif stored.ref.kind != ref.kind:
                unresolved_detail.append(
                    f"{ref.artifact_id}@v{ref.version} (expected kind={ref.kind.value!r}, "
                    f"actually stored as kind={stored.ref.kind.value!r})"
                )
                live_refs.append(ref)
            else:
                live_refs.append(ref)

        if unresolved_detail:
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.FAIL,
                checked_artifact=artifact,
                detail=f"missing/unresolved required artifact(s): {', '.join(unresolved_detail)}",
                input_versions=live_refs,
                missing_inputs=missing_refs,
            )
        return Evidence(
            constraint_id=constraint.constraint_id,
            outcome=CheckOutcome.PASS,
            checked_artifact=artifact,
            detail="every applies_to artifact resolved to a real stored version of the claimed kind",
            input_versions=live_refs,
        )

    def _resolve_input_refs(
        self, constraint: Constraint, extra: List[ArtifactRef]
    ) -> Tuple[List[ArtifactRef], List[ArtifactRef]]:
        """``constraint.applies_to`` plus whatever extra refs a RECOMPUTE
        mode consulted beyond it (``source_refs``/``source_ref``), deduped
        by ``(artifact_id, version)`` and then split by whether each one
        actually resolves in the store **right now**: a resolved ref goes
        toward ``Evidence.input_versions`` (live iff it stays the current,
        non-stale version); a ref whose exact version does not resolve
        goes toward ``Evidence.missing_inputs`` (live iff that exact
        version still does not exist) -- see that field's docstring for
        why putting an unresolved ref on ``input_versions`` instead is
        backwards.

        A second Stage 3 review found that a RECOMPUTE constraint
        declaring its source only via ``applies_to`` -- the same
        mechanism SCHEMA_CONTRACT uses, instead of duplicating it into
        ``context["recompute"][...]["source_refs"]`` -- ended up with an
        empty ``Evidence.input_versions`` under the original,
        ``source_refs``-only merge, so a PASS kept reading as live
        evidence forever even after that source artifact changed version.
        A third review found the fix after *that* (merging unconditionally
        into ``input_versions``) broke the opposite case: declaring the
        *same not-yet-created* source via both ``applies_to`` and
        ``source_ref``/``source_refs`` produced an ``Evidence`` requiring
        that ref to simultaneously exist (``input_versions``) and not
        exist (``missing_inputs``) -- self-contradictory, so it could
        never be live. Existence-checking every ref here, regardless of
        which of the two APIs a caller declared it through, is what
        removes that contradiction: a given ref ends up on exactly one of
        the two lists, never both.
        """
        combined = list(constraint.applies_to)
        seen = {(ref.artifact_id, ref.version) for ref in combined}
        for ref in extra:
            key = (ref.artifact_id, ref.version)
            if key not in seen:
                combined.append(ref)
                seen.add(key)

        found: List[ArtifactRef] = []
        missing: List[ArtifactRef] = []
        for ref in combined:
            if self._store.get_version(ref.artifact_id, ref.version) is not None:
                found.append(ref)
            else:
                missing.append(ref)
        return found, missing

    def _check_recompute(
        self, constraint: Constraint, artifact: ArtifactRef, values: Optional[Dict[str, Any]]
    ) -> Evidence:
        """Dispatches to recompute-from-source (when ``values`` carries
        ``source_ref``+``operation``) or direct comparison (when it
        carries ``expected`` instead) -- see this module's docstring for
        what each mode actually does. Missing/incomplete ``values`` is
        UNKNOWN/NOT_CHECKED, not a silent FAIL or an exception -- this
        checker was simply never handed what it needs, which is a
        different fact than the comparison having run and mismatched."""
        if not values or "observed" not in values:
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=(
                    "no observed value supplied in "
                    f"context['recompute'][{constraint.constraint_id!r}]"
                ),
                unknown_reason=UnknownReason.NOT_CHECKED,
            )

        observed = float(values["observed"])
        tolerance = float(values.get("tolerance", 0.0))
        tolerance_mode = values.get("tolerance_mode", "absolute")
        round_to = values.get("round_to")

        if "source_ref" in values and "operation" in values:
            return self._check_recompute_from_source(
                constraint, artifact, values, observed, tolerance, tolerance_mode, round_to
            )

        if "expected" in values:
            expected = float(values["expected"])
            raw_source_refs = values.get("source_refs") or []
            source_refs = [
                ref if isinstance(ref, ArtifactRef) else ArtifactRef.model_validate(ref)
                for ref in raw_source_refs
            ]
            input_versions, missing_inputs = self._resolve_input_refs(constraint, source_refs)
            return self._recompute_verdict(
                constraint, artifact, expected=expected, observed=observed, tolerance=tolerance,
                input_versions=input_versions, missing_inputs=missing_inputs, mode="comparison",
                tolerance_mode=tolerance_mode, round_to=round_to,
            )

        return Evidence(
            constraint_id=constraint.constraint_id,
            outcome=CheckOutcome.UNKNOWN,
            checked_artifact=artifact,
            detail=(
                "no expected value or source_ref/operation supplied in "
                f"context['recompute'][{constraint.constraint_id!r}]"
            ),
            unknown_reason=UnknownReason.NOT_CHECKED,
        )

    def _check_recompute_from_source(
        self,
        constraint: Constraint,
        artifact: ArtifactRef,
        values: Dict[str, Any],
        observed: float,
        tolerance: float,
        tolerance_mode: str = "absolute",
        round_to: Optional[int] = None,
    ) -> Evidence:
        """Actually recomputes ``expected`` from a stored source
        artifact's content -- the mode that earns "숫자 재계산" rather
        than mere "숫자 비교" (see this module's docstring)."""
        raw_ref = values["source_ref"]
        source_ref = raw_ref if isinstance(raw_ref, ArtifactRef) else ArtifactRef.model_validate(raw_ref)
        operation = values["operation"]

        if operation not in _RECOMPUTE_OPERATIONS:
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=(
                    f"unsupported recompute operation {operation!r} -- only "
                    f"{sorted(_RECOMPUTE_OPERATIONS)} implemented this stage"
                ),
                unknown_reason=UnknownReason.NOT_REPRESENTABLE,
            )

        stored_source = self._store.get_version(source_ref.artifact_id, source_ref.version)
        if stored_source is None:
            # source_ref's exact version was not found -- let
            # _resolve_input_refs re-derive that itself (rather than
            # hardcoding it into missing_inputs here) so any OTHER
            # applies_to-declared ref that also happens to be missing is
            # caught too, and so declaring this same source via BOTH
            # applies_to and source_ref never produces a ref on both
            # input_versions and missing_inputs at once (a third review's
            # exact repro: that self-contradiction made the Evidence
            # un-live no matter what).
            input_versions, missing_inputs = self._resolve_input_refs(constraint, [source_ref])
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=f"source artifact {source_ref.artifact_id}@v{source_ref.version} not found",
                unknown_reason=UnknownReason.MISSING_DEPENDENCY,
                input_versions=input_versions,
                missing_inputs=missing_inputs,
            )
        if stored_source.content is None:
            # The artifact exists (at this version) but was stored with no
            # content -- a real, trackable version, so source_ref itself
            # resolves to input_versions here; only some OTHER
            # applies_to-declared ref could still land on missing_inputs.
            input_versions, missing_inputs = self._resolve_input_refs(constraint, [source_ref])
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=f"source artifact {source_ref.artifact_id}@v{source_ref.version} has no content",
                unknown_reason=UnknownReason.MISSING_DEPENDENCY,
                input_versions=input_versions,
                missing_inputs=missing_inputs,
            )

        try:
            raw_numbers = json.loads(stored_source.content)
            if not isinstance(raw_numbers, list):
                raise ValueError("not a JSON array")
            numbers = [_coerce_recompute_number(n) for n in raw_numbers]
        except (ValueError, TypeError, ZeroDivisionError, json.JSONDecodeError):
            input_versions, missing_inputs = self._resolve_input_refs(constraint, [source_ref])
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=(
                    f"source artifact {source_ref.artifact_id}@v{source_ref.version}'s content is "
                    "not a JSON array of numbers (plain numbers, decimal strings, or 'a/b' fraction "
                    "strings) -- cannot recompute"
                ),
                unknown_reason=UnknownReason.NOT_REPRESENTABLE,
                input_versions=input_versions,
                missing_inputs=missing_inputs,
            )

        if len(numbers) > _MAX_RECOMPUTE_INPUTS:
            input_versions, missing_inputs = self._resolve_input_refs(constraint, [source_ref])
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=(
                    f"source artifact {source_ref.artifact_id}@v{source_ref.version} has "
                    f"{len(numbers)} values, over this checker's {_MAX_RECOMPUTE_INPUTS}-value "
                    "execution bound -- not recomputed"
                ),
                unknown_reason=UnknownReason.SOLVER_TIMEOUT,
                input_versions=input_versions,
                missing_inputs=missing_inputs,
            )

        required_arity = _RECOMPUTE_OPERATION_ARITY[operation]
        if required_arity is not None and len(numbers) != required_arity:
            input_versions, missing_inputs = self._resolve_input_refs(constraint, [source_ref])
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=(
                    f"operation {operation!r} requires exactly {required_arity} source value(s), "
                    f"source artifact {source_ref.artifact_id}@v{source_ref.version} has "
                    f"{len(numbers)} -- insufficient input, not guessed at"
                ),
                unknown_reason=UnknownReason.NOT_REPRESENTABLE,
                input_versions=input_versions,
                missing_inputs=missing_inputs,
            )
        if required_arity is None and not numbers:
            input_versions, missing_inputs = self._resolve_input_refs(constraint, [source_ref])
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=(
                    f"source artifact {source_ref.artifact_id}@v{source_ref.version} has no values "
                    f"-- cannot recompute {operation!r} over an empty input"
                ),
                unknown_reason=UnknownReason.NOT_REPRESENTABLE,
                input_versions=input_versions,
                missing_inputs=missing_inputs,
            )

        try:
            expected = float(_RECOMPUTE_OPERATIONS[operation](numbers))
        except (ZeroDivisionError, ValueError, OverflowError, TypeError) as exc:
            # Division by zero (ratio/percentage/percentage_change), a
            # negative base with a fractional exponent (power -> complex,
            # which float() then rejects), or an overflowing power -- all
            # genuinely "cannot represent this result", never a crash and
            # never silently coerced to some placeholder number.
            input_versions, missing_inputs = self._resolve_input_refs(constraint, [source_ref])
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=f"operation {operation!r} on {numbers!r} is not representable as a real number ({exc})",
                unknown_reason=UnknownReason.NOT_REPRESENTABLE,
                input_versions=input_versions,
                missing_inputs=missing_inputs,
            )
        input_versions, missing_inputs = self._resolve_input_refs(constraint, [source_ref])
        return self._recompute_verdict(
            constraint, artifact, expected=expected, observed=observed, tolerance=tolerance,
            input_versions=input_versions, missing_inputs=missing_inputs,
            mode=f"recomputed via {operation!r} from {source_ref.artifact_id}@v{source_ref.version}",
            tolerance_mode=tolerance_mode, round_to=round_to,
        )

    def _recompute_verdict(
        self,
        constraint: Constraint,
        artifact: ArtifactRef,
        *,
        expected: float,
        observed: float,
        tolerance: float,
        input_versions: List[ArtifactRef],
        missing_inputs: Optional[List[ArtifactRef]] = None,
        mode: str,
        tolerance_mode: str = "absolute",
        round_to: Optional[int] = None,
    ) -> Evidence:
        """Compares ``expected``/``observed`` -- optionally rounded first
        (B2's "반올림 자리"), with either an absolute tolerance (the
        original, default behavior) or a tolerance relative to
        ``abs(expected)`` (B2's "절대/상대 오차"). An unrecognized
        ``tolerance_mode`` is UNKNOWN/NOT_REPRESENTABLE, never silently
        treated as "absolute" -- a caller's typo in this parameter must
        not quietly change what tolerance was actually applied."""
        missing_inputs = missing_inputs or []
        if tolerance_mode not in ("absolute", "relative"):
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=f"unsupported tolerance_mode {tolerance_mode!r} -- only 'absolute'/'relative' implemented",
                unknown_reason=UnknownReason.NOT_REPRESENTABLE,
                input_versions=input_versions,
                missing_inputs=missing_inputs,
            )

        compare_expected, compare_observed = expected, observed
        rounding_note = ""
        if round_to is not None:
            compare_expected = round(expected, round_to)
            compare_observed = round(observed, round_to)
            rounding_note = f", rounded to {round_to} decimal place(s)"

        effective_tolerance = tolerance
        if tolerance_mode == "relative":
            effective_tolerance = tolerance * abs(compare_expected)

        diff = abs(compare_expected - compare_observed)
        tolerance_note = (
            f"tolerance={tolerance} ({tolerance_mode})"
            if tolerance_mode == "absolute"
            else f"tolerance={tolerance} relative to expected -> {effective_tolerance}"
        )
        if diff <= effective_tolerance:
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.PASS,
                checked_artifact=artifact,
                detail=(
                    f"observed={compare_observed} matches expected={compare_expected} ({mode}){rounding_note} "
                    f"within {tolerance_note}"
                ),
                input_versions=input_versions,
                missing_inputs=missing_inputs,
            )
        return Evidence(
            constraint_id=constraint.constraint_id,
            outcome=CheckOutcome.FAIL,
            checked_artifact=artifact,
            detail=(
                f"observed={compare_observed} does not match expected={compare_expected} ({mode}){rounding_note} "
                f"(|diff|={diff} > {tolerance_note})"
            ),
            input_versions=input_versions,
            missing_inputs=missing_inputs,
        )

    def _check_dependency(
        self, constraint: Constraint, artifact: ArtifactRef, values: Optional[Dict[str, Any]]
    ) -> Evidence:
        """DEPENDENCY constraint check -- Fix C (the separate "Revision
        (Additional verifiers)" plan), checklist item C4's "dependency
        위반 evidence를 기존 check·patch·replan에 연결한다".

        Caller supplies the relation-specific parameters via
        ``context["dependency"][constraint.constraint_id]`` -- the same
        extension-point convention ``_check_recompute`` already established
        for ``context["recompute"][...]`` (see this module's own
        docstring). Expected keys: ``relation`` (a ``DependencyRelation``
        value), ``allowed_labels`` (``List[str]``, for ``must_reference``),
        ``order_hint`` (``[source_index, target_index]`` or ``None``, for
        ``must_precede``). ``constraint.applies_to`` must name the source
        artifact -- resolved via ``_resolve_input_refs`` exactly like every
        other kind here, so a since-moved or still-missing source
        correctly invalidates/holds back this verdict through the same
        ``Evidence.input_versions``/``missing_inputs`` mechanism, no
        dependency-specific invalidation logic needed.

        This method is a thin **adapter**, not a second implementation of
        the actual decision: it builds one ``dependency_mapping.
        CrossSectionObligation`` from ``values`` + the resolved source ref,
        and delegates the real must_reference/must_precede verdict to
        ``dependency_verifier``'s own functions (the same ones
        ``dependency_verifier.verify_cross_section_obligations`` uses for
        whole-document reporting) -- see
        ``docs/av_stage2_dependency_mapping.md`` §... for why reusing,
        rather than reimplementing, that decision is the point: this
        checker and the whole-document report must never be able to
        disagree about the same fact.

        ``must_reuse``/``must_be_consistent_with`` (Fix C checklist item
        C3's second bullet, now that Fix B's B3 registry exists) delegate
        to ``dependency_verifier._verify_shared_value_relation`` the same
        way -- but ONLY when the caller supplies a non-empty
        ``relevant_symbols`` list in ``values``; an empty/absent one stays
        UNKNOWN/NOT_REPRESENTABLE exactly as before this round, since a
        shared-value check with no explicit symbol allowlist would have
        to guess which symbols are shared, which this checker (and
        ``shared_value_registry`` underneath it) never does -- see that
        module's own docstring. Optional extra keys: ``value_scope``,
        ``value_unit``, ``value_assumptions``, ``symbol_aliases``.
        """
        if not values or "relation" not in values:
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=(
                    "no dependency parameters supplied in "
                    f"context['dependency'][{constraint.constraint_id!r}]"
                ),
                unknown_reason=UnknownReason.NOT_CHECKED,
            )

        relation = DependencyRelation(values["relation"])
        input_versions, missing_inputs = self._resolve_input_refs(constraint, [])
        if missing_inputs:
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=f"source artifact(s) not found: {[r.artifact_id for r in missing_inputs]}",
                unknown_reason=UnknownReason.MISSING_DEPENDENCY,
                input_versions=input_versions,
                missing_inputs=missing_inputs,
            )

        source_section_id = constraint.applies_to[0].artifact_id if constraint.applies_to else None
        order_hint = values.get("order_hint")
        obligation = CrossSectionObligation(
            dependency_id=constraint.constraint_id,
            relation=relation,
            description=constraint.description,
            source_section_id=source_section_id or artifact.artifact_id,
            target_section_id=artifact.artifact_id,
            status=MappingStatus.MATCHED,
            matched_source_section_id=source_section_id,
            matched_target_section_id=artifact.artifact_id,
            allowed_labels=values.get("allowed_labels", []),
            order_hint=tuple(order_hint) if order_hint is not None else None,
        )

        stored_target = self._store.get_version(artifact.artifact_id, artifact.version)
        target_content = stored_target.content if stored_target is not None else None

        if relation == DependencyRelation.MUST_REFERENCE:
            rendered = {artifact.artifact_id: target_content} if target_content is not None else {}
            record = _verify_must_reference(obligation, rendered)
        elif relation == DependencyRelation.MUST_PRECEDE:
            record = _verify_must_precede(obligation)
        elif relation in (DependencyRelation.MUST_REUSE, DependencyRelation.MUST_BE_CONSISTENT_WITH):
            relevant_symbols = values.get("relevant_symbols") or []
            if not relevant_symbols:
                return Evidence(
                    constraint_id=constraint.constraint_id,
                    outcome=CheckOutcome.UNKNOWN,
                    checked_artifact=artifact,
                    detail=(
                        f"{relation.value} verification needs an explicit relevant_symbols "
                        "allowlist in context['dependency'][...] -- never guessed, to avoid "
                        "flagging an independently-scoped value as inconsistent."
                    ),
                    unknown_reason=UnknownReason.NOT_REPRESENTABLE,
                    input_versions=input_versions,
                    missing_inputs=missing_inputs,
                )
            source_ref_obj = constraint.applies_to[0] if constraint.applies_to else None
            source_content = None
            if source_ref_obj is not None:
                stored_source = self._store.get_version(source_ref_obj.artifact_id, source_ref_obj.version)
                source_content = stored_source.content if stored_source is not None else None
            rendered = {}
            if source_section_id and source_content is not None:
                rendered[source_section_id] = source_content
            if target_content is not None:
                rendered[artifact.artifact_id] = target_content
            record = _verify_shared_value_relation(
                obligation,
                rendered,
                relevant_symbols,
                scope=values.get("value_scope", "global"),
                unit=values.get("value_unit"),
                assumptions=values.get("value_assumptions", []),
                symbol_aliases=values.get("symbol_aliases"),
            )
        else:
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=f"unrecognized dependency relation {relation.value!r}",
                unknown_reason=UnknownReason.NOT_REPRESENTABLE,
                input_versions=input_versions,
                missing_inputs=missing_inputs,
            )

        return Evidence(
            constraint_id=constraint.constraint_id,
            outcome=record.outcome,
            checked_artifact=artifact,
            detail=record.detail,
            unknown_reason=record.unknown_reason,
            input_versions=input_versions,
            missing_inputs=missing_inputs,
        )


__all__ = ["BasicConstraintChecker"]

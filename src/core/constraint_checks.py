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

  Both modes record every artifact version the verdict actually consulted
  on ``Evidence.input_versions`` (direct mode: whatever ``source_refs``
  the caller declares; recompute mode: ``source_ref`` itself) --
  ``Validator.check(..., context=...)`` is the same extension point
  Stage 1's ``TextAgentValidatorAdapter`` already uses to carry whatever a
  given checker needs beyond the bare ``Constraint`` list.

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
from typing import Any, Dict, List, Optional

from src.core.artifact_graph import ArtifactGraphStore
from src.core.schemas import (
    ArtifactRef,
    CheckOutcome,
    Constraint,
    ConstraintKind,
    Evidence,
    UnknownReason,
)

# The one recompute operation this checker actually implements -- "숫자
# 재계산부터 구현하고" (checklist item 2) is satisfied by one concrete,
# narrow case (summing a JSON array of numbers read from a stored source
# artifact), not a general expression evaluator. Any other operation name
# is UNKNOWN/NOT_REPRESENTABLE, exactly like an unimplemented ConstraintKind.
_RECOMPUTE_OPERATIONS = {"sum": sum}

# The two constraint kinds this checker can actually evaluate; the other two
# (CSP_SMT, SYMBOLIC) always fall through to the final "not implemented"
# branch in ``check()``. Kept as an explicit set (not an if/elif chain
# spread across the method) so the one place that decides "do we have a
# narrow-enough mechanism for this kind at all" is unambiguous.
_IMPLEMENTED_KINDS = frozenset({ConstraintKind.SCHEMA_CONTRACT, ConstraintKind.RECOMPUTE})


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
                            "checker implemented yet (Stage 3 scope: only schema_contract and "
                            "recompute are implemented; csp_smt/symbolic are tracked as "
                            "follow-up work)"
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
            else:
                evidence.append(
                    self._check_recompute(constraint, artifact, recompute_inputs.get(constraint.constraint_id))
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

        ``input_versions`` is set to ``constraint.applies_to`` on both
        PASS and FAIL: even a PASS depends on those refs staying resolved
        at that exact version -- if one of them is later superseded, this
        verdict must stop being "live" (``ArtifactGraphStore.live_evidence``)
        even though ``checked_artifact`` itself never changed.
        """
        missing: List[str] = []
        for ref in constraint.applies_to:
            stored = self._store.get_version(ref.artifact_id, ref.version)
            if stored is None:
                missing.append(f"{ref.artifact_id}@v{ref.version} (not found)")
            elif stored.ref.kind != ref.kind:
                missing.append(
                    f"{ref.artifact_id}@v{ref.version} (expected kind={ref.kind.value!r}, "
                    f"actually stored as kind={stored.ref.kind.value!r})"
                )

        if missing:
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.FAIL,
                checked_artifact=artifact,
                detail=f"missing/unresolved required artifact(s): {', '.join(missing)}",
                input_versions=list(constraint.applies_to),
            )
        return Evidence(
            constraint_id=constraint.constraint_id,
            outcome=CheckOutcome.PASS,
            checked_artifact=artifact,
            detail="every applies_to artifact resolved to a real stored version of the claimed kind",
            input_versions=list(constraint.applies_to),
        )

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

        if "source_ref" in values and "operation" in values:
            return self._check_recompute_from_source(constraint, artifact, values, observed, tolerance)

        if "expected" in values:
            expected = float(values["expected"])
            raw_source_refs = values.get("source_refs") or []
            source_refs = [
                ref if isinstance(ref, ArtifactRef) else ArtifactRef.model_validate(ref)
                for ref in raw_source_refs
            ]
            return self._recompute_verdict(
                constraint, artifact, expected=expected, observed=observed, tolerance=tolerance,
                input_versions=source_refs, mode="comparison",
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
        if stored_source is None or stored_source.content is None:
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=f"source artifact {source_ref.artifact_id}@v{source_ref.version} not found or has no content",
                unknown_reason=UnknownReason.MISSING_DEPENDENCY,
                input_versions=[source_ref],
            )

        try:
            numbers = json.loads(stored_source.content)
            if not isinstance(numbers, list) or not all(isinstance(n, (int, float)) for n in numbers):
                raise ValueError("not a JSON array of numbers")
        except (ValueError, TypeError, json.JSONDecodeError):
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=(
                    f"source artifact {source_ref.artifact_id}@v{source_ref.version}'s content "
                    "is not a JSON array of numbers -- cannot recompute"
                ),
                unknown_reason=UnknownReason.NOT_REPRESENTABLE,
                input_versions=[source_ref],
            )

        expected = float(_RECOMPUTE_OPERATIONS[operation](numbers))
        return self._recompute_verdict(
            constraint, artifact, expected=expected, observed=observed, tolerance=tolerance,
            input_versions=[source_ref], mode=f"recomputed via {operation!r} from {source_ref.artifact_id}@v{source_ref.version}",
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
        mode: str,
    ) -> Evidence:
        diff = abs(expected - observed)
        if diff <= tolerance:
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.PASS,
                checked_artifact=artifact,
                detail=f"observed={observed} matches expected={expected} ({mode}) within tolerance={tolerance}",
                input_versions=input_versions,
            )
        return Evidence(
            constraint_id=constraint.constraint_id,
            outcome=CheckOutcome.FAIL,
            checked_artifact=artifact,
            detail=(
                f"observed={observed} does not match expected={expected} ({mode}) "
                f"(|diff|={diff} > tolerance={tolerance})"
            ),
            input_versions=input_versions,
        )


__all__ = ["BasicConstraintChecker"]

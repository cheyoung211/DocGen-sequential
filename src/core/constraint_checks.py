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
  on ``Evidence.input_versions`` (direct mode: ``constraint.applies_to``
  merged with whatever ``source_refs`` the caller additionally declares;
  recompute mode: ``applies_to`` merged with ``source_ref`` itself --
  ``_merge_input_refs``). A second Stage 3 review found that a caller
  declaring the source *only* via ``applies_to`` -- a legitimate choice,
  since that is the one mechanism SCHEMA_CONTRACT already uses -- got an
  empty ``input_versions`` under the original, ``source_refs``-only
  version of this, so a later change to that source never invalidated the
  verdict. A source artifact found to not exist *at all* goes on
  ``Evidence.missing_inputs`` instead, by id, never as a fabricated
  "existing" version -- see that field's own docstring for why conflating
  the two inverts ``ArtifactGraphStore.live_evidence``'s invalidation
  direction for exactly this case. ``Validator.check(..., context=...)``
  is the same extension point Stage 1's ``TextAgentValidatorAdapter``
  already uses to carry whatever a given checker needs beyond the bare
  ``Constraint`` list.

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

        A ref that resolved (found, right kind, *or* found with the wrong
        kind -- it still exists, just incorrectly) goes on
        ``Evidence.input_versions``: a later review found this check's own
        correctness also depends on that version staying the current one,
        PASS or FAIL alike. A ref that was not found at all -- the
        artifact_id has literally never been stored -- goes on
        ``Evidence.missing_inputs`` instead, by artifact_id alone, never
        a version number: a second review found that recording a
        not-found ref as if it were some trustworthy existing version (the
        original fix) inverted ``ArtifactGraphStore.live_evidence``'s
        invalidation direction for exactly this case -- see
        ``Evidence.missing_inputs``'s own docstring.
        """
        unresolved_detail: List[str] = []
        live_refs: List[ArtifactRef] = []
        missing_ids: List[str] = []
        for ref in constraint.applies_to:
            stored = self._store.get_version(ref.artifact_id, ref.version)
            if stored is None:
                unresolved_detail.append(f"{ref.artifact_id}@v{ref.version} (not found)")
                missing_ids.append(ref.artifact_id)
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
                missing_inputs=missing_ids,
            )
        return Evidence(
            constraint_id=constraint.constraint_id,
            outcome=CheckOutcome.PASS,
            checked_artifact=artifact,
            detail="every applies_to artifact resolved to a real stored version of the claimed kind",
            input_versions=live_refs,
        )

    def _merge_input_refs(self, constraint: Constraint, extra: List[ArtifactRef]) -> List[ArtifactRef]:
        """``constraint.applies_to`` plus whatever extra refs a RECOMPUTE
        mode consulted beyond it (``source_refs``/``source_ref``),
        deduped by ``(artifact_id, version)``. A second Stage 3 review
        found that a RECOMPUTE constraint declaring its source only via
        ``applies_to`` -- the same mechanism SCHEMA_CONTRACT uses, instead
        of duplicating it into ``context["recompute"][...]["source_refs"]``
        -- ended up with an empty ``Evidence.input_versions``, so a PASS
        kept reading as live evidence forever even after that source
        artifact changed version. Merging means it no longer matters which
        of the two ways a caller happened to declare the dependency."""
        combined = list(constraint.applies_to)
        seen = {(ref.artifact_id, ref.version) for ref in combined}
        for ref in extra:
            key = (ref.artifact_id, ref.version)
            if key not in seen:
                combined.append(ref)
                seen.add(key)
        return combined

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
                input_versions=self._merge_input_refs(constraint, source_refs), mode="comparison",
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
        if stored_source is None:
            # The source artifact_id was never stored at all (at this
            # version or any other) -- this goes on missing_inputs, not
            # input_versions, so ArtifactGraphStore.live_evidence treats
            # "still absent" as live and "now exists" as invalidating, not
            # the other way around. See Evidence.missing_inputs's docstring.
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=f"source artifact {source_ref.artifact_id}@v{source_ref.version} not found",
                unknown_reason=UnknownReason.MISSING_DEPENDENCY,
                input_versions=list(constraint.applies_to),
                missing_inputs=[source_ref.artifact_id],
            )
        if stored_source.content is None:
            # The artifact exists (at this version) but was stored with no
            # content -- a real, trackable version, so this belongs on
            # input_versions like any other resolved-but-unusable source.
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=f"source artifact {source_ref.artifact_id}@v{source_ref.version} has no content",
                unknown_reason=UnknownReason.MISSING_DEPENDENCY,
                input_versions=self._merge_input_refs(constraint, [source_ref]),
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
                input_versions=self._merge_input_refs(constraint, [source_ref]),
            )

        expected = float(_RECOMPUTE_OPERATIONS[operation](numbers))
        return self._recompute_verdict(
            constraint, artifact, expected=expected, observed=observed, tolerance=tolerance,
            input_versions=self._merge_input_refs(constraint, [source_ref]),
            mode=f"recomputed via {operation!r} from {source_ref.artifact_id}@v{source_ref.version}",
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

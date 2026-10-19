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
  resolve to a real, stored ``(artifact_id, version)`` in the graph.
  Listing a required component's ref is indistinguishable, at this level,
  from listing a referenced target's ref; both are "this must exist".
  **This cannot detect a reference that exists in an artifact's raw
  *content* but was never declared in ``applies_to``** -- that would be
  real reference-scanning, which Stage 2 already deferred explicitly
  (``docs/stage2_artifact_graph.md``, "하지 않은 것": "실제 레퍼런스 스캔
  기반 INFERRED edge 자동 탐지"). This checker only ever answers "are the
  artifacts this constraint says must exist actually there", not "does
  this artifact's content actually cite them".
- ``RECOMPUTE``: compares a caller-supplied expected/observed numeric pair
  within a tolerance. The checker does not itself extract numbers from
  document content or source data -- direction 3's "Python 기반
  재계산" is the caller's job (wherever the real artifact content and
  source-data artifact eventually get wired through, Stage 4/5); this
  checker's job is only the comparison and producing structured Evidence
  from it, via ``context["recompute"][constraint_id] = {"expected":
  ..., "observed": ..., "tolerance": ...}`` -- the same
  ``Validator.check(..., context=...)`` extension point Stage 1's
  ``TextAgentValidatorAdapter`` already uses to carry whatever a given
  checker needs beyond the bare ``Constraint`` list.

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
        resolves to a real stored ``(artifact_id, version)`` right now;
        FAIL lists exactly which ones do not. An empty ``applies_to`` is
        vacuously PASS -- a schema-contract constraint naming nothing to
        check for existence has nothing it can fail on."""
        missing: List[str] = []
        for ref in constraint.applies_to:
            stored = self._store.get_version(ref.artifact_id, ref.version)
            if stored is None:
                missing.append(f"{ref.artifact_id}@v{ref.version}")

        if missing:
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.FAIL,
                checked_artifact=artifact,
                detail=f"missing/unresolved required artifact(s): {', '.join(missing)}",
            )
        return Evidence(
            constraint_id=constraint.constraint_id,
            outcome=CheckOutcome.PASS,
            checked_artifact=artifact,
            detail="every applies_to artifact resolved to a real stored version",
        )

    def _check_recompute(
        self, constraint: Constraint, artifact: ArtifactRef, values: Optional[Dict[str, Any]]
    ) -> Evidence:
        """Compares ``values["observed"]`` against ``values["expected"]``
        within ``values.get("tolerance", 0.0)``. Missing/incomplete
        ``values`` is UNKNOWN/NOT_CHECKED, not a silent FAIL or an
        exception -- this checker was simply never handed what it needs to
        recompute against, which is a different fact than the recompute
        actually having been run and mismatched."""
        if not values or "expected" not in values or "observed" not in values:
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.UNKNOWN,
                checked_artifact=artifact,
                detail=(
                    "no expected/observed values supplied in "
                    f"context['recompute'][{constraint.constraint_id!r}]"
                ),
                unknown_reason=UnknownReason.NOT_CHECKED,
            )

        expected = float(values["expected"])
        observed = float(values["observed"])
        tolerance = float(values.get("tolerance", 0.0))
        diff = abs(expected - observed)

        if diff <= tolerance:
            return Evidence(
                constraint_id=constraint.constraint_id,
                outcome=CheckOutcome.PASS,
                checked_artifact=artifact,
                detail=f"observed={observed} matches expected={expected} within tolerance={tolerance}",
            )
        return Evidence(
            constraint_id=constraint.constraint_id,
            outcome=CheckOutcome.FAIL,
            checked_artifact=artifact,
            detail=(
                f"observed={observed} does not match expected={expected} "
                f"(|diff|={diff} > tolerance={tolerance})"
            ),
        )


__all__ = ["BasicConstraintChecker"]

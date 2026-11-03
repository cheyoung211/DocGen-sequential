"""A real ``ConstraintProvider`` (``repair_operations.ConstraintProvider =
Callable[[ArtifactRef], List[Constraint]]``) for DEPENDENCY constraints --
Fix C (the "Revision (Additional verifiers)" plan), checklist item C4's
"dependency 위반 evidence를 기존 check·patch·replan에 연결한다".

Stage 0 of that plan documented, as a known, deferred gap: "production
``ConstraintProvider``가 코드베이스 어디에도 없다" -- every Stage 1-6 test
across both revisions only ever supplies one from a test fake. This module
closes that gap specifically for dependency constraints, built from one
``dependency_mapping.DependencyMappingReport`` (C1).

A second, less obvious gap, found while building this: ``Validator.check()``
needs more than ``List[Constraint]`` for ``RECOMPUTE``/``DEPENDENCY`` --
relation-specific parameters travel through ``context["recompute"][...]``/
``context["dependency"][...]`` instead (``constraint_checks.py``'s own
established convention), and NEITHER real call site that invokes a
``ConstraintProvider`` today (``repair_operations.py``'s
``self._base_context`` kwarg, fixed at construction time; ``controller.py``'s
``CheckCall(..., context={})``) actually threads anything from the provider
into that context. This is not new to DEPENDENCY -- RECOMPUTE already had
this exact gap, just never named, since no RECOMPUTE ``ConstraintProvider``
existed yet either to notice it against. ``DependencyConstraintProvider``
exposes both halves from the one place that actually has the data
(``context_for``, alongside ``__call__``) so a future caller can combine
them; actually wiring that combination into ``repair_operations.py``/
``controller.py``'s own re-verification loop is further, separate work,
tracked here rather than silently assumed done.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.artifact_graph import ArtifactGraphStore
from src.core.dependency_mapping import CrossSectionObligation, DependencyMappingReport, MappingStatus
from src.core.schemas import ArtifactKind, ArtifactRef, Constraint, ConstraintKind


class DependencyConstraintProvider:
    """``__call__`` conforms to ``repair_operations.ConstraintProvider``.

    Built from one ``DependencyMappingReport`` + the ``ArtifactGraphStore``
    that resolves a source section id to its *current* version (never a
    guessed/stale one -- see ``_source_ref``). Every ``MATCHED``
    obligation targeting the asked-about artifact becomes one
    ``Constraint``, **regardless of relation** -- including
    ``must_reuse``/``must_be_consistent_with``, which ``_check_dependency``
    cannot yet resolve past UNKNOWN: naming them here is what keeps an
    unresolved dependency visible as a real, tracked constraint instead of
    simply never being asked about (the plan's own "미해결 상태를... 보고
    한다", never silently dropped). Obligations whose mapping ``status``
    isn't ``MATCHED`` (UNMATCHED/AMBIGUOUS/DUPLICATE_ID) are skipped here,
    not turned into a Constraint at all -- there is no real source/target
    pair yet for ``applies_to`` to even name; C1's own report already
    carries that unresolved state under ``data_quality_issues``/the
    mapping records themselves, so nothing is lost by not duplicating it
    into a second, parallel "constraint for a thing that doesn't exist"
    representation.
    """

    def __init__(self, report: DependencyMappingReport, store: ArtifactGraphStore) -> None:
        self._report = report
        self._store = store

    def _obligations_for(self, artifact_id: str) -> List[CrossSectionObligation]:
        return [
            obligation
            for obligation in self._report.cross_section_obligations
            if obligation.status == MappingStatus.MATCHED
            and obligation.matched_target_section_id == artifact_id
        ]

    def _source_ref(self, source_section_id: str) -> ArtifactRef:
        """The source's *current* stored version, or a version-1 reference
        existence-checked (and correctly found missing) by the checker's
        own ``_resolve_input_refs`` if it was never stored at all -- never
        a guessed version number for a source that DOES exist at some
        other version (that would pin ``input_versions`` to a stale
        version forever, defeating the whole point of version-bound
        invalidation -- see this module's own docstring)."""
        stored = self._store.latest(source_section_id)
        if stored is not None:
            return stored.ref
        return ArtifactRef(artifact_id=source_section_id, version=1, kind=ArtifactKind.SECTION)

    @staticmethod
    def _constraint_id(obligation: CrossSectionObligation) -> str:
        return f"{obligation.dependency_id}::{obligation.target_section_id}"

    def __call__(self, artifact: ArtifactRef) -> List[Constraint]:
        constraints: List[Constraint] = []
        for obligation in self._obligations_for(artifact.artifact_id):
            source_ref = self._source_ref(obligation.matched_source_section_id)
            constraints.append(
                Constraint(
                    constraint_id=self._constraint_id(obligation),
                    kind=ConstraintKind.DEPENDENCY,
                    description=obligation.description,
                    source_ref=obligation.dependency_id,
                    applies_to=[source_ref],
                )
            )
        return constraints

    def context_for(self, artifact: ArtifactRef) -> Dict[str, Any]:
        """The ``context["dependency"]`` payload matching, one-to-one,
        whatever ``__call__(artifact)`` just returned for the same
        artifact -- both read the exact same obligations list, so the two
        can never drift out of sync with each other."""
        dependency_context: Dict[str, Any] = {}
        for obligation in self._obligations_for(artifact.artifact_id):
            dependency_context[self._constraint_id(obligation)] = {
                "relation": obligation.relation.value,
                "allowed_labels": list(obligation.allowed_labels),
                "order_hint": list(obligation.order_hint) if obligation.order_hint is not None else None,
            }
        return {"dependency": dependency_context}


__all__ = ["DependencyConstraintProvider"]

"""Controlled fault injection (Stage 6 of the "Revision 2026.10.15" plan:
"실패·행동 데이터셋과 human gold 구축" -- the "자연 실패와 주입 실패를 구분해
수집" half of that checklist).

Implements the four concrete injection types revision direction 4 names
under "주입 예시": a theorem/proof missing a needed assumption, a table
whose denominator was altered, a symbol given a different meaning in
different sections, and a blueprint whose required components cannot
jointly be satisfied.

**Scope decision (confirmed with the user before implementing)**: this
module writes hand-authored corrupted *content* directly through
``ArtifactGraphStore.add_version`` -- it never calls a real ``Producer`` or
the local Qwen3.6 pipeline. This mirrors every earlier stage's own
discipline (Stages 3-5 all built mechanism + Protocol-conforming-fake tests
first, deferring a real LLM-calling pilot to a dedicated follow-up round):
the injector is exercised and reviewed here purely against
``ArtifactGraphStore``, which is already Stage 2's reviewed, deterministic
surface. Wiring a real pilot (perturbing an artifact mid-pipeline, letting
the real agents/validators run downstream of it) is explicitly left for
that follow-up, not this module.

Each ``inject_*`` method:

1. Reads the artifact(s) being corrupted via ``store.latest()`` (never
   assumes a caller-supplied ``ArtifactRef`` is still current -- the same
   "read the live state, don't trust a stale pointer" discipline
   ``repair_operations.py`` already follows).
2. Writes exactly one new version per touched artifact through the public
   ``add_version`` API, with ``read_input_versions`` echoing the artifact's
   OWN pre-injection ``read_input_versions`` -- i.e. whatever real upstream
   artifacts it was actually built from (``current.ref.read_input_versions``)
   -- never ``[current.ref]`` (the artifact's own previous version). A
   review caught that passing ``[current.ref]`` models "this version
   depends on its own prior version" as a real semantic ``DependencyEdge``,
   which is wrong on two counts: (a) version history is already tracked by
   the store's own version list, with no edge needed at all, and (b) doing
   so creates a self-loop edge that pollutes both
   ``direct_dependents``/``direct_dependencies`` (the corrupted artifact
   now appears to depend on, and be depended on by, itself) AND silently
   severs the REAL upstream edge: `_current_dependents`'s "currently
   applicable" filter drops any edge whose ``to_artifact`` is no longer
   that artifact's latest version, so the moment a new version is written
   with no real input re-declared, the genuine upstream dependency
   (recorded against the now-superseded old version) stops counting, and
   a later change to that real upstream artifact no longer propagates
   staleness forward at all. Re-declaring the real ``read_input_versions``
   keeps the genuine dependency chain intact across the injection -- the
   fault corrupts *content*, never the artifact's documented provenance.
3. Returns an ``InjectedFault`` recording what changed and, via
   ``store.impact_candidates`` evaluated *after* the write, which
   artifacts the injector itself predicts should be affected -- kept
   separate from whatever a later classification/repair step actually
   observes (see that schema's own docstring for why).

4. Captures ``store.direct_dependencies(artifact_id)`` -- EXPLICIT *and*
   INFERRED, the complete real dependency set -- immediately *before*
   writing the corrupted version, and (a) records it on ``InjectedFault.
   pre_injection_dependency_ids_by_artifact`` as the known reference
   baseline, and (b) re-registers, against the new version, any of those
   dependencies that came from an INFERRED edge (via
   ``_carry_forward_inferred_edges``). A review caught that point 2's own
   fix -- re-declaring ``read_input_versions`` -- only carries EXPLICIT
   deps forward: an INFERRED edge (``ArtifactGraphStore.add_inferred_edge``)
   never appears in any ``ArtifactRef.read_input_versions`` at all, so it
   still silently dropped the moment the corrupted version superseded the
   one the INFERRED edge pointed at (the same "edge only counts against
   the current version" filter point 2's docstring already explains, one
   layer deeper).
"""

from __future__ import annotations

import uuid
from typing import List, Optional, Set

from src.core.artifact_graph import ArtifactGraphStore
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    DependencyRelationKind,
    FaultInjectionKind,
    InjectedFault,
)


class UnsupportedInjectionTargetError(ValueError):
    """Raised when an injector is asked to corrupt an artifact whose kind
    or current content shape cannot represent the requested fault -- e.g.
    altering "the denominator" of an artifact that was never stored as a
    TABLE. Never silently injects a different kind of fault than the one
    requested."""


def _new_fault_id(kind: FaultInjectionKind) -> str:
    return f"fault_{kind.value}_{uuid.uuid4().hex[:8]}"


def _carry_forward_inferred_edges(
    store: ArtifactGraphStore,
    pre_injection_dependency_ids: Set[str],
    old_ref: ArtifactRef,
    new_ref: ArtifactRef,
) -> None:
    """Re-register, against ``new_ref``, every INFERRED edge that was live
    against ``old_ref`` immediately before an injection superseded it.

    ``add_version`` (called just before this, to write the corrupted
    content) only re-creates EXPLICIT edges, one per ``read_input_versions``
    entry -- an INFERRED edge lives solely in ``store.edges`` and is never
    implied by any ``ArtifactRef``, so without this call the moment the
    injected version becomes current, ``_current_dependents``'s "only live
    against the current version" filter drops the INFERRED relation
    entirely, exactly the way the original ``[current.ref]`` self-loop bug
    dropped EXPLICIT ones.

    Only the ids in ``pre_injection_dependency_ids`` that are NOT already
    covered by ``new_ref.read_input_versions`` can possibly be INFERRED
    (``direct_dependencies`` returns both kinds together) -- those are
    looked up against the actual pre-injection INFERRED edges so the
    original ``reason`` and exact ``from_artifact`` version are preserved,
    not fabricated.
    """
    declared_ids = {ref.artifact_id for ref in new_ref.read_input_versions}
    inferred_from_ids = pre_injection_dependency_ids - declared_ids
    if not inferred_from_ids:
        return
    inferred_edges_by_from_id = {
        edge.from_artifact.artifact_id: edge
        for edge in store.edges
        if edge.relation == DependencyRelationKind.INFERRED
        and edge.to_artifact.artifact_id == old_ref.artifact_id
        and edge.to_artifact.version == old_ref.version
    }
    for from_id in inferred_from_ids:
        edge = inferred_edges_by_from_id[from_id]
        store.add_inferred_edge(
            edge.from_artifact,
            new_ref,
            reason=f"carried forward across fault injection: {edge.reason}",
        )


class FaultInjector:
    """Applies one of the four named perturbation types to a live
    ``ArtifactGraphStore``. Stateless beyond the store it is given per call
    -- a single instance may be reused across many injections, including
    across different stores (e.g. one per branch in a checkpoint-branch
    comparison), unlike ``constraint_checks.BasicConstraintChecker``, which
    deliberately binds to one store at construction time. There is nothing
    here that would make store-binding meaningful: every method takes the
    store explicitly and reads/writes nothing else.
    """

    def inject_missing_assumption(
        self,
        store: ArtifactGraphStore,
        theorem_ref: ArtifactRef,
        *,
        assumption_text: str,
        produced_by: str = "fault_injection:missing_assumption",
    ) -> InjectedFault:
        """Drop a stated assumption from a theorem/claim-like artifact's
        content. ``assumption_text`` must be the exact substring to remove
        (the caller names the real assumption sentence present in the
        artifact's current content) -- raises rather than silently
        no-opping if it is not actually present, since an injection that
        does not change anything is not an injection.
        """
        current = store.latest(theorem_ref.artifact_id)
        if current is None:
            raise UnsupportedInjectionTargetError(
                f"'{theorem_ref.artifact_id}' has never been stored; cannot inject into it."
            )
        content = current.content or ""
        if assumption_text not in content:
            raise UnsupportedInjectionTargetError(
                f"assumption_text {assumption_text!r} was not found in "
                f"'{theorem_ref.artifact_id}'@v{current.ref.version}'s content -- "
                "an injection must remove something that was actually there."
            )
        pre_injection_dependency_ids = store.direct_dependencies(theorem_ref.artifact_id)
        corrupted = content.replace(assumption_text, "", 1)
        result = store.add_version(
            theorem_ref.artifact_id,
            current.ref.kind,
            content=corrupted,
            read_input_versions=current.ref.read_input_versions,
            produced_by=produced_by,
        )
        _carry_forward_inferred_edges(store, pre_injection_dependency_ids, current.ref, result.ref)
        predicted = sorted(store.impact_candidates(theorem_ref.artifact_id) - {theorem_ref.artifact_id})
        return InjectedFault(
            fault_id=_new_fault_id(FaultInjectionKind.MISSING_ASSUMPTION),
            kind=FaultInjectionKind.MISSING_ASSUMPTION,
            pre_injection_refs=[current.ref],
            injected_refs=[result.ref],
            description=(
                f"removed required assumption {assumption_text!r} from "
                f"'{theorem_ref.artifact_id}'@v{current.ref.version}"
            ),
            predicted_affected_artifact_ids=predicted,
            pre_injection_dependency_ids_by_artifact={
                theorem_ref.artifact_id: sorted(pre_injection_dependency_ids)
            },
        )

    def inject_altered_denominator(
        self,
        store: ArtifactGraphStore,
        table_ref: ArtifactRef,
        *,
        original_denominator: str,
        corrupted_denominator: str,
        produced_by: str = "fault_injection:altered_denominator",
    ) -> InjectedFault:
        """Replace a table's stated denominator (a literal substring of
        its content, e.g. ``"/ 120"``) with a different one, without
        changing any other number in the table -- the kind of error a
        RECOMPUTE constraint (``constraint_checks.BasicConstraintChecker``)
        is specifically built to catch once a value derived from this
        table disagrees with its own recomputation."""
        current = store.latest(table_ref.artifact_id)
        if current is None:
            raise UnsupportedInjectionTargetError(
                f"'{table_ref.artifact_id}' has never been stored; cannot inject into it."
            )
        if current.ref.kind != ArtifactKind.TABLE:
            raise UnsupportedInjectionTargetError(
                f"'{table_ref.artifact_id}' is stored as kind {current.ref.kind!r}, not TABLE -- "
                "altering a denominator only makes sense for a TABLE artifact."
            )
        content = current.content or ""
        if original_denominator not in content:
            raise UnsupportedInjectionTargetError(
                f"original_denominator {original_denominator!r} was not found in "
                f"'{table_ref.artifact_id}'@v{current.ref.version}'s content."
            )
        pre_injection_dependency_ids = store.direct_dependencies(table_ref.artifact_id)
        corrupted = content.replace(original_denominator, corrupted_denominator, 1)
        result = store.add_version(
            table_ref.artifact_id,
            ArtifactKind.TABLE,
            content=corrupted,
            read_input_versions=current.ref.read_input_versions,
            produced_by=produced_by,
        )
        _carry_forward_inferred_edges(store, pre_injection_dependency_ids, current.ref, result.ref)
        predicted = sorted(store.impact_candidates(table_ref.artifact_id) - {table_ref.artifact_id})
        return InjectedFault(
            fault_id=_new_fault_id(FaultInjectionKind.ALTERED_DENOMINATOR),
            kind=FaultInjectionKind.ALTERED_DENOMINATOR,
            pre_injection_refs=[current.ref],
            injected_refs=[result.ref],
            description=(
                f"altered denominator in '{table_ref.artifact_id}'@v{current.ref.version}: "
                f"{original_denominator!r} -> {corrupted_denominator!r}"
            ),
            predicted_affected_artifact_ids=predicted,
            pre_injection_dependency_ids_by_artifact={
                table_ref.artifact_id: sorted(pre_injection_dependency_ids)
            },
        )

    def inject_inconsistent_symbol_meaning(
        self,
        store: ArtifactGraphStore,
        symbol: str,
        section_a_ref: ArtifactRef,
        section_b_ref: ArtifactRef,
        *,
        meaning_a: str,
        meaning_b: str,
        produced_by: str = "fault_injection:inconsistent_symbol_meaning",
    ) -> InjectedFault:
        """Rewrite two sections so the same ``symbol`` is (re)defined with
        two different meanings across them -- touches exactly two
        artifacts, which is why this is the one injection kind
        ``InjectedFault`` needs ``pre_injection_refs``/``injected_refs`` to
        hold more than one entry for.
        """
        current_a = store.latest(section_a_ref.artifact_id)
        current_b = store.latest(section_b_ref.artifact_id)
        if current_a is None or current_b is None:
            missing = section_a_ref.artifact_id if current_a is None else section_b_ref.artifact_id
            raise UnsupportedInjectionTargetError(f"'{missing}' has never been stored; cannot inject into it.")
        if section_a_ref.artifact_id == section_b_ref.artifact_id:
            raise UnsupportedInjectionTargetError(
                "inject_inconsistent_symbol_meaning requires two distinct sections; "
                f"got the same artifact_id {section_a_ref.artifact_id!r} twice."
            )
        pre_injection_dependency_ids_a = store.direct_dependencies(section_a_ref.artifact_id)
        pre_injection_dependency_ids_b = store.direct_dependencies(section_b_ref.artifact_id)
        content_a = f"{current_a.content or ''}\n{symbol} denotes {meaning_a}."
        content_b = f"{current_b.content or ''}\n{symbol} denotes {meaning_b}."
        result_a = store.add_version(
            section_a_ref.artifact_id,
            current_a.ref.kind,
            content=content_a,
            read_input_versions=current_a.ref.read_input_versions,
            produced_by=produced_by,
        )
        _carry_forward_inferred_edges(store, pre_injection_dependency_ids_a, current_a.ref, result_a.ref)
        result_b = store.add_version(
            section_b_ref.artifact_id,
            current_b.ref.kind,
            content=content_b,
            read_input_versions=current_b.ref.read_input_versions,
            produced_by=produced_by,
        )
        _carry_forward_inferred_edges(store, pre_injection_dependency_ids_b, current_b.ref, result_b.ref)
        predicted = sorted(
            (store.impact_candidates(section_a_ref.artifact_id) | store.impact_candidates(section_b_ref.artifact_id))
            - {section_a_ref.artifact_id, section_b_ref.artifact_id}
        )
        return InjectedFault(
            fault_id=_new_fault_id(FaultInjectionKind.INCONSISTENT_SYMBOL_MEANING),
            kind=FaultInjectionKind.INCONSISTENT_SYMBOL_MEANING,
            pre_injection_refs=[current_a.ref, current_b.ref],
            injected_refs=[result_a.ref, result_b.ref],
            description=(
                f"symbol {symbol!r} means {meaning_a!r} in '{section_a_ref.artifact_id}' "
                f"but {meaning_b!r} in '{section_b_ref.artifact_id}'"
            ),
            predicted_affected_artifact_ids=predicted,
            pre_injection_dependency_ids_by_artifact={
                section_a_ref.artifact_id: sorted(pre_injection_dependency_ids_a),
                section_b_ref.artifact_id: sorted(pre_injection_dependency_ids_b),
            },
        )

    def inject_unsatisfiable_blueprint(
        self,
        store: ArtifactGraphStore,
        document_ref: ArtifactRef,
        *,
        contradictory_requirements: List[str],
        produced_by: str = "fault_injection:unsatisfiable_blueprint",
    ) -> InjectedFault:
        """Append two (or more) requirements to a document/blueprint
        artifact's content that cannot jointly be satisfied (the caller
        names them as plain-text requirement strings -- this method does
        not attempt to prove unsatisfiability itself, only to record that
        the requirements were deliberately chosen to contradict each
        other). Requires at least 2 requirements, since a single
        requirement cannot contradict itself."""
        if len(contradictory_requirements) < 2:
            raise UnsupportedInjectionTargetError(
                "inject_unsatisfiable_blueprint requires at least 2 contradictory_requirements; "
                f"got {len(contradictory_requirements)}."
            )
        current = store.latest(document_ref.artifact_id)
        if current is None:
            raise UnsupportedInjectionTargetError(
                f"'{document_ref.artifact_id}' has never been stored; cannot inject into it."
            )
        pre_injection_dependency_ids = store.direct_dependencies(document_ref.artifact_id)
        requirement_block = "\n".join(f"- {req}" for req in contradictory_requirements)
        corrupted = f"{current.content or ''}\n[contradictory requirements]\n{requirement_block}"
        result = store.add_version(
            document_ref.artifact_id,
            current.ref.kind,
            content=corrupted,
            read_input_versions=current.ref.read_input_versions,
            produced_by=produced_by,
        )
        _carry_forward_inferred_edges(store, pre_injection_dependency_ids, current.ref, result.ref)
        predicted = sorted(store.impact_candidates(document_ref.artifact_id) - {document_ref.artifact_id})
        return InjectedFault(
            fault_id=_new_fault_id(FaultInjectionKind.UNSATISFIABLE_BLUEPRINT),
            kind=FaultInjectionKind.UNSATISFIABLE_BLUEPRINT,
            pre_injection_refs=[current.ref],
            injected_refs=[result.ref],
            description=(
                f"'{document_ref.artifact_id}'@v{current.ref.version} now requires mutually "
                f"contradictory requirements: {contradictory_requirements!r}"
            ),
            predicted_affected_artifact_ids=predicted,
            pre_injection_dependency_ids_by_artifact={
                document_ref.artifact_id: sorted(pre_injection_dependency_ids)
            },
        )

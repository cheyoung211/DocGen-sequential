"""Versioned artifact graph: IDs/versions, typed edges, change-impact
computation, staleness, and checkpointing (Stage 2 of the "Revision
2026.10.15" plan: "2단계. Versioned artifact graph와 로그·checkpoint").

Builds directly on Stage 1's schemas: ``ArtifactRef``/``DependencyEdge``
were already defined in ``src/core/schemas.py`` specifically for this
stage to "own actually assigning/incrementing ``version`` and persisting
these pointers" (see that module's docstrings). Nothing here reads or
mutates ``src/common/state.py``'s ``DocumentGraph`` -- revision direction 1
is explicit that the execution graph (today's ``DocumentGraph``, and
eventually the LangGraph controller) and the artifact dependency graph are
kept separate; this module is only the latter. Wiring a real pipeline run
through this store is Stage 4/5's job, not this one's.

The dependency graph is explicitly **not** assumed acyclic -- the Notion
plan says so directly ("문서 내 상호 참조에는 순환이 있을 수 있으므로 모든
관계를 DAG로 강제하지 않는다": a document's cross-references can cycle,
e.g. section A's content cites section B's label and vice versa). Every
reachability computation below is a visited-set BFS for exactly this
reason, never a toposort.

Two invariants hold everywhere in this file, enforced after a user-caught
review found them both broken (see ``test/core/test_artifact_graph.py``'s
regression tests, named after the exact repro each one came from):

1. **Every object that crosses the store's boundary, in either direction,
   is an independent deep copy.** A parameter handed in (an ``ArtifactRef``
   declared as a read input, an ``Evidence``/``CostRecord`` being logged,
   an edge endpoint) is copied before anything internal ever holds a
   reference to it; anything handed back out (``latest()``, ``edges``,
   ``live_evidence()``, ``save_checkpoint()``, ...) is copied before the
   caller ever sees it. Mutating a parameter after the call, or mutating a
   return value, must never retroactively change stored history.
2. **Staleness has exactly one source of truth: ``_stale_ids``.**
   ``StoredArtifact.status`` is never trusted as a static field for the
   *current* latest version of an artifact -- every public accessor
   overlays ``_live_status(artifact_id)`` onto a copy of it at read time,
   so ``is_stale(x)`` and ``latest(x).status`` can never disagree, no
   matter how long ago ``x`` was actually produced relative to whenever
   something upstream of it last changed, and regardless of *which*
   mechanism last touched ``_stale_ids`` for it (``add_version``'s own
   propagation to descendants, or ``add_inferred_edge``'s retroactive
   invalidation when a missed relation is filled in later). Once a
   version is superseded by a newer one, its ``.status`` is frozen for
   good -- to ``_live_status`` at that exact moment -- rather than kept
   live forever or reset; see ``get_version``'s docstring for why a
   superseded version's own historical status must stay a fixed fact, not
   a value recomputed (or forced) at read time the way the current
   version's is.
"""

from __future__ import annotations

import copy
import uuid
from typing import Dict, List, Optional, Set

from src.core.run_log import RunLog
from src.core.schemas import (
    ActionLogEntry,
    AddVersionResult,
    ArtifactKind,
    ArtifactRef,
    ArtifactStatus,
    Checkpoint,
    CostRecord,
    DependencyEdge,
    DependencyRelationKind,
    Evidence,
    InvalidationRecord,
    RepairAction,
    StoredArtifact,
    ToolCall,
    ToolCallLogEntry,
    ToolObservation,
)


class UnknownArtifactError(ValueError):
    """Raised when a caller references an ``(artifact_id, version)`` the
    store never actually recorded -- e.g. claiming to have read it as an
    input, or declaring an inferred edge to/from it. Covers both "this
    ``artifact_id`` was never stored at all" and "this ``artifact_id``
    exists, but never at this specific ``version``" -- a version number
    that was never real cannot be told apart from a real one by
    ``add_version``'s *outdated*-input bookkeeping
    (``AddVersionResult.used_outdated_inputs``), which assumes every input
    it is handed at least once existed; both are hard errors here instead.

    Reading or depending on something that was never produced is a caller
    bug, not a reachable research scenario -- unlike an *outdated but
    real* input version (a real, past version that has since been
    superseded), which ``add_version`` accepts and only surfaces on its
    result.
    """


class ArtifactGraphStore:
    """One branch's mutable timeline of versioned artifacts, typed edges,
    staleness, and the action/evidence/cost log that goes with it.

    One instance == one branch (``branch_id``). Comparing repair actions
    from the *same* starting state (revision direction 4: "동일한 상태에서
    서로 다른 repair action으로 분기") means saving one ``Checkpoint`` and
    reconstructing *multiple independent* stores from it via
    ``from_checkpoint``, each given its own ``branch_id`` -- never
    interleaving two branches' writes into one store, which would make
    "the current latest version" ambiguous.
    """

    def __init__(self, branch_id: str = "main") -> None:
        self.branch_id = branch_id
        self._versions: Dict[str, List[StoredArtifact]] = {}
        self._edges: List[DependencyEdge] = []
        self._stale_ids: Set[str] = set()
        self._invalidation_history: List[InvalidationRecord] = []
        self.log = RunLog()
        self._checkpoints: Dict[str, Checkpoint] = {}

    # -- read access ----------------------------------------------------

    def _live_status(self, artifact_id: str) -> ArtifactStatus:
        """The current live staleness verdict for ``artifact_id``, derived
        solely from ``_stale_ids`` -- the one source of truth (see this
        module's docstring, invariant 2). Every place that needs "is this
        artifact currently stale, as a status value" computes it through
        this single method, so there is exactly one place that can get the
        derivation wrong."""
        return ArtifactStatus.STALE if artifact_id in self._stale_ids else ArtifactStatus.FRESH

    def _with_live_status(self, stored: StoredArtifact, artifact_id: str) -> StoredArtifact:
        """A deep copy of ``stored`` with ``.status`` overlaid from
        ``_live_status``. Only meaningful for an artifact's current latest
        version, which can still change status going forward; a superseded
        historical version's status is frozen for good the moment it stops
        being latest (see ``add_version``'s pre-append freeze step) and is
        never touched here."""
        copy_ = stored.model_copy(deep=True)
        copy_.status = self._live_status(artifact_id)
        return copy_

    def latest(self, artifact_id: str) -> Optional[StoredArtifact]:
        """A **deep copy** of the current latest version, with ``.status``
        always reflecting the *current* ``_stale_ids`` verdict (never a
        value frozen at creation time), or ``None`` if ``artifact_id`` has
        never been stored."""
        versions = self._versions.get(artifact_id)
        return self._with_live_status(versions[-1], artifact_id) if versions else None

    def get_version(self, artifact_id: str, version: int) -> Optional[StoredArtifact]:
        """A **deep copy** of one historical version.

        If ``version`` is the artifact's current latest, ``.status`` is
        live-overlaid exactly like ``latest()`` (it can still change). If
        ``version`` has since been superseded, its ``.status`` was already
        permanently frozen -- at the moment it stopped being latest, by
        ``add_version``'s pre-append freeze step -- to whatever
        ``_live_status`` said right then, so it is returned as-is: a
        version that really was stale right up until it got superseded
        reads ``STALE`` forever after, not reset to ``FRESH`` just because
        it is no longer current. This is also exactly what
        ``save_checkpoint`` writes into a non-latest ``Checkpoint.artifacts``
        entry, so the two can never disagree about a historical version's
        status the way an earlier version of this method (which forced
        ``FRESH`` on every non-latest read) did.
        """
        versions = self._versions.get(artifact_id, [])
        for stored in versions:
            if stored.ref.version != version:
                continue
            if stored.ref.version == versions[-1].ref.version:
                return self._with_live_status(stored, artifact_id)
            return stored.model_copy(deep=True)
        return None

    def is_stale(self, artifact_id: str) -> bool:
        return artifact_id in self._stale_ids

    @property
    def stale_artifacts(self) -> frozenset:
        return frozenset(self._stale_ids)

    @property
    def edges(self) -> List[DependencyEdge]:
        """A deep copy of every recorded edge, across all of history --
        mutating an entry in the returned list, or the list itself, has no
        effect on the store. Includes edges that no longer describe a
        *current* relation (see ``_current_dependents``) -- this is the
        full audit trail, not the live adjacency."""
        return copy.deepcopy(self._edges)

    @property
    def invalidation_history(self) -> List[InvalidationRecord]:
        """A deep copy of the full invalidation audit trail."""
        return copy.deepcopy(self._invalidation_history)

    def _validate_artifact_ref(self, ref: ArtifactRef) -> None:
        """Every ``ArtifactRef`` a caller hands in as a declared input (to
        ``add_version``) or edge endpoint (to ``add_inferred_edge``) must
        point at a version that was genuinely stored, with the kind it
        actually claims. A version number that was never stored, or a kind
        that doesn't match what that artifact was actually stored as, is a
        caller bug -- the resulting edge/version would reference content
        that ``from_checkpoint`` could never actually locate -- so both
        are hard errors, raised here before any state is mutated.

        Deliberately permits a real, past (non-latest) version: being
        outdated relative to the current latest is a *separate*, non-fatal
        concern surfaced via ``AddVersionResult.used_outdated_inputs``, not
        rejected here.
        """
        versions = self._versions.get(ref.artifact_id)
        if not versions:
            raise UnknownArtifactError(f"'{ref.artifact_id}' has never been stored.")
        exact = next((v for v in versions if v.ref.version == ref.version), None)
        if exact is None:
            raise UnknownArtifactError(
                f"'{ref.artifact_id}'@v{ref.version} was never stored "
                f"(stored versions: {[v.ref.version for v in versions]})."
            )
        if exact.ref.kind != ref.kind:
            raise ValueError(
                f"'{ref.artifact_id}' is stored as kind {exact.ref.kind!r}; "
                f"the given ref claims kind {ref.kind!r}."
            )

    # -- writing new versions --------------------------------------------

    def add_version(
        self,
        artifact_id: str,
        kind: ArtifactKind,
        content: Optional[str] = None,
        *,
        read_input_versions: Optional[List[ArtifactRef]] = None,
        produced_by: Optional[str] = None,
    ) -> AddVersionResult:
        """Record a new version of ``artifact_id``, create typed EXPLICIT
        edges from each declared input, and propagate staleness forward to
        every artifact currently reachable from ``artifact_id`` (revision
        direction 1's conservative candidate set ``I(R) = R ∪ Desc_G(R)``,
        minus ``R`` itself).

        Each entry in ``read_input_versions`` must name a real, previously
        stored ``(artifact_id, version)`` of the right ``kind`` --
        validated by ``_validate_artifact_ref`` *before* any state is
        touched, so a bad input never leaves the store half-mutated.
        Beyond that hard check, two independent, non-fatal conditions are
        detected and surfaced rather than rejected:

        - **outdated**: the input's version is not that artifact's current
          latest (something newer exists that this production did not
          read) -> ``AddVersionResult.used_outdated_inputs``.
        - **stale**: the input artifact is *currently* ``is_stale()`` --
          even if the exact version read happens to still be its latest,
          because nothing has regenerated it yet since *its own* upstream
          changed -> ``AddVersionResult.used_stale_inputs``.

        If either condition holds for any input, the artifact being
        produced right now is **born stale, not fresh**
        (``AddVersionResult.born_stale``): it was built from ground this
        store already knows is unreliable, so simply being newly-written
        does not make it trustworthy. This is what makes
        ``is_stale()``/``.status`` correct for both of the bugs a caught
        review found: (a) building C from an already-stale B used to make
        C look fresh and its Evidence live, even though B's own upstream
        had already changed; (b) "regenerating" B again from the same old,
        outdated A used to clear B's staleness simply by virtue of writing
        a new version, even though nothing was actually fixed. Neither
        case is rejected -- a producer deliberately pinning a stale/old
        input remains a permitted scenario -- but neither is silently
        treated as healthy, either.
        """
        read_input_versions = [ref.model_copy(deep=True) for ref in (read_input_versions or [])]

        existing_versions = self._versions.get(artifact_id, [])
        if existing_versions and existing_versions[-1].ref.kind != kind:
            raise ValueError(
                f"Artifact '{artifact_id}' was previously stored as kind "
                f"{existing_versions[-1].ref.kind!r}; got {kind!r} instead."
            )

        next_version = (existing_versions[-1].ref.version + 1) if existing_versions else 1

        used_outdated_inputs: List[ArtifactRef] = []
        used_stale_inputs: List[ArtifactRef] = []
        for input_ref in read_input_versions:
            self._validate_artifact_ref(input_ref)
            current_input = self._versions[input_ref.artifact_id][-1]
            if input_ref.version != current_input.ref.version:
                used_outdated_inputs.append(input_ref)
            if input_ref.artifact_id in self._stale_ids:
                used_stale_inputs.append(input_ref)

        born_stale = bool(used_outdated_inputs) or bool(used_stale_inputs)

        # Freeze the outgoing latest version's status as permanent history,
        # right before it stops being latest: whatever `_live_status` says
        # about `artifact_id` at this exact moment (before any of this
        # call's own bookkeeping for the NEW version below) is the true
        # "last known state" of the version about to be superseded -- see
        # `get_version`'s docstring for why this must happen here, not be
        # read back later from a value forced at read time.
        if existing_versions:
            existing_versions[-1].status = self._live_status(artifact_id)

        new_ref = ArtifactRef(
            artifact_id=artifact_id,
            version=next_version,
            kind=kind,
            read_input_versions=list(read_input_versions),
        )
        stored = StoredArtifact(
            ref=new_ref.model_copy(deep=True),
            content=content,
            produced_by=produced_by,
            status=ArtifactStatus.STALE if born_stale else ArtifactStatus.FRESH,
            branch_id=self.branch_id,
        )
        self._versions.setdefault(artifact_id, []).append(stored)

        if born_stale:
            self._stale_ids.add(artifact_id)
            unreliable_ids = sorted({r.artifact_id for r in used_outdated_inputs + used_stale_inputs})
            self._invalidation_history.append(
                InvalidationRecord(
                    changed_artifact=new_ref.model_copy(deep=True),
                    affected_artifact_id=artifact_id,
                    reason=(
                        f"produced from an outdated and/or already-stale input "
                        f"({', '.join(unreliable_ids)})"
                    ),
                )
            )
        else:
            self._stale_ids.discard(artifact_id)

        for input_ref in read_input_versions:
            edge = DependencyEdge(
                from_artifact=input_ref.model_copy(deep=True),
                to_artifact=new_ref.model_copy(deep=True),
                relation=DependencyRelationKind.EXPLICIT,
            )
            self._edges.append(edge)

        newly_stale = sorted(self._propagate_staleness(new_ref, artifact_id))
        return AddVersionResult(
            ref=new_ref.model_copy(deep=True),
            born_stale=born_stale,
            newly_stale_artifact_ids=newly_stale,
            used_outdated_inputs=[r.model_copy(deep=True) for r in used_outdated_inputs],
            used_stale_inputs=[r.model_copy(deep=True) for r in used_stale_inputs],
        )

    def add_inferred_edge(
        self, from_artifact: ArtifactRef, to_artifact: ArtifactRef, reason: str
    ) -> DependencyEdge:
        """Record a relationship discovered by some mechanism other than a
        Producer's own declared ``read_input_versions`` -- e.g. a reference
        scanner finding that one artifact's content cites another's label,
        or a human noticing *after the fact* that ``to_artifact`` actually
        used ``from_artifact`` all along even though that was never
        declared when ``to_artifact`` was produced. This is revision
        direction 1's INFERRED relation kind.

        Both endpoints are validated exactly like ``add_version``'s
        declared inputs (``_validate_artifact_ref``: must be a real,
        previously stored ``(artifact_id, version)`` of the claimed
        ``kind``). ``reason`` is stored on the edge itself
        (``DependencyEdge.reason``) so it survives a checkpoint
        save/restore and remains auditable later -- an earlier version of
        this method accepted ``reason`` but discarded it.

        **Retroactive invalidation.** Recording a structural fact is not
        itself the same event as ``from_artifact``'s content changing --
        simply adding knowledge of a relationship between two artifacts
        that are both currently fine must not mark anything stale just for
        being newly connected. But if ``to_artifact`` names the target's
        *current* latest version, and ``from_artifact`` turns out to be
        outdated or already-stale relative to the CURRENT state, this edge
        reveals that ``to_artifact`` (and everything reachable from it) has
        secretly been resting on unreliable ground the entire time it
        looked fine -- exactly the case this call exists for: the relation
        was missed when it mattered, and is only being filled in now. That
        case **does** retroactively invalidate ``to_artifact`` and its
        descendants, via the same ``_invalidate`` machinery
        ``add_version``'s own propagation uses -- the only difference is
        ``include_self=True`` here, since ``to_artifact`` itself was never
        rewritten and must be marked stale directly, not merely have its
        descendants affected.

        Two cases are deliberately exempt from this side effect, per the
        distinction above: (1) ``to_artifact`` is **not** the current
        latest version of its artifact_id -- this is a purely historical
        annotation about a superseded version, with nothing live to
        invalidate; (2) ``from_artifact`` is itself current and non-stale
        -- the relationship is healthy, so there is nothing to retroactively
        reveal.
        """
        self._validate_artifact_ref(from_artifact)
        self._validate_artifact_ref(to_artifact)

        edge = DependencyEdge(
            from_artifact=from_artifact.model_copy(deep=True),
            to_artifact=to_artifact.model_copy(deep=True),
            relation=DependencyRelationKind.INFERRED,
            reason=reason,
        )
        self._edges.append(edge)

        to_is_current = (
            to_artifact.version == self._versions[to_artifact.artifact_id][-1].ref.version
        )
        from_is_outdated = (
            from_artifact.version != self._versions[from_artifact.artifact_id][-1].ref.version
        )
        from_is_stale = self.is_stale(from_artifact.artifact_id)

        if to_is_current and (from_is_outdated or from_is_stale):
            unreliable_bits = []
            if from_is_outdated:
                unreliable_bits.append("outdated")
            if from_is_stale:
                unreliable_bits.append("stale")
            self._invalidate(
                to_artifact.artifact_id,
                changed_artifact=from_artifact.model_copy(deep=True),
                reason=(
                    f"inferred edge retroactively revealed that "
                    f"'{to_artifact.artifact_id}' depends on "
                    f"'{from_artifact.artifact_id}'@v{from_artifact.version}, which is "
                    f"{' and '.join(unreliable_bits)} relative to its current state"
                ),
                include_self=True,
            )

        return edge.model_copy(deep=True)

    # -- change impact / staleness ---------------------------------------

    def _current_dependents(self) -> Dict[str, Set[str]]:
        """Adjacency used for impact computation: ``from_artifact_id ->
        {artifact ids that CURRENTLY declare depending on it}``.

        Rebuilt from ``_edges``/``_versions`` on every call rather than
        incrementally maintained -- a caught review bug: an edge whose
        ``to_artifact`` is not that artifact's *current* latest version
        describes a relation that has since been **superseded** (the
        artifact was regenerated again, possibly with a different input or
        none at all) and must not keep contributing to the live impact
        graph, even though the edge itself stays in ``_edges`` forever as
        an audit trail. Recomputing from scratch -- rather than trying to
        incrementally retract a superseded edge's earlier contribution --
        is what this stage's scale actually needs and is far easier to
        prove correct than the incremental index it replaced, which is
        exactly what went wrong before.
        """
        dependents: Dict[str, Set[str]] = {}
        for edge in self._edges:
            to_id = edge.to_artifact.artifact_id
            versions = self._versions.get(to_id)
            if not versions:
                continue
            if edge.to_artifact.version == versions[-1].ref.version:
                dependents.setdefault(edge.from_artifact.artifact_id, set()).add(to_id)
        return dependents

    def impact_candidates(self, artifact_id: str) -> Set[str]:
        """``I(R) = R ∪ Desc_G(R)`` (revision direction 1): every artifact
        transitively reachable from ``artifact_id`` by following CURRENTLY
        applicable EXPLICIT or INFERRED edges forward, plus ``artifact_id``
        itself -- see ``_current_dependents`` for what "currently
        applicable" excludes.

        Pure query: calling this any number of times changes nothing on
        ``self`` (unlike ``add_version``, which is where staleness is
        actually applied -- see ``RepairPolicy.decide()``'s docstring in
        ``interfaces.py`` for why a "compute the candidates" step must stay
        just as side-effect-free here as a policy decision does). Cycle-safe
        by construction (visited set): the underlying graph is not assumed
        acyclic.
        """
        dependents = self._current_dependents()
        visited: Set[str] = {artifact_id}
        frontier = [artifact_id]
        while frontier:
            current = frontier.pop()
            for dependent in dependents.get(current, ()):
                if dependent not in visited:
                    visited.add(dependent)
                    frontier.append(dependent)
        return visited

    def _invalidate(
        self,
        artifact_id: str,
        changed_artifact: ArtifactRef,
        reason: str,
        *,
        include_self: bool,
    ) -> Set[str]:
        """Mark every artifact in ``impact_candidates(artifact_id)`` stale,
        recording one ``InvalidationRecord`` per affected id, attributing
        the cause to ``changed_artifact``.

        ``include_self`` is the one thing that differs between this
        helper's two call sites, and is required (no default) so neither
        can silently fall back to the other's semantics:

        - ``add_version``'s own propagation (``_propagate_staleness``)
          passes ``False`` -- the artifact that was just freshly written
          is, by construction, not stale from its own write; only its
          *descendants* are affected.
        - ``add_inferred_edge``'s retroactive path passes ``True`` --
          ``artifact_id`` here was never rewritten; a newly discovered
          relation reveals it was *already* resting on unreliable ground,
          so it must be marked stale directly, not just its descendants.
        """
        affected = self.impact_candidates(artifact_id)
        if not include_self:
            affected = affected - {artifact_id}
        for affected_id in affected:
            self._stale_ids.add(affected_id)
            self._invalidation_history.append(
                InvalidationRecord(
                    changed_artifact=changed_artifact,
                    affected_artifact_id=affected_id,
                    reason=reason,
                )
            )
        return affected

    def _propagate_staleness(
        self, changed_ref: ArtifactRef, changed_artifact_id: str
    ) -> Set[str]:
        return self._invalidate(
            changed_artifact_id,
            changed_artifact=changed_ref.model_copy(deep=True),
            reason=(
                f"upstream artifact '{changed_artifact_id}' changed to "
                f"version {changed_ref.version}"
            ),
            include_self=False,
        )

    # -- evidence ---------------------------------------------------------

    def record_evidence(self, evidence: Evidence) -> None:
        self.log.log_evidence(evidence)

    def _ref_is_live(self, ref: ArtifactRef) -> bool:
        """Whether ``ref`` still names a trustworthy, current version --
        the same version-match-AND-non-stale rule ``live_evidence`` applies
        to ``checked_artifact``, applied here to one arbitrary
        ``Evidence.input_versions`` entry. ``self.latest()``/``self.is_stale()``
        are both pure reads -- no side effect from calling this."""
        current = self.latest(ref.artifact_id)
        if current is None:
            return False
        if current.ref.version != ref.version:
            return False
        return not self.is_stale(ref.artifact_id)

    def live_evidence(self, artifact_id: str) -> List[Evidence]:
        """The single currently-valid verdict per ``constraint_id`` checked
        against ``artifact_id`` -- distinct from the full history (see
        ``evidence_history``). Three conditions, all required:

        1. Checked against the artifact's current latest version, AND
        2. recorded while that artifact was not (and has not since become)
           stale -- including an artifact that was born stale (see
           ``add_version``) and so was never trustworthy at any point in
           its current version's lifetime. The staleness check is not
           redundant with the version check: a shared-definition change
           marks its dependents stale *without* bumping their own version
           number, so Evidence checked against their unchanged version
           would still version-match even though it no longer reflects a
           trustworthy state.
        3. every one of ``Evidence.input_versions`` -- whatever *other*
           artifacts the check actually consulted beyond ``checked_artifact``
           itself (e.g. a SCHEMA_CONTRACT's ``applies_to`` refs, a
           RECOMPUTE's source-data ref) -- is *itself* still the current,
           non-stale version. Added after a Stage 3 review found that a
           PASS checked against a source artifact's v1 kept reading as
           live evidence even after that source moved to v2: condition 1/2
           alone only protect against *the checked artifact itself*
           changing, not against one of the other things the verdict
           actually depended on changing.

        A fourth condition handles **re-checking the same constraint**: if
        ``constraint_id`` was checked more than once against the current
        version (e.g. FAIL, then -- after an unrelated missing artifact was
        added -- PASS), only the most recently logged entry for that
        ``constraint_id`` is returned, never both; ``self.log.evidence`` is
        append-only in chronological order, so the last matching entry per
        ``constraint_id`` *is* the most recent one, with no separate
        timestamp/sequence field required. The earlier version of this
        method had no such dedup at all and could return a stale FAIL
        alongside the PASS that superseded it, with nothing on either
        entry indicating which one was current.

        Returns deep copies (``self.log.evidence`` already deep-copies) --
        mutating an entry in the returned list has no effect on this
        store's log.
        """
        current = self.latest(artifact_id)
        if current is None or self.is_stale(artifact_id):
            return []

        matching = [
            item
            for item in self.log.evidence
            if item.checked_artifact.artifact_id == artifact_id
            and item.checked_artifact.version == current.ref.version
            and all(self._ref_is_live(ref) for ref in item.input_versions)
        ]

        latest_by_constraint: Dict[str, Evidence] = {}
        for item in matching:
            latest_by_constraint[item.constraint_id] = item
        return list(latest_by_constraint.values())

    def evidence_history(self, artifact_id: str) -> List[Evidence]:
        """Every ``Evidence`` ever logged against *any* version of
        ``artifact_id`` -- undeduped, unfiltered by current staleness or
        input freshness. The full record, as opposed to ``live_evidence``'s
        single current verdict per constraint; added after a Stage 3
        review explicitly asked that the two be distinguishable rather
        than only the deduped view being reachable at all."""
        return [item for item in self.log.evidence if item.checked_artifact.artifact_id == artifact_id]

    # -- actions / cost -----------------------------------------------------

    def record_action(
        self, action: RepairAction, triggering_evidence: Optional[List[Evidence]] = None
    ) -> ActionLogEntry:
        """Log that ``action`` actually began executing. Must be called by
        an executor, never from inside a ``RepairPolicy.decide()`` -- see
        that Protocol's docstring in ``interfaces.py``."""
        return self.log.log_action(action, self.branch_id, triggering_evidence)

    def record_cost(self, cost: CostRecord) -> None:
        self.log.log_cost(cost)

    def record_tool_call(self, call: ToolCall, observation: ToolObservation) -> ToolCallLogEntry:
        """Log that ``call`` was executed and ``observation`` is what came
        back -- logged by ``ToolExecutor.execute()`` for every call,
        admitted or rejected (see ``ToolCallLogEntry``'s docstring)."""
        return self.log.log_tool_call(call, observation)

    # -- checkpoint / replay ----------------------------------------------

    def save_checkpoint(self, label: Optional[str] = None) -> Checkpoint:
        """Snapshot every artifact version, edge, stale id, and the full
        invalidation/action/evidence/cost history into a self-contained
        ``Checkpoint``.

        Each artifact's latest stored version has its ``.status`` corrected
        via ``_live_status`` at save time -- the same overlay
        ``latest()``/``get_version()`` apply on every read -- so the raw
        ``Checkpoint.artifacts`` data is itself already correct, not
        dependent on going through an accessor later to look right. Every
        *non*-latest (superseded) version's ``.status`` was already frozen
        permanently when it stopped being latest (``add_version``'s
        pre-append freeze step, see ``get_version``'s docstring), so it is
        copied as-is here too -- this is exactly what makes
        ``get_version(artifact_id, old_version).status`` and
        ``checkpoint.artifacts[artifact_id][i].status`` agree for the same
        historical version; an earlier version of this method only
        corrected the last entry while ``get_version`` separately forced
        every non-latest read to ``FRESH``, so the two disagreed whenever a
        version had genuinely been stale right up until it was superseded.

        The ``Checkpoint`` cached internally (for ``restore_checkpoint``)
        and the one returned to the caller are two *separately* deep-copied
        objects, neither aliased to the other: an earlier version returned
        the exact object it had just cached, so mutating the caller's copy
        silently corrupted what a later ``restore_checkpoint`` of the same
        id would produce.
        """
        materialized_artifacts: Dict[str, List[StoredArtifact]] = {}
        for artifact_id, versions in self._versions.items():
            copied_versions = copy.deepcopy(versions)
            copied_versions[-1].status = self._live_status(artifact_id)
            materialized_artifacts[artifact_id] = copied_versions

        checkpoint = Checkpoint(
            checkpoint_id=str(uuid.uuid4()),
            label=label,
            source_branch_id=self.branch_id,
            artifacts=materialized_artifacts,
            edges=copy.deepcopy(self._edges),
            stale_artifact_ids=sorted(self._stale_ids),
            invalidation_history=copy.deepcopy(self._invalidation_history),
            actions=self.log.actions,
            evidence=self.log.evidence,
            costs=self.log.costs,
            tool_calls=self.log.tool_calls,
        )
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint.model_copy(deep=True)
        return checkpoint.model_copy(deep=True)

    def restore_checkpoint(self, checkpoint_id: str, branch_id: str) -> "ArtifactGraphStore":
        """Reconstruct a new, independent store from a checkpoint this
        instance itself saved earlier.

        ``branch_id`` must be supplied explicitly every time: any write the
        new store makes next (in particular, re-calling a Producer, which
        may call an LLM again and get a different result -- revision
        direction 4) must be attributable to a specific branch, never
        silently folded back into the source branch's own history. See the
        Stage 2 completion criterion: "LLM을 다시 호출하는 확률적 후속
        실행은 별도 run으로 기록한다"."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint is None:
            raise KeyError(f"No checkpoint '{checkpoint_id}' on this store.")
        return ArtifactGraphStore.from_checkpoint(checkpoint, branch_id)

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint, branch_id: str) -> "ArtifactGraphStore":
        """Reconstruct a store purely from a ``Checkpoint``'s data -- does
        not require the original store instance that produced it (a real
        ``Checkpoint`` could be serialized via ``checkpoint.model_dump_json()``
        and handed to a different process entirely; see
        ``test/core/test_artifact_graph.py::test_from_checkpoint_rebuilds_from_serialized_json_alone``).

        No separate adjacency index needs rebuilding: ``impact_candidates``
        recomputes ``_current_dependents`` fresh from ``_edges``/
        ``_versions`` on every call, so restoring just those two fields is
        sufficient and cannot go stale the way an incrementally-maintained
        index could.
        """
        store = cls(branch_id=branch_id)
        store._versions = copy.deepcopy(checkpoint.artifacts)
        store._edges = copy.deepcopy(checkpoint.edges)
        store._stale_ids = set(checkpoint.stale_artifact_ids)
        store._invalidation_history = copy.deepcopy(checkpoint.invalidation_history)
        store.log.replace_all(
            checkpoint.actions, checkpoint.evidence, checkpoint.costs, checkpoint.tool_calls
        )
        return store


__all__ = ["ArtifactGraphStore", "UnknownArtifactError"]

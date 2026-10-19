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
)


class UnknownArtifactError(ValueError):
    """Raised when a caller references an ``artifact_id`` the store has
    never recorded a version for -- e.g. claiming to have read it as an
    input, or declaring an inferred edge to/from it.

    Reading or depending on something that was never produced is a caller
    bug, not a reachable research scenario, so this is a hard error rather
    than a silently accepted edge case -- unlike an *outdated but real*
    input version, which ``add_version`` accepts and only surfaces on its
    result (see ``AddVersionResult.used_outdated_inputs``).
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
        self._dependents: Dict[str, Set[str]] = {}
        self._stale_ids: Set[str] = set()
        self.invalidation_history: List[InvalidationRecord] = []
        self.log = RunLog()
        self._checkpoints: Dict[str, Checkpoint] = {}

    # -- read access ----------------------------------------------------

    def latest(self, artifact_id: str) -> Optional[StoredArtifact]:
        """A **deep copy** of the current latest version, or ``None`` if
        ``artifact_id`` has never been stored. A versioned artifact's past
        (and current) state must be immutable history -- the only way to
        change it is ``add_version`` minting a new version -- so this
        deliberately never hands out the live internal object; mutating
        the returned ``StoredArtifact`` has no effect on the store."""
        versions = self._versions.get(artifact_id)
        return versions[-1].model_copy(deep=True) if versions else None

    def get_version(self, artifact_id: str, version: int) -> Optional[StoredArtifact]:
        """A **deep copy** of one historical version -- same non-mutating
        guarantee as ``latest()``."""
        for stored in self._versions.get(artifact_id, []):
            if stored.ref.version == version:
                return stored.model_copy(deep=True)
        return None

    def is_stale(self, artifact_id: str) -> bool:
        return artifact_id in self._stale_ids

    @property
    def stale_artifacts(self) -> frozenset:
        return frozenset(self._stale_ids)

    @property
    def edges(self) -> List[DependencyEdge]:
        """A deep copy of every recorded edge -- mutating an entry in the
        returned list, or the list itself, has no effect on the store."""
        return copy.deepcopy(self._edges)

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
        minus ``R`` itself -- ``R`` is this artifact, which was just
        freshly produced, so it is never marked stale by its own write).

        An input in ``read_input_versions`` whose ``version`` is not that
        artifact's current latest -- something newer exists that this
        production did not actually read -- is **not** rejected: a
        producer legitimately pinning an older version (e.g. a controlled
        checkpoint-branch comparison) is a real scenario this stage does
        not forbid. It is surfaced on the returned
        ``AddVersionResult.used_outdated_inputs`` instead, so that case is
        never silently indistinguishable from reading the true current
        state.
        """
        read_input_versions = read_input_versions or []
        existing_versions = self._versions.get(artifact_id, [])
        if existing_versions and existing_versions[-1].ref.kind != kind:
            raise ValueError(
                f"Artifact '{artifact_id}' was previously stored as kind "
                f"{existing_versions[-1].ref.kind!r}; got {kind!r} instead."
            )

        next_version = (existing_versions[-1].ref.version + 1) if existing_versions else 1

        used_outdated_inputs: List[ArtifactRef] = []
        for input_ref in read_input_versions:
            current_input = self.latest(input_ref.artifact_id)
            if current_input is None:
                raise UnknownArtifactError(
                    f"'{artifact_id}' declares reading '{input_ref.artifact_id}' "
                    f"@v{input_ref.version}, but no version of it has ever been stored."
                )
            if input_ref.version != current_input.ref.version:
                used_outdated_inputs.append(input_ref)

        new_ref = ArtifactRef(
            artifact_id=artifact_id,
            version=next_version,
            kind=kind,
            read_input_versions=list(read_input_versions),
        )
        stored = StoredArtifact(
            ref=new_ref,
            content=content,
            produced_by=produced_by,
            status=ArtifactStatus.FRESH,
            branch_id=self.branch_id,
        )
        self._versions.setdefault(artifact_id, []).append(stored)
        self._stale_ids.discard(artifact_id)

        for input_ref in read_input_versions:
            edge = DependencyEdge(
                from_artifact=input_ref,
                to_artifact=new_ref,
                relation=DependencyRelationKind.EXPLICIT,
            )
            self._edges.append(edge)
            self._dependents.setdefault(input_ref.artifact_id, set()).add(artifact_id)

        newly_stale = sorted(self._propagate_staleness(new_ref, artifact_id))
        return AddVersionResult(
            ref=new_ref,
            newly_stale_artifact_ids=newly_stale,
            used_outdated_inputs=used_outdated_inputs,
        )

    def add_inferred_edge(
        self, from_artifact: ArtifactRef, to_artifact: ArtifactRef, reason: str
    ) -> DependencyEdge:
        """Record a relationship discovered by some mechanism other than a
        Producer's own declared ``read_input_versions`` -- e.g. a reference
        scanner finding that one artifact's content cites another's label.
        This is revision direction 1's INFERRED relation kind.

        Both artifacts must already exist (same invariant ``add_version``
        enforces for ``read_input_versions``). Unlike ``add_version``, this
        does **not** retroactively mark ``to_artifact`` stale: recording a
        previously-unknown structural fact is a different event from
        ``from_artifact``'s content actually changing, and conflating the
        two would mark things stale every time new inference knowledge is
        added, not only when content changes.
        """
        if self.latest(from_artifact.artifact_id) is None:
            raise UnknownArtifactError(f"'{from_artifact.artifact_id}' has never been stored.")
        if self.latest(to_artifact.artifact_id) is None:
            raise UnknownArtifactError(f"'{to_artifact.artifact_id}' has never been stored.")

        edge = DependencyEdge(
            from_artifact=from_artifact,
            to_artifact=to_artifact,
            relation=DependencyRelationKind.INFERRED,
        )
        self._edges.append(edge)
        self._dependents.setdefault(from_artifact.artifact_id, set()).add(
            to_artifact.artifact_id
        )
        return edge

    # -- change impact / staleness ---------------------------------------

    def impact_candidates(self, artifact_id: str) -> Set[str]:
        """``I(R) = R ∪ Desc_G(R)`` (revision direction 1): every artifact
        transitively reachable from ``artifact_id`` by following recorded
        EXPLICIT or INFERRED edges forward, plus ``artifact_id`` itself.

        Pure query: calling this any number of times changes nothing on
        ``self`` (unlike ``add_version``, which is where staleness is
        actually applied -- see ``RepairPolicy.decide()``'s docstring in
        ``interfaces.py`` for why a "compute the candidates" step must stay
        just as side-effect-free here as a policy decision does). Cycle-safe
        by construction (visited set): the underlying graph is not assumed
        acyclic.
        """
        visited: Set[str] = {artifact_id}
        frontier = [artifact_id]
        while frontier:
            current = frontier.pop()
            for dependent in self._dependents.get(current, ()):
                if dependent not in visited:
                    visited.add(dependent)
                    frontier.append(dependent)
        return visited

    def _propagate_staleness(
        self, changed_ref: ArtifactRef, changed_artifact_id: str
    ) -> Set[str]:
        descendants = self.impact_candidates(changed_artifact_id) - {changed_artifact_id}
        for affected_id in descendants:
            self._stale_ids.add(affected_id)
            self.invalidation_history.append(
                InvalidationRecord(
                    changed_artifact=changed_ref,
                    affected_artifact_id=affected_id,
                    reason=(
                        f"upstream artifact '{changed_artifact_id}' changed to "
                        f"version {changed_ref.version}"
                    ),
                )
            )
        return descendants

    # -- evidence ---------------------------------------------------------

    def record_evidence(self, evidence: Evidence) -> None:
        self.log.log_evidence(evidence)

    def live_evidence(self, artifact_id: str) -> List[Evidence]:
        """Evidence that still speaks to the artifact's *current* state:
        checked against the artifact's current latest version, AND
        recorded while that artifact was not (and has not since become)
        stale.

        The staleness check is not redundant with the version check: a
        shared-definition change marks its dependents stale *without*
        bumping their own version number (their content hasn't changed yet
        -- only the ground it rests on has), so Evidence checked against
        their unchanged version would still version-match even though it no
        longer reflects a trustworthy state. Both conditions are required
        for Evidence to count as live.
        """
        current = self.latest(artifact_id)
        if current is None or self.is_stale(artifact_id):
            return []
        return [
            item
            for item in self.log.evidence
            if item.checked_artifact.artifact_id == artifact_id
            and item.checked_artifact.version == current.ref.version
        ]

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

    # -- checkpoint / replay ----------------------------------------------

    def save_checkpoint(self, label: Optional[str] = None) -> Checkpoint:
        """Snapshot every artifact version, edge, stale id, and the full
        invalidation/action/evidence/cost history into a self-contained
        ``Checkpoint`` -- no live references back into ``self``. Verified
        by ``from_checkpoint`` reconstructing a fully independent store
        from it (see ``test/core/test_artifact_graph.py``'s
        checkpoint-independence tests)."""
        checkpoint = Checkpoint(
            checkpoint_id=str(uuid.uuid4()),
            label=label,
            source_branch_id=self.branch_id,
            artifacts=copy.deepcopy(self._versions),
            edges=copy.deepcopy(self._edges),
            stale_artifact_ids=sorted(self._stale_ids),
            invalidation_history=copy.deepcopy(self.invalidation_history),
            actions=copy.deepcopy(self.log.actions),
            evidence=copy.deepcopy(self.log.evidence),
            costs=copy.deepcopy(self.log.costs),
        )
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        return checkpoint

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

        The dependents index is rebuilt from ``checkpoint.edges`` rather
        than trusted as pre-computed state, so a hand-built or
        round-tripped ``Checkpoint`` reconstructs correctly even without
        the live index that originally produced it.
        """
        store = cls(branch_id=branch_id)
        store._versions = copy.deepcopy(checkpoint.artifacts)
        store._edges = copy.deepcopy(checkpoint.edges)
        store._stale_ids = set(checkpoint.stale_artifact_ids)
        store.invalidation_history = copy.deepcopy(checkpoint.invalidation_history)
        store.log.actions = copy.deepcopy(checkpoint.actions)
        store.log.evidence = copy.deepcopy(checkpoint.evidence)
        store.log.costs = copy.deepcopy(checkpoint.costs)
        for edge in store._edges:
            store._dependents.setdefault(edge.from_artifact.artifact_id, set()).add(
                edge.to_artifact.artifact_id
            )
        return store


__all__ = ["ArtifactGraphStore", "UnknownArtifactError"]

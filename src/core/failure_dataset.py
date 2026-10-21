"""Failure classification + repair gain/damage/cost (Stage 6 of the
"Revision 2026.10.15" plan: "실패·행동 데이터셋과 human gold 구축").

Builds on Stage 2's ``ArtifactGraphStore`` (dependency-graph queries) and
Stage 4's ``BranchExecutionResult``/``BranchComparisonResult`` (checkpoint-
branch repair outcomes) -- this module adds no new execution mechanism of
its own, only classification and aggregation over what those stages
already produce. See ``docs/stage6_failure_dataset.md`` for the full
design and the scope decisions confirmed with the user before this was
written (mechanism + Protocol-conforming-fake tests this round; a real
pilot against the local Qwen3.6 pipeline, to actually populate a dataset
with natural failures, is a deferred follow-up -- same discipline Stages
3-5 each used).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.artifact_graph import ArtifactGraphStore, UnknownArtifactError
from src.core.schemas import (
    ArtifactKind,
    BranchExecutionResult,
    CandidateActionRecord,
    Checkpoint,
    ContentPreservationScore,
    DependencyRelationKind,
    Evidence,
    FailureDifficultyProfile,
    FailureOrigin,
    FailureRecord,
    InjectedFault,
    RepairGainScore,
    RepairOutcome,
)

_NON_SHARED_KINDS = frozenset({ArtifactKind.BLOCK, ArtifactKind.SECTION, ArtifactKind.DOCUMENT})


def _dependency_depth(store: ArtifactGraphStore, artifact_id: str, *, _stack: frozenset = frozenset()) -> int:
    """Longest current upstream chain leading into ``artifact_id``, 0 if it
    has no current dependencies. Cycle-safe like ``impact_candidates``:
    revisiting an artifact already on the current recursion path
    contributes 0 extra depth from that edge rather than recursing forever
    -- true longest-*simple*-path in a graph with cycles is NP-hard in
    general, and the dependency graph is explicitly not assumed acyclic
    (``ArtifactGraphStore.impact_candidates``'s own docstring), so this is
    a deliberately bounded, terminating approximation, not an exact
    longest-path solver.
    """
    deps = store.direct_dependencies(artifact_id)
    if not deps:
        return 0
    next_stack = _stack | {artifact_id}
    return 1 + max(
        (0 if dep in next_stack else _dependency_depth(store, dep, _stack=next_stack)) for dep in deps
    )


def classify_failure_difficulty(store: ArtifactGraphStore, artifact_id: str) -> FailureDifficultyProfile:
    """The four axes Stage 6's checklist names
    ("의존 깊이·fan-out·공유 정의 사용·관계 누락에 따라 분류한다"), computed from
    ``store``'s current dependency-graph state around ``artifact_id``.

    Raises ``UnknownArtifactError`` if ``artifact_id`` was never stored --
    a failure must be classified against a real, previously-produced
    artifact, never a guessed/fabricated one.
    """
    current = store.latest(artifact_id)
    if current is None:
        raise UnknownArtifactError(f"'{artifact_id}' has never been stored; cannot classify its difficulty.")

    fan_out = len(store.direct_dependents(artifact_id))
    shared_definition_use = current.ref.kind not in _NON_SHARED_KINDS and fan_out > 1
    missing_relations_count = sum(
        1
        for edge in store.edges
        if edge.relation == DependencyRelationKind.INFERRED
        and (edge.from_artifact.artifact_id == artifact_id or edge.to_artifact.artifact_id == artifact_id)
    )
    return FailureDifficultyProfile(
        dependency_depth=_dependency_depth(store, artifact_id),
        fan_out=fan_out,
        shared_definition_use=shared_definition_use,
        missing_relations_count=missing_relations_count,
    )


def build_failure_record(
    store: ArtifactGraphStore,
    *,
    failure_id: str,
    origin: FailureOrigin,
    failing_artifact_id: str,
    evidence: List[Evidence],
    source_checkpoint_id: str,
    injected_fault: Optional[InjectedFault] = None,
    sample_id: Optional[str] = None,
) -> FailureRecord:
    """Assemble one ``FailureRecord`` from live store state --
    ``FailureRecord.model_post_init`` itself enforces that ``origin`` and
    ``injected_fault`` agree (NATURAL <-> absent, INJECTED <-> present), so
    a caller cannot build an internally-contradictory record even by
    mistake."""
    current = store.latest(failing_artifact_id)
    if current is None:
        raise UnknownArtifactError(f"'{failing_artifact_id}' has never been stored.")
    return FailureRecord(
        failure_id=failure_id,
        origin=origin,
        injected_fault=injected_fault,
        failing_artifact=current.ref,
        evidence=list(evidence),
        difficulty=classify_failure_difficulty(store, failing_artifact_id),
        source_checkpoint_id=source_checkpoint_id,
        sample_id=sample_id,
    )


def _merge_final_outcomes(branch: BranchExecutionResult) -> Dict[str, RepairOutcome]:
    """The one authoritative per-artifact outcome across an ENTIRE branch
    (every round, every execution within it) -- merged in true execution
    order (round order, then execution order within a round) so a later
    round's outcome always overwrites an earlier one for the same
    artifact_id, exactly the last-write-wins rule
    ``RepairExecutionResult.final_outcome_by_artifact`` already documents
    for a single execution, extended here across a whole branch's several
    executions. Never re-derives anything by scanning ``regenerated``/
    ``reverified_untouched`` directly -- only reads the one field those
    lists' own docstring says is safe to read.
    """
    merged: Dict[str, RepairOutcome] = {}
    for round_ in branch.rounds:
        for execution in round_.executions:
            merged.update(execution.final_outcome_by_artifact)
    return merged


def _compute_damage(
    before: Checkpoint, after: Checkpoint, touched_artifact_ids: Dict[str, RepairOutcome]
) -> ContentPreservationScore:
    """Compare ``before``/``after`` checkpoint content for every
    artifact_id ``before`` knew about that the branch's cascade never
    touched at all (absent from ``touched_artifact_ids``) -- any content
    difference found there is, by construction, damage unrelated to what
    the branch was actually trying to fix (revision directions 1/4's
    "정상 내용 보존"). An artifact_id ``before`` never had (impossible for
    ``after``'s set of ids to shrink, but a future artifact kind could in
    principle grow it) is simply not checked -- it was not part of the
    "unaffected content" this branch started with.
    """
    newly_broken: List[str] = []
    checked = 0
    for artifact_id, versions in before.artifacts.items():
        if artifact_id in touched_artifact_ids:
            continue
        after_versions = after.artifacts.get(artifact_id)
        if not after_versions:
            continue
        checked += 1
        before_content = versions[-1].content
        after_content = after_versions[-1].content
        if before_content != after_content:
            newly_broken.append(artifact_id)
    preserved_fraction = (checked - len(newly_broken)) / checked if checked > 0 else None
    return ContentPreservationScore(
        preserved_fraction=preserved_fraction,
        newly_broken_artifacts=sorted(newly_broken),
        not_implemented_reason=(
            None if checked > 0 else "no untouched artifact existed in the source checkpoint to compare against"
        ),
    )


def compute_repair_gain_score(before_checkpoint: Checkpoint, branch: BranchExecutionResult) -> RepairGainScore:
    """Revision direction 4's "repair gain, 손상, 총비용" for one branch,
    built entirely from already-computed, already-reviewed Stage 4 fields
    -- ``branch.total_cost_tokens`` directly (never re-summed from
    ``CostRecord``s here), and ``final_outcome_by_artifact`` merged across
    the whole branch (``_merge_final_outcomes``) rather than scanning the
    raw ``regenerated``/``reverified_untouched`` audit lists.
    """
    final_outcomes = _merge_final_outcomes(branch)
    outcome_counts: Dict[str, int] = {}
    for outcome in final_outcomes.values():
        outcome_counts[outcome.value] = outcome_counts.get(outcome.value, 0) + 1

    total = sum(outcome_counts.values())
    if total > 0:
        gain: Optional[float] = outcome_counts.get(RepairOutcome.RESOLVED.value, 0) / total
        not_implemented_reason = None
    else:
        gain = None
        not_implemented_reason = "no artifact in this branch ever reached a final outcome"

    damage = _compute_damage(before_checkpoint, branch.final_checkpoint, final_outcomes)
    return RepairGainScore(
        outcome_counts=outcome_counts,
        gain=gain,
        not_implemented_reason=not_implemented_reason,
        damage=damage,
        total_cost_tokens=branch.total_cost_tokens,
    )


def build_candidate_action_record(
    *,
    record_id: str,
    failure_id: str,
    before_checkpoint: Checkpoint,
    branch: BranchExecutionResult,
) -> CandidateActionRecord:
    """One ``(checkpoint, candidate action, downstream result)`` triple --
    revision direction 4's benchmark-storage unit. ``branch.rounds[0]`` is
    always the action that distinguishes this branch (``BranchExecutionResult``'s
    own docstring); every round after that belongs to the shared downstream
    policy, so ``action``/``branch_label`` here describe only the
    *initiating* choice this record exists to compare against sibling
    branches from the same ``before_checkpoint``.
    """
    if not branch.rounds or not branch.rounds[0].executions:
        raise ValueError(
            f"branch {branch.branch_id!r} has no initial round/execution -- "
            "a CandidateActionRecord needs at least one actually-attempted action."
        )
    initiating_action = branch.rounds[0].executions[0].action
    return CandidateActionRecord(
        record_id=record_id,
        failure_id=failure_id,
        source_checkpoint_id=branch.source_checkpoint_id,
        branch_label=branch.policy_label,
        action=initiating_action,
        outcome=compute_repair_gain_score(before_checkpoint, branch),
        rounds_executed=len(branch.rounds),
    )

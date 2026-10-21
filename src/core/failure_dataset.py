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

**Revised after a user review found 4 real bugs** (full detail in
``docs/stage6_failure_dataset.md`` §... and memory ``docgen_stage_review_rigor``):
``classify_failure_difficulty``'s 4th axis was renamed and given a properly
current-scoped sibling (§ below); ``compute_repair_gain_score``'s gain and
``_compute_damage``'s damage both used a branch-dependent, self-defined
target set (whatever ``final_outcome_by_artifact`` happened to contain)
instead of one COMMON ``target_artifact_ids`` set fixed once per checkpoint
-- see ``RepairGainScore``'s own (rewritten) docstring in ``schemas.py`` for
the full repro of both; and ``build_candidate_action_record`` recorded only
the first of a round's possibly-several initiating actions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from src.core.artifact_graph import ArtifactGraphStore, UnknownArtifactError
from src.core.schemas import (
    ArtifactKind,
    BranchComparisonResult,
    BranchExecutionResult,
    CandidateActionRecord,
    CheckOutcome,
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

# Sentinel bucket key for `RepairGainScore.outcome_counts` -- deliberately
# not a real `RepairOutcome` value. A target artifact absent from
# `RepairExecutionResult.final_outcome_by_artifact` (that schema's own
# docstring: "left_stale", or simply never reached by any round) must
# never be silently dropped from the denominator, and must never be
# promoted to FAIL/INCONCLUSIVE either -- it genuinely has no verdict yet.
LEFT_STALE_OR_UNRESOLVED = "left_stale_or_unresolved"


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


def count_missing_relations_against_reference(
    store: ArtifactGraphStore, artifact_id: str, reference_dependency_ids: Set[str]
) -> int:
    """How many ids in ``reference_dependency_ids`` (a caller-supplied,
    KNOWN-correct dependency set for ``artifact_id`` -- e.g. an
    ``InjectedFault``'s own pre-injection ``read_input_versions``) are NOT
    currently reflected in ``ArtifactGraphStore.direct_dependencies``.

    This is a comparison tool, not automatic detection: it answers "does
    the current graph still show what we already know it should show,"
    never "what SHOULD this artifact depend on" with no reference at all
    -- the latter would need a content/NLP scanner, explicitly out of
    scope for this stage (``ArtifactGraphStore.add_inferred_edge``'s own
    "no automatic reference scanner" non-goal).
    """
    current_dependency_ids = store.direct_dependencies(artifact_id)
    return sum(1 for reference_id in reference_dependency_ids if reference_id not in current_dependency_ids)


def classify_failure_difficulty(
    store: ArtifactGraphStore,
    artifact_id: str,
    *,
    reference_dependency_ids: Optional[Set[str]] = None,
) -> FailureDifficultyProfile:
    """The four axes Stage 6's checklist names
    ("의존 깊이·fan-out·공유 정의 사용·관계 누락에 따라 분류한다"), computed from
    ``store``'s current dependency-graph state around ``artifact_id``.

    ``reference_dependency_ids``, when supplied, additionally populates
    ``current_missing_relations_vs_reference`` -- see that field's own
    docstring in ``schemas.py`` for why it is a separate field from
    ``historical_inferred_relations_count`` rather than a redefinition of
    it (a review caught that the original single field conflated "ever
    discovered as missing, cumulative" with "missing right now," which are
    not the same claim: a historical INFERRED edge recorded against a
    since-superseded version stays counted forever even after the relation
    it describes stops applying to the artifact's current content at all).

    Raises ``UnknownArtifactError`` if ``artifact_id`` was never stored --
    a failure must be classified against a real, previously-produced
    artifact, never a guessed/fabricated one.
    """
    current = store.latest(artifact_id)
    if current is None:
        raise UnknownArtifactError(f"'{artifact_id}' has never been stored; cannot classify its difficulty.")

    fan_out = len(store.direct_dependents(artifact_id))
    shared_definition_use = current.ref.kind not in _NON_SHARED_KINDS and fan_out > 1
    historical_inferred_relations_count = sum(
        1
        for edge in store.edges
        if edge.relation == DependencyRelationKind.INFERRED
        and (edge.from_artifact.artifact_id == artifact_id or edge.to_artifact.artifact_id == artifact_id)
    )
    current_missing_relations_vs_reference = (
        count_missing_relations_against_reference(store, artifact_id, reference_dependency_ids)
        if reference_dependency_ids is not None
        else None
    )
    return FailureDifficultyProfile(
        dependency_depth=_dependency_depth(store, artifact_id),
        fan_out=fan_out,
        shared_definition_use=shared_definition_use,
        historical_inferred_relations_count=historical_inferred_relations_count,
        current_missing_relations_vs_reference=current_missing_relations_vs_reference,
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
    mistake.

    When ``injected_fault`` is given, its own pre-injection record of
    ``failing_artifact_id``'s real dependencies (``ArtifactRef.
    read_input_versions`` on whichever of ``injected_fault.
    pre_injection_refs`` matches this artifact_id) is used as the known
    reference for ``classify_failure_difficulty``'s
    ``current_missing_relations_vs_reference`` -- an injected fault is
    exactly the one case where this stage genuinely has ground truth for
    "what this artifact is supposed to depend on," without needing a
    general reference-graph mechanism.
    """
    current = store.latest(failing_artifact_id)
    if current is None:
        raise UnknownArtifactError(f"'{failing_artifact_id}' has never been stored.")

    reference_dependency_ids: Optional[Set[str]] = None
    if injected_fault is not None:
        matching_pre_ref = next(
            (ref for ref in injected_fault.pre_injection_refs if ref.artifact_id == failing_artifact_id), None
        )
        if matching_pre_ref is not None:
            reference_dependency_ids = {r.artifact_id for r in matching_pre_ref.read_input_versions}

    return FailureRecord(
        failure_id=failure_id,
        origin=origin,
        injected_fault=injected_fault,
        failing_artifact=current.ref,
        evidence=list(evidence),
        difficulty=classify_failure_difficulty(
            store, failing_artifact_id, reference_dependency_ids=reference_dependency_ids
        ),
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

    An artifact_id this branch's cascade reached but never resolved either
    way (``RepairExecutionResult.left_stale``) is, by that field's own
    documented design, simply ABSENT from every execution's
    ``final_outcome_by_artifact`` -- so it stays absent from this merged
    dict too. Callers (``compute_repair_gain_score``) must treat "absent"
    as its own distinct, genuine state, never as "nothing happened here."
    """
    merged: Dict[str, RepairOutcome] = {}
    for round_ in branch.rounds:
        for execution in round_.executions:
            merged.update(execution.final_outcome_by_artifact)
    return merged


def _compute_damage(before: Checkpoint, after: Checkpoint, non_target_artifact_ids: Set[str]) -> ContentPreservationScore:
    """Compare ``before``/``after`` state for every artifact_id that
    existed in ``before`` and is NOT in ``non_target_artifact_ids``'s
    complement -- i.e. every artifact that was never part of what this
    repair was trying to fix (revision directions 1/4's "정상 내용 보존").

    "Broken" is decided from the RECONSTRUCTED FINAL STORE's own ground
    truth (``is_stale`` / a live FAIL ``Evidence``), never from a bare
    before/after content-string comparison. A review caught two bugs in
    the original design at once: (1) it excluded every artifact_id this
    branch's cascade happened to visit from damage-checking at all, which
    let a previously-healthy artifact swept into a WIDE repair action
    (e.g. whole-document regeneration) that came out genuinely FAILING be
    silently excluded just because it had been "touched"; (2) it used
    ``before_content != after_content`` as the damage signal, which cannot
    distinguish a harmless content refresh (e.g. a successful regeneration
    whose wording simply changed, or a reaffirm) from an actual new error
    -- only a confirmed FAIL/stale state in the final graph counts as
    damage now.
    """
    after_store = ArtifactGraphStore.from_checkpoint(after, branch_id="damage-check")
    newly_broken: List[str] = []
    checked = 0
    for artifact_id in non_target_artifact_ids:
        if artifact_id not in before.artifacts or artifact_id not in after.artifacts:
            continue
        checked += 1
        is_broken = after_store.is_stale(artifact_id) or any(
            item.outcome == CheckOutcome.FAIL for item in after_store.live_evidence(artifact_id)
        )
        if is_broken:
            newly_broken.append(artifact_id)

    preserved_fraction = (checked - len(newly_broken)) / checked if checked > 0 else None
    return ContentPreservationScore(
        preserved_fraction=preserved_fraction,
        newly_broken_artifacts=sorted(newly_broken),
        not_implemented_reason=(
            None if checked > 0 else "no non-target artifact existed in the source checkpoint to compare against"
        ),
    )


def compute_common_target_artifact_ids(
    comparison: BranchComparisonResult, *, triggering_evidence: List[Evidence]
) -> Set[str]:
    """The ONE common evaluation target shared by every branch in
    ``comparison`` -- Stage 6's completion criterion "후보 행동 사이에 공통
    평가 대상을 정하고" made concrete.

    The union of: (a) every artifact_id that was actually FAILING in
    ``triggering_evidence`` (what every branch started from, before any of
    them acted), and (b) every artifact_id that ANY branch's own execution
    history (across every round) ever reported a final outcome for, OR
    left genuinely unresolved (``RepairExecutionResult.left_stale``).

    (b) matters even for a branch that never itself touched a given
    artifact_id: a review's repro had branch A's narrow action resolve
    only the originally-failing artifact while orphaning a dependent as
    ``left_stale``, and branch B's wider action resolve both -- counting
    that dependent against branch A's gain requires it to be IN branch
    A's own ``target_artifact_ids``, even though branch A's own cascade
    never discovered it was a problem at all. Computing the target set
    from only one branch's own history (the bug this function replaces)
    cannot see what a *sibling* branch discovered; computing it from every
    compared branch's combined history can.
    """
    target_ids: Set[str] = {
        item.checked_artifact.artifact_id for item in triggering_evidence if item.outcome == CheckOutcome.FAIL
    }
    for branch in comparison.branches:
        for round_ in branch.rounds:
            for execution in round_.executions:
                target_ids.update(execution.final_outcome_by_artifact.keys())
                target_ids.update(ref.artifact_id for ref in execution.left_stale)
    return target_ids


def compute_repair_gain_score(
    before_checkpoint: Checkpoint,
    branch: BranchExecutionResult,
    *,
    target_artifact_ids: Set[str],
) -> RepairGainScore:
    """Revision direction 4's "repair gain, 손상, 총비용" for one branch.

    ``target_artifact_ids`` is the COMMON evaluation target -- the
    artifact_ids that were actually failing before ANY branch from this
    checkpoint ran -- fixed once by the caller and passed identically to
    every branch being compared from the same checkpoint (see
    ``RepairGainScore``'s own, rewritten docstring in ``schemas.py`` for
    the full repro this replaces: a branch-dependent denominator derived
    from each branch's own ``final_outcome_by_artifact`` let a narrow
    action that simply never attempted some of the originally-failing
    artifacts report a perfect, misleading ``gain=1.0``).

    Every id in ``target_artifact_ids`` lands in exactly one
    ``outcome_counts`` bucket: its real ``RepairOutcome`` if
    ``_merge_final_outcomes`` has one, else the sentinel
    ``LEFT_STALE_OR_UNRESOLVED`` -- never silently omitted, and never
    promoted to FAIL/INCONCLUSIVE (an absent verdict is not a FAIL
    verdict). ``damage`` is computed over every OTHER artifact_id in
    ``before_checkpoint`` (i.e. ``before_checkpoint``'s ids minus
    ``target_artifact_ids``) -- the "was meant to stay healthy" set,
    decided at the same time and for the same reason as the gain
    denominator.
    """
    final_outcomes = _merge_final_outcomes(branch)
    outcome_counts: Dict[str, int] = {}
    for target_id in target_artifact_ids:
        outcome = final_outcomes.get(target_id)
        bucket = outcome.value if outcome is not None else LEFT_STALE_OR_UNRESOLVED
        outcome_counts[bucket] = outcome_counts.get(bucket, 0) + 1

    if target_artifact_ids:
        gain: Optional[float] = outcome_counts.get(RepairOutcome.RESOLVED.value, 0) / len(target_artifact_ids)
        not_implemented_reason = None
    else:
        gain = None
        not_implemented_reason = "target_artifact_ids is empty -- nothing was being evaluated for repair"

    non_target_artifact_ids = set(before_checkpoint.artifacts) - target_artifact_ids
    damage = _compute_damage(before_checkpoint, branch.final_checkpoint, non_target_artifact_ids)
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
    target_artifact_ids: Set[str],
) -> CandidateActionRecord:
    """One ``(checkpoint, candidate action(s), downstream result)`` triple
    -- revision direction 4's benchmark-storage unit. ``branch.rounds[0]``
    is always the round that distinguishes this branch
    (``BranchExecutionResult``'s own docstring); every round after that
    belongs to the shared downstream policy, so ``actions``/
    ``branch_label`` here describe only the *initiating* round's choice(s)
    this record exists to compare against sibling branches from the same
    ``before_checkpoint``.

    ``actions`` records EVERY action that initiating round actually ran
    (``branch.rounds[0].executions``), not only the first -- a review
    caught that recording just ``executions[0].action`` while ``outcome``
    reflected the combined effect of every action the round ran
    misattributed a multi-action round's full result to a single, partial
    cause.
    """
    if not branch.rounds or not branch.rounds[0].executions:
        raise ValueError(
            f"branch {branch.branch_id!r} has no initial round/execution -- "
            "a CandidateActionRecord needs at least one actually-attempted action."
        )
    initiating_actions = [execution.action for execution in branch.rounds[0].executions]
    return CandidateActionRecord(
        record_id=record_id,
        failure_id=failure_id,
        source_checkpoint_id=branch.source_checkpoint_id,
        branch_label=branch.policy_label,
        actions=initiating_actions,
        outcome=compute_repair_gain_score(before_checkpoint, branch, target_artifact_ids=target_artifact_ids),
        rounds_executed=len(branch.rounds),
    )

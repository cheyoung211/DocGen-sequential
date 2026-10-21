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

**Revised a second time after a second review found 3 more real bugs**: (1)
``compute_common_target_artifact_ids``'s COMMON ``target_artifact_ids`` --
the very fix the first review above added -- was then being reused as
``_compute_damage``'s exclusion set too, which reintroduced the *original*
damage bug through a new door: an artifact a WIDE branch swept into its own
action and broke got unioned into the common target by component (b) of
that function (every branch's own execution history), which then made
``_compute_damage`` skip it for EVERY branch being compared, including a
narrow sibling that never touched it at all. Fixed by giving damage its own,
narrower exclusion set -- ``_originally_failing_artifact_ids(triggering_evidence)``,
component (a) alone -- decoupled from whatever (b) later adds for gain's
purposes. (2) ``compute_repair_gain_score`` trusted ``_merge_final_outcomes``'s
last-recorded verdict even when a *later*, unrelated round's own change
re-invalidated that artifact without anything ever re-checking it
afterward -- a target resolved in round 1 and silently re-invalidated by
round 2 still read back as ``RESOLVED``. Fixed by cross-checking every
``RESOLVED`` verdict against the branch's own final checkpoint
(``_is_broken_in_final_state``, the same ground-truth check ``_compute_damage``
already uses) and downgrading a contradicted one to the
``LEFT_STALE_OR_UNRESOLVED`` sentinel -- never promoted to FAIL, exactly
the same discipline the first review's "판정불가를 FAIL로 바꾸라는 뜻은
아닙니다" already established. (3) ``build_failure_record``'s reference
dependency baseline came from ``injected_fault.pre_injection_refs[i].
read_input_versions`` alone, which can only ever hold EXPLICIT inputs --
an INFERRED dependency the artifact had right before injection (and which
the injector itself now carries forward, see ``fault_injection``'s own
revised docstring) was invisible to that baseline, so
``current_missing_relations_vs_reference`` silently read 0 even when the
INFERRED relation was the one actually at risk. Fixed by reading
``InjectedFault.pre_injection_dependency_ids_by_artifact`` instead, which
``ArtifactGraphStore.direct_dependencies`` populates with both kinds.

**Revised a third time after a third review found 3 more real bugs**: (1)
a target_id absent from a branch's own ``final_outcome_by_artifact`` was
unconditionally bucketed ``LEFT_STALE_OR_UNRESOLVED`` for gain -- correct
when that target genuinely needed fixing and nothing addressed it, but
wrong when the id only entered ``target_artifact_ids`` via a *sibling*
branch's wider action touching an artifact that was never actually broken
to begin with (e.g. a sibling redundantly regenerating an already-PASSing
artifact): a narrow branch that correctly left a genuinely-healthy
artifact alone was penalized as if it had ignored a real problem. Fixed by
adding a new sentinel, ``ALREADY_HEALTHY_UNTOUCHED``, for exactly this
case -- a target NOT in the original trigger set, never touched by this
branch, and confirmed non-broken in this branch's own final state -- and
counting it toward ``gain``'s numerator alongside ``RESOLVED`` (see
``compute_repair_gain_score``'s docstring for the full repro and why this
is NOT the same case ``RepairGainScore``'s gain-denominator fix already
handles). (2) ``_compute_damage``'s evaluation set assumed every artifact
NOT in the original trigger was healthy before the repair -- which is
false whenever some *other*, unrelated artifact was already stale/FAILing
in the ``before_checkpoint`` for a pre-existing reason that has nothing to
do with this repair at all; such an artifact, left completely untouched,
was reported as "newly broken" simply because it was still broken
afterward. Fixed with a new ``_pre_repair_healthy_artifact_ids`` that
additionally checks the ``before_checkpoint``'s own ground truth
(``_is_broken_in_final_state``), so only a genuine healthy-before to
broken-after TRANSITION counts as damage, never a pre-existing problem
that was simply never part of this repair's scope. (3) fault injection's
INFERRED-edge carry-forward (previous revision, point 4 below) diffed an
artifact-id-only set to decide which dependencies "must be INFERRED,"
which silently treated an INFERRED edge pointing at an OLDER version of an
id as already covered by an EXPLICIT edge to a NEWER version of the SAME
id -- dropping the INFERRED edge (and the retroactive staleness it alone
was responsible for) entirely. Fixed in ``fault_injection.py`` by looking
up INFERRED edges directly by exact ``(artifact_id, version)``, never by
id-only set subtraction -- see that module's own revised docstring.

**Revised a fourth time after a fourth review found 1 more real bug,
inside the previous round's own two new mechanisms at once**:
``ALREADY_HEALTHY_UNTOUCHED`` and ``_pre_repair_healthy_artifact_ids``
both asserted "healthy" from ``not _is_broken_in_final_state(...)`` --
i.e. "not confirmed broken" -- which silently also matches a THIRD state
neither confirmed-broken nor confirmed-healthy covers: no evidence at
all, or only live ``CheckOutcome.UNKNOWN`` entries. A target left on
UNKNOWN both before and after (never actually verified by anyone) still
counted as ``ALREADY_HEALTHY_UNTOUCHED`` for whichever branch happened
not to re-check it, while its sibling branch that DID re-check it and
honestly got ``INCONCLUSIVE`` again scored worse for being more thorough.
Fixed by adding ``_is_confirmed_healthy`` (requires an actual live PASS)
and using it -- never the negation of ``_is_broken_in_final_state`` --
everywhere this module asserts an artifact IS healthy; the two predicates
are allowed to both be ``False`` for the same artifact (unverified is
neither), and nothing in this module may treat that as a tie-break in
either direction.
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

# A second sentinel, on the opposite side of LEFT_STALE_OR_UNRESOLVED: a
# target_id this branch's own execution history never mentions at all, but
# which was NOT part of the original trigger and is confirmed non-broken in
# this branch's own final state -- i.e. there was genuinely nothing for
# this branch to do. A review's repro had a sibling branch redundantly
# regenerate an already-PASSing artifact (pulling it into the COMMON
# `target_artifact_ids` via `compute_common_target_artifact_ids`'s own
# component (b)), which made a narrow branch that correctly left that
# artifact untouched read as if it had ignored a real problem
# (LEFT_STALE_OR_UNRESOLVED, dragging gain down from a deserved 1.0 to
# 0.5). Counted toward `gain`'s numerator alongside RESOLVED -- see
# `compute_repair_gain_score`'s docstring for the full repro and why this
# is a different case from LEFT_STALE_OR_UNRESOLVED, not a relaxation of it.
ALREADY_HEALTHY_UNTOUCHED = "already_healthy_untouched"


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
    ``failing_artifact_id``'s real dependencies
    (``InjectedFault.pre_injection_dependency_ids_by_artifact``) is used as
    the known reference for ``classify_failure_difficulty``'s
    ``current_missing_relations_vs_reference`` -- an injected fault is
    exactly the one case where this stage genuinely has ground truth for
    "what this artifact is supposed to depend on," without needing a
    general reference-graph mechanism. Deliberately reads
    ``pre_injection_dependency_ids_by_artifact``, not
    ``pre_injection_refs[i].read_input_versions`` -- a review caught that
    ``read_input_versions`` can only ever hold EXPLICIT inputs, silently
    dropping any INFERRED dependency from the reference baseline.
    """
    current = store.latest(failing_artifact_id)
    if current is None:
        raise UnknownArtifactError(f"'{failing_artifact_id}' has never been stored.")

    reference_dependency_ids: Optional[Set[str]] = None
    if injected_fault is not None:
        matching_ids = injected_fault.pre_injection_dependency_ids_by_artifact.get(failing_artifact_id)
        if matching_ids is not None:
            reference_dependency_ids = set(matching_ids)

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


def _is_broken_in_final_state(store: ArtifactGraphStore, artifact_id: str) -> bool:
    """An artifact counts as currently broken iff the store's own ground
    truth says so: ``is_stale`` or a live FAIL ``Evidence`` -- NEVER a bare
    before/after content-string comparison (a review caught that a plain
    string diff conflates a harmless, legitimate content refresh with an
    actual new error). Shared by ``_compute_damage`` and
    ``compute_repair_gain_score`` so the two never silently disagree about
    what "actually broken right now" means.
    """
    return store.is_stale(artifact_id) or any(
        item.outcome == CheckOutcome.FAIL for item in store.live_evidence(artifact_id)
    )


def _is_confirmed_healthy(store: ArtifactGraphStore, artifact_id: str) -> bool:
    """An artifact counts as CONFIRMED healthy iff it has at least one live
    PASS ``Evidence`` -- deliberately NOT the negation of
    ``_is_broken_in_final_state``. A review caught that treating "not
    stale and no live FAIL" as "healthy" silently promotes a THIRD,
    genuinely different state -- no evidence at all, or only live
    ``CheckOutcome.UNKNOWN`` entries (a constraint that could not be
    evaluated) -- into a confirmed-fine verdict it never earned. Those two
    predicates are allowed to both be ``False`` for the same artifact at
    the same time: unverified is not broken, but it is not healthy
    either, and must never be silently counted as either one just because
    it fails to match the other's definition. Used wherever this module
    needs to assert an artifact IS fine (``ALREADY_HEALTHY_UNTOUCHED``,
    ``_pre_repair_healthy_artifact_ids``) -- as opposed to
    ``_is_broken_in_final_state``, used wherever it needs to assert an
    artifact IS broken (damage's "newly broken" check, the RESOLVED-
    verdict cross-check); an UNKNOWN/never-checked artifact correctly
    satisfies neither.
    """
    return any(item.outcome == CheckOutcome.PASS for item in store.live_evidence(artifact_id))


def _originally_failing_artifact_ids(triggering_evidence: List[Evidence]) -> Set[str]:
    """Every artifact_id that was actually FAILING in ``triggering_evidence``
    -- what every branch being compared started from, before any of them
    acted. This is deliberately NOT the same set as
    ``compute_common_target_artifact_ids``'s return value: that function
    unions this with (b) whatever any branch's own execution history later
    adds, which is correct for gain (a sibling branch can discover a
    problem this one's own cascade never found) but wrong for damage,
    which must stay anchored to what was *actually* failing up front --
    see ``compute_repair_gain_score``'s docstring for the bug this
    separation fixes.
    """
    return {item.checked_artifact.artifact_id for item in triggering_evidence if item.outcome == CheckOutcome.FAIL}


def _pre_repair_healthy_artifact_ids(before: Checkpoint, triggering_evidence: List[Evidence]) -> Set[str]:
    """Every artifact_id in ``before`` that is NEITHER part of the original
    trigger (``_originally_failing_artifact_ids``) NOR anything less than
    CONFIRMED healthy in ``before``'s own ground truth
    (``_is_confirmed_healthy``, read from a store reconstructed from
    ``before`` itself: at least one live PASS) -- the only artifacts for
    which a repair starting from this checkpoint can meaningfully be said
    to have caused *new* damage.

    A review caught that the previous version of this logic (inlined in
    ``compute_repair_gain_score`` as ``set(before.artifacts) -
    _originally_failing_artifact_ids(...)``) assumed every non-triggering
    artifact was healthy before the repair -- false whenever some *other*,
    unrelated artifact was already stale/FAILing in ``before`` for a
    pre-existing reason that has nothing to do with this repair at all. An
    artifact left completely untouched, still exactly as stale as it was
    in ``before``, was reported as "newly broken" purely because
    ``_compute_damage`` never checked whether it had actually ever been
    healthy to begin with.

    A *second* review then caught that the fix for that bug (checking
    ``_is_broken_in_final_state``, i.e. excluding only CONFIRMED-broken
    artifacts) still silently promoted a THIRD state -- no evidence at
    all, or only live ``UNKNOWN`` -- into "healthy," since "not confirmed
    broken" is not the same claim as "confirmed healthy." An artifact
    whose status was never actually verified is not a safe baseline for
    "this repair preserved it": there is nothing to preserve if nothing
    ever confirmed it was fine to begin with. Only a genuine CONFIRMED
    healthy-before -> broken-after TRANSITION is damage; a pre-existing
    problem, or a status this stage never verified either way, is neither
    this repair's achievement nor its fault.
    """
    before_store = ArtifactGraphStore.from_checkpoint(before, branch_id="damage-baseline")
    originally_failing_ids = _originally_failing_artifact_ids(triggering_evidence)
    return {
        artifact_id
        for artifact_id in before.artifacts
        if artifact_id not in originally_failing_ids and _is_confirmed_healthy(before_store, artifact_id)
    }


def _compute_damage(before: Checkpoint, after: Checkpoint, healthy_before_artifact_ids: Set[str]) -> ContentPreservationScore:
    """Compare ``before``/``after`` state for every artifact_id in
    ``healthy_before_artifact_ids`` -- i.e. every artifact that was never
    part of what this repair was trying to fix (revision directions 1/4's
    "정상 내용 보존"), decided independently of which branch touched what.

    "Broken" is decided from the RECONSTRUCTED FINAL STORE's own ground
    truth via ``_is_broken_in_final_state`` (``is_stale`` / a live FAIL
    ``Evidence``), never from a bare before/after content-string
    comparison. A review caught two bugs in the original design at once:
    (1) it excluded every artifact_id this branch's cascade happened to
    visit from damage-checking at all, which let a previously-healthy
    artifact swept into a WIDE repair action (e.g. whole-document
    regeneration) that came out genuinely FAILING be silently excluded
    just because it had been "touched"; (2) it used ``before_content !=
    after_content`` as the damage signal, which cannot distinguish a
    harmless content refresh from an actual new error.

    A *second* review then caught that fixing (1) by excluding
    ``target_artifact_ids`` (plural, the COMMON set shared across
    branches) reintroduced the exact same bug through a new door: once
    that common set also includes whatever a *sibling* branch's own
    history happened to touch, a branch that never touched a given
    artifact at all could still have it excluded from ITS OWN damage
    check, just because a sibling's wider action collided with it. See
    ``compute_repair_gain_score``'s docstring for why the caller must pass
    ``_originally_failing_artifact_ids(triggering_evidence)``'s complement
    here, never ``target_artifact_ids``'s.

    A *third* review then caught that this complement is still not quite
    "healthy before" -- an artifact that was already broken in ``before``
    for a pre-existing, unrelated reason is also not "originally failing"
    (it is not in ``triggering_evidence`` at all), so it was still ending
    up in this set and being reported as "newly broken" for simply
    staying exactly as broken as it already was. Callers must pass
    ``_pre_repair_healthy_artifact_ids(before, triggering_evidence)``,
    never the raw ``set(before.artifacts) -
    _originally_failing_artifact_ids(...)`` this replaces -- see that
    function's own docstring.
    """
    after_store = ArtifactGraphStore.from_checkpoint(after, branch_id="damage-check")
    newly_broken: List[str] = []
    checked = 0
    for artifact_id in healthy_before_artifact_ids:
        if artifact_id not in before.artifacts or artifact_id not in after.artifacts:
            continue
        checked += 1
        if _is_broken_in_final_state(after_store, artifact_id):
            newly_broken.append(artifact_id)

    preserved_fraction = (checked - len(newly_broken)) / checked if checked > 0 else None
    return ContentPreservationScore(
        preserved_fraction=preserved_fraction,
        newly_broken_artifacts=sorted(newly_broken),
        not_implemented_reason=(
            None if checked > 0 else "no artifact was healthy-before-repair in the source checkpoint to compare against"
        ),
    )


def compute_common_target_artifact_ids(
    comparison: BranchComparisonResult, *, triggering_evidence: List[Evidence]
) -> Set[str]:
    """The ONE common evaluation target shared by every branch in
    ``comparison`` -- Stage 6's completion criterion "후보 행동 사이에 공통
    평가 대상을 정하고" made concrete.

    The union of: (a) every artifact_id that was actually FAILING in
    ``triggering_evidence`` (``_originally_failing_artifact_ids``), and (b)
    every artifact_id that ANY branch's own execution history (across
    every round) ever reported a final outcome for, OR left genuinely
    unresolved (``RepairExecutionResult.left_stale``).

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

    This broader union is correct for GAIN, but must never be reused as
    damage's exclusion set -- see ``_originally_failing_artifact_ids``'s
    own docstring and ``compute_repair_gain_score``'s for why (a) alone,
    not this function's full return value, is what damage needs.
    """
    target_ids: Set[str] = set(_originally_failing_artifact_ids(triggering_evidence))
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
    triggering_evidence: List[Evidence],
) -> RepairGainScore:
    """Revision direction 4's "repair gain, 손상, 총비용" for one branch.

    ``target_artifact_ids`` is the COMMON evaluation target used for GAIN
    only -- fixed once by the caller (``compute_common_target_artifact_ids``)
    and passed identically to every branch being compared from the same
    checkpoint (see ``RepairGainScore``'s own, rewritten docstring in
    ``schemas.py`` for the full repro this replaces: a branch-dependent
    denominator derived from each branch's own ``final_outcome_by_artifact``
    let a narrow action that simply never attempted some of the
    originally-failing artifacts report a perfect, misleading
    ``gain=1.0``).

    Every id in ``target_artifact_ids`` lands in exactly one
    ``outcome_counts`` bucket: its real ``RepairOutcome`` if
    ``_merge_final_outcomes`` has one, else one of two sentinels --
    ``LEFT_STALE_OR_UNRESOLVED`` or ``ALREADY_HEALTHY_UNTOUCHED`` -- never
    silently omitted, and never promoted to FAIL/INCONCLUSIVE (an absent
    verdict is not a FAIL verdict). A ``RESOLVED`` verdict is additionally
    cross-checked against ``branch.final_checkpoint``'s own ground truth
    (``_is_broken_in_final_state``) before being trusted: a review's repro
    had an artifact resolved in round 1, then re-invalidated by a later,
    unrelated round's own change that nothing ever re-checked afterward --
    ``_merge_final_outcomes`` (last-write-wins) still read back
    ``RESOLVED`` even though the branch's actual final state showed it
    stale again. A verdict contradicted this way is downgraded to
    ``LEFT_STALE_OR_UNRESOLVED``, never promoted to FAIL -- the branch
    genuinely has no fresh verdict for it, which is exactly what that
    sentinel already means.

    A target_id this branch's own history never mentions at all is NOT
    automatically ``LEFT_STALE_OR_UNRESOLVED`` either: if it is not part
    of the original trigger (``_originally_failing_artifact_ids``) AND is
    CONFIRMED healthy in this branch's own final state
    (``_is_confirmed_healthy`` -- at least one live PASS, NOT merely "not
    confirmed broken"), it becomes ``ALREADY_HEALTHY_UNTOUCHED`` instead,
    and counts toward ``gain``'s numerator alongside ``RESOLVED``. A
    review's repro had A fail and B already sitting on a valid PASS; a
    narrow branch correctly repaired only A and left B alone, while a
    wide sibling branch redundantly regenerated both -- pulling B into the
    COMMON ``target_artifact_ids`` via
    ``compute_common_target_artifact_ids``'s component (b). Without this
    sentinel, the narrow branch's own, correct choice to leave an
    already-healthy B untouched read as if it had ignored a real problem
    (``gain=0.5`` instead of the deserved ``1.0``) -- a comparison that
    structurally penalizes the MORE selective, MORE efficient repair. This
    is a different case from ``LEFT_STALE_OR_UNRESOLVED``, which still
    applies whenever the untouched target genuinely IS broken in this
    branch's final state (e.g. a true dependent that cascaded into
    staleness and was simply never revisited) -- only a target that was
    NEVER actually a problem for this checkpoint at all is eligible for
    the new sentinel.

    A *second* review then caught that requiring merely "not confirmed
    broken" (``_is_broken_in_final_state``'s negation) for the sentinel
    above silently admitted a THIRD state: no evidence at all, or only
    live ``UNKNOWN`` entries (a constraint that could not be evaluated
    either way). A target B left on UNKNOWN before AND after -- genuinely
    never verified as healthy by either branch -- still counted as
    ``ALREADY_HEALTHY_UNTOUCHED`` for whichever branch happened not to
    touch it, giving that branch a perfect ``gain=1.0`` while its sibling
    (which DID re-check B, and got UNKNOWN again -- an honest
    ``INCONCLUSIVE``, never counted toward gain) scored only ``0.5``,
    even though NEITHER branch ever actually confirmed B was fine. Fixed
    by requiring ``_is_confirmed_healthy`` (an actual live PASS) for this
    sentinel, never just the absence of a confirmed FAIL/stale state --
    unverified is its own, third outcome, counted toward neither bucket.

    ``triggering_evidence`` -- the SAME list passed to
    ``compute_common_target_artifact_ids`` for this comparison -- is used
    for two things: the ``ALREADY_HEALTHY_UNTOUCHED`` check above, and to
    derive ``damage``'s evaluation set,
    ``_pre_repair_healthy_artifact_ids(before_checkpoint, triggering_evidence)``
    -- every artifact that both (a) was not part of the original trigger,
    and (b) is CONFIRMED healthy (not merely "not confirmed broken," for
    the identical reason as the previous paragraph) in
    ``before_checkpoint``'s own ground truth. This is deliberately NOT
    ``target_artifact_ids``'s complement either -- excluding the full
    common target (which also includes whatever a *sibling* branch's own
    history later adds) let an artifact THIS branch itself swept into a
    wide action and broke escape damage-checking, simply because some
    other branch's history happened to also mention it.
    """
    final_outcomes = _merge_final_outcomes(branch)
    final_store = ArtifactGraphStore.from_checkpoint(branch.final_checkpoint, branch_id="gain-check")
    originally_failing_ids = _originally_failing_artifact_ids(triggering_evidence)
    outcome_counts: Dict[str, int] = {}
    for target_id in target_artifact_ids:
        outcome = final_outcomes.get(target_id)
        confirmed_in_final_checkpoint = target_id in branch.final_checkpoint.artifacts
        if (
            outcome == RepairOutcome.RESOLVED
            and confirmed_in_final_checkpoint
            and _is_broken_in_final_state(final_store, target_id)
        ):
            bucket = LEFT_STALE_OR_UNRESOLVED
        elif (
            outcome is None
            and target_id not in originally_failing_ids
            and confirmed_in_final_checkpoint
            and _is_confirmed_healthy(final_store, target_id)
            and not _is_broken_in_final_state(final_store, target_id)
        ):
            bucket = ALREADY_HEALTHY_UNTOUCHED
        else:
            bucket = outcome.value if outcome is not None else LEFT_STALE_OR_UNRESOLVED
        outcome_counts[bucket] = outcome_counts.get(bucket, 0) + 1

    if target_artifact_ids:
        resolved_count = outcome_counts.get(RepairOutcome.RESOLVED.value, 0) + outcome_counts.get(
            ALREADY_HEALTHY_UNTOUCHED, 0
        )
        gain: Optional[float] = resolved_count / len(target_artifact_ids)
        not_implemented_reason = None
    else:
        gain = None
        not_implemented_reason = "target_artifact_ids is empty -- nothing was being evaluated for repair"

    healthy_before_artifact_ids = _pre_repair_healthy_artifact_ids(before_checkpoint, triggering_evidence)
    damage = _compute_damage(before_checkpoint, branch.final_checkpoint, healthy_before_artifact_ids)
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
    triggering_evidence: List[Evidence],
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
    cause. ``triggering_evidence`` is forwarded as-is to
    ``compute_repair_gain_score`` -- see that function's docstring for why
    it must be the same list used to build ``target_artifact_ids``.
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
        outcome=compute_repair_gain_score(
            before_checkpoint,
            branch,
            target_artifact_ids=target_artifact_ids,
            triggering_evidence=triggering_evidence,
        ),
        rounds_executed=len(branch.rounds),
    )

"""Human annotation tooling (Stage 6 of the "Revision 2026.10.15" plan:
"실패·행동 데이터셋과 human gold 구축" -- the human-gold half of that
checklist, operationalizing ``docs/stage1_experiment_design.md`` §2's
protocol, which that document deliberately left as protocol-only until
this stage).

**Scope decision (confirmed with the user before implementing)**: an AI
agent cannot act as a genuine independent human annotator, so this module
builds the tooling real annotators (the user, or hired labelers) would use
-- task sampling/export, a raw-label import shape, and the disagreement-
resolution/agreement/weighting computation Stage 1 §2 specifies. It does
not, and cannot, produce real labels itself. Any example data constructed
against this module in tests is explicitly synthetic, used only to
exercise the merge/resolution code path -- never represented as real human
judgment.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List

from src.core.schemas import (
    AnnotationAgreementReport,
    AnnotationLabel,
    AnnotationOutcome,
    AnnotationTask,
    CheckOutcome,
    ResolvedAnnotation,
    ResolvedLabelStatus,
    StratumWeight,
    VerifierPrecisionRecall,
)


def select_balanced_sample(
    candidates: List[AnnotationTask],
    *,
    min_per_stratum: int,
    seed: int = 0,
) -> List[AnnotationTask]:
    """``docs/stage1_experiment_design.md`` §2's "표본 추출"/"층화": draw up
    to ``min_per_stratum`` tasks from each distinct ``AnnotationStratum.key``
    present in ``candidates`` (trigger/non-trigger crossed with whatever
    domain/difficulty each task carries), rather than drawing a single
    pool-wide random sample that could, by chance, under-represent a rare
    stratum (e.g. non-trigger cases, if triggers dominate the pool).

    Deterministic given the same ``candidates``/``seed`` -- a per-stratum
    ``random.Random(seed)`` shuffle, iterated in sorted stratum-key order
    so the result does not depend on ``candidates``' own input order or on
    dict-iteration order.
    """
    rng = random.Random(seed)
    by_stratum: Dict[str, List[AnnotationTask]] = {}
    for task in candidates:
        by_stratum.setdefault(task.stratum.key, []).append(task)

    selected: List[AnnotationTask] = []
    for key in sorted(by_stratum):
        bucket = list(by_stratum[key])
        rng.shuffle(bucket)
        selected.extend(bucket[:min_per_stratum])
    return selected


def resolve_annotations(labels_by_task: Dict[str, List[AnnotationLabel]]) -> Dict[str, ResolvedAnnotation]:
    """``docs/stage1_experiment_design.md`` §2's disagreement procedure,
    applied independently per task: two non-adjudicator labels that agree
    resolve directly; two that disagree require at least one adjudicator
    label and resolve by majority across ALL labels (primary + adjudicator);
    a majority that still does not exist resolves to ``UNRESOLVED``.

    The Notion plan's own wording ("재검토 후에도 불일치하면 unknown으로
    기록") is implemented as ``status=UNRESOLVED``/``outcome=None`` here,
    not as ``outcome=AnnotationOutcome.UNKNOWN`` -- the latter would make a
    *procedural give-up* indistinguishable from annotators independently,
    genuinely judging the task "unknown" (a real, agreed-upon label in its
    own right). ``UNRESOLVED`` already achieves the Notion text's actual
    requirement -- exclusion from any precision/recall denominator -- via
    ``compute_verifier_precision_recall`` simply skipping any task without
    a resolved outcome; it does not also need to borrow the real
    ``AnnotationOutcome.UNKNOWN`` value to do that.

    Raises ``ValueError`` if: a label's own ``task_id`` does not match the
    key it was grouped under; any two labels for the same task (primary or
    adjudicator) come from the same ``annotator_id`` (a review's explicit
    concern -- the protocol requires independent annotators, and neither
    "the same primary labeler counted twice" nor "a primary labeler also
    adjudicating their own disagreement" is independence, and neither was
    being checked at all before); a task has fewer than 2 primary labels;
    or its primary labels disagree with no adjudicator supplied. All are
    caller/data errors, not states this function should silently paper
    over.
    """
    resolved: Dict[str, ResolvedAnnotation] = {}
    for task_id, labels in labels_by_task.items():
        resolved[task_id] = _resolve_one(task_id, labels)
    return resolved


def _resolve_one(task_id: str, labels: List[AnnotationLabel]) -> ResolvedAnnotation:
    mismatched = [label for label in labels if label.task_id != task_id]
    if mismatched:
        raise ValueError(
            f"labels_by_task[{task_id!r}] contains a label actually stamped with task_id "
            f"{mismatched[0].task_id!r} -- labels must be grouped under their own task_id, "
            "never reassigned to a different key."
        )

    annotator_ids = [label.annotator_id for label in labels]
    duplicated = {annotator_id for annotator_id in annotator_ids if annotator_ids.count(annotator_id) > 1}
    if duplicated:
        raise ValueError(
            f"task {task_id!r} has more than one label from annotator(s) {sorted(duplicated)!r} -- "
            "the protocol requires independent annotators: the same person cannot contribute two "
            "labels for one task, whether as two 'independent' primaries, or as both a primary "
            "labeler and that same task's own adjudicator."
        )

    primary = [label for label in labels if not label.is_adjudicator]
    adjudicators = [label for label in labels if label.is_adjudicator]
    if len(primary) < 2:
        raise ValueError(
            f"task {task_id!r} has only {len(primary)} primary (non-adjudicator) label(s); "
            "the protocol requires at least 2 independent, blind primary labels."
        )

    primary_outcomes = [label.outcome for label in primary]
    if len(set(primary_outcomes)) == 1:
        return ResolvedAnnotation(
            task_id=task_id,
            outcome=primary_outcomes[0],
            status=ResolvedLabelStatus.AGREED,
            label_count=len(labels),
        )

    if not adjudicators:
        raise ValueError(
            f"task {task_id!r}'s primary labels disagree ({primary_outcomes!r}) but no "
            "adjudicator label was supplied -- the protocol requires a third labeler's "
            "review before a disagreement can be resolved."
        )

    counts = Counter(primary_outcomes + [label.outcome for label in adjudicators])
    top_count = max(counts.values())
    winners = [outcome for outcome, count in counts.items() if count == top_count]
    if len(winners) == 1:
        return ResolvedAnnotation(
            task_id=task_id,
            outcome=winners[0],
            status=ResolvedLabelStatus.ADJUDICATED,
            label_count=len(labels),
        )
    return ResolvedAnnotation(
        task_id=task_id,
        outcome=None,
        status=ResolvedLabelStatus.UNRESOLVED,
        label_count=len(labels),
    )


def compute_agreement_report(
    tasks_by_id: Dict[str, AnnotationTask], resolved: Dict[str, ResolvedAnnotation]
) -> AnnotationAgreementReport:
    """§2's "불일치율 자체를 별도로 보고한다", overall and per
    ``AnnotationStratum.key``. ``ADJUDICATED`` and ``UNRESOLVED`` both count
    as a disagreement for this rate -- a task that needed a third labeler
    at all disagreed initially, whether or not that third opinion broke the
    tie.
    """
    total = len(resolved)
    agreed = sum(1 for item in resolved.values() if item.status == ResolvedLabelStatus.AGREED)
    adjudicated = sum(1 for item in resolved.values() if item.status == ResolvedLabelStatus.ADJUDICATED)
    unresolved = sum(1 for item in resolved.values() if item.status == ResolvedLabelStatus.UNRESOLVED)
    disagreement_rate = (adjudicated + unresolved) / total if total > 0 else 0.0

    totals_by_stratum: Dict[str, int] = {}
    disagree_by_stratum: Dict[str, int] = {}
    for task_id, item in resolved.items():
        task = tasks_by_id.get(task_id)
        if task is None:
            continue
        key = task.stratum.key
        totals_by_stratum[key] = totals_by_stratum.get(key, 0) + 1
        if item.status != ResolvedLabelStatus.AGREED:
            disagree_by_stratum[key] = disagree_by_stratum.get(key, 0) + 1

    return AnnotationAgreementReport(
        total_tasks=total,
        agreed_count=agreed,
        adjudicated_count=adjudicated,
        unresolved_count=unresolved,
        disagreement_rate=disagreement_rate,
        disagreement_rate_by_stratum={
            key: disagree_by_stratum.get(key, 0) / count for key, count in totals_by_stratum.items()
        },
    )


def compute_stratum_weights(
    population_counts: Dict[str, int], sample_counts: Dict[str, int]
) -> Dict[str, StratumWeight]:
    """Stage 6's completion criterion "계층별 표본을 사용했다면 전체 비율
    추정에 필요한 가중치를 적용한다": ``weight = population_count /
    sample_count`` per ``AnnotationStratum.key``, so a stratum oversampled
    relative to its true population share (e.g. rare trigger cases,
    deliberately over-collected for annotation efficiency) gets a weight
    below 1.0 when the sample is used to estimate a population-level rate.

    Raises if any stratum present in ``population_counts`` has no matching
    entry in ``sample_counts`` (an un-sampled stratum cannot be weighted at
    all -- silently omitting it from the weighted estimate would bias that
    estimate without saying so) or if any sample count is 0 (division by
    zero would otherwise silently become an infinite/undefined weight).
    """
    missing = set(population_counts) - set(sample_counts)
    if missing:
        raise ValueError(
            f"population strata {sorted(missing)} have no matching sample -- "
            "cannot compute a population-rate weight for a stratum that was never sampled."
        )
    weights: Dict[str, StratumWeight] = {}
    for key, sample_count in sample_counts.items():
        if sample_count <= 0:
            raise ValueError(f"stratum {key!r} has sample_count={sample_count}; weighting requires sample_count > 0.")
        population_count = population_counts.get(key, 0)
        weights[key] = StratumWeight(
            stratum_key=key,
            population_count=population_count,
            sample_count=sample_count,
            weight=population_count / sample_count,
        )
    return weights


def _included_gold_items(gold: Dict[str, ResolvedAnnotation], predictions: Dict[str, CheckOutcome]):
    """Shared filter for both the unweighted and weighted precision/recall
    computations below -- a task is included only if its gold is a real,
    valid PASS/FAIL human consensus (excludes ``UNRESOLVED``, where
    ``outcome`` is structurally ``None``, AND a review-caught gap: a
    genuine, AGREED gold of ``AnnotationOutcome.UNKNOWN`` -- annotators who
    independently agreed they could not tell -- which the first draft let
    fall through to "not actual_fail," silently scoring it as if the
    annotators had agreed PASS) and the verifier's own prediction for it is
    PASS or FAIL (excludes a verifier UNKNOWN the same way). Yields
    ``(task_id, resolved, predicted_fail, actual_fail)`` for each included
    task so both callers apply the identical inclusion rule -- a precision/
    recall pair computed with two different filters would not be
    comparable at all.
    """
    for task_id, resolved in gold.items():
        if resolved.outcome not in (AnnotationOutcome.PASS, AnnotationOutcome.FAIL):
            continue
        predicted = predictions.get(task_id)
        if predicted not in (CheckOutcome.PASS, CheckOutcome.FAIL):
            continue
        yield task_id, resolved, predicted == CheckOutcome.FAIL, resolved.outcome == AnnotationOutcome.FAIL


def compute_verifier_precision_recall(
    predictions: Dict[str, CheckOutcome],
    gold: Dict[str, ResolvedAnnotation],
) -> VerifierPrecisionRecall:
    """RAW, UNWEIGHTED sample-level online ``Validator`` precision/recall
    against the resolved human gold, with FAIL as the positive class
    (direction 2's own framing: a verifier "triggers" by reporting FAIL).
    ``interfaces.FinalEvaluator``'s docstring already forbids reusing the
    online stream as the gold label itself -- ``gold`` here must come only
    from ``resolve_annotations``, never from ``predictions``' own source.

    This is the per-sample metric, treating every included task as equally
    weighted regardless of which stratum it came from -- see
    ``compute_weighted_verifier_precision_recall`` for the SEPARATE,
    population-level estimate a stratified sample needs instead (a review
    caught that conflating the two silently returned the unweighted number
    where a population estimate was expected).

    ``precision``/``recall`` stay ``None`` when their respective
    denominator (predicted-FAIL count / actual-FAIL count) is 0, rather
    than fabricated as 0.0 -- the same "never silently backfill a rate from
    an empty denominator" discipline every other score in this codebase
    already follows.
    """
    true_positive = 0
    predicted_positive = 0
    actual_positive = 0
    sample_size = 0

    for _task_id, _resolved, predicted_fail, actual_fail in _included_gold_items(gold, predictions):
        sample_size += 1
        if predicted_fail:
            predicted_positive += 1
        if actual_fail:
            actual_positive += 1
        if predicted_fail and actual_fail:
            true_positive += 1

    precision = (true_positive / predicted_positive) if predicted_positive > 0 else None
    recall = (true_positive / actual_positive) if actual_positive > 0 else None
    return VerifierPrecisionRecall(precision=precision, recall=recall, sample_size=sample_size)


def compute_weighted_verifier_precision_recall(
    predictions: Dict[str, CheckOutcome],
    gold: Dict[str, ResolvedAnnotation],
    task_strata: Dict[str, str],
    weights: Dict[str, StratumWeight],
) -> VerifierPrecisionRecall:
    """POPULATION-LEVEL, stratum-weighted precision/recall -- Stage 6's
    completion criterion "계층별 표본을 사용했다면 전체 비율 추정에 필요한
    가중치를 적용한다" actually applied to a metric, not merely available as
    a separate, never-used weight table (a review caught that
    ``compute_stratum_weights`` existed but nothing in this module actually
    used its output: a stratified sample that deliberately oversampled a
    rare stratum reported the SAME recall as the naive unweighted version,
    silently overweighting that stratum's cases in the result).

    Standard stratified (Horvitz-Thompson-style) estimator: every
    included task's contribution to true-positive/predicted-positive/
    actual-positive is scaled by ``weights[task_strata[task_id]].weight``
    before summing, so a stratum sampled above its true population rate
    contributes proportionally less to the final estimate, and one
    undersampled contributes more. Uses the SAME inclusion filter as
    ``compute_verifier_precision_recall`` (``_included_gold_items``) --
    the two numbers must differ only in weighting, never in which tasks
    they are computed over.

    ``sample_size`` on the returned object is still the RAW, unweighted
    count of included tasks (how many real human-labeled comparisons went
    into this estimate) -- only ``precision``/``recall`` themselves are
    weighted; conflating the two would make it impossible to tell a
    population estimate's sample size from its (differently-scaled)
    statistical weight.

    Raises ``ValueError`` if any included task's stratum
    (``task_strata[task_id]``) has no entry in ``weights`` -- the same
    "never silently omit a stratum from a weighted estimate" discipline
    ``compute_stratum_weights`` itself already enforces when building that
    table.
    """
    weighted_true_positive = 0.0
    weighted_predicted_positive = 0.0
    weighted_actual_positive = 0.0
    sample_size = 0

    for task_id, _resolved, predicted_fail, actual_fail in _included_gold_items(gold, predictions):
        sample_size += 1
        stratum_key = task_strata.get(task_id)
        if stratum_key is None or stratum_key not in weights:
            raise ValueError(
                f"task {task_id!r} has no resolvable stratum weight "
                f"(stratum_key={stratum_key!r}) -- every included task must have an entry "
                "in both task_strata and weights; omitting one would silently bias the "
                "population estimate without saying so."
            )
        weight = weights[stratum_key].weight
        if predicted_fail:
            weighted_predicted_positive += weight
        if actual_fail:
            weighted_actual_positive += weight
        if predicted_fail and actual_fail:
            weighted_true_positive += weight

    precision = (weighted_true_positive / weighted_predicted_positive) if weighted_predicted_positive > 0 else None
    recall = (weighted_true_positive / weighted_actual_positive) if weighted_actual_positive > 0 else None
    return VerifierPrecisionRecall(precision=precision, recall=recall, sample_size=sample_size)

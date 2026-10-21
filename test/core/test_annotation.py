"""Coverage for src/core/annotation.py (Stage 6): stratified task sampling,
the disagreement-resolution procedure (agree / adjudicate / unresolved),
agreement reporting, stratum weighting, and both raw and population-
weighted verifier precision/recall against resolved human gold. Every
label in this file is synthetic test fixture data, not real human
judgment -- this module cannot produce real labels itself (see its own
module docstring).

Revised after a user review found 2 real bugs (full detail in
docs/stage6_failure_dataset.md and memory docgen_stage_review_rigor) --
every fix below has at least one regression test named after the repro
that caught it.
"""

from __future__ import annotations

from typing import Optional

import pytest

from src.core.annotation import (
    compute_agreement_report,
    compute_stratum_weights,
    compute_verifier_precision_recall,
    compute_weighted_verifier_precision_recall,
    resolve_annotations,
    select_balanced_sample,
)
from src.core.schemas import (
    AnnotationLabel,
    AnnotationOutcome,
    AnnotationStratum,
    AnnotationTask,
    ArtifactKind,
    ArtifactRef,
    CheckOutcome,
    ResolvedAnnotation,
    ResolvedLabelStatus,
    StratumWeight,
)


def _task(
    task_id: str,
    *,
    trigger: bool,
    domain: str = "mathematics",
    success: Optional[bool] = None,
    stage: Optional[str] = None,
) -> AnnotationTask:
    return AnnotationTask(
        task_id=task_id,
        artifact=ArtifactRef(artifact_id=task_id, version=1, kind=ArtifactKind.BLOCK),
        content_shown="some content",
        stratum=AnnotationStratum(domain=domain, difficulty="basic", stage=stage, success=success, trigger=trigger),
    )


def _label(task_id: str, annotator_id: str, outcome: AnnotationOutcome, *, is_adjudicator: bool = False) -> AnnotationLabel:
    return AnnotationLabel(task_id=task_id, annotator_id=annotator_id, outcome=outcome, is_adjudicator=is_adjudicator)


# ---------------------------------------------------------------------------
# select_balanced_sample
# ---------------------------------------------------------------------------


def test_select_balanced_sample_draws_min_per_stratum_from_each_stratum() -> None:
    candidates = [_task(f"trig_{i}", trigger=True) for i in range(5)] + [
        _task(f"non_{i}", trigger=False) for i in range(5)
    ]

    selected = select_balanced_sample(candidates, min_per_stratum=2, seed=0)

    triggers = [t for t in selected if t.stratum.trigger]
    non_triggers = [t for t in selected if not t.stratum.trigger]
    assert len(triggers) == 2
    assert len(non_triggers) == 2


def test_select_balanced_sample_takes_everything_when_a_stratum_is_smaller_than_the_minimum() -> None:
    candidates = [_task("only_trigger", trigger=True)] + [_task(f"non_{i}", trigger=False) for i in range(5)]
    selected = select_balanced_sample(candidates, min_per_stratum=3, seed=0)
    assert sum(1 for t in selected if t.stratum.trigger) == 1
    assert sum(1 for t in selected if not t.stratum.trigger) == 3


def test_select_balanced_sample_is_deterministic_given_the_same_seed() -> None:
    candidates = [_task(f"t_{i}", trigger=True) for i in range(10)]
    first = select_balanced_sample(candidates, min_per_stratum=4, seed=42)
    second = select_balanced_sample(candidates, min_per_stratum=4, seed=42)
    assert [t.task_id for t in first] == [t.task_id for t in second]


def test_select_balanced_sample_treats_trigger_and_success_as_independent_axes() -> None:
    """Notion Stage 6's own checklist line names trigger/non-trigger AND
    success/failure as separate required sample coverage -- a non-trigger
    sample from a FAILED trajectory (a miss: the online verifier never
    fired, yet the trajectory still failed) and one from a SUCCEEDED
    trajectory must be sampled as distinct strata, not collapsed into one
    just because both happen to be non-trigger."""
    candidates = (
        [_task(f"miss_{i}", trigger=False, success=False) for i in range(5)]
        + [_task(f"ok_{i}", trigger=False, success=True) for i in range(5)]
    )

    selected = select_balanced_sample(candidates, min_per_stratum=2, seed=0)

    misses = [t for t in selected if t.stratum.success is False]
    oks = [t for t in selected if t.stratum.success is True]
    assert len(misses) == 2
    assert len(oks) == 2


def test_annotation_stratum_key_distinguishes_success_and_stage() -> None:
    base = AnnotationStratum(domain="mathematics", difficulty="basic", trigger=False)
    different_success = AnnotationStratum(domain="mathematics", difficulty="basic", success=False, trigger=False)
    different_stage = AnnotationStratum(
        domain="mathematics", difficulty="basic", stage="content_generation", trigger=False
    )
    assert base.key != different_success.key
    assert base.key != different_stage.key
    assert different_success.key != different_stage.key


# ---------------------------------------------------------------------------
# resolve_annotations
# ---------------------------------------------------------------------------


def test_resolve_annotations_agreement() -> None:
    labels = {"t1": [_label("t1", "ann1", AnnotationOutcome.FAIL), _label("t1", "ann2", AnnotationOutcome.FAIL)]}
    resolved = resolve_annotations(labels)
    assert resolved["t1"].status == ResolvedLabelStatus.AGREED
    assert resolved["t1"].outcome == AnnotationOutcome.FAIL


def test_resolve_annotations_disagreement_adjudicated_by_majority() -> None:
    labels = {
        "t1": [
            _label("t1", "ann1", AnnotationOutcome.FAIL),
            _label("t1", "ann2", AnnotationOutcome.PASS),
            _label("t1", "adj", AnnotationOutcome.FAIL, is_adjudicator=True),
        ]
    }
    resolved = resolve_annotations(labels)
    assert resolved["t1"].status == ResolvedLabelStatus.ADJUDICATED
    assert resolved["t1"].outcome == AnnotationOutcome.FAIL


def test_resolve_annotations_three_way_disagreement_is_unresolved() -> None:
    labels = {
        "t1": [
            _label("t1", "ann1", AnnotationOutcome.FAIL),
            _label("t1", "ann2", AnnotationOutcome.PASS),
            _label("t1", "adj", AnnotationOutcome.UNKNOWN, is_adjudicator=True),
        ]
    }
    resolved = resolve_annotations(labels)
    assert resolved["t1"].status == ResolvedLabelStatus.UNRESOLVED
    assert resolved["t1"].outcome is None


def test_resolve_annotations_raises_with_fewer_than_two_primary_labels() -> None:
    labels = {"t1": [_label("t1", "ann1", AnnotationOutcome.FAIL)]}
    with pytest.raises(ValueError):
        resolve_annotations(labels)


def test_resolve_annotations_raises_on_disagreement_with_no_adjudicator() -> None:
    labels = {
        "t1": [_label("t1", "ann1", AnnotationOutcome.FAIL), _label("t1", "ann2", AnnotationOutcome.PASS)]
    }
    with pytest.raises(ValueError):
        resolve_annotations(labels)


def test_resolve_annotations_raises_on_duplicate_annotator_for_the_same_task() -> None:
    """Repro from the user's review: the same annotator_id appearing
    twice as 'independent' primaries for one task must be rejected --
    independence is a condition on WHO labeled, not just HOW MANY labels
    exist."""
    labels = {"t1": [_label("t1", "ann1", AnnotationOutcome.FAIL), _label("t1", "ann1", AnnotationOutcome.FAIL)]}
    with pytest.raises(ValueError):
        resolve_annotations(labels)


def test_resolve_annotations_raises_when_adjudicator_is_also_a_primary_labeler() -> None:
    """An adjudicator must be a third, independent person -- not one of
    the original two primaries breaking the tie on their own disagreement."""
    labels = {
        "t1": [
            _label("t1", "ann1", AnnotationOutcome.FAIL),
            _label("t1", "ann2", AnnotationOutcome.PASS),
            _label("t1", "ann1", AnnotationOutcome.FAIL, is_adjudicator=True),
        ]
    }
    with pytest.raises(ValueError):
        resolve_annotations(labels)


def test_resolve_annotations_raises_on_task_id_mismatch() -> None:
    """A label actually stamped with a different task_id than the key it
    was grouped under is a caller/data bug, not something to silently
    resolve under the wrong key."""
    mismatched_label = AnnotationLabel(task_id="t2", annotator_id="ann1", outcome=AnnotationOutcome.FAIL)
    labels = {"t1": [mismatched_label, _label("t1", "ann2", AnnotationOutcome.FAIL)]}
    with pytest.raises(ValueError):
        resolve_annotations(labels)


def test_resolved_annotation_rejects_outcome_set_while_unresolved() -> None:
    with pytest.raises(ValueError):
        ResolvedAnnotation(
            task_id="t1", outcome=AnnotationOutcome.FAIL, status=ResolvedLabelStatus.UNRESOLVED, label_count=3
        )


def test_resolved_annotation_rejects_missing_outcome_when_resolved() -> None:
    with pytest.raises(ValueError):
        ResolvedAnnotation(task_id="t1", outcome=None, status=ResolvedLabelStatus.AGREED, label_count=2)


# ---------------------------------------------------------------------------
# compute_agreement_report
# ---------------------------------------------------------------------------


def test_compute_agreement_report_overall_and_per_stratum() -> None:
    tasks = {"t1": _task("t1", trigger=True), "t2": _task("t2", trigger=True), "t3": _task("t3", trigger=False)}
    resolved = {
        "t1": ResolvedAnnotation(task_id="t1", outcome=AnnotationOutcome.FAIL, status=ResolvedLabelStatus.AGREED, label_count=2),
        "t2": ResolvedAnnotation(task_id="t2", outcome=AnnotationOutcome.FAIL, status=ResolvedLabelStatus.ADJUDICATED, label_count=3),
        "t3": ResolvedAnnotation(task_id="t3", outcome=AnnotationOutcome.PASS, status=ResolvedLabelStatus.AGREED, label_count=2),
    }

    report = compute_agreement_report(tasks, resolved)

    assert report.total_tasks == 3
    assert report.agreed_count == 2
    assert report.adjudicated_count == 1
    assert report.unresolved_count == 0
    assert report.disagreement_rate == pytest.approx(1 / 3)
    trigger_key = tasks["t1"].stratum.key
    non_trigger_key = tasks["t3"].stratum.key
    assert report.disagreement_rate_by_stratum[trigger_key] == pytest.approx(0.5)
    assert report.disagreement_rate_by_stratum[non_trigger_key] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_stratum_weights
# ---------------------------------------------------------------------------


def test_compute_stratum_weights_basic() -> None:
    weights = compute_stratum_weights(population_counts={"A": 100, "B": 50}, sample_counts={"A": 10, "B": 50})
    assert weights["A"].weight == pytest.approx(10.0)
    assert weights["B"].weight == pytest.approx(1.0)


def test_compute_stratum_weights_raises_when_a_population_stratum_was_never_sampled() -> None:
    with pytest.raises(ValueError):
        compute_stratum_weights(population_counts={"A": 100, "B": 50}, sample_counts={"A": 10})


def test_compute_stratum_weights_raises_on_zero_sample_count() -> None:
    with pytest.raises(ValueError):
        compute_stratum_weights(population_counts={"A": 100}, sample_counts={"A": 0})


# ---------------------------------------------------------------------------
# compute_verifier_precision_recall (raw, unweighted)
# ---------------------------------------------------------------------------


def test_compute_verifier_precision_recall() -> None:
    predictions = {
        "t1": CheckOutcome.FAIL,
        "t2": CheckOutcome.PASS,
        "t3": CheckOutcome.FAIL,
        "t4": CheckOutcome.UNKNOWN,
    }
    gold = {
        "t1": ResolvedAnnotation(task_id="t1", outcome=AnnotationOutcome.FAIL, status=ResolvedLabelStatus.AGREED, label_count=2),
        "t2": ResolvedAnnotation(task_id="t2", outcome=AnnotationOutcome.FAIL, status=ResolvedLabelStatus.AGREED, label_count=2),
        "t3": ResolvedAnnotation(task_id="t3", outcome=AnnotationOutcome.PASS, status=ResolvedLabelStatus.AGREED, label_count=2),
        "t4": ResolvedAnnotation(task_id="t4", outcome=AnnotationOutcome.FAIL, status=ResolvedLabelStatus.AGREED, label_count=2),
        "t5": ResolvedAnnotation(task_id="t5", outcome=None, status=ResolvedLabelStatus.UNRESOLVED, label_count=3),
    }

    result = compute_verifier_precision_recall(predictions, gold)

    # t4 excluded (verifier UNKNOWN), t5 excluded (unresolved gold) -> sample_size=3 (t1,t2,t3)
    assert result.sample_size == 3
    assert result.precision == pytest.approx(0.5)  # TP=1 (t1) / predicted_positive=2 (t1,t3)
    assert result.recall == pytest.approx(0.5)  # TP=1 (t1) / actual_positive=2 (t1,t2)


def test_compute_verifier_precision_recall_none_when_denominator_is_zero() -> None:
    predictions = {"t1": CheckOutcome.PASS}
    gold = {
        "t1": ResolvedAnnotation(task_id="t1", outcome=AnnotationOutcome.PASS, status=ResolvedLabelStatus.AGREED, label_count=2)
    }
    result = compute_verifier_precision_recall(predictions, gold)
    assert result.precision is None  # nothing predicted FAIL
    assert result.recall is None  # nothing is actually FAIL
    assert result.sample_size == 1


def test_compute_verifier_precision_recall_excludes_a_genuine_agreed_unknown_gold() -> None:
    """Repro from the user's review: a genuinely AGREED gold of UNKNOWN
    (annotators independently agreed they could not tell) was falling
    through to 'not actual_fail', so a FAIL prediction against it was
    wrongly counted as a false positive (precision dragged to 0.0 by a
    single such case). A valid gold for this metric is PASS or FAIL only."""
    predictions = {"t1": CheckOutcome.FAIL}
    gold = {
        "t1": ResolvedAnnotation(
            task_id="t1", outcome=AnnotationOutcome.UNKNOWN, status=ResolvedLabelStatus.AGREED, label_count=2
        )
    }

    result = compute_verifier_precision_recall(predictions, gold)

    assert result.sample_size == 0
    assert result.precision is None
    assert result.recall is None


# ---------------------------------------------------------------------------
# compute_weighted_verifier_precision_recall (population-level estimate)
# ---------------------------------------------------------------------------


def test_compute_weighted_verifier_precision_recall_differs_from_the_unweighted_sample_metric() -> None:
    """Repro from the user's review: compute_stratum_weights existed but
    nothing ever applied its output to a metric -- a stratified sample
    that deliberately oversampled a rare stratum reported the SAME recall
    as the naive unweighted version. Here: t1 (caught, light stratum
    weight 1.0) and t2 (missed, heavy stratum weight 9.0) give an
    unweighted recall of 0.5 (1 of 2 caught) but a population-weighted
    recall of 0.1 -- the missed case counts far more once its stratum's
    true population share is accounted for."""
    predictions = {"t1": CheckOutcome.FAIL, "t2": CheckOutcome.PASS}
    gold = {
        "t1": ResolvedAnnotation(task_id="t1", outcome=AnnotationOutcome.FAIL, status=ResolvedLabelStatus.AGREED, label_count=2),
        "t2": ResolvedAnnotation(task_id="t2", outcome=AnnotationOutcome.FAIL, status=ResolvedLabelStatus.AGREED, label_count=2),
    }
    task_strata = {"t1": "A", "t2": "B"}
    weights = {
        "A": StratumWeight(stratum_key="A", population_count=10, sample_count=10, weight=1.0),
        "B": StratumWeight(stratum_key="B", population_count=90, sample_count=10, weight=9.0),
    }

    unweighted = compute_verifier_precision_recall(predictions, gold)
    weighted = compute_weighted_verifier_precision_recall(predictions, gold, task_strata, weights)

    assert unweighted.recall == pytest.approx(0.5)
    assert weighted.recall == pytest.approx(0.1)
    assert weighted.sample_size == unweighted.sample_size == 2


def test_compute_weighted_verifier_precision_recall_uses_the_same_inclusion_filter_as_unweighted() -> None:
    """A verifier UNKNOWN prediction and an UNRESOLVED/UNKNOWN gold must
    be excluded here exactly as they are in the unweighted version --
    weighting changes how included tasks are combined, never which tasks
    are included."""
    predictions = {"t1": CheckOutcome.FAIL, "t2": CheckOutcome.UNKNOWN}
    gold = {
        "t1": ResolvedAnnotation(task_id="t1", outcome=AnnotationOutcome.FAIL, status=ResolvedLabelStatus.AGREED, label_count=2),
        "t2": ResolvedAnnotation(task_id="t2", outcome=AnnotationOutcome.FAIL, status=ResolvedLabelStatus.AGREED, label_count=2),
    }
    task_strata = {"t1": "A", "t2": "A"}
    weights = {"A": StratumWeight(stratum_key="A", population_count=10, sample_count=10, weight=1.0)}

    result = compute_weighted_verifier_precision_recall(predictions, gold, task_strata, weights)
    assert result.sample_size == 1  # only t1


def test_compute_weighted_verifier_precision_recall_raises_for_an_unmapped_stratum() -> None:
    predictions = {"t1": CheckOutcome.FAIL}
    gold = {
        "t1": ResolvedAnnotation(task_id="t1", outcome=AnnotationOutcome.FAIL, status=ResolvedLabelStatus.AGREED, label_count=2)
    }
    with pytest.raises(ValueError):
        compute_weighted_verifier_precision_recall(predictions, gold, task_strata={}, weights={})

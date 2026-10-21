"""Coverage for src/core/annotation.py (Stage 6): stratified task sampling,
the disagreement-resolution procedure (agree / adjudicate / unresolved),
agreement reporting, stratum weighting, and verifier precision/recall
against resolved human gold. Every label in this file is synthetic test
fixture data, not real human judgment -- this module cannot produce real
labels itself (see its own module docstring)."""

from __future__ import annotations

import pytest

from src.core.annotation import (
    compute_agreement_report,
    compute_stratum_weights,
    compute_verifier_precision_recall,
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
)


def _task(task_id: str, *, trigger: bool, domain: str = "mathematics") -> AnnotationTask:
    return AnnotationTask(
        task_id=task_id,
        artifact=ArtifactRef(artifact_id=task_id, version=1, kind=ArtifactKind.BLOCK),
        content_shown="some content",
        stratum=AnnotationStratum(domain=domain, difficulty="basic", trigger=trigger),
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
# compute_verifier_precision_recall
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

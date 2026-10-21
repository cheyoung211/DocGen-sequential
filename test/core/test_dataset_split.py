"""Coverage for src/core/dataset_split.py (Stage 6): family-grouped
train/dev/test assignment (no template/topic/graph-structure family ever
splits across the boundary) and the held-out test accessor's explicit
acknowledge-to-read gate."""

from __future__ import annotations

from typing import Dict

import pytest

from src.core.dataset_split import (
    HeldOutSplitLockedError,
    build_fixed_split,
    dev_sample_ids,
    held_out_test_sample_ids,
    train_sample_ids,
)
from src.core.schemas import SplitFamilyKey, SplitName


def _families(n_families: int, samples_per_family: int) -> dict:
    """Every family gets a fully distinct document_type/topic/
    graph_structure_bucket -- i.e. families here share NOTHING with each
    other on any axis, so this fixture stays non-degenerate under the
    real (any-shared-attribute) grouping relation. A fixture that reused
    one attribute (e.g. the same graph_structure_bucket) across every
    family would, correctly, merge all of them into a single group -- see
    the dedicated cross-attribute tests below for that case instead."""
    out = {}
    for f in range(n_families):
        key = SplitFamilyKey(
            document_type=f"doc_type_{f}", topic=f"topic_{f}", graph_structure_bucket=f"bucket_{f}"
        )
        for s in range(samples_per_family):
            out[f"sample_{f}_{s}"] = key
    return out


def test_build_fixed_split_assigns_every_sample_exactly_once() -> None:
    families = _families(n_families=10, samples_per_family=3)
    split = build_fixed_split(families, seed=1)

    all_assigned = [a.sample_id for a in split.assignments]
    assert sorted(all_assigned) == sorted(families.keys())
    assert len(all_assigned) == len(set(all_assigned))


def test_build_fixed_split_never_splits_a_family_across_boundaries() -> None:
    families = _families(n_families=8, samples_per_family=4)
    split = build_fixed_split(families, seed=3)

    split_by_sample = {a.sample_id: a.split for a in split.assignments}
    by_family: dict = {}
    for sample_id, family_key in families.items():
        by_family.setdefault(family_key.key, set()).add(split_by_sample[sample_id])

    for family_key, splits_seen in by_family.items():
        assert len(splits_seen) == 1, f"family {family_key!r} was split across {splits_seen}"


def test_build_fixed_split_merges_samples_sharing_any_single_attribute() -> None:
    """Repro from the user's review: two samples sharing the same
    document_type (template) and graph_structure_bucket, but a different
    topic, must still never cross the train/dev/test boundary -- each of
    the three axes is individually protected, not only their exact joint
    combination (the bug: the first draft grouped only on exact-tuple
    equality, so this pair could and did land in different splits)."""
    families = {
        "s1": SplitFamilyKey(document_type="lecture_note", topic="topic_a", graph_structure_bucket="g1"),
        "s2": SplitFamilyKey(document_type="lecture_note", topic="topic_b", graph_structure_bucket="g1"),
    }
    split = build_fixed_split(families, seed=0)
    split_by_sample = {a.sample_id: a.split for a in split.assignments}
    assert split_by_sample["s1"] == split_by_sample["s2"]


def test_build_fixed_split_transitively_merges_through_a_shared_attribute_chain() -> None:
    """s1/s2 share document_type; s2/s3 share topic; s1 and s3 share
    neither attribute directly with each other, but must still land in the
    same split because they are connected through s2 -- correctly
    implementing "no axis value crosses the boundary" requires the full
    transitive closure, not just pairwise tuple checks."""
    families = {
        "s1": SplitFamilyKey(document_type="doc_a", topic="topic_1", graph_structure_bucket="g1"),
        "s2": SplitFamilyKey(document_type="doc_a", topic="topic_2", graph_structure_bucket="g2"),
        "s3": SplitFamilyKey(document_type="doc_b", topic="topic_2", graph_structure_bucket="g3"),
    }
    split = build_fixed_split(families, seed=0)
    split_by_sample = {a.sample_id: a.split for a in split.assignments}
    assert split_by_sample["s1"] == split_by_sample["s2"] == split_by_sample["s3"]


def test_no_individual_attribute_value_ever_crosses_a_split_boundary() -> None:
    """General property, beyond the tuple-identity check: for ANY single
    document_type, topic, or graph_structure_bucket value present in the
    input, every sample carrying it must land in exactly one split."""
    families = _families(n_families=8, samples_per_family=3)
    # Deliberately overlap one extra sample on exactly one attribute with
    # an existing family -- the base fixture's families are otherwise
    # fully disjoint, so without this the property would hold vacuously.
    families["extra_shared_doc_type"] = SplitFamilyKey(
        document_type="doc_type_0", topic="unique_topic_x", graph_structure_bucket="unique_bucket_x"
    )

    split = build_fixed_split(families, seed=2)
    split_by_sample = {a.sample_id: a.split for a in split.assignments}

    splits_seen_by_attribute: Dict[str, set] = {}
    for sample_id, key in families.items():
        for attribute_value in (
            f"doc:{key.document_type}",
            f"topic:{key.topic}",
            f"graph:{key.graph_structure_bucket}",
        ):
            splits_seen_by_attribute.setdefault(attribute_value, set()).add(split_by_sample[sample_id])

    for attribute_value, splits_seen in splits_seen_by_attribute.items():
        assert len(splits_seen) == 1, f"attribute {attribute_value!r} crossed split boundaries: {splits_seen}"


def test_build_fixed_split_is_deterministic_given_the_same_seed() -> None:
    families = _families(n_families=12, samples_per_family=2)
    first = build_fixed_split(families, seed=7)
    second = build_fixed_split(families, seed=7)
    assert {a.sample_id: a.split for a in first.assignments} == {a.sample_id: a.split for a in second.assignments}


def test_build_fixed_split_is_independent_of_input_iteration_order() -> None:
    """Repro from the user's review: the union-find's internal root label
    depends on processing order, and build_fixed_split was sorting/
    shuffling groups BY that root -- so the same 20 samples with the same
    seed, with only the input dict's iteration order reversed, moved
    samples to a different split (12 of 20 in the user's own repro). The
    actual partition (which samples belong together) is provably
    order-independent; only the (now fixed) group-keying was leaking
    order into the result.

    Sample ids are deliberately NOT in family/numeric order -- with
    strictly-increasing ids (one family per consecutive pair), the old
    buggy root happens to still sort the same way after a plain reversal,
    making the bug invisible; scrambled ids expose it (confirmed against
    the pre-fix code before relying on this fixture)."""
    ids_per_family = [
        ["m3", "a1"], ["z9", "b2"], ["k4", "c0"], ["q7", "d5"], ["x2", "e8"],
        ["n6", "f1"], ["t0", "g3"], ["y5", "h9"], ["p8", "i2"], ["v1", "j4"],
    ]
    families: Dict[str, SplitFamilyKey] = {}
    for f, ids in enumerate(ids_per_family):
        key = SplitFamilyKey(document_type=f"doc_type_{f}", topic=f"topic_{f}", graph_structure_bucket=f"bucket_{f}")
        for sample_id in ids:
            families[sample_id] = key

    forward = build_fixed_split(families, seed=1)
    reversed_families = dict(reversed(list(families.items())))
    backward = build_fixed_split(reversed_families, seed=1)

    assert {a.sample_id: a.split for a in forward.assignments} == {
        a.sample_id: a.split for a in backward.assignments
    }


def test_build_fixed_split_locked_by_default() -> None:
    split = build_fixed_split(_families(4, 2), seed=1)
    assert split.locked is True


def test_build_fixed_split_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        build_fixed_split({}, seed=1)


def test_build_fixed_split_rejects_fractions_not_summing_to_one() -> None:
    with pytest.raises(ValueError):
        build_fixed_split(_families(2, 2), seed=1, train_frac=0.5, dev_frac=0.5, test_frac=0.5)


def test_train_dev_test_accessors_partition_every_sample() -> None:
    families = _families(n_families=15, samples_per_family=2)
    split = build_fixed_split(families, seed=5)

    train = set(train_sample_ids(split))
    dev = set(dev_sample_ids(split))
    test = set(held_out_test_sample_ids(split, acknowledge_test_use=True))

    assert train & dev == set()
    assert train & test == set()
    assert dev & test == set()
    assert train | dev | test == set(families.keys())


def test_held_out_test_sample_ids_refuses_without_explicit_acknowledgement() -> None:
    split = build_fixed_split(_families(6, 2), seed=1)
    with pytest.raises(HeldOutSplitLockedError):
        held_out_test_sample_ids(split)


def test_held_out_test_sample_ids_succeeds_with_explicit_acknowledgement() -> None:
    split = build_fixed_split(_families(6, 2), seed=1)
    result = held_out_test_sample_ids(split, acknowledge_test_use=True)
    assert isinstance(result, list)

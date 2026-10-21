"""Coverage for src/core/dataset_split.py (Stage 6): family-grouped
train/dev/test assignment (no template/topic/graph-structure family ever
splits across the boundary) and the held-out test accessor's explicit
acknowledge-to-read gate."""

from __future__ import annotations

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
    out = {}
    for f in range(n_families):
        key = SplitFamilyKey(document_type=f"doc_type_{f}", topic=f"topic_{f}", graph_structure_bucket="bucket_a")
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


def test_build_fixed_split_is_deterministic_given_the_same_seed() -> None:
    families = _families(n_families=12, samples_per_family=2)
    first = build_fixed_split(families, seed=7)
    second = build_fixed_split(families, seed=7)
    assert {a.sample_id: a.split for a in first.assignments} == {a.sample_id: a.split for a in second.assignments}


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

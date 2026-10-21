"""Train/dev/test split (Stage 6 of the "Revision 2026.10.15" plan:
"실패·행동 데이터셋과 human gold 구축" -- the "template·topic·graph 구조를
고려한 train/dev/test 분할을 고정하고, held-out test를 정책 개선에 사용하지
않는다" half of that checklist).

Reuses ``docs/stage1_experiment_design.md`` §7's own framing: group samples
by family (document_type/topic/graph-structure-bucket) first, then assign
whole families to a split -- never individual samples -- so a template,
topic, or graph structure that appears in train/dev never also appears in
test.
"""

from __future__ import annotations

import random
from typing import Dict, List

from src.core.schemas import DatasetSplit, SplitAssignment, SplitFamilyKey, SplitName


class HeldOutSplitLockedError(RuntimeError):
    """Raised by ``held_out_test_sample_ids`` when called without
    ``acknowledge_test_use=True``. Stage 6's completion criterion is "held-
    out test를 정책 개선에 사용하지 않는다" -- enforced here at the one
    accessor that can read the test split out at all, applied
    unconditionally regardless of ``DatasetSplit.locked`` (an informational/
    audit field, not itself the gate), rather than left as a convention a
    future Stage 7 policy-tuning script could violate simply by forgetting
    it (the same "apply to the type itself, not a checklist of call sites"
    preference this project's own review history keeps landing on).

    Deliberately not named with a ``Test``/``test_`` prefix despite what it
    is about -- pytest's default collection treats any top-level ``test_*``
    callable or ``Test*`` class as a test case, and both this error and the
    accessor below would otherwise either be silently mis-collected or
    (for the function) actually invoked as a zero-argument test by pytest
    the moment a caller does ``from src.core.dataset_split import
    held_out_test_sample_ids``, exactly the failure mode a first draft of
    this module's own test suite hit immediately.
    """


def build_fixed_split(
    sample_families: Dict[str, SplitFamilyKey],
    *,
    seed: int,
    train_frac: float = 0.7,
    dev_frac: float = 0.15,
    test_frac: float = 0.15,
    dataset_version: str = "v1",
) -> DatasetSplit:
    """Assign every ``sample_id`` in ``sample_families`` to TRAIN/DEV/TEST.

    Samples sharing the same ``SplitFamilyKey.key`` are always assigned
    together, as one unit -- the hard constraint this function exists to
    satisfy. Families are then greedily assigned, in a ``seed``-shuffled
    order, to whichever split is currently furthest below its target count
    (``train_frac``/``dev_frac``/``test_frac`` of the total sample count) --
    a best-effort approximation of the target ratios, since the hard
    family-grouping constraint can make an exact ratio impossible (e.g. one
    family containing most of the pool). Deterministic given the same
    inputs and ``seed``.

    Raises ``ValueError`` if ``sample_families`` is empty or the three
    fractions do not sum to 1.0.
    """
    if not sample_families:
        raise ValueError("sample_families must not be empty.")
    total_frac = train_frac + dev_frac + test_frac
    if abs(total_frac - 1.0) > 1e-9:
        raise ValueError(f"train_frac + dev_frac + test_frac must sum to 1.0, got {total_frac}.")

    families: Dict[str, List[str]] = {}
    key_objects: Dict[str, SplitFamilyKey] = {}
    for sample_id, family_key in sample_families.items():
        families.setdefault(family_key.key, []).append(sample_id)
        key_objects[family_key.key] = family_key

    family_order = sorted(families)
    random.Random(seed).shuffle(family_order)

    total_samples = len(sample_families)
    targets = {
        SplitName.TRAIN: train_frac * total_samples,
        SplitName.DEV: dev_frac * total_samples,
        SplitName.TEST: test_frac * total_samples,
    }
    assigned_counts = {SplitName.TRAIN: 0, SplitName.DEV: 0, SplitName.TEST: 0}

    assignments: List[SplitAssignment] = []
    for family_key_str in family_order:
        sample_ids = sorted(families[family_key_str])
        chosen = max(SplitName, key=lambda split: targets[split] - assigned_counts[split])
        assigned_counts[chosen] += len(sample_ids)
        for sample_id in sample_ids:
            assignments.append(
                SplitAssignment(
                    dataset_version=dataset_version,
                    sample_id=sample_id,
                    family_key=key_objects[family_key_str],
                    split=chosen,
                )
            )

    return DatasetSplit(dataset_version=dataset_version, seed=seed, assignments=assignments, locked=True)


def train_sample_ids(split: DatasetSplit) -> List[str]:
    return [a.sample_id for a in split.assignments if a.split == SplitName.TRAIN]


def dev_sample_ids(split: DatasetSplit) -> List[str]:
    return [a.sample_id for a in split.assignments if a.split == SplitName.DEV]


def held_out_test_sample_ids(split: DatasetSplit, *, acknowledge_test_use: bool = False) -> List[str]:
    """The held-out test split's sample ids -- refuses by default.

    Stage 7 (budget-aware policy improvement) must only ever call
    ``train_sample_ids``/``dev_sample_ids``; Stage 8's one-time final
    evaluation is the only intended caller of this function with
    ``acknowledge_test_use=True``. Passing that flag does not change
    anything about ``split`` itself (no mutation, no unlocking) -- it is
    purely an explicit admission from the caller, visible in a code review,
    that it intends to look at the held-out set right now.
    """
    if not acknowledge_test_use:
        raise HeldOutSplitLockedError(
            "the test split is held out from policy improvement by design "
            "(Notion Stage 6: 'held-out test를 정책 개선에 사용하지 않는다') -- "
            "pass acknowledge_test_use=True only from a final, one-time evaluation "
            "(Stage 8), never from policy tuning (Stage 7)."
        )
    return [a.sample_id for a in split.assignments if a.split == SplitName.TEST]

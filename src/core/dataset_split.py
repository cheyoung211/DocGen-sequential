"""Train/dev/test split (Stage 6 of the "Revision 2026.10.15" plan:
"실패·행동 데이터셋과 human gold 구축" -- the "template·topic·graph 구조를
고려한 train/dev/test 분할을 고정하고, held-out test를 정책 개선에 사용하지
않는다" half of that checklist).

Reuses ``docs/stage1_experiment_design.md`` §7's own framing: group samples
by family (document_type/topic/graph-structure-bucket) first, then assign
whole families to a split -- never individual samples -- so a template,
topic, or graph structure that appears in train/dev never also appears in
test.

**Grouping is connected-components over the three attributes individually,
not an exact-tuple match.** A review caught that the first draft grouped
samples only when ALL THREE of ``document_type``/``topic``/
``graph_structure_bucket`` matched at once (``SplitFamilyKey.key``, a
plain tuple-equality key) -- two samples sharing the same ``document_type``
and ``graph_structure_bucket`` but a different ``topic`` landed in
*different* groups and could legally be split across train/dev/test, even
though that is exactly the leak the checklist's "template·topic·graph 구조를
고려한" wording forbids for EACH axis independently, not only for their
joint combination. "No value of any single axis crosses a split boundary"
is only satisfiable by grouping every sample that shares ANY one of the
three attribute values into one connected component (``_group_by_shared_attribute``)
-- a strictly coarser, and the only correct, partition: if two samples
sharing one axis value ended up in different components, that axis value
would necessarily appear in two different splits.

**Group keys are ``min(members)``, never the raw union-find root.** A
second review caught that the union-find's internal root label is an
artifact of processing order (which member happened to get unioned into
which), and ``build_fixed_split`` was sorting/shuffling groups BY that
root -- so the exact same ``sample_families`` dict, same ``seed``, with
only its iteration order reversed, produced a materially different split
(12 of 20 samples moved in the reported repro) even though the actual
groups (which samples belong together) are themselves provably
order-independent. Re-keying each group by the minimum member id --
a pure function of group membership alone -- fixes this without changing
anything about which samples end up grouped together.
"""

from __future__ import annotations

import random
from typing import Dict, List

from src.core.schemas import DatasetSplit, SplitAssignment, SplitFamilyKey, SplitName


def _group_by_shared_attribute(sample_families: Dict[str, SplitFamilyKey]) -> Dict[str, List[str]]:
    """Union-find over ``sample_families``: two samples end up in the same
    group iff they are connected by a chain of shared ``document_type``,
    ``topic``, or ``graph_structure_bucket`` values -- not only when all
    three match simultaneously. Returns ``{min(member sample_ids):
    [member sample_ids]}``.

    The key is deliberately ``min(members)``, NOT whichever sample_id the
    union-find's internal root happened to settle on. A review caught
    that the raw root depends on processing order (which member got
    unioned into which first, itself driven by ``sample_families``'s own
    dict/list iteration order) even though the actual partition -- which
    samples end up grouped together -- does not: connected components are
    a pure function of the "shares an attribute value" relation, order-
    independent by construction. ``build_fixed_split`` sorts and shuffles
    groups BY this dict's keys, so leaking the order-dependent raw root
    into those keys let the exact same data and ``seed`` produce a
    different split merely because ``sample_families`` was iterated in a
    different order. ``min(members)`` is a pure function of a group's
    final membership alone, so it cannot carry that leak.
    """
    parent: Dict[str, str] = {sample_id: sample_id for sample_id in sample_families}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    by_attribute_value: Dict[str, List[str]] = {}
    for sample_id, family_key in sample_families.items():
        for attribute_value in (
            f"document_type:{family_key.document_type}",
            f"topic:{family_key.topic}",
            f"graph_structure_bucket:{family_key.graph_structure_bucket}",
        ):
            by_attribute_value.setdefault(attribute_value, []).append(sample_id)

    for members in by_attribute_value.values():
        for other in members[1:]:
            union(members[0], other)

    raw_groups: Dict[str, List[str]] = {}
    for sample_id in sample_families:
        raw_groups.setdefault(find(sample_id), []).append(sample_id)
    return {min(members): members for members in raw_groups.values()}


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

    Samples connected by a shared ``document_type``, ``topic``, or
    ``graph_structure_bucket`` value -- transitively, via
    ``_group_by_shared_attribute`` -- are always assigned together, as one
    unit; this, not plain ``SplitFamilyKey`` tuple-equality, is the hard
    constraint this function exists to satisfy (see the module docstring).
    Groups are then greedily assigned, in a ``seed``-shuffled order, to
    whichever split is currently furthest below its target count
    (``train_frac``/``dev_frac``/``test_frac`` of the total sample count) --
    a best-effort approximation of the target ratios, since the hard
    grouping constraint can make an exact ratio impossible, and is now
    *more* likely to dominate than a naive tuple-based grouping would
    suggest: a single widely-reused ``document_type`` (or ``topic``, or
    ``graph_structure_bucket``) value can transitively pull a large
    fraction of all samples into one group. Deterministic given the same
    inputs and ``seed``. Each ``SplitAssignment.family_key`` records that
    sample's OWN ``document_type``/``topic``/``graph_structure_bucket`` for
    audit purposes -- not the (otherwise-meaningless) internal union-find
    group id it was actually grouped by.

    Raises ``ValueError`` if ``sample_families`` is empty or the three
    fractions do not sum to 1.0.
    """
    if not sample_families:
        raise ValueError("sample_families must not be empty.")
    total_frac = train_frac + dev_frac + test_frac
    if abs(total_frac - 1.0) > 1e-9:
        raise ValueError(f"train_frac + dev_frac + test_frac must sum to 1.0, got {total_frac}.")

    groups = _group_by_shared_attribute(sample_families)
    group_order = sorted(groups)
    random.Random(seed).shuffle(group_order)

    total_samples = len(sample_families)
    targets = {
        SplitName.TRAIN: train_frac * total_samples,
        SplitName.DEV: dev_frac * total_samples,
        SplitName.TEST: test_frac * total_samples,
    }
    assigned_counts = {SplitName.TRAIN: 0, SplitName.DEV: 0, SplitName.TEST: 0}

    assignments: List[SplitAssignment] = []
    for group_root in group_order:
        sample_ids = sorted(groups[group_root])
        chosen = max(SplitName, key=lambda split: targets[split] - assigned_counts[split])
        assigned_counts[chosen] += len(sample_ids)
        for sample_id in sample_ids:
            assignments.append(
                SplitAssignment(
                    dataset_version=dataset_version,
                    sample_id=sample_id,
                    family_key=sample_families[sample_id],
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

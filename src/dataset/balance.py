"""Domain/difficulty/document-type balancing (spec sections 7-8) and
selection of the ``dev_debug`` subset. All randomness here is seeded for
reproducibility (spec section 11: "deterministic random seed").
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Dict, List

from src.dataset.schemas import BenchmarkItem

logger = logging.getLogger(__name__)


def _value(x) -> str:
    return x.value if hasattr(x, "value") else str(x)


def enforce_domain_count(items: List[BenchmarkItem], target_count: int, *, rng: random.Random) -> List[BenchmarkItem]:
    """Deterministically truncate to at most ``target_count`` items."""
    if len(items) <= target_count:
        return items
    indices = list(range(len(items)))
    rng.shuffle(indices)
    keep = sorted(indices[:target_count])
    return [items[i] for i in keep]


def difficulty_distribution(items: List[BenchmarkItem]) -> Dict[str, int]:
    return dict(Counter(_value(item.difficulty) for item in items))


def document_type_distribution(items: List[BenchmarkItem]) -> Dict[str, int]:
    return dict(Counter(_value(item.document_type) for item in items))


def domain_distribution(items: List[BenchmarkItem]) -> Dict[str, int]:
    return dict(Counter(_value(item.domain) for item in items))


def check_difficulty_balance(
    items: List[BenchmarkItem],
    expected_ratios: Dict[str, float],
    *,
    tolerance: float = 0.05,
) -> List[str]:
    """Return human-readable warnings when the observed difficulty mix
    deviates from the configured target ratios by more than ``tolerance``.
    """
    if not items:
        return ["No items to check difficulty balance for."]
    warnings: List[str] = []
    counts = difficulty_distribution(items)
    total = len(items)
    for difficulty, expected_ratio in expected_ratios.items():
        observed = counts.get(difficulty, 0) / total
        if abs(observed - expected_ratio) > tolerance:
            warnings.append(
                f"difficulty={difficulty}: observed ratio {observed:.3f} deviates from "
                f"expected {expected_ratio:.3f} by more than tolerance {tolerance}."
            )
    return warnings


def select_dev_debug_subset(
    items_by_domain: Dict[str, List[BenchmarkItem]],
    size: int,
    *,
    rng: random.Random,
) -> List[BenchmarkItem]:
    """Pick a small, domain-balanced debug subset for smoke-testing consumers.

    This is a debug-only convenience subset, never a training split (see
    spec section 10: no train/validation/test splits are produced here).
    """
    domains = [d for d, pool in items_by_domain.items() if pool]
    if not domains:
        return []
    per_domain = max(1, size // len(domains))
    subset: List[BenchmarkItem] = []
    for domain in domains:
        pool = items_by_domain[domain]
        indices = list(range(len(pool)))
        rng.shuffle(indices)
        chosen = sorted(indices[:per_domain])
        subset.extend(pool[i] for i in chosen)
    return subset[:size]

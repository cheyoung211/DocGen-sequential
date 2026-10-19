"""Compute ``data/metadata/dataset_statistics.json`` and ``license_report.json``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.dataset.balance import difficulty_distribution, document_type_distribution, domain_distribution
from src.dataset.schemas import BenchmarkItem
from src.dataset.source_registry import SourceRegistry

logger = logging.getLogger(__name__)

_QUALITY_FIELDS = [
    "specificity", "evaluability", "latex_relevance",
    "long_form_suitability", "constraint_clarity", "source_safety",
]


def compute_dataset_statistics(items: List[BenchmarkItem]) -> Dict[str, Any]:
    if not items:
        return {"total_items": 0}

    word_counts = [item.target_length.min_words for item in items]
    section_counts = [len(item.required_sections) for item in items]
    component_counts = [len(item.required_components) for item in items]
    dependency_counts = [len(item.cross_section_dependencies) for item in items]

    return {
        "total_items": len(items),
        "by_domain": domain_distribution(items),
        "by_difficulty": difficulty_distribution(items),
        "by_document_type": document_type_distribution(items),
        "avg_min_words": round(sum(word_counts) / len(word_counts), 1),
        "avg_required_sections": round(sum(section_counts) / len(section_counts), 2),
        "avg_required_components": round(sum(component_counts) / len(component_counts), 2),
        "avg_cross_section_dependencies": round(sum(dependency_counts) / len(dependency_counts), 2),
        "quality_score_averages": _average_quality_scores(items),
    }


def _average_quality_scores(items: List[BenchmarkItem]) -> Dict[str, float]:
    scored = [item.quality_score for item in items if item.quality_score is not None]
    if not scored:
        return {}
    return {
        field: round(sum(getattr(qs, field) for qs in scored) / len(scored), 4)
        for field in _QUALITY_FIELDS
    }


def compute_license_report(registry: SourceRegistry) -> Dict[str, Any]:
    records = registry.all()
    by_license: Dict[str, int] = {}
    for record in records:
        by_license[record.license_name] = by_license.get(record.license_name, 0) + 1
    return {
        "total_sources": len(records),
        "by_license": by_license,
        "manual_review_required": registry.manual_review_ids(),
        "sources": [record.model_dump(mode="json") for record in records],
    }


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", path)

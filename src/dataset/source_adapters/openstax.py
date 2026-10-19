"""OpenStax metadata adapter (Tier 1 source, mathematics / statistics_ml).

Queries OpenStax's public CMS API v2 (``openstax.org/apps/cms/api/v2/pages``)
for book titles and table-of-contents metadata only. It never fetches book
prose, exercises, or examples -- OpenStax editions carry Creative Commons
licenses in general, but the exact license and edition can differ per title,
so this adapter deliberately does **not** assume a universal license: every
``SourceRecord`` it produces is marked ``manual_review_required=True`` until
a human confirms the license for that specific edition, per the dataset
spec's conservative source policy.

Verified live (2026-08-05): ``GET /apps/cms/api/v2/pages/?type=books.Book
&fields=title,slug&limit=N`` returns
``{"items": [{"title": ..., "meta": {"slug": ..., "html_url": ...}}]}``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.dataset.schemas import (
    ExtractionMethod,
    Seed,
    SeedSourceRef,
    SourceRecord,
    SourceType,
)
from src.dataset.source_adapters.base import CachedHttpClient, SourceAdapter, SourceUnavailableError
from src.dataset.source_registry import SourceRegistry

logger = logging.getLogger(__name__)

BOOKS_ENDPOINT = "https://openstax.org/apps/cms/api/v2/pages/"

# Keyword hints used to match a requested subdomain to relevant OpenStax
# book titles. Extend as new subdomains are added.
SUBDOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "real_analysis": ["calculus"],
    "calculus": ["calculus"],
    "linear_algebra": ["algebra"],
    "probability_theory": ["statistics"],
    "discrete_mathematics": ["algebra"],
    "numerical_analysis": ["calculus"],
    "descriptive_statistics": ["statistics"],
    "probability_distributions": ["statistics"],
    "estimation": ["statistics", "business statistics"],
    "hypothesis_testing": ["statistics"],
    "regression": ["statistics", "business statistics"],
}


class OpenStaxAdapter(SourceAdapter):
    source_type = SourceType.OPENSTAX.value

    def __init__(self, http_client: CachedHttpClient, registry: SourceRegistry) -> None:
        self._http = http_client
        self._registry = registry

    def fetch_seeds(self, subdomain: str, limit: int) -> List[Seed]:
        keywords = SUBDOMAIN_KEYWORDS.get(subdomain, [subdomain.replace("_", " ")])
        try:
            payload = self._http.get_json(
                BOOKS_ENDPOINT,
                params={"type": "books.Book", "fields": "title,slug", "limit": 100},
            )
        except SourceUnavailableError:
            logger.warning("OpenStax API unavailable; returning no seeds for %r.", subdomain)
            return []

        items: List[Dict[str, Any]] = payload.get("items", [])
        matches = [
            item
            for item in items
            if any(keyword.lower() in item.get("title", "").lower() for keyword in keywords)
        ][:limit]

        seeds: List[Seed] = []
        for item in matches:
            seed = self._to_seed(item, subdomain)
            if seed is not None:
                seeds.append(seed)
        return seeds

    def _to_seed(self, item: Dict[str, Any], subdomain: str) -> Optional[Seed]:
        title = item.get("title")
        meta = item.get("meta", {})
        slug = meta.get("slug")
        html_url = meta.get("html_url")
        if not title or not slug:
            return None

        source_id = f"openstax_{slug.replace('-', '_')}"
        record = SourceRecord(
            source_id=source_id,
            source_type=SourceType.OPENSTAX,
            source_title=title,
            source_url=html_url,
            publisher_or_owner="OpenStax (Rice University)",
            edition_or_version="unknown-edition (verify before use)",
            access_date=_today(),
            license_name="CC BY 4.0 (OpenStax typical; unverified for this specific edition)",
            license_url="https://openstax.org/general/licensing/",
            allowed_use="topic and learning-objective seed only; no prose reuse",
            attribution_text=f"Topic inspired by OpenStax's \"{title}\" (title/heading metadata only).",
            extraction_method=ExtractionMethod.API_METADATA,
            copied_text_flag=False,
            manual_review_required=True,
            notes=[
                "License auto-recorded as OpenStax's typical CC BY 4.0; a human "
                "must confirm the exact edition's license before this source "
                "record is used to justify redistribution of anything beyond "
                "title/heading metadata.",
            ],
        )
        self._registry.register(record)

        return Seed(
            seed_id=f"seed_{source_id}",
            domain=_domain_for_subdomain(subdomain),
            subdomain=subdomain,
            source=SeedSourceRef(
                source_id=record.source_id,
                source_title=record.source_title,
                source_url=record.source_url,
                license_name=record.license_name,
                access_date=record.access_date,
            ),
            topic=title,
            subtopic=subdomain.replace("_", " "),
            source_headings=[title],
            learning_objectives=[f"Explain the core concepts of {title} relevant to {subdomain.replace('_', ' ')}."],
            key_concepts=[subdomain.replace("_", " ")],
            prerequisites=[],
            candidate_notation=[],
            candidate_equations=[],
            candidate_components=["definition", "worked_example"],
            source_notes=[f"Seed topic derived from OpenStax book title only: {title!r}."],
            difficulty_hint="intermediate",
            document_type_hint="lecture_note",
        )


def _domain_for_subdomain(subdomain: str) -> str:
    stats_subdomains = {
        "descriptive_statistics",
        "probability_distributions",
        "estimation",
        "hypothesis_testing",
        "regression",
        "bias_variance_tradeoff",
        "bayesian_inference",
        "model_evaluation",
        "experimental_design",
    }
    return "statistics_ml" if subdomain in stats_subdomains else "mathematics"


def _today() -> str:
    from datetime import date

    return date.today().isoformat()

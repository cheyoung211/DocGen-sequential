"""NASA open-data metadata adapter (Tier 1 source, scientific_engineering).

Queries NASA's public CKAN catalog (``data.nasa.gov``) for dataset titles,
descriptions, and organization/discipline metadata, used only to derive
realistic scientific report topics -- never as a file input and never as a
source of prose to copy. The ``notes`` (description) field returned by the
API is used only to pick a topic label; it is not copied into any benchmark
item field.

Verified live (2026-08-05): ``GET /api/3/action/package_search?q=...&rows=N``
returns ``{"result": {"results": [{"title", "notes", "license_id",
"license_title", "organization": {"title"}, ...}]}}``.
"""

from __future__ import annotations

import logging
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

PACKAGE_SEARCH_ENDPOINT = "https://data.nasa.gov/api/3/action/package_search"

SUBDOMAIN_QUERIES: Dict[str, str] = {
    "astronomy_and_planetary_science": "planetary science",
    "climate_and_earth_science": "climate",
    "physics": "physics",
    "computer_science": "data science",
    "electrical_or_mechanical_engineering": "engineering",
    "environmental_science": "earth science",
    "scientific_methodology": "research data",
}


class NasaMetadataAdapter(SourceAdapter):
    source_type = SourceType.NASA_METADATA.value

    def __init__(self, http_client: CachedHttpClient, registry: SourceRegistry) -> None:
        self._http = http_client
        self._registry = registry

    def fetch_seeds(self, subdomain: str, limit: int) -> List[Seed]:
        query = SUBDOMAIN_QUERIES.get(subdomain, subdomain.replace("_", " "))
        try:
            payload = self._http.get_json(
                PACKAGE_SEARCH_ENDPOINT,
                params={"q": query, "rows": limit},
            )
        except SourceUnavailableError:
            logger.warning("NASA data catalog unavailable; returning no seeds for %r.", subdomain)
            return []

        results: List[Dict[str, Any]] = payload.get("result", {}).get("results", [])

        seeds: List[Seed] = []
        for item in results[:limit]:
            seed = self._to_seed(item, subdomain)
            if seed is not None:
                seeds.append(seed)
        return seeds

    def _to_seed(self, item: Dict[str, Any], subdomain: str) -> Optional[Seed]:
        title = item.get("title")
        name = item.get("name")
        if not title or not name:
            return None

        license_title = item.get("license_title") or "unspecified"
        organization = (item.get("organization") or {}).get("title", "NASA")
        landing_page = f"https://data.nasa.gov/dataset/{name}"

        source_id = f"nasa_{name.replace('-', '_')}"
        record = SourceRecord(
            source_id=source_id,
            source_type=SourceType.NASA_METADATA,
            source_title=title,
            source_url=landing_page,
            publisher_or_owner=organization,
            edition_or_version=None,
            access_date=_today(),
            license_name=license_title if license_title != "unspecified" else "US Government Work",
            license_url=None,
            allowed_use="topic and terminology seed only; dataset itself is not used as a model input",
            attribution_text=f"Topic inspired by the NASA open-data catalog entry \"{title}\" (metadata only).",
            extraction_method=ExtractionMethod.API_METADATA,
            copied_text_flag=False,
            manual_review_required=(license_title == "unspecified" or "other" in license_title.lower()),
            notes=["Dataset description used only to pick a topic label; not copied into any output field."],
        )
        self._registry.register(record)

        return Seed(
            seed_id=f"seed_{source_id}",
            domain="scientific_engineering",
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
            learning_objectives=[f"Explain the phenomenon or system behind \"{title}\" at a report-appropriate level."],
            key_concepts=[subdomain.replace("_", " ")],
            prerequisites=[],
            candidate_notation=[],
            candidate_equations=[],
            candidate_components=["classification_table", "figure_placeholder"],
            source_notes=[f"Seed topic derived from NASA catalog title only: {title!r}."],
            difficulty_hint="intermediate",
            document_type_hint="technical_report",
        )


def _today() -> str:
    from datetime import date

    return date.today().isoformat()

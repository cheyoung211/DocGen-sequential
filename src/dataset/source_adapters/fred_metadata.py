"""FRED (Federal Reserve Economic Data) series-metadata adapter (Tier 1
source, financial_economic domain).

Queries the official FRED API (``api.stlouisfed.org/fred``) for *series
metadata only* -- title, category, units, frequency, seasonal-adjustment
status, release name -- never actual time-series observations. Per the
dataset spec, v1 must not download or attach real economic data as a model
input; only metadata is used to construct realistic report topics.

Requires ``FRED_API_KEY`` to be set in the environment (never hard-coded).
If it is missing, :meth:`fetch_seeds` logs a warning and returns ``[]`` --
this is the adapter's graceful-degradation path, and is exercised by
default since ``network_sources_enabled`` is off in ``configs/dataset_v1.yaml``.

Endpoint shape (per FRED API docs, https://fred.stlouisfed.org/docs/api/fred/):
``GET /fred/series/search?search_text=...&api_key=...&file_type=json``
returns ``{"seriess": [{"id", "title", "frequency", "units",
"seasonal_adjustment", "notes", ...}]}``.
"""

from __future__ import annotations

import logging
import os
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

SERIES_SEARCH_ENDPOINT = "https://api.stlouisfed.org/fred/series/search"

SUBDOMAIN_QUERIES: Dict[str, str] = {
    "macroeconomic_analysis": "GDP growth",
    "monetary_policy": "federal funds rate",
    "market_structure": "unemployment rate",
    "economic_indicator_comparison": "consumer price index",
}


class FredMetadataAdapter(SourceAdapter):
    source_type = SourceType.FRED_METADATA.value

    def __init__(self, http_client: CachedHttpClient, registry: SourceRegistry) -> None:
        self._http = http_client
        self._registry = registry

    def fetch_seeds(self, subdomain: str, limit: int) -> List[Seed]:
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            logger.warning(
                "FRED_API_KEY is not set; skipping FredMetadataAdapter for %r "
                "(this is expected when network_sources_enabled is off).",
                subdomain,
            )
            return []

        query = SUBDOMAIN_QUERIES.get(subdomain, subdomain.replace("_", " "))
        try:
            payload = self._http.get_json(
                SERIES_SEARCH_ENDPOINT,
                params={
                    "search_text": query,
                    "api_key": api_key,
                    "file_type": "json",
                    "limit": limit,
                },
            )
        except SourceUnavailableError:
            logger.warning("FRED API unavailable; returning no seeds for %r.", subdomain)
            return []

        series_list: List[Dict[str, Any]] = payload.get("seriess", [])

        seeds: List[Seed] = []
        for item in series_list[:limit]:
            seed = self._to_seed(item, subdomain)
            if seed is not None:
                seeds.append(seed)
        return seeds

    def _to_seed(self, item: Dict[str, Any], subdomain: str) -> Optional[Seed]:
        series_id = item.get("id")
        title = item.get("title")
        if not series_id or not title:
            return None

        source_id = f"fred_{series_id.lower()}"
        record = SourceRecord(
            source_id=source_id,
            source_type=SourceType.FRED_METADATA,
            source_title=title,
            source_url=f"https://fred.stlouisfed.org/series/{series_id}",
            publisher_or_owner="Federal Reserve Bank of St. Louis",
            edition_or_version=None,
            access_date=_today(),
            license_name="FRED public data (attribution requested; verify current terms)",
            license_url="https://fred.stlouisfed.org/legal/",
            allowed_use="series metadata as topic seed only; no observation values used",
            attribution_text=f"Topic inspired by FRED series \"{title}\" ({series_id}), metadata only.",
            extraction_method=ExtractionMethod.API_METADATA,
            copied_text_flag=False,
            manual_review_required=True,
            notes=["Series metadata only; no time-series values were fetched or stored."],
        )
        self._registry.register(record)

        units = item.get("units", "unspecified units")
        frequency = item.get("frequency", "unspecified frequency")

        return Seed(
            seed_id=f"seed_{source_id}",
            domain="financial_economic",
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
            learning_objectives=[f"Interpret {title} ({units}, {frequency}) in a policy or analytical context."],
            key_concepts=[subdomain.replace("_", " "), units, frequency],
            prerequisites=[],
            candidate_notation=[],
            candidate_equations=[],
            candidate_components=["terminology_table", "comparison_table"],
            source_notes=[f"Seed topic derived from FRED series title only: {title!r}."],
            difficulty_hint="intermediate",
            document_type_hint="economic_policy_report",
        )


def _today() -> str:
    from datetime import date

    return date.today().isoformat()

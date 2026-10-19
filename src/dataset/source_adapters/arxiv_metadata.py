"""STUB adapter for arXiv paper metadata (Tier 2 source).

Intended use (per the dataset spec): pull *metadata and section structure
only* -- paper titles, abstract-derived topic phrases, and category-implied
subdomain hints from arXiv's Atom API
(``http://export.arxiv.org/api/query?search_query=cat:math.CA&...``) -- used
purely as structural inspiration for seed topics/subdomains, never as a
source of prose to copy into a benchmark item.

TODO(v1.1): implement against the arXiv API:
  1. Query ``export.arxiv.org/api/query`` with a category + keyword filter
     per subdomain (e.g. ``cat:math.CA`` for real analysis, ``cat:stat.ML``
     for statistics/ML).
  2. Parse the Atom XML response for ``<entry><title>`` and
     ``<entry><category>`` only -- do not extract ``<summary>`` text into
     ``candidate_equations``/``key_concepts`` verbatim; use it only to pick a
     topic label, then write original phrasing.
  3. Record a ``SourceRecord`` with ``source_type=SourceType.ARXIV_METADATA``,
     ``license_name="arXiv non-exclusive license (metadata only)"``, and
     ``extraction_method=ExtractionMethod.STRUCTURAL_ONLY``.
  4. Respect arXiv's rate-limit guidance (>= 3 seconds between requests) via
     ``CachedHttpClient``'s ``min_request_interval_seconds``.
"""

from __future__ import annotations

import logging
from typing import List

from src.dataset.schemas import Seed, SourceType
from src.dataset.source_adapters.base import SourceAdapter

logger = logging.getLogger(__name__)


class ArxivMetadataAdapter(SourceAdapter):
    source_type = SourceType.ARXIV_METADATA.value

    def fetch_seeds(self, subdomain: str, limit: int) -> List[Seed]:
        logger.info(
            "ArxivMetadataAdapter.fetch_seeds(subdomain=%r) is a stub (TODO v1.1); "
            "returning no seeds.",
            subdomain,
        )
        return []

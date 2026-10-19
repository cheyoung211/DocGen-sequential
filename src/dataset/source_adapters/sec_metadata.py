"""STUB adapter for SEC EDGAR filing metadata and XBRL taxonomy concepts
(Tier 1 source, financial_economic domain).

Intended use (per the dataset spec): pull filing *section names*, XBRL
*concept names*, standard financial-statement categories, and risk-factor
category labels -- never long filing passages.

TODO(v1.1): implement against SEC's official metadata endpoints:
  1. ``https://www.sec.gov/cgi-bin/browse-edgar`` or the JSON submissions API
     at ``https://data.sec.gov/submissions/CIK##########.json`` for filing
     *types* (10-K, 10-Q, 8-K) and standard section names -- not filing text.
  2. ``https://xbrl.sec.gov/`` taxonomy concept lists (e.g. us-gaap concept
     names like ``Revenues``, ``GrossProfit``, ``OperatingIncomeLoss``) as
     ``candidate_notation``/``candidate_equations`` inspiration for
     terminology tables, not as data values.
  3. SEC requires a descriptive ``User-Agent`` header identifying the
     requester (see ``CachedHttpClient(user_agent=...)``) and a fair-access
     rate limit -- do not exceed a few requests per second.
  4. Record a ``SourceRecord`` with ``source_type=SourceType.SEC_METADATA``,
     ``license_name="US Government Work"``,
     ``extraction_method=ExtractionMethod.API_METADATA``.
  5. Never fetch or store actual filing numbers/figures; this stays a
     metadata-only, text-free adapter for v1.
"""

from __future__ import annotations

import logging
from typing import List

from src.dataset.schemas import Seed, SourceType
from src.dataset.source_adapters.base import SourceAdapter

logger = logging.getLogger(__name__)


class SecMetadataAdapter(SourceAdapter):
    source_type = SourceType.SEC_METADATA.value

    def fetch_seeds(self, subdomain: str, limit: int) -> List[Seed]:
        logger.info(
            "SecMetadataAdapter.fetch_seeds(subdomain=%r) is a stub (TODO v1.1); "
            "returning no seeds.",
            subdomain,
        )
        return []

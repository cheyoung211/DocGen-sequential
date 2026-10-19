"""STUB adapter for World Bank Indicators metadata (Tier 1 source,
financial_economic domain).

Intended use (per the dataset spec): pull indicator *metadata* -- indicator
code, indicator name, unit, source note, topic, source organization -- to
build economic comparison and policy-report topics. Never actual time-series
observations.

TODO(v1.1): implement against the official World Bank Indicators API:
  1. ``https://api.worldbank.org/v2/indicator?format=json&per_page=...`` for
     the indicator catalog (code, name, sourceNote, sourceOrganization,
     topics).
  2. ``https://api.worldbank.org/v2/topic?format=json`` to map indicators to
     economics topics (e.g. "Economic Policy & Debt") for subdomain
     assignment.
  3. Record a ``SourceRecord`` with
     ``source_type=SourceType.WORLD_BANK_METADATA``,
     ``license_name="CC BY 4.0"`` (verify per
     https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets at
     access time -- do not hardcode without checking),
     ``extraction_method=ExtractionMethod.API_METADATA``.
  4. Use ``sourceNote`` text only to pick a topic/comparison angle; do not
     copy it verbatim into ``natural_language_instruction`` or any
     ``required_points``.
"""

from __future__ import annotations

import logging
from typing import List

from src.dataset.schemas import Seed, SourceType
from src.dataset.source_adapters.base import SourceAdapter

logger = logging.getLogger(__name__)


class WorldBankMetadataAdapter(SourceAdapter):
    source_type = SourceType.WORLD_BANK_METADATA.value

    def fetch_seeds(self, subdomain: str, limit: int) -> List[Seed]:
        logger.info(
            "WorldBankMetadataAdapter.fetch_seeds(subdomain=%r) is a stub (TODO v1.1); "
            "returning no seeds.",
            subdomain,
        )
        return []

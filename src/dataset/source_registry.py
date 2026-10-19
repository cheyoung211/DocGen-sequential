"""Registry of every ``SourceRecord`` used anywhere in the benchmark.

Enforces the conservative source policy from the dataset spec: an allowlist
of approved source domains and licenses, with a ``manual_review_required``
flag automatically set for anything outside that allowlist. Persists to
``data/metadata/source_registry.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from src.dataset.schemas import SourceRecord

logger = logging.getLogger(__name__)

# Tier 1/2 domains from the spec's source priority list. Anything not on
# this list (e.g. a commercial textbook publisher) is still recordable, but
# is force-flagged for manual review rather than silently trusted.
APPROVED_SOURCE_DOMAINS = {
    "openstax.org",
    "cnx.org",
    "data.nasa.gov",
    "data.gov",
    "earthdata.nasa.gov",
    "science.nasa.gov",
    "www.sec.gov",
    "sec.gov",
    "fred.stlouisfed.org",
    "api.stlouisfed.org",
    "data.worldbank.org",
    "api.worldbank.org",
    "arxiv.org",
    "export.arxiv.org",
}

# Licenses this pipeline is willing to trust without a manual review flag.
# "Standard curriculum" covers manually authored Tier-3 seeds that contain no
# reused copyrighted prose in the first place.
APPROVED_LICENSES = {
    "CC BY 4.0",
    "CC BY-SA 4.0",
    "CC0 1.0",
    "US Government Work",
    "Public Domain (US Government)",
    "Standard curriculum (no copyrighted prose reused)",
}


class SourceRegistry:
    """In-memory + on-disk store of ``SourceRecord`` entries, keyed by id."""

    def __init__(self) -> None:
        self._records: Dict[str, SourceRecord] = {}

    def register(self, record: SourceRecord) -> SourceRecord:
        """Add (or re-affirm) a source record, applying the allowlist.

        Registering the same ``source_id`` twice with identical content is
        allowed (many seeds legitimately share one source record); doing so
        with *different* content raises, since that would silently rewrite
        provenance history for items already generated against it.
        """
        normalized = self._apply_allowlist(record)
        existing = self._records.get(normalized.source_id)
        if existing is not None and existing != normalized:
            raise ValueError(
                f"source_id '{normalized.source_id}' is already registered "
                "with different content. Use a distinct source_id for a "
                "different edition/URL/license."
            )
        self._records[normalized.source_id] = normalized
        return normalized

    def _apply_allowlist(self, record: SourceRecord) -> SourceRecord:
        needs_review = record.manual_review_required
        if record.license_name not in APPROVED_LICENSES:
            needs_review = True
        if record.source_url:
            host = urlparse(record.source_url).netloc.lower()
            if host and host not in APPROVED_SOURCE_DOMAINS:
                needs_review = True
        if needs_review != record.manual_review_required:
            return record.model_copy(update={"manual_review_required": needs_review})
        return record

    def get(self, source_id: str) -> Optional[SourceRecord]:
        return self._records.get(source_id)

    def require(self, source_id: str) -> SourceRecord:
        record = self.get(source_id)
        if record is None:
            raise KeyError(
                f"Unknown source_id '{source_id}'. Register it in the "
                "SourceRegistry before referencing it from a Seed."
            )
        return record

    def all(self) -> List[SourceRecord]:
        return sorted(self._records.values(), key=lambda r: r.source_id)

    def manual_review_ids(self) -> List[str]:
        return [r.source_id for r in self.all() if r.manual_review_required]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [r.model_dump(mode="json") for r in self.all()]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote %d source records to %s", len(payload), path)

    @classmethod
    def load(cls, path: Path) -> "SourceRegistry":
        registry = cls()
        if not path.exists():
            logger.info("No existing source registry at %s; starting empty.", path)
            return registry
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload:
            registry.register(SourceRecord(**entry))
        return registry

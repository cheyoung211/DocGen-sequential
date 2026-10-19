"""Shared adapter interface and HTTP utilities for ``source_adapters/*``.

Every real adapter (``openstax``, ``nasa_metadata``, ``fred_metadata``)
subclasses :class:`SourceAdapter` and uses :class:`CachedHttpClient` for
retried, cached, rate-limited GET requests. Stub adapters (``arxiv_metadata``,
``sec_metadata``, ``world_bank_metadata``) subclass :class:`SourceAdapter`
too, but their ``fetch_seeds`` currently returns an empty list with a logged
TODO -- see each module's docstring for the intended endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.dataset.schemas import Seed

logger = logging.getLogger(__name__)


class SourceUnavailableError(RuntimeError):
    """Raised when a source cannot be reached after retries."""


class SourceAdapter(ABC):
    """Common interface for every Stage-1 seed source."""

    source_type: str  # matches schemas.SourceType value

    @abstractmethod
    def fetch_seeds(self, subdomain: str, limit: int) -> List[Seed]:
        """Return up to ``limit`` normalized :class:`Seed` records for a subdomain.

        Implementations must not raise on network/availability problems --
        catch them internally, log a warning, and return ``[]`` so the
        pipeline degrades gracefully. :func:`safe_fetch_seeds` below adds a
        second layer of protection for adapters that don't.
        """
        raise NotImplementedError


def safe_fetch_seeds(adapter: SourceAdapter, subdomain: str, limit: int) -> List[Seed]:
    """Call ``adapter.fetch_seeds`` and never let it take the build down."""
    try:
        return adapter.fetch_seeds(subdomain, limit)
    except Exception:  # noqa: BLE001 - intentional: a source's failure must not abort the build
        logger.warning(
            "Source adapter %s failed for subdomain=%r; continuing without it.",
            type(adapter).__name__,
            subdomain,
            exc_info=True,
        )
        return []


class CachedHttpClient:
    """Rate-limited, retried, disk-cached JSON GET client shared by real adapters."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        user_agent: str = "docgen-benchmark-builder/1.0",
        timeout_seconds: float = 15.0,
        max_retries: int = 4,
        backoff_seconds: float = 1.0,
        min_request_interval_seconds: float = 0.34,  # ~3 req/s ceiling
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout_seconds,
            follow_redirects=True,
        )
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._min_interval = min_request_interval_seconds
        self._last_request_ts = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CachedHttpClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _cache_path(self, url: str, params: Optional[Dict[str, Any]]) -> Path:
        key_material = url + "?" + json.dumps(params or {}, sort_keys=True)
        digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def get_json(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
    ) -> Any:
        """GET a JSON endpoint with disk caching plus retried, backed-off requests.

        Raises :class:`SourceUnavailableError` if all retries are exhausted;
        callers (adapters) are expected to catch this and degrade gracefully.
        """
        cache_path = self._cache_path(url, params)
        if use_cache and cache_path.exists():
            logger.debug("Cache hit for %s", url)
            return json.loads(cache_path.read_text(encoding="utf-8"))

        payload = self._get_json_with_retry(url, params=params, headers=headers)

        if use_cache:
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _get_json_with_retry(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
    ) -> Any:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=self._backoff_seconds, min=1.0, max=20.0),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        )
        def _do_request() -> Any:
            self._rate_limit()
            response = self._client.get(url, params=params, headers=headers)
            self._last_request_ts = time.monotonic()
            response.raise_for_status()
            return response.json()

        try:
            return _do_request()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            raise SourceUnavailableError(f"GET {url} failed after retries: {exc}") from exc

"""Integration tests for source_adapters/*, using mocked API responses
(no real network calls) -- these fixture payloads mirror the real response
shapes verified live against openstax.org and data.nasa.gov during
development (see the docstrings in openstax.py / nasa_metadata.py)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.source_adapters.arxiv_metadata import ArxivMetadataAdapter  # noqa: E402
from src.dataset.source_adapters.fred_metadata import FredMetadataAdapter  # noqa: E402
from src.dataset.source_adapters.manual_seed import ManualSeedAdapter, seed_bank_size  # noqa: E402
from src.dataset.source_adapters.nasa_metadata import NasaMetadataAdapter  # noqa: E402
from src.dataset.source_adapters.openstax import OpenStaxAdapter  # noqa: E402
from src.dataset.source_adapters.sec_metadata import SecMetadataAdapter  # noqa: E402
from src.dataset.source_adapters.world_bank_metadata import WorldBankMetadataAdapter  # noqa: E402
from src.dataset.source_registry import SourceRegistry  # noqa: E402

OPENSTAX_FIXTURE = {
    "meta": {"total_count": 1},
    "items": [
        {
            "id": 38,
            "meta": {
                "slug": "calculus-volume-1",
                "type": "books.Book",
                "html_url": "https://openstax.org/details/books/calculus-volume-1",
            },
            "title": "Calculus Volume 1",
        }
    ],
}

NASA_FIXTURE = {
    "result": {
        "results": [
            {
                "title": "Planetary Formation Survey Dataset",
                "name": "planetary-formation-survey-dataset",
                "license_title": "other-license-specified",
                "organization": {"title": "NASA"},
            }
        ]
    }
}

FRED_FIXTURE = {
    "seriess": [
        {"id": "GDPC1", "title": "Real Gross Domestic Product", "units": "Billions of Dollars", "frequency": "Quarterly"}
    ]
}


class _FakeHttpClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def get_json(self, url, *, params=None, headers=None, use_cache=True):
        self.calls.append((url, params))
        return self._payload


def test_openstax_adapter_parses_fixture_and_flags_manual_review():
    registry = SourceRegistry()
    http = _FakeHttpClient(OPENSTAX_FIXTURE)
    adapter = OpenStaxAdapter(http, registry)

    seeds = adapter.fetch_seeds("real_analysis", limit=5)

    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.topic == "Calculus Volume 1"
    assert seed.domain.value == "mathematics"
    record = registry.get(seed.source.source_id)
    assert record is not None
    assert record.copied_text_flag is False
    assert record.manual_review_required is True  # license unverified for this specific edition


def test_nasa_adapter_parses_fixture_and_flags_unspecified_license():
    registry = SourceRegistry()
    http = _FakeHttpClient(NASA_FIXTURE)
    adapter = NasaMetadataAdapter(http, registry)

    seeds = adapter.fetch_seeds("astronomy_and_planetary_science", limit=5)

    assert len(seeds) == 1
    assert seeds[0].domain.value == "scientific_engineering"
    record = registry.get(seeds[0].source.source_id)
    assert record.manual_review_required is True  # license_title contains "other"


def test_fred_adapter_skips_gracefully_without_api_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    registry = SourceRegistry()
    http = _FakeHttpClient(FRED_FIXTURE)
    adapter = FredMetadataAdapter(http, registry)

    seeds = adapter.fetch_seeds("macroeconomic_analysis", limit=5)

    assert seeds == []
    assert http.calls == []  # never attempted a request without a key


def test_fred_adapter_parses_fixture_when_key_present(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key-not-real")
    registry = SourceRegistry()
    http = _FakeHttpClient(FRED_FIXTURE)
    adapter = FredMetadataAdapter(http, registry)

    seeds = adapter.fetch_seeds("macroeconomic_analysis", limit=5)

    assert len(seeds) == 1
    assert seeds[0].topic == "Real Gross Domestic Product"
    assert seeds[0].domain.value == "financial_economic"


def test_stub_adapters_return_no_seeds():
    assert ArxivMetadataAdapter().fetch_seeds("real_analysis", 5) == []
    assert SecMetadataAdapter().fetch_seeds("risk_analysis", 5) == []
    assert WorldBankMetadataAdapter().fetch_seeds("macroeconomic_analysis", 5) == []


def test_manual_seed_adapter_covers_all_domains_with_30_each():
    registry = SourceRegistry()
    adapter = ManualSeedAdapter(registry)
    for domain in ["mathematics", "statistics_ml", "scientific_engineering", "financial_economic"]:
        assert seed_bank_size(domain) == 30
        seeds = adapter.fetch_seeds_for_domain(domain, 100)
        assert len(seeds) == 30
        assert len({s.seed_id for s in seeds}) == 30

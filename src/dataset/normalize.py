"""Stage 1: gather ``Seed`` records from configured source adapters and
persist them to ``data/normalized_seeds/<domain>.jsonl``.

By default (``network_sources_enabled: false`` in ``configs/dataset_v1.yaml``)
only the manual seed bank runs, which is what produces the committed
``benchmark_v1.jsonl``. When ``network_sources_enabled: true``, the real
OpenStax/NASA/FRED adapters are also queried (best-effort, per subdomain) to
top up each domain's seed pool; the stub adapters (arXiv/SEC/World Bank)
return no seeds regardless (see their modules).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.dataset.schemas import Seed
from src.dataset.source_adapters.base import CachedHttpClient, safe_fetch_seeds
from src.dataset.source_adapters.arxiv_metadata import ArxivMetadataAdapter
from src.dataset.source_adapters.fred_metadata import FredMetadataAdapter
from src.dataset.source_adapters.manual_seed import ManualSeedAdapter
from src.dataset.source_adapters.nasa_metadata import NasaMetadataAdapter
from src.dataset.source_adapters.openstax import OpenStaxAdapter
from src.dataset.source_adapters.sec_metadata import SecMetadataAdapter
from src.dataset.source_adapters.world_bank_metadata import WorldBankMetadataAdapter
from src.dataset.source_registry import SourceRegistry

logger = logging.getLogger(__name__)

SUBDOMAINS_BY_DOMAIN: Dict[str, List[str]] = {
    "mathematics": [
        "real_analysis", "linear_algebra", "probability_theory",
        "discrete_mathematics", "calculus", "numerical_analysis",
    ],
    "statistics_ml": [
        "descriptive_statistics", "probability_distributions", "estimation",
        "hypothesis_testing", "regression", "bias_variance_tradeoff",
        "bayesian_inference", "model_evaluation", "experimental_design",
    ],
    "scientific_engineering": [
        "astronomy_and_planetary_science", "climate_and_earth_science", "physics",
        "computer_science", "electrical_or_mechanical_engineering",
        "environmental_science", "scientific_methodology",
    ],
    "financial_economic": [
        "corporate_financial_analysis", "financial_statement_interpretation",
        "risk_analysis", "macroeconomic_analysis", "monetary_policy",
        "market_structure", "valuation_concepts", "economic_indicator_comparison",
    ],
}

_NETWORK_ADAPTER_FACTORIES = {
    "openstax": lambda http, registry: OpenStaxAdapter(http, registry),
    "nasa_metadata": lambda http, registry: NasaMetadataAdapter(http, registry),
    "fred_metadata": lambda http, registry: FredMetadataAdapter(http, registry),
    "arxiv_metadata": lambda http, registry: ArxivMetadataAdapter(),
    "sec_metadata": lambda http, registry: SecMetadataAdapter(),
    "world_bank_metadata": lambda http, registry: WorldBankMetadataAdapter(),
}


def collect_seeds_for_domain(
    domain: str,
    *,
    enabled_sources: List[str],
    target_count: int,
    manual_adapter: ManualSeedAdapter,
    registry: SourceRegistry,
    http_client: Optional[CachedHttpClient],
    network_sources_enabled: bool,
) -> List[Seed]:
    """Collect up to ``target_count`` seeds for a domain from its enabled sources."""
    seeds: List[Seed] = []
    seen_ids = set()

    def _add(new_seeds: List[Seed]) -> None:
        for seed in new_seeds:
            if seed.seed_id in seen_ids:
                continue
            seen_ids.add(seed.seed_id)
            seeds.append(seed)

    if "manual_seed" in enabled_sources:
        _add(manual_adapter.fetch_seeds_for_domain(domain, target_count))
    else:
        logger.warning(
            "manual_seed is not enabled for domain %r; this domain will rely "
            "entirely on network sources, which is not the v1 default.",
            domain,
        )

    if network_sources_enabled and http_client is not None:
        for source_name in enabled_sources:
            if source_name == "manual_seed" or len(seeds) >= target_count:
                continue
            factory = _NETWORK_ADAPTER_FACTORIES.get(source_name)
            if factory is None:
                logger.warning("Unknown source %r for domain %r; skipping.", source_name, domain)
                continue
            adapter = factory(http_client, registry)
            for subdomain in SUBDOMAINS_BY_DOMAIN.get(domain, []):
                remaining = target_count - len(seeds)
                if remaining <= 0:
                    break
                _add(safe_fetch_seeds(adapter, subdomain, remaining))

    if len(seeds) < target_count:
        logger.info(
            "Collected only %d/%d seeds for domain %r from sources %s.",
            len(seeds), target_count, domain, enabled_sources,
        )

    return seeds[:target_count]


def write_seeds_jsonl(seeds: List[Seed], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for seed in seeds:
            f.write(seed.model_dump_json() + "\n")
    logger.info("Wrote %d seeds to %s", len(seeds), path)


def load_seeds_jsonl(path: Path) -> List[Seed]:
    if not path.exists():
        return []
    seeds = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(Seed(**json.loads(line)))
    return seeds

"""CLI entry point for building the LaTeX-focused evaluation benchmark dataset.

This dataset is for test-time / inference-time evaluation of the document
generation system only -- it is never used for model training, so no
train/validation/test split is produced (spec section 10).

Usage:
    python -m src.dataset.build_dataset --config configs/dataset_v1.yaml --output-dir data --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.balance import (  # noqa: E402
    check_difficulty_balance,
    enforce_domain_count,
    select_dev_debug_subset,
)
from src.dataset.instruction_builder import build_items_for_domain  # noqa: E402
from src.dataset.normalize import collect_seeds_for_domain, write_seeds_jsonl  # noqa: E402
from src.dataset.schemas import BenchmarkItem  # noqa: E402
from src.dataset.source_adapters.base import CachedHttpClient  # noqa: E402
from src.dataset.source_adapters.manual_seed import ManualSeedAdapter  # noqa: E402
from src.dataset.source_registry import SourceRegistry  # noqa: E402
from src.dataset.statistics import compute_dataset_statistics, compute_license_report, write_json  # noqa: E402
from src.dataset.validators import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    MIN_WORDS_FOR_LONG_FORM,
    validate_items,
)

logger = logging.getLogger(__name__)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_jsonl(items: List[BenchmarkItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json_line() + "\n")
    logger.info("Wrote %d items to %s", len(items), path)


def write_json_array(items: List[BenchmarkItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [json.loads(item.model_dump_json_line()) for item in items]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d items to %s", len(items), path)


def write_rejected_jsonl(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rejected records to %s", len(records), path)


def build_dataset(config: Dict[str, Any], output_dir: Path, seed: int) -> Dict[str, Any]:
    """Run the full Stage 1 + Stage 2 + validation + balancing pipeline.

    Returns a small summary dict (mainly for tests); all real outputs are
    written to disk under ``output_dir`` per spec section 10.
    """
    rng = random.Random(seed)

    output_cfg = config.get("output", {})
    benchmark_dir = output_dir / output_cfg.get("benchmark_dirname", "benchmark")
    seeds_dir = output_dir / output_cfg.get("seeds_dirname", "normalized_seeds")
    metadata_dir = output_dir / output_cfg.get("metadata_dirname", "metadata")
    raw_sources_dir = output_dir / output_cfg.get("raw_sources_dirname", "raw_sources")
    splits_dir = benchmark_dir / "splits"

    registry = SourceRegistry()
    manual_adapter = ManualSeedAdapter(registry)

    network_sources_enabled = bool(config.get("network_sources_enabled", False))
    http_cfg = config.get("http", {})
    http_client = None
    if network_sources_enabled:
        http_client = CachedHttpClient(
            cache_dir=raw_sources_dir,
            user_agent=http_cfg.get("user_agent", "docgen-benchmark-builder/1.0"),
            timeout_seconds=float(http_cfg.get("timeout_seconds", 15.0)),
            max_retries=int(http_cfg.get("max_retries", 4)),
            backoff_seconds=float(http_cfg.get("backoff_seconds", 1.0)),
        )

    thresholds = {**DEFAULT_THRESHOLDS, **config.get("quality_thresholds", {})}
    difficulty_ratios = config.get("difficulty_distribution", {})

    # Verifier-evaluation project phase: a config can opt out of the
    # long-form contract (see instruction_builder.build_benchmark_item and
    # validators._MIN_WORDS_FOR_LONG_FORM). Both default to the original
    # long-form behavior, so an existing config with neither key (e.g.
    # dataset_v1.yaml) builds byte-for-byte the same dataset as before.
    long_form = bool(config.get("long_form", True))
    benchmark_version = str(config.get("benchmark_version", "v1"))
    min_words_threshold = int(config.get("long_form_word_threshold", MIN_WORDS_FOR_LONG_FORM))

    all_accepted: List[BenchmarkItem] = []
    all_rejected: List[dict] = []
    items_by_domain: Dict[str, List[BenchmarkItem]] = {}

    try:
        for domain, domain_cfg in config.get("domains", {}).items():
            target_count = int(domain_cfg.get("count", 0))
            enabled_sources = list(domain_cfg.get("enabled_sources", []))

            seeds = collect_seeds_for_domain(
                domain,
                enabled_sources=enabled_sources,
                target_count=target_count,
                manual_adapter=manual_adapter,
                registry=registry,
                http_client=http_client,
                network_sources_enabled=network_sources_enabled,
            )
            write_seeds_jsonl(seeds, seeds_dir / f"{domain}.jsonl")

            candidate_items = build_items_for_domain(
                seeds, long_form=long_form, benchmark_version=benchmark_version
            )
            candidate_items = enforce_domain_count(candidate_items, target_count, rng=rng)

            accepted, rejected = validate_items(
                candidate_items, thresholds=thresholds, min_words_threshold=min_words_threshold
            )
            for entry in rejected:
                entry["domain"] = domain
            all_rejected.extend(rejected)

            items_by_domain[domain] = accepted
            all_accepted.extend(accepted)

            logger.info(
                "Domain %s: %d/%d accepted (%d rejected, %d seeds collected).",
                domain, len(accepted), target_count, len(rejected), len(seeds),
            )
            for warning in check_difficulty_balance(accepted, difficulty_ratios):
                logger.warning("Domain %s: %s", domain, warning)

            write_jsonl(accepted, splits_dir / f"{domain}.jsonl")
    finally:
        if http_client is not None:
            http_client.close()

    write_jsonl(all_accepted, benchmark_dir / f"benchmark_{benchmark_version}.jsonl")
    write_json_array(all_accepted, benchmark_dir / f"benchmark_{benchmark_version}.json")

    dev_debug_size = int(config.get("dev_debug_size", 12))
    dev_debug_items = select_dev_debug_subset(items_by_domain, dev_debug_size, rng=rng)
    write_jsonl(dev_debug_items, benchmark_dir / "dev_debug.jsonl")

    registry.save(metadata_dir / "source_registry.json")
    write_json(compute_license_report(registry), metadata_dir / "license_report.json")
    write_json(compute_dataset_statistics(all_accepted), metadata_dir / "dataset_statistics.json")
    write_rejected_jsonl(all_rejected, metadata_dir / "rejected_items.jsonl")

    logger.info(
        "Build complete: %d accepted items across %d domains, %d rejected.",
        len(all_accepted), len(items_by_domain), len(all_rejected),
    )

    return {
        "total_accepted": len(all_accepted),
        "total_rejected": len(all_rejected),
        "by_domain": {domain: len(items) for domain, items in items_by_domain.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the LaTeX-focused evaluation benchmark dataset.")
    parser.add_argument("--config", required=True, help="Path to a dataset config YAML, e.g. configs/dataset_v1.yaml")
    parser.add_argument("--output-dir", required=True, help="Output directory, e.g. data")
    parser.add_argument("--seed", type=int, default=None, help="Override the config file's random seed.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config(Path(args.config))
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))

    build_dataset(config, Path(args.output_dir), seed)


if __name__ == "__main__":
    main()

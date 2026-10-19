"""End-to-end smoke test for src/dataset/build_dataset.py, using a small
in-memory config so the test runs fast and never touches the network."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.build_dataset import build_dataset  # noqa: E402

_SMALL_CONFIG = {
    "benchmark_version": "v1",
    "seed": 42,
    "network_sources_enabled": False,
    "dev_debug_size": 4,
    "domains": {
        "mathematics": {"count": 5, "enabled_sources": ["manual_seed"]},
        "statistics_ml": {"count": 5, "enabled_sources": ["manual_seed"]},
        "scientific_engineering": {"count": 5, "enabled_sources": ["manual_seed"]},
        "financial_economic": {"count": 5, "enabled_sources": ["manual_seed"]},
    },
    "difficulty_distribution": {"basic": 0.3333, "intermediate": 0.3333, "advanced": 0.3334},
    "quality_thresholds": {
        "specificity": 0.8, "evaluability": 0.85, "latex_relevance": 0.8,
        "long_form_suitability": 0.8, "constraint_clarity": 0.85, "source_safety": 1.0,
    },
    "output": {
        "benchmark_dirname": "benchmark",
        "seeds_dirname": "normalized_seeds",
        "metadata_dirname": "metadata",
        "raw_sources_dirname": "raw_sources",
    },
}


def test_build_dataset_end_to_end(tmp_path):
    summary = build_dataset(_SMALL_CONFIG, tmp_path, seed=42)

    assert summary["total_accepted"] == 20
    assert summary["total_rejected"] == 0
    assert summary["by_domain"] == {
        "mathematics": 5, "statistics_ml": 5, "scientific_engineering": 5, "financial_economic": 5,
    }

    benchmark_path = tmp_path / "benchmark" / "benchmark_v1.jsonl"
    assert benchmark_path.exists()
    lines = benchmark_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 20

    items = [json.loads(line) for line in lines]
    sample_ids = [item["sample_id"] for item in items]
    assert len(sample_ids) == len(set(sample_ids))
    assert all(item["quality_score"] is not None for item in items)

    assert (tmp_path / "benchmark" / "benchmark_v1.json").exists()
    assert (tmp_path / "benchmark" / "dev_debug.jsonl").exists()
    dev_debug_lines = (tmp_path / "benchmark" / "dev_debug.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert 1 <= len(dev_debug_lines) <= 4

    for domain in _SMALL_CONFIG["domains"]:
        split_path = tmp_path / "benchmark" / "splits" / f"{domain}.jsonl"
        assert split_path.exists()
        assert len(split_path.read_text(encoding="utf-8").strip().splitlines()) == 5

    assert (tmp_path / "metadata" / "source_registry.json").exists()
    assert (tmp_path / "metadata" / "license_report.json").exists()
    assert (tmp_path / "metadata" / "dataset_statistics.json").exists()

    stats = json.loads((tmp_path / "metadata" / "dataset_statistics.json").read_text(encoding="utf-8"))
    assert stats["total_items"] == 20

    rejected_path = tmp_path / "metadata" / "rejected_items.jsonl"
    assert rejected_path.exists()
    assert rejected_path.read_text(encoding="utf-8").strip() == ""


def test_build_dataset_is_deterministic(tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    build_dataset(_SMALL_CONFIG, out1, seed=42)
    build_dataset(_SMALL_CONFIG, out2, seed=42)

    content1 = (out1 / "benchmark" / "benchmark_v1.jsonl").read_text(encoding="utf-8")
    content2 = (out2 / "benchmark" / "benchmark_v1.jsonl").read_text(encoding="utf-8")
    assert content1 == content2

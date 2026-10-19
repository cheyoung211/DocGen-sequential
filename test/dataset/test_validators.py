"""Unit tests for src/dataset/validators.py."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.instruction_builder import build_benchmark_item  # noqa: E402
from src.dataset.source_adapters.manual_seed import ManualSeedAdapter  # noqa: E402
from src.dataset.source_registry import SourceRegistry  # noqa: E402
from src.dataset.validators import DEFAULT_THRESHOLDS, validate_item, validate_items  # noqa: E402


def _sample_item(sample_index: int = 1):
    registry = SourceRegistry()
    adapter = ManualSeedAdapter(registry)
    seed = adapter.fetch_seeds_for_domain("mathematics", 1)[0]
    return build_benchmark_item(seed, sample_index)


def test_generated_item_passes_validation():
    item = _sample_item()
    result = validate_item(item, seen_sample_ids=set(), thresholds=DEFAULT_THRESHOLDS)
    assert result.accepted, [i.message for i in result.issues]
    assert result.quality_score.source_safety == 1.0


def test_duplicate_sample_id_is_rejected():
    item = _sample_item()
    result = validate_item(item, seen_sample_ids={item.sample_id}, thresholds=DEFAULT_THRESHOLDS)
    assert not result.accepted
    assert any(i.rule_id == "unique_sample_id" for i in result.issues)


def test_bad_component_placement_is_rejected():
    item = _sample_item()
    broken = item.required_components[0].model_copy(update={"placement": "not_a_real_section"})
    item = item.model_copy(update={"required_components": [broken, *item.required_components[1:]]})
    result = validate_item(item, seen_sample_ids=set(), thresholds=DEFAULT_THRESHOLDS)
    assert not result.accepted
    assert any(i.rule_id == "placement_id_exists" for i in result.issues)


def test_bad_dependency_target_is_rejected():
    item = _sample_item()
    broken = item.cross_section_dependencies[0].model_copy(update={"target_sections": ["nonexistent_section"]})
    item = item.model_copy(update={"cross_section_dependencies": [broken, *item.cross_section_dependencies[1:]]})
    result = validate_item(item, seen_sample_ids=set(), thresholds=DEFAULT_THRESHOLDS)
    assert not result.accepted
    assert any(i.rule_id == "dependency_target_ids_exist" for i in result.issues)


def test_vague_short_item_is_rejected():
    item = _sample_item()
    item = item.model_copy(update={
        "required_sections": item.required_sections[:2],
        "required_components": item.required_components[:1],
        "target_length": item.target_length.model_copy(update={"min_words": 100}),
    })
    result = validate_item(item, seen_sample_ids=set(), thresholds=DEFAULT_THRESHOLDS)
    assert not result.accepted
    assert any(i.rule_id == "not_vague" for i in result.issues)


def test_instruction_missing_constraint_text_is_rejected():
    item = _sample_item()
    item = item.model_copy(update={"natural_language_instruction": "A generic instruction with nothing specific."})
    result = validate_item(item, seen_sample_ids=set(), thresholds=DEFAULT_THRESHOLDS)
    assert not result.accepted
    rule_ids = {i.rule_id for i in result.issues}
    assert "instruction_covers_sections" in rule_ids or "instruction_covers_constraints" in rule_ids


def test_validate_items_batch_attaches_quality_score():
    registry = SourceRegistry()
    adapter = ManualSeedAdapter(registry)
    seeds = adapter.fetch_seeds_for_domain("mathematics", 3)
    items = [build_benchmark_item(seed, i) for i, seed in enumerate(seeds, start=1)]

    accepted, rejected = validate_items(items)
    assert len(accepted) == 3
    assert rejected == []
    assert all(item.quality_score is not None for item in accepted)

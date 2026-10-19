"""Unit tests for src/dataset/instruction_builder.py.

Checks the spec's section-8 structural rules (section/component counts by
difficulty) and section-13 hard rules (>=2 LaTeX-native components, >=1
cross-section dependency) hold for every domain x difficulty combination in
the real manual seed bank, not just a hand-built fixture.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.instruction_builder import build_benchmark_item  # noqa: E402
from src.dataset.source_adapters.manual_seed import ManualSeedAdapter  # noqa: E402
from src.dataset.source_registry import SourceRegistry  # noqa: E402

_NATIVE_LATEX_TYPES = {
    "definition", "theorem", "lemma", "proposition", "corollary", "proof",
    "proof_sketch", "equation", "aligned_equations", "table", "notation_table",
    "comparison_table", "terminology_table", "classification_table", "boxed_result",
}

_EXPECTED_SECTION_COUNT = {"basic": 5, "intermediate": 8, "advanced": 10}
_EXPECTED_COMPONENT_COUNT = {"basic": 2, "intermediate": 4, "advanced": 6}


def _all_seeds():
    registry = SourceRegistry()
    adapter = ManualSeedAdapter(registry)
    seeds = []
    for domain in ["mathematics", "statistics_ml", "scientific_engineering", "financial_economic"]:
        seeds.extend(adapter.fetch_seeds_for_domain(domain, 100))
    return seeds


_SEEDS = _all_seeds()


@pytest.mark.parametrize("seed", _SEEDS, ids=lambda s: s.seed_id)
def test_section_and_component_counts_match_difficulty(seed):
    item = build_benchmark_item(seed, 1)
    difficulty = seed.difficulty_hint.value if hasattr(seed.difficulty_hint, "value") else seed.difficulty_hint
    assert len(item.required_sections) == _EXPECTED_SECTION_COUNT[difficulty]
    assert len(item.required_components) == _EXPECTED_COMPONENT_COUNT[difficulty]


@pytest.mark.parametrize("seed", _SEEDS, ids=lambda s: s.seed_id)
def test_item_satisfies_section13_hard_rules(seed):
    item = build_benchmark_item(seed, 1)
    # Rule 11: every task must contain at least one document-level dependency.
    assert len(item.cross_section_dependencies) >= 1
    # Rule 12: every task must contain at least two LaTeX-native components.
    present_types = {c.type.value for c in item.required_components}
    assert len(present_types & _NATIVE_LATEX_TYPES) >= 2
    # Sections/components/dependencies must reference only ids that exist.
    section_ids = set(item.section_ids)
    component_ids = set(item.component_ids)
    for comp in item.required_components:
        assert comp.placement in section_ids
        assert all(dep in component_ids for dep in comp.depends_on)
    for dep in item.cross_section_dependencies:
        assert dep.source_section in section_ids
        assert all(t in section_ids for t in dep.target_sections)


@pytest.mark.parametrize("seed", _SEEDS, ids=lambda s: s.seed_id)
def test_instruction_covers_structured_fields(seed):
    item = build_benchmark_item(seed, 1)
    instruction_lower = item.natural_language_instruction.lower()
    for section in item.required_sections:
        assert section.title.lower() in instruction_lower
    for hc in item.hard_constraints:
        assert hc.description.lower() in instruction_lower


def test_sample_id_format():
    item = build_benchmark_item(_SEEDS[0], 7)
    assert item.sample_id.endswith("_0007")
    assert item.sample_id.startswith("math_")


def test_advanced_items_include_limitations_and_extra_dependency():
    advanced_seed = next(s for s in _SEEDS if s.difficulty_hint.value == "advanced")
    item = build_benchmark_item(advanced_seed, 1)
    assert any("limitation" in s.title.lower() for s in item.required_sections)
    assert len(item.cross_section_dependencies) == 3

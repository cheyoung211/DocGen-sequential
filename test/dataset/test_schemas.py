"""Unit tests for src/dataset/schemas.py."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.schemas import (  # noqa: E402
    BenchmarkItem,
    CrossSectionDependency,
    EvaluationContract,
    HardConstraint,
    LayoutRequirements,
    RequiredComponent,
    RequiredSection,
    Seed,
    SeedSourceRef,
    SourceProvenanceEntry,
    SourceRecord,
    TargetLength,
)


def _minimal_item_kwargs():
    return dict(
        sample_id="math_real_analysis_0001",
        domain="mathematics",
        subdomain="real_analysis",
        document_type="lecture_note",
        difficulty="basic",
        title="Test Topic",
        overview="An overview.",
        audience="advanced undergraduate students",
        document_purpose="Explain the topic.",
        target_length=TargetLength(target_pages=4, min_words=1000, max_words=2000, min_sections=4, max_sections=6),
        required_sections=[
            RequiredSection(section_id="introduction", title="Introduction", purpose="Motivate.", target_words=200),
            RequiredSection(section_id="formal_definition", title="Formal Definition", purpose="Define.", target_words=200),
        ],
        required_components=[
            RequiredComponent(component_id="definition_core", type="definition", placement="formal_definition", description="Define the topic."),
            RequiredComponent(component_id="notation_table", type="notation_table", placement="introduction", description="List symbols."),
        ],
        hard_constraints=[
            HardConstraint(constraint_id="terminology_consistency", type="terminology_consistency", description="Use the term consistently.", applicable_to=["all_sections"]),
        ],
        cross_section_dependencies=[
            CrossSectionDependency(dependency_id="reuse_def", source_section="formal_definition", target_sections=["introduction"], relation="must_reuse", description="Reuse the definition."),
        ],
        layout_requirements=LayoutRequirements(preferred_layout_profile="technical_lecture_note"),
        evaluation_contract=EvaluationContract(),
        natural_language_instruction="Write a document about the topic, including Introduction and Formal Definition.",
        source_provenance=[SourceProvenanceEntry(source_id="curriculum_mathematics", usage="topic seed", copied_text=False)],
    )


def test_benchmark_item_valid_construction():
    item = BenchmarkItem(**_minimal_item_kwargs())
    assert item.sample_id == "math_real_analysis_0001"
    assert item.section_ids == ["introduction", "formal_definition"]
    assert item.component_ids == ["definition_core", "notation_table"]


def test_benchmark_item_rejects_bad_section_id():
    with pytest.raises(ValidationError):
        RequiredSection(section_id="Bad-ID!", title="Introduction", purpose="Motivate.", target_words=200)


def test_benchmark_item_rejects_empty_required_sections():
    kwargs = _minimal_item_kwargs()
    kwargs["required_sections"] = []
    with pytest.raises(ValidationError):
        BenchmarkItem(**kwargs)


def test_benchmark_item_rejects_copied_text_provenance():
    with pytest.raises(ValidationError):
        SourceProvenanceEntry(source_id="x", usage="topic seed", copied_text=True)


def test_target_length_rejects_inverted_word_bounds():
    with pytest.raises(ValidationError):
        TargetLength(target_pages=4, min_words=3000, max_words=2000, min_sections=4, max_sections=6)


def test_source_record_rejects_copied_text_flag_true():
    with pytest.raises(ValidationError):
        SourceRecord(
            source_id="bad_source",
            source_type="manual_curation",
            source_title="Some Book",
            publisher_or_owner="Someone",
            access_date="2026-01-01",
            license_name="Standard curriculum (no copyrighted prose reused)",
            allowed_use="topic seed",
            attribution_text="attribution",
            extraction_method="manual",
            copied_text_flag=True,
        )


def test_seed_requires_non_empty_objectives_and_concepts():
    with pytest.raises(ValidationError):
        Seed(
            seed_id="seed_math_real_analysis_001",
            domain="mathematics",
            subdomain="real_analysis",
            source=SeedSourceRef(source_id="curriculum_mathematics", source_title="x", license_name="Standard curriculum (no copyrighted prose reused)", access_date="2026-01-01"),
            topic="Topic",
            subtopic="subtopic",
            learning_objectives=[],
            key_concepts=["a"],
            difficulty_hint="basic",
            document_type_hint="lecture_note",
        )


def test_export_json_schema_is_a_dict():
    from src.dataset.schemas import export_benchmark_item_json_schema

    schema = export_benchmark_item_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema

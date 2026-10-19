"""Pydantic schemas for the LaTeX-focused, evaluation-only benchmark dataset.

This dataset is used for test-time / inference-time evaluation of the
document-generation system. It is never used for model training, so no
"label" or "gold document" field exists here -- the gold annotation is the
structured document contract (``BenchmarkItem``) itself.

Two record types are defined:

- ``SourceRecord`` / ``Seed``: Stage 1 (seed extraction) records, one per
  topic pulled from a source adapter, tracked in the source registry and
  ``data/normalized_seeds/*.jsonl``.
- ``BenchmarkItem``: the Stage 2 (instruction generation) output, one line
  of ``data/benchmark/benchmark_v1.jsonl``.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Identifiers are written into JSONL/JSON output and cross-referenced by id
# between required_sections, required_components, and cross_section_dependencies,
# so they are constrained the same way src.common.state.SAFE_IDENTIFIER
# constrains node ids: lowercase snake_case, safe to use as a dict key or a
# LaTeX \label suffix.
SAFE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_safe_id(value: str, *, field_name: str = "id") -> str:
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} '{value}' must be lowercase snake_case "
            "(letters, digits, underscores; starting with a letter)."
        )
    return value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Domain(str, Enum):
    MATHEMATICS = "mathematics"
    STATISTICS_ML = "statistics_ml"
    SCIENTIFIC_ENGINEERING = "scientific_engineering"
    FINANCIAL_ECONOMIC = "financial_economic"


class Difficulty(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXTREME = "extreme"


class DocumentType(str, Enum):
    # Mathematics
    LECTURE_NOTE = "lecture_note"
    THEOREM_ORIENTED_ARTICLE = "theorem_oriented_explanatory_article"
    CONCEPT_TUTORIAL = "concept_tutorial"
    PROOF_ORIENTED_TECHNICAL_NOTE = "proof_oriented_technical_note"
    CHAPTER_STYLE_EDUCATIONAL_DOCUMENT = "chapter_style_educational_document"
    # Statistics / ML
    STATISTICAL_CONCEPT_REPORT = "statistical_concept_report"
    METHOD_COMPARISON_ARTICLE = "method_comparison_article"
    DERIVATION_ORIENTED_TECHNICAL_NOTE = "derivation_oriented_technical_note"
    MODEL_EVALUATION_REPORT = "model_evaluation_report"
    # Scientific / Engineering
    TECHNICAL_REPORT = "technical_report"
    SCIENTIFIC_ARTICLE_STYLE = "scientific_article_style_document"
    EXPLANATORY_RESEARCH_NOTE = "explanatory_research_note"
    METHOD_SURVEY = "method_survey"
    SYSTEM_DESIGN_REPORT = "system_design_report"
    # Financial / Economic
    FINANCIAL_ANALYSIS_REPORT = "financial_analysis_report"
    ECONOMIC_POLICY_REPORT = "economic_policy_report"
    RISK_REPORT = "risk_report"
    COMPARATIVE_MARKET_REPORT = "comparative_market_report"
    EDUCATIONAL_FINANCIAL_NOTE = "educational_financial_note"


class ComponentType(str, Enum):
    DEFINITION = "definition"
    THEOREM = "theorem"
    LEMMA = "lemma"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"
    PROOF = "proof"
    PROOF_SKETCH = "proof_sketch"
    EQUATION = "equation"
    ALIGNED_EQUATIONS = "aligned_equations"
    TABLE = "table"
    NOTATION_TABLE = "notation_table"
    COMPARISON_TABLE = "comparison_table"
    TERMINOLOGY_TABLE = "terminology_table"
    CLASSIFICATION_TABLE = "classification_table"
    WORKED_EXAMPLE = "worked_example"
    CALLOUT = "callout"
    RISK_CALLOUT = "risk_callout"
    BOXED_RESULT = "boxed_result"
    FIGURE_PLACEHOLDER = "figure_placeholder"
    TIMELINE_PLACEHOLDER = "timeline_placeholder"
    EXERCISE = "exercise"
    SUMMARY_BOX = "summary_box"


class ConstraintType(str, Enum):
    NOTATION_CONSISTENCY = "notation_consistency"
    TERMINOLOGY_CONSISTENCY = "terminology_consistency"
    STRUCTURAL_ORDERING = "structural_ordering"
    CROSS_REFERENCE = "cross_reference"
    HYPOTHETICAL_LABELING = "hypothetical_labeling"
    CATEGORY_CONSISTENCY = "category_consistency"
    OTHER = "other"


class DependencyRelation(str, Enum):
    MUST_REUSE = "must_reuse"
    MUST_REFERENCE = "must_reference"
    MUST_BE_CONSISTENT_WITH = "must_be_consistent_with"
    MUST_PRECEDE = "must_precede"


class SourceType(str, Enum):
    OPENSTAX = "openstax"
    ARXIV_METADATA = "arxiv_metadata"
    NASA_METADATA = "nasa_metadata"
    SEC_METADATA = "sec_metadata"
    FRED_METADATA = "fred_metadata"
    WORLD_BANK_METADATA = "world_bank_metadata"
    MANUAL_CURATION = "manual_curation"
    STANDARD_CURRICULUM = "standard_curriculum"


class ExtractionMethod(str, Enum):
    MANUAL = "manual"
    API_METADATA = "api_metadata"
    STRUCTURAL_ONLY = "structural_only"


# ---------------------------------------------------------------------------
# Source policy records (spec section 3)
# ---------------------------------------------------------------------------


class SourceRecord(BaseModel):
    """One entry in the source registry; never contains copied source prose."""

    source_id: str
    source_type: SourceType
    source_title: str
    source_url: Optional[str] = None
    publisher_or_owner: str
    edition_or_version: Optional[str] = None
    access_date: str
    license_name: str
    license_url: Optional[str] = None
    allowed_use: str
    attribution_text: str
    extraction_method: ExtractionMethod
    copied_text_flag: bool = False
    manual_review_required: bool = False
    notes: List[str] = Field(default_factory=list)

    @field_validator("source_id")
    @classmethod
    def _safe_source_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="source_id")

    @field_validator("access_date")
    @classmethod
    def _valid_access_date(cls, value: str) -> str:
        date.fromisoformat(value)  # raises ValueError if malformed
        return value

    @model_validator(mode="after")
    def _copied_text_requires_no_manual_review_conflict(self) -> "SourceRecord":
        if self.copied_text_flag:
            raise ValueError(
                f"source '{self.source_id}': copied_text_flag must be False. "
                "This pipeline never stores copied source prose; only "
                "metadata, headings, and curricular structure are permitted."
            )
        return self


class SeedSourceRef(BaseModel):
    """Slim reference into the source registry, embedded in a Seed."""

    source_id: str
    source_title: str
    source_url: Optional[str] = None
    license_name: str
    access_date: str


class Seed(BaseModel):
    """Stage 1 normalized seed record (spec section 4).

    Extends the literal spec schema with two additive fields --
    ``difficulty_hint`` and ``document_type_hint`` -- because Stage 2 in this
    implementation is a deterministic template expansion rather than an LLM
    call, so it needs an explicit difficulty/document-type target instead of
    inferring one from prose.
    """

    seed_id: str
    domain: Domain
    subdomain: str
    source: SeedSourceRef
    topic: str
    subtopic: str
    source_headings: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    candidate_notation: List[str] = Field(default_factory=list)
    candidate_equations: List[str] = Field(default_factory=list)
    candidate_components: List[str] = Field(default_factory=list)
    source_notes: List[str] = Field(default_factory=list)
    difficulty_hint: Difficulty
    document_type_hint: DocumentType

    @field_validator("seed_id")
    @classmethod
    def _safe_seed_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="seed_id")

    @field_validator("learning_objectives", "key_concepts")
    @classmethod
    def _non_empty(cls, value: List[str], info) -> List[str]:
        if not value:
            raise ValueError(f"{info.field_name} must not be empty.")
        return value


# ---------------------------------------------------------------------------
# Benchmark item sub-models (spec section 5)
# ---------------------------------------------------------------------------


class TargetLength(BaseModel):
    target_pages: int = Field(gt=0)
    min_words: int = Field(gt=0)
    max_words: int = Field(gt=0)
    min_sections: int = Field(gt=0)
    max_sections: int = Field(gt=0)

    @model_validator(mode="after")
    def _bounds_consistent(self) -> "TargetLength":
        if self.min_words >= self.max_words:
            raise ValueError("min_words must be < max_words.")
        if self.min_sections > self.max_sections:
            raise ValueError("min_sections must be <= max_sections.")
        return self


class RequiredSection(BaseModel):
    section_id: str
    title: str
    purpose: str
    required_points: List[str] = Field(default_factory=list)
    target_words: int = Field(gt=0)

    @field_validator("section_id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="section_id")


class RequiredComponent(BaseModel):
    component_id: str
    type: ComponentType
    required: bool = True
    placement: str
    description: str
    reference_required: bool = False
    depends_on: List[str] = Field(default_factory=list)

    @field_validator("component_id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="component_id")


class HardConstraint(BaseModel):
    constraint_id: str
    type: ConstraintType
    description: str
    severity: str = "hard"
    applicable_to: List[str] = Field(default_factory=list)

    @field_validator("constraint_id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="constraint_id")


class CrossSectionDependency(BaseModel):
    dependency_id: str
    source_section: str
    target_sections: List[str] = Field(min_length=1)
    relation: DependencyRelation
    description: str

    @field_validator("dependency_id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="dependency_id")


class TerminologyConstraint(BaseModel):
    canonical_term: str
    allowed_variants: List[str] = Field(default_factory=list)
    forbidden_variants: List[str] = Field(default_factory=list)


class NotationConstraint(BaseModel):
    symbol: str
    meaning: str
    scope: str = "global"
    must_be_defined: bool = True


class LayoutRequirements(BaseModel):
    table_of_contents: bool = True
    numbered_sections: bool = True
    numbered_equations: bool = True
    numbered_theorems: bool = True
    required_callout_types: List[str] = Field(default_factory=list)
    preferred_layout_profile: str


class EvaluationContract(BaseModel):
    required_section_coverage: bool = True
    required_component_coverage: bool = True
    cross_section_dependency_check: bool = True
    terminology_consistency_check: bool = True
    notation_consistency_check: bool = True
    latex_rendering_check: bool = True
    length_check: bool = True


class SourceProvenanceEntry(BaseModel):
    source_id: str
    usage: str
    copied_text: bool = False

    @model_validator(mode="after")
    def _never_copied(self) -> "SourceProvenanceEntry":
        if self.copied_text:
            raise ValueError(
                f"source_provenance entry for '{self.source_id}' claims "
                "copied_text=True, which this benchmark never permits."
            )
        return self


class QualityScore(BaseModel):
    specificity: float = Field(ge=0.0, le=1.0)
    evaluability: float = Field(ge=0.0, le=1.0)
    latex_relevance: float = Field(ge=0.0, le=1.0)
    long_form_suitability: float = Field(ge=0.0, le=1.0)
    constraint_clarity: float = Field(ge=0.0, le=1.0)
    source_safety: float = Field(ge=0.0, le=1.0)


class BenchmarkItem(BaseModel):
    """One line of ``data/benchmark/benchmark_v1.jsonl`` (spec section 5).

    This is a document *contract*, not a gold reference document: evaluation
    later checks whether a generated LaTeX document satisfies this contract,
    it never diffs against a stored reference text.
    """

    sample_id: str
    benchmark_version: str = "v1"
    domain: Domain
    subdomain: str
    document_type: DocumentType
    difficulty: Difficulty
    title: str
    overview: str
    audience: str
    document_purpose: str
    target_length: TargetLength
    required_sections: List[RequiredSection] = Field(min_length=1)
    required_components: List[RequiredComponent] = Field(min_length=1)
    hard_constraints: List[HardConstraint] = Field(min_length=1)
    cross_section_dependencies: List[CrossSectionDependency] = Field(min_length=1)
    terminology_constraints: List[TerminologyConstraint] = Field(default_factory=list)
    notation_constraints: List[NotationConstraint] = Field(default_factory=list)
    layout_requirements: LayoutRequirements
    forbidden_content: List[str] = Field(default_factory=list)
    evaluation_contract: EvaluationContract
    natural_language_instruction: str
    source_provenance: List[SourceProvenanceEntry] = Field(min_length=1)
    quality_score: Optional[QualityScore] = None

    @field_validator("sample_id")
    @classmethod
    def _safe_sample_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="sample_id")

    @property
    def section_ids(self) -> List[str]:
        return [s.section_id for s in self.required_sections]

    @property
    def component_ids(self) -> List[str]:
        return [c.component_id for c in self.required_components]

    def model_dump_json_line(self) -> str:
        """Serialize to a single compact JSON line for JSONL output."""
        return self.model_dump_json(exclude_none=True)


# ---------------------------------------------------------------------------
# Validation results (used by validators.py / balance.py / statistics.py)
# ---------------------------------------------------------------------------


class ValidationIssue(BaseModel):
    rule_id: str
    message: str


class ValidationResult(BaseModel):
    sample_id: str
    accepted: bool
    quality_score: QualityScore
    issues: List[ValidationIssue] = Field(default_factory=list)


def export_benchmark_item_json_schema() -> Dict[str, Any]:
    """Optional JSON Schema export (spec section 11), via pydantic's
    built-in exporter -- no extra dependency (e.g. ``jsonschema``) required.
    """
    return BenchmarkItem.model_json_schema()

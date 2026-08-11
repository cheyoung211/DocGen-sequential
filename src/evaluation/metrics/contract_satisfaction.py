"""Metric 7 (Contract Satisfaction Rate), hard/deterministic categories only.

Checks a generated document (a project_dir under ``outputs/...``) against
the benchmark's ``BenchmarkItem`` contract. ``content_coverage`` (topic/
objective coverage) and ``cross_section_dependency`` (logical dependency
between sections) are semantic checks the research spec explicitly permits
deferring -- they are never attempted here; see ``ContractResult``.

Required Section Coverage matching is NOT an exact-title hard constraint --
the benchmark schema (``src/dataset/schemas.py``) never says matching must be
by exact title; that was this module's own earlier implementation choice.
Matching now tries three tiers, in priority order, per required section:

1. ``section_id`` equality against the generated blueprint's own section ids
   (structurally correct, but rarely fires today -- confirmed by grepping
   the whole generation pipeline that the Planner is never shown
   ``RequiredSection.section_id``, only the title and ``purpose``, embedded
   in prose inside ``natural_language_instruction``; a real generated
   ``plan.json`` shows the Planner inventing its own ``sec_``-prefixed ids
   independent of the benchmark's scheme).
2. Normalized-title equality (casefold + whitespace-collapse).
3. Keyword overlap between the required section's ``title``/``purpose`` and
   a still-unclaimed generated section's title -- catches the common case a
   pure title match misses, where the Planner elaborates a required title
   into a longer one (observed live: "Introduction" -> "Motivation and
   Structure of Convergence of Real Sequences") but the elaboration still
   shares real keywords with the required section's own stated purpose.
   Every required section's ``purpose`` also restates the *document's own
   topic* (e.g. every section of a "Basic Set Operations and Venn Diagrams"
   document has that phrase in its purpose text, and the Planner echoes it
   in every generated section title too) -- counting those topic words at
   full weight let two unrelated sections outscore the real match on shared
   topic vocabulary alone (caught live: "Prerequisites and Notation"
   matched to a generated "Synthesis of ..." section over the real
   "Prerequisites and Notation for ..." one). Keywords drawn from
   ``benchmark_item.title`` are therefore down-weighted, not excluded, so
   they can still break a tie when nothing else distinguishes two
   candidates.

Once a required section is matched and drafted, its status is no longer
existence-only: ``required_points`` (deterministic keyword-evidence search
over that section's drafted text, not an LLM judgment) decides between
``satisfied`` (every point evidenced) and ``partially_satisfied`` (some/none
evidenced). This is still a lexical, not semantic, check -- true meaning-
level content coverage remains out of scope here (``content_coverage`` stays
``None``/``not_implemented``; see ``ContractResult``) and is deferred to a
future semantic-evaluation pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from src.common.schemas import PlannerOutput, TextAgentOutput
from src.common.state import DocumentBlueprint, LayoutBlockKind, SectionLayout
from src.dataset.schemas import BenchmarkItem, ComponentType, RequiredSection
from src.evaluation.schemas import ContractCategoryDetail, ContractItemStatus, ContractResult

# Built from the Planner's own documented kind vocabulary
# (PLANNER_STRATEGY_PROMPT's "kind vocabulary" section in
# src/agents/planner.py) -- the authoritative mapping the Planner is
# actually instructed to use, not a guess at what "should" correspond to
# what. A ComponentType mapped to None has no LayoutBlockKind capable of
# expressing it at all in this pipeline today (PROPOSITION/COROLLARY/full
# PROOF/TIMELINE_PLACEHOLDER/EXERCISE/SUMMARY_BOX) -- evaluate_contract
# reports these as "not_representable", never conflated with "missing", so
# a permanent pipeline capability gap is never misattributed as this run's
# failure.
COMPONENT_TYPE_TO_LAYOUT_KIND: Dict[ComponentType, Optional[LayoutBlockKind]] = {
    ComponentType.DEFINITION: LayoutBlockKind.DEFINITION,
    ComponentType.THEOREM: LayoutBlockKind.THEOREM,
    ComponentType.LEMMA: LayoutBlockKind.THEOREM,
    ComponentType.PROPOSITION: None,
    ComponentType.COROLLARY: None,
    ComponentType.PROOF: None,
    ComponentType.PROOF_SKETCH: LayoutBlockKind.PROOF_SKETCH,
    ComponentType.EQUATION: LayoutBlockKind.EQUATION,
    ComponentType.ALIGNED_EQUATIONS: LayoutBlockKind.EQUATION,
    ComponentType.TABLE: LayoutBlockKind.TABLE,
    ComponentType.NOTATION_TABLE: LayoutBlockKind.NOTATION_TABLE,
    ComponentType.COMPARISON_TABLE: LayoutBlockKind.COMPARISON,
    ComponentType.TERMINOLOGY_TABLE: LayoutBlockKind.TABLE,
    ComponentType.CLASSIFICATION_TABLE: LayoutBlockKind.TABLE,
    ComponentType.WORKED_EXAMPLE: LayoutBlockKind.CASE_STUDY,
    ComponentType.CALLOUT: LayoutBlockKind.WARNING,
    ComponentType.RISK_CALLOUT: LayoutBlockKind.WARNING,
    ComponentType.BOXED_RESULT: LayoutBlockKind.KEY_TAKEAWAY,
    ComponentType.FIGURE_PLACEHOLDER: LayoutBlockKind.FIGURE,
    ComponentType.TIMELINE_PLACEHOLDER: None,
    ComponentType.EXERCISE: None,
    ComponentType.SUMMARY_BOX: None,
}


def _normalize_title(title: str) -> str:
    return " ".join(title.split()).casefold()


# Small, fixed stopword list for the keyword-overlap match tier and the
# required_points evidence check -- deliberately not a general-purpose NLP
# stopword list, just enough to strip connective/instructional words
# ("state", "present", "outline", ...) that would otherwise inflate overlap
# scores between any two unrelated sections' purpose text.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "that", "this", "these", "those", "into", "your", "each", "every",
    "any", "all", "its", "their", "other", "also", "using", "used", "use",
    "as", "is", "are", "was", "were", "be", "by", "from", "at", "if",
    "then", "state", "states", "present", "presents", "describe",
    "describes", "explain", "explains", "outline", "outlines", "discuss",
    "discusses", "include", "includes", "reference", "references",
    "restate", "restates", "establish", "establishes", "provide",
    "provides", "introduce", "introduces", "show", "shows", "list", "lists",
    "document", "section", "sections",
})


def _keywords(text: str) -> Set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.casefold())
    return {w for w in words if w not in _STOPWORDS and len(w) >= 4}


def _drafted_section_ids(project_dir: Path, blueprint: DocumentBlueprint) -> set:
    sections_dir = project_dir / "sections"
    return {
        section.section_id
        for section in blueprint.sections
        if (sections_dir / f"{section.section_id}.blocks.json").is_file()
    }


def _section_text(project_dir: Path, section: SectionLayout) -> str:
    """Rendered LaTeX when compilation reached this section, else the
    persisted semantic blocks -- whichever the run actually produced.
    Single-section counterpart to ``_collect_document_text``, used by the
    required_points evidence check."""
    sections_dir = project_dir / "sections"
    tex_path = sections_dir / f"{section.section_id}.tex"
    if tex_path.is_file():
        return tex_path.read_text(encoding="utf-8", errors="replace")
    blocks_path = sections_dir / f"{section.section_id}.blocks.json"
    if not blocks_path.is_file():
        return ""
    try:
        output = TextAgentOutput.model_validate(
            json.loads(blocks_path.read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, ValueError):
        return ""
    parts = [section.title]
    for block in output.blocks:
        if block.title:
            parts.append(block.title)
        parts.append(block.content)
    return "\n".join(parts)


def _point_is_evidenced(point: str, haystack_casefold: str) -> bool:
    """Deterministic, lexical stand-in for "was this required point covered"
    -- not a semantic judgment. A point with no extractable keywords (all
    stopwords/short words) can't be meaningfully checked this way, so it
    counts as evidenced rather than silently failing every such point."""
    point_keywords = _keywords(point)
    if not point_keywords:
        return True
    hits = sum(1 for kw in point_keywords if kw in haystack_casefold)
    return hits >= max(1, len(point_keywords) // 2)


#: Weight given to a shared keyword that's also part of the document's own
#: topic title (see the module docstring's tier-3 explanation) -- low enough
#: that any genuinely distinguishing shared word wins, but non-zero so topic
#: words can still break a tie when nothing else does.
_TOPIC_KEYWORD_WEIGHT = 0.2


def _match_required_section(
    req: RequiredSection,
    blueprint: DocumentBlueprint,
    claimed_ids: Set[str],
    topic_keywords: Set[str],
) -> "tuple[Optional[SectionLayout], str]":
    """Resolve which generated section (if any) corresponds to one required
    section, trying section_id, then normalized title, then purpose/title
    keyword overlap -- see this module's docstring for why in that order."""
    by_id = {s.section_id: s for s in blueprint.sections}
    candidate = by_id.get(req.section_id)
    if candidate is not None and candidate.section_id not in claimed_ids:
        return candidate, "section_id"

    normalized_req_title = _normalize_title(req.title)
    for section in blueprint.sections:
        if section.section_id in claimed_ids:
            continue
        if _normalize_title(section.title) == normalized_req_title:
            return section, "title"

    req_keywords = _keywords(req.title) | _keywords(req.purpose)
    if req_keywords:
        best: Optional[SectionLayout] = None
        best_score = 0.0
        for section in blueprint.sections:
            if section.section_id in claimed_ids:
                continue
            shared = req_keywords & _keywords(section.title)
            score = sum(
                _TOPIC_KEYWORD_WEIGHT if kw in topic_keywords else 1.0 for kw in shared
            )
            if score > best_score:
                best, best_score = section, score
        if best is not None:
            return best, "keyword_overlap"

    return None, "unmatched"


def _collect_document_text(project_dir: Path, blueprint: DocumentBlueprint) -> str:
    """Rendered LaTeX when compilation reached this section, else the
    persisted semantic blocks -- whichever the run actually produced."""
    parts: List[str] = []
    sections_dir = project_dir / "sections"
    for section in blueprint.sections:
        tex_path = sections_dir / f"{section.section_id}.tex"
        if tex_path.is_file():
            parts.append(tex_path.read_text(encoding="utf-8", errors="replace"))
            continue
        blocks_path = sections_dir / f"{section.section_id}.blocks.json"
        if not blocks_path.is_file():
            continue
        try:
            output = TextAgentOutput.model_validate(
                json.loads(blocks_path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, ValueError):
            continue
        for block in output.blocks:
            if block.title:
                parts.append(block.title)
            parts.append(block.content)
    return "\n".join(parts)


def _score_required_sections(
    project_dir: Path, blueprint: DocumentBlueprint, benchmark_item: BenchmarkItem
) -> "tuple[Optional[float], List[ContractCategoryDetail]]":
    drafted_ids = _drafted_section_ids(project_dir, blueprint)
    topic_keywords = _keywords(benchmark_item.title)

    details: List[ContractCategoryDetail] = []
    scores: List[float] = []
    claimed_ids: Set[str] = set()
    for req in benchmark_item.required_sections:
        matched, matched_via = _match_required_section(req, blueprint, claimed_ids, topic_keywords)

        if matched is None:
            details.append(ContractCategoryDetail(
                item_id=req.section_id, status="missing",
                reason=(
                    f"no generated section corresponds to {req.title!r} "
                    "(checked section_id, normalized title, and purpose-keyword overlap)"
                ),
                matched_via="unmatched",
            ))
            scores.append(0.0)
            continue

        claimed_ids.add(matched.section_id)
        if matched.section_id not in drafted_ids:
            details.append(ContractCategoryDetail(
                item_id=req.section_id, status="missing",
                reason=f"matched generated section {matched.section_id!r} (via {matched_via}) "
                       "was planned but never drafted",
                matched_via=matched_via,
            ))
            scores.append(0.0)
            continue

        if req.required_points:
            haystack = _section_text(project_dir, matched).casefold()
            evidenced = sum(1 for p in req.required_points if _point_is_evidenced(p, haystack))
            points_coverage = evidenced / len(req.required_points)
            points_note = f"{evidenced}/{len(req.required_points)} required_points evidenced"
        else:
            points_coverage = 1.0
            points_note = "no required_points to verify"

        status: ContractItemStatus = "satisfied" if points_coverage >= 1.0 else "partially_satisfied"
        scores.append(points_coverage)
        details.append(ContractCategoryDetail(
            item_id=req.section_id, status=status,
            reason=f"matched generated section {matched.section_id!r} via {matched_via}; {points_note}",
            matched_via=matched_via,
            required_points_coverage=points_coverage,
        ))

    total = len(benchmark_item.required_sections)
    score = (sum(scores) / total) if total else None
    return score, details


def _score_required_components(
    project_dir: Path, blueprint: DocumentBlueprint, benchmark_item: BenchmarkItem
) -> "tuple[Optional[float], List[ContractCategoryDetail]]":
    drafted_ids = _drafted_section_ids(project_dir, blueprint)
    kinds_present = {
        block.kind
        for section in blueprint.sections
        if section.section_id in drafted_ids
        for block in section.blocks
    }

    details: List[ContractCategoryDetail] = []
    satisfied = 0
    applicable = 0
    for comp in benchmark_item.required_components:
        layout_kind = COMPONENT_TYPE_TO_LAYOUT_KIND.get(comp.type)
        if layout_kind is None:
            details.append(ContractCategoryDetail(
                item_id=comp.component_id, status="not_representable",
                reason=f"ComponentType.{comp.type.value} has no corresponding LayoutBlockKind",
            ))
            continue
        applicable += 1
        if layout_kind in kinds_present:
            satisfied += 1
            details.append(ContractCategoryDetail(item_id=comp.component_id, status="satisfied"))
        else:
            details.append(ContractCategoryDetail(
                item_id=comp.component_id, status="missing",
                reason=f"no drafted block of kind '{layout_kind.value}' found",
            ))

    score = (satisfied / applicable) if applicable else None
    return score, details


def _score_terminology(
    document_text: str, benchmark_item: BenchmarkItem
) -> "tuple[Optional[float], List[ContractCategoryDetail]]":
    constraints = benchmark_item.terminology_constraints
    if not benchmark_item.evaluation_contract.terminology_consistency_check or not constraints:
        return None, []

    text_casefold = document_text.casefold()
    details: List[ContractCategoryDetail] = []
    satisfied = 0
    for constraint in constraints:
        candidates = [constraint.canonical_term, *constraint.allowed_variants]
        term_used = any(c.casefold() in text_casefold for c in candidates if c)
        forbidden_used = any(
            v.casefold() in text_casefold for v in constraint.forbidden_variants if v
        )
        if term_used and not forbidden_used:
            satisfied += 1
            details.append(
                ContractCategoryDetail(item_id=constraint.canonical_term, status="satisfied")
            )
        else:
            reason = (
                "forbidden variant present" if forbidden_used
                else "canonical term (or an allowed variant) never appears in the document"
            )
            details.append(ContractCategoryDetail(
                item_id=constraint.canonical_term, status="missing", reason=reason
            ))

    return satisfied / len(constraints), details


def _score_notation(
    document_text: str, benchmark_item: BenchmarkItem
) -> "tuple[Optional[float], List[ContractCategoryDetail]]":
    constraints = benchmark_item.notation_constraints
    if not benchmark_item.evaluation_contract.notation_consistency_check or not constraints:
        return None, []

    details: List[ContractCategoryDetail] = []
    satisfied = 0
    for constraint in constraints:
        if constraint.symbol in document_text:
            satisfied += 1
            details.append(ContractCategoryDetail(item_id=constraint.symbol, status="satisfied"))
        else:
            details.append(ContractCategoryDetail(
                item_id=constraint.symbol, status="missing",
                reason="symbol never appears in the document",
            ))

    return satisfied / len(constraints), details


def evaluate_contract(project_dir: Path, benchmark_item: BenchmarkItem) -> ContractResult:
    """Pure, read-only, no LLM/network calls. Reads ``project_dir/plan.json``
    (the blueprint) and ``project_dir/sections/*``. Returns a null-filled
    ``ContractResult`` if ``plan.json`` doesn't exist yet (planning itself
    never completed)."""
    plan_path = project_dir / "plan.json"
    if not plan_path.is_file():
        return ContractResult()

    plan = PlannerOutput.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))
    blueprint = plan.blueprint

    required_sections_score, section_details = _score_required_sections(
        project_dir, blueprint, benchmark_item
    )
    required_components_score, component_details = _score_required_components(
        project_dir, blueprint, benchmark_item
    )
    document_text = _collect_document_text(project_dir, blueprint)
    terminology_score, terminology_details = _score_terminology(document_text, benchmark_item)
    notation_score, notation_details = _score_notation(document_text, benchmark_item)

    applicable_scores = [
        s for s in (required_sections_score, required_components_score, terminology_score, notation_score)
        if s is not None
    ]
    overall = (sum(applicable_scores) / len(applicable_scores)) if applicable_scores else None

    return ContractResult(
        overall=overall,
        required_sections=required_sections_score,
        required_components=required_components_score,
        terminology=terminology_score,
        notation=notation_score,
        details={
            "required_sections": section_details,
            "required_components": component_details,
            "terminology": terminology_details,
            "notation": notation_details,
        },
    )


def aggregate_contract_satisfaction(results: Sequence[ContractResult]) -> dict:
    """Category-level mean across a batch, skipping None (not-applicable/not-computed) per sample."""

    def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
        present = [v for v in values if v is not None]
        return (sum(present) / len(present)) if present else None

    return {
        "overall": _mean([r.overall for r in results]),
        "required_sections": _mean([r.required_sections for r in results]),
        "required_components": _mean([r.required_components for r in results]),
        "content_coverage": None,
        "terminology": _mean([r.terminology for r in results]),
        "notation": _mean([r.notation for r in results]),
        "cross_section_dependency": None,
        "not_implemented": ["content_coverage", "cross_section_dependency"],
        "sample_count": len(results),
    }

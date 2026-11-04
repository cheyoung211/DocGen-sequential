"""Requirement-to-artifact mapping for the "Revision (Additional verifiers)"
plan's Fix C, checklist item C1 (``docs/av_stage2_dependency_mapping.md``).

Builds, from one ``BenchmarkItem`` (the authoring-time contract,
``src/dataset/schemas.py``) and one generated ``DocumentGraph`` (blueprint +
label registry, ``src/common/state.py``), a structural map from every
``RequiredSection``/``RequiredComponent``/``CrossSectionDependency`` id to
the real generated section/block/label that satisfies it -- or an explicit,
never-silently-dropped reason why it doesn't.

Reuses, rather than reimplements, the one matcher this codebase already has
for this problem: ``_match_required_section``/``_keywords``/
``COMPONENT_TYPE_TO_LAYOUT_KIND`` (``src/evaluation/metrics/
contract_satisfaction.py``) -- the same functions ``src/agents/planner.py``
already imports for its own blueprint validation. This module does not
duplicate that logic for components; it extends the same kind/keyword
matching one level deeper (section -> block), using the fact that the
benchmark's own generator (``src/dataset/instruction_builder.py``) always
sets ``RequiredComponent.placement`` and ``CrossSectionDependency.
source_section``/``target_sections`` to a real ``RequiredSection.section_id``
from the same item -- confirmed by reading that generator, not assumed (see
the design doc's "0. 현재 코드 상태" section).

What this module deliberately does NOT do (see the design doc's "4. 명시적으로
미룬 것"): judge adjacent-label acceptance, check actual content reuse/
consistency, check whether a cited section's content really supports what
cites it, or decide a ``must_precede`` violation. It only produces the
structural references + obligation records those later checks need.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.common.state import DocumentBlueprint, DocumentGraph, LayoutBlock, default_block_label
from src.dataset.schemas import (
    BenchmarkItem,
    CrossSectionDependency,
    DependencyRelation,
    RequiredComponent,
    RequiredSection,
)
from src.evaluation.metrics.contract_satisfaction import (
    COMPONENT_TYPE_TO_LAYOUT_KIND,
    _TOPIC_KEYWORD_WEIGHT,
    _keywords,
    _match_required_section,
    _normalize_title,
)

#: Same literal string Stage 0 pinned as ``AdditionalVerifierRunMetadata.
#: rubric_version``'s default (``src/evaluation/schemas.py``) -- the Gold
#: Labeling page's own last-edited date as of when this module's matching
#: rules were extracted from it. Kept as one shared constant so a future
#: change to either side doesn't silently drift out of sync.
DEPENDENCY_MAPPING_RUBRIC_VERSION = "gold_labeling_draft_2026-10-26"


class MappingStatus(str, Enum):
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"
    NOT_REPRESENTABLE = "not_representable"
    DUPLICATE_ID = "duplicate_id"


class SectionMapping(BaseModel):
    section_id: str
    status: MappingStatus
    matched_section_id: Optional[str] = None
    matched_via: Optional[str] = None
    reason: str = ""


class ComponentMapping(BaseModel):
    component_id: str
    status: MappingStatus
    placement_section_id: str
    #: ``RequiredComponent.reference_required`` carried straight through,
    #: regardless of ``status`` -- this is a fact about the benchmark's own
    #: requirement (principle 4), not something matching produces, so it
    #: must survive even an UNMATCHED/AMBIGUOUS/NOT_REPRESENTABLE/
    #: DUPLICATE_ID component exactly as declared. A later round (C3/B)
    #: checking "is this component actually cited somewhere" needs this
    #: flag to know whether an absent citation is a FAIL or trivially a
    #: PASS (principle 4: ``reference_required=False`` needs no citation
    #: at all) -- without it, that distinction cannot be recovered from the
    #: report alone.
    reference_required: bool = False
    matched_section_id: Optional[str] = None
    matched_block_id: Optional[str] = None
    matched_label: Optional[str] = None
    matched_via: Optional[str] = None
    reason: str = ""


class CrossSectionObligation(BaseModel):
    dependency_id: str
    relation: DependencyRelation
    description: str
    source_section_id: str
    target_section_id: str
    status: MappingStatus
    matched_source_section_id: Optional[str] = None
    matched_target_section_id: Optional[str] = None
    allowed_labels: List[str] = Field(default_factory=list)
    content_check_required: bool = False
    adjacent_label_possible: bool = False
    order_hint: Optional[Tuple[int, int]] = None
    reason: str = ""


class ComponentDependencyObligation(BaseModel):
    component_id: str
    depends_on_component_id: str
    status: MappingStatus
    matched_component_block_id: Optional[str] = None
    matched_depends_on_block_id: Optional[str] = None
    content_check_required: bool = True
    reason: str = ""


class DependencyMappingReport(BaseModel):
    rubric_version: str = DEPENDENCY_MAPPING_RUBRIC_VERSION
    section_mappings: List[SectionMapping] = Field(default_factory=list)
    component_mappings: List[ComponentMapping] = Field(default_factory=list)
    cross_section_obligations: List[CrossSectionObligation] = Field(default_factory=list)
    component_dependency_obligations: List[ComponentDependencyObligation] = Field(default_factory=list)
    data_quality_issues: List[str] = Field(default_factory=list)


def _duplicate_ids(ids: List[str], *, label: str) -> Tuple[Set[str], List[str]]:
    """Returns (set of ids that appear more than once, one report line per
    such id) -- never silently keeps "the last one wins"."""
    seen: Dict[str, int] = {}
    for id_ in ids:
        seen[id_] = seen.get(id_, 0) + 1
    duplicates = {id_ for id_, count in seen.items() if count > 1}
    issues = [f"{label} id {id_!r} is declared {seen[id_]} times in this BenchmarkItem." for id_ in sorted(duplicates)]
    return duplicates, issues


def _candidate_blocks(
    blocks: List[LayoutBlock], claimed_block_ids: Set[str], layout_kind
) -> List[LayoutBlock]:
    return [b for b in blocks if b.id not in claimed_block_ids and b.kind == layout_kind]


def _best_keyword_match(
    comp: RequiredComponent, candidates: List[LayoutBlock]
) -> Tuple[Optional[LayoutBlock], bool]:
    """Returns (best candidate or None, is_ambiguous). Mirrors
    ``_match_required_section``'s keyword-overlap tier, but never falls back
    to "pick anything" the way that tier's caller does for sections -- a tie
    (including an all-zero tie) here is reported AMBIGUOUS, not resolved by
    arbitrary list order, per the design doc's §2 rationale."""
    comp_keywords = _keywords(comp.description)
    scored: List[Tuple[int, LayoutBlock]] = []
    for block in candidates:
        block_keywords = _keywords(f"{block.title or ''} {block.instructions}")
        scored.append((len(comp_keywords & block_keywords), block))
    if not scored:
        return None, False
    max_score = max(score for score, _ in scored)
    top = [block for score, block in scored if score == max_score]
    if len(top) == 1:
        return top[0], False
    return None, True


def _tied_section_candidates(
    req: RequiredSection,
    blueprint: DocumentBlueprint,
    claimed_ids: Set[str],
    topic_keywords: Set[str],
    matched_via: str,
) -> List[str]:
    """Recomputes, for the SAME tier ``_match_required_section`` actually
    matched on, whether more than one unclaimed section ties for that
    match -- without re-implementing which section *wins* a tier (that
    stays ``_match_required_section``'s own job, reused as-is). A review
    found that reusing it as-is silently hid this: its title/keyword tiers
    both resolve a tie by first-seen list order, so reversing the order of
    two identically-titled (or identically-scoring) sections changes which
    one a required section matches to, with both runs reporting MATCHED --
    an unreported ambiguity. ``matched_via == "section_id"`` can never be
    ambiguous (an exact ``section_id`` lookup names exactly one node) and
    is not re-checked here."""
    if matched_via == "title":
        normalized_req_title = _normalize_title(req.title)
        tied = [
            s.section_id for s in blueprint.sections
            if s.section_id not in claimed_ids and _normalize_title(s.title) == normalized_req_title
        ]
        return sorted(tied) if len(tied) > 1 else []

    if matched_via == "keyword_overlap":
        req_keywords = _keywords(req.title) | _keywords(req.purpose)
        if not req_keywords:
            return []
        scored: List[Tuple[float, str]] = []
        for section in blueprint.sections:
            if section.section_id in claimed_ids:
                continue
            shared = req_keywords & _keywords(section.title)
            score = sum(_TOPIC_KEYWORD_WEIGHT if kw in topic_keywords else 1.0 for kw in shared)
            scored.append((score, section.section_id))
        if not scored:
            return []
        max_score = max(score for score, _ in scored)
        if max_score <= 0:
            return []
        tied = sorted(sid for score, sid in scored if score == max_score)
        return tied if len(tied) > 1 else []

    return []


def build_section_mappings(benchmark_item: BenchmarkItem, graph: DocumentGraph) -> List[SectionMapping]:
    """Section-level mapping -- thin wrapper over the existing shared
    matcher (``_match_required_section``), not a reimplementation."""
    blueprint = graph.blueprint
    assert blueprint is not None, "build_section_mappings requires graph.blueprint to be set."

    duplicate_ids, dup_issues = _duplicate_ids(
        [s.section_id for s in benchmark_item.required_sections], label="required_section"
    )
    topic_keywords = _keywords(benchmark_item.title)
    claimed_ids: Set[str] = set()
    mappings: List[SectionMapping] = []
    for req in benchmark_item.required_sections:
        if req.section_id in duplicate_ids:
            mappings.append(
                SectionMapping(
                    section_id=req.section_id,
                    status=MappingStatus.DUPLICATE_ID,
                    reason="this section_id is declared more than once in required_sections; skipped rather than guessed.",
                )
            )
            continue
        matched, matched_via = _match_required_section(req, blueprint, claimed_ids, topic_keywords)
        if matched is None:
            mappings.append(
                SectionMapping(
                    section_id=req.section_id,
                    status=MappingStatus.UNMATCHED,
                    reason="no generated section matched via section_id, normalized title, or keyword overlap.",
                )
            )
            continue
        tied = _tied_section_candidates(req, blueprint, claimed_ids, topic_keywords, matched_via)
        if tied:
            mappings.append(
                SectionMapping(
                    section_id=req.section_id,
                    status=MappingStatus.AMBIGUOUS,
                    matched_via=matched_via,
                    reason=(
                        f"{len(tied)} unclaimed sections tie on {matched_via!r} for this required "
                        "section, not claiming any of them by arbitrary order: " + ", ".join(tied)
                    ),
                )
            )
            continue
        claimed_ids.add(matched.section_id)
        mappings.append(
            SectionMapping(
                section_id=req.section_id,
                status=MappingStatus.MATCHED,
                matched_section_id=matched.section_id,
                matched_via=matched_via,
            )
        )
    # dup_issues are surfaced by the top-level build_dependency_mapping_report,
    # not from this helper (it only returns List[SectionMapping]); callers
    # that need them call _duplicate_ids themselves. Kept local here so this
    # function stays a pure, independently testable mapper.
    del dup_issues
    return mappings


def build_dependency_mapping_report(
    benchmark_item: BenchmarkItem, graph: DocumentGraph
) -> DependencyMappingReport:
    blueprint = graph.blueprint
    assert blueprint is not None, "build_dependency_mapping_report requires graph.blueprint to be set."

    data_quality_issues: List[str] = []

    section_dup_ids, section_dup_issues = _duplicate_ids(
        [s.section_id for s in benchmark_item.required_sections], label="required_section"
    )
    data_quality_issues.extend(section_dup_issues)
    section_mappings = build_section_mappings(benchmark_item, graph)
    section_by_req_id = {m.section_id: m for m in section_mappings}

    component_dup_ids, component_dup_issues = _duplicate_ids(
        [c.component_id for c in benchmark_item.required_components], label="required_component"
    )
    data_quality_issues.extend(component_dup_issues)

    dependency_dup_ids, dependency_dup_issues = _duplicate_ids(
        [d.dependency_id for d in benchmark_item.cross_section_dependencies], label="cross_section_dependency"
    )
    data_quality_issues.extend(dependency_dup_issues)

    sections_by_id = {s.section_id: s for s in blueprint.sections}
    claimed_block_ids: Set[str] = set()
    component_mappings: List[ComponentMapping] = []
    for comp in benchmark_item.required_components:
        if comp.component_id in component_dup_ids:
            component_mappings.append(
                ComponentMapping(
                    component_id=comp.component_id,
                    status=MappingStatus.DUPLICATE_ID,
                    placement_section_id=comp.placement,
                    reference_required=comp.reference_required,
                    reason="this component_id is declared more than once in required_components; skipped rather than guessed.",
                )
            )
            continue

        placement_mapping = section_by_req_id.get(comp.placement)
        if placement_mapping is None or placement_mapping.status != MappingStatus.MATCHED:
            component_mappings.append(
                ComponentMapping(
                    component_id=comp.component_id,
                    status=MappingStatus.UNMATCHED,
                    placement_section_id=comp.placement,
                    reference_required=comp.reference_required,
                    reason=(
                        f"placement section {comp.placement!r} did not resolve to a real generated "
                        "section (see its own SectionMapping entry for why)."
                    ),
                )
            )
            continue

        layout_kind = COMPONENT_TYPE_TO_LAYOUT_KIND.get(comp.type)
        if layout_kind is None:
            component_mappings.append(
                ComponentMapping(
                    component_id=comp.component_id,
                    status=MappingStatus.NOT_REPRESENTABLE,
                    placement_section_id=comp.placement,
                    reference_required=comp.reference_required,
                    matched_section_id=placement_mapping.matched_section_id,
                    reason=f"ComponentType.{comp.type.value} has no corresponding LayoutBlockKind in this pipeline.",
                )
            )
            continue

        section_layout = sections_by_id[placement_mapping.matched_section_id]
        candidates = _candidate_blocks(section_layout.blocks, claimed_block_ids, layout_kind)
        if not candidates:
            component_mappings.append(
                ComponentMapping(
                    component_id=comp.component_id,
                    status=MappingStatus.UNMATCHED,
                    placement_section_id=comp.placement,
                    reference_required=comp.reference_required,
                    matched_section_id=placement_mapping.matched_section_id,
                    reason=f"no unclaimed block of kind {layout_kind.value!r} in section {placement_mapping.matched_section_id!r}.",
                )
            )
            continue

        if len(candidates) == 1:
            best, matched_via = candidates[0], "kind_only"
        else:
            best, ambiguous = _best_keyword_match(comp, candidates)
            if ambiguous or best is None:
                component_mappings.append(
                    ComponentMapping(
                        component_id=comp.component_id,
                        status=MappingStatus.AMBIGUOUS,
                        placement_section_id=comp.placement,
                        reference_required=comp.reference_required,
                        matched_section_id=placement_mapping.matched_section_id,
                        reason=(
                            f"{len(candidates)} unclaimed blocks of kind {layout_kind.value!r} in section "
                            f"{placement_mapping.matched_section_id!r} tie on keyword overlap with this "
                            "component's description: " + ", ".join(sorted(b.id for b in candidates))
                        ),
                    )
                )
                continue
            matched_via = "kind_and_keyword"

        claimed_block_ids.add(best.id)
        matched_label = default_block_label(best.kind, best.id)
        component_mappings.append(
            ComponentMapping(
                component_id=comp.component_id,
                status=MappingStatus.MATCHED,
                placement_section_id=comp.placement,
                reference_required=comp.reference_required,
                matched_section_id=placement_mapping.matched_section_id,
                matched_block_id=best.id,
                matched_label=matched_label,
                matched_via=matched_via,
            )
        )

    component_by_id = {c.component_id: c for c in component_mappings}

    cross_section_obligations: List[CrossSectionObligation] = []
    for dep in benchmark_item.cross_section_dependencies:
        if dep.dependency_id in dependency_dup_ids:
            for target in dep.target_sections:
                cross_section_obligations.append(
                    CrossSectionObligation(
                        dependency_id=dep.dependency_id,
                        relation=dep.relation,
                        description=dep.description,
                        source_section_id=dep.source_section,
                        target_section_id=target,
                        status=MappingStatus.DUPLICATE_ID,
                        reason="this dependency_id is declared more than once in cross_section_dependencies; skipped rather than guessed.",
                    )
                )
            continue

        source_mapping = section_by_req_id.get(dep.source_section)
        for target_section_id in dep.target_sections:
            target_mapping = section_by_req_id.get(target_section_id)
            obligation = _build_cross_section_obligation(
                dep, target_section_id, source_mapping, target_mapping, component_mappings, blueprint
            )
            cross_section_obligations.append(obligation)

    component_dependency_obligations: List[ComponentDependencyObligation] = []
    for comp in benchmark_item.required_components:
        for dep_component_id in comp.depends_on:
            if comp.component_id in component_dup_ids:
                # A duplicated component_id's depends_on edges are real,
                # declared requirements too -- report them as DUPLICATE_ID
                # rather than dropping them with no trace just because the
                # component itself was excluded from matching above.
                component_dependency_obligations.append(
                    ComponentDependencyObligation(
                        component_id=comp.component_id,
                        depends_on_component_id=dep_component_id,
                        status=MappingStatus.DUPLICATE_ID,
                        reason=(
                            f"component {comp.component_id!r} is declared more than once in "
                            "required_components; its depends_on edges are skipped rather than guessed."
                        ),
                    )
                )
                continue
            this_mapping = component_by_id.get(comp.component_id)
            dep_mapping = component_by_id.get(dep_component_id)
            component_dependency_obligations.append(
                _build_component_dependency_obligation(comp.component_id, dep_component_id, this_mapping, dep_mapping)
            )

    return DependencyMappingReport(
        section_mappings=section_mappings,
        component_mappings=component_mappings,
        cross_section_obligations=cross_section_obligations,
        component_dependency_obligations=component_dependency_obligations,
        data_quality_issues=data_quality_issues,
    )


def _combine_status(source: Optional[SectionMapping], target: Optional[SectionMapping]) -> MappingStatus:
    """Propagates the more specific "why this isn't matched" reason rather
    than inventing a new one -- a DUPLICATE_ID or AMBIGUOUS source/target
    outranks a plain UNMATCHED in the combined status, since it carries more
    information about what actually went wrong."""
    for candidate in (source, target):
        if candidate is not None and candidate.status == MappingStatus.DUPLICATE_ID:
            return MappingStatus.DUPLICATE_ID
    for candidate in (source, target):
        if candidate is not None and candidate.status == MappingStatus.AMBIGUOUS:
            return MappingStatus.AMBIGUOUS
    if source is None or source.status != MappingStatus.MATCHED:
        return MappingStatus.UNMATCHED
    if target is None or target.status != MappingStatus.MATCHED:
        return MappingStatus.UNMATCHED
    return MappingStatus.MATCHED


def _build_cross_section_obligation(
    dep: CrossSectionDependency,
    target_section_id: str,
    source_mapping: Optional[SectionMapping],
    target_mapping: Optional[SectionMapping],
    component_mappings: List[ComponentMapping],
    blueprint: DocumentBlueprint,
) -> CrossSectionObligation:
    status = _combine_status(source_mapping, target_mapping)
    if status != MappingStatus.MATCHED:
        reasons = []
        if source_mapping is None or source_mapping.status != MappingStatus.MATCHED:
            reasons.append(f"source_section {dep.source_section!r} did not resolve to a real generated section.")
        if target_mapping is None or target_mapping.status != MappingStatus.MATCHED:
            reasons.append(f"target_section {target_section_id!r} did not resolve to a real generated section.")
        return CrossSectionObligation(
            dependency_id=dep.dependency_id,
            relation=dep.relation,
            description=dep.description,
            source_section_id=dep.source_section,
            target_section_id=target_section_id,
            status=status,
            matched_source_section_id=(
                source_mapping.matched_section_id
                if source_mapping is not None and source_mapping.status == MappingStatus.MATCHED
                else None
            ),
            matched_target_section_id=(
                target_mapping.matched_section_id
                if target_mapping is not None and target_mapping.status == MappingStatus.MATCHED
                else None
            ),
            reason=" ".join(reasons),
        )

    matched_source = source_mapping.matched_section_id
    matched_target = target_mapping.matched_section_id
    source_components = [
        c for c in component_mappings
        if c.status == MappingStatus.MATCHED and c.placement_section_id == dep.source_section
    ]
    allowed_labels = [f"sec:{matched_source}"] + [
        c.matched_label for c in source_components if c.matched_label is not None
    ]
    adjacent_label_possible = any(c.matched_label is None for c in source_components)

    order_hint: Optional[Tuple[int, int]] = None
    section_order = [s.section_id for s in blueprint.sections]
    if matched_source in section_order and matched_target in section_order:
        order_hint = (section_order.index(matched_source), section_order.index(matched_target))

    return CrossSectionObligation(
        dependency_id=dep.dependency_id,
        relation=dep.relation,
        description=dep.description,
        source_section_id=dep.source_section,
        target_section_id=target_section_id,
        status=MappingStatus.MATCHED,
        matched_source_section_id=matched_source,
        matched_target_section_id=matched_target,
        allowed_labels=allowed_labels,
        content_check_required=dep.relation in (DependencyRelation.MUST_REUSE, DependencyRelation.MUST_BE_CONSISTENT_WITH),
        adjacent_label_possible=adjacent_label_possible,
        order_hint=order_hint,
    )


def _build_component_dependency_obligation(
    component_id: str,
    depends_on_component_id: str,
    this_mapping: Optional[ComponentMapping],
    dep_mapping: Optional[ComponentMapping],
) -> ComponentDependencyObligation:
    this_ok = this_mapping is not None and this_mapping.status == MappingStatus.MATCHED
    dep_ok = dep_mapping is not None and dep_mapping.status == MappingStatus.MATCHED
    if this_ok and dep_ok:
        return ComponentDependencyObligation(
            component_id=component_id,
            depends_on_component_id=depends_on_component_id,
            status=MappingStatus.MATCHED,
            matched_component_block_id=this_mapping.matched_block_id,
            matched_depends_on_block_id=dep_mapping.matched_block_id,
        )
    reasons = []
    if not this_ok:
        reasons.append(f"component {component_id!r} did not resolve to a real generated block.")
    if not dep_ok:
        reasons.append(f"depends_on component {depends_on_component_id!r} did not resolve to a real generated block.")
    return ComponentDependencyObligation(
        component_id=component_id,
        depends_on_component_id=depends_on_component_id,
        status=MappingStatus.UNMATCHED,
        reason=" ".join(reasons),
    )


__all__ = [
    "DEPENDENCY_MAPPING_RUBRIC_VERSION",
    "MappingStatus",
    "SectionMapping",
    "ComponentMapping",
    "CrossSectionObligation",
    "ComponentDependencyObligation",
    "DependencyMappingReport",
    "build_section_mappings",
    "build_dependency_mapping_report",
]

"""Coverage for src/core/dependency_constraint_provider.py (Fix C,
checklist item C4: a real ConstraintProvider for DEPENDENCY constraints,
closing Stage 0's documented "no production ConstraintProvider exists
anywhere" gap for this constraint family)."""

from __future__ import annotations

from src.core.artifact_graph import ArtifactGraphStore
from src.core.dependency_constraint_provider import DependencyConstraintProvider
from src.core.dependency_mapping import CrossSectionObligation, DependencyMappingReport, MappingStatus
from src.core.schemas import ArtifactKind, ArtifactRef, ConstraintKind
from src.dataset.schemas import DependencyRelation


def _report(obligations):
    return DependencyMappingReport(
        section_mappings=[], component_mappings=[],
        cross_section_obligations=obligations, component_dependency_obligations=[],
    )


def test_returns_one_constraint_per_matched_obligation_targeting_this_artifact():
    store = ArtifactGraphStore()
    store.add_version("intro", ArtifactKind.SECTION, content="x")
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_REFERENCE, description="cite intro",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary", allowed_labels=["sec:intro"],
    )
    provider = DependencyConstraintProvider(_report([obligation]), store)
    target_ref = ArtifactRef(artifact_id="summary", version=1, kind=ArtifactKind.SECTION)

    constraints = provider(target_ref)
    assert len(constraints) == 1
    assert constraints[0].kind == ConstraintKind.DEPENDENCY
    assert constraints[0].constraint_id == "d1::summary"
    assert constraints[0].applies_to[0].artifact_id == "intro"
    assert constraints[0].applies_to[0].version == 1  # store.latest("intro") is version 1


def test_unmatched_obligations_produce_no_constraint():
    store = ArtifactGraphStore()
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_REFERENCE, description="x",
        source_section_id="intro", target_section_id="ghost", status=MappingStatus.UNMATCHED,
    )
    provider = DependencyConstraintProvider(_report([obligation]), store)
    target_ref = ArtifactRef(artifact_id="ghost", version=1, kind=ArtifactKind.SECTION)
    assert provider(target_ref) == []


def test_other_artifacts_obligations_are_not_returned():
    store = ArtifactGraphStore()
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_REFERENCE, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
    )
    provider = DependencyConstraintProvider(_report([obligation]), store)
    other_ref = ArtifactRef(artifact_id="other_section", version=1, kind=ArtifactKind.SECTION)
    assert provider(other_ref) == []


def test_source_ref_uses_the_stores_current_version_not_a_guess():
    store = ArtifactGraphStore()
    store.add_version("intro", ArtifactKind.SECTION, content="v1")
    store.add_version("intro", ArtifactKind.SECTION, content="v2")  # now at version 2
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_REFERENCE, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
    )
    provider = DependencyConstraintProvider(_report([obligation]), store)
    target_ref = ArtifactRef(artifact_id="summary", version=1, kind=ArtifactKind.SECTION)
    constraints = provider(target_ref)
    assert constraints[0].applies_to[0].version == 2


def test_source_ref_is_a_safe_placeholder_when_the_source_was_never_stored():
    store = ArtifactGraphStore()
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_REFERENCE, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
    )
    provider = DependencyConstraintProvider(_report([obligation]), store)
    target_ref = ArtifactRef(artifact_id="summary", version=1, kind=ArtifactKind.SECTION)
    constraints = provider(target_ref)
    # version=1 for a never-stored artifact_id is correctly detected as
    # missing by _resolve_input_refs -- confirm it really doesn't resolve.
    assert store.get_version("intro", constraints[0].applies_to[0].version) is None


def test_every_relation_gets_a_constraint_including_ones_not_yet_checkable():
    # must_reuse/must_be_consistent_with still need a Constraint so the
    # unresolved dependency stays visible/trackable, even though
    # _check_dependency can only answer UNKNOWN for them right now.
    store = ArtifactGraphStore()
    obligations = [
        CrossSectionObligation(
            dependency_id=f"d{i}", relation=relation, description="x",
            source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
            matched_source_section_id="intro", matched_target_section_id="summary",
        )
        for i, relation in enumerate(DependencyRelation)
    ]
    provider = DependencyConstraintProvider(_report(obligations), store)
    target_ref = ArtifactRef(artifact_id="summary", version=1, kind=ArtifactKind.SECTION)
    assert len(provider(target_ref)) == len(list(DependencyRelation))


def test_context_for_matches_call_one_to_one():
    store = ArtifactGraphStore()
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_PRECEDE, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
        order_hint=(0, 3),
    )
    provider = DependencyConstraintProvider(_report([obligation]), store)
    target_ref = ArtifactRef(artifact_id="summary", version=1, kind=ArtifactKind.SECTION)

    constraints = provider(target_ref)
    context = provider.context_for(target_ref)
    assert set(context["dependency"].keys()) == {c.constraint_id for c in constraints}
    assert context["dependency"]["d1::summary"] == {
        "relation": "must_precede", "allowed_labels": [], "order_hint": [0, 3],
        "relevant_symbols": [],
    }


def test_relevant_symbols_default_to_empty_without_opting_in():
    # Fix B's B3 registry needs an explicit per-dependency symbol
    # allowlist (never guessed) -- a caller who doesn't pass
    # relevant_symbols_by_dependency gets exactly the pre-B3 shape.
    store = ArtifactGraphStore()
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
    )
    provider = DependencyConstraintProvider(_report([obligation]), store)
    target_ref = ArtifactRef(artifact_id="summary", version=1, kind=ArtifactKind.SECTION)
    context = provider.context_for(target_ref)
    assert context["dependency"]["d1::summary"]["relevant_symbols"] == []


def test_relevant_symbols_are_threaded_through_when_supplied():
    store = ArtifactGraphStore()
    obligation = CrossSectionObligation(
        dependency_id="d1", relation=DependencyRelation.MUST_BE_CONSISTENT_WITH, description="x",
        source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
        matched_source_section_id="intro", matched_target_section_id="summary",
    )
    provider = DependencyConstraintProvider(
        _report([obligation]), store, relevant_symbols_by_dependency={"d1": ["n", "\\alpha"]}
    )
    target_ref = ArtifactRef(artifact_id="summary", version=1, kind=ArtifactKind.SECTION)
    context = provider.context_for(target_ref)
    assert context["dependency"]["d1::summary"]["relevant_symbols"] == ["n", "\\alpha"]


def test_two_targets_of_the_same_dependency_id_get_distinct_constraint_ids():
    store = ArtifactGraphStore()
    obligations = [
        CrossSectionObligation(
            dependency_id="reuse_core", relation=DependencyRelation.MUST_REFERENCE, description="x",
            source_section_id="intro", target_section_id="summary", status=MappingStatus.MATCHED,
            matched_source_section_id="intro", matched_target_section_id="summary",
        ),
        CrossSectionObligation(
            dependency_id="reuse_core", relation=DependencyRelation.MUST_REFERENCE, description="x",
            source_section_id="intro", target_section_id="worked_examples", status=MappingStatus.MATCHED,
            matched_source_section_id="intro", matched_target_section_id="worked_examples",
        ),
    ]
    provider = DependencyConstraintProvider(_report(obligations), store)
    summary_ref = ArtifactRef(artifact_id="summary", version=1, kind=ArtifactKind.SECTION)
    we_ref = ArtifactRef(artifact_id="worked_examples", version=1, kind=ArtifactKind.SECTION)
    assert provider(summary_ref)[0].constraint_id != provider(we_ref)[0].constraint_id

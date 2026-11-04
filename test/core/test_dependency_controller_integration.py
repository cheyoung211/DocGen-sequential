"""End-to-end coverage for Fix C checklist item C4 ("dependency 위반
evidence를 기존 check·patch·replan에 연결한다"): a DEPENDENCY constraint,
built from a real DependencyMappingReport via DependencyConstraintProvider,
dispatched through the REAL ToolExecutor/ArtifactGraphStore/
FixedGraphAwareRepairPolicy machinery from the other ("Graph artifacts")
revision -- not a disconnected unit test of constraint_checks.py alone."""

from __future__ import annotations

from src.core.artifact_graph import ArtifactGraphStore
from src.core.constraint_checks import BasicConstraintChecker
from src.core.controller import ControllerState, RuleBasedController
from src.core.dependency_constraint_provider import DependencyConstraintProvider
from src.core.dependency_mapping import CrossSectionObligation, DependencyMappingReport, MappingStatus
from src.core.dependency_verifier import verify_cross_section_obligations
from src.core.repair_operations import FixedGraphAwareRepairPolicy
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    CheckCall,
    CheckOutcome,
    RepairActionType,
)
from src.core.tool_executor import ToolExecutor
from src.dataset.schemas import DependencyRelation


def _report(obligations):
    return DependencyMappingReport(
        section_mappings=[], component_mappings=[],
        cross_section_obligations=obligations, component_dependency_obligations=[],
    )


def _executor(store: ArtifactGraphStore) -> ToolExecutor:
    return ToolExecutor(store, BasicConstraintChecker(store), budget=Budget(cap=1000), max_actions=50)


def test_check_call_through_the_real_executor_passes_and_is_recorded():
    store = ArtifactGraphStore()
    store.add_version("intro", ArtifactKind.SECTION, content="the introduction")
    store.add_version("summary", ArtifactKind.SECTION, content="As shown, \\Cref{sec:intro} holds.")
    obligation = CrossSectionObligation(
        dependency_id="reference_key_result", relation=DependencyRelation.MUST_REFERENCE,
        description="summary must cite intro", source_section_id="intro", target_section_id="summary",
        status=MappingStatus.MATCHED, matched_source_section_id="intro", matched_target_section_id="summary",
        allowed_labels=["sec:intro"],
    )
    report = _report([obligation])
    provider = DependencyConstraintProvider(report, store)
    target_ref = store.latest("summary").ref

    executor = _executor(store)
    observation = executor.execute(
        CheckCall(artifact=target_ref, constraints=provider(target_ref), context=provider.context_for(target_ref))
    )

    assert observation.ok is True
    assert observation.output.evidence[0].outcome == CheckOutcome.PASS
    # _dispatch_check auto-records evidence -- confirms the budget-cutoff/
    # resume/"never reuse a stale PASS" machinery actually sees this result,
    # not just the direct return value.
    live = store.live_evidence("summary")
    assert len(live) == 1 and live[0].outcome == CheckOutcome.PASS


def test_per_artifact_checker_path_agrees_with_the_whole_document_report():
    """The same fact (does 'summary' satisfy must_reference against
    'intro') must never be answered differently by the two paths: C3's
    direct dependency_verifier.verify_cross_section_obligations (whole-
    document reporting) and C4's BasicConstraintChecker/ConstraintProvider
    path (per-artifact, wired into the controller machinery)."""
    store = ArtifactGraphStore()
    store.add_version("intro", ArtifactKind.SECTION, content="x")
    store.add_version("summary", ArtifactKind.SECTION, content="no citation here")
    obligation = CrossSectionObligation(
        dependency_id="reference_key_result", relation=DependencyRelation.MUST_REFERENCE,
        description="x", source_section_id="intro", target_section_id="summary",
        status=MappingStatus.MATCHED, matched_source_section_id="intro", matched_target_section_id="summary",
        allowed_labels=["sec:intro"],
    )
    report = _report([obligation])

    direct_result = verify_cross_section_obligations(report, {"summary": "no citation here"})

    provider = DependencyConstraintProvider(report, store)
    target_ref = store.latest("summary").ref
    checker_evidence = BasicConstraintChecker(store).check(
        target_ref, provider(target_ref), context=provider.context_for(target_ref)
    )

    assert direct_result.records[0].outcome == CheckOutcome.FAIL
    assert checker_evidence[0].outcome == direct_result.records[0].outcome


def test_a_missing_citation_fail_scopes_repair_to_the_target_section():
    """C4 checklist bullet 1's "누락된 인용은 해당 target... 복구" -- no new
    repair-policy code needed, FixedGraphAwareRepairPolicy already scopes
    any FAIL to its own checked_artifact by kind."""
    store = ArtifactGraphStore()
    store.add_version("intro", ArtifactKind.SECTION, content="x")
    store.add_version("summary", ArtifactKind.SECTION, content="no citation here")
    obligation = CrossSectionObligation(
        dependency_id="reference_key_result", relation=DependencyRelation.MUST_REFERENCE,
        description="x", source_section_id="intro", target_section_id="summary",
        status=MappingStatus.MATCHED, matched_source_section_id="intro", matched_target_section_id="summary",
        allowed_labels=["sec:intro"],
    )
    report = _report([obligation])
    provider = DependencyConstraintProvider(report, store)
    target_ref = store.latest("summary").ref

    executor = _executor(store)
    observation = executor.execute(
        CheckCall(artifact=target_ref, constraints=provider(target_ref), context=provider.context_for(target_ref))
    )
    fail_evidence = observation.output.evidence
    assert fail_evidence[0].outcome == CheckOutcome.FAIL

    document_ref = ArtifactRef(artifact_id="document", version=1, kind=ArtifactKind.DOCUMENT)
    policy = FixedGraphAwareRepairPolicy(document_ref=document_ref)
    actions = policy.decide(fail_evidence, dependency_edges=[], budget=Budget(cap=100))

    assert len(actions) == 1
    assert actions[0].action_type == RepairActionType.REGENERATE_SECTION
    assert actions[0].target_artifacts == [target_ref]
    assert actions[0].triggering_constraint_id == "reference_key_result::summary"


def test_rule_based_controller_decide_now_threads_context_so_the_check_actually_fails():
    """A review found this exact repro: the same missing-citation
    CrossSectionObligation came back FAIL when checked with
    ``provider.context_for(ref)``, but UNKNOWN/``not_checked`` when
    checked with the empty ``context`` ``RuleBasedController.decide()``
    used to pass to its own ``CheckCall``. Drives the REAL
    ``RuleBasedController.decide()`` (not a hand-built ``CheckCall``) to
    confirm the ``repair_operations._context_for_check`` fix actually
    reaches this controller path end to end."""
    store = ArtifactGraphStore()
    store.add_version("intro", ArtifactKind.SECTION, content="x")
    store.add_version("summary", ArtifactKind.SECTION, content="no citation here")
    obligation = CrossSectionObligation(
        dependency_id="reference_key_result", relation=DependencyRelation.MUST_REFERENCE,
        description="summary must cite intro", source_section_id="intro", target_section_id="summary",
        status=MappingStatus.MATCHED, matched_source_section_id="intro", matched_target_section_id="summary",
        allowed_labels=["sec:intro"],
    )
    provider = DependencyConstraintProvider(_report([obligation]), store)
    target_ref = store.latest("summary").ref

    policy = RuleBasedController(target_ref, provider, request_id="req1", output_dir="/tmp/x")
    state = ControllerState(
        document_ref=target_ref, checkpoint=store.save_checkpoint(), budget=Budget(cap=1000.0),
        current_evidence=[], tracked_artifact_ids=set(), compiled=False,
    )
    call = policy.decide(state, store, _executor(store))

    assert call.action.value == "check"
    assert call.context == provider.context_for(target_ref)  # not {}

    observation = _executor(store).execute(call)
    assert observation.output.evidence[0].outcome == CheckOutcome.FAIL  # not UNKNOWN/NOT_CHECKED


def test_unresolved_dependency_evidence_survives_checkpoint_and_resume():
    """C4 checklist bullet 2's "budget cutoff·resume 뒤에도 미해결
    dependency를 유지하고, 재검증 없이 이전 PASS를 재사용하지 않는다" --
    exercised against the real Checkpoint/restore_checkpoint mechanism,
    not asserted from this module's own code."""
    store = ArtifactGraphStore()
    store.add_version("intro", ArtifactKind.SECTION, content="x")
    store.add_version("summary", ArtifactKind.SECTION, content="no citation here")
    obligations = [
        CrossSectionObligation(
            dependency_id="reference_key_result", relation=DependencyRelation.MUST_REFERENCE,
            description="x", source_section_id="intro", target_section_id="summary",
            status=MappingStatus.MATCHED, matched_source_section_id="intro", matched_target_section_id="summary",
            allowed_labels=["sec:intro"],
        ),
        CrossSectionObligation(
            # Still UNKNOWN (Fix B not implemented) -- must remain visibly
            # unresolved after resume, never silently dropped.
            dependency_id="reuse_core_definition", relation=DependencyRelation.MUST_REUSE,
            description="x", source_section_id="intro", target_section_id="summary",
            status=MappingStatus.MATCHED, matched_source_section_id="intro", matched_target_section_id="summary",
        ),
    ]
    report = _report(obligations)
    provider = DependencyConstraintProvider(report, store)
    target_ref = store.latest("summary").ref

    executor = _executor(store)
    executor.execute(
        CheckCall(artifact=target_ref, constraints=provider(target_ref), context=provider.context_for(target_ref))
    )

    before = {e.constraint_id: e.outcome for e in store.live_evidence("summary")}
    assert before["reference_key_result::summary"] == CheckOutcome.FAIL
    assert before["reuse_core_definition::summary"] == CheckOutcome.UNKNOWN

    checkpoint = store.save_checkpoint()
    restored = ArtifactGraphStore.from_checkpoint(checkpoint, branch_id="resumed")

    after = {e.constraint_id: e.outcome for e in restored.live_evidence("summary")}
    assert after == before  # both the FAIL and the still-UNKNOWN survive resume unchanged

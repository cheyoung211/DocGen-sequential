"""Coverage for src/core/adapters.py (Stage 1 of the "Revision 2026.10.15"
plan): proves today's pipeline is expressible in the Stage 1 interfaces
without changing its behavior."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.agents.text_agent import TextAgent
from src.common.schemas import ContentBlock, PlannerOutput, SemanticBlockType, TextAgentOutput
from src.common.state import (
    DocumentBlueprint,
    DocumentGraph,
    DocumentKind,
    DocumentNode,
    DocumentSpec,
    LayoutBlock,
    LayoutBlockKind,
    NodeStatus,
    NodeType,
    SectionLayout,
)
from src.core.adapters import (
    BlindRetryRepairPolicy,
    BlockPatchProducer,
    ContractOnlyFinalEvaluator,
    PlannerAgentProducer,
    TextAgentProducer,
    TextAgentValidatorAdapter,
    TodayFixedRepairPolicyAdapter,
)
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    CheckOutcome,
    Constraint,
    ConstraintKind,
    Evidence,
    ProductionStatus,
    RepairActionType,
    UnknownReason,
)
from src.dataset.schemas import (
    BenchmarkItem,
    ComponentType,
    ConstraintType,
    CrossSectionDependency,
    Difficulty,
    Domain,
    DocumentType,
    DependencyRelation,
    EvaluationContract,
    HardConstraint,
    LayoutRequirements,
    RequiredComponent,
    RequiredSection,
    SourceProvenanceEntry,
    TargetLength,
)


def _benchmark_item() -> BenchmarkItem:
    # Mirrors test/evaluation/test_contract_satisfaction.py's fixture shape
    # (BenchmarkItem requires at least one required_component/hard_constraint/
    # cross_section_dependency) -- trimmed to the minimum valid set, since
    # this test only needs evaluate_contract() to run, not to exercise every
    # contract category.
    return BenchmarkItem(
        sample_id="test_item_0001",
        domain=Domain.MATHEMATICS,
        subdomain="real_analysis",
        document_type=DocumentType.LECTURE_NOTE,
        difficulty=Difficulty.BASIC,
        title="Convergence",
        overview="A short note.",
        audience="students",
        document_purpose="teach",
        target_length=TargetLength(
            target_pages=1, min_words=100, max_words=500, min_sections=1, max_sections=3
        ),
        required_sections=[
            RequiredSection(section_id="introduction", title="Introduction", purpose="intro", target_words=50),
        ],
        required_components=[
            RequiredComponent(
                component_id="definition_core", type=ComponentType.DEFINITION, placement="introduction",
                description="Define convergence.",
            ),
        ],
        hard_constraints=[
            HardConstraint(
                constraint_id="terminology_consistency",
                type=ConstraintType.TERMINOLOGY_CONSISTENCY,
                description="d",
            ),
        ],
        cross_section_dependencies=[
            CrossSectionDependency(
                dependency_id="dep_1", source_section="introduction", target_sections=["introduction"],
                relation=DependencyRelation.MUST_REUSE, description="d",
            ),
        ],
        layout_requirements=LayoutRequirements(preferred_layout_profile="technical_lecture_note"),
        evaluation_contract=EvaluationContract(),
        natural_language_instruction="Write about convergence.",
        source_provenance=[SourceProvenanceEntry(source_id="standard_curriculum", usage="topic seed")],
    )


def _write_generated_output(project_dir: Path) -> None:
    blueprint = DocumentBlueprint(
        document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Convergence"),
        document_class="article",
        sections=[
            SectionLayout(
                section_id="sec_intro",
                title="Introduction",
                blocks=[LayoutBlock(id="def1", kind=LayoutBlockKind.DEFINITION)],
            ),
        ],
    )
    plan = PlannerOutput(
        request_id="test-run",
        nodes_to_create=[{"id": "sec_intro", "type": "SECTION", "title": "Introduction"}],
        hierarchy_edges=[],
        global_context="",
        blueprint=blueprint,
    )
    (project_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    sections_dir = project_dir / "sections"
    sections_dir.mkdir(parents=True)
    output = TextAgentOutput(
        request_id="test-run",
        section_id="sec_intro",
        blocks=[
            ContentBlock(
                block_id="def1",
                type=SemanticBlockType.DEFINITION,
                content="A sequence exhibits convergence as n grows.",
            )
        ],
    )
    (sections_dir / "sec_intro.blocks.json").write_text(output.model_dump_json(indent=2), encoding="utf-8")


def test_contract_only_final_evaluator_matches_evaluate_contract_directly() -> None:
    from src.evaluation.metrics.contract_satisfaction import evaluate_contract

    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        _write_generated_output(project_dir)
        benchmark_item = _benchmark_item()

        direct = evaluate_contract(project_dir, benchmark_item)
        evaluator = ContractOnlyFinalEvaluator(project_dir)
        result = evaluator.evaluate(final_artifacts=[], benchmark_contract=benchmark_item)

    assert result.quality.overall == direct.overall
    assert result.quality.category_scores["required_sections"] == direct.required_sections
    # The two forward-looking axes are explicitly unimplemented, not
    # silently defaulted to a passing score.
    assert result.content_preservation.preserved_fraction is None
    assert result.content_preservation.not_implemented_reason is not None
    assert result.verifier_precision_recall.sample_size == 0


# ---------------------------------------------------------------------------
# Producer: generation failure must be visible in the return value.
# ---------------------------------------------------------------------------


class _StubTextAgent:
    """Duck-typed stand-in for TextAgent -- TextAgentProducer only ever
    calls .process_node(...), so a stub controlling exactly what that
    returns/raises is enough to test the adapter's own logic without
    driving a real (slow, LLM-backed) retry loop."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def process_node(self, node_id, graph, request_id, **kwargs):
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class _StubPlannerAgent:
    def __init__(self, result: Any) -> None:
        self._result = result

    def generate_plan(self, planner_input, **kwargs):
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


def _graph_with_one_section() -> DocumentGraph:
    graph = DocumentGraph()
    graph.add_node(DocumentNode(id="sec_intro", type=NodeType.SECTION, title="Introduction"))
    return graph


def test_text_agent_producer_surfaces_exhausted_retries_as_failure() -> None:
    """TextAgent.process_node returns None (not an exception) once retries
    are exhausted, after recording the real reason on the graph node
    (text_agent.py:818-821) -- the producer must read that back, not just
    report "None was returned" with no explanation, and must not return a
    look-alike-successful ArtifactRef."""

    graph = _graph_with_one_section()
    graph.update_node_status(
        "sec_intro", NodeStatus.ERROR, error="3 attempt(s) exhausted: schema_validation_error"
    )
    producer = TextAgentProducer(_StubTextAgent(None))

    result = producer.produce("sec_intro", {"graph": graph, "request_id": "req-1"})

    assert result.status == ProductionStatus.FAILED
    assert result.artifact is None
    assert result.error == "3 attempt(s) exhausted: schema_validation_error"


def test_text_agent_producer_surfaces_a_raised_precondition_failure() -> None:
    graph = DocumentGraph()  # no "missing_node" node registered at all
    producer = TextAgentProducer(_StubTextAgent(ValueError("Node 'missing_node' not found in graph.")))

    result = producer.produce("missing_node", {"graph": graph, "request_id": "req-1"})

    assert result.status == ProductionStatus.FAILED
    assert result.artifact is None
    assert "missing_node" in result.error


def test_text_agent_producer_reports_success_with_a_real_artifact() -> None:
    graph = _graph_with_one_section()
    stub_output = object()  # stand-in for a real TextAgentOutput
    producer = TextAgentProducer(_StubTextAgent(stub_output))

    result = producer.produce("sec_intro", {"graph": graph, "request_id": "req-1"})

    assert result.status == ProductionStatus.SUCCESS
    assert result.artifact.artifact_id == "sec_intro"
    assert producer.last_output is stub_output


def test_planner_agent_producer_surfaces_exhausted_retries_as_failure() -> None:
    """PlannerAgent.generate_plan raises (does not return a sentinel) once
    retries are exhausted (planner.py:714-716) -- the opposite failure
    mechanism from TextAgent, normalized into the same ProductionResult
    shape."""
    producer = PlannerAgentProducer(_StubPlannerAgent(ValueError("3 attempt(s) exhausted")))

    result = producer.produce("doc", {"planner_input": object()})

    assert result.status == ProductionStatus.FAILED
    assert result.artifact is None
    assert result.error == "3 attempt(s) exhausted"


def test_planner_agent_producer_reports_success_with_a_real_artifact() -> None:
    stub_plan = object()
    producer = PlannerAgentProducer(_StubPlannerAgent(stub_plan))

    result = producer.produce("doc", {"planner_input": object()})

    assert result.status == ProductionStatus.SUCCESS
    assert result.artifact.artifact_id == "doc"
    assert producer.last_output is stub_plan


# ---------------------------------------------------------------------------
# BlockPatchProducer (Stage 4): wraps TextAgent.regenerate_blocks, giving
# PATCH_BLOCK a real backing producer.
# ---------------------------------------------------------------------------


class _StubTextAgentForPatch:
    """Duck-typed stand-in -- BlockPatchProducer only ever calls
    .regenerate_blocks(...); records the call's arguments so a test can
    assert node_id/block_id/error_message were threaded through correctly,
    not just that *some* call happened."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list = []

    def regenerate_blocks(self, node_id, block_id, error_message, graph, request_id, **kwargs):
        self.calls.append(
            {
                "node_id": node_id,
                "block_id": block_id,
                "error_message": error_message,
                "request_id": request_id,
                **kwargs,
            }
        )
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


def test_block_patch_producer_surfaces_exhausted_retries_as_failure() -> None:
    """regenerate_blocks shares process_node's None-on-exhausted-retries
    failure mode (text_agent.py) -- same normalization as
    TextAgentProducer, read back off graph.nodes[node_id].last_error."""
    graph = _graph_with_one_section()
    graph.update_node_status(
        "sec_intro", NodeStatus.ERROR, error="2 attempt(s) exhausted: compile_error"
    )
    producer = BlockPatchProducer(_StubTextAgentForPatch(None))

    result = producer.produce(
        "def1",
        {"node_id": "sec_intro", "graph": graph, "request_id": "req-1", "error_message": "bad bracket"},
    )

    assert result.status == ProductionStatus.FAILED
    assert result.artifact is None
    assert result.error == "2 attempt(s) exhausted: compile_error"


def test_block_patch_producer_surfaces_a_raised_precondition_failure() -> None:
    graph = DocumentGraph()  # no "missing_node" registered at all
    producer = BlockPatchProducer(
        _StubTextAgentForPatch(ValueError("Node 'missing_node' not found in graph."))
    )

    result = producer.produce(
        "def1",
        {"node_id": "missing_node", "graph": graph, "request_id": "req-1", "error_message": "x"},
    )

    assert result.status == ProductionStatus.FAILED
    assert result.artifact is None
    assert "missing_node" in result.error


def test_block_patch_producer_reports_success_with_a_real_artifact() -> None:
    graph = _graph_with_one_section()
    stub_output = object()
    producer = BlockPatchProducer(_StubTextAgentForPatch(stub_output))

    result = producer.produce(
        "def1",
        {"node_id": "sec_intro", "graph": graph, "request_id": "req-1", "error_message": "bad bracket"},
    )

    assert result.status == ProductionStatus.SUCCESS
    assert result.artifact.artifact_id == "def1"
    assert result.artifact.kind == ArtifactKind.BLOCK
    assert producer.last_output is stub_output


def test_block_patch_producer_requires_error_message_in_context() -> None:
    """regenerate_blocks has no parameter to omit error_message (unlike
    process_node) -- see BlockPatchProducer's docstring on the resulting
    "blind patch" gap. A caller must make an explicit choice, not have one
    silently manufactured."""
    graph = _graph_with_one_section()
    producer = BlockPatchProducer(_StubTextAgentForPatch(object()))

    with pytest.raises(ValueError, match="error_message"):
        producer.produce("def1", {"node_id": "sec_intro", "graph": graph, "request_id": "req-1"})


def test_block_patch_producer_forwards_node_id_and_block_id_separately() -> None:
    """artifact_id is the block id; node_id (the containing section) comes
    from context and is a distinct, required field -- regenerate_blocks
    needs both as separate parameters, unlike TextAgentProducer where
    artifact_id alone is the section."""
    graph = _graph_with_one_section()
    stub_agent = _StubTextAgentForPatch(object())
    producer = BlockPatchProducer(stub_agent)

    producer.produce(
        "def1",
        {"node_id": "sec_intro", "graph": graph, "request_id": "req-1", "error_message": "msg"},
    )

    assert len(stub_agent.calls) == 1
    assert stub_agent.calls[0]["node_id"] == "sec_intro"
    assert stub_agent.calls[0]["block_id"] == "def1"
    assert stub_agent.calls[0]["error_message"] == "msg"


# ---------------------------------------------------------------------------
# Validator: config mutability and constraint-to-evidence mapping.
# ---------------------------------------------------------------------------


def _constraint(constraint_id: str) -> Constraint:
    return Constraint(constraint_id=constraint_id, kind=ConstraintKind.SCHEMA_CONTRACT, description=constraint_id)


def test_config_snapshot_is_read_only_set_enabled_propagates_to_real_checks() -> None:
    text_agent = TextAgent(llm_client=MagicMock())
    adapter = TextAgentValidatorAdapter(text_agent)

    snapshot = adapter.config
    snapshot["forbidden_environment"] = False
    # Mutating the returned snapshot must have no effect on the real,
    # live dict _validate_one_block actually reads.
    assert text_agent.enabled_validators["forbidden_environment"] is True

    adapter.set_enabled("forbidden_environment", False)
    assert text_agent.enabled_validators["forbidden_environment"] is False
    assert adapter.config["forbidden_environment"] is False

    with pytest.raises(ValueError):
        adapter.set_enabled("not_a_real_code", False)


def test_validator_adapter_maps_every_requested_constraint_to_evidence() -> None:
    text_agent = TextAgent(llm_client=MagicMock())
    adapter = TextAgentValidatorAdapter(text_agent)
    adapter.set_enabled("plain_title", False)

    artifact = ArtifactRef(artifact_id="def1", kind=ArtifactKind.BLOCK)
    layout_block = LayoutBlock(id="def1", kind=LayoutBlockKind.DEFINITION)
    # Correctly-typed, non-empty DEFINITION block (so the gated checks
    # below are actually reached) that violates forbidden_environment via
    # a document-structuring command the block-level contract never allows.
    content_block = ContentBlock(
        block_id="def1", type=SemanticBlockType.DEFINITION, content="\\section{Not allowed}"
    )

    requested = [
        _constraint("forbidden_environment"),       # violated -> FAIL
        _constraint("manual_label"),                # checked, not violated -> PASS
        _constraint("plain_title"),                 # trigger disabled -> UNKNOWN/NOT_CHECKED
        _constraint("equation_shape"),               # wrong kind for this check -> UNKNOWN/NOT_REPRESENTABLE
        _constraint("cross_section_dependency"),     # not a code this adapter knows at all -> UNKNOWN/NOT_REPRESENTABLE
    ]
    evidence = {
        e.constraint_id: e
        for e in adapter.check(artifact, requested, context={"content_block": content_block, "layout_block": layout_block})
    }

    assert len(evidence) == len(requested)  # nothing requested was silently dropped
    assert evidence["forbidden_environment"].outcome == CheckOutcome.FAIL
    assert evidence["manual_label"].outcome == CheckOutcome.PASS
    assert evidence["plain_title"].outcome == CheckOutcome.UNKNOWN
    assert evidence["plain_title"].unknown_reason == UnknownReason.NOT_CHECKED
    assert evidence["equation_shape"].outcome == CheckOutcome.UNKNOWN
    assert evidence["equation_shape"].unknown_reason == UnknownReason.NOT_REPRESENTABLE
    assert evidence["cross_section_dependency"].outcome == CheckOutcome.UNKNOWN
    assert evidence["cross_section_dependency"].unknown_reason == UnknownReason.NOT_REPRESENTABLE


def test_validator_adapter_reports_not_checked_when_short_circuited() -> None:
    """A type_mismatch failure makes _validate_one_block return immediately
    (text_agent.py:1020-1032) -- every gated check below that point must
    come back UNKNOWN/NOT_CHECKED, never a silent PASS just because the
    code that would have caught a real violation never ran."""
    text_agent = TextAgent(llm_client=MagicMock())
    adapter = TextAgentValidatorAdapter(text_agent)
    artifact = ArtifactRef(artifact_id="def1", kind=ArtifactKind.BLOCK)
    layout_block = LayoutBlock(id="def1", kind=LayoutBlockKind.DEFINITION)
    mismatched_block = ContentBlock(block_id="def1", type=SemanticBlockType.PARAGRAPH, content="wrong type")

    requested = [_constraint("type_mismatch"), _constraint("forbidden_environment")]
    evidence = {
        e.constraint_id: e
        for e in adapter.check(artifact, requested, context={"content_block": mismatched_block, "layout_block": layout_block})
    }

    assert evidence["type_mismatch"].outcome == CheckOutcome.FAIL
    assert evidence["forbidden_environment"].outcome == CheckOutcome.UNKNOWN
    assert evidence["forbidden_environment"].unknown_reason == UnknownReason.NOT_CHECKED


def test_validator_adapter_reports_empty_content_as_not_checked_when_type_mismatch_fires_first() -> None:
    """Regression for a real bug: a block with the wrong type AND empty
    content used to report empty_content as PASS, because
    _validate_one_block returns immediately on type_mismatch
    (text_agent.py:1020-1032) and never reaches the empty-content check at
    all -- "no empty_content error in the result" was wrongly read as
    "checked, and found not empty" instead of "never checked"."""
    text_agent = TextAgent(llm_client=MagicMock())
    adapter = TextAgentValidatorAdapter(text_agent)
    artifact = ArtifactRef(artifact_id="def1", kind=ArtifactKind.BLOCK)
    layout_block = LayoutBlock(id="def1", kind=LayoutBlockKind.DEFINITION)
    wrong_type_and_empty = ContentBlock(block_id="def1", type=SemanticBlockType.PARAGRAPH, content="")

    requested = [_constraint("type_mismatch"), _constraint("empty_content")]
    evidence = {
        e.constraint_id: e
        for e in adapter.check(
            artifact, requested, context={"content_block": wrong_type_and_empty, "layout_block": layout_block}
        )
    }

    assert evidence["type_mismatch"].outcome == CheckOutcome.FAIL
    assert evidence["empty_content"].outcome == CheckOutcome.UNKNOWN
    assert evidence["empty_content"].unknown_reason == UnknownReason.NOT_CHECKED


def test_validator_adapter_short_circuits_on_empty_content_too() -> None:
    """The second structural check short-circuits everything after it
    (text_agent.py:1033-1037), the same way type_mismatch does -- and,
    when type actually matches, empty_content itself must be a real
    FAIL, not swallowed by the same short-circuit logic that protects it."""
    text_agent = TextAgent(llm_client=MagicMock())
    adapter = TextAgentValidatorAdapter(text_agent)
    artifact = ArtifactRef(artifact_id="def1", kind=ArtifactKind.BLOCK)
    layout_block = LayoutBlock(id="def1", kind=LayoutBlockKind.DEFINITION)
    empty_block = ContentBlock(block_id="def1", type=SemanticBlockType.DEFINITION, content="   ")

    requested = [_constraint("type_mismatch"), _constraint("empty_content"), _constraint("forbidden_environment")]
    evidence = {
        e.constraint_id: e
        for e in adapter.check(artifact, requested, context={"content_block": empty_block, "layout_block": layout_block})
    }

    assert evidence["type_mismatch"].outcome == CheckOutcome.PASS  # type really did match
    assert evidence["empty_content"].outcome == CheckOutcome.FAIL  # really is empty, really checked
    assert evidence["forbidden_environment"].outcome == CheckOutcome.UNKNOWN
    assert evidence["forbidden_environment"].unknown_reason == UnknownReason.NOT_CHECKED


def test_validator_adapter_prioritizes_kind_mismatch_over_short_circuit() -> None:
    """equation_shape is both inapplicable to a DEFINITION block AND
    short-circuited by type_mismatch here -- the kind mismatch is the more
    fundamental, context-independent reason (it would never have run
    regardless of the short-circuit) and must take priority."""
    text_agent = TextAgent(llm_client=MagicMock())
    adapter = TextAgentValidatorAdapter(text_agent)
    artifact = ArtifactRef(artifact_id="def1", kind=ArtifactKind.BLOCK)
    layout_block = LayoutBlock(id="def1", kind=LayoutBlockKind.DEFINITION)
    mismatched_block = ContentBlock(block_id="def1", type=SemanticBlockType.PARAGRAPH, content="wrong type")

    evidence = adapter.check(
        artifact, [_constraint("equation_shape")],
        context={"content_block": mismatched_block, "layout_block": layout_block},
    )[0]

    assert evidence.outcome == CheckOutcome.UNKNOWN
    assert evidence.unknown_reason == UnknownReason.NOT_REPRESENTABLE


def test_validator_adapter_returns_nothing_when_nothing_was_requested() -> None:
    text_agent = TextAgent(llm_client=MagicMock())
    adapter = TextAgentValidatorAdapter(text_agent)
    artifact = ArtifactRef(artifact_id="def1", kind=ArtifactKind.BLOCK)
    layout_block = LayoutBlock(id="def1", kind=LayoutBlockKind.DEFINITION)
    content_block = ContentBlock(block_id="def1", type=SemanticBlockType.DEFINITION, content="fine")

    assert adapter.check(artifact, [], context={"content_block": content_block, "layout_block": layout_block}) == []


# ---------------------------------------------------------------------------
# RepairPolicy: grouping, deferred attempt-counting, and shared scope rules.
# ---------------------------------------------------------------------------


def test_today_fixed_policy_only_acts_on_the_first_failure_per_call() -> None:
    first = Evidence(
        constraint_id="c1", outcome=CheckOutcome.FAIL,
        checked_artifact=ArtifactRef(artifact_id="def1", kind=ArtifactKind.BLOCK),
    )
    second = Evidence(
        constraint_id="c2", outcome=CheckOutcome.FAIL,
        checked_artifact=ArtifactRef(artifact_id="def2", kind=ArtifactKind.BLOCK),
    )

    # Mirrors regenerable[0] in the real _repair_and_recompile: one location
    # per round, not a batch of every current failure.
    actions = TodayFixedRepairPolicyAdapter().decide([first, second], [], Budget(cap=1000.0))

    assert len(actions) == 1
    assert actions[0].target_artifacts[0].artifact_id == "def1"


def test_today_fixed_policy_only_gives_feedback_on_the_block_branch() -> None:
    """Regression for a real bug: the real pipeline's section-repair call
    (TextAgent.process_node, run_full_pipeline.py:508-511) is never handed
    the compiler's error message at all -- only the block-repair call
    (TextAgent.regenerate_blocks, run_full_pipeline.py:500-504) is, via its
    explicit error_message parameter. An earlier version of this adapter
    set use_diagnostic_feedback=True for both branches."""
    block_evidence = Evidence(
        constraint_id="forbidden_environment", outcome=CheckOutcome.FAIL,
        checked_artifact=ArtifactRef(artifact_id="def1", kind=ArtifactKind.BLOCK),
        detail="verbatim environment is not allowed here",
    )
    section_evidence = Evidence(
        constraint_id="forbidden_environment", outcome=CheckOutcome.FAIL,
        checked_artifact=ArtifactRef(artifact_id="sec_intro", kind=ArtifactKind.SECTION),
        detail="verbatim environment is not allowed here",
    )
    budget = Budget(cap=1000.0)
    today = TodayFixedRepairPolicyAdapter()

    block_action = today.decide([block_evidence], [], budget)[0]
    section_action = today.decide([section_evidence], [], budget)[0]

    assert block_action.action_type == RepairActionType.PATCH_BLOCK
    assert block_action.use_diagnostic_feedback is True
    assert block_action.feedback_text == "verbatim environment is not allowed here"

    assert section_action.action_type == RepairActionType.REGENERATE_SECTION
    assert section_action.use_diagnostic_feedback is False
    assert section_action.feedback_text is None


def test_blind_retry_shares_the_wrapped_policys_scope_rule() -> None:
    """Proves the fix for the earlier hardcoded-scope design: wrapping
    TodayFixedRepairPolicyAdapter must pick the exact same action_type (and,
    on the block branch, inherit the same real feedback-text content before
    it gets stripped) that policy would alone -- for both a block- and a
    section-level failure, evaluated one at a time so each gets its own
    decision (see test_blind_retry_matches_the_wrapped_policys_cardinality_
    exactly for what happens when both are offered in the same call)."""
    budget = Budget(cap=1000.0)
    today = TodayFixedRepairPolicyAdapter()
    blind = BlindRetryRepairPolicy(wrapped=TodayFixedRepairPolicyAdapter())

    for evidence, expected_scope in (
        (
            Evidence(
                constraint_id="forbidden_environment", outcome=CheckOutcome.FAIL,
                checked_artifact=ArtifactRef(artifact_id="def1", kind=ArtifactKind.BLOCK),
                detail="verbatim environment is not allowed here",
            ),
            RepairActionType.PATCH_BLOCK,
        ),
        (
            Evidence(
                constraint_id="forbidden_environment", outcome=CheckOutcome.FAIL,
                checked_artifact=ArtifactRef(artifact_id="sec_intro", kind=ArtifactKind.SECTION),
                detail="verbatim environment is not allowed here",
            ),
            RepairActionType.REGENERATE_SECTION,
        ),
    ):
        today_action = today.decide([evidence], [], budget)[0]
        blind_action = blind.decide([evidence], [], budget)[0]

        assert today_action.action_type == expected_scope
        assert blind_action.action_type == today_action.action_type
        assert blind_action.target_artifacts == today_action.target_artifacts
        # Feedback is the only field blind retry is allowed to differ on --
        # including when today's own policy already withholds it (the
        # section branch), where stripping it again is simply a no-op.
        assert blind_action.use_diagnostic_feedback is False
        assert blind_action.feedback_text is None


def test_blind_retry_matches_the_wrapped_policys_cardinality_exactly() -> None:
    """Regression for a real bug: an earlier version of BlindRetryRepairPolicy
    called wrapped.decide([item], ...) once per distinct failing artifact,
    so for 2 distinct failing artifacts TodayFixedRepairPolicyAdapter alone
    returned 1 action (only the first, per regenerable[0]) while
    BlindRetryRepairPolicy wrapping it returned 2 -- not actually "the same
    policy, minus feedback". Fixed by calling wrapped.decide() exactly ONCE
    on one (retry-cap-restricted) evidence list and only stripping feedback
    from whatever it returns, so cardinality comes through unchanged."""
    first = Evidence(
        constraint_id="c1", outcome=CheckOutcome.FAIL,
        checked_artifact=ArtifactRef(artifact_id="def1", kind=ArtifactKind.BLOCK),
    )
    second = Evidence(
        constraint_id="c2", outcome=CheckOutcome.FAIL,
        checked_artifact=ArtifactRef(artifact_id="def2", kind=ArtifactKind.BLOCK),
    )
    budget = Budget(cap=1000.0)

    today_actions = TodayFixedRepairPolicyAdapter().decide([first, second], [], budget)
    blind_actions = BlindRetryRepairPolicy(wrapped=TodayFixedRepairPolicyAdapter()).decide(
        [first, second], [], budget
    )

    assert len(today_actions) == len(blind_actions) == 1
    assert (
        today_actions[0].target_artifacts[0].artifact_id
        == blind_actions[0].target_artifacts[0].artifact_id
        == "def1"
    )


def test_blind_retry_only_counts_an_attempt_once_execution_is_recorded() -> None:
    evidence = Evidence(
        constraint_id="c1", outcome=CheckOutcome.FAIL,
        checked_artifact=ArtifactRef(artifact_id="def1", kind=ArtifactKind.BLOCK),
    )
    policy = BlindRetryRepairPolicy(wrapped=TodayFixedRepairPolicyAdapter(), max_retries_per_artifact=1)
    budget = Budget(cap=1000.0)

    # decide() alone, called repeatedly, must never exhaust the retry cap --
    # nothing was ever actually executed yet.
    for _ in range(5):
        actions = policy.decide([evidence], [], budget)
        assert len(actions) == 1

    # Only once the executor signals the action actually started does the
    # cap advance.
    policy.record_attempt_started(actions[0])
    assert policy.decide([evidence], [], budget) == []


def test_blind_retry_respects_budget_exhaustion() -> None:
    evidence = Evidence(
        constraint_id="c1", outcome=CheckOutcome.FAIL,
        checked_artifact=ArtifactRef(artifact_id="def1", kind=ArtifactKind.BLOCK),
    )
    policy = BlindRetryRepairPolicy(wrapped=TodayFixedRepairPolicyAdapter())
    exhausted_budget = Budget(cap=10.0, spent=10.0)
    assert policy.decide([evidence], [], exhausted_budget) == []

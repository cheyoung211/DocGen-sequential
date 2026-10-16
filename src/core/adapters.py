"""Wrappers that express today's pipeline in the Stage 1 interfaces,
without changing its behavior (Stage 1 of the "Revision 2026.10.15" plan).

Nothing here is wired into ``scripts/run_full_pipeline.py`` -- every class
below delegates to an existing agent method (or, where that method is
inseparably bundled with real I/O side effects, faithfully reproduces just
its *decision rule*, documented per class) to prove the interfaces in
``interfaces.py`` can actually represent the existing setup. Swapping the
real orchestrator over to call through these interfaces is Stage 5's job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.planner import PlannerAgent
from src.agents.text_agent import TextAgent
from src.common.schemas import ContentBlock, PlannerOutput
from src.common.state import DocumentGraph, LayoutBlock
from src.core.interfaces import ValidatorConfig
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    CheckOutcome,
    Constraint,
    ContentPreservationScore,
    Evidence,
    FinalEvaluationResult,
    QualityScore,
    RepairAction,
    RepairActionType,
    VerifierPrecisionRecall,
)
from src.dataset.schemas import BenchmarkItem
from src.evaluation.metrics.contract_satisfaction import evaluate_contract


class PlannerAgentProducer:
    """Wraps ``PlannerAgent.generate_plan`` as a ``Producer``.

    ``context`` must carry ``"planner_input"`` (a ``PlannerInput``); the
    real ``PlannerOutput`` (far richer than a single ``ArtifactRef``) is
    kept on ``self.last_output`` for a caller that needs it, since the
    ``Producer`` Protocol's return type is deliberately a bare
    ``ArtifactRef`` -- generality for Stage 2+'s controller, which only
    needs to know *that* an artifact was produced, not its full content.
    """

    def __init__(self, planner_agent: PlannerAgent) -> None:
        self._planner = planner_agent
        self.last_output: Optional[PlannerOutput] = None

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ArtifactRef:
        self.last_output = self._planner.generate_plan(
            context["planner_input"],
            event_logger=context.get("event_logger"),
            usage_sink=context.get("usage_sink"),
        )
        return ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.DOCUMENT)


class TextAgentProducer:
    """Wraps ``TextAgent.process_node`` as a ``Producer``.

    ``context`` must carry ``"graph"`` (``DocumentGraph``) and
    ``"request_id"``; ``output_dir``/``event_logger``/``usage_sink``/
    ``outer_iteration`` are forwarded when present, matching
    ``process_node``'s own optional parameters exactly.
    """

    def __init__(self, text_agent: TextAgent) -> None:
        self._text_agent = text_agent
        self.last_output: Any = None

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ArtifactRef:
        graph: DocumentGraph = context["graph"]
        self.last_output = self._text_agent.process_node(
            artifact_id,
            graph,
            context["request_id"],
            output_dir=context.get("output_dir", "outputs/generations"),
            event_logger=context.get("event_logger"),
            usage_sink=context.get("usage_sink"),
            outer_iteration=context.get("outer_iteration"),
        )
        return ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.SECTION)


class TextAgentValidatorAdapter:
    """Wraps ``TextAgent._validate_one_block`` as a ``Validator``.

    This is the real per-block checker, unmodified -- the one
    ``DEFAULT_ENABLED_VALIDATORS`` (``src/agents/text_agent.py:89-99``)
    already gates. ``.config`` exposes that same dict as the *trigger* axis
    (``ValidatorConfig``); ``check()`` requires ``context={"content_block":
    ..., "layout_block": ...}`` since ``_validate_one_block`` needs both
    concrete objects, not just an ``ArtifactRef``.
    """

    def __init__(self, text_agent: TextAgent) -> None:
        self._text_agent = text_agent

    @property
    def config(self) -> ValidatorConfig:
        return ValidatorConfig(self._text_agent.enabled_validators)

    def check(
        self,
        artifact: ArtifactRef,
        constraints: List[Constraint],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Evidence]:
        context = context or {}
        content_block: ContentBlock = context["content_block"]
        layout_block: LayoutBlock = context["layout_block"]
        errors = self._text_agent._validate_one_block(content_block, layout_block)
        if not errors:
            return [
                Evidence(
                    constraint_id="text_agent_block_validation",
                    outcome=CheckOutcome.PASS,
                    checked_artifact=artifact,
                    detail="no violations",
                )
            ]
        return [
            Evidence(
                constraint_id=error.code,
                outcome=CheckOutcome.FAIL,
                checked_artifact=artifact,
                detail=str(error),
            )
            for error in errors
        ]


class BlindRetryRepairPolicy:
    """A repair baseline that does not exist in the pipeline today --
    defined fresh here as the comparison point revision review problem #1
    asks for ("동일한 허용 복구 정책과 비용 상한에서 blind retry와
    비교한다").

    Both axes are held constant regardless of the failing artifact: scope
    is always ``scope`` (default ``REGENERATE_SECTION`` -- the policy does
    not even inspect the artifact's kind to decide), and diagnostic
    feedback is always withheld (the producer is asked to simply try again
    from the same instructions, blind to *why* the previous attempt
    failed). Contrast with ``TodayFixedRepairPolicyAdapter``, whose scope
    *does* depend on the artifact and whose feedback is always on.
    """

    def __init__(
        self,
        scope: RepairActionType = RepairActionType.REGENERATE_SECTION,
        max_retries_per_artifact: int = 3,
    ) -> None:
        self._scope = scope
        self._max_retries = max_retries_per_artifact
        self._attempts: Dict[str, int] = {}

    def decide(
        self, evidence: List[Evidence], dependency_edges: List[Any], budget: Budget
    ) -> List[RepairAction]:
        actions: List[RepairAction] = []
        for item in evidence:
            if item.outcome != CheckOutcome.FAIL or budget.remaining <= 0:
                continue
            artifact_id = item.checked_artifact.artifact_id
            attempts = self._attempts.get(artifact_id, 0)
            if attempts >= self._max_retries:
                continue
            self._attempts[artifact_id] = attempts + 1
            actions.append(
                RepairAction(
                    action_type=self._scope,
                    target_artifacts=[item.checked_artifact],
                    use_diagnostic_feedback=False,
                    feedback_text=None,
                    triggering_constraint_id=item.constraint_id,
                )
            )
        return actions


class TodayFixedRepairPolicyAdapter:
    """Expresses today's actual compile-repair scope rule
    (``DocGenPipeline._repair_and_recompile``, ``scripts/run_full_pipeline.py:425-537``)
    as a ``RepairPolicy.decide()`` call.

    That method is inseparable from real I/O (it recompiles, re-assembles,
    and loops against the filesystem), so this adapter does not call it
    directly -- it reproduces just its *decision rule*: patch the block if
    the compiler's error location resolved to one via
    ``LatexAssembler.locate_block_at_line`` (here: ``checked_artifact.kind
    == ArtifactKind.BLOCK``), otherwise regenerate the whole section
    (``run_full_pipeline.py:491-511``). Diagnostic feedback is always on,
    matching the real method, which always forwards the compiler's error
    message (``location.message``) as the next attempt's instruction.
    """

    def decide(
        self, evidence: List[Evidence], dependency_edges: List[Any], budget: Budget
    ) -> List[RepairAction]:
        actions: List[RepairAction] = []
        for item in evidence:
            if item.outcome != CheckOutcome.FAIL or budget.remaining <= 0:
                continue
            action_type = (
                RepairActionType.PATCH_BLOCK
                if item.checked_artifact.kind == ArtifactKind.BLOCK
                else RepairActionType.REGENERATE_SECTION
            )
            actions.append(
                RepairAction(
                    action_type=action_type,
                    target_artifacts=[item.checked_artifact],
                    use_diagnostic_feedback=True,
                    feedback_text=item.detail,
                    triggering_constraint_id=item.constraint_id,
                )
            )
        return actions


class ContractOnlyFinalEvaluator:
    """Expresses today's only final-score mechanism (``evaluate_contract``
    plus compile success) as a ``FinalEvaluator``.

    Calls ``evaluate_contract`` unmodified and repackages its
    ``ContractResult`` into ``QualityScore``; explicitly leaves
    ``content_preservation``/``verifier_precision_recall`` unset
    (``not_implemented``), since neither exists until Stage 2's versioned
    artifact graph (preservation) or Stage 6's human gold (precision/
    recall). Receives no online ``Evidence``/``VerificationEvent`` stream at
    all -- by construction, matching ``FinalEvaluator``'s docstring.

    ``final_artifacts`` is accepted for ``FinalEvaluator`` conformance but
    unused today: ``evaluate_contract`` reads ``plan.json``/``sections/*``
    straight off disk under ``project_dir`` rather than through artifact
    references, since no artifact store exists yet (Stage 2).
    """

    def __init__(self, project_dir: Path) -> None:
        self._project_dir = project_dir

    def evaluate(
        self, final_artifacts: List[ArtifactRef], benchmark_contract: BenchmarkItem
    ) -> FinalEvaluationResult:
        contract_result = evaluate_contract(self._project_dir, benchmark_contract)
        quality = QualityScore(
            overall=contract_result.overall,
            category_scores={
                "required_sections": contract_result.required_sections,
                "required_components": contract_result.required_components,
                "content_coverage": contract_result.content_coverage,
                "terminology": contract_result.terminology,
                "notation": contract_result.notation,
                "cross_section_dependency": contract_result.cross_section_dependency,
            },
            not_implemented=list(contract_result.not_implemented),
        )
        return FinalEvaluationResult(
            quality=quality,
            content_preservation=ContentPreservationScore(
                not_implemented_reason=(
                    "requires Stage 2's versioned artifact graph "
                    "(before/after comparison across a repair)"
                ),
            ),
            verifier_precision_recall=VerifierPrecisionRecall(sample_size=0),
        )


__all__ = [
    "PlannerAgentProducer",
    "TextAgentProducer",
    "TextAgentValidatorAdapter",
    "BlindRetryRepairPolicy",
    "TodayFixedRepairPolicyAdapter",
    "ContractOnlyFinalEvaluator",
]

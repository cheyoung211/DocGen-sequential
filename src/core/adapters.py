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
from src.agents.text_agent import SemanticBlockError, TextAgent
from src.common.schemas import ContentBlock, PlannerOutput
from src.common.state import DocumentGraph, LayoutBlock, LayoutBlockKind
from src.core.interfaces import RepairPolicy, ValidatorConfig
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    CheckOutcome,
    Constraint,
    ContentPreservationScore,
    Evidence,
    FinalEvaluationResult,
    ProductionResult,
    ProductionStatus,
    QualityScore,
    RepairAction,
    RepairActionType,
    UnknownReason,
    VerifierPrecisionRecall,
)
from src.dataset.schemas import BenchmarkItem
from src.evaluation.metrics.contract_satisfaction import evaluate_contract


class PlannerAgentProducer:
    """Wraps ``PlannerAgent.generate_plan`` as a ``Producer``.

    ``context`` must carry ``"planner_input"`` (a ``PlannerInput``); the
    real ``PlannerOutput`` (far richer than a single ``ArtifactRef``) is
    kept on ``self.last_output`` for a caller that needs it, since the
    ``Producer`` Protocol's return type carries only a reference, not full
    content -- generality for Stage 2+'s controller, which only needs to
    know *that* an artifact was produced, not its full content.

    ``generate_plan`` raises (does not return a sentinel) when it exhausts
    its retries (``src/agents/planner.py:714-716``) -- caught here and
    translated into ``ProductionResult(status=FAILED, error=...)`` so a
    caller never needs to wrap this in its own try/except to learn that
    planning failed.
    """

    def __init__(self, planner_agent: PlannerAgent) -> None:
        self._planner = planner_agent
        self.last_output: Optional[PlannerOutput] = None

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ProductionResult:
        try:
            self.last_output = self._planner.generate_plan(
                context["planner_input"],
                event_logger=context.get("event_logger"),
                usage_sink=context.get("usage_sink"),
            )
        except Exception as exc:
            return ProductionResult(status=ProductionStatus.FAILED, error=str(exc))
        return ProductionResult(
            status=ProductionStatus.SUCCESS,
            artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.DOCUMENT),
        )


class TextAgentProducer:
    """Wraps ``TextAgent.process_node`` as a ``Producer``.

    ``context`` must carry ``"graph"`` (``DocumentGraph``) and
    ``"request_id"``; ``output_dir``/``event_logger``/``usage_sink``/
    ``outer_iteration`` are forwarded when present, matching
    ``process_node``'s own optional parameters exactly.

    ``process_node`` has two different failure modes, both normalized here
    into ``ProductionResult(status=FAILED, ...)``: it raises on a
    precondition failure (node/layout lookup, ``text_agent.py:610-621``),
    and it returns ``None`` -- after marking the graph node ``ERROR`` and
    recording ``node.last_error`` -- when every generation attempt is
    exhausted (``text_agent.py:818-821``). Earlier versions of this adapter
    ignored the return value entirely and always reported success; a
    controller reading only the returned ``ArtifactRef`` had no way to
    notice a generation failure. Both paths are checked here, and the
    second one's error message is read back off ``graph.nodes[artifact_id]
    .last_error`` rather than invented, since that is the pipeline's own
    record of why it failed.
    """

    def __init__(self, text_agent: TextAgent) -> None:
        self._text_agent = text_agent
        self.last_output: Any = None

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ProductionResult:
        graph: DocumentGraph = context["graph"]
        try:
            self.last_output = self._text_agent.process_node(
                artifact_id,
                graph,
                context["request_id"],
                output_dir=context.get("output_dir", "outputs/generations"),
                event_logger=context.get("event_logger"),
                usage_sink=context.get("usage_sink"),
                outer_iteration=context.get("outer_iteration"),
            )
        except Exception as exc:
            return ProductionResult(status=ProductionStatus.FAILED, error=str(exc))

        if self.last_output is None:
            node = graph.nodes.get(artifact_id)
            error = (node.last_error if node is not None else None) or (
                "process_node returned None with no recorded error"
            )
            return ProductionResult(status=ProductionStatus.FAILED, error=error)

        return ProductionResult(
            status=ProductionStatus.SUCCESS,
            artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.SECTION),
        )


# Error codes TextAgent._validate_one_block can produce unconditionally,
# before any enabled_validators-gated check runs -- and whose presence
# means every gated check below was skipped entirely, not merely not
# triggered (text_agent.py:1020-1037's early ``return errors``).
_STRUCTURAL_CODES = frozenset({"type_mismatch", "empty_content"})

# Gated error code -> the DEFAULT_ENABLED_VALIDATORS key that gates it
# (text_agent.py:1046-1143). "figure_content" gates two distinct codes.
_GATE_KEY_BY_CODE: Dict[str, str] = {
    "forbidden_environment": "forbidden_environment",
    "manual_label": "manual_label",
    "table_columns": "table_columns",
    "table_shape": "table_shape",
    "equation_shape": "equation_shape",
    "plain_title": "plain_title",
    "unicode_math_symbol": "unicode_math_symbol",
    "bare_math_notation": "bare_math_notation",
    "figure_asset_mismatch": "figure_content",
    "figure_forbidden_content": "figure_content",
}

# Gated error code -> the LayoutBlockKinds _validate_one_block actually
# runs that check for. Absent from this dict (and from the exclusion dict
# below) means "every kind". See text_agent.py:1046-1143 for each gate's
# own kind condition.
_APPLICABLE_KINDS_BY_CODE: Dict[str, frozenset] = {
    "table_columns": frozenset(
        {LayoutBlockKind.TABLE, LayoutBlockKind.NOTATION_TABLE, LayoutBlockKind.COMPARISON}
    ),
    "table_shape": frozenset({LayoutBlockKind.TABLE, LayoutBlockKind.NOTATION_TABLE}),
    "equation_shape": frozenset({LayoutBlockKind.EQUATION}),
    "figure_asset_mismatch": frozenset({LayoutBlockKind.FIGURE}),
    "figure_forbidden_content": frozenset({LayoutBlockKind.FIGURE}),
}
# bare_math_notation is the one exclusion-shaped rule (every kind except
# EQUATION) rather than an inclusion list -- text_agent.py:1117-1124.
_EXCLUDED_KINDS_BY_CODE: Dict[str, frozenset] = {
    "bare_math_notation": frozenset({LayoutBlockKind.EQUATION}),
}


class TextAgentValidatorAdapter:
    """Wraps ``TextAgent._validate_one_block`` as a ``Validator``.

    This is the real per-block checker, unmodified -- the one
    ``DEFAULT_ENABLED_VALIDATORS`` (``src/agents/text_agent.py:89-99``)
    already gates. ``check()`` requires ``context={"content_block": ...,
    "layout_block": ...}`` since ``_validate_one_block`` needs both
    concrete objects, not just an ``ArtifactRef``.

    ``check()`` answers for every ``Constraint`` passed in, keyed by
    ``constraint_id`` matching one of ``_validate_one_block``'s own error
    codes (e.g. ``"forbidden_environment"``, ``"type_mismatch"``) -- not by
    silently reporting whatever the checker happened to find and ignoring
    ``constraints`` otherwise. Four distinct reasons a constraint does not
    get a plain PASS/FAIL, each a different ``Evidence``:

    - unrecognized ``constraint_id`` -> UNKNOWN / NOT_REPRESENTABLE (this
      adapter has no check by that name at all).
    - recognized, but ``layout_block.kind`` is one ``_validate_one_block``
      never runs that check for (e.g. ``equation_shape`` on a non-equation
      block) -> UNKNOWN / NOT_REPRESENTABLE.
    - recognized and applicable, but this block already failed
      ``type_mismatch``/``empty_content``, so every gated check below that
      short-circuit never ran at all -> UNKNOWN / NOT_CHECKED.
    - recognized, applicable, and reached, but its gate in
      ``enabled_validators`` is off for this run -> UNKNOWN / NOT_CHECKED.
    """

    def __init__(self, text_agent: TextAgent) -> None:
        self._text_agent = text_agent

    @property
    def config(self) -> ValidatorConfig:
        """Read-only snapshot of ``text_agent.enabled_validators`` at the
        time of the call. Mutating the returned dict has **no effect** on
        the wrapped ``TextAgent`` -- it is a copy, not the live dict
        ``_validate_one_block`` reads. To actually change which checks run,
        call ``set_enabled()``."""
        return ValidatorConfig(self._text_agent.enabled_validators)

    def set_enabled(self, code: str, enabled: bool) -> None:
        """Change whether ``code`` is checked, in a way that *does* take
        effect on the next ``check()`` call: mutates
        ``self._text_agent.enabled_validators`` in place -- the same dict
        ``_validate_one_block`` reads live (``text_agent.py:1046`` etc.),
        not a copy."""
        if code not in self._text_agent.enabled_validators:
            raise ValueError(
                f"Unknown validator code: {code!r}. Valid codes: "
                f"{sorted(self._text_agent.enabled_validators)}."
            )
        self._text_agent.enabled_validators[code] = enabled

    def check(
        self,
        artifact: ArtifactRef,
        constraints: List[Constraint],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Evidence]:
        context = context or {}
        content_block: ContentBlock = context["content_block"]
        layout_block: LayoutBlock = context["layout_block"]

        errors_by_code: Dict[str, SemanticBlockError] = {
            error.code: error
            for error in self._text_agent._validate_one_block(content_block, layout_block)
        }
        short_circuited = bool(_STRUCTURAL_CODES & errors_by_code.keys())

        evidence: List[Evidence] = []
        for constraint in constraints:
            code = constraint.constraint_id

            if code in _STRUCTURAL_CODES:
                evidence.append(self._pass_or_fail(code, errors_by_code, artifact))
                continue

            if code not in _GATE_KEY_BY_CODE:
                evidence.append(
                    Evidence(
                        constraint_id=code,
                        outcome=CheckOutcome.UNKNOWN,
                        checked_artifact=artifact,
                        detail="not a check this adapter supports",
                        unknown_reason=UnknownReason.NOT_REPRESENTABLE,
                    )
                )
                continue

            applicable_kinds = _APPLICABLE_KINDS_BY_CODE.get(code)
            excluded_kinds = _EXCLUDED_KINDS_BY_CODE.get(code)
            is_applicable = (
                applicable_kinds is None or layout_block.kind in applicable_kinds
            ) and (excluded_kinds is None or layout_block.kind not in excluded_kinds)
            if not is_applicable:
                evidence.append(
                    Evidence(
                        constraint_id=code,
                        outcome=CheckOutcome.UNKNOWN,
                        checked_artifact=artifact,
                        detail=f"this check does not apply to block kind '{layout_block.kind.value}'",
                        unknown_reason=UnknownReason.NOT_REPRESENTABLE,
                    )
                )
                continue

            if short_circuited:
                evidence.append(
                    Evidence(
                        constraint_id=code,
                        outcome=CheckOutcome.UNKNOWN,
                        checked_artifact=artifact,
                        detail="block failed an earlier structural check, so this check never ran",
                        unknown_reason=UnknownReason.NOT_CHECKED,
                    )
                )
                continue

            gate_key = _GATE_KEY_BY_CODE[code]
            if not self._text_agent.enabled_validators.get(gate_key, True):
                evidence.append(
                    Evidence(
                        constraint_id=code,
                        outcome=CheckOutcome.UNKNOWN,
                        checked_artifact=artifact,
                        detail=f"trigger disabled for this run (enabled_validators[{gate_key!r}]=False)",
                        unknown_reason=UnknownReason.NOT_CHECKED,
                    )
                )
                continue

            evidence.append(self._pass_or_fail(code, errors_by_code, artifact))

        return evidence

    @staticmethod
    def _pass_or_fail(
        code: str, errors_by_code: Dict[str, SemanticBlockError], artifact: ArtifactRef
    ) -> Evidence:
        error = errors_by_code.get(code)
        return Evidence(
            constraint_id=code,
            outcome=CheckOutcome.FAIL if error is not None else CheckOutcome.PASS,
            checked_artifact=artifact,
            detail=str(error) if error is not None else "no violation",
        )


class TodayFixedRepairPolicyAdapter:
    """Expresses today's actual compile-repair scope rule
    (``DocGenPipeline._repair_and_recompile``, ``scripts/run_full_pipeline.py:425-537``)
    as a ``RepairPolicy.decide()`` call.

    That method is inseparable from real I/O (it recompiles, re-assembles,
    and loops against the filesystem), so this adapter does not call it
    directly -- it reproduces just its *decision rule*. Two parts of that
    rule matter, and both are reproduced:

    1. Scope: patch the block if the compiler's error location resolved to
       one via ``LatexAssembler.locate_block_at_line`` (here:
       ``checked_artifact.kind == ArtifactKind.BLOCK``), otherwise
       regenerate the whole section (``run_full_pipeline.py:491-511``).
       Diagnostic feedback is always on, matching the real method, which
       always forwards the compiler's error message as the next attempt's
       instruction.
    2. Cardinality: the real method acts on exactly **one** location per
       round -- ``regenerable[0]`` (``run_full_pipeline.py:466-475``) --
       then recompiles and re-reads the *new* log before deciding again. It
       never batches every currently-failing location into one shot. This
       ``decide()`` reproduces that: given several FAIL ``Evidence``, it
       returns at most one ``RepairAction``, for the first one, in order --
       an executor reproducing the real loop calls ``decide()`` again with
       freshly re-verified Evidence for the next round, the same way
       ``_repair_and_recompile`` re-reads ``compile.log`` every round.

    Carries no retry-cap state of its own: the real method's cap
    (``max_compile_repair_attempts``) lives in the *outer* loop
    (``_repair_and_recompile``'s own ``for attempt in range(...)``),
    external to this decision rule.
    """

    def decide(
        self, evidence: List[Evidence], dependency_edges: List[Any], budget: Budget
    ) -> List[RepairAction]:
        if budget.remaining <= 0:
            return []
        for item in evidence:
            if item.outcome != CheckOutcome.FAIL:
                continue
            action_type = (
                RepairActionType.PATCH_BLOCK
                if item.checked_artifact.kind == ArtifactKind.BLOCK
                else RepairActionType.REGENERATE_SECTION
            )
            return [
                RepairAction(
                    action_type=action_type,
                    target_artifacts=[item.checked_artifact],
                    use_diagnostic_feedback=True,
                    feedback_text=item.detail,
                    triggering_constraint_id=item.constraint_id,
                )
            ]
        return []


class BlindRetryRepairPolicy:
    """Strips diagnostic feedback from another ``RepairPolicy``'s
    decisions -- "비진단 재시도" (non-diagnostic retry), exactly as Notion
    review problem #1 frames the comparison: "동일한 허용 복구 정책과 비용
    상한에서 blind retry와 비교한다" (compare against blind retry under the
    *same admissible repair policy* and cost cap). Wraps ``wrapped`` (e.g.
    ``TodayFixedRepairPolicyAdapter()``) and shares its scope decisions by
    construction, instead of inventing an independent scope rule -- an
    earlier version of this class picked its own hardcoded constant scope,
    which was not actually comparable to anything else on a single,
    isolated axis and has been replaced by this wrapper form.

    Concretely: ``BlindRetryRepairPolicy(wrapped=TodayFixedRepairPolicyAdapter())``
    picks the exact same ``action_type`` as ``TodayFixedRepairPolicyAdapter()``
    alone would for the same ``Evidence``, and differs *only* in
    ``use_diagnostic_feedback``/``feedback_text`` -- see
    ``test/core/test_adapters.py::test_blind_retry_shares_the_wrapped_policys_scope_rule``.

    Also enforces its own per-artifact retry cap and budget check, grouping
    ``Evidence`` by target artifact first so one ``decide()`` call never
    returns two actions for the same artifact (previously, two FAIL
    ``Evidence`` entries naming the same artifact produced two separate
    ``RepairAction``s for it in a single call). The retry counter is
    advanced **only** by ``record_attempt_started()``, never inside
    ``decide()`` -- see ``RepairPolicy.decide()``'s docstring for why
    counting there is wrong (it exhausts the cap before anything was
    actually executed, and double-counts on a repeated/speculative
    ``decide()`` call).
    """

    def __init__(self, wrapped: RepairPolicy, max_retries_per_artifact: int = 3) -> None:
        self._wrapped = wrapped
        self._max_retries = max_retries_per_artifact
        self._attempts: Dict[str, int] = {}

    def record_attempt_started(self, action: RepairAction) -> None:
        """Call this once, when (and only when) an executor actually begins
        carrying out ``action`` -- never from inside ``decide()``. Advances
        the retry count for every artifact the action targets."""
        for target in action.target_artifacts:
            self._attempts[target.artifact_id] = self._attempts.get(target.artifact_id, 0) + 1

    def decide(
        self, evidence: List[Evidence], dependency_edges: List[Any], budget: Budget
    ) -> List[RepairAction]:
        first_failure_by_artifact: Dict[str, Evidence] = {}
        for item in evidence:
            if item.outcome != CheckOutcome.FAIL:
                continue
            first_failure_by_artifact.setdefault(item.checked_artifact.artifact_id, item)

        actions: List[RepairAction] = []
        for artifact_id, item in first_failure_by_artifact.items():
            if budget.remaining <= 0:
                break
            if self._attempts.get(artifact_id, 0) >= self._max_retries:
                continue
            for wrapped_action in self._wrapped.decide([item], dependency_edges, budget):
                actions.append(
                    wrapped_action.model_copy(
                        update={"use_diagnostic_feedback": False, "feedback_text": None}
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

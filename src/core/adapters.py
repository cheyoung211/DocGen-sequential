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
            # The real output, serialized -- a Stage 4 review found a caller
            # (RepairOperationExecutor) was persisting content=None instead
            # of what was actually produced; ProductionResult.content exists
            # so this adapter can report it honestly. read_input_versions
            # simply echoes back what the caller told this call to use
            # (context["read_input_versions"]) -- PlannerAgent has no
            # finer-grained tracking of its own ArtifactRef-level inputs to
            # report instead.
            content=self.last_output.model_dump_json(),
            read_input_versions=context.get("read_input_versions"),
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

    **``produce()`` has no way to suppress ``process_node``'s own internal
    retry-with-feedback loop.** Every inner attempt inside
    ``_run_generation_loop`` unconditionally feeds its own previous
    attempt's error back into the next attempt's prompt
    (``text_agent.py:765`` passes ``previous_error=last_error``) -- there is
    no parameter on ``process_node`` to turn that off, and this adapter
    does not add one (that would be an agent-code change, outside Stage
    1's scope). A caller building a "blind" condition out of
    ``RepairAction.use_diagnostic_feedback=False`` is only blind at the
    outer level -- which diagnostic text, if any, this adapter *was handed*
    to pass along to the Producer call. It cannot, by itself, make
    ``TextAgent`` stop learning from its own in-flight attempts; see
    ``RepairAction``'s docstring for the same point from the schema side.
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
            # See PlannerAgentProducer's identical comment -- same Stage 4
            # review finding, same fix.
            content=self.last_output.model_dump_json(),
            read_input_versions=context.get("read_input_versions"),
        )


class BlockPatchProducer:
    """Wraps ``TextAgent.regenerate_blocks`` as a ``Producer`` -- added in
    Stage 4 to give ``RepairActionType.PATCH_BLOCK`` a real backing
    producer (Stage 1 only had producers for ``process_node``/
    ``generate_plan``, i.e. SECTION/DOCUMENT; nothing wrapped the
    block-level repair call).

    ``artifact_id`` here is the block id (matching this class's role: one
    instance call regenerates exactly one block), and ``context`` must
    carry ``"node_id"`` (the section the block belongs to -- unlike
    ``TextAgentProducer``, where ``artifact_id`` already *is* the section,
    ``regenerate_blocks`` needs both the section and the block id as
    separate parameters, so there is no way to derive one from the other
    here). ``"graph"``/``"request_id"`` are required exactly like
    ``TextAgentProducer``; ``output_dir``/``event_logger``/``usage_sink``/
    ``outer_iteration`` are forwarded when present.

    ``"error_message"`` is read from ``context`` as the diagnostic text to
    seed the regeneration with -- but ``regenerate_blocks`` takes
    ``error_message`` as a required, non-optional positional parameter
    (``text_agent.py:656-658``), unlike ``process_node``, which has no such
    parameter at all. This means PATCH_BLOCK has no way to be truly
    message-free the way ``RepairAction.use_diagnostic_feedback=False``
    implies for the section-repair branch (see that field's and
    ``RepairAction``'s own docstrings for the identical gap one level up,
    at the repair-policy level): a caller that wants a "blind" patch must
    still supply *some* string. This adapter does not invent one -- it
    requires ``context["error_message"]`` unconditionally and raises a
    clear ``KeyError``-free ``ValueError`` if it is missing, so a caller
    choosing to go "blind" at this scope must make an explicit choice about
    what placeholder text to pass (e.g. a content-free ``"retry"``) rather
    than this adapter silently manufacturing one that could be mistaken for
    real diagnostic feedback.

    Failure-mode normalization mirrors ``TextAgentProducer`` exactly, since
    ``regenerate_blocks`` shares the same two failure paths as
    ``process_node`` (a precondition raise, or a ``None`` return with
    ``graph.nodes[node_id].last_error`` set) -- both documented on that
    class and not repeated here.
    """

    def __init__(self, text_agent: TextAgent) -> None:
        self._text_agent = text_agent
        self.last_output: Any = None

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ProductionResult:
        node_id = context["node_id"]
        graph: DocumentGraph = context["graph"]
        if "error_message" not in context:
            raise ValueError(
                "BlockPatchProducer requires context['error_message'] -- "
                "regenerate_blocks has no parameter to omit it; see this "
                "class's docstring for the 'blind patch' gap."
            )
        try:
            self.last_output = self._text_agent.regenerate_blocks(
                node_id,
                artifact_id,
                context["error_message"],
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
            node = graph.nodes.get(node_id)
            error = (node.last_error if node is not None else None) or (
                "regenerate_blocks returned None with no recorded error"
            )
            return ProductionResult(status=ProductionStatus.FAILED, error=error)

        return ProductionResult(
            status=ProductionStatus.SUCCESS,
            artifact=ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.BLOCK),
            # See PlannerAgentProducer's identical comment -- same Stage 4
            # review finding, same fix.
            content=self.last_output.model_dump_json(),
            read_input_versions=context.get("read_input_versions"),
        )


# The two unconditional (never enabled_validators-gated) checks
# _validate_one_block runs before anything else, IN THIS ORDER
# (text_agent.py:1020-1037) -- each one's own early ``return errors`` means
# every check *after* it in this tuple, and every gated check below both,
# never ran at all. Order matters: type_mismatch is checked first and can
# short-circuit empty_content, but not vice versa.
_STRUCTURAL_CHECK_ORDER = ("type_mismatch", "empty_content")
_STRUCTURAL_CODES = frozenset(_STRUCTURAL_CHECK_ORDER)


def _short_circuited_before(code: str, errors_by_code: Dict[str, SemanticBlockError]) -> bool:
    """True if ``_validate_one_block`` returned before ``code``'s own check
    could run at all, because an earlier check in ``_STRUCTURAL_CHECK_ORDER``
    already failed -- meaning a later code's absence from ``errors_by_code``
    reflects "never evaluated", not "evaluated and passed". Every gated
    check (anything not in ``_STRUCTURAL_CHECK_ORDER``) is short-circuited
    by either structural failure, since both sit before the gated checks
    in the method body.
    """
    for earlier_code in _STRUCTURAL_CHECK_ORDER:
        if earlier_code == code:
            return False
        if earlier_code in errors_by_code:
            return True
    return False

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
    ``constraints`` otherwise. Checked in this order, each a different
    reason a constraint does not get a plain PASS/FAIL:

    1. unrecognized ``constraint_id`` -> UNKNOWN / NOT_REPRESENTABLE (this
       adapter has no check by that name at all).
    2. recognized, but ``layout_block.kind`` is one ``_validate_one_block``
       never runs that check for (e.g. ``equation_shape`` on a
       non-equation block) -> UNKNOWN / NOT_REPRESENTABLE. Checked *before*
       #3 below: a kind mismatch is true independent of anything else that
       did or didn't short-circuit, so it is the more fundamental reason
       and takes priority when both would otherwise apply.
    3. kind-applicable, but ``_validate_one_block`` returned before this
       code's own check could run, because an *earlier* unconditional
       check -- ``type_mismatch``, then ``empty_content``
       (``_STRUCTURAL_CHECK_ORDER``) -- already failed first. This can
       happen to ``empty_content`` itself (short-circuited by
       ``type_mismatch``), not only to a gated check -> UNKNOWN /
       NOT_CHECKED.
    4. kind-applicable and reached, but its gate in ``enabled_validators``
       is off for this run -> UNKNOWN / NOT_CHECKED.
    5. everything else: actually evaluated -> PASS or FAIL.
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

        evidence: List[Evidence] = []
        for constraint in constraints:
            code = constraint.constraint_id

            if code not in _STRUCTURAL_CODES and code not in _GATE_KEY_BY_CODE:
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

            # Kind-applicability is checked before short-circuit below: if
            # this check could never apply to this block's kind at all
            # (e.g. table_shape on a DEFINITION block), that is true
            # regardless of whether something else also short-circuited
            # _validate_one_block -- it is the more fundamental reason, and
            # does not depend on errors_by_code at all. _APPLICABLE_KINDS_
            # BY_CODE/_EXCLUDED_KINDS_BY_CODE have no entry for a
            # structural code, so this is always "applicable" for those.
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

            # Checked next, including for a structural code itself:
            # type_mismatch is never short-circuited (it's first in
            # _STRUCTURAL_CHECK_ORDER), but empty_content IS short-
            # circuited by type_mismatch -- so a structural code can still
            # be UNKNOWN/NOT_CHECKED, not just a gated one. This is the fix
            # for a real bug: a block that fails type_mismatch AND would
            # also fail empty_content (wrong type, and empty) used to
            # report empty_content as PASS, because _validate_one_block
            # never actually reached that check to fail it -- "absent from
            # errors_by_code" was wrongly read as "checked and passed".
            if _short_circuited_before(code, errors_by_code):
                evidence.append(
                    Evidence(
                        constraint_id=code,
                        outcome=CheckOutcome.UNKNOWN,
                        checked_artifact=artifact,
                        detail="an earlier structural check failed first, so this check never ran",
                        unknown_reason=UnknownReason.NOT_CHECKED,
                    )
                )
                continue

            if code in _STRUCTURAL_CODES:
                evidence.append(self._pass_or_fail(code, errors_by_code, artifact))
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

    1. Scope *and* feedback, together -- the real method's two call sites
       differ in both, not just scope (``run_full_pipeline.py:499-511``):
       when the error location resolved to a block, it calls
       ``TextAgent.regenerate_blocks(section_id, block_id, location.message,
       ...)``, passing the compiler's error message as the explicit
       ``error_message`` argument -- feedback *is* delivered. When it
       didn't resolve to a block, it calls
       ``TextAgent.process_node(section_id, graph, request_id, ...)`` --
       this call has no error-message parameter at all, so the compiler's
       message is computed (``location.message``) but never actually
       handed to the producer. So: ``PATCH_BLOCK`` ->
       ``use_diagnostic_feedback=True``; ``REGENERATE_SECTION`` ->
       ``use_diagnostic_feedback=False``. An earlier version of this
       adapter set feedback ``True`` for both branches, which did not match
       the real method's section-repair call.
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
            is_block = item.checked_artifact.kind == ArtifactKind.BLOCK
            action_type = RepairActionType.PATCH_BLOCK if is_block else RepairActionType.REGENERATE_SECTION
            return [
                RepairAction(
                    action_type=action_type,
                    target_artifacts=[item.checked_artifact],
                    # Matches regenerate_blocks (gets location.message) vs
                    # process_node (gets nothing) -- see this method's
                    # docstring, point 1.
                    use_diagnostic_feedback=is_block,
                    feedback_text=item.detail if is_block else None,
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
    ``TodayFixedRepairPolicyAdapter()``) and shares its decisions by
    construction, instead of inventing independent logic of its own.

    **Calls ``wrapped.decide()`` exactly once per ``decide()`` call, on one
    shared evidence list, then only strips feedback from whatever it
    returns.** An earlier version called ``wrapped.decide([item], ...)``
    separately *per distinct failing artifact* -- which seemed like a
    reasonable way to also enforce a per-artifact retry cap, but silently
    changed more than feedback: ``TodayFixedRepairPolicyAdapter`` alone,
    given 2 distinct failing artifacts, returns exactly 1 action (its own
    ``regenerable[0]``-style cardinality -- see its docstring); the
    per-artifact-call version of this wrapper returned 2, one per artifact,
    because each singleton call let the wrapped policy "see" only one
    failure at a time and naturally act on it. That was no longer "the same
    policy, minus feedback" -- it was a different policy that also happened
    to drop feedback. Fixed by computing the retry cap's effect as a
    *restriction on the evidence list itself* (evidence for an
    already-exhausted artifact is simply left out, as if that failure
    weren't reported this round) and then calling ``wrapped.decide()``
    once on what remains -- so cardinality, ordering, and every other
    aspect of the wrapped policy's behavior come through unchanged; see
    ``test/core/test_adapters.py::test_blind_retry_matches_the_wrapped_policys_cardinality_exactly``.

    The retry counter behind that restriction is advanced **only** by
    ``record_attempt_started()``, never inside ``decide()`` -- see
    ``RepairPolicy.decide()``'s docstring for why counting there is wrong
    (it exhausts the cap before anything was actually executed, and
    double-counts on a repeated/speculative ``decide()`` call).
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
        if budget.remaining <= 0:
            return []
        eligible_evidence = [
            item
            for item in evidence
            if item.outcome != CheckOutcome.FAIL
            or self._attempts.get(item.checked_artifact.artifact_id, 0) < self._max_retries
        ]
        wrapped_actions = self._wrapped.decide(eligible_evidence, dependency_edges, budget)
        return [
            action.model_copy(update={"use_diagnostic_feedback": False, "feedback_text": None})
            for action in wrapped_actions
        ]


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
    "BlockPatchProducer",
    "TextAgentValidatorAdapter",
    "BlindRetryRepairPolicy",
    "TodayFixedRepairPolicyAdapter",
    "ContractOnlyFinalEvaluator",
]

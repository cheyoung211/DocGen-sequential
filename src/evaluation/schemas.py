"""Structured records for the Foundation evaluation layer.

These models capture what each pipeline stage's existing validation/retry
logic already does -- they add no new validation behavior of their own, only
a record of it. See ``event_logger.py`` for how they get populated and the
"Verification Evaluation Metrics" research spec for the metrics computed from
them (``metrics/*.py``).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Stage(str, Enum):
    """The pipeline stages a VerificationEvent can originate from.

    ``COMPOSER`` and ``COMPILE`` re-verify an artifact another stage already
    produced (they have no LLM output of their own to regenerate) --
    see GENERATION_STAGES below for the subset that does.
    """

    PLANNER = "planner"
    TEXT_AGENT = "text_agent"
    IMAGE_AGENT = "image_agent"
    COMPOSER = "composer"
    COMPILE = "compile"


# Used by EventLogger.total_generation_attempts: a composer/compile check
# re-verifies an artifact another stage already generated, it doesn't itself
# generate one, so it shouldn't count toward "generation attempts".
GENERATION_STAGES = frozenset({Stage.PLANNER, Stage.TEXT_AGENT, Stage.IMAGE_AGENT})


class RecoveryAction(str, Enum):
    """What kind of recovery a compile failure would need -- classification
    only, no retry loop consumes this yet. See
    compile_log_parser.classify_recovery_action.
    """

    REGENERATE_CONTENT = "regenerate_content"
    DETERMINISTIC_REPAIR = "deterministic_repair"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    UNKNOWN = "unknown"

VerificationResultValue = Literal["pass", "fail"]

# No real verification-config toggle system exists yet (see
# VerificationConfigSnapshot below) -- every run today uses the same fixed
# configuration, so every RunResult can point at one shared name instead of
# inventing a per-run identity that would always be identical.
DEFAULT_CONFIG_NAME = "always_on_v1"


class VerificationEvent(BaseModel):
    """One verification mechanism's pass/fail judgment of one artifact/attempt.

    Deviates from the research spec's literal ``VerificationEvent`` class in
    two additive ways -- both required to compute the metrics the spec itself
    asks for, not decorative: ``failure_id`` is the join key a
    ``RecoveryEvent`` needs to reference *this* failure (Metric 2's own
    worked JSON example includes one; the spec's formal class omits it), and
    ``producer_model`` is needed for Metric 1's explicit "per producer model"
    aggregation requirement (no other field could supply it).

    ``failure_id`` is set by the caller, not auto-assigned here: its
    presence or absence is a structural fact about whether this stage *has* a
    recovery mechanism at all (``None`` for image_agent/compile, which have
    none), not a per-event choice -- see ``EventLogger.new_failure_id`` and
    ``AttemptTracker``.

    ``sequence_id`` and ``outer_iteration`` close a real provenance gap:
    within one inner retry loop, ``attempt`` is monotonic, but
    ``run_full_pipeline.py``'s *outer* Phase-2 loop re-dispatches a node with
    a brand-new inner loop on every iteration, so ``attempt`` alone resets to
    1 each time -- the same ``(run_id, stage, verifier, artifact_id,
    attempt)`` tuple can legitimately repeat across outer iterations.
    ``sequence_id`` is assigned by ``EventLogger`` itself, globally monotonic
    across every verification *and* recovery event for one run, so call
    order is always unambiguous and collision-free even when ``attempt``
    resets; ``outer_iteration`` additionally records *which* Phase-2 pass
    produced this event where that concept applies (``None`` for
    planner/composer/compile, which only ever run once per run).
    """

    run_id: str
    stage: Stage
    verifier: str
    artifact_id: Optional[str] = None
    section_id: Optional[str] = None
    block_id: Optional[str] = None
    attempt: int
    outer_iteration: Optional[int] = None
    sequence_id: Optional[int] = None
    result: VerificationResultValue
    signal_type: Optional[str] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None
    producer_model: Optional[str] = None
    failure_id: Optional[str] = None


class RecoveryEvent(BaseModel):
    """One recovery action's outcome, joined to its triggering failure via ``failure_id``.

    ``attempt`` is the attempt index of the retry that produced this outcome
    (i.e. the attempt that resolved the *preceding* failure) -- following the
    spec's formal ``RecoveryEvent`` class shape rather than its illustrative
    JSON example's looser ``recovery_attempts`` count field, since the two
    are not the same thing and the formal class is the one with a `success`/
    `attempt` pair general enough to represent every retry outcome, not just
    a final tally.
    """

    run_id: str
    failure_id: str
    stage: Stage
    artifact_id: Optional[str] = None
    recovery_action: str
    attempt: int
    outer_iteration: Optional[int] = None
    sequence_id: Optional[int] = None
    success: bool
    resulting_artifact_id: Optional[str] = None


class CompileResult(BaseModel):
    """The richer per-compile record Metric 6's own JSON example asks for.

    Not part of the research spec's formal ``RunResult`` class (which only
    has a flat ``compile_success: bool``) -- persisted alongside the existing
    ``compile.log`` as ``compile_result.json``, and referenced from
    ``RunResult.compile_result``. ``compile_success`` is computed
    independently of ``fatal_error_count``/``warning_count`` -- a
    warnings-only compile can still be ``compile_success=True`` (per the
    spec's explicit caution that these must be tracked as separate values).
    """

    compile_success: bool
    engine: Optional[str] = None
    return_code: Optional[int] = None
    fatal_error_count: int = 0
    warning_count: int = 0
    first_error_type: Optional[str] = None
    # Populated by classify_recovery_action() from first_error_type -- no
    # retry loop consumes this yet, classification only.
    recovery_action: Optional[RecoveryAction] = None
    pdf_exists: bool = False


ContractItemStatus = Literal[
    "satisfied", "partially_satisfied", "missing", "not_representable", "not_applicable"
]


class ContractCategoryDetail(BaseModel):
    """Why one contract item did or didn't count as satisfied.

    Not in the spec's minimal ``contract_satisfaction`` JSON example (which
    only shows category-level floats) -- added for the same reason the spec
    asks Metrics 1-2 to keep raw events, not just aggregates: a category
    score alone can't distinguish a component that was actually missing from
    one that was never representable in this pipeline's current block-kind
    vocabulary (see ``metrics/contract_satisfaction.py``'s mapping table).

    ``matched_via``/``required_points_coverage`` are populated only for
    ``required_sections`` items -- see that check's docstring in
    ``metrics/contract_satisfaction.py`` for why a required section's status
    now depends on more than bare existence.
    """

    item_id: str
    status: ContractItemStatus
    reason: Optional[str] = None
    matched_via: Optional[str] = None
    required_points_coverage: Optional[float] = None


class ContractResult(BaseModel):
    """Category-level Contract Satisfaction Rate for one generated document.

    ``content_coverage`` and ``cross_section_dependency`` are semantic checks
    the research spec explicitly permits deferring. They are always ``None``
    here and listed in ``not_implemented`` -- never silently scored as 0 or
    dropped from the output.
    """

    overall: Optional[float] = None
    required_sections: Optional[float] = None
    required_components: Optional[float] = None
    content_coverage: Optional[float] = None
    terminology: Optional[float] = None
    notation: Optional[float] = None
    cross_section_dependency: Optional[float] = None
    not_implemented: List[str] = Field(
        default_factory=lambda: ["content_coverage", "cross_section_dependency"]
    )
    details: Dict[str, List[ContractCategoryDetail]] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    by_stage: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class VerificationConfigSnapshot(BaseModel):
    """What verification/recovery machinery is actually active for a run.

    No toggle system exists yet -- that's deferred to a future pass. Every
    field here just records today's always-on-or-nonexistent reality (for
    example ``compiler_recovery=False``, since no such mechanism exists), so
    this snapshot is already meaningful and its shape won't need to change
    once a real toggle system lands.
    """

    planner_validation: bool = True
    planner_fallback: bool = True
    text_validation: bool = True
    text_retry_feedback: bool = True
    image_validation: bool = True
    image_retry_feedback: bool = False
    composer_structural_validation: bool = True
    compiler_feedback: bool = False
    compiler_recovery: bool = False


class RunResult(BaseModel):
    """Everything the Foundation metrics need about one pipeline run.

    ``run_id`` is the pipeline's existing ``request_id`` string, reused
    rather than duplicated as a separate identity. ``producer_models`` is
    additive beyond the spec's single ``producer_model`` field: this
    pipeline genuinely allows 4 independently-chosen models per run
    (planner/text/image/integrator), so collapsing to one string would lose
    real information for a mixed-model run; ``producer_model`` is kept too,
    set to the text model, since TextAgent is the pipeline's bulk producer.
    """

    run_id: str
    benchmark_id: Optional[str] = None
    producer_model: str
    producer_models: Dict[str, str] = Field(default_factory=dict)
    config_name: str = DEFAULT_CONFIG_NAME
    verification_config: VerificationConfigSnapshot = Field(
        default_factory=VerificationConfigSnapshot
    )
    completed: bool
    # More granular than `completed`: which of run_full_pipeline.py's 4
    # outcome buckets this run landed in ("success", "content_generation_error",
    # "integration_error", "compile_error") -- preserved from the pre-Foundation
    # run_result.json shape so no information is lost once this schema
    # becomes the file's top-level shape.
    status: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    compile_success: bool
    compile_result: Optional[CompileResult] = None
    verification_events: List[VerificationEvent] = Field(default_factory=list)
    recovery_events: List[RecoveryEvent] = Field(default_factory=list)
    contract_result: Optional[ContractResult] = None
    total_generation_attempts: int = 0
    total_recovery_attempts: int = 0
    token_usage: Optional[TokenUsage] = None
    elapsed_time: Optional[float] = None

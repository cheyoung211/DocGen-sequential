"""Artifact, constraint, evidence, action, and cost schemas (Stage 1 of the
"Revision 2026.10.15" plan: 검증·복구·평가 분리와 실험 설계).

These are new, additive concepts -- they do not replace anything in
``src/common/state.py`` (today's section-only ``DocumentGraph``) or
``src/evaluation/schemas.py`` (today's online ``VerificationEvent``/
``RecoveryEvent``/``TokenUsage`` records). Where an existing model already
covers part of a concept, the class docstring below says so explicitly and
links to it by name instead of redefining it.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from src.evaluation.schemas import CompileResult


class ArtifactKind(str, Enum):
    """The artifact vocabulary from revision direction 1 ("definition,
    symbol, equation, table, claim, block, source data, calculation 등").
    Finer-grained than ``src/common/state.py::NodeType`` (only SECTION/
    SUBSECTION/TABLE/FIGURE/BIBLIOGRAPHY) -- that enum is for today's
    section-only graph and is left untouched; this one is for the Stage 2
    artifact graph."""

    DOCUMENT = "document"
    SECTION = "section"
    BLOCK = "block"
    DEFINITION = "definition"
    SYMBOL = "symbol"
    EQUATION = "equation"
    TABLE = "table"
    CLAIM = "claim"
    FIGURE = "figure"
    SOURCE_DATA = "source_data"
    CALCULATION = "calculation"


class ArtifactRef(BaseModel):
    """A pointer to one versioned artifact -- not its content.

    ``read_input_versions`` records this version's generation provenance
    (revision direction 1: "생성 근거, 읽은 입력 버전"). Stage 1 only defines
    the shape; Stage 2 owns actually assigning/incrementing ``version`` and
    persisting these pointers -- today every agent overwrites
    ``DocumentNode.content`` in place (``src/common/state.py``), which has
    no version field at all.
    """

    artifact_id: str
    version: int = 1
    kind: ArtifactKind
    read_input_versions: List["ArtifactRef"] = Field(default_factory=list)


class ProductionStatus(str, Enum):
    """Whether a ``Producer.produce()`` call actually generated the
    artifact. Added because today's two real producers disagree on how they
    signal running out of retries: ``TextAgent.process_node`` returns
    ``None`` (``src/agents/text_agent.py:818-821``), while
    ``PlannerAgent.generate_plan`` raises instead
    (``src/agents/planner.py:714-716``). A ``Producer`` adapter must
    normalize both into this one explicit outcome -- a caller must never
    have to know which legacy failure mechanism a given agent happens to
    use."""

    SUCCESS = "success"
    FAILED = "failed"


class ProductionResult(BaseModel):
    """``Producer.produce()``'s return shape. ``artifact`` is set if and
    only if ``status`` is ``SUCCESS`` -- enforced below, so a controller
    reading this can never mistake a failed generation for a real artifact
    reference.

    ``content``/``read_input_versions`` are the real output and the real
    inputs this production call actually used -- added in Stage 4 after a
    review found that a caller (``repair_operations.RepairOperationExecutor``)
    was persisting ``content=None`` and a *guessed* input set instead of
    what was actually produced, making every repaired artifact's stored
    content and provenance fabricated rather than real. Both are
    best-effort: ``content`` stays ``None`` when a ``Producer`` has no
    practical way to serialize its real output (not a lie -- an honest "not
    reported", the same discipline ``ContentPreservationScore.
    not_implemented_reason`` already uses elsewhere). ``read_input_versions``
    stays ``None`` (not ``[]`` -- the two mean different things: "not
    reported" vs. "really used zero inputs") when the ``Producer`` has no
    way to know which ``ArtifactRef``s it actually read; a conforming
    ``Producer`` that was told via ``context["read_input_versions"]`` what
    it was asked to use, and has no finer-grained tracking of its own, may
    honestly echo that list back here (it is reporting its own caller's
    instruction, not inventing one) -- see ``adapters.py``'s producers for
    the concrete pattern.
    """

    status: ProductionStatus
    artifact: Optional[ArtifactRef] = None
    error: Optional[str] = None
    content: Optional[str] = None
    read_input_versions: Optional[List[ArtifactRef]] = None

    def model_post_init(self, __context: object) -> None:
        if self.status == ProductionStatus.SUCCESS and self.artifact is None:
            raise ValueError("ProductionResult with status=SUCCESS requires an artifact.")
        if self.status == ProductionStatus.FAILED and self.artifact is not None:
            raise ValueError("ProductionResult with status=FAILED must not carry an artifact.")


class DependencyRelationKind(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class DependencyEdge(BaseModel):
    """One directed edge: an upstream artifact pointing to a downstream
    artifact that reads it. ``relation`` distinguishes a dependency the
    producer actually declared (EXPLICIT) from one inferred later from
    usage (INFERRED) -- revision direction 1 requires this distinction so a
    repair policy can treat them with different confidence. Conceptually
    mirrors, but does not replace, ``DocumentNode.dependencies: List[str]``
    (``src/common/state.py:258``), which today is an unused, same-type,
    direction-less id list.

    ``reason`` is only ever populated for INFERRED edges -- the free-form
    justification passed to ``ArtifactGraphStore.add_inferred_edge``. Left
    ``None`` for EXPLICIT edges: their "reason" is simply that the producer
    declared this artifact as a read input, which ``relation`` alone
    already says, so inventing a canned string there would tell an auditor
    nothing new.
    """

    from_artifact: ArtifactRef
    to_artifact: ArtifactRef
    relation: DependencyRelationKind
    reason: Optional[str] = None


class CheckOutcome(str, Enum):
    """Mirrors ``src/evaluation/schemas.py::VerificationResultValue``
    (``Literal["pass", "fail"]``) but adds UNKNOWN -- today's online
    verifiers have no way to report "could not determine": every check is
    forced to resolve to pass or fail even when, e.g., a solver times out or
    a symbolic check can't represent the claim. See ``UnknownReason`` for
    why a given check came out UNKNOWN."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class UnknownReason(str, Enum):
    """Why a constraint check resolved to ``CheckOutcome.UNKNOWN`` instead
    of pass/fail. Revision direction 3 explicitly asks that "판정 불가
    사유" be recorded, not just a bare UNKNOWN flag."""

    NOT_REPRESENTABLE = "not_representable"
    SOLVER_TIMEOUT = "solver_timeout"
    NOT_CHECKED = "not_checked"
    MISSING_DEPENDENCY = "missing_dependency"
    STALE_EVIDENCE = "stale_evidence"


class ConstraintKind(str, Enum):
    """The four checkable-constraint families from revision direction 3's
    table (schema contract / CSP-SMT / recompute / symbolic)."""

    SCHEMA_CONTRACT = "schema_contract"
    CSP_SMT = "csp_smt"
    RECOMPUTE = "recompute"
    SYMBOLIC = "symbolic"


class Constraint(BaseModel):
    """One checkable requirement against one or more artifacts.

    ``source_ref`` optionally points at the authoring-time requirement this
    constraint was instantiated from: ``src/dataset/schemas.py``'s
    ``HardConstraint.constraint_id``, ``TerminologyConstraint.canonical_term``,
    or ``NotationConstraint.symbol`` -- that module already defines
    ``ConstraintType``/``HardConstraint``/``CrossSectionDependency``/
    ``TerminologyConstraint``/``NotationConstraint`` for the benchmark
    *contract* (what a document must satisfy). This class is the runtime,
    check-time object evaluated against one specific artifact version; it
    links to the contract item by id rather than redefining it.
    """

    constraint_id: str
    kind: ConstraintKind
    description: str
    source_ref: Optional[str] = None
    applies_to: List[ArtifactRef] = Field(default_factory=list)


class Evidence(BaseModel):
    """One constraint check's structured result: "pass/fail만이 아니라
    constraint ID, evidence, 관련 artifact 버전, 판정 불가 사유" (revision
    direction 3). Produced only by a ``Validator`` (see ``interfaces.py``);
    a ``FinalEvaluator`` must never produce or consume this as its quality
    signal -- see that Protocol's docstring for why.

    ``input_versions`` records every artifact version (besides
    ``checked_artifact`` itself) this verdict actually depended on -- e.g.
    a SCHEMA_CONTRACT check's ``Constraint.applies_to`` refs, or a
    RECOMPUTE check's source-data artifact. Added after a Stage 3 review
    found that without this, a verdict could silently keep looking "live"
    after one of its real inputs changed version: ``checked_artifact``
    alone is not the complete set of things a check's correctness depends
    on. ``ArtifactGraphStore.live_evidence()`` requires every entry here to
    still be the current, non-stale version of its artifact -- see that
    method's docstring. Defaults to empty for a check that genuinely
    depends on nothing beyond ``checked_artifact`` itself (e.g. a
    structural check with no cross-references).

    ``missing_inputs`` is the counterpart for an input a check found to
    *not exist at all* at check time -- e.g. a SCHEMA_CONTRACT FAIL whose
    required artifact was never produced. A second Stage 3 review found
    that cramming this case into ``input_versions`` instead (as some
    already-stored version number that happened to be expected) actively
    inverted ``live_evidence()``'s invalidation logic: a verdict asserting
    "X is absent" was treated as invalid *while X stayed absent* (because
    ``input_versions``' liveness rule requires the referenced version to
    currently exist), and, backwards, started being treated as *live again*
    the moment X was finally produced -- exactly backwards from "the thing
    this verdict depends on changed, so it needs rechecking".

    Holds the full ``ArtifactRef`` (artifact_id **and** version), not just
    an artifact_id -- a third Stage 3 review found that id-only tracking
    conflated two different facts: "this artifact_id has never been
    stored at all" and "this artifact_id exists, but not at the exact
    version a constraint required" (e.g. ``table@1`` is stored but
    ``table@2`` was required). With id-only tracking, the moment *any*
    version of that id appeared (``table@1``, already present before the
    check even ran), the FAIL was wrongly invalidated even though the
    specific missing thing it named -- ``table@2`` -- still did not
    exist. An entry here stays live only while that *exact*
    ``(artifact_id, version)`` is still, right now, never-stored
    (``ArtifactGraphStore._still_missing``); the moment that exact version
    appears, this verdict must invalidate, the same direction
    ``input_versions`` invalidates when its ref moves to a different
    version. Never put the same ``ArtifactRef`` on both lists at once --
    see ``constraint_checks.BasicConstraintChecker._resolve_input_refs``
    for why every input must be existence-checked before being routed to
    either list, not assumed from which API a caller happened to declare
    it through.
    """

    constraint_id: str
    outcome: CheckOutcome
    checked_artifact: ArtifactRef
    detail: str = ""
    unknown_reason: Optional[UnknownReason] = None
    input_versions: List[ArtifactRef] = Field(default_factory=list)
    missing_inputs: List[ArtifactRef] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if self.outcome == CheckOutcome.UNKNOWN and self.unknown_reason is None:
            raise ValueError("Evidence with outcome=UNKNOWN requires unknown_reason.")


class RepairActionType(str, Enum):
    """The repair-scope vocabulary from revision directions 1/2 ("patch
    block, section regeneration, subgraph replan, whole-document
    regeneration")."""

    PATCH_BLOCK = "patch_block"
    REGENERATE_SECTION = "regenerate_section"
    REPLAN_SUBGRAPH = "replan_subgraph"
    REGENERATE_DOCUMENT = "regenerate_document"


class RepairAction(BaseModel):
    """One ``RepairPolicy`` decision: what scope to regenerate/re-check
    (``action_type``), and whether the producer is handed the triggering
    ``Evidence``'s detail text ("diagnostic feedback", ``use_diagnostic_feedback``).

    These two fields are independent axes by construction: a policy can
    hold ``action_type`` fixed while varying ``use_diagnostic_feedback``
    (the "blind retry" vs "diagnostic retry" comparison revision direction 2
    asks for), or vice versa. ``adapters.BlindRetryRepairPolicy`` demonstrates
    this by wrapping another ``RepairPolicy`` (e.g.
    ``adapters.TodayFixedRepairPolicyAdapter``) and stripping only
    ``use_diagnostic_feedback``/``feedback_text`` from its decisions, while
    sharing its exact ``action_type`` choice.

    ``use_diagnostic_feedback`` governs only the diagnostic text a
    ``RepairPolicy`` attaches to *this one action* -- what an executor would
    pass to a ``Producer`` call site when carrying it out (e.g.
    ``TextAgent.regenerate_blocks``'s ``error_message`` parameter). It says
    nothing about, and cannot disable, a ``Producer``'s own *internal*
    retry-with-feedback loop (``TextAgent.process_node``'s inner attempts
    feed their own self-discovered validation errors into the next inner
    attempt's prompt unconditionally -- see ``text_agent.py``'s
    ``_run_generation_loop``, which has no parameter to turn that off). A
    "blind" condition built from this flag alone is blind only at the
    outer, repair-policy-chosen-diagnostic level; see
    ``adapters.TextAgentProducer``'s docstring for the concrete gap.
    """

    action_type: RepairActionType
    target_artifacts: List[ArtifactRef]
    use_diagnostic_feedback: bool
    feedback_text: Optional[str] = None
    triggering_constraint_id: Optional[str] = None


class CostRecord(BaseModel):
    """Cost of one action -- a generation call, a check, or a repair
    attempt, including a *failed* one (revision direction 2: "비용에는
    초기 생성, 검사, 실패한 시도, 수정 후 downstream 재계산을 포함한다").

    Field names deliberately overlap ``src/evaluation/schemas.py::
    TokenUsage`` (``total_input_tokens``/``total_output_tokens``/
    ``total_tokens``/``call_count``) rather than inventing new ones, so a
    batch of ``CostRecord``s can be summed into a ``TokenUsage`` with no
    translation layer. ``latency_s`` and ``gpu_seconds`` are additive
    beyond it: revision direction 2 explicitly says API cost and local
    inference cost must not be assumed equal.
    """

    stage: str
    action_type: Optional[RepairActionType] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    gpu_seconds: Optional[float] = None
    succeeded: bool = True


class Budget(BaseModel):
    """Running budget state a ``RepairPolicy`` reads before deciding an
    action (revision direction 2's controller state: "누적 비용, 남은
    예산"). ``cap`` is fixed per run/comparison condition; ``spent`` is the
    running sum of ``CostRecord.total_tokens`` charged so far. Which unit a
    given experiment actually caps on (tokens, dollars, wall-clock) is
    decided per condition in ``docs/stage1_experiment_design.md``'s cost-cap
    section -- this class is unit-agnostic.
    """

    cap: float
    spent: float = 0.0

    @property
    def remaining(self) -> float:
        return self.cap - self.spent

    def charge(self, record: CostRecord) -> "Budget":
        return self.model_copy(update={"spent": self.spent + record.total_tokens})


class QualityScore(BaseModel):
    """Independent-rubric quality score for one finished document.

    ``category_scores`` mirrors ``src/evaluation/schemas.py::ContractResult``'s
    per-category floats (required_sections, required_components, ...) by
    key name, not by re-deriving them differently -- an evaluator that
    computes them the way today's ``evaluate_contract`` does (see
    ``adapters.ContractOnlyFinalEvaluator``) can populate this directly from
    a ``ContractResult``; a human-rubric evaluator (Stage 6) populates the
    same keys from a different source.
    """

    overall: Optional[float] = None
    category_scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    not_implemented: List[str] = Field(default_factory=list)


class ContentPreservationScore(BaseModel):
    """How much *unrelated* content a repair damaged -- revision directions
    1/4's "정상 내용 보존". Computing this requires comparing two versions
    of the artifact graph (before/after a repair), which does not exist
    until Stage 2's versioned artifact store lands. Until then every
    ``FinalEvaluator`` must leave the score fields ``None`` and set
    ``not_implemented_reason`` -- never fabricate a number.

    First real implementation: ``failure_dataset._compute_damage`` (Stage
    6). ``newly_broken_artifacts`` means "has a live FAIL (or is currently
    stale) in the final checkpoint" -- NEVER a bare before/after content
    string comparison. A review caught that a plain string diff conflates
    a harmless, legitimate content refresh (e.g. a successful regeneration
    whose wording simply changed) with an actual new error; the two must
    stay distinguishable, and only the latter counts as damage.
    """

    preserved_fraction: Optional[float] = None
    newly_broken_artifacts: List[str] = Field(default_factory=list)
    not_implemented_reason: Optional[str] = None


class VerifierPrecisionRecall(BaseModel):
    """Online ``Validator`` precision/recall against an independent human
    gold label (review problem #3 in the Notion revision doc). Requires the
    human-annotation protocol in ``docs/stage1_experiment_design.md`` to
    have actually been run (Stage 6); until then ``sample_size`` stays 0 and
    the rate fields stay ``None`` -- never backfilled from the online
    signal itself, which would make this tautological.
    """

    precision: Optional[float] = None
    recall: Optional[float] = None
    sample_size: int = 0


class FinalEvaluationResult(BaseModel):
    """``FinalEvaluator.evaluate()``'s return shape -- see that Protocol's
    docstring in ``interfaces.py`` for the constraint that none of these
    three fields may be computed from the run's online verification
    stream."""

    quality: QualityScore
    content_preservation: Optional[ContentPreservationScore] = None
    verifier_precision_recall: Optional[VerifierPrecisionRecall] = None


# ---------------------------------------------------------------------------
# Stage 2 ("Versioned artifact graph와 로그·checkpoint") schemas. The store
# and graph logic that actually uses these lives in
# ``src/core/artifact_graph.py``/``src/core/run_log.py`` -- these classes
# are plain data shapes only, following this file's existing split between
# schema (here) and behavior (``interfaces.py``/elsewhere).
# ---------------------------------------------------------------------------


class ArtifactStatus(str, Enum):
    """Whether a stored artifact's *current* version can still be trusted
    as-is.

    FRESH: nothing this version (transitively) reads from has moved to a
    newer version since it was produced. STALE: at least one upstream input
    has -- the content itself has not changed, only the ground it was
    produced against has. Stage 2 only flags this; deciding what to do
    about it (modify / re-verify / regenerate, revision direction 1's
    three-way split) is Stage 4's job.
    """

    FRESH = "fresh"
    STALE = "stale"


class StoredArtifact(BaseModel):
    """One versioned artifact as actually persisted by
    ``artifact_graph.ArtifactGraphStore``. ``ArtifactRef`` above is the bare
    pointer Stage 1 defined ("Stage 1 only defines the shape; Stage 2 owns
    actually assigning/incrementing ``version`` and persisting these
    pointers" -- see that class's own docstring); this is Stage 2's
    concrete record around it.

    ``produced_by`` is this version's "출처" (provenance, revision
    direction 1) beyond ``ref.read_input_versions``: a free-form label for
    what produced it (a ``Producer`` class name, an executed action's id,
    ...) -- today's ``DocumentNode`` (``src/common/state.py``) has no
    equivalent field at all.
    """

    ref: ArtifactRef
    content: Optional[str] = None
    produced_by: Optional[str] = None
    status: ArtifactStatus = ArtifactStatus.FRESH
    branch_id: str = "main"
    created_at: datetime = Field(default_factory=datetime.now)


class InvalidationRecord(BaseModel):
    """Audit trail entry for one "artifact X changed -> artifact Y marked
    stale" propagation step (revision direction 1's change-impact
    computation; direction 4's requirement that repairs be auditable).

    Never mutates or deletes anything it describes --
    ``ArtifactGraphStore`` appends one of these per affected descendant on
    every ``add_version`` call, and this list is the only record of *why*
    a given artifact is currently stale.
    """

    changed_artifact: ArtifactRef
    affected_artifact_id: str
    reason: str
    created_at: datetime = Field(default_factory=datetime.now)


class ActionLogEntry(BaseModel):
    """One ``RepairAction`` actually carried out -- not merely decided.
    ``RepairPolicy.decide()`` must stay side-effect-free (``interfaces.py``),
    so this is logged by an executor, when and only when an action actually
    begins executing, mirroring ``BlindRetryRepairPolicy.
    record_attempt_started``'s same principle at the logging layer.

    ``triggering_evidence`` is kept alongside the action so a later audit
    can see exactly which ``Evidence`` led to it, without re-joining against
    a separate evidence log by timestamp guesswork.
    """

    action: RepairAction
    branch_id: str
    triggering_evidence: List[Evidence] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class AddVersionResult(BaseModel):
    """What happened as a side effect of one
    ``ArtifactGraphStore.add_version`` call: the new pointer, which
    existing artifacts this change just invalidated, and whether this
    version's own declared inputs were trustworthy at the moment it was
    recorded.

    ``used_outdated_inputs``/``used_stale_inputs`` are deliberately
    informational, not a rejected call: a producer pinning an older or
    already-stale input version on purpose is a real, permitted scenario
    (e.g. a controlled checkpoint-branch comparison) -- but it must be
    visible on the result, never silently indistinguishable from reading
    the true, trustworthy current state. ``born_stale`` is simply
    ``bool(used_outdated_inputs) or bool(used_stale_inputs)``, spelled out
    explicitly rather than left for a caller to re-derive: this is also
    exactly whether ``ref``'s own artifact was stored with
    ``ArtifactStatus.STALE`` instead of ``FRESH`` -- an artifact built from
    unreliable ground is never itself trustworthy just because it is new.
    """

    ref: ArtifactRef
    born_stale: bool = False
    newly_stale_artifact_ids: List[str] = Field(default_factory=list)
    used_outdated_inputs: List[ArtifactRef] = Field(default_factory=list)
    used_stale_inputs: List[ArtifactRef] = Field(default_factory=list)


class Checkpoint(BaseModel):
    """A fully self-contained snapshot of one ``ArtifactGraphStore`` at one
    moment -- revision direction 4: "복구 전 상태를 versioned checkpoint로
    저장한다".

    Carries every artifact version, every typed edge, the current stale
    set, and the full invalidation/action/evidence/cost/tool-call history
    -- no live Python object references back into the store that produced
    it. That is what lets ``ArtifactGraphStore.from_checkpoint`` reconstruct
    a fully independent store sharing no mutable state with the original,
    which is the whole point of "branch into different repair actions from
    the same state" (direction 4): a ``Checkpoint`` could in principle be
    serialized via ``model_dump_json()`` and handed to a different process
    entirely.

    ``tool_calls`` (``ToolCallLogEntry``, defined later in this file) was
    added in the Stage 3 review -- without it, a branch restored from a
    checkpoint had no record of which tool calls/observations led up to
    that state, even though Stage 4's branching re-execution needs exactly
    that to connect an action to its downstream result.
    """

    checkpoint_id: str
    label: Optional[str] = None
    source_branch_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    artifacts: Dict[str, List[StoredArtifact]] = Field(default_factory=dict)
    edges: List[DependencyEdge] = Field(default_factory=list)
    stale_artifact_ids: List[str] = Field(default_factory=list)
    invalidation_history: List[InvalidationRecord] = Field(default_factory=list)
    actions: List[ActionLogEntry] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    costs: List[CostRecord] = Field(default_factory=list)
    tool_calls: List["ToolCallLogEntry"] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 3 ("도구 실행기, 기본 제약 검사, 비용 집계") schemas. The seven
# action kinds are direction 2's controller action space ("inspect_artifact,
# query_dependencies, check_constraints/check_calculation, patch_block,
# replan_subgraph, compile_document, finish"), collapsed to the Stage 3
# checklist's own seven short names ("inspect/query/check/patch/replan/
# compile/finish의 입력·출력 schema") -- the checklist itself folds
# direction 2's "check_constraints / check_calculation" slash-pair into one
# "check" action, so there is exactly one CHECK kind here, not two; which
# ``ConstraintKind``s a given check call actually exercises is a property
# of the ``Constraint`` list passed in, not of the action name.
#
# The behavior behind each kind lives in ``src/core/tool_executor.py``
# (dispatch, budget/action-count/suspend enforcement),
# ``src/core/constraint_checks.py`` (CHECK), and ``src/core/cost_tracking.py``
# (cost/latency/GPU measurement for the one real LLM-calling surface that
# exists today, Stage 1's ``Producer``). These classes are plain data shapes
# only, following this file's existing split between schema (here) and
# behavior (elsewhere) -- see ``interfaces.py``'s and ``artifact_graph.py``'s
# own module docstrings for the same split at Stage 1/2.
# ---------------------------------------------------------------------------


class ToolActionKind(str, Enum):
    """The controller's action vocabulary (revision direction 2), exactly as
    the Stage 3 checklist names it. Through Stage 3, ``PATCH``/``REPLAN``
    were intentionally not implemented -- see ``tool_executor.ToolExecutor``'s
    docstring: actually deciding regeneration scope and carrying it out was
    Stage 4's "고정 복구 정책". From Stage 5 on, a ``ToolExecutor`` constructed
    with ``producer_registry``/``constraint_provider`` dispatches both for
    real, through ``repair_operations.RepairOperationExecutor`` -- an
    executor constructed without those still returns ``NOT_IMPLEMENTED``
    for both, so every pre-Stage-5 caller's behavior is unchanged.

    ``GENERATE`` is not one of the controller's seven selectable actions
    (direction 2 never lists "generate" as something the controller
    chooses -- initial generation happens upstream of the controller's
    decision loop, via Planner/TextAgent). It exists here only so
    ``ToolExecutor.run_generation()`` -- added after a Stage 3 review found
    "초기 생성" had no connected cost/budget path at all -- can log and
    charge a production call through the exact same ``ToolExecutionError``/
    ``CostRecord`` machinery every other action uses, rather than inventing
    a second, parallel error/cost vocabulary just for this one case."""

    INSPECT = "inspect"
    QUERY = "query"
    CHECK = "check"
    PATCH = "patch"
    REPLAN = "replan"
    COMPILE = "compile"
    FINISH = "finish"
    GENERATE = "generate"


class ToolErrorCode(str, Enum):
    """The "실행 오류 형식" (execution error format) the Stage 3 checklist
    asks every tool action to share, rather than each action inventing its
    own ad hoc error shape."""

    BUDGET_EXCEEDED = "budget_exceeded"
    MAX_ACTIONS_EXCEEDED = "max_actions_exceeded"
    SUSPENDED = "suspended"
    NOT_IMPLEMENTED = "not_implemented"
    UNKNOWN_ARTIFACT = "unknown_artifact"
    TIMEOUT = "timeout"
    EXECUTION_FAILED = "execution_failed"


BUDGET_ERROR_CODES = frozenset(
    {ToolErrorCode.BUDGET_EXCEEDED, ToolErrorCode.MAX_ACTIONS_EXCEEDED, ToolErrorCode.SUSPENDED}
)
"""Every ``ToolErrorCode`` that means "no room was left to even attempt this
call" -- as opposed to ``EXECUTION_FAILED``/``TIMEOUT``/``NOT_IMPLEMENTED``/
``UNKNOWN_ARTIFACT``, which mean a call genuinely ran (or was dispatched)
and something about the call itself failed, not this executor's own
admission state. A caller distinguishing "stop the whole cascade/branch,
nothing further can run" from "this one call just errored, everything else
can still proceed" must branch on membership in this set, never treat every
``not observation.ok`` as the former -- a Stage 4 review found exactly this
collapse in both ``repair_operations.RepairOperationExecutor._check()`` and
``branching_runner.BranchingRunner``'s own ``check_timing="final_only"``
initial check, each of which previously read a ``Validator.check()`` that
raised (surfaced as ``EXECUTION_FAILED``) as if it were a budget
exhaustion."""


class ToolExecutionError(BaseModel):
    """One structured execution failure for one ``ToolCall``. Never a bare
    exception/string: ``code`` is always one of ``ToolErrorCode`` so a
    controller can branch on it programmatically, and ``action`` identifies
    which call this refers to even once unwrapped from its
    ``ToolObservation``."""

    code: ToolErrorCode
    action: ToolActionKind
    message: str


class InspectArtifactCall(BaseModel):
    """inspect_artifact: read one artifact's current (or one historical)
    stored state -- no side effects, no cost beyond the call's own
    measured latency."""

    action: Literal[ToolActionKind.INSPECT] = ToolActionKind.INSPECT
    artifact_id: str
    version: Optional[int] = None


class InspectArtifactResult(BaseModel):
    """``stored`` reuses ``StoredArtifact`` directly (ref/content/
    produced_by/status/branch_id/created_at) rather than re-deriving a
    narrower view of it -- same principle ``CostRecord`` already applies to
    ``TokenUsage``: reuse an existing shape in full instead of inventing a
    thinner one. ``found=False`` with ``stored=None`` means the
    ``(artifact_id, version)`` was never stored -- not an error, since
    "does this exist" is a legitimate question to ask."""

    action: Literal[ToolActionKind.INSPECT] = ToolActionKind.INSPECT
    found: bool
    stored: Optional[StoredArtifact] = None


class QueryDependenciesCall(BaseModel):
    """query_dependencies: the change-impact and typed-edge view of one
    artifact, read straight through to ``ArtifactGraphStore.impact_candidates``/
    ``.edges`` -- see that module for why both are pure queries."""

    action: Literal[ToolActionKind.QUERY] = ToolActionKind.QUERY
    artifact_id: str


class QueryDependenciesResult(BaseModel):
    """``impact_candidates`` is ``I(R) = R ∪ Desc_G(R)`` (direction 1) for
    ``artifact_id``, sorted for determinism. ``edges`` is filtered to only
    those naming ``artifact_id`` as either endpoint -- the store's full
    ``edges`` property is every edge ever recorded across every artifact,
    which would make every ``QueryDependenciesResult`` carry the whole
    run's edge history regardless of which artifact was asked about."""

    action: Literal[ToolActionKind.QUERY] = ToolActionKind.QUERY
    impact_candidates: List[str] = Field(default_factory=list)
    edges: List[DependencyEdge] = Field(default_factory=list)


class CheckCall(BaseModel):
    """check: evaluate ``constraints`` against ``artifact`` via whichever
    ``Validator`` the executor was constructed with (Stage 3's default is
    ``constraint_checks.BasicConstraintChecker``). ``context`` is forwarded
    to ``Validator.check()`` unchanged -- e.g. a RECOMPUTE constraint's
    expected/observed values, keyed by ``constraint_id`` under
    ``context["recompute"]`` (see that module), mirroring how Stage 1's
    ``TextAgentValidatorAdapter.check()`` already uses ``context`` to carry
    whatever a given checker needs beyond the bare ``Constraint`` list.

    Always runs a fresh check -- never a cache lookup against previously
    recorded ``Evidence`` -- so a stale verification result can never be
    silently reused in place of re-checking (Stage 3's completion
    criterion: "오래된 검증 결과를 재사용하지 않으며")."""

    action: Literal[ToolActionKind.CHECK] = ToolActionKind.CHECK
    artifact: ArtifactRef
    constraints: List[Constraint]
    context: Dict[str, Any] = Field(default_factory=dict)


class CheckResult(BaseModel):
    action: Literal[ToolActionKind.CHECK] = ToolActionKind.CHECK
    evidence: List[Evidence] = Field(default_factory=list)


class PatchCall(BaseModel):
    """patch: request a narrow, single-target repair. Schema-only through
    Stage 4 (always ``ToolErrorCode.NOT_IMPLEMENTED`` when the executor was
    not given a repair executor -- see ``tool_executor.ToolExecutor``'s
    ``producer_registry``/``constraint_provider`` constructor args). From
    Stage 5 on, an executor configured with those dispatches this for
    real through ``repair_operations.RepairOperationExecutor``: ``target``'s
    own ``ArtifactKind`` picks the recorded ``RepairActionType`` label
    (BLOCK -> PATCH_BLOCK, SECTION -> REGENERATE_SECTION, anything else ->
    REPLAN_SUBGRAPH with this one target) -- a labeling choice only, since
    ``RepairOperationExecutor`` runs all three through the identical
    selective cascade regardless of which label is attached."""

    action: Literal[ToolActionKind.PATCH] = ToolActionKind.PATCH
    target: ArtifactRef
    feedback_text: Optional[str] = None


class ReplanCall(BaseModel):
    """replan: request a wider repair naming one or more targets. Same
    Stage-3-schema-only / Stage-5-real split as ``PatchCall``. ``targets``'
    shape picks the recorded ``RepairActionType``: a single DOCUMENT-kind
    target -> REGENERATE_DOCUMENT (carried out by
    ``RepairOperationExecutor``'s unconditional, non-selective path -- see
    that class's docstring for why that one is deliberately not the same
    mechanism as the other three); a single SECTION-kind target ->
    REGENERATE_SECTION; anything else (multiple targets, or one
    non-document/non-section target) -> REPLAN_SUBGRAPH over exactly the
    given ``targets``. The controller, not this call or its dispatcher,
    decides what belongs in ``targets`` -- this schema only describes the
    shape it was given, it never consults a ``RepairPolicy`` of its own to
    expand or narrow it."""

    action: Literal[ToolActionKind.REPLAN] = ToolActionKind.REPLAN
    targets: List[ArtifactRef]
    reason: Optional[str] = None


class PatchToolResult(BaseModel):
    """Real result of a dispatched ``PatchCall`` (Stage 5) -- wraps
    ``repair_operations.RepairExecutionResult`` unchanged, the same "wrap
    the real thing, don't re-derive a thinner view" principle
    ``CompileToolResult``/``GenerateResult`` already use. ``RepairExecutionResult``
    is defined later in this file (Stage 4 section); referencing it here by
    name works because this module uses ``from __future__ import
    annotations`` (deferred string annotations, resolved by pydantic the
    first time this model is actually used/validated, by which point the
    whole module -- including ``RepairExecutionResult`` -- has finished
    executing)."""

    action: Literal[ToolActionKind.PATCH] = ToolActionKind.PATCH
    result: RepairExecutionResult


class ReplanToolResult(BaseModel):
    """Real result of a dispatched ``ReplanCall`` (Stage 5) -- see
    ``PatchToolResult``'s docstring; the same applies here."""

    action: Literal[ToolActionKind.REPLAN] = ToolActionKind.REPLAN
    result: RepairExecutionResult


class CompileCall(BaseModel):
    """compile_document: dispatches to the real, already-deterministic
    ``LatexCompiler.compile_pdf`` (``src/agents/latex_compiler.py``) --
    direction 2 frames the compiler as a ready-made deterministic tool, and
    unlike patch/replan there is no new decision logic to defer to Stage 4
    here, only plumbing. ``request_id``/``output_dir`` match
    ``compile_pdf``'s own parameters exactly (``base_dir = output_dir /
    request_id``, matching ``main.tex``'s expected location).

    ``timeout_s`` is forwarded to ``compile_pdf``'s own ``timeout``
    parameter (added after a Stage 3 review: a local engine that hangs --
    an infinite loop in a malformed document, say -- would otherwise never
    return at all). Defaults to a finite 120s rather than ``None``
    (unbounded) specifically for calls routed through
    ``ToolExecutor`` -- a direct, non-executor caller of ``compile_pdf``
    elsewhere in the codebase is unaffected, since that function's own
    default stays ``None``."""

    action: Literal[ToolActionKind.COMPILE] = ToolActionKind.COMPILE
    request_id: str
    output_dir: str = "outputs/generations"
    timeout_s: Optional[float] = 120.0


class CompileToolResult(BaseModel):
    """``compile_result`` is ``Optional`` only because ``ToolExecutor``
    could in principle be asked to compile without ever constructing an
    ``EventLogger`` -- in practice it is never actually ``None``:
    ``ToolExecutor`` always constructs a fresh one before calling
    ``compile_pdf``, and every one of that method's four return paths
    (missing ``main.tex``, no engine found, a failed compile, and success)
    calls ``_finalize_compile_result``, which unconditionally sets
    ``event_logger.compile_result`` whenever ``event_logger`` is not
    ``None`` (``latex_compiler.py``). ``compile_success`` always mirrors
    ``compile_pdf``'s own boolean return.

    Typed as the real ``CompileResult`` (``src/evaluation/schemas.py``),
    not ``Any`` -- a second Stage 3 review found that ``Any`` let a round
    trip through ``Checkpoint.model_dump_json()`` /
    ``model_validate_json()`` silently turn this field into a plain
    ``dict`` (pydantic has no schema to reconstruct against for ``Any``),
    so ``observation.output.compile_result.pdf_exists`` worked before
    serializing a checkpoint and broke after restoring one, with the
    underlying data otherwise unchanged. ``src/core/adapters.py`` already
    imports from ``src.evaluation.metrics...`` the same direction, and
    ``src/evaluation/schemas.py`` imports nothing from ``src.core`` at
    all, so this adds no import cycle."""

    action: Literal[ToolActionKind.COMPILE] = ToolActionKind.COMPILE
    compile_success: bool
    compile_result: Optional[CompileResult] = None


class FinishCall(BaseModel):
    """finish: end the episode and record what, if anything, is left
    unresolved -- direction 2: "종료 조건과 미해결 항목을 기록하고 종료".
    Always admissible through ``ToolExecutor.execute`` regardless of
    suspension or exhausted budget/action-count: those conditions are
    exactly when an episode needs to be able to wrap up cleanly, not when
    it should be blocked from doing so."""

    action: Literal[ToolActionKind.FINISH] = ToolActionKind.FINISH
    reason: str
    unresolved_artifact_ids: List[str] = Field(default_factory=list)


class FinishToolResult(BaseModel):
    action: Literal[ToolActionKind.FINISH] = ToolActionKind.FINISH
    acknowledged: bool = True


class GenerateResult(BaseModel):
    """Output of ``ToolExecutor.run_generation()`` -- not reachable through
    ``execute()``'s normal ``ToolCall`` dispatch (see ``ToolActionKind.GENERATE``'s
    docstring). ``production`` is the wrapped ``Producer``'s own
    ``ProductionResult`` unchanged -- ``ok=True`` here means only "the
    executor allowed this call to run and it didn't raise", exactly
    mirroring ``CompileToolResult``'s existing ``ok=True``-with-
    ``compile_success=False`` pattern: a routine ``ProductionStatus.FAILED``
    is data on ``production.status``, not an executor-level error."""

    action: Literal[ToolActionKind.GENERATE] = ToolActionKind.GENERATE
    production: ProductionResult


ToolCall = Union[
    InspectArtifactCall,
    QueryDependenciesCall,
    CheckCall,
    PatchCall,
    ReplanCall,
    CompileCall,
    FinishCall,
]

ToolCallResult = Union[
    InspectArtifactResult,
    QueryDependenciesResult,
    CheckResult,
    CompileToolResult,
    FinishToolResult,
    GenerateResult,
    PatchToolResult,
    ReplanToolResult,
]


class ToolObservation(BaseModel):
    """The one structured shape a controller reads back from every
    ``ToolExecutor.execute()`` call -- Stage 3's completion criterion:
    "controller가 읽을 수 있는 구조화된 observation". Exactly one of
    ``output``/``error`` is set -- never both, and never neither --
    matching ``ProductionResult``'s existing success-xor-failure
    discipline (``model_post_init`` below) rather than leaving a caller to
    infer which one is meaningful, or to handle an observation that
    carries neither, from ``ok`` alone. A Stage 3 review found the
    original version of this check only forbade the *wrong* field from
    being set, not that the *right* one must be -- ``ok=True`` with
    ``output=None`` validated successfully, as did an ``action`` that
    disagreed with what ``output``/``error`` themselves were tagged with
    (e.g. ``action=CHECK`` wrapping a ``FinishToolResult``); both are now
    checked explicitly.

    ``cost`` is populated whenever the executor actually dispatched the
    call (whether it then succeeded or raised) -- a rejected call (budget/
    max-actions/suspended, or an unimplemented action) carries no
    ``CostRecord`` at all, since nothing was actually run; see
    ``tool_executor.ToolExecutor`` for why an admission-rejected call must
    not be charged.
    """

    action: ToolActionKind
    ok: bool
    output: Optional[ToolCallResult] = Field(default=None, discriminator="action")
    error: Optional[ToolExecutionError] = None
    cost: Optional[CostRecord] = None

    def model_post_init(self, __context: object) -> None:
        if self.ok:
            if self.error is not None:
                raise ValueError("ToolObservation with ok=True must not carry an error.")
            if self.output is None:
                raise ValueError("ToolObservation with ok=True must carry an output.")
            if self.output.action != self.action:
                raise ValueError(
                    f"ToolObservation.action={self.action!r} does not match "
                    f"output.action={self.output.action!r}."
                )
        else:
            if self.output is not None:
                raise ValueError("ToolObservation with ok=False must not carry an output.")
            if self.error is None:
                raise ValueError("ToolObservation with ok=False must carry an error.")
            if self.error.action != self.action:
                raise ValueError(
                    f"ToolObservation.action={self.action!r} does not match "
                    f"error.action={self.error.action!r}."
                )


class ToolCallLogEntry(BaseModel):
    """One ``ToolExecutor.execute()`` call paired with the observation it
    produced -- added after a Stage 3 review found that neither the call
    nor the observation was persisted anywhere once ``execute()`` returned
    (e.g. a ``FinishCall``'s ``reason``/``unresolved_artifact_ids`` had no
    storage path at all), which Stage 4's branching re-execution (revision
    direction 4: comparing different actions from the *same* checkpointed
    state) needs connected, not just the ``Evidence``/``CostRecord``
    fragments a call happened to also produce along the way.

    Logged for **every** ``execute()`` call, including a rejected one
    (SUSPENDED/BUDGET_EXCEEDED/MAX_ACTIONS_EXCEEDED/NOT_IMPLEMENTED) --
    "a tool call was attempted and here is exactly what came back" is
    worth keeping regardless of whether it was admitted, same as
    ``InvalidationRecord``/``ActionLogEntry`` never skip logging an
    outcome just because it wasn't the successful one.

    Does not cover ``ToolExecutor.run_generation()`` -- that call's
    ``Producer`` reference is not itself loggable/serializable the way a
    plain ``ToolCall`` is; see that method's docstring for the resulting,
    explicitly scoped gap.
    """

    call: ToolCall = Field(discriminator="action")
    observation: ToolObservation
    created_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Stage 4 ("고정 복구 정책과 분기 재실행") schemas. The behavior that produces
# these lives in ``src/core/repair_operations.py`` (carrying out one
# ``RepairAction`` -- regenerate its named target(s), then cascade through
# ``ArtifactGraphStore.impact_candidates`` re-verifying everything reachable,
# regenerating only what still fails and re-affirming the rest unchanged) and
# ``src/core/branching_runner.py`` (branching the same checkpoint into several
# independent episodes, one per initial policy, continuing each under one
# shared fixed downstream policy). Plain data shapes only here, same
# schema/behavior split as every earlier stage.
# ---------------------------------------------------------------------------


class RepairOutcome(str, Enum):
    """What happened to one artifact a repair touched or tried to touch --
    revision direction 1's three-way split ("수정 대상", "재검사 대상",
    "재생성 대상") made into an explicit, inspectable result instead of a
    bare success/fail bool, and the mechanism behind the Stage 4 completion
    criterion "실패한 복구도 남는다" (a failed repair must also be recorded,
    never silently dropped).
    """

    RESOLVED = "resolved"
    """Regenerated, and a fresh re-check right afterward found at least one
    PASS and no FAIL -- not merely "no FAIL": a check round that came back
    UNKNOWN-only (or produced no evidence at all) is ``INCONCLUSIVE``, never
    this. A first-pass implementation conflated "no FAIL" with "resolved",
    which silently counted an unchecked or unverifiable artifact as fixed --
    see ``INCONCLUSIVE``."""

    STILL_FAILING = "still_failing"
    """Regenerated, but a fresh re-check right afterward still found a
    FAIL. ``RepairOperationExecutor.execute()`` does not retry internally --
    see its docstring for why another attempt is a later round's, driven by
    the caller re-calling a ``RepairPolicy``, not something buried inside
    one execution."""

    INCONCLUSIVE = "inconclusive"
    """Regenerated (or reaffirmed), but the fresh re-check came back with no
    FAIL and no clean PASS either -- every constraint resolved UNKNOWN, or
    the check produced no evidence at all despite being asked. Mirrors the
    Notion plan's own ``recovery_outcome`` vocabulary
    ("not_attempted/success/failure/inconclusive", stage 0's terminology
    doc) -- an artifact in this state must stay flagged stale and must
    never be read as resolved just because nothing explicitly failed."""

    PRODUCTION_FAILED = "production_failed"
    """The ``Producer.produce()`` call itself returned
    ``ProductionStatus.FAILED`` (e.g. exhausted retries) -- nothing new was
    written, so nothing propagates from here."""

    NO_PRODUCER = "no_producer"
    """``target.kind`` has no entry in the executor's producer registry --
    e.g. one of direction 1's finer-grained kinds (DEFINITION/SYMBOL/
    EQUATION/...) that, in this pipeline, has no independent generation call
    of its own; see ``repair_operations.py``'s module docstring for why this
    is a deliberate, bounded scope decision rather than a bug."""


class RegeneratedArtifactRecord(BaseModel):
    """One artifact a repair actually attempted to regenerate (called its
    Producer and, on success, wrote a new version) -- as opposed to one that
    was only re-verified and left untouched because it still passed (see
    ``RepairExecutionResult.reverified_untouched``, which holds bare
    ``ArtifactRef``s instead, since nothing was produced for those).

    ``pre_repair_ref`` is the version being replaced, so a reader can always
    answer "what did this used to be" without a separate lookup even after
    the store has moved past it.
    """

    pre_repair_ref: ArtifactRef
    production: ProductionResult
    outcome: RepairOutcome
    new_ref: Optional[ArtifactRef] = None
    reverify_evidence: List[Evidence] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if self.outcome in (RepairOutcome.RESOLVED, RepairOutcome.STILL_FAILING, RepairOutcome.INCONCLUSIVE):
            if self.new_ref is None:
                raise ValueError(
                    f"RegeneratedArtifactRecord with outcome={self.outcome!r} requires new_ref -- "
                    "something was actually written in all three cases, only the re-check's verdict differs."
                )
        if self.outcome in (RepairOutcome.PRODUCTION_FAILED, RepairOutcome.NO_PRODUCER):
            if self.new_ref is not None:
                raise ValueError(
                    f"RegeneratedArtifactRecord with outcome={self.outcome!r} must not carry new_ref "
                    "-- nothing was actually written."
                )


class RepairExecutionResult(BaseModel):
    """What actually happened when ``RepairOperationExecutor.execute()``
    carried out one ``RepairAction`` end to end: the directly-named
    target(s), plus every artifact the graph-aware cascade additionally
    touched as a consequence -- never the whole conservative
    ``impact_candidates`` set blindly regenerated, only whatever a fresh
    re-check actually found broken (``regenerated``) vs. confirmed still
    good (``reverified_untouched``). This pairing is the concrete evidence
    behind the Stage 4 checklist line "필요한 항목만 재생성한다": a reader can
    see exactly which artifacts were spared, not just trust a claim that
    some were.

    ``left_stale`` is the third, previously-missing bucket a review found:
    a candidate that is currently stale but was never resolved one way or
    the other this round -- either because no ``Constraint`` was ever
    registered for it (nothing to check it against), or because its fresh
    check came back ``INCONCLUSIVE`` (UNKNOWN-only / no evidence). Neither
    case belongs in ``reverified_untouched`` (which means "checked and
    genuinely confirmed still good") -- conflating "confirmed fine" with
    "we simply don't know" was the bug this field exists to stop.

    ``budget_exhausted`` is set when this execution stopped partway through
    its own cascade because the underlying ``ToolExecutor`` ran out of
    budget/actions -- whatever is left in ``left_stale``/unprocessed is
    *not* a resolved state, it is simply where the cascade had to stop.

    ``final_outcome_by_artifact`` is the one authoritative answer to "what
    is this artifact_id's state as of the END of this execution" -- a
    review found that ``regenerated``/``reverified_untouched`` are both
    plain append-in-chronological-order audit logs, and the SAME
    artifact_id can legitimately appear more than once across either list
    within one multi-sweep execution (reaffirmed PASS in an early sweep,
    then re-invalidated and found FAIL in a later one, or the reverse).
    Any consumer that scans those lists for "does this id appear with
    outcome X anywhere" -- rather than "what did the LAST thing that
    happened to this id say" -- silently lets a stale early record outvote
    a later, truer one, in either direction. ``RepairOperationExecutor``
    computes this map internally in true execution order (last write
    wins) and is the one place both this class's own ``overall_outcome``
    and any external caller (``BranchingRunner``'s ``pending``/
    ``unresolved_ids`` bookkeeping) must read from -- never re-derive their
    own interpretation of ``regenerated``/``reverified_untouched`` by
    scanning them independently. Those two lists remain exactly what they
    were: a full, superseded-entries-included audit trail, not pruned or
    reordered -- only this map resolves "which one is current" per id.
    Absent from this map entirely means "never actually resolved one way
    or the other" -- i.e. it is instead in ``left_stale``.
    """

    action: RepairAction
    regenerated: List[RegeneratedArtifactRecord] = Field(default_factory=list)
    reverified_untouched: List[ArtifactRef] = Field(default_factory=list)
    left_stale: List[ArtifactRef] = Field(default_factory=list)
    final_outcome_by_artifact: Dict[str, RepairOutcome] = Field(default_factory=dict)
    overall_outcome: RepairOutcome
    cost: List[CostRecord] = Field(default_factory=list)
    budget_exhausted: bool = False


class RepairRound(BaseModel):
    """Every ``RepairExecutionResult`` produced by actually carrying out
    **every** action one ``RepairPolicy.decide()`` call returned -- never
    just the first one. A review found the original design executed only
    ``decisions[0]`` and silently discarded the rest, which made two
    independent failures named by the same ``decide()`` call (e.g. A and B,
    both FAIL, with no dependency between them) collapse into "only A was
    ever attempted, B's failure vanished from the record."
    """

    round_index: int
    policy_label: str
    executions: List[RepairExecutionResult] = Field(default_factory=list)


class BranchExecutionResult(BaseModel):
    """One independent branch's full result, from an initial checkpoint
    through a fixed number of rounds -- revision direction 4: "동일한
    상태에서 서로 다른 repair action으로 분기"/"효과를 ... checkpoint·행동·후속
    정책 조건에 한정해서 해석한다".

    ``rounds[0]`` is always this branch's own distinguishing initial round
    (decided by this branch's own ``BranchSpec.policy``); every round after
    that is decided by the one shared downstream policy. Keeping all rounds
    in one ordered list (rather than a separate ``initial_round`` field)
    is what makes "execute every action a round's decide() call returned"
    representable at all -- a single ``Optional[RepairExecutionResult]``
    field cannot hold more than one execution per round.
    """

    branch_id: str
    source_checkpoint_id: str
    policy_label: str
    rounds: List[RepairRound] = Field(default_factory=list)
    stopped_reason: str
    final_checkpoint: Checkpoint
    total_cost_tokens: float = 0.0


class BranchComparisonResult(BaseModel):
    """Every branch spun up from one shared source checkpoint -- the unit a
    caller actually reads to compare repair actions "동일한 초기 상태에서"
    (from the same initial state), per the Stage 4 completion criterion.
    """

    source_checkpoint_id: str
    branches: List[BranchExecutionResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 5 ("LangGraph 기반 자율 controller MVP") schemas. Behavior lives in
# ``src/core/controller.py`` (``ControllerPolicy`` Protocol,
# ``RuleBasedController``/``LLMController``, the LangGraph graph itself) --
# same schema/behavior split every earlier stage follows. Nothing below
# imports ``langgraph`` -- these stay plain, testable, serializable data
# shapes regardless of which graph runtime (if any) ends up driving them.
# ---------------------------------------------------------------------------


class ControllerStopReason(str, Enum):
    """Why one controller episode stopped -- direction 2's completion
    criterion ("종료 조건과 미해결 항목을 기록하고 종료") plus the Stage 5
    checklist's explicit anti-infinite-loop/budget-overrun guards, each
    given its own distinct value rather than folded into one generic
    "finished" so a caller can tell a clean stop from a forced one."""

    FINISHED = "finished"
    MAX_STEPS_REACHED = "max_steps_reached"
    REPEATED_ACTION_DETECTED = "repeated_action_detected"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ControllerStepRecord(BaseModel):
    """One (action, observation) pair in the controller's own execution
    trace -- the Stage 5 completion criterion's "관찰 결과가 달라지면 다음
    행동과 복구 범위가 달라지는 실행 trace". Kept directly on
    ``ControllerState`` rather than left only reconstructible from
    ``ArtifactGraphStore.log.tool_calls``, because this list travels with
    the LangGraph state itself across a checkpoint/resume -- a durable
    resume reconstructs the artifact store under a *new* ``branch_id``
    (Stage 2's own "복원 후 다시 LLM을 호출하는 후속 실행은 별도 run으로
    기록한다" rule), which would otherwise leave the pre-resume portion of
    this trace stranded in a branch nothing still reads.

    ``call_summary`` is a short, deterministic one-line identity for the
    dispatched call (e.g. ``"check:doc@3"``, ``"patch:block_2@1"``) -- used
    both for the human-readable trace and, by equality, as this stage's
    "동일 수정 반복" (repeated-fix) detector; see
    ``controller._repeated_action``.

    ``detail`` is the full observation this step actually produced --
    added after a user review found that without it, an ``LLMController``
    asked to decide the *next* action could only see "ok"/an error code
    for what INSPECT/QUERY/CHECK/COMPILE just returned, never the real
    payload (inspected content, queried dependencies, a compile's actual
    diagnostic). ``call_summary`` stays a short identity key (used for
    repeat-detection); ``detail`` is the separate, free-text field for
    "what did this step actually observe" -- see
    ``controller._describe_observation``."""

    step_index: int
    action: ToolActionKind
    call_summary: str
    ok: bool
    error_code: Optional[ToolErrorCode] = None
    detail: str = ""


class ControllerState(BaseModel):
    """Typed state threaded through the Stage 5 LangGraph loop (a
    ``controller`` node decides the next action, a ``tool`` node executes
    it, a conditional edge routes back to ``controller`` or to ``END``) --
    the checklist's own "typed state, conditional routing, checkpoint/
    resume" triad made concrete as one pydantic model.

    ``checkpoint`` is a periodic snapshot of the live ``ArtifactGraphStore``
    driving this episode (refreshed by the ``tool`` node after every real
    dispatch) -- not the live store itself, which stays held by
    ``controller.ControllerRuntime`` across node calls within one episode,
    so a normal step does not fragment one continuous run into a new
    ``branch_id`` per step (``ArtifactGraphStore.from_checkpoint`` always
    requires a fresh ``branch_id``, by design -- see that method's
    docstring). This field exists so (a) LangGraph's own checkpointer can
    snapshot/replay the whole typed state in-process, and (b) an actual
    crash/interruption has something durable
    (``Checkpoint.model_dump_json()``-serializable) to resume from under a
    new ``branch_id`` -- see ``controller.ControllerRuntime.resume_from_checkpoint``.

    ``pending_tool_call`` is set by the ``controller`` node and consumed by
    the ``tool`` node in the very next step -- deciding and executing never
    happen in the same node, mirroring this project's existing
    decide-then-separately-execute discipline (``interfaces.RepairPolicy``'s
    docstring).

    Fields added after a user review of the first Stage 5 pass (full
    detail in ``docs/stage5_langgraph_controller.md``):

    - ``tracked_artifact_ids``: every artifact_id this episode has ever
      dispatched a call against, or that a PATCH/REPLAN's cascade touched
      or discovered broken -- the set a controller checks ``store.
      is_stale()`` against to find unresolved work that has no *fresh*
      Evidence of its own (anything a cascade ran out of budget before
      re-examining stays ``is_stale()``-true without ever producing a
      fresh FAIL to react to -- see ``controller.RuleBasedController.
      decide()``).
    - ``blocked_artifact_ids``: artifacts whose most recent repair attempt
      could not even produce new content (``RepairOutcome.
      PRODUCTION_FAILED``/``NO_PRODUCER``) -- added after a user review
      found ``tracked_artifact_ids``/``is_stale()`` alone insufficient for
      exactly this case: an artifact failing its own *content* check (not
      a dependency that moved) is never marked stale in the first place,
      so a production failure against it leaves no trace in
      ``is_stale()`` at all. Cleared the moment a later attempt actually
      produces something, regardless of whether that content then
      passes, fails, or is unknown (those outcomes DO get a fresh
      Evidence entry to react to).
    - ``actions_taken``/``suspended``/``suspend_reason``: mirror
      ``tool_executor.ToolExecutor``'s own admission state exactly, so a
      durable resume (``controller.ControllerRuntime.resume_from_checkpoint``)
      can reconstruct that executor with the *same* action-count/
      suspension state it had right before the interruption, instead of
      silently reopening a full fresh allowance.
    - ``last_compile_failed``: whether the most recent COMPILE dispatch
      reported ``compile_success=False`` -- lets a rule-based policy stop
      cleanly with an honest reason instead of blindly repeating an
      identical COMPILE call until the repeated-action guard eventually
      intervenes.
    - ``needs_recheck_keys``: ``(artifact_id, constraint_id)`` pairs whose
      cached evidence was just dropped because one of its own inputs moved
      (``ArtifactGraphStore.is_evidence_still_valid`` turned False for
      it), added after a second user review found that dropping an
      invalidated verdict and simply leaving nothing behind was not
      enough: a document with constraint A=FAIL and constraint B=PASS,
      where A's (but not B's) input changes through a relation recorded
      only on ``Evidence.input_versions`` (no matching graph edge, so the
      document itself never becomes ``is_stale()``), lost exactly A's
      now-invalid FAIL and kept B's still-valid PASS -- leaving
      ``_classify()`` reading a clean "PASS" and proceeding straight to
      COMPILE/FINISH without ever re-checking the one constraint that
      actually needed it. Keyed per-constraint, not just per-artifact
      (a third user review: a single CHECK can legitimately name only
      SOME of an artifact's constraints -- ``CheckCall.constraints`` is
      never required to equal the full set a ``ConstraintProvider`` would
      return -- so clearing every pending obligation for an artifact just
      because *any* CHECK touched it could clear A's own obligation while
      only B was actually re-examined). A pair is "owed a fresh CHECK"
      regardless of what evidence (if any) still happens to remain for
      that artifact; cleared only once a fresh CHECK (or a repair's own
      re-verification) actually produces new, currently-valid evidence
      for that *exact* ``(artifact_id, constraint_id)`` pair again.
    """

    document_ref: ArtifactRef
    checkpoint: Checkpoint
    budget: Budget
    current_evidence: List[Evidence] = Field(default_factory=list)
    tracked_artifact_ids: Set[str] = Field(default_factory=set)
    blocked_artifact_ids: Set[str] = Field(default_factory=set)
    needs_recheck_keys: Set[Tuple[str, str]] = Field(default_factory=set)
    compiled: bool = False
    last_compile_failed: bool = False
    step_count: int = 0
    history: List[ControllerStepRecord] = Field(default_factory=list)
    pending_tool_call: Optional["ToolCall"] = Field(default=None, discriminator="action")
    done: bool = False
    stop_reason: Optional[ControllerStopReason] = None
    actions_taken: int = 0
    suspended: bool = False
    suspend_reason: Optional[str] = None


class LLMControllerDecision(BaseModel):
    """The LLM controller's structured-output contract: exactly one next
    ``ToolCall``, plus a short rationale kept only for the execution trace
    -- never fed back into any decision logic itself, the same "an LLM's
    own reported confidence/rationale is not a calibrated signal" caution
    direction 2 gives generally. A plain wrapper, not the bare ``ToolCall``
    union directly, because most structured-output APIs require one
    top-level JSON object as the schema root, not a bare discriminated
    union."""

    call: "ToolCall" = Field(discriminator="action")
    rationale: str = ""


# ---------------------------------------------------------------------------
# Stage 6 ("실패·행동 데이터셋과 human gold 구축") schemas. Behavior lives in
# ``src/core/fault_injection.py`` (controlled perturbations),
# ``src/core/failure_dataset.py`` (natural/injected-failure classification +
# repair gain/damage/cost per checkpoint), ``src/core/annotation.py`` (human
# annotation tooling + disagreement resolution), and
# ``src/core/dataset_split.py`` (train/dev/test split) -- same schema/
# behavior split every earlier stage follows.
# ---------------------------------------------------------------------------


class FaultInjectionKind(str, Enum):
    """The four concrete injected-failure types revision direction 4 names
    under "주입 예시": a theorem/proof missing a needed assumption, a table
    whose denominator was altered, a symbol given a different meaning in
    different sections, and a blueprint whose required components cannot
    jointly be satisfied."""

    MISSING_ASSUMPTION = "missing_assumption"
    ALTERED_DENOMINATOR = "altered_denominator"
    INCONSISTENT_SYMBOL_MEANING = "inconsistent_symbol_meaning"
    UNSATISFIABLE_BLUEPRINT = "unsatisfiable_blueprint"


class InjectedFault(BaseModel):
    """One fault-injection event, as actually carried out against an
    ``ArtifactGraphStore`` (every injector writes through ``add_version``,
    never mutates stored content in place -- same discipline as every other
    writer in this codebase).

    ``pre_injection_refs``/``injected_refs`` are parallel, same-length lists
    -- one ``(old version, new corrupted version)`` pair per artifact this
    single injection event touched. Most injection kinds touch exactly one
    artifact; ``INCONSISTENT_SYMBOL_MEANING`` touches two (the two sections
    disagreeing about the symbol) by construction, so a singular
    ``injected_ref`` field would not fit every kind uniformly.

    ``predicted_affected_artifact_ids`` is the injector's OWN prediction
    (``ArtifactGraphStore.impact_candidates`` at injection time) of what
    should propagate -- kept separate from whatever a later repair/
    classification step actually observes as affected, so "predicted
    propagation" and "observed propagation" are never silently conflated
    into one field (the same discipline ``RepairExecutionResult`` already
    applies to its own "what should be checked" vs "what a fresh check
    actually found").

    ``pre_injection_dependency_ids_by_artifact`` maps each touched
    ``artifact_id`` to its COMPLETE real dependency set (EXPLICIT *and*
    INFERRED) exactly as it stood immediately before this injection wrote
    a new version -- captured via ``ArtifactGraphStore.direct_dependencies``,
    never derived from ``pre_injection_refs[i].read_input_versions``. A
    review caught that ``read_input_versions`` can only ever hold EXPLICIT
    inputs (``add_inferred_edge`` never appends to it, by design -- see
    that method's own docstring), so a caller using it alone as "the known
    reference dependency set" (``failure_dataset.build_failure_record``)
    silently dropped every INFERRED dependency from the baseline, making a
    genuinely-lost INFERRED relation invisible to
    ``current_missing_relations_vs_reference``. This field is also what
    the injector itself uses to decide which INFERRED edges need
    re-registering against the freshly-injected version (see
    ``fault_injection._carry_forward_inferred_edges``) -- the same root
    cause as the original ``[current.ref]`` self-loop bug, one layer
    deeper: re-declaring ``read_input_versions`` alone carries EXPLICIT
    deps forward but silently drops any INFERRED one.
    """

    fault_id: str
    kind: FaultInjectionKind
    pre_injection_refs: List[ArtifactRef]
    injected_refs: List[ArtifactRef]
    pre_injection_dependency_ids_by_artifact: Dict[str, List[str]] = Field(default_factory=dict)
    description: str
    predicted_affected_artifact_ids: List[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if len(self.pre_injection_refs) != len(self.injected_refs):
            raise ValueError(
                "pre_injection_refs and injected_refs must be the same length "
                "(one old->new pair per artifact this injection touched); got "
                f"{len(self.pre_injection_refs)} vs {len(self.injected_refs)}."
            )
        if not self.pre_injection_refs:
            raise ValueError("an InjectedFault must touch at least one artifact.")


class FailureOrigin(str, Enum):
    """Revision direction 4's required separation: "자연 발생 실패와 통제된
    fault injection은 별도로 보고한다" -- kept as an explicit field on every
    ``FailureRecord`` rather than left inferable from whether
    ``injected_fault`` happens to be set, so a query/report can filter on it
    directly without re-deriving the distinction each time."""

    NATURAL = "natural"
    INJECTED = "injected"


class FailureDifficultyProfile(BaseModel):
    """The four classification axes Stage 6's checklist names verbatim
    ("의존 깊이·fan-out·공유 정의 사용·관계 누락에 따라 분류한다"), computed purely
    from the dependency graph state around one failing artifact at the
    moment of failure -- never from that failure's eventual repair outcome,
    which would make difficulty circular with the thing later analysis
    (Stage 7/8) measures difficulty against.

    ``dependency_depth``: length of the longest current upstream chain
    leading into this artifact (0 for an artifact with no current
    dependencies at all).
    ``fan_out``: number of artifacts currently declaring a direct
    dependency on this one (``ArtifactGraphStore.direct_dependents``).
    ``shared_definition_use``: whether this artifact is a non-block/
    section/document kind with ``fan_out > 1`` -- the same "shared
    definition-like artifact with more than one direct dependent" condition
    ``repair_operations.FixedGraphAwareRepairPolicy._scope_for`` already
    uses to decide when a failure's blast radius extends beyond itself;
    reused here rather than redefined, so "difficulty" and "repair scope
    selection" never quietly drift apart.
    ``historical_inferred_relations_count``: number of ``DependencyEdge``s
    of ``relation=INFERRED`` ever recorded touching this artifact (as
    either endpoint), across the FULL edge history -- each one is, by
    construction, a relation that was NOT declared at generation time and
    had to be added after the fact (``ArtifactGraphStore.add_inferred_edge``).
    Named (and renamed from an earlier ``missing_relations_count``) to be
    explicit that this is a cumulative, historical count, never a
    current-state one: a review caught that an INFERRED edge recorded
    against a since-superseded version of this artifact stayed counted
    here forever, even after that version stopped being current and the
    relation it described stopped applying at all (``ArtifactGraphStore.
    direct_dependencies`` would no longer return it) -- reading this field
    as "how many relations are missing RIGHT NOW" is exactly the mistake
    that review was reacting to. It remains a legitimate difficulty signal
    in its own right (an artifact whose documented dependencies have
    needed correction more often is plausibly harder/more error-prone),
    just not a snapshot-at-this-moment one.
    ``current_missing_relations_vs_reference``: ``None`` unless a caller
    supplied a known-correct reference dependency set for this artifact
    (``classify_failure_difficulty``'s ``reference_dependency_ids``) --
    e.g. an ``InjectedFault``'s own pre-injection ``read_input_versions``,
    which IS a genuine, currently-known ground truth for what this
    artifact is supposed to depend on. When supplied, this is the count of
    reference ids NOT currently reflected in ``ArtifactGraphStore.
    direct_dependencies`` -- a properly current-scoped "관계 누락" measure,
    kept as its own field rather than overloading the historical count
    above with a second, incompatible meaning. General automatic
    current-missingness detection (inferring the "correct" dependency set
    with no reference at all) is out of scope here, the same as Stage 2's
    own "no automatic reference scanner" non-goal (``ArtifactGraphStore.
    add_inferred_edge``'s docstring).
    """

    dependency_depth: int = Field(ge=0)
    fan_out: int = Field(ge=0)
    shared_definition_use: bool
    historical_inferred_relations_count: int = Field(ge=0)
    current_missing_relations_vs_reference: Optional[int] = Field(default=None, ge=0)


class FailureRecord(BaseModel):
    """One collected failure event -- natural or injected -- tied to the
    checkpoint it was observed at and classified by its dependency-graph
    difficulty profile. ``dataset_version`` is this record's OWN schema
    version (the Stage 6 completion criterion "버전이 있는 데이터 명세") --
    bumped whenever this shape changes; independent of any single stored
    record's content or of ``src/dataset/schemas.py::BenchmarkItem.
    benchmark_version``, which versions a different thing (the benchmark
    contract, not this failure/action dataset).
    """

    dataset_version: str = "v1"
    failure_id: str
    origin: FailureOrigin
    injected_fault: Optional[InjectedFault] = None
    failing_artifact: ArtifactRef
    evidence: List[Evidence] = Field(default_factory=list)
    difficulty: FailureDifficultyProfile
    source_checkpoint_id: str
    sample_id: Optional[str] = None

    def model_post_init(self, __context: object) -> None:
        if self.origin == FailureOrigin.INJECTED and self.injected_fault is None:
            raise ValueError("FailureRecord with origin=INJECTED requires injected_fault.")
        if self.origin == FailureOrigin.NATURAL and self.injected_fault is not None:
            raise ValueError("FailureRecord with origin=NATURAL must not carry injected_fault.")


class RepairGainScore(BaseModel):
    """Net effect of one candidate repair action from one shared source
    checkpoint -- revision direction 4's "repair gain, 손상, 총비용" made
    concrete on top of Stage 4's already-computed, already-reviewed
    ``BranchExecutionResult``.

    A review caught that the first draft's denominator -- "every artifact
    this branch's own cascade happened to produce a final outcome for" --
    was branch-dependent: a narrow action that only ever attempted 1 of 2
    originally-failing artifacts reported ``gain=1.0`` (1/1) by silently
    never counting the one it ignored, and a branch that left an artifact
    ``left_stale`` (cascaded into it, but never got back to re-checking it
    before stopping) dropped that artifact from the denominator entirely
    for the identical reason -- ``RepairExecutionResult.left_stale``'s own
    docstring already says such an artifact is absent from
    ``final_outcome_by_artifact`` by design, which this class's gain
    computation must NOT read as "nothing to count here."

    Fixed by requiring every caller (``failure_dataset.
    compute_repair_gain_score``) to supply one COMMON
    ``target_artifact_ids`` set -- decided once per source checkpoint,
    from the artifacts that were actually failing before ANY branch ran,
    identical across every branch/candidate action being compared from
    that checkpoint -- rather than letting each branch's own cascade
    silently define its own denominator. ``outcome_counts`` maps each
    ``RepairOutcome`` value seen, PLUS the sentinel key
    ``"left_stale_or_unresolved"`` (never a real ``RepairOutcome`` value --
    an UNKNOWN/left-stale target must never be silently promoted to FAIL or
    quietly dropped either), to how many of ``target_artifact_ids``
    resolved to it; every id in ``target_artifact_ids`` is accounted for in
    exactly one bucket, so ``sum(outcome_counts.values()) ==
    len(target_artifact_ids)`` always. ``gain`` is
    ``outcome_counts.get("resolved", 0) / len(target_artifact_ids)``; the
    only way it is left ``None`` (with ``not_implemented_reason`` set) is
    an empty ``target_artifact_ids`` itself -- never a 0-sized internal
    denominator, since the denominator is now fixed up front.

    ``damage`` is computed over every artifact_id that existed in the
    source checkpoint and was NOT actually failing before any branch ran
    (``failure_dataset._originally_failing_artifact_ids(triggering_evidence)``'s
    complement) -- i.e. the set of artifacts that were healthy BEFORE this
    repair started and were never meant to be touched. A review found the
    original "skip whatever this branch's own cascade happened to visit"
    rule let a previously-healthy artifact that got swept into a WIDE
    repair action (e.g. whole-document regeneration) and came out actually
    FAILING be silently excluded from damage just because it was, in fact,
    visited. "Visited by this branch" and "was this artifact supposed to
    be touched at all" are different questions; only the second one should
    gate damage.

    A *second* review then found that the first fix's replacement rule --
    excluding ``target_artifact_ids`` itself, this class's own GAIN
    denominator -- reintroduced the identical bug through a new door:
    ``target_artifact_ids`` is the COMMON set shared across every branch
    being compared, which also includes whatever a *sibling* branch's own
    history happens to touch. That let an artifact THIS branch swept into
    a wide action and broke escape ITS OWN damage check too, merely
    because a different branch's history also named it. Damage's
    exclusion set must stay anchored to what was actually failing *before
    any branch ran at all* -- a strictly narrower, branch-independent
    fact -- never to the broader, gain-only common target.

    A *third* review found two more problems with this same pairing.
    First: a target_id absent from a branch's own history is not always
    ``LEFT_STALE_OR_UNRESOLVED`` -- if it entered the COMMON
    ``target_artifact_ids`` only because a *sibling* branch's wider action
    redundantly touched an artifact that was never actually broken (not
    because that artifact ever needed fixing), a branch that correctly
    left it alone was scored as if it had ignored a real problem, making
    the MORE selective repair look worse. ``outcome_counts`` gained a
    second sentinel, ``"already_healthy_untouched"``, for exactly this
    case -- never touched, not part of the original trigger, confirmed
    non-broken in THIS branch's own final state -- counted toward
    ``gain``'s numerator alongside ``"resolved"``. Second: ``damage``'s
    exclusion set (the previous paragraph's fix) still assumed every
    non-triggering artifact was healthy before the repair, which is false
    whenever some *other*, unrelated artifact was already broken in the
    source checkpoint for a pre-existing reason -- such an artifact,
    left completely untouched, kept reading as "newly broken" simply for
    remaining exactly as broken as it always was. Both exclusion/inclusion
    sets must check the checkpoint's OWN ground truth
    (``failure_dataset._is_broken_in_final_state``), not just trigger
    membership -- see ``failure_dataset._pre_repair_healthy_artifact_ids``
    and ``compute_repair_gain_score``'s own (rewritten) docstring.
    """

    outcome_counts: Dict[str, int] = Field(default_factory=dict)
    gain: Optional[float] = None
    not_implemented_reason: Optional[str] = None
    damage: ContentPreservationScore
    total_cost_tokens: float = 0.0


class CandidateActionRecord(BaseModel):
    """One ``(checkpoint, candidate repair action(s), downstream result)``
    triple -- the unit revision direction 4's benchmark spec asks to store
    ("초기 상태, 오류 근거, 의존관계, 후보 복구 범위, 독립 평가 기준, 행동별
    결과와 비용"). Links back to the ``FailureRecord`` that triggered this
    branch rather than re-embedding it, so the same failure's several
    candidate actions (one ``CandidateActionRecord`` per ``BranchSpec``)
    never duplicate the difficulty-classification/evidence payload.

    ``actions`` (plural) holds EVERY ``RepairAction`` the branch's own
    initiating round actually executed, not only the first. A review
    caught that a policy's ``decide()`` legitimately returns more than one
    independent action in one call (e.g. ``FixedGraphAwareRepairPolicy``
    returns one action per distinct failing artifact) -- recording only
    ``actions[0]`` left ``outcome``'s gain/damage/cost reflecting the
    combined effect of EVERY action that round actually ran, while
    ``actions`` silently named only one of them, misattributing the full
    combined result to a single, partial cause. This is the same
    "``List[X]`` is a plural contract, consuming only element 0 is an
    unstated cardinality assumption" lesson ``branching_runner.py`` already
    had to learn once, recurring here one layer up, at the recording
    layer rather than the execution layer.
    """

    dataset_version: str = "v1"
    record_id: str
    failure_id: str
    source_checkpoint_id: str
    branch_label: str
    actions: List[RepairAction] = Field(min_length=1)
    outcome: RepairGainScore
    rounds_executed: int = Field(ge=0)


class AnnotationOutcome(str, Enum):
    """Mirrors ``CheckOutcome`` (pass/fail/unknown) but kept as its own enum
    rather than reused directly -- ``CheckOutcome.UNKNOWN`` always carries a
    machine ``UnknownReason``; a human annotator's "unknown" has no such
    structured reason, only free-text ``AnnotationLabel.rationale``, so
    reusing the same type would let a human label silently satisfy
    ``Evidence.model_post_init``'s ``unknown_reason`` requirement with
    nothing meaningful behind it."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class AnnotationStratum(BaseModel):
    """The sampling/reporting strata ``docs/stage1_experiment_design.md``
    §2 names: ``Domain``/``Difficulty``/stage (``src/dataset/schemas.py``/
    ``src/evaluation/schemas.py::Stage``) plus the online-trigger split --
    all optional except ``trigger`` because a failure/action-dataset task
    may not always have a resolvable domain/difficulty/stage (e.g. one
    built from a synthetic fake in a test).

    ``success`` was added after a review found the ORIGINAL Stage 6
    checklist line itself -- "trigger/non-trigger와 성공/실패를 포함한 표본을
    사람이 평가하고" -- names success/failure as its own required coverage
    axis, independent of ``trigger``: a sample can be non-trigger (the
    online verifier never fired) yet still belong to an ultimately FAILED
    trajectory (a miss), or trigger yet belong to a trajectory that
    succeeded after repair. Collapsing these into ``trigger`` alone would
    make the trigger x success/failure 2x2 -- exactly what a verifier
    precision/recall analysis needs to see broken out (true/false
    positives/negatives) -- unrepresentable as a stratum at all. ``stage``
    was added for the same reason the design doc's own stratification list
    already named it but this model had never carried it.
    """

    domain: Optional[str] = None
    difficulty: Optional[str] = None
    stage: Optional[str] = None
    success: Optional[bool] = None
    trigger: bool

    @property
    def key(self) -> str:
        """A single hashable/printable stratum identity, used by
        ``annotation.compute_stratum_weights``/agreement-by-stratum
        reporting -- defined once here so two call sites can never compute
        slightly different strings for "the same" stratum."""
        return (
            f"domain={self.domain}|difficulty={self.difficulty}|stage={self.stage}|"
            f"success={self.success}|trigger={self.trigger}"
        )


class AnnotationTask(BaseModel):
    """One blind-labeling task -- ``docs/stage1_experiment_design.md`` §2's
    protocol made concrete. Deliberately carries no online-validator name or
    result anywhere on this model: showing that to an annotator would make
    the eventual verifier precision/recall comparison circular
    (``interfaces.FinalEvaluator``'s own docstring on the same point).
    ``stratum.trigger`` is used only to build a *balanced* sample and is
    never shown to the annotator and never written back onto the resulting
    label -- it is this model's own field, not something read off the
    label.
    """

    dataset_version: str = "v1"
    task_id: str
    artifact: ArtifactRef
    content_shown: str
    stratum: AnnotationStratum
    sample_id: Optional[str] = None


class AnnotationLabel(BaseModel):
    """One annotator's raw, blind judgment on one ``AnnotationTask``.
    ``is_adjudicator`` marks the (at most one) third-labeler call that only
    happens after the first two disagree -- ``annotation.
    resolve_annotations`` uses it to tell a genuine tie-break label apart
    from a third independent opinion gathered for some other reason."""

    task_id: str
    annotator_id: str
    outcome: AnnotationOutcome
    rationale: str = ""
    is_adjudicator: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class ResolvedLabelStatus(str, Enum):
    AGREED = "agreed"
    ADJUDICATED = "adjudicated"
    UNRESOLVED = "unresolved"


class ResolvedAnnotation(BaseModel):
    """``docs/stage1_experiment_design.md`` §2's disagreement procedure
    applied to one task's raw labels: two independent (blind) labels that
    agree resolve directly (``AGREED``); two that disagree require a third,
    adjudicating label and resolve by majority (``ADJUDICATED``); a
    disagreement with no majority even after adjudication resolves to
    ``outcome=None``/``UNRESOLVED`` and must be excluded from any
    precision/recall denominator -- but the disagreement itself is still
    counted, never silently dropped (§2: "불일치율 자체를 별도로 보고한다";
    see ``AnnotationAgreementReport``).
    """

    task_id: str
    outcome: Optional[AnnotationOutcome] = None
    status: ResolvedLabelStatus
    label_count: int = Field(ge=0)

    def model_post_init(self, __context: object) -> None:
        if self.status == ResolvedLabelStatus.UNRESOLVED and self.outcome is not None:
            raise ValueError("ResolvedAnnotation with status=UNRESOLVED must not carry an outcome.")
        if self.status != ResolvedLabelStatus.UNRESOLVED and self.outcome is None:
            raise ValueError(f"ResolvedAnnotation with status={self.status!r} requires an outcome.")


class AnnotationAgreementReport(BaseModel):
    """Aggregate inter-annotator agreement over one batch of tasks --
    overall and per ``AnnotationStratum.key``, so a stratum with unusually
    high disagreement is visible rather than averaged away."""

    total_tasks: int = Field(ge=0)
    agreed_count: int = Field(ge=0)
    adjudicated_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    disagreement_rate: float = Field(ge=0.0, le=1.0)
    disagreement_rate_by_stratum: Dict[str, float] = Field(default_factory=dict)


class StratumWeight(BaseModel):
    """Inverse-probability weight for one ``AnnotationStratum.key`` --
    Stage 6's completion criterion "계층별 표본을 사용했다면 전체 비율 추정에
    필요한 가중치를 적용한다". ``weight = population_count / sample_count``;
    a stratum sampled at the same rate as its population share gets
    weight 1.0, one oversampled (e.g. rare trigger cases, deliberately
    over-collected for annotation efficiency) gets a weight below 1.0 so a
    population-level estimate built from the sample does not overweight it.
    """

    stratum_key: str
    population_count: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    weight: float = Field(ge=0.0)


class SplitName(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class SplitFamilyKey(BaseModel):
    """The grouping key whose members must never cross a split boundary --
    Stage 6's checklist: "template·topic·graph 구조를 고려한 train/dev/test
    분할을 고정하고". Two samples sharing this key are always assigned to the
    same split together, never split across train/dev vs. test, which is
    what actually prevents the "같은 template/topic/graph 구조가 양쪽에
    중복되어 평가가 쉬워지는" leak the checklist names.
    """

    document_type: str
    topic: str
    graph_structure_bucket: str

    @property
    def key(self) -> str:
        return f"{self.document_type}|{self.topic}|{self.graph_structure_bucket}"


class SplitAssignment(BaseModel):
    dataset_version: str = "v1"
    sample_id: str
    family_key: SplitFamilyKey
    split: SplitName


class DatasetSplit(BaseModel):
    """A fixed train/dev/test partition. ``seed``/``dataset_version`` make
    the assignment reproducible and auditable. ``locked`` defaults to
    ``True``: Stage 6's completion criterion is "held-out test를 정책
    개선에 사용하지 않는다" -- ``dataset_split.held_out_test_sample_ids``
    is the one place that is actually enforced (requires an explicit
    ``acknowledge_test_use=True``, never a silent default), rather than left
    as a convention a future Stage 7 script could violate by accident.
    """

    dataset_version: str = "v1"
    seed: int
    assignments: List[SplitAssignment] = Field(default_factory=list)
    locked: bool = True

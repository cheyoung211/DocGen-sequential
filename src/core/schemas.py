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
from typing import Any, Dict, List, Literal, Optional, Union

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
    reference."""

    status: ProductionStatus
    artifact: Optional[ArtifactRef] = None
    error: Optional[str] = None

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
    the Stage 3 checklist names it. ``PATCH``/``REPLAN`` are intentionally
    not implemented this stage -- see ``tool_executor.ToolExecutor``'s
    docstring: actually deciding regeneration scope and carrying it out is
    Stage 4's "고정 복구 정책" / Stage 5's controller, not Stage 3's.

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
    """patch_block: schema only this stage -- see ``ToolActionKind``'s
    docstring. Defined now so the input shape a Stage 4 executor will
    actually dispatch on is already fixed and testable; calling it through
    ``ToolExecutor`` always returns ``ToolErrorCode.NOT_IMPLEMENTED``."""

    action: Literal[ToolActionKind.PATCH] = ToolActionKind.PATCH
    target: ArtifactRef
    feedback_text: Optional[str] = None


class ReplanCall(BaseModel):
    """replan_subgraph: schema only this stage -- see ``PatchCall``'s
    docstring; the same applies here."""

    action: Literal[ToolActionKind.REPLAN] = ToolActionKind.REPLAN
    targets: List[ArtifactRef]
    reason: Optional[str] = None


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
    """Regenerated, and a fresh re-check right afterward found no FAIL."""

    STILL_FAILING = "still_failing"
    """Regenerated, but a fresh re-check right afterward still found a
    FAIL. ``RepairOperationExecutor.execute()`` does not retry internally --
    see its docstring for why another attempt is a later round's, driven by
    the caller re-calling a ``RepairPolicy``, not something buried inside
    one execution."""

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
        if self.outcome == RepairOutcome.RESOLVED and self.new_ref is None:
            raise ValueError("RegeneratedArtifactRecord with outcome=RESOLVED requires new_ref.")
        if self.outcome == RepairOutcome.STILL_FAILING and self.new_ref is None:
            raise ValueError("RegeneratedArtifactRecord with outcome=STILL_FAILING requires new_ref.")
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
    """

    action: RepairAction
    regenerated: List[RegeneratedArtifactRecord] = Field(default_factory=list)
    reverified_untouched: List[ArtifactRef] = Field(default_factory=list)
    overall_outcome: RepairOutcome
    cost: List[CostRecord] = Field(default_factory=list)


class BranchExecutionResult(BaseModel):
    """One independent branch's full result, from an initial checkpoint
    through a fixed number of rounds -- revision direction 4: "동일한
    상태에서 서로 다른 repair action으로 분기"/"효과를 ... checkpoint·행동·후속
    정책 조건에 한정해서 해석한다".

    ``initial_action``/``initial_round`` (``None`` if the branch's own
    initial policy decided there was nothing to do) are kept separate from
    ``downstream_rounds`` precisely so a reader can isolate "what this
    branch's own distinguishing choice was" from "what every branch did
    afterward under the same shared policy" -- conflating the two would
    make the checkpoint-branch comparison meaningless (``docs/
    stage1_experiment_design.md`` §5, point 3: "그렇지 않으면 1단계 선택 효과와
    이후 정책 효과가 혼재된다").
    """

    branch_id: str
    source_checkpoint_id: str
    policy_label: str
    initial_action: Optional[RepairAction] = None
    initial_round: Optional[RepairExecutionResult] = None
    downstream_rounds: List[RepairExecutionResult] = Field(default_factory=list)
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

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
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


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
    """

    constraint_id: str
    outcome: CheckOutcome
    checked_artifact: ArtifactRef
    detail: str = ""
    unknown_reason: Optional[UnknownReason] = None

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
    set, and the full invalidation/action/evidence/cost history -- no live
    Python object references back into the store that produced it. That is
    what lets ``ArtifactGraphStore.from_checkpoint`` reconstruct a fully
    independent store sharing no mutable state with the original, which is
    the whole point of "branch into different repair actions from the same
    state" (direction 4): a ``Checkpoint`` could in principle be serialized
    via ``model_dump_json()`` and handed to a different process entirely.
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

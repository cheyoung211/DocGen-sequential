"""Producer / Validator / RepairPolicy / FinalEvaluator interfaces (Stage 1
of the "Revision 2026.10.15" plan).

These are ``typing.Protocol`` classes, not ``abc.ABC``: today's agents
(``src/agents/planner.py::PlannerAgent``, ``src/agents/text_agent.py::
TextAgent``, ...) already have the right *shape* of method under a
pipeline-specific name (``generate_plan``, ``process_node``, ...). A
Protocol lets ``adapters.py`` wrap them without editing a single line of
agent code -- the point of Stage 1 is to prove the existing pipeline is
*expressible* in these interfaces, not to rewrite it (that rewrite is
Stage 5's LangGraph controller MVP).

The central design point of this stage: **trigger, diagnostic feedback,
and repair scope are three independent axes**, not one fused decision the
way today's ``DocGenPipeline._repair_and_recompile``
(``scripts/run_full_pipeline.py:425-537``) makes it. Concretely:

- *trigger* -- which checks even run at all -> ``Validator.config``
  (``ValidatorConfig``, the same shape as today's ``DEFAULT_ENABLED_VALIDATORS``).
- *diagnostic feedback* -- whether a failing check's message text is handed
  back to the Producer -> ``RepairAction.use_diagnostic_feedback``.
- *repair scope* -- how much gets regenerated/re-checked ->
  ``RepairAction.action_type``.

A ``RepairPolicy`` can vary any one of these while holding the others
fixed. For example "blind retry" (``adapters.BlindRetryRepairPolicy``) is
trigger=on, feedback=off, scope=constant; today's actual pipeline
(``adapters.TodayFixedRepairPolicyAdapter``) is trigger=on, feedback=on,
scope=block-or-section depending on the error location. Neither axis is
hard-coded into the Protocols below -- only into a given policy's
*implementation* of ``decide()``, so a new policy can pick a new
combination without changing this file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.core.schemas import (
    ArtifactRef,
    Budget,
    Constraint,
    Evidence,
    FinalEvaluationResult,
    RepairAction,
)


@runtime_checkable
class Producer(Protocol):
    """Generates or regenerates one artifact's content.

    Maps to today's ``PlannerAgent.generate_plan`` / ``TextAgent.process_node``
    / ``TextAgent.regenerate_blocks`` / ``ImageAgent.process_figure_node`` --
    see ``adapters.PlannerAgentProducer`` / ``adapters.TextAgentProducer``.
    ``context`` carries whatever the wrapped agent actually needs
    (``DocumentGraph``, ``request_id``, ``event_logger``, ...) -- kept
    generic here because today's agents each need a different set of
    pipeline-specific objects that this Stage does not standardize away
    (that standardization is Stage 2's artifact graph).
    """

    def produce(self, artifact_id: str, context: Dict[str, Any]) -> ArtifactRef:
        ...


class ValidatorConfig(Dict[str, bool]):
    """Which named checks are "on" for a given run -- the *trigger* axis.

    Deliberately the same shape as today's ``DEFAULT_ENABLED_VALIDATORS``
    (``src/agents/text_agent.py:89-99``) / ``DEFAULT_ENABLED_PLANNER_VALIDATORS``
    (``src/agents/planner.py:368-373``): a plain ``{code: bool}`` mapping, so
    today's config dict is already a valid ``ValidatorConfig`` with no
    conversion step.
    """


@runtime_checkable
class Validator(Protocol):
    """Checks one artifact against a set of constraints and returns
    structured ``Evidence`` -- never a bare bool, and never the artifact's
    final quality score (that is ``FinalEvaluator``'s job, kept separate on
    purpose -- see its docstring below)."""

    config: ValidatorConfig

    def check(
        self,
        artifact: ArtifactRef,
        constraints: List[Constraint],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Evidence]:
        ...


@runtime_checkable
class RepairPolicy(Protocol):
    """Decides what to do about failing/unknown ``Evidence``: which scope to
    regenerate (``RepairAction.action_type``) and whether to pass along
    diagnostic feedback (``RepairAction.use_diagnostic_feedback``) -- see
    this module's docstring for why those are independent axes.

    Reads ``budget`` so a cost-aware policy can choose a narrower scope, or
    stop entirely (return ``[]``), once ``budget.remaining`` runs low; a
    budget-blind policy is simply one that ignores the argument.
    ``dependency_edges`` is accepted so a graph-aware policy (Stage 4+) can
    widen or narrow scope using them -- a policy that does not use the
    dependency graph at all (e.g. ``adapters.BlindRetryRepairPolicy``) is
    simply one that ignores this argument too.
    """

    def decide(
        self,
        evidence: List[Evidence],
        dependency_edges: List[Any],
        budget: Budget,
    ) -> List[RepairAction]:
        ...


@runtime_checkable
class FinalEvaluator(Protocol):
    """Independently judges the *finished* document: quality, content
    preservation, and (once human gold exists) online-verifier precision/
    recall.

    Must NOT take the run's online ``Evidence``/``VerificationEvent`` stream
    as an input to the quality score it produces. Doing so would let the
    same mechanism that triggered a retry also certify that the retry
    worked -- exactly the "온라인 keyword rule과 최종 평가의 중복" problem
    the revision review calls out (see ``docs/stage1_experiment_design.md``,
    "독립 평가 rubric" section). The online stream is only a valid input to
    the *verifier precision/recall* axis, where it is compared against an
    independent gold label it did not help produce -- never substituted for
    the gold label itself. This is why ``evaluate()`` takes no event-log
    parameter at all: a conforming implementation has nothing to smuggle it
    in through.
    """

    def evaluate(
        self, final_artifacts: List[ArtifactRef], benchmark_contract: Any
    ) -> FinalEvaluationResult:
        ...

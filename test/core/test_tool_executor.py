"""Coverage for src/core/tool_executor.py (Stage 3): dispatch for
inspect/query/check/compile, NOT_IMPLEMENTED for patch/replan, FINISH's
suspension exemption, and budget/max-actions/suspend-resume enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.artifact_graph import ArtifactGraphStore
from src.core.constraint_checks import BasicConstraintChecker
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    Budget,
    CheckCall,
    CompileCall,
    Constraint,
    ConstraintKind,
    FinishCall,
    InspectArtifactCall,
    PatchCall,
    QueryDependenciesCall,
    ReplanCall,
    ToolErrorCode,
)
from src.core.tool_executor import ToolExecutor


def _executor(store: ArtifactGraphStore | None = None, *, cap: float = 1000.0, max_actions: int = 100) -> ToolExecutor:
    store = store or ArtifactGraphStore()
    return ToolExecutor(store, BasicConstraintChecker(store), budget=Budget(cap=cap), max_actions=max_actions)


# -- INSPECT ------------------------------------------------------------


def test_inspect_found_returns_the_stored_artifact() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.BLOCK, content="hello")
    executor = _executor(store)
    obs = executor.execute(InspectArtifactCall(artifact_id="a"))
    assert obs.ok is True
    assert obs.output.found is True
    assert obs.output.stored.content == "hello"


def test_inspect_not_found_is_ok_with_found_false_not_an_error() -> None:
    executor = _executor()
    obs = executor.execute(InspectArtifactCall(artifact_id="nope"))
    assert obs.ok is True
    assert obs.output.found is False
    assert obs.output.stored is None


def test_inspect_a_specific_historical_version() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.BLOCK, content="v1")
    store.add_version("a", ArtifactKind.BLOCK, content="v2")
    executor = _executor(store)
    obs = executor.execute(InspectArtifactCall(artifact_id="a", version=1))
    assert obs.output.stored.content == "v1"
    assert obs.output.stored.ref.version == 1


def test_inspecting_the_returned_stored_artifact_does_not_corrupt_the_store() -> None:
    """Mutating a ToolObservation's own output must never retroactively
    change the store's history -- same deep-copy discipline Stage 2
    already enforces on every ArtifactGraphStore accessor."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.BLOCK, content="original")
    executor = _executor(store)
    obs = executor.execute(InspectArtifactCall(artifact_id="a"))
    obs.output.stored.content = "corrupted"
    assert store.latest("a").content == "original"


# -- QUERY ----------------------------------------------------------------


def test_query_returns_impact_candidates_and_only_relevant_edges() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="def")
    store.add_version("thm_1", ArtifactKind.CLAIM, content="thm", read_input_versions=[store.latest("def_1").ref])
    store.add_version("unrelated", ArtifactKind.TABLE, content="t")
    executor = _executor(store)
    obs = executor.execute(QueryDependenciesCall(artifact_id="def_1"))
    assert obs.output.impact_candidates == ["def_1", "thm_1"]
    assert len(obs.output.edges) == 1
    assert obs.output.edges[0].to_artifact.artifact_id == "thm_1"


def test_query_is_a_pure_read_does_not_change_store_state() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.BLOCK, content="x")
    executor = _executor(store)
    before = store.stale_artifacts
    executor.execute(QueryDependenciesCall(artifact_id="a"))
    executor.execute(QueryDependenciesCall(artifact_id="a"))
    assert store.stale_artifacts == before


# -- CHECK ------------------------------------------------------------------


def test_check_dispatches_to_the_configured_checker_and_records_evidence() -> None:
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.BLOCK, content="x")
    executor = _executor(store)
    constraint = Constraint(constraint_id="c1", kind=ConstraintKind.SCHEMA_CONTRACT, description="x", applies_to=[])
    obs = executor.execute(CheckCall(artifact=store.latest("a").ref, constraints=[constraint]))
    assert obs.ok is True
    assert obs.output.evidence[0].outcome.value == "pass"
    assert len(store.log.evidence) == 1


def test_check_runs_fresh_every_time_never_reuses_a_prior_result() -> None:
    """Completion criterion: "오래된 검증 결과를 재사용하지 않으며" --
    calling check() twice on evidence that is designed to change (by
    passing different recompute values) must reflect the *new* call's
    inputs, not whatever was cached from the first one."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.BLOCK, content="x")
    executor = _executor(store)
    ref = store.latest("a").ref
    constraint = Constraint(constraint_id="r1", kind=ConstraintKind.RECOMPUTE, description="x")

    first = executor.execute(
        CheckCall(artifact=ref, constraints=[constraint], context={"recompute": {"r1": {"expected": 1, "observed": 1}}})
    )
    second = executor.execute(
        CheckCall(artifact=ref, constraints=[constraint], context={"recompute": {"r1": {"expected": 1, "observed": 999}}})
    )
    assert first.output.evidence[0].outcome.value == "pass"
    assert second.output.evidence[0].outcome.value == "fail"
    assert len(store.log.evidence) == 2


# -- COMPILE -----------------------------------------------------------------


def test_compile_dispatches_to_the_real_latex_compiler(tmp_path: Path) -> None:
    request_id = "req1"
    project_dir = tmp_path / request_id
    project_dir.mkdir()
    (project_dir / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}Hello\\end{document}\n", encoding="utf-8"
    )
    executor = _executor()
    obs = executor.execute(CompileCall(request_id=request_id, output_dir=str(tmp_path)))
    assert obs.ok is True
    assert obs.output.compile_success is True
    assert obs.output.compile_result is not None
    assert obs.output.compile_result.pdf_exists is True


def test_compile_reports_failure_when_main_tex_is_missing(tmp_path: Path) -> None:
    executor = _executor()
    obs = executor.execute(CompileCall(request_id="does_not_exist", output_dir=str(tmp_path)))
    assert obs.ok is True  # the tool call itself executed without raising
    assert obs.output.compile_success is False
    assert obs.output.compile_result.first_error_type == "main_tex_missing"


def test_compile_reports_failure_for_a_real_latex_syntax_error(tmp_path: Path) -> None:
    """Distinct from the missing-main.tex branch above: main.tex exists
    and the engine actually runs, but the document itself is invalid --
    a different failure path through compile_pdf entirely, exercised
    separately so a fix for one branch is not mistaken for covering both
    (docgen_stage_review_rigor pattern #6: check every branch, not just
    the first one tested)."""
    request_id = "req_bad"
    project_dir = tmp_path / request_id
    project_dir.mkdir()
    (project_dir / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}\\badcommandthatdoesnotexist\\end{document}\n",
        encoding="utf-8",
    )
    executor = _executor()
    obs = executor.execute(CompileCall(request_id=request_id, output_dir=str(tmp_path)))
    assert obs.ok is True
    assert obs.output.compile_success is False
    assert obs.output.compile_result.first_error_type == "undefined_control_sequence"
    assert obs.output.compile_result.pdf_exists is False


# -- PATCH / REPLAN (schema-only this stage) --------------------------------


@pytest.mark.parametrize(
    "call",
    [
        PatchCall(target=ArtifactRef(artifact_id="a", version=1, kind=ArtifactKind.BLOCK)),
        ReplanCall(targets=[ArtifactRef(artifact_id="a", version=1, kind=ArtifactKind.BLOCK)]),
    ],
)
def test_patch_and_replan_are_not_implemented(call) -> None:
    executor = _executor()
    obs = executor.execute(call)
    assert obs.ok is False
    assert obs.error.code == ToolErrorCode.NOT_IMPLEMENTED
    assert obs.cost is None


def test_patch_and_replan_do_not_consume_budget_or_action_count() -> None:
    """Nothing was actually run, so neither the token budget nor the
    action-count cap -- both of which exist to bound real work -- should
    move at all."""
    executor = _executor(cap=10.0, max_actions=5)
    before_budget = executor.budget.spent
    before_actions = executor.actions_taken
    executor.execute(PatchCall(target=ArtifactRef(artifact_id="a", version=1, kind=ArtifactKind.BLOCK)))
    executor.execute(ReplanCall(targets=[ArtifactRef(artifact_id="a", version=1, kind=ArtifactKind.BLOCK)]))
    assert executor.budget.spent == before_budget
    assert executor.actions_taken == before_actions


# -- FINISH -------------------------------------------------------------------


def test_finish_is_ok_and_acknowledged() -> None:
    executor = _executor()
    obs = executor.execute(FinishCall(reason="done", unresolved_artifact_ids=["a"]))
    assert obs.ok is True
    assert obs.output.acknowledged is True


def test_finish_is_admissible_even_while_suspended_by_budget_exhaustion() -> None:
    executor = _executor(cap=0.0)
    blocked = executor.execute(InspectArtifactCall(artifact_id="a"))
    assert blocked.ok is False
    assert executor.is_suspended is True

    obs = executor.execute(FinishCall(reason="wrap up despite exhaustion"))
    assert obs.ok is True


def test_finish_is_admissible_even_while_suspended_by_max_actions() -> None:
    executor = _executor(max_actions=1)
    executor.execute(InspectArtifactCall(artifact_id="a"))
    assert executor.is_suspended is True

    obs = executor.execute(FinishCall(reason="wrap up"))
    assert obs.ok is True


# -- budget / max-actions / suspend-resume -----------------------------------


def test_budget_exhausted_at_construction_rejects_every_non_finish_call() -> None:
    executor = _executor(cap=0.0)
    obs = executor.execute(InspectArtifactCall(artifact_id="a"))
    assert obs.ok is False
    assert obs.error.code == ToolErrorCode.BUDGET_EXCEEDED
    assert obs.cost is None  # rejected calls are never charged
    assert executor.is_suspended is True


def test_max_actions_suspends_after_the_limit_is_reached() -> None:
    """The executor auto-suspends as soon as actions_taken reaches
    max_actions (right after the call that brings it there, in _charge) --
    so by the time a *third* call is even attempted, admission already
    sees is_suspended=True and reports SUSPENDED, not a fresh
    MAX_ACTIONS_EXCEEDED discovery. The original cause survives in
    suspend_reason."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.BLOCK, content="x")
    executor = _executor(store, max_actions=2)

    first = executor.execute(InspectArtifactCall(artifact_id="a"))
    assert executor.is_suspended is False
    second = executor.execute(InspectArtifactCall(artifact_id="a"))
    assert executor.is_suspended is True  # auto-suspended immediately after reaching the cap
    assert "max action" in (executor.suspend_reason or "")
    third = executor.execute(InspectArtifactCall(artifact_id="a"))

    assert first.ok and second.ok
    assert third.ok is False
    assert third.error.code == ToolErrorCode.SUSPENDED
    assert executor.actions_taken == 2  # the rejected third call never incremented this


def test_suspend_rejects_everything_except_finish_until_resumed() -> None:
    executor = _executor()
    executor.suspend("manual pause for inspection")
    obs = executor.execute(InspectArtifactCall(artifact_id="a"))
    assert obs.ok is False
    assert obs.error.code == ToolErrorCode.SUSPENDED

    executor.resume()
    assert executor.is_suspended is False
    obs2 = executor.execute(InspectArtifactCall(artifact_id="a"))
    assert obs2.ok is True


def test_resume_with_extra_budget_and_actions_actually_raises_the_caps() -> None:
    executor = _executor(cap=0.0, max_actions=0)
    executor.execute(InspectArtifactCall(artifact_id="a"))  # suspends immediately (already exhausted)
    assert executor.is_suspended is True

    executor.resume(extra_budget=100.0, extra_actions=5)
    assert executor.is_suspended is False
    obs = executor.execute(InspectArtifactCall(artifact_id="a"))
    assert obs.ok is True


def test_a_rejected_call_is_never_charged_a_cost_record() -> None:
    store = ArtifactGraphStore()
    executor = _executor(store, cap=0.0)
    before = len(store.log.costs)
    executor.execute(InspectArtifactCall(artifact_id="a"))
    assert len(store.log.costs) == before


def test_every_dispatched_call_is_logged_to_the_store_even_zero_token_ones() -> None:
    """Completion criterion: a dispatched tool call's cost must never be
    missing from the ledger, even when it has zero token cost (every
    Stage 3 action is deterministic/local, so none of them incur tokens
    today -- see docs/stage3_tool_executor.md)."""
    store = ArtifactGraphStore()
    store.add_version("a", ArtifactKind.BLOCK, content="x")
    executor = _executor(store)
    executor.execute(InspectArtifactCall(artifact_id="a"))
    assert len(store.log.costs) == 1
    assert store.log.costs[0].stage == "tool_executor:inspect"


def test_dispatch_exception_is_converted_to_a_structured_error_not_raised() -> None:
    """An internal dispatch failure must still come back as a
    ToolObservation(ok=False, error=...), charged like any other executed
    call -- never propagate past execute() as a raw exception, mirroring
    Producer.produce()'s own "never raise past this boundary" contract."""
    store = ArtifactGraphStore()
    executor = _executor(store)
    constraint = Constraint(constraint_id="c1", kind=ConstraintKind.SCHEMA_CONTRACT, description="x", applies_to=[])
    # A ref naming an artifact_id with a version that was never stored at
    # all makes the underlying checker/store raise internally.
    bad_ref = ArtifactRef(artifact_id="never_stored", version=1, kind=ArtifactKind.BLOCK)
    obs = executor.execute(CheckCall(artifact=bad_ref, constraints=[constraint]))
    assert obs.ok is True  # applies_to is empty, so this particular call actually succeeds
    # Exercise a genuine internal failure instead: a Constraint whose
    # applies_to references an artifact_id/version combination that does
    # not exist is handled gracefully by _check_schema_contract (FAIL, not
    # a raise) -- so to prove execute()'s except-path works at all, call
    # through a checker that actually raises.
    class _RaisingChecker:
        config: dict = {}

        def check(self, artifact, constraints, context=None):
            raise RuntimeError("checker exploded")

    raising_executor = ToolExecutor(store, _RaisingChecker(), budget=Budget(cap=1000.0), max_actions=10)
    obs2 = raising_executor.execute(CheckCall(artifact=bad_ref, constraints=[constraint]))
    assert obs2.ok is False
    assert obs2.error.code == ToolErrorCode.EXECUTION_FAILED
    assert "checker exploded" in obs2.error.message
    assert obs2.cost is not None
    assert obs2.cost.succeeded is False

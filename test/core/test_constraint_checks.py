"""Coverage for src/core/constraint_checks.py (Stage 3 of the "Revision
2026.10.15" plan): BasicConstraintChecker over SCHEMA_CONTRACT/RECOMPUTE,
with CSP_SMT/SYMBOLIC left explicitly unimplemented."""

from __future__ import annotations

import pytest

from src.core.artifact_graph import ArtifactGraphStore
from src.core.constraint_checks import BasicConstraintChecker
from src.core.interfaces import Validator
from src.core.schemas import (
    ArtifactKind,
    ArtifactRef,
    CheckOutcome,
    Constraint,
    ConstraintKind,
    UnknownReason,
)


def _store_with_one_artifact(artifact_id: str = "a") -> tuple[ArtifactGraphStore, ArtifactRef]:
    store = ArtifactGraphStore()
    store.add_version(artifact_id, ArtifactKind.BLOCK, content="content")
    return store, store.latest(artifact_id).ref


def test_checker_satisfies_the_validator_protocol() -> None:
    store, _ = _store_with_one_artifact()
    assert isinstance(BasicConstraintChecker(store), Validator)


def test_schema_contract_passes_when_every_applies_to_ref_exists() -> None:
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="def")
    store.add_version("thm_1", ArtifactKind.CLAIM, content="thm")
    checker = BasicConstraintChecker(store)
    constraint = Constraint(
        constraint_id="c1",
        kind=ConstraintKind.SCHEMA_CONTRACT,
        description="thm must cite def_1",
        applies_to=[store.latest("def_1").ref],
    )
    evidence = checker.check(store.latest("thm_1").ref, [constraint])
    assert len(evidence) == 1
    assert evidence[0].outcome == CheckOutcome.PASS
    assert evidence[0].constraint_id == "c1"


def test_schema_contract_fails_and_names_every_missing_ref() -> None:
    store, checked = _store_with_one_artifact("thm_1")
    checker = BasicConstraintChecker(store)
    missing_1 = ArtifactRef(artifact_id="ghost_1", version=1, kind=ArtifactKind.DEFINITION)
    missing_2 = ArtifactRef(artifact_id="ghost_2", version=1, kind=ArtifactKind.TABLE)
    constraint = Constraint(
        constraint_id="c1", kind=ConstraintKind.SCHEMA_CONTRACT, description="x", applies_to=[missing_1, missing_2]
    )
    evidence = checker.check(checked, [constraint])
    assert evidence[0].outcome == CheckOutcome.FAIL
    assert "ghost_1" in evidence[0].detail and "ghost_2" in evidence[0].detail


def test_schema_contract_rejects_a_stale_version_number_not_just_missing_id() -> None:
    """A required artifact that exists at v1 but is being asked about at a
    version number that was never actually stored (e.g. a stale pointer to
    a since-superseded ref) must still be reported as missing -- existence
    of the artifact_id alone is not existence of the exact version named."""
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="v1")
    store.add_version("thm_1", ArtifactKind.CLAIM, content="thm")
    never_stored_version = ArtifactRef(artifact_id="def_1", version=99, kind=ArtifactKind.DEFINITION)
    checker = BasicConstraintChecker(store)
    constraint = Constraint(
        constraint_id="c1", kind=ConstraintKind.SCHEMA_CONTRACT, description="x", applies_to=[never_stored_version]
    )
    evidence = checker.check(store.latest("thm_1").ref, [constraint])
    assert evidence[0].outcome == CheckOutcome.FAIL
    assert "def_1@v99" in evidence[0].detail


def test_schema_contract_with_empty_applies_to_is_vacuously_pass() -> None:
    store, checked = _store_with_one_artifact()
    checker = BasicConstraintChecker(store)
    constraint = Constraint(constraint_id="c1", kind=ConstraintKind.SCHEMA_CONTRACT, description="x", applies_to=[])
    evidence = checker.check(checked, [constraint])
    assert evidence[0].outcome == CheckOutcome.PASS


def test_recompute_without_context_is_unknown_not_checked() -> None:
    store, checked = _store_with_one_artifact()
    checker = BasicConstraintChecker(store)
    constraint = Constraint(constraint_id="r1", kind=ConstraintKind.RECOMPUTE, description="sum")
    evidence = checker.check(checked, [constraint])
    assert evidence[0].outcome == CheckOutcome.UNKNOWN
    assert evidence[0].unknown_reason == UnknownReason.NOT_CHECKED


def test_recompute_matches_within_tolerance() -> None:
    store, checked = _store_with_one_artifact()
    checker = BasicConstraintChecker(store)
    constraint = Constraint(constraint_id="r1", kind=ConstraintKind.RECOMPUTE, description="sum")
    evidence = checker.check(
        checked, [constraint], context={"recompute": {"r1": {"expected": 10.0, "observed": 10.04, "tolerance": 0.05}}}
    )
    assert evidence[0].outcome == CheckOutcome.PASS


def test_recompute_fails_outside_tolerance() -> None:
    store, checked = _store_with_one_artifact()
    checker = BasicConstraintChecker(store)
    constraint = Constraint(constraint_id="r1", kind=ConstraintKind.RECOMPUTE, description="sum")
    evidence = checker.check(
        checked, [constraint], context={"recompute": {"r1": {"expected": 10.0, "observed": 9.0}}}
    )
    assert evidence[0].outcome == CheckOutcome.FAIL
    assert "9.0" in evidence[0].detail and "10.0" in evidence[0].detail


@pytest.mark.parametrize("kind", [ConstraintKind.CSP_SMT, ConstraintKind.SYMBOLIC])
def test_csp_smt_and_symbolic_are_unknown_not_representable(kind: ConstraintKind) -> None:
    """No narrow constraint instance needing Z3/SymPy exists yet, and
    neither is a dependency of this project -- see this module's own
    docstring and docs/stage3_tool_executor.md. These two kinds must never
    silently resolve to PASS."""
    store, checked = _store_with_one_artifact()
    checker = BasicConstraintChecker(store)
    constraint = Constraint(constraint_id="s1", kind=kind, description="x")
    evidence = checker.check(checked, [constraint])
    assert evidence[0].outcome == CheckOutcome.UNKNOWN
    assert evidence[0].unknown_reason == UnknownReason.NOT_REPRESENTABLE


def test_set_enabled_rejects_an_unimplemented_kind() -> None:
    """CSP_SMT/SYMBOLIC have no trigger to toggle at all -- they are not
    "disabled", they are "not implemented". Trying to enable one must fail
    loudly, not silently create a trigger that then does nothing."""
    store, _ = _store_with_one_artifact()
    checker = BasicConstraintChecker(store)
    with pytest.raises(ValueError):
        checker.set_enabled(ConstraintKind.CSP_SMT, True)


def test_config_is_a_read_only_snapshot_not_the_live_trigger_state() -> None:
    """Mirrors adapters.TextAgentValidatorAdapter.config's exact contract:
    mutating the returned dict must not affect what check() actually runs
    next -- only set_enabled() does."""
    store, checked = _store_with_one_artifact()
    checker = BasicConstraintChecker(store)
    snapshot = checker.config
    snapshot[ConstraintKind.SCHEMA_CONTRACT.value] = False

    constraint = Constraint(constraint_id="c1", kind=ConstraintKind.SCHEMA_CONTRACT, description="x", applies_to=[])
    evidence = checker.check(checked, [constraint])
    assert evidence[0].outcome == CheckOutcome.PASS  # still enabled; snapshot mutation had no effect


def test_set_enabled_false_then_check_reports_not_checked() -> None:
    store, checked = _store_with_one_artifact()
    checker = BasicConstraintChecker(store)
    checker.set_enabled(ConstraintKind.SCHEMA_CONTRACT, False)
    constraint = Constraint(constraint_id="c1", kind=ConstraintKind.SCHEMA_CONTRACT, description="x", applies_to=[])
    evidence = checker.check(checked, [constraint])
    assert evidence[0].outcome == CheckOutcome.UNKNOWN
    assert evidence[0].unknown_reason == UnknownReason.NOT_CHECKED


def test_check_answers_every_requested_constraint_not_a_subset() -> None:
    """Validator.check()'s contract (interfaces.py): never silently drop a
    requested constraint. A mixed batch -- one of each kind -- must come
    back with exactly as many Evidence entries, in order."""
    store, checked = _store_with_one_artifact()
    checker = BasicConstraintChecker(store)
    constraints = [
        Constraint(constraint_id="schema", kind=ConstraintKind.SCHEMA_CONTRACT, description="x", applies_to=[]),
        Constraint(constraint_id="recompute", kind=ConstraintKind.RECOMPUTE, description="x"),
        Constraint(constraint_id="csp", kind=ConstraintKind.CSP_SMT, description="x"),
        Constraint(constraint_id="symbolic", kind=ConstraintKind.SYMBOLIC, description="x"),
    ]
    evidence = checker.check(checked, constraints, context={"recompute": {"recompute": {"expected": 1, "observed": 1}}})
    assert [e.constraint_id for e in evidence] == ["schema", "recompute", "csp", "symbolic"]

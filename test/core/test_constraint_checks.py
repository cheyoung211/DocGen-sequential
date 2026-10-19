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


def test_schema_contract_fails_on_kind_mismatch_even_when_id_and_version_resolve() -> None:
    """Regression for a user-caught bug: a constraint requiring kind=TABLE
    at (artifact_id, version) that actually resolves to a stored
    DEFINITION must FAIL, not silently PASS just because something real
    exists at that exact (id, version)."""
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="def")
    store.add_version("thm_1", ArtifactKind.CLAIM, content="thm")
    checker = BasicConstraintChecker(store)
    wrong_kind_ref = ArtifactRef(artifact_id="def_1", version=1, kind=ArtifactKind.TABLE)
    constraint = Constraint(
        constraint_id="c1", kind=ConstraintKind.SCHEMA_CONTRACT, description="x", applies_to=[wrong_kind_ref]
    )
    evidence = checker.check(store.latest("thm_1").ref, [constraint])
    assert evidence[0].outcome == CheckOutcome.FAIL
    assert "table" in evidence[0].detail and "definition" in evidence[0].detail


def test_schema_contract_evidence_carries_applies_to_as_input_versions() -> None:
    """So ArtifactGraphStore.live_evidence can invalidate this verdict if
    one of the referenced artifacts later changes version -- even on a
    PASS, since a PASS depends on those refs staying resolved too."""
    store, checked = _store_with_one_artifact("thm_1")
    store.add_version("def_1", ArtifactKind.DEFINITION, content="def")
    checker = BasicConstraintChecker(store)
    ref = store.latest("def_1").ref
    constraint = Constraint(constraint_id="c1", kind=ConstraintKind.SCHEMA_CONTRACT, description="x", applies_to=[ref])
    evidence = checker.check(checked, [constraint])
    assert evidence[0].outcome == CheckOutcome.PASS
    assert evidence[0].input_versions == [ref]
    assert evidence[0].missing_inputs == []


def test_schema_contract_fail_records_missing_refs_by_id_not_as_a_fabricated_existing_version() -> None:
    """Regression for a second user-caught repro: a not-found ref must go
    on missing_inputs by artifact_id, never on input_versions as if that
    version genuinely existed -- otherwise ArtifactGraphStore.live_evidence
    treats a "this is missing" FAIL as already invalid (nothing named in
    input_versions actually exists yet to be live), and resurrects it the
    moment the artifact is finally produced -- backwards either way."""
    store, checked = _store_with_one_artifact("thm_1")
    checker = BasicConstraintChecker(store)
    missing_1 = ArtifactRef(artifact_id="ghost_1", version=1, kind=ArtifactKind.DEFINITION)
    missing_2 = ArtifactRef(artifact_id="ghost_2", version=1, kind=ArtifactKind.TABLE)
    constraint = Constraint(
        constraint_id="c1", kind=ConstraintKind.SCHEMA_CONTRACT, description="x", applies_to=[missing_1, missing_2]
    )
    evidence = checker.check(checked, [constraint])
    assert evidence[0].outcome == CheckOutcome.FAIL
    assert sorted(evidence[0].missing_inputs) == ["ghost_1", "ghost_2"]
    assert evidence[0].input_versions == []


def test_schema_contract_wrong_kind_ref_is_tracked_as_input_versions_not_missing() -> None:
    """A ref that resolved to the wrong kind still exists -- it belongs on
    input_versions (so a later version change on it invalidates this
    verdict), not on missing_inputs (which means "does not exist at
    all")."""
    store = ArtifactGraphStore()
    store.add_version("def_1", ArtifactKind.DEFINITION, content="def")
    store.add_version("thm_1", ArtifactKind.CLAIM, content="thm")
    checker = BasicConstraintChecker(store)
    wrong_kind_ref = ArtifactRef(artifact_id="def_1", version=1, kind=ArtifactKind.TABLE)
    constraint = Constraint(
        constraint_id="c1", kind=ConstraintKind.SCHEMA_CONTRACT, description="x", applies_to=[wrong_kind_ref]
    )
    evidence = checker.check(store.latest("thm_1").ref, [constraint])
    assert evidence[0].outcome == CheckOutcome.FAIL
    assert evidence[0].input_versions == [wrong_kind_ref]
    assert evidence[0].missing_inputs == []


def test_direct_comparison_mode_tracks_applies_to_even_without_source_refs_in_context() -> None:
    """Regression for a second user-caught repro: declaring the source
    solely via Constraint.applies_to (the same mechanism SCHEMA_CONTRACT
    uses), with no "source_refs" key in context["recompute"][...] at all,
    previously left input_versions empty -- so a PASS kept reading as live
    evidence forever even after that source changed version."""
    store, checked = _store_with_one_artifact("checked")
    store.add_version("src", ArtifactKind.SOURCE_DATA, content="[1, 2]")
    src_ref = store.latest("src").ref
    checker = BasicConstraintChecker(store)
    constraint = Constraint(
        constraint_id="r1", kind=ConstraintKind.RECOMPUTE, description="x", applies_to=[src_ref]
    )
    evidence = checker.check(checked, [constraint], context={"recompute": {"r1": {"expected": 3.0, "observed": 3.0}}})
    assert evidence[0].outcome == CheckOutcome.PASS
    assert evidence[0].input_versions == [src_ref]


def test_direct_comparison_mode_merges_applies_to_and_source_refs_without_duplicating() -> None:
    store, checked = _store_with_one_artifact("checked")
    store.add_version("src_a", ArtifactKind.SOURCE_DATA, content="[1]")
    store.add_version("src_b", ArtifactKind.SOURCE_DATA, content="[2]")
    ref_a = store.latest("src_a").ref
    ref_b = store.latest("src_b").ref
    checker = BasicConstraintChecker(store)
    constraint = Constraint(
        constraint_id="r1", kind=ConstraintKind.RECOMPUTE, description="x", applies_to=[ref_a]
    )
    evidence = checker.check(
        checked,
        [constraint],
        context={"recompute": {"r1": {"expected": 3.0, "observed": 3.0, "source_refs": [ref_a, ref_b]}}},
    )
    assert evidence[0].outcome == CheckOutcome.PASS
    assert evidence[0].input_versions == [ref_a, ref_b]  # ref_a from applies_to, not duplicated


class TestRecomputeFromSource:
    """The second RECOMPUTE mode: actually recomputing `expected` from a
    stored, structured source artifact -- distinct from the pre-existing
    direct-comparison mode, which only compares two caller-supplied
    numbers and was previously (inaccurately) the only thing "RECOMPUTE"
    meant."""

    def _store_with_source(self, content: str = "[10, 20, 5.5]"):
        store = ArtifactGraphStore()
        store.add_version("checked", ArtifactKind.BLOCK, content="x")
        store.add_version("data_1", ArtifactKind.SOURCE_DATA, content=content)
        return store, store.latest("checked").ref, store.latest("data_1").ref

    def test_sum_matches_within_tolerance(self) -> None:
        store, checked, source_ref = self._store_with_source()
        checker = BasicConstraintChecker(store)
        constraint = Constraint(constraint_id="r1", kind=ConstraintKind.RECOMPUTE, description="x")
        evidence = checker.check(
            checked,
            [constraint],
            context={"recompute": {"r1": {"source_ref": source_ref, "operation": "sum", "observed": 35.5, "tolerance": 0.01}}},
        )
        assert evidence[0].outcome == CheckOutcome.PASS
        assert evidence[0].input_versions == [source_ref]
        assert "recomputed via 'sum'" in evidence[0].detail

    def test_sum_mismatch_fails(self) -> None:
        store, checked, source_ref = self._store_with_source()
        checker = BasicConstraintChecker(store)
        constraint = Constraint(constraint_id="r1", kind=ConstraintKind.RECOMPUTE, description="x")
        evidence = checker.check(
            checked,
            [constraint],
            context={"recompute": {"r1": {"source_ref": source_ref, "operation": "sum", "observed": 999.0}}},
        )
        assert evidence[0].outcome == CheckOutcome.FAIL

    def test_unsupported_operation_is_unknown_not_representable(self) -> None:
        store, checked, source_ref = self._store_with_source()
        checker = BasicConstraintChecker(store)
        constraint = Constraint(constraint_id="r1", kind=ConstraintKind.RECOMPUTE, description="x")
        evidence = checker.check(
            checked,
            [constraint],
            context={"recompute": {"r1": {"source_ref": source_ref, "operation": "average", "observed": 1.0}}},
        )
        assert evidence[0].outcome == CheckOutcome.UNKNOWN
        assert evidence[0].unknown_reason == UnknownReason.NOT_REPRESENTABLE

    def test_missing_source_artifact_is_unknown_missing_dependency(self) -> None:
        store, checked, _ = self._store_with_source()
        checker = BasicConstraintChecker(store)
        constraint = Constraint(constraint_id="r1", kind=ConstraintKind.RECOMPUTE, description="x")
        ghost_ref = ArtifactRef(artifact_id="ghost", version=1, kind=ArtifactKind.SOURCE_DATA)
        evidence = checker.check(
            checked,
            [constraint],
            context={"recompute": {"r1": {"source_ref": ghost_ref, "operation": "sum", "observed": 1.0}}},
        )
        assert evidence[0].outcome == CheckOutcome.UNKNOWN
        assert evidence[0].unknown_reason == UnknownReason.MISSING_DEPENDENCY
        # Regression for a second user-caught repro: a wholly-absent source
        # must go on missing_inputs by id, never on input_versions as a
        # fabricated "existing" version -- see test_constraint_checks.py's
        # top-level test of the same fix for SCHEMA_CONTRACT.
        assert evidence[0].missing_inputs == ["ghost"]
        assert evidence[0].input_versions == []

    def test_recompute_from_source_also_tracks_constraints_applies_to_refs(self) -> None:
        """The same applies_to-merge fix as the direct-comparison mode:
        a constraint can additionally declare dependencies via applies_to
        (e.g. an unrelated required artifact), and those must still be
        tracked on input_versions alongside source_ref itself."""
        store, checked, source_ref = self._store_with_source()
        store.add_version("extra", ArtifactKind.DEFINITION, content="also depended on")
        extra_ref = store.latest("extra").ref
        checker = BasicConstraintChecker(store)
        constraint = Constraint(
            constraint_id="r1", kind=ConstraintKind.RECOMPUTE, description="x", applies_to=[extra_ref]
        )
        evidence = checker.check(
            checked,
            [constraint],
            context={"recompute": {"r1": {"source_ref": source_ref, "operation": "sum", "observed": 35.5, "tolerance": 0.01}}},
        )
        assert evidence[0].outcome == CheckOutcome.PASS
        assert evidence[0].input_versions == [extra_ref, source_ref]

    def test_non_numeric_json_content_is_unknown_not_representable(self) -> None:
        store, checked, source_ref = self._store_with_source(content='{"not": "a list of numbers"}')
        checker = BasicConstraintChecker(store)
        constraint = Constraint(constraint_id="r1", kind=ConstraintKind.RECOMPUTE, description="x")
        evidence = checker.check(
            checked,
            [constraint],
            context={"recompute": {"r1": {"source_ref": source_ref, "operation": "sum", "observed": 1.0}}},
        )
        assert evidence[0].outcome == CheckOutcome.UNKNOWN
        assert evidence[0].unknown_reason == UnknownReason.NOT_REPRESENTABLE

    def test_direct_comparison_mode_still_works_and_carries_declared_source_refs(self) -> None:
        """The original mode (caller supplies `expected` directly) is kept
        for callers who already computed it elsewhere -- now correctly
        understood as comparison, not recomputation, and still able to
        declare provenance via `source_refs` for live-evidence invalidation."""
        store, checked, source_ref = self._store_with_source()
        checker = BasicConstraintChecker(store)
        constraint = Constraint(constraint_id="r1", kind=ConstraintKind.RECOMPUTE, description="x")
        evidence = checker.check(
            checked,
            [constraint],
            context={"recompute": {"r1": {"expected": 10.0, "observed": 10.0, "source_refs": [source_ref]}}},
        )
        assert evidence[0].outcome == CheckOutcome.PASS
        assert evidence[0].input_versions == [source_ref]


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

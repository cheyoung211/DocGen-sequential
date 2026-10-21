"""Coverage for src/core/fault_injection.py (Stage 6): the four named
injection kinds, each writing a hand-authored corrupted version through
the real, already-reviewed ArtifactGraphStore.add_version -- never a real
Producer/LLM call, per this stage's own scope decision."""

from __future__ import annotations

import pytest

from src.core.artifact_graph import ArtifactGraphStore
from src.core.fault_injection import FaultInjector, UnsupportedInjectionTargetError
from src.core.schemas import ArtifactKind, ArtifactRef, FaultInjectionKind


def _ref(artifact_id: str, version: int, kind: ArtifactKind) -> ArtifactRef:
    return ArtifactRef(artifact_id=artifact_id, version=version, kind=kind)


# ---------------------------------------------------------------------------
# inject_missing_assumption
# ---------------------------------------------------------------------------


def test_inject_missing_assumption_removes_the_named_text_and_bumps_version() -> None:
    store = ArtifactGraphStore()
    store.add_version(
        "thm_1", ArtifactKind.CLAIM, content="Let X be real. Assume X > 0. Then X^2 > 0.", produced_by="seed"
    )
    theorem_ref = store.latest("thm_1").ref

    fault = FaultInjector().inject_missing_assumption(
        store, theorem_ref, assumption_text="Assume X > 0. "
    )

    assert fault.kind == FaultInjectionKind.MISSING_ASSUMPTION
    assert fault.pre_injection_refs == [theorem_ref]
    new_content = store.latest("thm_1").content
    assert new_content == "Let X be real. Then X^2 > 0."
    assert fault.injected_refs[0].version == theorem_ref.version + 1
    assert store.is_stale("thm_1") is False  # injection itself is the new ground truth, not "built from stale input"


def test_inject_missing_assumption_includes_dependents_in_predicted_affected() -> None:
    store = ArtifactGraphStore()
    store.add_version("thm_1", ArtifactKind.CLAIM, content="Assume X > 0. Conclusion.", produced_by="seed")
    theorem_ref = store.latest("thm_1").ref
    store.add_version(
        "example_1", ArtifactKind.BLOCK, content="uses thm_1", read_input_versions=[theorem_ref], produced_by="seed"
    )

    fault = FaultInjector().inject_missing_assumption(store, theorem_ref, assumption_text="Assume X > 0. ")

    assert "example_1" in fault.predicted_affected_artifact_ids
    assert store.is_stale("example_1") is True


def test_inject_missing_assumption_raises_for_unknown_artifact() -> None:
    store = ArtifactGraphStore()
    with pytest.raises(UnsupportedInjectionTargetError):
        FaultInjector().inject_missing_assumption(
            store, _ref("ghost", 1, ArtifactKind.CLAIM), assumption_text="Assume X > 0."
        )


def test_inject_missing_assumption_raises_when_text_not_present() -> None:
    store = ArtifactGraphStore()
    store.add_version("thm_1", ArtifactKind.CLAIM, content="no assumptions here", produced_by="seed")
    theorem_ref = store.latest("thm_1").ref
    with pytest.raises(UnsupportedInjectionTargetError):
        FaultInjector().inject_missing_assumption(store, theorem_ref, assumption_text="Assume X > 0.")


# ---------------------------------------------------------------------------
# inject_altered_denominator
# ---------------------------------------------------------------------------


def test_inject_altered_denominator_replaces_only_the_named_denominator() -> None:
    store = ArtifactGraphStore()
    store.add_version("table_1", ArtifactKind.TABLE, content="ratio = revenue / 120", produced_by="seed")
    table_ref = store.latest("table_1").ref

    fault = FaultInjector().inject_altered_denominator(
        store, table_ref, original_denominator="/ 120", corrupted_denominator="/ 90"
    )

    assert fault.kind == FaultInjectionKind.ALTERED_DENOMINATOR
    assert store.latest("table_1").content == "ratio = revenue / 90"


def test_inject_altered_denominator_rejects_non_table_kind() -> None:
    store = ArtifactGraphStore()
    store.add_version("block_1", ArtifactKind.BLOCK, content="x / 120", produced_by="seed")
    block_ref = store.latest("block_1").ref
    with pytest.raises(UnsupportedInjectionTargetError):
        FaultInjector().inject_altered_denominator(
            store, block_ref, original_denominator="/ 120", corrupted_denominator="/ 90"
        )


# ---------------------------------------------------------------------------
# inject_inconsistent_symbol_meaning
# ---------------------------------------------------------------------------


def test_inject_inconsistent_symbol_meaning_touches_exactly_both_sections() -> None:
    store = ArtifactGraphStore()
    store.add_version("sec_a", ArtifactKind.SECTION, content="intro", produced_by="seed")
    store.add_version("sec_b", ArtifactKind.SECTION, content="methods", produced_by="seed")
    ref_a = store.latest("sec_a").ref
    ref_b = store.latest("sec_b").ref

    fault = FaultInjector().inject_inconsistent_symbol_meaning(
        store, "x", ref_a, ref_b, meaning_a="cost in dollars", meaning_b="item count"
    )

    assert fault.kind == FaultInjectionKind.INCONSISTENT_SYMBOL_MEANING
    assert len(fault.pre_injection_refs) == 2
    assert len(fault.injected_refs) == 2
    assert "cost in dollars" in (store.latest("sec_a").content or "")
    assert "item count" in (store.latest("sec_b").content or "")


def test_inject_inconsistent_symbol_meaning_rejects_the_same_artifact_twice() -> None:
    store = ArtifactGraphStore()
    store.add_version("sec_a", ArtifactKind.SECTION, content="intro", produced_by="seed")
    ref_a = store.latest("sec_a").ref
    with pytest.raises(UnsupportedInjectionTargetError):
        FaultInjector().inject_inconsistent_symbol_meaning(
            store, "x", ref_a, ref_a, meaning_a="cost", meaning_b="count"
        )


# ---------------------------------------------------------------------------
# inject_unsatisfiable_blueprint
# ---------------------------------------------------------------------------


def test_inject_unsatisfiable_blueprint_appends_both_requirements() -> None:
    store = ArtifactGraphStore()
    store.add_version("doc_1", ArtifactKind.DOCUMENT, content="A report.", produced_by="seed")
    doc_ref = store.latest("doc_1").ref

    fault = FaultInjector().inject_unsatisfiable_blueprint(
        store,
        doc_ref,
        contradictory_requirements=["must be under 2 pages", "must include 5 full worked examples"],
    )

    assert fault.kind == FaultInjectionKind.UNSATISFIABLE_BLUEPRINT
    content = store.latest("doc_1").content or ""
    assert "must be under 2 pages" in content
    assert "must include 5 full worked examples" in content


def test_inject_unsatisfiable_blueprint_requires_at_least_two_requirements() -> None:
    store = ArtifactGraphStore()
    store.add_version("doc_1", ArtifactKind.DOCUMENT, content="A report.", produced_by="seed")
    doc_ref = store.latest("doc_1").ref
    with pytest.raises(UnsupportedInjectionTargetError):
        FaultInjector().inject_unsatisfiable_blueprint(store, doc_ref, contradictory_requirements=["only one"])

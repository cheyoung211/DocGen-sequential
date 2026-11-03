"""Regression tests for DocumentGraph.compute_ready_sections/
find_dependency_cycles (Revision "Additional verifiers" plan, Fix C
checklist item C2's "compute_ready_artifacts의 개념을 최초 생성과 복구
경로에 연결한다... content-read 의존성의 누락·stale 입력·순환을 처리") --
mirrors src/core/controller.py::compute_ready_artifacts's readiness
definition for this legacy, single-version DocumentGraph."""

from __future__ import annotations

import unittest

from src.common.state import DocumentGraph, DocumentNode, NodeStatus, NodeType


def _node(node_id: str, status: NodeStatus, dependencies=None) -> DocumentNode:
    return DocumentNode(
        id=node_id, type=NodeType.SECTION, title=node_id, status=status,
        dependencies=dependencies or [],
    )


class ComputeReadySectionsTest(unittest.TestCase):
    def test_a_node_with_no_dependencies_is_ready(self) -> None:
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.PENDING))
        self.assertEqual(graph.compute_ready_sections(["a"]), {"a"})

    def test_a_node_waiting_on_an_undrafted_dependency_is_not_ready(self) -> None:
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.PENDING))
        graph.add_node(_node("b", NodeStatus.PENDING, dependencies=["a"]))
        self.assertEqual(graph.compute_ready_sections(["b"]), set())

    def test_a_node_whose_dependency_is_drafted_is_ready(self) -> None:
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.DRAFTED))
        graph.add_node(_node("b", NodeStatus.PENDING, dependencies=["a"]))
        self.assertEqual(graph.compute_ready_sections(["b"]), {"b"})

    def test_reviewed_and_validated_also_count_as_satisfying_a_dependency(self) -> None:
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.REVIEWED))
        graph.add_node(_node("b", NodeStatus.VALIDATED))
        graph.add_node(_node("c", NodeStatus.PENDING, dependencies=["a", "b"]))
        self.assertEqual(graph.compute_ready_sections(["c"]), {"c"})

    def test_an_already_drafted_node_is_not_reported_ready(self) -> None:
        # Mirrors compute_ready_artifacts's "stale AND..." -- a node with
        # nothing left to do is not "ready to generate", it's just done.
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.DRAFTED))
        self.assertEqual(graph.compute_ready_sections(["a"]), set())

    def test_a_dependency_on_a_nonexistent_node_is_treated_as_unmet(self) -> None:
        graph = DocumentGraph()
        graph.add_node(_node("b", NodeStatus.PENDING, dependencies=["ghost"]))
        self.assertEqual(graph.compute_ready_sections(["b"]), set())

    def test_only_candidate_ids_are_evaluated(self) -> None:
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.PENDING))
        graph.add_node(_node("b", NodeStatus.PENDING))
        self.assertEqual(graph.compute_ready_sections(["a"]), {"a"})


class FindDependencyCyclesTest(unittest.TestCase):
    def test_no_cycle_in_a_simple_chain(self) -> None:
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.PENDING))
        graph.add_node(_node("b", NodeStatus.PENDING, dependencies=["a"]))
        graph.add_node(_node("c", NodeStatus.PENDING, dependencies=["b"]))
        self.assertEqual(graph.find_dependency_cycles(), [])

    def test_direct_two_node_cycle_is_detected(self) -> None:
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.PENDING, dependencies=["b"]))
        graph.add_node(_node("b", NodeStatus.PENDING, dependencies=["a"]))
        cycles = graph.find_dependency_cycles()
        self.assertEqual(len(cycles), 1)
        self.assertEqual(set(cycles[0]), {"a", "b"})

    def test_longer_cycle_is_detected(self) -> None:
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.PENDING, dependencies=["c"]))
        graph.add_node(_node("b", NodeStatus.PENDING, dependencies=["a"]))
        graph.add_node(_node("c", NodeStatus.PENDING, dependencies=["b"]))
        cycles = graph.find_dependency_cycles()
        self.assertEqual(len(cycles), 1)
        self.assertEqual(set(cycles[0]), {"a", "b", "c"})

    def test_cycle_outside_candidate_ids_is_not_reported(self) -> None:
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.PENDING, dependencies=["b"]))
        graph.add_node(_node("b", NodeStatus.PENDING, dependencies=["a"]))
        graph.add_node(_node("c", NodeStatus.PENDING))
        self.assertEqual(graph.find_dependency_cycles(candidate_ids=["c"]), [])

    def test_a_cycle_makes_every_member_permanently_unready(self) -> None:
        # The combined, real-world point of both functions together: a
        # cyclic pair never becomes ready no matter how long you wait,
        # which compute_ready_sections alone cannot distinguish from
        # "just not ready yet" -- find_dependency_cycles is what surfaces
        # that distinction explicitly.
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.PENDING, dependencies=["b"]))
        graph.add_node(_node("b", NodeStatus.PENDING, dependencies=["a"]))
        self.assertEqual(graph.compute_ready_sections(["a", "b"]), set())
        self.assertEqual(len(graph.find_dependency_cycles(["a", "b"])), 1)


class FindStaleDependentsTest(unittest.TestCase):
    def test_node_with_no_recorded_versions_is_not_stale(self) -> None:
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.DRAFTED))
        graph.add_node(_node("b", NodeStatus.DRAFTED))
        self.assertEqual(graph.find_stale_dependents(["b"]), set())

    def test_node_whose_recorded_snapshot_still_matches_is_not_stale(self) -> None:
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.DRAFTED))
        b = _node("b", NodeStatus.DRAFTED)
        b.spec["dependency_versions_used"] = [
            {
                "depends_on": "a",
                "attempt_count": graph.nodes["a"].attempt_count,
                "last_updated": graph.nodes["a"].last_updated.isoformat(),
            }
        ]
        graph.add_node(b)
        self.assertEqual(graph.find_stale_dependents(["b"]), set())

    def test_node_is_stale_after_its_dependency_is_regenerated(self) -> None:
        graph = DocumentGraph()
        graph.add_node(_node("a", NodeStatus.DRAFTED))
        b = _node("b", NodeStatus.DRAFTED)
        b.spec["dependency_versions_used"] = [
            {
                "depends_on": "a",
                "attempt_count": graph.nodes["a"].attempt_count,
                "last_updated": graph.nodes["a"].last_updated.isoformat(),
            }
        ]
        graph.add_node(b)
        # Simulate a repair regenerating "a" after "b" already read it.
        graph.update_node_status("a", NodeStatus.RUNNING)
        graph.update_node_status("a", NodeStatus.DRAFTED, content="revised content")
        self.assertEqual(graph.find_stale_dependents(["b"]), {"b"})

    def test_node_is_stale_if_its_recorded_dependency_no_longer_exists(self) -> None:
        graph = DocumentGraph()
        b = _node("b", NodeStatus.DRAFTED)
        b.spec["dependency_versions_used"] = [
            {"depends_on": "ghost", "attempt_count": 1, "last_updated": "2026-01-01T00:00:00"}
        ]
        graph.add_node(b)
        self.assertEqual(graph.find_stale_dependents(["b"]), {"b"})


if __name__ == "__main__":
    unittest.main()

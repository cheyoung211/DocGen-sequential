"""Unit coverage for run_full_pipeline._redraft_stale_dependents_until_settled
(Fix C, checklist item C2's "stale 입력... 처리" applied to the compile-repair
path): a review found the first version of this logic computed the stale-
dependent list ONCE and redrafted only that fixed list -- missing a chain
source -> A -> B, where redrafting A (itself found stale from source's
change) makes B stale too, and a one-shot list never revisits B. This test
drives the real fixed-point function against a hand-built DocumentGraph, the
same DocumentGraph/run_full_pipeline.py path the review named as uncovered by
the existing ArtifactGraph-side multi-step regeneration tests."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_full_pipeline import _redraft_stale_dependents_until_settled
from src.common.state import DocumentGraph, DocumentNode, NodeStatus, NodeType


def _node(node_id: str, *, attempt_count: int = 0, last_updated: datetime) -> DocumentNode:
    return DocumentNode(
        id=node_id, type=NodeType.SECTION, title=node_id, status=NodeStatus.DRAFTED,
        content=f"content:{node_id}", attempt_count=attempt_count, last_updated=last_updated,
    )


def _snapshot(source: DocumentNode) -> Dict[str, object]:
    return {
        "depends_on": source.id,
        "attempt_count": source.attempt_count,
        "last_updated": source.last_updated.isoformat(),
    }


class _FakeTextAgent:
    """Models what a real redraft actually does to the two fields
    find_stale_dependents reads: bumps the redrafted node's own
    attempt_count/last_updated, and refreshes its dependency_versions_used
    snapshot to the CURRENT state of whatever it depends on -- exactly like
    TextAgent._build_dependency_context really would on a fresh call."""

    def __init__(self, graph: DocumentGraph, depends_on: Dict[str, List[str]]) -> None:
        self.graph = graph
        self.depends_on = depends_on
        self.calls: List[str] = []

    def redraft(self, node_id: str) -> None:
        self.calls.append(node_id)
        node = self.graph.nodes[node_id]
        node.attempt_count += 1
        node.last_updated = datetime.now()
        node.spec["dependency_versions_used"] = [
            _snapshot(self.graph.nodes[dep_id]) for dep_id in self.depends_on.get(node_id, [])
        ]


class RedraftStaleDependentsUntilSettledTest(unittest.TestCase):
    def test_a_chain_propagates_through_every_level_not_just_the_first(self) -> None:
        """source -> A -> B. Only source has already changed when this is
        called (as _repair_and_recompile calls it right after regenerating
        location.section_id) -- A is stale now, B is not stale YET (its
        snapshot of A still matches A's current, untouched state). Once A
        is redrafted, B must be discovered stale on a LATER sweep and
        redrafted too."""
        t0 = datetime.now() - timedelta(hours=1)
        graph = DocumentGraph()
        source = _node("source", attempt_count=5, last_updated=datetime.now())  # just "changed"
        a = _node("a", attempt_count=0, last_updated=t0)
        b = _node("b", attempt_count=0, last_updated=t0)
        graph.add_node(source)
        graph.add_node(a)
        graph.add_node(b)
        a.spec["dependency_versions_used"] = [_snapshot(_node("source", attempt_count=0, last_updated=t0))]
        b.spec["dependency_versions_used"] = [_snapshot(a)]  # matches a's CURRENT (not-yet-touched) state

        fake = _FakeTextAgent(graph, depends_on={"a": ["source"], "b": ["a"]})
        settled = _redraft_stale_dependents_until_settled(graph, fake.redraft)

        self.assertTrue(settled)
        self.assertEqual(fake.calls, ["a", "b"])  # a resolved first, which is what made b stale
        # Final state really is consistent -- not just "a function returned True".
        self.assertEqual(graph.find_stale_dependents({"a", "b"}), set())
        self.assertEqual(a.spec["dependency_versions_used"][0]["attempt_count"], source.attempt_count)
        self.assertEqual(b.spec["dependency_versions_used"][0]["attempt_count"], a.attempt_count)

    def test_nothing_stale_redrafts_nothing(self) -> None:
        t0 = datetime.now()
        graph = DocumentGraph()
        source = _node("source", attempt_count=0, last_updated=t0)
        a = _node("a", attempt_count=0, last_updated=t0)
        graph.add_node(source)
        graph.add_node(a)
        a.spec["dependency_versions_used"] = [_snapshot(source)]

        fake = _FakeTextAgent(graph, depends_on={"a": ["source"]})
        settled = _redraft_stale_dependents_until_settled(graph, fake.redraft)

        self.assertTrue(settled)
        self.assertEqual(fake.calls, [])

    def test_a_raising_redraft_reports_failure(self) -> None:
        t0 = datetime.now() - timedelta(hours=1)
        graph = DocumentGraph()
        source = _node("source", attempt_count=1, last_updated=datetime.now())
        a = _node("a", attempt_count=0, last_updated=t0)
        graph.add_node(source)
        graph.add_node(a)
        a.spec["dependency_versions_used"] = [_snapshot(_node("source", attempt_count=0, last_updated=t0))]

        def _raising_redraft(node_id: str) -> None:
            raise RuntimeError("boom")

        self.assertFalse(_redraft_stale_dependents_until_settled(graph, _raising_redraft))

    def test_a_redraft_that_leaves_the_node_not_drafted_reports_failure(self) -> None:
        t0 = datetime.now() - timedelta(hours=1)
        graph = DocumentGraph()
        source = _node("source", attempt_count=1, last_updated=datetime.now())
        a = _node("a", attempt_count=0, last_updated=t0)
        graph.add_node(source)
        graph.add_node(a)
        a.spec["dependency_versions_used"] = [_snapshot(_node("source", attempt_count=0, last_updated=t0))]

        def _failed_redraft(node_id: str) -> None:
            graph.nodes[node_id].status = NodeStatus.ERROR  # mirrors a real content-agent failure

        self.assertFalse(_redraft_stale_dependents_until_settled(graph, _failed_redraft))

    def test_a_redraft_that_never_refreshes_its_own_snapshot_never_settles(self) -> None:
        """A pathological redraft that "succeeds" (reaches DRAFTED, never
        raises) but forgets to refresh its own dependency_versions_used --
        the same node is found stale every sweep forever. Must report
        failure once the bounded sweep count is exhausted, never loop
        forever and never silently report success."""
        t0 = datetime.now() - timedelta(hours=1)
        graph = DocumentGraph()
        source = _node("source", attempt_count=1, last_updated=datetime.now())
        a = _node("a", attempt_count=0, last_updated=t0)
        graph.add_node(source)
        graph.add_node(a)
        a.spec["dependency_versions_used"] = [_snapshot(_node("source", attempt_count=0, last_updated=t0))]

        calls: List[str] = []

        def _broken_redraft(node_id: str) -> None:
            calls.append(node_id)  # status stays DRAFTED, snapshot is never touched

        self.assertFalse(_redraft_stale_dependents_until_settled(graph, _broken_redraft))
        self.assertEqual(len(calls), len(graph.nodes) + 1)  # ran for exactly the bounded sweep count

    def test_log_callback_is_optional(self) -> None:
        t0 = datetime.now()
        graph = DocumentGraph()
        graph.add_node(_node("source", attempt_count=0, last_updated=t0))
        settled = _redraft_stale_dependents_until_settled(graph, lambda node_id: None)
        self.assertTrue(settled)  # no dependency_versions_used anywhere -- trivially settled


if __name__ == "__main__":
    unittest.main()

"""Unit coverage for run_full_pipeline._dependency_ordered: the execution
loop must draft a node's dependencies before the node itself within one
outer iteration's pending batch, so TextAgent's dependency context (see
test_text_agent_dependency_context.py) actually finds something drafted to
show instead of silently seeing nothing every time."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_full_pipeline import _dependency_ordered
from src.common.state import DocumentNode, NodeType


def _node(node_id: str, dependencies: list[str] | None = None) -> DocumentNode:
    return DocumentNode(
        id=node_id, type=NodeType.SECTION, title=node_id, dependencies=dependencies or []
    )


class DependencyOrderedTest(unittest.TestCase):
    def test_a_dependency_is_ordered_before_its_dependent(self) -> None:
        a, b = _node("a"), _node("b", ["a"])
        # Feed it in dependent-first order to prove it isn't a no-op.
        ordered = _dependency_ordered([b, a])
        self.assertEqual([n.id for n in ordered], ["a", "b"])

    def test_a_dependency_outside_the_batch_never_blocks_readiness(self) -> None:
        # "x" isn't part of this batch at all (e.g. already drafted in a
        # previous outer iteration, or not a real node) -- must not stall c.
        c = _node("c", ["x"])
        ordered = _dependency_ordered([c])
        self.assertEqual([n.id for n in ordered], ["c"])

    def test_a_cycle_falls_back_to_original_order_instead_of_hanging(self) -> None:
        d, e = _node("d", ["e"]), _node("e", ["d"])
        ordered = _dependency_ordered([d, e])
        self.assertEqual([n.id for n in ordered], ["d", "e"])

    def test_independent_nodes_keep_their_relative_order(self) -> None:
        x, y, z = _node("x"), _node("y"), _node("z")
        ordered = _dependency_ordered([x, y, z])
        self.assertEqual([n.id for n in ordered], ["x", "y", "z"])


if __name__ == "__main__":
    unittest.main()

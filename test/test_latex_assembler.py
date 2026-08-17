"""Regression tests for semantic-block-to-LaTeX composition."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.agents.latex_assembler import LatexAssembler
from src.common.schemas import ContentBlock, SemanticBlockType, TextAgentOutput
from src.common.state import (
    DocumentBlueprint,
    DocumentGraph,
    DocumentKind,
    DocumentNode,
    DocumentSpec,
    FigureLayout,
    FigurePlacement,
    LayoutBlock,
    LayoutBlockKind,
    LatexComponent,
    NodeType,
    SectionLayout,
)


class SemanticIntegratorTest(unittest.TestCase):
    def _graph(self) -> DocumentGraph:
        blocks = [
            LayoutBlock(id="intro", kind=LayoutBlockKind.PARAGRAPH),
            LayoutBlock(id="takeaway", kind=LayoutBlockKind.KEY_TAKEAWAY, title="Core result"),
            LayoutBlock(id="definition", kind=LayoutBlockKind.DEFINITION),
            LayoutBlock(id="warning", kind=LayoutBlockKind.WARNING),
            LayoutBlock(id="case", kind=LayoutBlockKind.CASE_STUDY, component=LatexComponent.MINIPAGE),
            LayoutBlock(id="comparison", kind=LayoutBlockKind.COMPARISON, component=LatexComponent.MINIPAGE),
            LayoutBlock(id="list", kind=LayoutBlockKind.LIST),
            LayoutBlock(id="table", kind=LayoutBlockKind.TABLE, component=LatexComponent.TABULARX),
            LayoutBlock(id="equation", kind=LayoutBlockKind.EQUATION),
            LayoutBlock(id="figure-ref", kind=LayoutBlockKind.FIGURE, asset_id="system-figure"),
        ]
        blueprint = DocumentBlueprint(
            document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Semantic rendering test"),
            document_class="article",
            sections=[SectionLayout(section_id="overview", title="Overview", blocks=blocks)],
            figure_slots={
                "system-figure": FigurePlacement(
                    figure_id="system-figure",
                    owner_section="overview",
                    anchor_after_block="figure-ref",
                    layout=FigureLayout.FLOAT,
                    width=r"0.60\textwidth",
                    label="fig:system",
                    caption="System boundary",
                )
            },
            layout_policy={"insert_float_barrier_after_section": True},
        )
        graph = DocumentGraph()
        graph.add_node(DocumentNode(id="overview", type=NodeType.SECTION, title="Overview"))
        graph.add_node(DocumentNode(id="system-figure", type=NodeType.FIGURE, title="System boundary"))
        graph.set_blueprint(blueprint)
        return graph

    @staticmethod
    def _semantic_output() -> TextAgentOutput:
        return TextAgentOutput(
            request_id="semantic-test",
            section_id="overview",
            blocks=[
                ContentBlock(block_id="intro", type=SemanticBlockType.PARAGRAPH, content="A 10% increase & x_y."),
                ContentBlock(block_id="takeaway", type=SemanticBlockType.KEY_TAKEAWAY, content="Keep evidence traceable."),
                ContentBlock(block_id="definition", type=SemanticBlockType.DEFINITION, content="A blueprint is a rendering contract."),
                ContentBlock(block_id="warning", type=SemanticBlockType.WARNING, content="Do not duplicate assets."),
                ContentBlock(block_id="case", type=SemanticBlockType.CASE_STUDY, content="A focused example validates the contract."),
                ContentBlock(
                    block_id="comparison",
                    type=SemanticBlockType.COMPARISON,
                    content="| Option | Result |\n| --- | --- |\n| Semantic blocks | Reliable |",
                ),
                ContentBlock(block_id="list", type=SemanticBlockType.LIST, content="- Validate IDs\n- Render deterministically"),
                ContentBlock(
                    block_id="table",
                    type=SemanticBlockType.TABLE,
                    title="Reliability checks",
                    content="| Check | State |\n| --- | --- |\n| Block IDs | Pass |",
                ),
                ContentBlock(block_id="equation", type=SemanticBlockType.EQUATION, content=r"E = mc^2"),
                ContentBlock(
                    block_id="figure-ref",
                    type=SemanticBlockType.FIGURE_REFERENCE,
                    asset_id="system-figure",
                    content=r"Figure~\ref{fig:system} shows the system boundary.",
                ),
            ],
        )

    def test_composes_from_blocks_json_not_draft_tex(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            os.chdir(temporary_directory)
            try:
                graph = self._graph()
                base = Path("outputs/generations/semantic-test")
                sections = base / "sections"
                assets = base / "assets"
                sections.mkdir(parents=True)
                assets.mkdir(parents=True)
                (assets / "system-figure.png").write_bytes(b"placeholder")
                graph.register_figure_asset("system-figure", "assets/system-figure.png")
                (sections / "overview.blocks.json").write_text(
                    self._semantic_output().model_dump_json(indent=2), encoding="utf-8"
                )
                # This must never be used as source input by the composer.
                (sections / "overview.draft.tex").write_text("THIS MUST NOT APPEAR", encoding="utf-8")

                LatexAssembler().assemble("semantic-test", graph)

                rendered = (sections / "overview.tex").read_text(encoding="utf-8")
                preamble = (base / "preamble.tex").read_text(encoding="utf-8")
                self.assertNotIn("THIS MUST NOT APPEAR", rendered)
                self.assertIn(r"A 10\% increase \& x\_y.", rendered)
                self.assertGreaterEqual(rendered.count(r"\begin{tcolorbox}"), 4)
                self.assertIn(r"\begin{minipage}", rendered)
                self.assertIn(r"\begin{tabularx}", rendered)
                # Bare-formula EQUATION blocks render as a numbered, labeled
                # `equation` environment (not the older unnumbered `\[...\]`)
                # so "Eq. (N)"-style cross-references have something real to
                # point at.
                self.assertIn(r"\begin{equation}", rendered)
                self.assertIn(r"\label{eq:equation}", rendered)
                self.assertEqual(rendered.count(r"\includegraphics"), 1)
                self.assertIn(r"\usepackage{tcolorbox}", preamble)
                self.assertIn(r"\usepackage{tabularx}", preamble)
                self.assertIn(r"\usepackage{amsmath}", preamble)
                self.assertEqual(graph.validate_figure_usage(), [])
            finally:
                os.chdir(original_cwd)

    def test_missing_blocks_json_is_a_hard_error(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            os.chdir(temporary_directory)
            try:
                graph = self._graph()
                base = Path("outputs/generations/semantic-test")
                (base / "assets").mkdir(parents=True)
                (base / "assets/system-figure.png").write_bytes(b"placeholder")
                graph.register_figure_asset("system-figure", "assets/system-figure.png")
                with self.assertRaisesRegex(FileNotFoundError, "blocks.json"):
                    LatexAssembler().assemble("semantic-test", graph)
            finally:
                os.chdir(original_cwd)


class MatrixEnvironmentInAnyBlockTypeTest(unittest.TestCase):
    """Regression coverage for the worked_examples incident: a pmatrix in a
    CASE_STUDY/PROOF_SKETCH block used to hit forbidden_environment on every
    retry (ALLOWED_ENVIRONMENTS had no entry at all for those types, so
    .get(block_type, set()) allowed nothing), exhausting TextAgent's 3
    attempts and failing the whole pipeline run -- see MATRIX_ENVIRONMENTS /
    validate_fragment_environments in latex_assembler.py."""

    def test_pmatrix_allowed_in_case_study_and_proof_sketch(self) -> None:
        from src.agents.latex_assembler import validate_fragment_environments

        content = (
            r"The eigenvectors are $\begin{pmatrix} 1 \\ 0 \end{pmatrix}$ and "
            r"$\begin{pmatrix} 0 \\ 1 \end{pmatrix}$."
        )
        for block_type in (SemanticBlockType.CASE_STUDY, SemanticBlockType.PROOF_SKETCH):
            validate_fragment_environments(block_type, content, "b1")  # must not raise

    def test_every_matrix_variant_allowed_in_a_prose_type(self) -> None:
        from src.agents.latex_assembler import validate_fragment_environments

        for env in ("matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix", "smallmatrix"):
            content = f"$\\begin{{{env}}} 1 & 0 \\\\ 0 & 1 \\end{{{env}}}$"
            validate_fragment_environments(SemanticBlockType.PARAGRAPH, content, "b1")

    def test_numbered_display_environment_still_rejected_in_prose_types(self) -> None:
        from src.agents.latex_assembler import validate_fragment_environments

        content = r"\begin{align} x &= 1 \end{align}"
        for block_type in (SemanticBlockType.CASE_STUDY, SemanticBlockType.PARAGRAPH):
            with self.assertRaises(ValueError):
                validate_fragment_environments(block_type, content, "b1")

    def test_document_framing_environments_still_rejected_alongside_matrix(self) -> None:
        from src.agents.latex_assembler import validate_fragment_environments

        content = r"\begin{figure}\begin{pmatrix} 1 \end{pmatrix}\end{figure}"
        with self.assertRaises(ValueError):
            validate_fragment_environments(SemanticBlockType.CASE_STUDY, content, "b1")


class MarkdownTablePipeEscapingTest(unittest.TestCase):
    """Regression coverage for a raw '|' inside a table cell (e.g. conditional
    probability written as 'P(A|B)') desynchronizing the column count -- see
    UNESCAPED_PIPE / _split_markdown_row in latex_assembler.py."""

    def setUp(self) -> None:
        self.agent = LatexAssembler()

    def test_unescaped_pipe_in_cell_raises_actionable_error(self) -> None:
        table = (
            "| Concept | Formula |\n"
            "|---|---|\n"
            r"| Conditional probability | $P(A|B) = P(A \cap B)/P(B)$ |"
        )
        with self.assertRaises(ValueError) as ctx:
            self.agent._markdown_table_to_tabularx(table)
        message = str(ctx.exception)
        self.assertIn("row 1 has 3 column(s)", message)
        self.assertIn("P(A|B)", message)
        self.assertIn(r"\mid", message)

    def test_escaped_pipe_is_restored_as_a_literal_pipe_in_the_cell(self) -> None:
        table = (
            "| Concept | Formula |\n"
            "|---|---|\n"
            r"| Conditional probability | $P(A\|B) = P(A \cap B)/P(B)$ |"
        )
        rendered = self.agent._markdown_table_to_tabularx(table)
        self.assertIn(r"$P(A|B) = P(A \cap B)/P(B)$", rendered)
        self.assertIn(r"\begin{tabularx}", rendered)

    def test_mid_notation_avoids_the_pipe_issue_entirely(self) -> None:
        table = (
            "| Concept | Formula |\n"
            "|---|---|\n"
            r"| Conditional probability | $P(A \mid B) = P(A \cap B)/P(B)$ |"
        )
        rendered = self.agent._markdown_table_to_tabularx(table)
        self.assertIn(r"$P(A \mid B) = P(A \cap B)/P(B)$", rendered)

    def test_well_formed_table_is_unaffected(self) -> None:
        table = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |"
        rendered = self.agent._markdown_table_to_tabularx(table)
        self.assertIn(r"\textbf{A} & \textbf{B} & \textbf{C}", rendered)
        self.assertIn("1 & 2 & 3", rendered)


class TheoremProofSketchNotationTableTest(unittest.TestCase):
    """Regression coverage for the theorem/proof_sketch/notation_table kinds
    added after the LaTeX-focused benchmark's math-domain items were found to
    require a 'theorem' LayoutBlockKind that did not exist yet (planner
    validation failure: "Input should be 'paragraph', ... or 'equation'
    [input_value='theorem']")."""

    def _graph(self) -> DocumentGraph:
        blocks = [
            LayoutBlock(id="notation", kind=LayoutBlockKind.NOTATION_TABLE),
            LayoutBlock(id="thm", kind=LayoutBlockKind.THEOREM, title="Uniqueness of Limits"),
            LayoutBlock(id="proof", kind=LayoutBlockKind.PROOF_SKETCH),
        ]
        blueprint = DocumentBlueprint(
            document=DocumentSpec(kind=DocumentKind.ARTICLE, title="Theorem rendering test"),
            document_class="article",
            sections=[SectionLayout(section_id="overview", title="Overview", blocks=blocks)],
        )
        graph = DocumentGraph()
        graph.add_node(DocumentNode(id="overview", type=NodeType.SECTION, title="Overview"))
        graph.set_blueprint(blueprint)
        return graph

    @staticmethod
    def _semantic_output() -> TextAgentOutput:
        return TextAgentOutput(
            request_id="theorem-test",
            section_id="overview",
            blocks=[
                ContentBlock(
                    block_id="notation",
                    type=SemanticBlockType.NOTATION_TABLE,
                    content="| Symbol | Meaning |\n| --- | --- |\n| $a_n$ | The $n$-th term |",
                ),
                ContentBlock(
                    block_id="thm",
                    type=SemanticBlockType.THEOREM,
                    content="If a sequence converges, its limit is unique.",
                ),
                ContentBlock(
                    block_id="proof",
                    type=SemanticBlockType.PROOF_SKETCH,
                    content="Suppose $a_n \\to L_1$ and $a_n \\to L_2$ with $L_1 \\neq L_2$; derive a contradiction.",
                ),
            ],
        )

    def test_theorem_proof_and_notation_table_render_and_load_amsthm(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            os.chdir(temporary_directory)
            try:
                graph = self._graph()
                base = Path("outputs/generations/theorem-test")
                sections = base / "sections"
                sections.mkdir(parents=True)
                (sections / "overview.blocks.json").write_text(
                    self._semantic_output().model_dump_json(indent=2), encoding="utf-8"
                )

                LatexAssembler().assemble("theorem-test", graph)

                rendered = (sections / "overview.tex").read_text(encoding="utf-8")
                preamble = (base / "preamble.tex").read_text(encoding="utf-8")

                # notation_table renders exactly like a table, with a default caption.
                self.assertIn(r"\begin{tabularx}", rendered)
                self.assertIn(r"\caption{Notation}", rendered)

                # theorem: composer-owned environment with the block's title as
                # the named subtitle; the model's prose is not re-wrapped itself.
                self.assertIn(r"\begin{theorem}[Uniqueness of Limits]", rendered)
                self.assertIn("If a sequence converges, its limit is unique.", rendered)
                self.assertIn(r"\end{theorem}", rendered)

                # proof_sketch: amsthm's built-in proof environment, renamed.
                self.assertIn(r"\begin{proof}[Proof Sketch]", rendered)
                self.assertIn(r"\end{proof}", rendered)

                # amsthm must be loaded and the theorem counter declared.
                self.assertIn(r"\usepackage{amsthm}", preamble)
                self.assertIn(r"\newtheorem{theorem}{Theorem}[section]", preamble)
                self.assertIn(r"\usepackage{amsmath}", preamble)
                self.assertIn(r"\usepackage{amssymb}", preamble)
            finally:
                os.chdir(original_cwd)

    def test_theorem_block_content_cannot_smuggle_its_own_environment(self) -> None:
        """theorem/proof_sketch are prose-only, like definition/warning --
        ALLOWED_ENVIRONMENTS has no entry for them, so a model-supplied
        \\begin{...} must still be rejected."""
        agent = LatexAssembler()
        block = ContentBlock(
            block_id="thm",
            type=SemanticBlockType.THEOREM,
            content="\\begin{theorem}\nSmuggled environment.\n\\end{theorem}",
        )
        with self.assertRaises(ValueError):
            agent._validate_fragment(block)


if __name__ == "__main__":
    unittest.main()

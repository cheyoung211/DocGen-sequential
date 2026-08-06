"""Semantic content generation aligned with a DocumentBlueprint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.llm_client import LLMClient
from src.common.schemas import (
    ContentBlock,
    SemanticBlockType,
    TextAgentOutput,
    ToolResponse,
)
from src.common.state import DocumentGraph, LayoutBlock, LayoutBlockKind, NodeStatus
from src.helpers.tools import pandas_csv_to_latex_table


SYSTEM_PROMPT = r"""
You are a senior technical writer for a blueprint-driven LaTeX document system.
Generate semantic content blocks for exactly the layout blocks supplied in the
user request. Do not create a preamble, document class, section command,
figure environment, includegraphics command, or a new block ID. The LaTeX
Composer owns document structure and figure placement.

Write valid LaTeX fragments where needed. Escape literal LaTeX special
characters in prose. A figure_reference may refer to an existing figure label,
but it must never contain a figure environment or includegraphics command.

Every "content" value is a JSON string. Where a list item, table row, or
paragraph break needs a line break, put an actual newline character in that
JSON string (the standard, single-escaped `\n` that JSON already supports) --
never write a literal backslash followed by the letter n or t as prose text.
A literal `\n`/`\t` left in the LaTeX output is read by the compiler as an
undefined control sequence and halts the build.

Block-content contract (the composer owns all document-level environments):
- paragraph, key_takeaway, definition, warning, and case_study: prose with only
  standard inline LaTeX such as \emph{}, \textbf{}, $...$, and \Cref{}.
- list: Markdown-style bullet/numbered lines, or a complete itemize/enumerate
  environment. Do not place a list inside a box environment yourself.
- comparison: concise prose or a Markdown pipe table. Do not use tcolorbox,
  minipage, figure, or table environments.
- table: a Markdown pipe table, a tabular/tabularx fragment, or a complete
  table/longtable returned by a data tool. Use `caption` and `label` fields for
  a composer-owned table wrapper; never create a figure.
- notation_table: exactly like `table`, for a table whose rows are symbol /
  meaning pairs (or similar term-definition pairs).
- theorem: prose only -- the theorem or lemma statement itself, with standard
  inline LaTeX and math. Do not write `\begin{theorem}...\end{theorem}`,
  `\newtheorem`, or a label for the environment yourself; the composer wraps
  the content in a numbered theorem environment. An optional `title` becomes
  the theorem's named subtitle (for example "Uniqueness of Limits").
- proof_sketch: prose only -- the proof steps. Do not write
  `\begin{proof}...\end{proof}` yourself; the composer wraps the content in
  amsthm's `proof` environment, labeled "Proof Sketch".

For any Markdown pipe table (in a `table`, `notation_table`, or `comparison`
block): every row, including the header separator row, must have exactly the
same number of `|`-delimited columns as the header. A cell must never contain
a raw `|` character -- it is silently read as an extra column boundary and
breaks the table. Conditional-probability and set-builder notation must use
`\mid` inside math mode (for example `$P(A \mid B)$`, not `$P(A|B)$`); if a
literal pipe is genuinely unavoidable in a cell, escape it as `\|`.
- equation: only a math body (for example `E = mc^2`) or a complete supported
  amsmath equation environment; do not create a section or document wrapper.
- figure_reference: a sentence such as `Figure~\ref{fig:...} shows ...`.
  It must name the supplied asset_id and must never place an image.

Return only valid JSON in this exact shape:
{
  "content_blocks": [
    {
      "block_id": "blueprint block id",
      "type": "paragraph | key_takeaway | definition | warning | case_study | comparison | table | notation_table | theorem | proof_sketch | equation | list | figure_reference",
      "title": "optional title",
      "content": "LaTeX fragment or prose",
      "asset_id": "required only for figure_reference",
      "caption": "optional, for a table only",
      "label": "optional valid LaTeX label, for a table only"
    }
  ],
  "tool_requests": [
    {
      "tool_name": "pandas_csv_to_latex_table",
      "arguments": {"file_path": "...", "caption": "...", "label": "..."},
      "reasoning": "why the data tool is needed"
    }
  ]
}
"""


SEMANTIC_TYPE_BY_LAYOUT_KIND = {
    LayoutBlockKind.PARAGRAPH: SemanticBlockType.PARAGRAPH,
    LayoutBlockKind.LIST: SemanticBlockType.LIST,
    LayoutBlockKind.KEY_TAKEAWAY: SemanticBlockType.KEY_TAKEAWAY,
    LayoutBlockKind.DEFINITION: SemanticBlockType.DEFINITION,
    LayoutBlockKind.WARNING: SemanticBlockType.WARNING,
    LayoutBlockKind.CASE_STUDY: SemanticBlockType.CASE_STUDY,
    LayoutBlockKind.COMPARISON: SemanticBlockType.COMPARISON,
    LayoutBlockKind.TABLE: SemanticBlockType.TABLE,
    LayoutBlockKind.FIGURE: SemanticBlockType.FIGURE_REFERENCE,
    LayoutBlockKind.EQUATION: SemanticBlockType.EQUATION,
    LayoutBlockKind.THEOREM: SemanticBlockType.THEOREM,
    LayoutBlockKind.PROOF_SKETCH: SemanticBlockType.PROOF_SKETCH,
    LayoutBlockKind.NOTATION_TABLE: SemanticBlockType.NOTATION_TABLE,
}


class TextAgent:
    """Generate and persist strict semantic blocks for one section node."""

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm = llm_client or LLMClient()

    def process_node(
        self,
        node_id: str,
        graph: DocumentGraph,
        request_id: str,
        output_dir: str = "outputs/generations",
    ) -> Optional[TextAgentOutput]:
        node = graph.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node '{node_id}' not found in graph.")

        layout_blocks = self._layout_blocks_for_node(node_id, graph)
        sections_dir = Path(output_dir) / request_id / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)
        graph.update_node_status(node_id, NodeStatus.RUNNING)

        try:
            raw_response = self.llm.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=self._build_prompt_from_node(node, graph, layout_blocks),
                temperature=0.3,
                max_new_tokens=4096,
            )
            data = json.loads(self.llm.extract_json_block(raw_response))
            blocks = self._validate_semantic_blocks(data.get("content_blocks"), layout_blocks)
            used_tools = self._execute_tools(data.get("tool_requests", []))

            output = TextAgentOutput(
                request_id=request_id,
                section_id=node_id,
                blocks=blocks,
                used_tools=used_tools,
            )
            self._write_semantic_artifacts(sections_dir, node_id, output)

            draft_content = self._render_draft_with_anchors(blocks)
            graph.update_node_status(node_id, NodeStatus.DRAFTED, content=draft_content)
            print(f"[TextAgent] Saved semantic blocks for section: {node_id}")
            return output
        except Exception as exc:
            graph.update_node_status(node_id, NodeStatus.ERROR, error=str(exc))
            print(f"[TextAgent] Semantic block generation failed for '{node_id}': {exc}")
            return None

    @staticmethod
    def _layout_blocks_for_node(node_id: str, graph: DocumentGraph) -> List[LayoutBlock]:
        if graph.blueprint is None:
            raise ValueError("TextAgent requires a DocumentBlueprint on the graph.")
        layout = next(
            (section for section in graph.blueprint.sections if section.section_id == node_id),
            None,
        )
        if layout is None:
            raise ValueError(f"No section layout exists for node '{node_id}'.")
        if not layout.blocks:
            raise ValueError(f"Section layout '{node_id}' has no semantic blocks.")
        return layout.blocks

    def _build_prompt_from_node(
        self,
        node: Any,
        graph: DocumentGraph,
        layout_blocks: List[LayoutBlock],
    ) -> str:
        parent_id = next((parent for parent, children in graph.hierarchy.items() if node.id in children), None)
        parent_context = graph.nodes[parent_id].title if parent_id in graph.nodes else "Root level"
        figure_context = []
        if graph.blueprint:
            for figure_id, placement in graph.blueprint.figure_slots.items():
                if placement.owner_section == node.id:
                    figure_context.append(
                        {
                            "asset_id": figure_id,
                            "label": placement.label or f"fig:{figure_id}",
                            "caption": placement.caption,
                            "anchor_after_block": placement.anchor_after_block,
                        }
                    )

        planned_blocks = [
            {
                "block_id": block.id,
                "required_type": SEMANTIC_TYPE_BY_LAYOUT_KIND[block.kind].value,
                "title": block.title,
                "instructions": block.instructions,
                "asset_id": block.asset_id,
            }
            for block in layout_blocks
        ]
        return f"""
Section node:
- ID: {node.id}
- Title: {node.title}
- Contextual role: {node.spec.get('contextual_role')}
- Key points: {json.dumps(node.spec.get('key_points', []), ensure_ascii=False)}
- Required tools: {json.dumps(node.spec.get('required_tools', []), ensure_ascii=False)}
- Parent context: {parent_context}
- Prior completed context: {graph.get_full_context(node.id)}

Required semantic blocks, in this exact order:
{json.dumps(planned_blocks, ensure_ascii=False, indent=2)}

Figures owned by this section; refer to these only when relevant:
{json.dumps(figure_context, ensure_ascii=False, indent=2)}
""".strip()

    @staticmethod
    def _validate_semantic_blocks(
        raw_blocks: Any,
        layout_blocks: List[LayoutBlock],
    ) -> List[ContentBlock]:
        if not isinstance(raw_blocks, list):
            raise ValueError("content_blocks must be a JSON array.")

        blocks = [ContentBlock.model_validate(raw_block) for raw_block in raw_blocks]
        expected_ids = [block.id for block in layout_blocks]
        actual_ids = [block.block_id for block in blocks]
        if actual_ids != expected_ids:
            raise ValueError(
                "Semantic block IDs must match the blueprint blocks in order. "
                f"Expected {expected_ids}, received {actual_ids}."
            )

        for content_block, layout_block in zip(blocks, layout_blocks):
            expected_type = SEMANTIC_TYPE_BY_LAYOUT_KIND[layout_block.kind]
            if content_block.type != expected_type:
                raise ValueError(
                    f"Block '{layout_block.id}' must use type '{expected_type.value}', "
                    f"not '{content_block.type.value}'."
                )
            if not content_block.content.strip():
                raise ValueError(f"Block '{layout_block.id}' has empty content.")
            if layout_block.kind == LayoutBlockKind.FIGURE:
                if content_block.asset_id != layout_block.asset_id:
                    raise ValueError(
                        f"Figure reference block '{layout_block.id}' must use asset_id "
                        f"'{layout_block.asset_id}'."
                    )
                forbidden = ("\\begin{figure}", "\\includegraphics")
                if any(token in content_block.content for token in forbidden):
                    raise ValueError(
                        f"Figure reference block '{layout_block.id}' must not render a figure."
                    )
        return blocks

    @staticmethod
    def _render_draft_with_anchors(blocks: List[ContentBlock]) -> str:
        rendered = []
        for block in blocks:
            rendered.append(f"% <layout-anchor:{block.block_id}>")
            rendered.append(block.content.strip())
        return "\n\n".join(rendered) + "\n"

    @staticmethod
    def _write_semantic_artifacts(
        sections_dir: Path,
        node_id: str,
        output: TextAgentOutput,
    ) -> None:
        blocks_path = sections_dir / f"{node_id}.blocks.json"
        blocks_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        draft_path = sections_dir / f"{node_id}.draft.tex"
        draft_path.write_text(
            TextAgent._render_draft_with_anchors(output.blocks),
            encoding="utf-8",
        )

    @staticmethod
    def _execute_tools(requests: Any) -> List[ToolResponse]:
        if not isinstance(requests, list):
            raise ValueError("tool_requests must be a JSON array when provided.")

        responses: List[ToolResponse] = []
        for request in requests:
            if not isinstance(request, dict):
                raise ValueError("Each tool request must be a JSON object.")
            if request.get("tool_name") != "pandas_csv_to_latex_table":
                raise ValueError(f"Unsupported TextAgent tool: {request.get('tool_name')}")
            result = pandas_csv_to_latex_table(**request.get("arguments", {}))
            responses.append(
                ToolResponse(
                    success=result["success"],
                    output=result.get("output"),
                    observation="Table generated successfully" if result["success"] else result.get("error", ""),
                    error=result.get("error"),
                )
            )
        return responses

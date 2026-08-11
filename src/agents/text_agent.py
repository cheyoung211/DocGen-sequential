"""Semantic content generation aligned with a DocumentBlueprint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from scripts.llm_client import LLMClient
from src.agents.latex_integrator import (
    validate_fragment_environments,
    validate_markdown_table_columns,
    validate_no_manual_labels,
    validate_plain_title,
)
from src.common.schemas import (
    ContentBlock,
    SemanticBlockType,
    TextAgentOutput,
    ToolResponse,
)
from src.common.state import DocumentGraph, LayoutBlock, LayoutBlockKind, NodeStatus
from src.evaluation.event_logger import AttemptTracker, EventLogger
from src.helpers.tools import pandas_csv_to_latex_table


def _classify_generation_error(exc: Exception) -> str:
    """Map a process_node() retry-loop exception to a small, stable signal_type
    vocabulary for the evaluation layer's Verification Trigger Rate."""
    if isinstance(exc, json.JSONDecodeError):
        return "json_parse_error"
    if isinstance(exc, ValidationError):
        return "schema_validation_error"
    if isinstance(exc, ValueError):
        # Covers _validate_semantic_blocks' own raises (id/type mismatch,
        # empty content, ...) and the LaTeX fragment validators it calls
        # (validate_fragment_environments/validate_no_manual_labels/
        # validate_markdown_table_columns) -- the dominant failure mode here.
        return "content_validation_error"
    return "generation_error"

SYSTEM_PROMPT = r"""
You are a senior technical writer in a blueprint-driven LaTeX document system.
Generate content for exactly the requested layout blocks.

<Scope>
- Return content only for supplied block IDs. Do not add, remove, rename, or
  reorder block IDs.
- The Composer owns the preamble, document class, sections, labels, numbering,
  box environments, table/figure wrappers, and image placement.
- Never produce a preamble, \section command, \label command, figure
  environment, or \includegraphics command.
- Use valid LaTeX fragments where needed and escape literal LaTeX special
  characters in prose.
- Return only valid JSON matching the schema below.

<Cross-references and titles>
- Use \Cref{} or \ref{} only with an exact label listed in
  `available_cross_reference_labels`. Never invent labels.
- Titles are plain text: never use LaTeX commands, `$...$`, or math notation
  in `title`. Put mathematical notation in `content`.

<Content rules by block type>

| Type | Required content |
|---|---|
| paragraph, key_takeaway, warning, case_study, definition | Prose with inline LaTeX only. |
| list | Markdown bullets/numbered lines, or a complete `itemize`/`enumerate` fragment. |
| comparison | Concise prose or a Markdown pipe table; no LaTex layout environments. |
| table, notation_table | Markdown pipe table, `tabular`/`tabularx` fragment, or complete table returned by a data tool. |
| theorem | Statement only; no theorem environment or label. |
| proof_sketch | Proof steps only; no proof environment. |
| equation | Math body or a supported amsmath environment only. |
| figure_reference | One sentence referencing an existing supplied asset and figure label; no image or figure environment. |

For Markdown pipe tables:
- Every row, including the separator row, has the header's column count.
- Do not use raw `|` inside cells; use `\mid` in mathematical conditional or
  set notation, or `\|` when a literal pipe is necessary.

<Tools>
Request `pandas_csv_to_latex_table` only when the requested block requires
data from a CSV. Use the exact tool schema and provide a short `purpose`;
do not call or request unlisted tools.

<Output schema>
{
  "content_blocks": [
    {
      "block_id": "supplied blueprint block id",
      "type": "supplied block type",
      "title": "optional plain-text title",
      "content": "content for this block",
      "asset_id": "required only for figure_reference",
      "caption": "optional; table only"
    }
  ],
  "tool_requests": [
    {
      "tool_name": "pandas_csv_to_latex_table",
      "arguments": {"file_path": "...", "caption": "...", "label": "tab:<block_id>"},
      "purpose": "short explanation"
    }
  ]
}
"""
# SYSTEM_PROMPT = r"""
# You are a senior technical writer for a blueprint-driven LaTeX document system.
# Generate semantic content blocks for exactly the layout blocks supplied in the
# user request. Do not create a preamble, document class, section command,
# figure environment, includegraphics command, or a new block ID. The LaTeX
# Composer owns document structure and figure placement.

# Write valid LaTeX fragments where needed. Escape literal LaTeX special
# characters in prose. A figure_reference may refer to an existing figure label,
# but it must never contain a figure environment or includegraphics command.

# Every "content" value is a JSON string. Where a list item, table row, or
# paragraph break needs a line break, put an actual newline character in that
# JSON string (the standard, single-escaped `\n` that JSON already supports) --
# never write a literal backslash followed by the letter n or t as prose text.
# A literal `\n`/`\t` left in the LaTeX output is read by the compiler as an
# undefined control sequence and halts the build.

# Block-content contract (the composer owns all document-level environments):
# - paragraph, key_takeaway, warning, and case_study: prose with only standard
#   inline LaTeX such as \emph{}, \textbf{}, $...$, and \Cref{}. These are not
#   numbered, so they are never a valid \Cref{}/\ref{} target -- don't cite one
#   by number ("as Warning 2 notes") since there is no such number.
# - definition: prose only, like the kinds above, but the composer DOES number
#   it (as "Definition N") and give it a citable label automatically -- see
#   "Available cross-reference labels" below to cite one.
# - list: Markdown-style bullet/numbered lines, or a complete itemize/enumerate
#   environment. Do not place a list inside a box environment yourself.
# - comparison: concise prose or a Markdown pipe table. Do not use tcolorbox,
#   minipage, figure, or table environments.
# - table: a Markdown pipe table, a tabular/tabularx fragment, or a complete
#   table/longtable returned by a data tool. Use `caption` for a composer-owned
#   table wrapper; never create a figure.
# - notation_table: exactly like `table`, for a table whose rows are symbol /
#   meaning pairs (or similar term-definition pairs).
# - theorem: prose only -- the theorem or lemma statement itself, with standard
#   inline LaTeX and math. Do not write `\begin{theorem}...\end{theorem}`,
#   `\newtheorem`, or a label for the environment yourself; the composer wraps
#   the content in a numbered theorem environment. An optional `title` becomes
#   the theorem's named subtitle (for example "Uniqueness of Limits").
# - proof_sketch: prose only -- the proof steps. Do not write
#   `\begin{proof}...\end{proof}` yourself; the composer wraps the content in
#   amsthm's `proof` environment, labeled "Proof Sketch". Like the callouts
#   above, this is not numbered and not a valid \Cref{}/\ref{} target.

# Labels (`\label{}`) for theorem/definition/equation/table/notation_table/
# section are assigned automatically by the composer -- never write `\label{}`
# in a block's own "content" yourself. (The one exception is the
# `pandas_csv_to_latex_table` tool's `label` argument below, which still needs
# one, since that tool renders its own complete table environment the composer
# doesn't touch.) To cite an automatically-labeled block elsewhere (with
# `\Cref{}`/`\ref{}`), copy the exact label string from "Available
# cross-reference labels" in the request -- never invent or guess a label
# name; if what you want to cite isn't in that list, it doesn't exist yet and
# can't be cited.

# For any Markdown pipe table (in a `table`, `notation_table`, or `comparison`
# block): every row, including the header separator row, must have exactly the
# same number of `|`-delimited columns as the header. A cell must never contain
# a raw `|` character -- it is silently read as an extra column boundary and
# breaks the table. Conditional-probability and set-builder notation must use
# `\mid` inside math mode (for example `$P(A \mid B)$`, not `$P(A|B)$`); if a
# literal pipe is genuinely unavoidable in a cell, escape it as `\|`.
# - equation: only a math body (for example `E = mc^2`) or a complete supported
#   amsmath equation environment; do not create a section or document wrapper.
# - figure_reference: a sentence such as `Figure~\ref{fig:...} shows ...`.
#   It must name the supplied asset_id and must never place an image.

# Every "title" is rendered as plain text, not run through LaTeX at all --
# a backslash or `$` there prints literally (for example `\pi(\theta)` shows
# up on the page as the text "\pi(\theta)", not the symbols). Never put a
# LaTeX command, math delimiter, or any other LaTeX syntax in a "title";
# math notation belongs in "content" instead, inside `$...$`.

# Return only valid JSON in this exact shape:
# {
#   "content_blocks": [
#     {
#       "block_id": "blueprint block id",
#       "type": "paragraph | key_takeaway | definition | warning | case_study | comparison | table | notation_table | theorem | proof_sketch | equation | list | figure_reference",
#       "title": "optional title",
#       "content": "LaTeX fragment or prose",
#       "asset_id": "required only for figure_reference",
#       "caption": "optional, for a table only"
#     }
#   ],
#   "tool_requests": [
#     {
#       "tool_name": "pandas_csv_to_latex_table",
#       "arguments": {"file_path": "...", "caption": "...", "label": "tab:<block_id>"},
#       "reasoning": "why the data tool is needed"
#     }
#   ]
# }
# """


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
        max_attempts: int = 3,
        event_logger: Optional[EventLogger] = None,
        usage_sink: Optional[list] = None,
    ) -> Optional[TextAgentOutput]:
        """Generate and persist semantic blocks for one section node.

        A validation failure here (an instructions/kind mismatch such as
        asking for a numbered, labeled equation inside a `definition` block,
        which only allows inline math -- see ``validate_fragment_environments``)
        used to be handed back unchanged to the outer per-iteration retry in
        ``run_full_pipeline.run()``, which just resends the exact same
        instructions again next iteration. Against a genuinely self-
        contradictory blueprint instruction that never succeeds, no matter
        how many iterations -- it silently exhausts every iteration still
        ERROR, and only then does the pipeline notice, in Phase 3, with a
        confusing ``FileNotFoundError`` for a section that was never drafted.
        Retrying here instead, with the validation error fed back into the
        next attempt's prompt, gives the model a chance to actually correct
        course (e.g. fall back to inline math) instead of repeating the same
        mistake for free.
        """
        node = graph.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node '{node_id}' not found in graph.")

        layout_blocks = self._layout_blocks_for_node(node_id, graph)
        sections_dir = Path(output_dir) / request_id / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)
        graph.update_node_status(node_id, NodeStatus.RUNNING)

        tracker = AttemptTracker(
            event_logger,
            stage="text_agent",
            verifier="text_semantic_block_validator",
            recovery_action="retry_with_feedback",
            artifact_id=node_id,
            section_id=node_id,
            producer_model=self.llm.model_name,
        )
        last_error: Optional[str] = None
        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = self.llm.generate(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=self._build_prompt_from_node(
                        node, graph, layout_blocks, previous_error=last_error
                    ),
                    temperature=0.3,
                    max_new_tokens=4096,
                    usage_sink=usage_sink,
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
                tracker.record_success(attempt, resulting_artifact_id=f"{node_id}.blocks.json")
                print(f"[TextAgent] Saved semantic blocks for section: {node_id}")
                return output
            except Exception as exc:
                last_error = str(exc)
                tracker.record_failure(attempt, _classify_generation_error(exc), str(exc))
                print(f"[TextAgent] Attempt {attempt}/{max_attempts} failed for '{node_id}': {exc}")

        tracker.record_exhausted()
        graph.update_node_status(node_id, NodeStatus.ERROR, error=last_error)
        print(f"[TextAgent] Semantic block generation failed for '{node_id}' after {max_attempts} attempts: {last_error}")
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
        previous_error: Optional[str] = None,
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
        # Built once from the blueprint (see DocumentGraph.set_blueprint), so
        # every citable label in the whole document -- including ones in
        # sections not drafted yet -- is already known here. Figures are
        # excluded: they're already listed separately in figure_context,
        # with the same label. This is what makes a cross-reference to
        # another section's content a real label instead of a guess (see
        # the `thm:taylor_convergence` incident: a label a different
        # section's independent model call assumed would exist, without it
        # ever being defined anywhere).
        citable_labels = [
            {"label": label, "refers_to": entry.description, "in_section": entry.section_id}
            for label, entry in graph.label_registry.items()
            if entry.kind != "figure"
        ]
        retry_notice = (
            f"""
Your previous attempt for this section was rejected: {previous_error}
Fix this specific problem in your next response. If a block's required_type
only allows prose (paragraph, key_takeaway, definition, warning, case_study),
do not wrap a value in a numbered/labeled display-equation environment there
even if the instructions ask for one to be "numbered" or "labeled" -- write
it as inline math ($...$) instead; a numbered, cross-referenceable version
belongs in a dedicated `equation` block only.
"""
            if previous_error
            else ""
        )
        return f"""
{retry_notice}Section node:
- ID: {node.id}
- Title: {node.title}
- Contextual role: {node.spec.get('contextual_role')}
- Key points: {json.dumps(node.spec.get('key_points', []), ensure_ascii=False)}
- Required tools: {json.dumps(node.spec.get('required_tools', []), ensure_ascii=False)}
- Parent context: {parent_context}
- Prior completed context: {graph.get_full_context(node.id)}

Required semantic blocks, in this exact order:
{json.dumps(planned_blocks, ensure_ascii=False, indent=2)}

Available cross-reference labels (cite with \\Cref{{...}}/\\ref{{...}} using
the exact "label" string; never invent or guess one not listed here -- if
what you want to cite isn't here, it doesn't exist and can't be cited):
{json.dumps(citable_labels, ensure_ascii=False, indent=2)}

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
            # Same environment/command whitelist the composer enforces in
            # LatexIntegratorAgent._validate_fragment, run here too so a
            # violation raises inside process_node's try/except -- which
            # marks the node ERROR and lets the pipeline's iteration loop
            # retry it with a fresh sample -- instead of surfacing for the
            # first time during Phase 3 integration, where nothing retries
            # and the whole request dies.
            validate_fragment_environments(
                content_block.type, content_block.content, content_block.block_id
            )
            # Same reasoning: both used to be composer-only (Phase 3), where
            # a violation surfaces only after every other section is already
            # drafted and nothing retries -- a manually-written \label{}
            # colliding with another independently-drafted section's guess
            # at the "same" label (the eq:energy_balance incident), or an
            # unescaped '|' inside a table cell (e.g. 'E[Y|X=x]' instead of
            # '\mid'), used to kill the whole document there instead of
            # retrying just this block.
            validate_no_manual_labels(
                content_block.type, content_block.content, content_block.block_id
            )
            if layout_block.kind in (
                LayoutBlockKind.TABLE,
                LayoutBlockKind.NOTATION_TABLE,
                LayoutBlockKind.COMPARISON,
            ):
                validate_markdown_table_columns(content_block.content, content_block.block_id)
            validate_plain_title(content_block.title, f"Block '{layout_block.id}'")
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

from __future__ import annotations
import json
from typing import Dict, List, Optional

from src.common.schemas import PlannerInput, PlannerOutput
from src.common.state import DocumentKind, DocumentBlueprint, LayoutBlockKind, NodeType
from src.helpers.tools import get_all_tool_schemas
from scripts.llm_client import LLMClient

PLANNER_SYSTEM_PROMPT = """
You are the Lead Document Architect. Your goal is to design both a hierarchical Document Graph and a LaTeX DocumentBlueprint.

### Your Responsibilities:
1. **Analyze Requirements:** Review the user query and available data files (CSVs, etc.).
2. **Build the Graph Structure:** - Define nodes (Sections, Tables, Figures).
   - Define hierarchy edges (e.g., Section 1 is the parent of Table 1.1).
3. **Strategic Planning:** For each node, specify:
   - `contextual_role`: How this node contributes to the document's logic.
   - `required_tools`: Which specific tools (from the provided list) should be used.
   - `key_points`: Specific data or insights that MUST be included.
   - `contextual_role` and `spec' should be described in more than 50 words.
   - `title` should not be just 'section 1'. It should contain the main concept of each node.
4. **LaTeX Layout Planning:** Design the document frame before any prose is drafted.
   - Choose `document.kind`: `report`, `article`, or `section_draft`.
   - For a report or article, provide a descriptive title and an appropriate document class.
   - Create one section layout for every `SECTION` and `SUBSECTION` node --
     and ONLY for `SECTION`/`SUBSECTION` nodes. A `TABLE` or `FIGURE` node
     (a notation table, a comparison table, a figure, ...) is never a section
     layout on its own; it is a `LayoutBlock` placed *inside* an existing
     section's `blocks` list (for a table, set that block's `node_id` to the
     TABLE node's id; for a figure, use `figure_slots` as described below).
     Giving a TABLE/FIGURE node its own entry in `blueprint.sections` is
     invalid and will be rejected.
   - Use semantic blocks and components deliberately: `tcolorbox` for takeaways, definitions, warnings, or case studies; `tabularx` or `longtable` for tables; `minipage` for direct comparisons; `wrapfigure` only for small supporting visuals; and `figure` for primary visual assets.
   - Every `FIGURE` node must have exactly one figure slot. A slot must name an existing owner section and an existing layout-block anchor. Do not place an image in more than one section.
   - `kind` vocabulary and how to map a requested element to it:
     - `theorem`: a numbered theorem/lemma statement. The composer wraps it in
       a numbered `theorem` environment -- write `instructions` asking only
       for the statement itself, never for `\\begin{{theorem}}` or a label.
     - `proof_sketch`: a proof's reasoning steps. The composer wraps it in
       amsthm's `proof` environment -- write `instructions` asking only for
       the steps themselves.
     - `notation_table`: a symbol/meaning (or term/definition) table. Behaves
       exactly like `table`.
     - There is no separate kind for every possible request wording -- map
       these onto the closest existing kind instead of inventing a new one:
       a "worked example" -> `case_study`; a "boxed result" or "boxed
       decomposition" -> `key_takeaway`; a "comparison table" -> `comparison`
       (or `table`/`notation_table` if it is not a direct two-way contrast);
       a "terminology table" or "classification table" -> `table`; a "risk
       callout" or generic "callout" -> `warning`; "aligned equations" or a
       "derivation" -> `equation` (write a complete `align`/`aligned`
       environment as the block's content); a "figure placeholder" -> `figure`.

### Critical Depth Rule:
- For a professional report, aim for at least 5-8 distinct nodes.
- Do NOT create a single large 'Introduction' node. Instead, break it into:
  1. Historical Background
  2. Current Market/Technical Status
  3. Scope of this Report
- If the user query is complex, favor creating SUBSECTION nodes under SECTION nodes.

### Available Tools:
{tool_schemas}

### Output Requirements:
The 'spec' field within each node MUST be a JSON object, not a string.
- Example: "spec": {{ "instruction": "Write about X", "key_points": ["point 1"] }}
You MUST return a JSON object that matches the PlannerOutput schema:
{{
  "nodes_to_create": [
    {{
      "id": "unique_id",
      "type": "SECTION | FIGURE | TABLE",
      "title": "Title of the section/figure/table",
      "spec": "Detailed drafting instructions",
      "contextual_role": "Strategic role",
      "required_tools": ["tool_name"]
    }}
  ],
  "hierarchy_edges": [
    {{
      "from_id": "parent_id",
      "to_id": "child_id",
      "relation_type": "hierarchy"
    }}
  ],
  "global_context": "Overall style and facts for the document",
  "blueprint": {{
    "document": {{
      "kind": "report | article | section_draft",
      "title": "Document title",
      "subtitle": "Optional subtitle",
      "author": "Optional author",
      "language": "en",
      "theme": "technical_report",
      "include_front_matter": ["title_page", "table_of_contents"]
    }},
    "document_class": "scrreprt | article",
    "packages": ["geometry", "graphicx", "booktabs", "tabularx", "tcolorbox", "hyperref", "cleveref", "placeins", "amsmath", "amssymb", "amsthm"],
    "sections": [
      {{
        "section_id": "an existing SECTION or SUBSECTION node id",
        "title": "The same section title",
        "level": "section | subsection | chapter",
        "blocks": [
          {{
            "id": "unique_block_id",
            "kind": "paragraph | key_takeaway | definition | warning | case_study | comparison | table | notation_table | theorem | proof_sketch | figure | equation | list",
            "component": "plain | tcolorbox | tabularx | longtable | minipage | wrapfigure | figure",
            "node_id": "optional existing graph node id",
            "asset_id": "a FIGURE node id only for a figure block",
            "instructions": "What this block must communicate"
          }}
        ]
      }}
    ],
    "figure_slots": {{
      "figure_node_id": {{
        "figure_id": "the same FIGURE node id",
        "owner_section": "one existing section id",
        "anchor_after_block": "one block id in that owner section",
        "layout": "figure | wrapfigure | minipage",
        "width": "0.72\\\\textwidth",
        "float_options": "htbp",
        "label": "fig:descriptive_label",
        "caption": "Descriptive caption",
        "required": true
      }}
    }},
    "layout_policy": {{
      "maximum_full_width_figures_per_section": 1,
      "insert_float_barrier_after_section": true,
      "allow_repeated_assets": false
    }}
  }}
}}

Return ONLY valid JSON. No commentary.
"""

class PlannerAgent:
    """
    Transforms a user query into an initial Document Graph structure.
    Acts as the 'Architect' of the multi-agent system.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm = llm_client or LLMClient()
        # Fetch tool definitions to inform the LLM of available capabilities
        self.available_tools = get_all_tool_schemas()

    def generate_plan(self, planner_input: PlannerInput, max_attempts: int = 3) -> PlannerOutput:
        """
        Communicates with the LLM to design the document structure.

        A single LLM call has no guaranteed way to enforce the graph/layout
        cross-references ``_validate_blueprint`` checks (matching node and
        section IDs, valid figure references, one layout per section, ...),
        so a plan that fails validation is retried with a fresh LLM sample
        instead of failing the whole request outright.
        """
        # Prepare the system prompt with tool metadata
        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            tool_schemas=json.dumps(self.available_tools, indent=2)
        )

        user_prompt = self._build_detailed_user_prompt(planner_input)

        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = self.llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.2, ##### 원래 0.2 정도; Low temperature for structural consistency
                    max_new_tokens=12000,
                )

                json_block = self.llm.extract_json_block(raw_response)
                plan_data = json.loads(json_block)

                plan = PlannerOutput.model_validate(plan_data)
                if plan.request_id is None:
                    plan.request_id = planner_input.request_id
                self._validate_blueprint(plan)
                return plan
            except Exception as exc:
                last_error = exc
                print(f"[PlannerAgent] Attempt {attempt}/{max_attempts} failed: {exc}")

        raise ValueError(
            f"PlannerAgent failed to produce a valid plan after {max_attempts} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _validate_blueprint(plan: PlannerOutput) -> None:
        """Reject graph/layout contracts that cannot be composed safely.

        The LLM chooses the layout, but this deterministic validation prevents a
        later composer from receiving orphaned sections, invalid figure anchors,
        or a figure that has more than one planned location.
        """
        node_ids = [node.get("id") for node in plan.nodes_to_create if node.get("id")]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Planner output contains duplicate node IDs.")

        nodes_by_id: Dict[str, Dict] = {
            node["id"]: node for node in plan.nodes_to_create if node.get("id")
        }
        section_ids = {
            node_id
            for node_id, node in nodes_by_id.items()
            if node.get("type") in {NodeType.SECTION.value, NodeType.SUBSECTION.value}
        }
        figure_ids = {
            node_id
            for node_id, node in nodes_by_id.items()
            if node.get("type") == NodeType.FIGURE.value
        }
        blueprint = plan.blueprint

        if blueprint.document.kind != DocumentKind.SECTION_DRAFT and not blueprint.document.title:
            raise ValueError("A report or article blueprint must define a document title.")

        layouts_by_id = {layout.section_id: layout for layout in blueprint.sections}
        if len(layouts_by_id) != len(blueprint.sections):
            raise ValueError("DocumentBlueprint contains duplicate section layouts.")
        if set(layouts_by_id) != section_ids:
            missing = sorted(section_ids - set(layouts_by_id))
            unexpected = sorted(set(layouts_by_id) - section_ids)
            raise ValueError(
                "DocumentBlueprint section layouts must match graph sections exactly. "
                f"Missing: {missing}; unexpected: {unexpected}."
            )

        all_block_ids = set()
        for section_id, layout in layouts_by_id.items():
            block_ids = set()
            for block in layout.blocks:
                if block.id in all_block_ids:
                    raise ValueError(f"DocumentBlueprint contains duplicate block ID '{block.id}'.")
                all_block_ids.add(block.id)
                block_ids.add(block.id)
                if block.node_id and block.node_id not in nodes_by_id:
                    raise ValueError(
                        f"Layout block '{block.id}' refers to unknown node '{block.node_id}'."
                    )
                if block.kind == LayoutBlockKind.FIGURE and not block.asset_id:
                    raise ValueError(f"Figure block '{block.id}' must declare an asset_id.")
                if block.asset_id and block.asset_id not in figure_ids:
                    raise ValueError(
                        f"Layout block '{block.id}' refers to unknown figure '{block.asset_id}'."
                    )

        if set(blueprint.figure_slots) != figure_ids:
            missing = sorted(figure_ids - set(blueprint.figure_slots))
            unexpected = sorted(set(blueprint.figure_slots) - figure_ids)
            raise ValueError(
                "Every graph figure must have exactly one blueprint figure slot. "
                f"Missing: {missing}; unexpected: {unexpected}."
            )

        for figure_id, placement in blueprint.figure_slots.items():
            if placement.figure_id != figure_id:
                raise ValueError(
                    f"Figure slot key '{figure_id}' does not match figure_id '{placement.figure_id}'."
                )
            if placement.owner_section not in section_ids:
                raise ValueError(
                    f"Figure '{figure_id}' has unknown owner section '{placement.owner_section}'."
                )
            if placement.anchor_after_block:
                owner_layout = layouts_by_id[placement.owner_section]
                owner_block_ids = {block.id for block in owner_layout.blocks}
                if placement.anchor_after_block not in owner_block_ids:
                    raise ValueError(
                        f"Figure '{figure_id}' has an invalid anchor "
                        f"'{placement.anchor_after_block}' in section '{placement.owner_section}'."
                    )

    def _build_detailed_user_prompt(self, planner_input: PlannerInput) -> str:
        """
        Creates a context-rich prompt including file inputs and template requirements.
        """
        input_files = "\n".join([f"- {f.name}: {f.path} ({f.file_type})" for f in planner_input.inputs])
        
        prompt = f"""
        Document Request:
        - Project ID: {planner_input.request_id}
        - Type: {planner_input.doc_type}
        - Language: {planner_input.language}
        - User Query: "{planner_input.user_query}"

        Template Info:
        - Main Template: {planner_input.template.name}
        - Suggested Order: {planner_input.template.section_order}

        Available Data Inputs:
        {input_files if input_files else "No external data files provided."}

        Generate the 'nodes_to_create' and 'hierarchy_edges' now.
        """
        return prompt.strip()

    @staticmethod
    def save_plan(plan: PlannerOutput, output_path: str):
        """Saves the generated plan to a JSON file for tracking."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(plan.model_dump_json(indent=2))

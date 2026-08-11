from __future__ import annotations
import json
import re
from typing import Dict, List, Literal, Optional, TypeAlias

from pydantic import BaseModel, ConfigDict

from src.agents.latex_integrator import (
    KNOWN_SECTION_LEVELS,
    SAFE_DOCUMENT_CLASSES,
    validate_plain_title,
)
from src.common.schemas import PlannerInput, PlannerOutput
from src.common.state import DocumentKind, DocumentBlueprint, LayoutBlockKind, NodeType, SAFE_IDENTIFIER
from src.helpers.tools import get_all_tool_schemas
from scripts.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Structured-output wire schema (OpenAI / Gemini)
#
# This is a *separate*, deliberately narrower schema from PlannerOutput /
# DocumentBlueprint (src/common/state.py, src/common/schemas.py). Those stay
# the internal representation used by every downstream agent; this is only
# the contract handed to the provider's structured-output feature so the API
# itself -- not a prompt instruction -- guarantees the LLM's JSON has every
# required key (in particular ``blueprint``, whose omission is what used to
# make plans fail validation).
#
# Both OpenAI's strict json_schema mode and pydantic's own conventions push
# toward the same shape: every field must be present in the output (no
# silently-omittable fields), so nothing here has a default -- an
# actually-optional field is spelled ``Optional[X]`` with no default, which
# forces the key to be present while still allowing ``null``. OpenAI's strict
# mode also has no way to express "object with arbitrary keys" (needs
# ``additionalProperties: false`` on every object), so the two dict-shaped
# spots in the internal model -- ``figure_slots`` and ``layout_policy`` --
# are written here as a list / a fixed set of fields instead, and converted
# back to the internal shape in ``_wire_output_to_plan_data`` below.
# ---------------------------------------------------------------------------

_SECTION_NODE_TYPES: TypeAlias = Literal["SECTION", "SUBSECTION"]
# FIGURE only -- TABLE used to be offered here too, but nothing downstream
# (TextAgent, ImageAgent) ever reads a table's node -- LayoutBlock.node_id
# was validated for existence and then never consumed again, so a "TABLE
# asset node" did literally nothing except exist. The model was observed
# generalizing that same "important content gets its own node" pattern to
# theorem/case_study ("worked example") content, and since TABLE/FIGURE were
# the only asset node types available, it mistyped those as TABLE or FIGURE
# nodes -- which then fail _validate_blueprint's figure_slots check and burn
# a retry. Removing the dead TABLE option (and node_id below) removes both
# the false affordance and the temptation.
_ASSET_NODE_TYPES: TypeAlias = Literal["FIGURE"]
# Built from the composer's own allowlist (src/agents/latex_integrator.py)
# rather than duplicated here, so the two can't drift apart the way
# document_class did: the wire schema left it as a bare `str`, so a model
# had no structural reason not to write "article (11pt)" -- a value that
# only fails much later, in Phase 3's _write_preamble.
_DOCUMENT_CLASSES: TypeAlias = Literal[tuple(sorted(SAFE_DOCUMENT_CLASSES))]
_LAYOUT_BLOCK_KINDS: TypeAlias = Literal[
    "paragraph", "list", "key_takeaway", "definition", "warning", "case_study",
    "comparison", "table", "figure", "equation", "theorem", "proof_sketch",
    "notation_table",
]
_LATEX_COMPONENTS: TypeAlias = Literal[
    "plain", "tcolorbox", "tabularx", "longtable", "minipage", "figure",
]
# "wrapfigure" (text wrapped around a figure) is deliberately not offered --
# see the comment on state.FigureLayout for why it's disabled outright.
_FIGURE_LAYOUTS: TypeAlias = Literal["figure", "minipage"]
_FRONT_MATTER_ITEMS: TypeAlias = Literal[
    "title_page", "abstract", "table_of_contents", "list_of_figures", "list_of_tables",
]
_SECTION_LEVELS: TypeAlias = Literal[tuple(sorted(KNOWN_SECTION_LEVELS))]
_DOCUMENT_KINDS: TypeAlias = Literal["report", "article", "section_draft"]


class _WireNodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction: str
    key_points: List[str]


class _WireAssetNode(BaseModel):
    """A FIGURE node -- a real image ImageAgent will generate -- referenced
    by asset_id from inside a section's blocks, never a section layout of
    its own. A table, theorem, case study, or any other non-visual content
    never belongs here: it's written entirely inside a LayoutBlock's own
    `instructions` (see PLANNER_STRATEGY_PROMPT)."""

    model_config = ConfigDict(extra="forbid")
    id: str
    type: _ASSET_NODE_TYPES
    title: str
    spec: _WireNodeSpec
    contextual_role: str
    required_tools: List[str]


class _WireEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    from_id: str
    to_id: str


class _WireBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: _LAYOUT_BLOCK_KINDS
    component: _LATEX_COMPONENTS
    title: Optional[str]
    instructions: str
    asset_id: Optional[str]


class _WireSectionNode(BaseModel):
    """A SECTION/SUBSECTION node fused with its own layout.

    The internal model (state.DocumentBlueprint) keeps these as two separate
    lists -- ``nodes_to_create`` and ``blueprint.sections``, matched up by a
    ``section_id`` string that has to be retyped identically in both places.
    A model has no structural reason to keep the two in sync (a real failure
    mode: "Missing: ['sec_prerequisites']; unexpected: ['sec_prereq']" --
    same section, two different id spellings for it in the same response),
    so the wire schema asks for it exactly once here and
    ``_wire_output_to_plan_data`` derives both of the internal shape's lists
    from this single object -- the two ids literally cannot diverge because
    there's only one id to begin with.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    type: _SECTION_NODE_TYPES
    title: str
    spec: _WireNodeSpec
    contextual_role: str
    required_tools: List[str]
    level: _SECTION_LEVELS
    blocks: List[_WireBlock]


class _WireFigureSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    figure_id: str
    owner_section: str
    anchor_after_block: Optional[str]
    layout: _FIGURE_LAYOUTS
    width: str
    float_options: str
    label: Optional[str]
    caption: Optional[str]
    required: bool


class _WireDocumentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: _DOCUMENT_KINDS
    title: Optional[str]
    subtitle: Optional[str]
    author: Optional[str]
    language: str
    include_front_matter: List[_FRONT_MATTER_ITEMS]
    # The only legitimate place for something like "11pt"/"twocolumn" --
    # document_class itself must stay a bare class name (see
    # _DOCUMENT_CLASSES above), or _write_preamble rejects it outright.
    class_options: List[str]


class _WireLayoutPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    maximum_full_width_figures_per_section: int
    insert_float_barrier_after_section: bool
    allow_repeated_assets: bool


class _WireBlueprint(BaseModel):
    """Everything about the document frame except the section layouts
    themselves -- those now live on each ``_WireSectionNode`` (see its
    docstring for why), and ``_wire_output_to_plan_data`` rebuilds
    ``blueprint.sections`` from ``section_nodes`` to restore the internal
    model's shape."""

    model_config = ConfigDict(extra="forbid")
    document: _WireDocumentSpec
    document_class: _DOCUMENT_CLASSES
    packages: List[str]
    figure_slots: List[_WireFigureSlot]
    layout_policy: _WireLayoutPolicy


class PlannerWireOutput(BaseModel):
    """The exact JSON shape requested from the provider's structured-output feature."""
    model_config = ConfigDict(extra="forbid")
    section_nodes: List[_WireSectionNode]
    asset_nodes: List[_WireAssetNode]
    hierarchy_edges: List[_WireEdge]
    global_context: str
    blueprint: _WireBlueprint


def _wire_output_to_plan_data(wire: dict) -> dict:
    """Reshape a validated ``PlannerWireOutput`` dict into ``PlannerOutput``'s shape.

    ``PlannerOutput.model_validate()`` (and ``_validate_blueprint`` after it)
    is unchanged by structured output -- it's still the source of truth for
    the semantic/cross-reference checks (valid figure anchors, one layout
    per section, ...) that no JSON schema can express. This function
    reverses the shape adjustments the wire schema needed for strict mode:
    ``figure_slots`` list -> dict keyed by ``figure_id``; each
    ``_WireSectionNode`` -> one ``nodes_to_create`` entry (its node fields)
    plus one ``blueprint.sections`` entry (its ``level``/``blocks``), both
    keyed by the exact same ``id`` since they're read off the same object;
    and dropping back in the internal model's defaults for fields the wire
    schema doesn't ask the LLM to decide (``document.date``/``document.theme``).
    """
    blueprint = wire["blueprint"]
    figure_slots = {slot["figure_id"]: slot for slot in blueprint["figure_slots"]}

    nodes_to_create = list(wire["asset_nodes"])
    sections = []
    for section_node in wire["section_nodes"]:
        nodes_to_create.append({
            "id": section_node["id"],
            "type": section_node["type"],
            "title": section_node["title"],
            "spec": section_node["spec"],
            "contextual_role": section_node["contextual_role"],
            "required_tools": section_node["required_tools"],
        })
        sections.append({
            "section_id": section_node["id"],
            "title": section_node["title"],
            "level": section_node["level"],
            "blocks": section_node["blocks"],
        })

    return {
        "nodes_to_create": nodes_to_create,
        "hierarchy_edges": wire["hierarchy_edges"],
        "global_context": wire["global_context"],
        "blueprint": {
            "document": blueprint["document"],
            "document_class": blueprint["document_class"],
            "packages": blueprint["packages"],
            "sections": sections,
            "figure_slots": figure_slots,
            "layout_policy": blueprint["layout_policy"],
        },
    }


_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9_-]")
# Matches a title the model copied straight from a numbered list in the user
# query ("1 Introduction", "1.1 Historical Background", "Section 3: ...") --
# \section{}/\subsection{} already number themselves, so this prints twice.
# Deliberately requires digits followed by '.'/':'/')'/whitespace (not just
# any leading digit) so a title like "3D Printing Techniques" doesn't match.
_LEADING_SECTION_NUMBER = re.compile(r"^(section\s+)?\d+(\.\d+)*[.:)]?\s", re.IGNORECASE)


def _safe_id(value: object) -> object:
    """Coerce an LLM-authored id into something ``state.SAFE_IDENTIFIER`` accepts.

    Models reliably reach for LaTeX-label-style ids (``sec:introduction``,
    ``fig:capm-formula``) even though every id in this pipeline is also a
    filesystem path component (``sections/{node_id}.blocks.json``) and so is
    restricted to ``[A-Za-z0-9_-]``. Rejecting that outright wastes a whole
    retry attempt on something a straightforward, deterministic rewrite can
    fix -- and since the same input always maps to the same output, rewriting
    every occurrence of an id independently keeps cross-references (a block's
    ``asset_id`` pointing at a FIGURE node's ``id``, a figure slot's
    ``owner_section`` pointing at a ``section_id``, ...) consistent with
    each other.
    """
    if not isinstance(value, str) or not value or SAFE_IDENTIFIER.fullmatch(value):
        return value
    cleaned = _UNSAFE_ID_CHARS.sub("_", value)
    if not cleaned or not cleaned[0].isalnum():
        cleaned = f"id_{cleaned}"
    return cleaned


def _sanitize_plan_identifiers(plan_data: dict) -> dict:
    """Rewrite every id-like string in a parsed plan to a safe identifier, in place.

    Runs on the same ``PlannerOutput``-shaped dict regardless of whether it
    came from structured output or the legacy prompt-only path, right before
    ``PlannerOutput.model_validate()`` -- the one point both paths converge on.
    """
    for node in plan_data.get("nodes_to_create") or []:
        if isinstance(node, dict) and "id" in node:
            node["id"] = _safe_id(node["id"])

    for edge in plan_data.get("hierarchy_edges") or []:
        if isinstance(edge, dict):
            for key in ("from_id", "to_id", "parent", "child"):
                if edge.get(key):
                    edge[key] = _safe_id(edge[key])
        elif isinstance(edge, list):
            for i, value in enumerate(edge):
                edge[i] = _safe_id(value)

    blueprint = plan_data.get("blueprint") or {}
    for section in blueprint.get("sections") or []:
        if section.get("section_id"):
            section["section_id"] = _safe_id(section["section_id"])
        for block in section.get("blocks") or []:
            if block.get("id"):
                block["id"] = _safe_id(block["id"])
            if block.get("asset_id"):
                block["asset_id"] = _safe_id(block["asset_id"])

    figure_slots = blueprint.get("figure_slots") or {}
    if isinstance(figure_slots, dict):
        sanitized_slots = {}
        for key, slot in figure_slots.items():
            if isinstance(slot, dict):
                if slot.get("figure_id"):
                    slot["figure_id"] = _safe_id(slot["figure_id"])
                if slot.get("owner_section"):
                    slot["owner_section"] = _safe_id(slot["owner_section"])
                if slot.get("anchor_after_block"):
                    slot["anchor_after_block"] = _safe_id(slot["anchor_after_block"])
            sanitized_slots[_safe_id(key)] = slot
        blueprint["figure_slots"] = sanitized_slots

    return plan_data

PLANNER_STRATEGY_PROMPT = """
You are the Lead Document Architect. Your goal is to design both a hierarchical Document Graph and a LaTeX DocumentBlueprint.

### Your Responsibilities:
1. **Analyze Requirements:** Review the user query.
2. **Build the Graph Structure:** - Define nodes (Sections, Tables, Figures).
   - Define hierarchy edges (e.g., Section 1 is the parent of Table 1.1).
3. **Strategic Planning:** For each node, specify:
   - `contextual_role`: How this node contributes to the document's logic.
   - `required_tools`: Which specific tools (from the provided list) should be used.
   - `key_points`: Specific data or insights that MUST be included.
   - `contextual_role` and `spec` should be described in more than 50 words.
   - `title` should not be just 'section 1'. It should contain the main concept of each node.
   - Never put a section/subsection number in a `title` (no `"1 Introduction"`,
     `"1.1 Historical Background"`, `"Section 3: ..."`, ...) -- even if the
     user query lists sections as a numbered list, strip the leading number
     before writing the title. `\section{{}}`/`\subsection{{}}` number
     themselves automatically; a number baked into the title text prints twice.
4. **LaTeX Layout Planning:** Design the document frame before any prose is drafted.
   - Choose `document.kind`: `report`, `article`, or `section_draft`.
   - For a report or article, provide a descriptive title and an appropriate document class.
   - Every `SECTION`/`SUBSECTION` node carries its own layout (`level` and
     `blocks`) directly -- ONLY `SECTION`/`SUBSECTION` nodes belong in
     `section_nodes`. `asset_nodes` holds ONLY `FIGURE` nodes: a real image
     that ImageAgent will render as a file. A table, notation table,
     comparison, theorem, case study, definition, warning, key takeaway, or
     any other non-visual content is NEVER an asset node, no matter how
     important or richly specified it seems -- it is written entirely as a
     `LayoutBlock` *inside* an existing section's `blocks` list instead, and
     all of its detail (data source, required tool, key facts to include)
     goes directly in that block's own `instructions` string. In particular,
     a "notation table" or "symbol table" is a `notation_table` block, and a
     "worked example" is a `case_study` block -- NEVER a FIGURE, even though
     the section discussing them could also separately contain a real
     illustrative figure. Only a `FIGURE` node gets a `figure_slots` entry,
     as described below, and every `FIGURE` node MUST also be referenced by
     exactly one `figure`-kind `LayoutBlock` (same `asset_id`) in some
     section's `blocks` -- an image with a slot but no block referencing it
     is rejected, because no prose would ever explain what it is.
   - Use semantic blocks and components deliberately: `tcolorbox` for takeaways, definitions, warnings, or case studies; `tabularx` or `longtable` for tables; `minipage` for direct comparisons; and `figure` for every visual asset. Every figure slot's `layout` must be `figure` (a full-width float) or `minipage` (for a side-by-side pair) -- never a text-wrapped layout; a figure needs a real caption and enough surrounding content on its own, not a specific paragraph length near it to wrap around.
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

### Available Tools:
{tool_schemas}
"""

# Only used when the provider has no native structured-output support (see
# LLMClient.supports_structured_output) and the plan has to be requested via
# prompt instruction + best-effort JSON extraction instead. When structured
# output is available, the provider's JSON-schema enforcement (built from
# PlannerWireOutput above) already guarantees this exact shape, so repeating
# it in the prompt would be redundant.
PLANNER_LEGACY_OUTPUT_FORMAT_PROMPT = """
### Output Requirements:
The 'spec' field within each node MUST be a JSON object, not a string.
- Example: "spec": {{ "instruction": "Write about X", "key_points": ["point 1"] }}
You MUST return a JSON object that matches the PlannerOutput schema:
{{
  "nodes_to_create": [
    {{
      "id": "unique_id",
      "type": "SECTION | SUBSECTION | FIGURE -- FIGURE only for a real image; a table/theorem/case_study/etc. is a LayoutBlock, never a node",
      "title": "Title of the section/figure",
      "spec": {{ "instruction": "Detailed drafting instructions", "key_points": ["point 1"] }},
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
      "include_front_matter": ["title_page", "table_of_contents"],
      "class_options": ["11pt"]
    }},
    "document_class": "article | report | scrartcl | scrreprt -- a bare class name ONLY, never with options like '(11pt)'; put font size/column options in document.class_options instead",
    "packages": ["geometry", "graphicx", "booktabs", "tabularx", "tcolorbox", "hyperref", "cleveref", "placeins", "amsmath", "amssymb", "amsthm"],
    "sections": [
      {{
        "section_id": "an existing SECTION or SUBSECTION node id",
        "title": "The same section title",
        "level": "section | subsection | subsubsection | chapter",
        "blocks": [
          {{
            "id": "unique_block_id",
            "kind": "paragraph | key_takeaway | definition | warning | case_study | comparison | table | notation_table | theorem | proof_sketch | figure | equation | list",
            "component": "plain | tcolorbox | tabularx | longtable | minipage | figure",
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
        "layout": "figure | minipage",
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

blueprint is REQUIRED -- never return nodes_to_create/hierarchy_edges/global_context alone.
Return ONLY valid JSON. No commentary.
"""

# Appended only when PlannerInput.allow_figures is False (no image-generation
# model available for this run). Overrides every FIGURE-related instruction
# in PLANNER_STRATEGY_PROMPT above -- content that would have been a figure
# must be re-expressed as a non-visual block instead of simply omitted, so
# the document doesn't lose the information the figure would have carried.
PLANNER_NO_IMAGES_PROMPT = """
### Image Generation Is Disabled For This Run
No image-generation model is available. Regardless of anything said above:
- Do not create any `FIGURE` node or any `asset_nodes` entry.
- Do not add any `figure_slots` entry to the blueprint.
- Do not use `kind: "figure"` / `component: "figure"` for any LayoutBlock.
If content would otherwise have been a real illustrative image, represent it
instead as a `table`, `notation_table`, `comparison`, or descriptive
`paragraph` block -- never as a FIGURE node.
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

        When the provider supports structured output (OpenAI, Gemini -- see
        ``LLMClient.supports_structured_output``), the request enforces
        ``PlannerWireOutput`` server-side, so the response is guaranteed to
        include every required key (``blueprint`` in particular -- its
        omission used to be the most common validation failure here). Other
        providers fall back to the legacy prompt-instruction + best-effort
        JSON extraction path.

        Either way, a single LLM call still has no way to enforce the
        graph/layout *cross-references* ``_validate_blueprint`` checks
        (matching node and section IDs, valid figure anchors, one layout per
        section, ...) -- no JSON schema can express those -- so a plan that
        fails that validation is retried instead of failing the whole
        request outright. The validation error is fed back into the next
        attempt's prompt (mirroring ``TextAgent.process_node``'s retry loop)
        rather than resampling blind: some mistakes -- notably the model
        inventing an unreferenced FIGURE node for a notation table or worked
        example -- were observed recurring across independent samples at the
        same rate, so a plain resample wastes attempts on the same error
        instead of correcting it.
        """
        use_structured = self.llm.supports_structured_output()
        system_prompt = PLANNER_STRATEGY_PROMPT.format(
            tool_schemas=json.dumps(self.available_tools, indent=2)
        )
        if not planner_input.allow_figures:
            system_prompt += PLANNER_NO_IMAGES_PROMPT
        if not use_structured:
            system_prompt += PLANNER_LEGACY_OUTPUT_FORMAT_PROMPT

        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                user_prompt = self._build_detailed_user_prompt(
                    planner_input, previous_error=str(last_error) if last_error else None
                )
                if use_structured:
                    raw_response = self.llm.generate_structured(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        response_model=PlannerWireOutput,
                        schema_name="planner_output",
                        temperature=0.2,
                        max_new_tokens=12000,
                    )
                    plan_data = _wire_output_to_plan_data(json.loads(raw_response))
                else:
                    raw_response = self.llm.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.2, ##### 원래 0.2 정도; Low temperature for structural consistency
                        max_new_tokens=12000,
                    )
                    json_block = self.llm.extract_json_block(raw_response)
                    plan_data = json.loads(json_block)

                plan_data = _sanitize_plan_identifiers(plan_data)
                plan = PlannerOutput.model_validate(plan_data)
                if plan.request_id is None:
                    plan.request_id = planner_input.request_id
                self._validate_blueprint(plan, allow_figures=planner_input.allow_figures)
                return plan
            except Exception as exc:
                last_error = exc
                print(f"[PlannerAgent] Attempt {attempt}/{max_attempts} failed: {exc}")

        raise ValueError(
            f"PlannerAgent failed to produce a valid plan after {max_attempts} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _validate_blueprint(plan: PlannerOutput, allow_figures: bool = True) -> None:
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
        if not allow_figures and figure_ids:
            raise ValueError(
                "Image generation is disabled for this run -- the plan must not include any "
                f"FIGURE node: {sorted(figure_ids)}. Represent this content as a table, "
                "notation_table, comparison, or paragraph block instead."
            )
        blueprint = plan.blueprint

        if blueprint.document.kind != DocumentKind.SECTION_DRAFT and not blueprint.document.title:
            raise ValueError("A report or article blueprint must define a document title.")
        validate_plain_title(blueprint.document.title, "Document")
        validate_plain_title(blueprint.document.subtitle, "Document subtitle")

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
        referenced_figure_ids = set()
        for section_id, layout in layouts_by_id.items():
            validate_plain_title(layout.title, f"Section '{section_id}'")
            if _LEADING_SECTION_NUMBER.match(layout.title):
                raise ValueError(
                    f"Section '{section_id}' title {layout.title!r} starts with a number "
                    "-- \\section{}/\\subsection{} already number themselves automatically; "
                    "remove the leading number from the title."
                )
            block_ids = set()
            for block in layout.blocks:
                if block.id in all_block_ids:
                    raise ValueError(f"DocumentBlueprint contains duplicate block ID '{block.id}'.")
                all_block_ids.add(block.id)
                block_ids.add(block.id)
                validate_plain_title(block.title, f"Layout block '{block.id}'")
                if block.kind == LayoutBlockKind.FIGURE and not block.asset_id:
                    raise ValueError(f"Figure block '{block.id}' must declare an asset_id.")
                if block.asset_id and block.asset_id not in figure_ids:
                    raise ValueError(
                        f"Layout block '{block.id}' refers to unknown figure '{block.asset_id}'."
                    )
                if block.kind == LayoutBlockKind.FIGURE:
                    referenced_figure_ids.add(block.asset_id)

        # A FIGURE node with a slot but no `kind=figure` LayoutBlock pointing
        # back at it renders with nothing in the prose that ever explains or
        # even mentions it -- observed live as the model creating orphan
        # FIGURE nodes named e.g. "fig_notation_table"/"fig_worked_example"
        # (a plausible-sounding illustration for a section) with a real
        # figure_slot but no block anywhere referencing them, so no section's
        # drafted text ever discusses what the image is. Requiring the same
        # asset_id/block link TextAgent's figure_reference type already
        # relies on closes that gap structurally instead of trusting the
        # model to remember to add one.
        if figure_ids - referenced_figure_ids:
            raise ValueError(
                "Every FIGURE node must be referenced by a `figure`-kind LayoutBlock "
                "(matching asset_id) in some section, so the drafted prose actually "
                "introduces it -- unreferenced: "
                f"{sorted(figure_ids - referenced_figure_ids)}. If this content isn't a "
                "genuine image, use a `table`/`notation_table`/`case_study` block "
                "instead of a FIGURE node."
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

    def _build_detailed_user_prompt(
        self, planner_input: PlannerInput, previous_error: Optional[str] = None
    ) -> str:
        """
        Creates a context-rich prompt including the request and template requirements.
        """
        retry_notice = (
            f"""
        Your previous plan was rejected: {previous_error}
        Fix this specific problem in your next response.
        """
            if previous_error
            else ""
        )
        prompt = f"""
        {retry_notice}Document Request:
        - Project ID: {planner_input.request_id}
        - Type: {planner_input.doc_type}
        - Language: {planner_input.language}
        - User Query: "{planner_input.user_query}"

        Template Info:
        - Main Template: {planner_input.template.name}
        - Suggested Order: {planner_input.template.section_order}

        Generate the 'nodes_to_create' and 'hierarchy_edges' now.
        """
        return prompt.strip()

    @staticmethod
    def save_plan(plan: PlannerOutput, output_path: str):
        """Saves the generated plan to a JSON file for tracking."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(plan.model_dump_json(indent=2))

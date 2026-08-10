"""Shared document, layout, and asset state for the generation pipeline.

The graph stores semantic content state.  ``DocumentBlueprint`` stores the
LaTeX presentation contract decided by the layout agent, and ``FigureRegistry``
enforces that a generated visual has one intentional location in the document.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# Section and node ids are written into filesystem paths verbatim (for example
# ``sections/{node_id}.blocks.json``), so they must be safe path components even
# though they ultimately originate from LLM-generated planner output.
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _validate_safe_identifier(value: str) -> str:
    if not SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"'{value}' is not a safe identifier: use only letters, digits, '_' or '-', "
            "starting with a letter or digit."
        )
    return value


class NodeStatus(str, Enum):
    """Workflow status for an individual document node."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DRAFTED = "DRAFTED"
    REVIEWED = "REVIEWED"
    VALIDATED = "VALIDATED"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


class NodeType(str, Enum):
    """Semantic types of content stored in a document graph."""

    SECTION = "SECTION"
    SUBSECTION = "SUBSECTION"
    TABLE = "TABLE"
    FIGURE = "FIGURE"
    BIBLIOGRAPHY = "BIBLIOGRAPHY"


class DocumentKind(str, Enum):
    """Top-level LaTeX products supported by the layout agent."""

    REPORT = "report"
    ARTICLE = "article"
    SECTION_DRAFT = "section_draft"


class FrontMatterItem(str, Enum):
    TITLE_PAGE = "title_page"
    ABSTRACT = "abstract"
    TABLE_OF_CONTENTS = "table_of_contents"
    LIST_OF_FIGURES = "list_of_figures"
    LIST_OF_TABLES = "list_of_tables"


class LayoutBlockKind(str, Enum):
    """Meaningful blocks that the LaTeX composer can render."""

    PARAGRAPH = "paragraph"
    LIST = "list"
    KEY_TAKEAWAY = "key_takeaway"
    DEFINITION = "definition"
    WARNING = "warning"
    CASE_STUDY = "case_study"
    COMPARISON = "comparison"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    THEOREM = "theorem"
    PROOF_SKETCH = "proof_sketch"
    NOTATION_TABLE = "notation_table"


class LatexComponent(str, Enum):
    """LaTeX component selected by the layout agent for a semantic block."""

    PLAIN = "plain"
    TCOLORBOX = "tcolorbox"
    TABULARX = "tabularx"
    LONGTABLE = "longtable"
    MINIPAGE = "minipage"
    FIGURE_FLOAT = "figure"


class FigureLayout(str, Enum):
    """Permitted layouts for a single registered figure.

    ``wrapfigure`` (text wrapped around the figure) used to be a third
    option here, but wrapfig only lays out correctly when there's enough
    surrounding paragraph text to wrap around the figure's full height --
    nothing in the pipeline checks that before committing to the layout, so
    a short trailing paragraph next to a large figure silently overflows
    past the paragraph and collides with whatever comes next. Disabled
    outright rather than patched, since detecting "enough trailing text"
    reliably needs more than a blueprint-time heuristic.
    """

    FLOAT = "figure"
    MINIPAGE = "minipage"


class DocumentSpec(BaseModel):
    """User-facing identity and frame requirements for a document."""

    kind: DocumentKind = DocumentKind.REPORT
    title: Optional[str] = None
    subtitle: Optional[str] = None
    author: Optional[str] = None
    date: str = r"\today"
    language: str = "en"
    theme: str = "technical_report"
    include_front_matter: List[FrontMatterItem] = Field(default_factory=list)
    class_options: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LayoutBlock(BaseModel):
    """A planned block within a section, before its text is generated."""

    id: str
    kind: LayoutBlockKind
    component: LatexComponent = LatexComponent.PLAIN
    title: Optional[str] = None
    instructions: str = ""
    asset_id: Optional[str] = None


class SectionLayout(BaseModel):
    """Ordered LaTeX layout contract for one section or subsection."""

    section_id: str
    title: str
    level: str = "section"
    blocks: List[LayoutBlock] = Field(default_factory=list)

    @field_validator("section_id")
    @classmethod
    def _safe_section_id(cls, value: str) -> str:
        return _validate_safe_identifier(value)


class FigurePlacement(BaseModel):
    """The sole permitted rendering location for a figure asset."""

    figure_id: str
    owner_section: str
    anchor_after_block: Optional[str] = None
    layout: FigureLayout = FigureLayout.FLOAT
    width: str = r"0.72\textwidth"
    float_options: str = "htbp"
    label: Optional[str] = None
    caption: Optional[str] = None
    required: bool = True


class DocumentBlueprint(BaseModel):
    """LaTeX layout plan produced before prose and figures are composed."""

    document: DocumentSpec = Field(default_factory=DocumentSpec)
    document_class: str = "scrreprt"
    packages: List[str] = Field(default_factory=list)
    sections: List[SectionLayout] = Field(default_factory=list)
    figure_slots: Dict[str, FigurePlacement] = Field(default_factory=dict)
    layout_policy: Dict[str, Any] = Field(default_factory=dict)

    def section_for_figure(self, figure_id: str) -> Optional[str]:
        """Return the only section authorized to render ``figure_id``."""
        placement = self.figure_slots.get(figure_id)
        return placement.owner_section if placement else None


class FigureRegistryEntry(BaseModel):
    """Lifecycle state for one generated figure and its unique placement."""

    figure_id: str
    asset_path: Optional[str] = None
    owner_section: Optional[str] = None
    label: Optional[str] = None
    caption: Optional[str] = None
    placement_count: int = 0
    rendered_in_section: Optional[str] = None
    required: bool = True
    last_updated: datetime = Field(default_factory=datetime.now)


# Only kinds that render as a genuinely numbered LaTeX object (a
# ``\newtheorem``-backed environment or counter, a numbered ``equation``, a
# captioned ``table``) get a default label. ``\ref``/``\Cref`` to an
# unnumbered tcolorbox callout (key_takeaway/warning/case_study/comparison)
# would just silently print whatever the nearest *other* counter happens to
# be -- wrong output, not merely a missing label -- so those are left out
# rather than given a label that would render incorrectly.
_LABEL_PREFIX_BY_KIND: Dict[LayoutBlockKind, str] = {
    LayoutBlockKind.THEOREM: "thm",
    LayoutBlockKind.DEFINITION: "def",
    LayoutBlockKind.EQUATION: "eq",
    LayoutBlockKind.TABLE: "tab",
    LayoutBlockKind.NOTATION_TABLE: "tab",
}


def default_block_label(kind: LayoutBlockKind, block_id: str) -> Optional[str]:
    """The deterministic label a block gets if it doesn't supply its own.

    ``block_id`` is already guaranteed unique (``PlannerAgent._validate_blueprint``
    rejects duplicate block IDs), so ``{prefix}:{block_id}`` is a label that's
    knowable the moment the blueprint exists -- before any section is drafted.
    """
    prefix = _LABEL_PREFIX_BY_KIND.get(kind)
    return f"{prefix}:{block_id}" if prefix else None


class LabelRegistryEntry(BaseModel):
    """One citable label and what it refers to.

    Built once, eagerly, in ``DocumentGraph.set_blueprint`` -- every id it's
    derived from (section_id, block_id, figure_id) is already fixed by the
    blueprint, so the full set of valid labels is knowable before any section
    is drafted. Later sections are handed this registry (see
    ``TextAgent._build_prompt_from_node``) so a cross-reference cites
    something that actually exists, instead of guessing at a label name a
    different section's independent model call happened to pick -- the
    `thm:taylor_convergence` incident this was added for was exactly that: a
    label no section ever actually defined.
    """

    label: str
    kind: str  # "section", a LayoutBlockKind value, or "figure"
    section_id: str
    block_id: Optional[str] = None
    description: str


class DocumentNode(BaseModel):
    """An individual semantic content node in a document graph."""

    id: str = Field(..., description="Unique identifier, for example 'sec_1' or 'fig_1'.")
    type: NodeType = Field(..., description="The type of the node.")
    title: str = Field(..., description="Section title or figure caption.")
    content: str = Field("", description="Generated semantic or LaTeX content.")
    status: NodeStatus = Field(NodeStatus.PENDING, description="Current workflow status.")
    spec: Dict[str, Any] = Field(default_factory=dict, description="Planner instructions for the node.")
    assets: List[str] = Field(default_factory=list, description="Paths to linked assets.")
    dependencies: List[str] = Field(default_factory=list)
    layout_block_ids: List[str] = Field(default_factory=list)
    attempt_count: int = 0
    last_error: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.now)

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        return _validate_safe_identifier(value)


class DocumentGraph(BaseModel):
    """Central state shared by planner, specialist agents, and LaTeX composer."""

    nodes: Dict[str, DocumentNode] = Field(default_factory=dict)
    hierarchy: Dict[str, List[str]] = Field(default_factory=dict)
    references: Dict[str, List[str]] = Field(default_factory=dict)
    blueprint: Optional[DocumentBlueprint] = None
    figure_registry: Dict[str, FigureRegistryEntry] = Field(default_factory=dict)
    label_registry: Dict[str, LabelRegistryEntry] = Field(default_factory=dict)

    def add_node(self, node: DocumentNode) -> None:
        self.nodes[node.id] = node
        self.hierarchy.setdefault(node.id, [])

    def add_edge(self, from_id: str, to_id: str, relation_type: str = "hierarchy") -> None:
        if relation_type == "hierarchy":
            if from_id in self.nodes and to_id in self.nodes:
                children = self.hierarchy.setdefault(from_id, [])
                if to_id not in children:
                    children.append(to_id)
        elif relation_type == "reference":
            targets = self.references.setdefault(from_id, [])
            if to_id not in targets:
                targets.append(to_id)

    def update_node_status(
        self,
        node_id: str,
        status: NodeStatus,
        content: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        if status == NodeStatus.RUNNING:
            node.attempt_count += 1
        node.status = status
        if content is not None:
            node.content = content
        if error is not None:
            node.last_error = error
        elif status != NodeStatus.ERROR:
            node.last_error = None
        node.last_updated = datetime.now()

    def set_blueprint(self, blueprint: DocumentBlueprint) -> None:
        """Attach a layout plan, reserve each figure's single render slot, and
        (re)build the label registry from it."""
        self.blueprint = blueprint
        for figure_id, placement in blueprint.figure_slots.items():
            entry = self.figure_registry.get(figure_id)
            if entry is None:
                self.figure_registry[figure_id] = FigureRegistryEntry(
                    figure_id=figure_id,
                    owner_section=placement.owner_section,
                    label=placement.label or f"fig:{figure_id}",
                    caption=placement.caption,
                    required=placement.required,
                )
                continue

            if entry.placement_count:
                raise ValueError(
                    f"Cannot replace the placement of figure '{figure_id}' after it was rendered."
                )
            entry.owner_section = placement.owner_section
            entry.label = placement.label or entry.label or f"fig:{figure_id}"
            entry.caption = placement.caption or entry.caption
            entry.required = placement.required
            entry.last_updated = datetime.now()

        # Entirely derived from the blueprint (section/block/figure ids never
        # change after planning), so a full rebuild on every call is correct
        # -- unlike figure_registry above, there's no accumulated rendering
        # state (placement_count, asset_path, ...) to preserve across a
        # re-compose.
        self.label_registry = {}
        for section in blueprint.sections:
            section_label = f"sec:{section.section_id}"
            self.label_registry[section_label] = LabelRegistryEntry(
                label=section_label,
                kind="section",
                section_id=section.section_id,
                description=section.title,
            )
            for block in section.blocks:
                label = default_block_label(block.kind, block.id)
                if label is None:
                    continue
                self.label_registry[label] = LabelRegistryEntry(
                    label=label,
                    kind=block.kind.value,
                    section_id=section.section_id,
                    block_id=block.id,
                    description=block.title or block.instructions[:80],
                )
        for figure_id, placement in blueprint.figure_slots.items():
            label = placement.label or f"fig:{figure_id}"
            self.label_registry[label] = LabelRegistryEntry(
                label=label,
                kind="figure",
                section_id=placement.owner_section,
                block_id=figure_id,
                description=placement.caption or figure_id,
            )

    def register_figure_asset(
        self,
        figure_id: str,
        asset_path: str,
        *,
        caption: Optional[str] = None,
    ) -> FigureRegistryEntry:
        """Associate a generated image file with a reserved figure identifier."""
        entry = self.figure_registry.get(figure_id)
        if entry is None:
            entry = FigureRegistryEntry(
                figure_id=figure_id,
                label=f"fig:{figure_id}",
            )
            self.figure_registry[figure_id] = entry

        if entry.asset_path and entry.asset_path != asset_path:
            raise ValueError(
                f"Figure '{figure_id}' is already registered with a different asset path."
            )
        entry.asset_path = asset_path
        if caption:
            entry.caption = caption
        entry.last_updated = datetime.now()
        return entry

    def claim_figure_placement(self, figure_id: str, section_id: str) -> FigureRegistryEntry:
        """Record the sole final-document placement of a generated figure.

        The composer must call this immediately before emitting the figure's
        ``\\includegraphics``. Calling it twice is an error by design.
        """
        entry = self.figure_registry.get(figure_id)
        if entry is None:
            raise ValueError(f"Figure '{figure_id}' is not registered.")
        if not entry.asset_path:
            raise ValueError(f"Figure '{figure_id}' has no generated asset path.")
        if entry.owner_section and entry.owner_section != section_id:
            raise ValueError(
                f"Figure '{figure_id}' belongs in '{entry.owner_section}', not '{section_id}'."
            )
        if entry.placement_count >= 1:
            raise ValueError(f"Figure '{figure_id}' may appear only once in the document.")

        entry.placement_count = 1
        entry.rendered_in_section = section_id
        entry.last_updated = datetime.now()
        return entry

    def reset_figure_placements(self) -> None:
        """Prepare the registry for a complete re-compose of the document."""
        for entry in self.figure_registry.values():
            entry.placement_count = 0
            entry.rendered_in_section = None
            entry.last_updated = datetime.now()

    def validate_figure_usage(self) -> List[str]:
        """Return registry violations for the pre-compile validation tool."""
        errors: List[str] = []
        for figure_id, entry in self.figure_registry.items():
            if entry.placement_count > 1:
                errors.append(f"Figure '{figure_id}' was rendered more than once.")
            if entry.required and not entry.asset_path:
                errors.append(f"Required figure '{figure_id}' has no generated asset.")
            if entry.required and entry.asset_path and entry.placement_count != 1:
                errors.append(f"Required figure '{figure_id}' must be rendered exactly once.")
        return errors

    def get_full_context(self, node_id: str) -> str:
        """Return concise completed parent and sibling context for a node."""
        node = self.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node '{node_id}' not found.")

        parent_id = next(
            (parent for parent, children in self.hierarchy.items() if node_id in children),
            None,
        )
        context_nodes: List[DocumentNode] = []
        if parent_id and parent_id in self.nodes:
            context_nodes.append(self.nodes[parent_id])
            context_nodes.extend(
                sibling
                for sibling_id in self.hierarchy.get(parent_id, [])
                if sibling_id != node_id
                and (sibling := self.nodes.get(sibling_id)) is not None
                and sibling.status in {NodeStatus.DRAFTED, NodeStatus.REVIEWED, NodeStatus.VALIDATED}
            )

        parts = []
        for context_node in context_nodes:
            content = context_node.content.strip()
            if content:
                parts.append(f"[{context_node.id}: {context_node.title}]\n{content}")
        return "\n\n".join(parts)

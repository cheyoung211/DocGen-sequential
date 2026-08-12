from __future__ import annotations
import re
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

from src.common.state import DocumentBlueprint

# Some LLM responses double-escape backslashes when they emit a JSON string
# whose value itself contains a markdown/LaTeX newline, so json.loads() leaves
# behind the two literal characters '\' + 'n' (or '\' + 't') instead of an
# actual newline/tab. TeX then reads that literal text as an undefined control
# sequence and halts compilation.
#
# A first version of this fix guarded against corrupting a real command like
# \newpage by refusing to touch \n/\t when a letter followed. That guard was
# backwards in practice: the artifact is a markdown line break, so the letter
# right after it is almost always just the first letter of the next line
# ("\nBuilds speed...", "\nPreserves reserve...") -- the common case got
# skipped, and only the rare "\n followed by punctuation" case was fixed.
#
# TeX control words are a backslash plus the *maximal* run of letters that
# follows, so "\nBuilds" is read as one undefined command "\nBuilds", not
# "\n" + "Builds". There is no way to tell, from the decoded string alone,
# whether "\n" + letters was meant as an escaped newline followed by prose or
# as a genuine multi-letter command -- so instead of guessing from context,
# only the finite set of real LaTeX/amsmath commands this pipeline's blocks
# are actually allowed to contain (see text_agent.SYSTEM_PROMPT and
# latex_assembler.FORBIDDEN_LATEX_COMMANDS) is treated as "real"; anything
# else starting with \n or \t is the escape artifact and gets unescaped, with
# the trailing letters preserved as the prose that follows.
_REAL_LATEX_COMMANDS_BY_LEADING_LETTER = {
    "n": {
        "nabla", "ne", "neq", "newpage", "newline", "noindent", "nonumber",
        "notag", "nolimits", "negthinspace", "negmedspace", "negthickspace",
    },
    "t": {
        "times", "theta", "tiny", "today", "toprule", "tabularx", "title",
        "tableofcontents", "listoftables", "thanks", "thispagestyle",
        "thesection", "thechapter", "thepage", "thefootnote", "textbf",
        "textit", "textsc", "textsf", "texttt", "textrm", "textsl", "textup",
        "textmd", "textnormal", "textcolor", "textwidth", "textheight",
        "textbackslash", "textasciicircum", "textasciitilde",
        "textsuperscript", "textemdash", "textendash", "textquotedblleft",
        "textquotedblright",
        # "to" (\lim_{n \to \infty}) and "text" (bare \text{...} from amsmath,
        # e.g. \text{Bias}) were missing here and were observed being mangled
        # into a literal tab character in generated math-domain content --
        # both are extremely common in the LaTeX benchmark's math/statistics
        # items (limits, bias-variance decompositions, etc.), unlike the
        # already-whitelisted text* variants (textbf, textit, ...), which are
        # compound commands that happen not to collide with this artifact.
        "to", "text",
    },
}
_ESCAPE_ARTIFACT_PATTERN = re.compile(r"\\([nt])([A-Za-z]*)")
_ESCAPE_ARTIFACT_REPLACEMENTS = {"n": "\n", "t": "\t"}


def unescape_literal_newlines_and_tabs(text: str) -> str:
    """Turn a double-escaped literal '\\n'/'\\t' back into a real newline/tab,
    leaving whitelisted real commands (\\newpage, \\textbf, ...) untouched."""

    def _replace(match: "re.Match[str]") -> str:
        leading_letter, rest = match.group(1), match.group(2)
        if leading_letter + rest in _REAL_LATEX_COMMANDS_BY_LEADING_LETTER[leading_letter]:
            return match.group(0)
        return _ESCAPE_ARTIFACT_REPLACEMENTS[leading_letter] + rest

    return _ESCAPE_ARTIFACT_PATTERN.sub(_replace, text)

# ---------- 1. 기초 및 설정 관련 ----------

class FileInput(BaseModel):
    name: str
    path: str
    file_type: str = Field("csv", description="csv, pdf, txt 등 입력 파일 형식")

class TemplateSpec(BaseModel):
    name: str
    main_tex_template: str
    cls_path: Optional[str] = None  # 비즈니스용 커스텀 클래스 지원 
    section_order: Optional[List[str]] = None
    options: Dict[str, Any] = Field(default_factory=dict)

# --- Enums for State Management ---

class NodeStatus(str, Enum):
    """Status of a document node in the workflow."""
    PENDING = "PENDING"      # Waiting to be drafted
    DRAFTED = "DRAFTED"      # Content generated
    REVIEWED = "REVIEWED"    # Verified by a reviewer
    ERROR = "ERROR"          # Refinement needed due to compilation/logic error

class NodeType(str, Enum):
    """Type of content within the document graph."""
    SECTION = "SECTION"
    FIGURE = "FIGURE"
    TABLE = "TABLE"
    APPENDIX = "APPENDIX" 

    # --- Graph Structure Specifications ---

class NodeSpec(BaseModel):
    """Blueprint for creating a DocumentNode."""
    id: str = Field(..., description="Unique identifier for the node (e.g., 'sec_1', 'fig_1')")
    type: NodeType = Field(..., description="The type of the node [SECTION, FIGURE, TABLE]")
    spec: str = Field(..., description="Detailed instructions for the agent (e.g., 'Analyze market share using Pandas')")
    contextual_role: str = Field(..., description="The strategic role (e.g., 'Executive Summary', 'Data Visualization')")
    required_tools: List[str] = Field(default_factory=list, description="List of tools the agent should use (e.g., ['pandas_tool'])")

class EdgeSpec(BaseModel):
    """Defines a relationship between two nodes."""
    from_id: str = Field(..., description="Source node ID")
    to_id: str = Field(..., description="Destination node ID")
    relation_type: str = Field("hierarchy", description="Type of edge: 'hierarchy' (parent-child) or 'reference'")

# ---------- 2. 도구(Tool) 실행 관련 (신설) ----------

class ToolRequest(BaseModel):
    """에이전트가 특정 도구 사용을 요청할 때 사용"""
    tool_name: str
    arguments: Dict[str, Any]
    reasoning: str = Field(..., description="이 도구를 왜 사용하는지에 대한 에이전트의 논리")

class ToolResponse(BaseModel):
    """도구 실행 결과 및 관찰 내용"""
    success: bool
    output: Any  # 데이터프레임 요약, 이미지 경로, 또는 에러 메시지
    observation: str
    error: Optional[str] = None

# ---------- 3. LaTeX 자산(Asset) 관리 (신설) ----------

class AssetType(str, Enum):
    TEX_SNIPPET = "tex"
    IMAGE = "image"
    TABLE_CSV = "csv"
    BIB = "bib"

class LaTeXAsset(BaseModel):
    """생성된 모든 물리적 파일 자산을 추적"""
    asset_id: str
    file_path: str
    type: AssetType
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ---------- 4. Planner 및 에이전트 입출력 고도화 ----------

class PlannerInput(BaseModel):
    """사용자의 요청을 PlannerAgent에 전달할 때 사용"""
    request_id: str
    language: str = "en"
    doc_type: str = "internal_report"
    user_query: str
    inputs: List[FileInput] = Field(default_factory=list)
    template: TemplateSpec
    allow_figures: bool = Field(
        True,
        description="If False, no image-generation model is available for this run -- "
        "the planner must not create any FIGURE node, asset_node, or figure_slots entry.",
    )

class SectionSpec(BaseModel):
    """state.py의 DocumentNode 내 spec 필드에 들어갈 상세 지침"""
    title: str
    contextual_role: str = Field(..., description="문서 내에서의 논리적 역할 (예: 배경 설명, 결과 분석)")
    key_points: List[str] = Field(default_factory=list, description="반드시 포함되어야 할 핵심 내용")
    required_tools: List[str] = Field(default_factory=list, description="사용 권장 도구 (Pandas, Search 등)")
    style_guide: Dict[str, Any] = Field(default_factory=dict)

# class PlannerOutput(BaseModel):
#     """사용자 쿼리를 해석하여 초기 그래프 구조를 제안"""
#     request_id: str
#     # 노드 정의: {node_id: DocumentNode 초기 데이터}
#     # nodes_to_create: List[Dict[str, Any]] 
#     # 관계 정의: [ (parent_id, child_id), ... ]
#     # hierarchy_edges: List[tuple[str, str]]
#     # global_context: str = Field(..., description="문서 전체의 톤앤매너 및 전제 조건")
#     nodes_to_create: List[NodeSpec] = Field(..., description="List of nodes to initialize the DocumentGraph")
#     hierarchy_edges: List[EdgeSpec] = Field(..., description="List of edges defining the document structure")
#     global_context: str = Field(..., description="Global style guide and facts for all agents to follow")
class PlannerOutput(BaseModel):
    # Make request_id optional so it doesn't crash if LLM forgets it
    request_id: Optional[str] = None
    nodes_to_create: List[Dict[str, Any]]
    # Accept both List of Lists AND List of Dicts for edges
    hierarchy_edges: List[Union[List[str], Dict[str, Any]]]
    global_context: str
    blueprint: DocumentBlueprint

    @validator('nodes_to_create', pre=True)
    def ensure_node_titles(cls, v):
        """Ensures every node has a 'title' field, using the ID as a fallback."""
        if not isinstance(v, list):
            return v
        for node in v:
            if isinstance(node, dict) and 'title' not in node:
                # Fallback: Convert 'sec_1' to 'Sec 1'
                node_id = node.get('id', 'untitled')
                node['title'] = node_id.replace('_', ' ').title()
        return v

    @validator('hierarchy_edges', pre=True)
    def normalize_edges(cls, v):
        """Standardizes edges to [parent, child] format regardless of LLM output style."""
        normalized = []
        for edge in v:
            if isinstance(edge, dict):
                parent = edge.get("from_id") or edge.get("parent")
                child = edge.get("to_id") or edge.get("child")
                if parent and child:
                    normalized.append([parent, child])
            else:
                normalized.append(edge)
        return normalized

# ---------- 5. Text & Content 관련 ----------

class SemanticBlockType(str, Enum):
    """Content semantics consumed by the blueprint-driven LaTeX composer."""
    PARAGRAPH = "paragraph"
    KEY_TAKEAWAY = "key_takeaway"
    DEFINITION = "definition"
    WARNING = "warning"
    CASE_STUDY = "case_study"
    COMPARISON = "comparison"
    TABLE = "table"
    EQUATION = "equation"
    LIST = "list"
    FIGURE_REFERENCE = "figure_reference"
    THEOREM = "theorem"
    PROOF_SKETCH = "proof_sketch"
    NOTATION_TABLE = "notation_table"


class ContentBlock(BaseModel):
    """One semantic content unit matched to a planned layout block."""
    block_id: Optional[str] = Field(None, description="DocumentBlueprint LayoutBlock ID")
    type: SemanticBlockType
    title: Optional[str] = None
    content: str = Field(..., alias="text")
    asset_id: Optional[str] = None
    caption: Optional[str] = None
    assets: List[str] = Field(default_factory=list, description="Related LaTeX asset IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        # Accept both the legacy `text` field and the explicit `content` field.
        populate_by_name = True

    @validator("content", pre=True)
    def normalize_escaped_whitespace(cls, v):
        """Repair double-escaped '\\n'/'\\t' before any downstream renderer
        splits this content on real newlines (see latex_assembler._render_list
        and _markdown_table_to_tabularx, which rely on actual line breaks)."""
        if not isinstance(v, str):
            return v
        return unescape_literal_newlines_and_tabs(v)

class TextAgentOutput(BaseModel):
    request_id: str
    section_id: str
    blocks: List[ContentBlock]
    used_tools: List[ToolResponse] = Field(default_factory=list)
    self_critique: Optional[str] = Field(None, description="작성 내용에 대한 에이전트 스스로의 품질 평가")

"""Blueprint-driven LaTeX document composer and static assembler."""

from __future__ import annotations

import difflib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.common.schemas import ContentBlock, SemanticBlockType, TextAgentOutput

from src.common.state import (
    DocumentBlueprint,
    DocumentGraph,
    DocumentKind,
    FigureLayout,
    FigurePlacement,
    LatexComponent,
    LayoutBlock,
    LayoutBlockKind,
    SectionLayout,
    default_block_label,
)
from src.evaluation.event_logger import EventLogger


SAFE_PACKAGES = {
    "geometry",
    "graphicx",
    "booktabs",
    "tabularx",
    "longtable",
    "tcolorbox",
    "caption",
    "hyperref",
    "cleveref",
    "placeins",
    "enumitem",
    "siunitx",
    "amsmath",
    "amssymb",
    "amsthm",
    "array",
    "xcolor",
    "etoolbox",
}
PACKAGE_ORDER = (
    "geometry",
    "graphicx",
    "xcolor",
    "amsmath",
    "amssymb",
    "amsthm",
    "array",
    "booktabs",
    "tabularx",
    "longtable",
    "tcolorbox",
    "caption",
    "enumitem",
    "siunitx",
    "hyperref",
    "cleveref",
    "placeins",
    "etoolbox",
)
SAFE_DOCUMENT_CLASSES = {"article", "report", "scrartcl", "scrreprt"}
# report/scrreprt define \chapter to force a page break before it starts
# (\if@openright\cleardoublepage\else\clearpage\fi). Neither article/scrartcl
# nor \section/\subsection/\subsubsection have this behavior.
CHAPTER_FORCES_PAGE_BREAK_CLASSES = {"report", "scrreprt"}
# Where that literal page-break snippet actually lives differs by class
# family. Stock report (classes.dtx) embeds it directly in \chapter's own
# macro body. KOMA-Script's scrreprt (scrkernel-sections.dtx) instead defines
# \chapter as nothing but a call to a shared internal helper,
# \scr@startchapter{chapter}, and the page-break snippet lives inside that
# helper instead. Patching \chapter itself is therefore a silent no-op for
# scrreprt -- \patchcmd never finds the text it's searching for there, and
# with empty success/failure branches it fails without any warning -- so each
# class family needs its own patch target.
CHAPTER_PAGEBREAK_PATCH_TARGET = {
    "report": r"\chapter",
    "scrreprt": r"\scr@startchapter",
}

# This mapping belongs in the composer as well as in TextAgent.  JSON files are
# artifacts that can outlive an agent process, so the composer must validate the
# contract itself instead of trusting that TextAgent happened to validate it.
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

# Blocks may contain only their semantic content.  Document framing, section
# headings, figure placement, and file inclusion stay under composer control.
FORBIDDEN_LATEX_COMMANDS = re.compile(
    r"\\(?:documentclass|usepackage|RequirePackage|input|include|"
    r"includegraphics|openin|openout|read|write|immediate|catcode|"
    r"csname|def|gdef|edef|xdef|newcommand|renewcommand|newenvironment|"
    r"section|subsection|subsubsection|chapter|part)\b",
    re.IGNORECASE,
)
FORBIDDEN_LAYOUT_ENVIRONMENTS = {
    "document",
    "figure",
    "wrapfigure",
    "tcolorbox",
    "minipage",
}
# Pure matrix notation (no numbering, no cross-reference target of its own --
# it only ever appears nested inside a $...$/equation body) is allowed in
# every block type's content, not just EQUATION. Splitting these out of
# EQUATION's set is what lets a CASE_STUDY/PROOF_SKETCH-style prose block
# write out an eigenvector calculation as \begin{pmatrix}...\end{pmatrix}
# directly, instead of forbidden_environment forcing a false choice between
# "no matrix at all" and "wrap it in a whole separate numbered equation
# block" -- see the worked_examples incident this was pulled out to fix
# (pmatrix in CASE_STUDY/PROOF_SKETCH hit forbidden_environment every retry,
# exhausting the section's 3 attempts and failing the whole pipeline run).
# Numbered/labeled *display* environments (equation, align, gather, ...) stay
# EQUATION-only: those need the composer's own label bookkeeping, and are
# exactly what REMEDIATION_HINTS["forbidden_environment"] tells a prose block
# to use a dedicated `equation` block for instead.
MATRIX_ENVIRONMENTS = {
    "matrix",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "Vmatrix",
    "smallmatrix",
}
ALLOWED_ENVIRONMENTS = {
    SemanticBlockType.LIST: {"itemize", "enumerate", "description"},
    SemanticBlockType.COMPARISON: {"tabular", "tabularx", "array"},
    SemanticBlockType.TABLE: {"table", "tabular", "tabularx", "longtable", "array", "center"},
    SemanticBlockType.NOTATION_TABLE: {"table", "tabular", "tabularx", "longtable", "array", "center"},
    SemanticBlockType.EQUATION: {
        "equation",
        "equation*",
        "align",
        "align*",
        "aligned",
        "gather",
        "gather*",
        "multline",
        "multline*",
        "split",
        "cases",
    }
    | MATRIX_ENVIRONMENTS,
}
ENVIRONMENT_TOKEN = re.compile(r"\\(?P<action>begin|end)\s*\{(?P<name>[A-Za-z*]+)\}")


def validate_fragment_environments(block_type: SemanticBlockType, content: str, block_id: Optional[str]) -> str:
    """Reject a LaTeX command/environment a block's ``type`` isn't allowed to use.

    Pulled out of ``LatexAssembler._validate_fragment`` so TextAgent can
    run the exact same check right after generating a block -- instead of
    only discovering the violation here, in the composer, after the content
    is already persisted and the per-node retry loop has moved on. Only the
    part of ``_validate_fragment`` that doesn't need the composer's
    document-wide label registry lives here; label-uniqueness checking stays
    on the instance method.

    Returns the normalized (``\\r\\n`` -> ``\\n``) content so callers that
    still need to scan it (for labels, say) don't have to normalize twice.
    """
    normalized = content.replace("\r\n", "\n")
    if "\x00" in normalized:
        raise ValueError(f"Block '{block_id}' contains a null byte.")
    if FORBIDDEN_LATEX_COMMANDS.search(normalized):
        raise ValueError(
            f"Block '{block_id}' contains a document-level or unsafe LaTeX command."
        )

    # Matrix notation is allowed in every block type's content (see
    # MATRIX_ENVIRONMENTS), not just whatever ALLOWED_ENVIRONMENTS lists for
    # this specific block_type -- a type with no entry there at all (e.g.
    # CASE_STUDY, PROOF_SKETCH) still gets `set()` from .get() and must not
    # lose the matrix allowance to that default.
    allowed = ALLOWED_ENVIRONMENTS.get(block_type, set()) | MATRIX_ENVIRONMENTS
    stack: List[str] = []
    for token in ENVIRONMENT_TOKEN.finditer(normalized):
        name = token.group("name")
        if name in FORBIDDEN_LAYOUT_ENVIRONMENTS or name not in allowed:
            raise ValueError(
                f"Block '{block_id}' may not use LaTeX environment '{name}'."
            )
        if token.group("action") == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            raise ValueError(
                f"Block '{block_id}' has an unbalanced LaTeX environment '{name}'."
            )
    if stack:
        raise ValueError(
            f"Block '{block_id}' has unclosed LaTeX environment(s): {', '.join(stack)}."
        )
    return normalized


_TITLE_UNSAFE_CHAR = re.compile(r"\\")


def _reject_plain_text_markup(value: Optional[str], source: str, field_label: str) -> None:
    """Shared rule for a field that is NEVER passed through LaTeX rendering
    at all -- only ``LatexAssembler._escape_latex``, a blunt, command-blind
    escaper that assumes plain text. A backslash command survives that as
    mangled literal text (see ``validate_plain_title``'s own history); a
    Markdown emphasis marker (``**bold**``/``__bold__``) survives it even
    worse, since ``_escape_latex`` has no notion of Markdown at all and
    prints the literal asterisks/underscores with no error. Both must be
    caught here, before rendering, for the same reason.

    Round-2 review finding: ``find_markdown_emphasis_matches`` was called
    with its default math-exclusion mask on, which treats a bare ``$`` as
    the start of a math region to protect. A title/caption field is never
    rendered through real LaTeX math at all -- a bare ``$`` here is just a
    literal dollar sign that ``_escape_latex`` prints correctly (see
    ``validate_plain_title``'s own docstring) -- so an unterminated
    ``$...`` (e.g. "Cost $5 **discount**") wrongly "protected" every
    character after it, hiding a real violation. The backslash check above
    already guarantees ``value`` has no backslash by the time this call is
    reached, so there is nothing left for the math mask to legitimately
    protect anyway (no ``\\(``, ``\\[``, ``\\verb``, ``\\texttt{}`` can
    appear); passing ``apply_literal_math_mask=False`` makes that explicit.
    """
    if not value:
        return
    if _TITLE_UNSAFE_CHAR.search(value):
        raise ValueError(
            f"{source} {field_label} {value!r} contains a LaTeX command (backslash), which "
            f"renders as literal text, not a LaTeX command -- {field_label} must be plain text."
        )
    candidates = find_markdown_emphasis_matches(
        value.replace("\r\n", "\n"), apply_literal_math_mask=False
    )
    if candidates:
        sample = candidates[0]
        raise ValueError(
            f"{source} {field_label} {value!r} contains Markdown emphasis syntax "
            f"('{sample.marker}{sample.inner}{sample.marker}'), which renders as literal "
            f"asterisks/underscores (not bold), not a LaTeX command either -- {field_label} "
            "must be plain text with no emphasis markup at all."
        )


def validate_plain_title(title: Optional[str], source: str) -> None:
    """Reject a LaTeX command (backslash) or Markdown emphasis syntax in a
    title-like field.

    Titles (document/section/block titles) are rendered through
    ``LatexAssembler._escape_latex`` -- a blunt, command-blind escaper that
    assumes the string is plain text and turns every backslash into a
    literal, visible "\\textbackslash{}". A title that legitimately needs
    math notation (``\\pi``, ``\\theta``, ...) has no way to survive that;
    the fix is to catch it here, before rendering, rather than let it print
    as mangled text with no error at all. The same is true of Markdown
    emphasis (``**bold**``): there is no LaTeX rendering pass here to turn
    it into bold, so it would print as literal asterisks instead.

    A bare ``$`` (e.g. a dollar figure in a finance-domain title like "$500
    Cost Overrun") is deliberately not flagged here: ``_escape_latex``
    already turns it into a literal, correctly-rendered dollar sign, so it
    isn't actually broken -- only a backslash command or emphasis markup is.
    """
    _reject_plain_text_markup(title, source, "title")


def validate_plain_caption(caption: Optional[str], source: str) -> None:
    """Same rule as ``validate_plain_title``, for a block's ``caption``
    field -- also rendered through ``_escape_latex`` (see
    ``LatexAssembler._render_table``/``_render_figure``), never through
    actual LaTeX, so it must be equally plain.
    """
    _reject_plain_text_markup(caption, source, "caption")


LABEL_TOKEN = re.compile(r"\\label\s*\{(?P<label>[^{}]+)\}")
VALID_LABEL = re.compile(r"[A-Za-z0-9:._-]+")
VALID_GRAPHICS_PATH = re.compile(r"[A-Za-z0-9_./-]+")

# The one sanctioned way a block's content may already contain a hand-placed
# label: a TABLE block pasting in the complete environment
# pandas_csv_to_latex_table returned (see text_agent.SYSTEM_PROMPT), which
# necessarily carries the caller-supplied `label` argument since that tool
# renders its own table wrapper the composer never touches.
_TOOL_RENDERED_TABLE_ENV = re.compile(r"\\begin\s*\{(?:table|longtable)\}")


def validate_no_manual_labels(block_type: SemanticBlockType, content: str, block_id: Optional[str]) -> None:
    """Reject a block that writes its own ``\\label{}`` instead of citing one.

    Labels for theorem/definition/equation/table/notation_table/section are
    assigned automatically by the composer -- text_agent.SYSTEM_PROMPT tells
    the model this explicitly -- but nothing used to enforce it, so a
    violation only surfaced (if at all) as a "Duplicate LaTeX label" crash in
    the composer, with no retry, whenever two independently-drafted sections
    happened to invent the same label string for what each considered the
    "same" equation (the ``eq:energy_balance`` incident: two different
    sections both wrote ``\\label{eq:energy_balance}`` for their own copy of
    the same physical relation). Running this right after generation (see
    ``validate_fragment_environments`` for the same reasoning) turns that
    into a retryable error instead, and also catches the many cases that
    never collide but still silently diverge from the label the
    cross-reference registry already promised other sections.
    """
    if block_type == SemanticBlockType.TABLE and _TOOL_RENDERED_TABLE_ENV.search(content):
        return
    if LABEL_TOKEN.search(content):
        raise ValueError(
            f"Block '{block_id}' must not write its own \\label{{}} -- labels are "
            "assigned automatically by the composer. Cite an existing one from "
            "\"Available cross-reference labels\" instead of declaring a new one."
        )


def _find_math_spans(row: str) -> List["tuple[int, int]"]:
    """Return (start, end) index ranges for each top-level math region in
    ``row`` -- ``\\(...\\)``, ``\\[...\\]``, ``$...$`` -- tracking a ``$$``
    run as one toggle, not two, the same as ``_sanitize_prose``/
    ``_validate_bare_math_commands``."""
    spans: List["tuple[int, int]"] = []
    length = len(row)
    index = 0
    in_math = False
    span_start = 0
    while index < length:
        char = row[index]
        if char == "\\" and index + 1 < length:
            two_char = row[index : index + 2]
            if two_char in (r"\(", r"\[") and not in_math:
                in_math = True
                span_start = index
                index += 2
                continue
            if two_char in (r"\)", r"\]") and in_math:
                spans.append((span_start, index + 2))
                in_math = False
                index += 2
                continue
            index += 2
            continue
        if char == "$":
            if index + 1 < length and row[index + 1] == "$":
                if not in_math:
                    in_math = True
                    span_start = index
                else:
                    spans.append((span_start, index + 2))
                    in_math = False
                index += 2
                continue
            if not in_math:
                in_math = True
                span_start = index
            else:
                spans.append((span_start, index + 1))
                in_math = False
            index += 1
            continue
        index += 1
    if in_math:
        spans.append((span_start, length))
    return spans


@dataclass
class MarkdownEmphasisMatch:
    """One ``**bold**``/``__bold__``-shaped span found in one line of block
    content. ``line_number`` is 1-indexed; ``start``/``end`` are indices
    within that one line (not the whole, possibly multi-line, content) --
    ``end`` is just past the closing marker. ``ambiguous`` spans must not be
    mechanically converted (see ``find_markdown_emphasis_matches``); ``reason``
    is only set when ``ambiguous`` is true.
    """

    line_number: int
    marker: str
    start: int
    end: int
    inner: str
    ambiguous: bool
    reason: str = ""


_EMPHASIS_MARKERS = ("**", "__")
_MARKER_RUN_CHARS = ("*", "_")
_VERB_COMMAND = "\\verb"
_TEXTT_COMMAND_OPEN = "\\texttt{"
# A backslash-escaped occurrence of one of these characters renders as that
# literal character, not as the syntax it would otherwise trigger --
# `\$` is a literal dollar sign, never the start of math; `\{`/`\}` are
# literal braces, never a LaTeX group delimiter; `\\` is a literal
# backslash. Round-2 review finding: `_literal_or_math_mask` used to treat
# every bare `$`/`{`/`}` as live syntax regardless of a preceding escaping
# backslash, so "The price is \\$5 and **important**." mistook `\$` for
# the start of unterminated math and excluded everything after it, and
# "\\texttt{\\{} then **bold**." mistook the escaped `\{` for a second,
# unmatched group-open inside the \texttt{} argument and excluded the rest
# of the line along with it.
_ESCAPED_LITERAL_CHARS = ("$", "{", "}", "\\")


def _is_escaped_literal(text: str, index: int, length: int) -> bool:
    return text[index] == "\\" and index + 1 < length and text[index + 1] in _ESCAPED_LITERAL_CHARS


def _literal_or_math_mask(text: str) -> List[bool]:
    """``True`` for every index in ``text`` that must never be scanned for
    Markdown emphasis -- three kinds of "this is not prose", each wrong to
    touch for a different reason:

    - ``$...$``/``\\(...\\)``/``\\[...\\]`` math, tracked with ONE piece of
      state across the *entire* string, never reset per line -- a
      multi-line ``\\[ ... \\]`` must stay excluded on every line it spans
      (the earlier per-line ``_find_math_spans`` reuse was wrong for
      exactly this: that helper's own contract is "one Markdown table
      row," not "math state carried across a whole multi-line fragment,"
      which is what ``LatexAssembler._sanitize_prose`` actually does for
      the same reason).
    - ``\\verb<delim>...<delim>`` -- LaTeX's literal/verbatim command; its
      argument renders byte-for-byte, so an emphasis marker inside it is
      the literal text being quoted, not emphasis.
    - A ``\\texttt{...}`` argument -- this project's own convention for
      code/identifier-like text, so a doubled-underscore run inside it
      (``\\texttt{__init__}``) is the author explicitly marking it
      literal/monospace, not writing Markdown bold.

    A backslash-escaped ``$``/``{``/``}``/``\\`` (see
    ``_is_escaped_literal``) is consumed as a literal pair everywhere
    below, before it ever reaches the ``$`` toggle or the ``\\texttt{}``
    brace-depth counter -- round-2 review fix (see ``_ESCAPED_LITERAL_CHARS``).
    """
    length = len(text)
    mask = [False] * length
    in_math = False
    index = 0
    while index < length:
        char = text[index]
        if char == "\\" and not in_math and text.startswith(_VERB_COMMAND, index):
            delim_index = index + len(_VERB_COMMAND)
            if delim_index < length and not text[delim_index].isalpha():
                close = text.find(text[delim_index], delim_index + 1)
                end = (close + 1) if close != -1 else length
                for i in range(index, end):
                    mask[i] = True
                index = end
                continue
        if char == "\\" and not in_math and text.startswith(_TEXTT_COMMAND_OPEN, index):
            depth = 1
            i = index + len(_TEXTT_COMMAND_OPEN)
            while i < length and depth > 0:
                if _is_escaped_literal(text, i, length):
                    i += 2
                    continue
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            for j in range(index, i):
                mask[j] = True
            index = i
            continue
        if char == "\\" and index + 1 < length and text[index : index + 2] in (r"\(", r"\["):
            mask[index] = mask[index + 1] = True
            in_math = True
            index += 2
            continue
        if char == "\\" and index + 1 < length and text[index : index + 2] in (r"\)", r"\]"):
            mask[index] = mask[index + 1] = True
            in_math = False
            index += 2
            continue
        if _is_escaped_literal(text, index, length):
            if in_math:
                mask[index] = mask[index + 1] = True
            index += 2
            continue
        if char == "$":
            if index + 1 < length and text[index + 1] == "$":
                mask[index] = mask[index + 1] = True
                in_math = not in_math
                index += 2
                continue
            mask[index] = True
            in_math = not in_math
            index += 1
            continue
        if in_math:
            mask[index] = True
        index += 1
    return mask


def _has_balanced_braces(inner: str) -> bool:
    """Whether wrapping ``inner`` in ``\\textbf{...}`` keeps the enclosing
    LaTeX group structure well-formed.

    Round-2 review repro: "**text { brace**" has inner text "text { brace"
    -- one naked, unescaped '{' with no matching '}'. Converting it to
    "\\textbf{text { brace}" leaves two opens and one close, a brace
    imbalance the composer only catches later (as a confusing "unclosed
    environment" error far from the actual cause). A backslash-escaped
    ``\\{``/``\\}`` is a literal brace, not a group delimiter, and does not
    count (same convention as ``_literal_or_math_mask``'s ``\\texttt{}``
    depth counter).
    """
    depth = 0
    index = 0
    length = len(inner)
    while index < length:
        char = inner[index]
        if char == "\\" and index + 1 < length:
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
        index += 1
    return depth == 0


def _classify_pair(
    line: str, marker: str, open_idx: int, close_idx: int, line_number: int
) -> MarkdownEmphasisMatch:
    """Classify a pair already matched by ``_find_matches_for_marker``'s
    stack walk -- every pair reaching here is a genuine emphasis attempt
    (code-shaped occurrences never get this far at all); this only decides
    whether it is SAFE to mechanically convert or AMBIGUOUS (left
    untouched, routed to block repair).
    """
    inner_start = open_idx + len(marker)
    inner = line[inner_start:close_idx]
    end = close_idx + len(marker)
    if not inner or inner[0].isspace() or inner[-1].isspace():
        return MarkdownEmphasisMatch(
            line_number, marker, open_idx, end, inner, True,
            "emphasis content is empty or starts/ends with whitespace",
        )
    if "|" in inner:
        return MarkdownEmphasisMatch(
            line_number, marker, open_idx, end, inner, True,
            "crosses a Markdown table cell boundary ('|')",
        )
    if not _has_balanced_braces(inner):
        return MarkdownEmphasisMatch(
            line_number, marker, open_idx, end, inner, True,
            "contains an unbalanced brace ('{' or '}') and is not safe to wrap in \\textbf{...}",
        )
    return MarkdownEmphasisMatch(line_number, marker, open_idx, end, inner, False)


def _find_matches_for_marker(
    line: str, marker: str, occurrences: List[int], line_number: int
) -> List[MarkdownEmphasisMatch]:
    """Pair ``marker`` occurrences on one line with a 1-deep stack.

    An occurrence always closes whatever is already pending, so the stack
    never grows past depth 1 -- the same marker genuinely cannot nest in
    itself (CommonMark agrees), so this alone is enough to resolve a run
    like "**a **b** c**" into two separate, non-nested spans rather than
    one containing the other.

    An occurrence with nothing pending opens a new pending span, but ONLY
    if it is not immediately preceded by an alphanumeric character.
    Excluding it here -- before any pairing, not after -- is what keeps
    "x**2"'s own "**" from ever being treated as a delimiter at all: under
    the previous blind-sequential-pairing design, that occurrence got
    paired with whatever the *next* "**" happened to be (even a real
    bold's opening delimiter), which silently broke the real pair instead
    of being ignored. There is deliberately no equivalent restriction on
    what a CLOSING delimiter may be followed by -- Korean (and other
    languages without whitespace-separated grammatical particles)
    routinely attaches a verb ending or particle directly after a bolded
    word with no space ("**중요**합니다"), which is normal prose, not a
    code-glued pattern; only gluing on the *opening* side reliably signals
    "this is the middle of some other token," so only that side is
    checked.

    A pending open still unclosed at end of line is reported as an
    unterminated, ambiguous match (it was already confirmed not to be
    code-shaped when it was pushed).
    """
    matches: List[MarkdownEmphasisMatch] = []
    pending_open: Optional[int] = None
    for idx in occurrences:
        if pending_open is not None:
            matches.append(_classify_pair(line, marker, pending_open, idx, line_number))
            pending_open = None
            continue
        before = line[idx - 1] if idx > 0 else ""
        if not before.isalnum():
            pending_open = idx
        # else: glued to a preceding alphanumeric character with nothing
        # pending to close (e.g. "x**2") -- not an emphasis attempt at
        # all, silently skipped (neither converted nor reported).
    if pending_open is not None:
        end = pending_open + len(marker)
        matches.append(
            MarkdownEmphasisMatch(
                line_number, marker, pending_open, len(line), line[end:], True,
                "unterminated -- no matching closing marker on this line",
            )
        )
    return matches


def _flag_cross_marker_overlaps(line_matches: List[MarkdownEmphasisMatch]) -> None:
    """Mark both spans ambiguous whenever a ``**`` span and a ``__`` span
    on the same line overlap at all -- nested (``**foo __bar__ baz**``) or
    partially crossing (``**foo __bar** baz__``) alike.

    Nesting is deliberately NOT treated as independently convertible:
    converting the inner span first and then splicing the outer span's
    *already-stale* ``inner``/``end`` (captured before that edit) in on
    top of it corrupts the text -- the real bug this rule exists to
    prevent, found via "**foo __bar__ baz**" producing
    "\\textbf{foo __bar__ baz}baz**". Routing both to block repair instead
    is what the plan itself asks for nested/ambiguous emphasis. The same
    marker can never reach this function already nested in itself: the
    1-deep stack in ``_find_matches_for_marker`` makes that structurally
    impossible.
    """
    for i, first in enumerate(line_matches):
        for second in line_matches[i + 1 :]:
            if first.marker == second.marker:
                continue
            if first.start < second.end and second.start < first.end:
                reason = "overlaps a different emphasis marker"
                first.ambiguous = True
                first.reason = first.reason or reason
                second.ambiguous = True
                second.reason = second.reason or reason


def find_markdown_emphasis_matches(
    text: str, *, apply_literal_math_mask: bool = True
) -> List[MarkdownEmphasisMatch]:
    """Find every ``**bold**``/``__bold__``-shaped span in ``text``,
    classifying each as safe to mechanically convert to ``\\textbf{...}`` or
    ambiguous (needs a human/model rewrite via block repair instead).

    Deliberately not a full CommonMark emphasis parser -- just enough
    structure-awareness to avoid the failure modes a naive "does this
    string contain '**'" check or a global regex replace would hit:

    - A delimiter starting inside math (tracked across the whole string,
      not per line), inside ``\\verb<delim>...<delim>``, or inside a
      ``\\texttt{...}`` argument is never considered at all (see
      ``_literal_or_math_mask``) -- unless ``apply_literal_math_mask`` is
      False, which skips that mask entirely (see its own docstring note
      below).
    - An occurrence immediately preceded by an alphanumeric character,
      with nothing already pending to close, is excluded *before*
      pairing -- this is what lets "x**2" sit next to a real
      "**important**" on the same line without disrupting it (see
      ``_find_matches_for_marker``'s docstring for why only the opening
      side is checked, not the closing side -- the latter would wrongly
      reject languages that attach grammar directly after a closing
      marker with no space).
    - A maximal run of 3+ of the same marker character (``***bold***``,
      a compound bold+italic marker this project's Markdown subset does
      not support) is never treated as an ordinary 2-char marker at all --
      round-2 review repro: the old 2-char-window scan read "***bold***"
      as "**" + "*bold" + "**" + "*", converting it to
      "\\textbf{*bold}*" and leaving a stray asterisk. Each such run is
      reported directly as its own ambiguous match instead (see the
      run-scanning loop below); a run of exactly 2 is the ordinary marker,
      a run of 1 is not a marker this project recognizes at all.
    - A matched pair whose inner text has an unbalanced brace is ambiguous
      too (see ``_classify_pair``/``_has_balanced_braces``) -- round-2
      review repro: "**text { brace**" converted to
      "\\textbf{text { brace}", a brace-imbalanced result.
    - Spans never cross a line break or a raw (non-math) '|' -- the
      latter is what keeps a forgotten closing '**' in one Markdown table
      cell from swallowing the next cell's content into one (wrong) bold
      span.
    - Pairing is a 1-deep stack (open, then whatever closes it), so the
      same marker can never nest in itself; a different-marker span that
      overlaps at all -- nested or merely crossing -- is ambiguous, never
      independently converted (see ``_flag_cross_marker_overlaps``).

    ``apply_literal_math_mask=False`` is for a plain-text field (title/
    caption) that is never rendered through real LaTeX math at all -- see
    ``_reject_plain_text_markup`` -- where applying the math mask would be
    wrong, not merely unnecessary: a bare ``$`` there is just a literal
    dollar sign, not the start of something to protect.
    """
    excluded = _literal_or_math_mask(text) if apply_literal_math_mask else [False] * len(text)
    matches: List[MarkdownEmphasisMatch] = []
    offset = 0
    for line_number, line in enumerate(text.split("\n"), start=1):
        length = len(line)

        occurrences_by_marker: Dict[str, List[int]] = {marker: [] for marker in _EMPHASIS_MARKERS}
        line_matches: List[MarkdownEmphasisMatch] = []
        index = 0
        while index < length:
            char = line[index]
            if char == "\\":
                index += 2
                continue
            if char in _MARKER_RUN_CHARS and not excluded[offset + index]:
                run_start = index
                while (
                    index < length
                    and line[index] == char
                    and not excluded[offset + index]
                ):
                    index += 1
                run_length = index - run_start
                if run_length == 2:
                    occurrences_by_marker[char * 2].append(run_start)
                elif run_length >= 3:
                    run_text = line[run_start:index]
                    line_matches.append(
                        MarkdownEmphasisMatch(
                            line_number, char * 2, run_start, index, run_text, True,
                            f"compound emphasis marker run ('{run_text}') is not "
                            "safely convertible",
                        )
                    )
                # run_length == 1: a lone '*'/'_' is not one of this
                # project's two supported markers at all -- ignored.
                continue
            index += 1

        for marker, occurrences in occurrences_by_marker.items():
            line_matches.extend(_find_matches_for_marker(line, marker, occurrences, line_number))

        _flag_cross_marker_overlaps(line_matches)
        matches.extend(line_matches)
        offset += length + 1  # +1 for the '\n' consumed by str.split("\n")
    return matches


def convert_markdown_emphasis(
    content: str,
    block_id: Optional[str] = None,
    *,
    matches: Optional[List[MarkdownEmphasisMatch]] = None,
) -> "tuple[str, List[str]]":
    """Convert every SAFE ``**bold**``/``__bold__`` span in ``content`` to
    ``\\textbf{...}`` in place; return the (possibly modified) content plus
    one description per AMBIGUOUS span left untouched, each naming its
    location and reason -- ready to feed a ``SemanticBlockError``/block
    repair verbatim instead of auto-converting something uncertain.

    Idempotent: the output contains no ``**``/``__`` delimiters that
    ``find_markdown_emphasis_matches`` would call safe (every one was just
    converted into a ``\\textbf{...}`` call, which contains none), so
    re-running this on its own output changes nothing.

    Pure -- never call this and then discard the possibility that nothing
    changed; the caller decides what to do with the returned content
    (persist it) and is responsible for recording that decision (see
    ``persist_markdown_emphasis_pass``). This function itself has no side
    effects.

    ``matches`` lets a caller that already ran ``find_markdown_emphasis_
    matches`` on this exact (newline-normalized) text -- e.g. to measure
    and persist that call's own latency/counts -- pass the result straight
    through instead of paying for the scan twice.
    """
    text = content.replace("\r\n", "\n")
    lines = text.split("\n")
    if matches is None:
        matches = find_markdown_emphasis_matches(text)

    safe_by_line: Dict[int, List[MarkdownEmphasisMatch]] = {}
    violations: List[str] = []
    for match in matches:
        if match.ambiguous:
            location = f"block '{block_id}'" if block_id else "block"
            # The exact source substring, not marker+inner+marker
            # reconstructed -- for a compound marker run (e.g. "***bold***")
            # `inner` already IS the whole flagged run, so re-wrapping it in
            # `marker` on both sides would double it into a garbled
            # "*******"-style display.
            raw = lines[match.line_number - 1][match.start : match.end]
            violations.append(f"{location}, line {match.line_number}: {raw!r} {match.reason}")
        else:
            safe_by_line.setdefault(match.line_number, []).append(match)

    for line_number, line_matches in safe_by_line.items():
        line = lines[line_number - 1]
        for match in sorted(line_matches, key=lambda m: m.start, reverse=True):
            line = line[: match.start] + f"\\textbf{{{match.inner}}}" + line[match.end :]
        lines[line_number - 1] = line

    return "\n".join(lines), violations


# Bumped whenever find_markdown_emphasis_matches/convert_markdown_emphasis's
# detection or safety rules change, so a persisted
# markdown_emphasis_conversion_*.json record (see
# persist_markdown_emphasis_pass) stays attributable to the exact algorithm
# that produced it even after the rules move on:
#   1 -- initial Stage 1 implementation (symmetric alnum-adjacency check on
#        both sides of a pair, blind sequential pairing, per-line math
#        reset).
#   2 -- round-1 review fixes: 1-deep stack pairing, asymmetric
#        (opening-side-only) alnum check, whole-text literal/math mask
#        covering \verb and \texttt{} and multi-line math, cross-marker
#        overlap always ambiguous.
#   3 -- round-2 review fixes: escape-aware literal/math mask (\$, \{, \},
#        \\), plain title/caption fields skip the math mask entirely,
#        compound marker runs (***bold***) and unbalanced-brace spans are
#        never auto-converted, and the pure per-block check fails on ANY
#        remaining match (not only ambiguous ones).
MARKDOWN_EMPHASIS_RULE_VERSION = 3


def persist_markdown_emphasis_pass(
    sections_dir: Path,
    node_id: str,
    *,
    phase: str,
    block_id: Optional[str],
    original_content: Optional[str],
    converted_content: Optional[str],
    safe_count: int,
    ambiguous_count: int,
    latency_s: float,
) -> None:
    """Persist one provenance record for one markdown_emphasis check/convert
    pass over one block's field, whatever the outcome.

    Round-2 review finding: the original version of this record was only
    written when ``converted_content`` differed from ``original_content``,
    so three kinds of real cost went completely unrecorded -- a block with
    nothing to convert, a block whose emphasis is entirely ambiguous (so
    content never changes even though the check still ran and still
    failed), and the composer's own pre-assembly residual check (``phase=
    "assembler_residual_check"``/``"assembler_figure_caption_check"``,
    called from ``LatexAssembler``). A record is written here every time,
    regardless of whether anything changed or any violation was found;
    ``converted_content`` stays ``None`` when nothing was (or could be)
    converted. ``rule_version`` stamps ``MARKDOWN_EMPHASIS_RULE_VERSION``
    so a historical record is always attributable to the detection/safety
    rules that produced it. File naming uses its own, separately-scanned
    sequence shared by every phase that calls this for the same
    ``node_id`` -- the same convention ``TextAgent._persist_failed_attempt``/
    ``_snapshot_before_overwrite`` use.
    """
    existing = list(sections_dir.glob(f"{node_id}.markdown_emphasis_conversion_*.json"))
    next_index = len(existing) + 1
    diff = ""
    if converted_content is not None and original_content is not None:
        diff = "\n".join(
            difflib.unified_diff(
                original_content.splitlines(),
                converted_content.splitlines(),
                fromfile="original",
                tofile="converted",
                lineterm="",
            )
        )
    record = {
        "node_id": node_id,
        "block_id": block_id,
        "phase": phase,
        "rule_version": MARKDOWN_EMPHASIS_RULE_VERSION,
        "original_content": original_content,
        "converted_content": converted_content,
        "diff": diff,
        "safe_match_count": safe_count,
        "ambiguous_match_count": ambiguous_count,
        "transform_latency_s": latency_s,
    }
    path = sections_dir / f"{node_id}.markdown_emphasis_conversion_{next_index}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def _paired_pipe_indices(row: str, spans: List["tuple[int, int]"]) -> "set[int]":
    """Indices of raw (non-escaped) '|' characters inside a math span whose
    span contains an *even*, non-zero count of them.

    A single raw '|' inside math (``P(A|B)``) is the same ambiguous
    conditional-probability notation the project's convention already
    requires writing as ``\\mid`` instead -- still a real column-count
    violation. A *pair* of raw '|' inside one math span (``|x - c|``) is
    unambiguous absolute-value notation with nothing to disambiguate, so
    those -- and only those -- are exempted from splitting.
    """
    paired: "set[int]" = set()
    for start, end in spans:
        positions = []
        i = start
        while i < end:
            if row[i] == "\\" and i + 1 < end:
                i += 2
                continue
            if row[i] == "|":
                positions.append(i)
            i += 1
        if positions and len(positions) % 2 == 0:
            paired.update(positions)
    return paired


def _split_markdown_row(row: str) -> List[str]:
    """Split one Markdown table row on unescaped '|', except a paired
    absolute-value-style '|' inside math mode -- see ``_paired_pipe_indices``.

    A plain ``row.split("|")`` (or a naive "'|' not preceded by a backslash"
    regex) treats every raw '|' as a column boundary, including
    ``\\( |x - c| \\)``'s absolute-value bars -- a table_columns violation
    the model had no correct way to satisfy, since that math was already
    written correctly (the "block_notation_table" incident this was pulled
    out to fix). ``\\|`` is always restored to a literal, non-delimiting '|'
    in the cell, in or out of math mode.
    """
    spans = _find_math_spans(row)
    paired = _paired_pipe_indices(row, spans)
    cells: List[str] = []
    current: List[str] = []
    length = len(row)
    index = 0
    while index < length:
        char = row[index]
        if char == "\\" and index + 1 < length and row[index + 1] == "|":
            current.append("|")
            index += 2
            continue
        if char == "|" and index not in paired:
            cells.append("".join(current).strip())
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    cells.append("".join(current).strip())
    return cells


def _split_leading_markdown_table(content: str) -> "tuple[List[str], str]":
    """Split the leading contiguous run of non-blank lines (the table) from
    any trailing prose that follows a blank line in the same block.

    A model was observed writing a valid Markdown table and then, in the
    same block, appending an explanatory paragraph directly below it with a
    blank line in between (a `comparison` block explaining its own table)
    instead of using a separate `paragraph` block. The old implementation
    folded every non-blank line in the *entire* content into the table, so
    that trailing paragraph became a "row" with however many columns as it
    happened to contain unescaped '|' characters (almost always a mismatch
    against the header) -- a parser boundary bug, not a model mistake. Only
    the leading run up to the first blank line is table data.
    """
    lines = content.splitlines()
    table_lines: List[str] = []
    index = 0
    while index < len(lines) and lines[index].strip():
        table_lines.append(lines[index].strip().strip("|"))
        index += 1
    remainder = "\n".join(lines[index:]).strip()
    return table_lines, remainder


def is_markdown_table(content: str) -> bool:
    table_lines, _ = _split_leading_markdown_table(content)
    return (
        len(table_lines) >= 2
        and "|" in table_lines[0]
        and bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?", table_lines[1]))
    )


def validate_markdown_table_columns(content: str, block_id: Optional[str]) -> None:
    """Reject a Markdown table whose data rows don't all have the header's
    column count.

    Pulled out of the composer's table renderer so TextAgent can run the
    same check right after generation (see ``validate_fragment_environments``
    for the same reasoning): a ragged table -- almost always an unescaped
    '|' inside a cell, e.g. conditional-probability notation like
    'E[Y|X=x]' written without '\\mid' -- used to only be discovered in the
    composer, where nothing retries and the error kills the whole document
    even though every other section already drafted correctly. Only checks
    the leading contiguous table block (see ``_split_leading_markdown_table``
    for why); trailing prose in the same block is not table data.
    """
    table_lines, _ = _split_leading_markdown_table(content)
    if not is_markdown_table(content):
        return
    raw_rows = [line for index, line in enumerate(table_lines) if index != 1]
    rows = [_split_markdown_row(row) for row in raw_rows]
    column_count = len(rows[0])
    if column_count == 0:
        raise ValueError(f"Markdown table header has no columns: {raw_rows[0]!r}")
    bad_rows = [(i, row, raw) for i, (row, raw) in enumerate(zip(rows, raw_rows)) if len(row) != column_count]
    if bad_rows:
        details = "; ".join(
            f"row {i} has {len(row)} column(s) (raw text: {raw!r})" for i, row, raw in bad_rows
        )
        raise ValueError(
            f"Block '{block_id}': Markdown table rows must all have the same number of "
            f"columns as the header ({column_count} column(s), from header {raw_rows[0]!r}); "
            f"{details}. The most common cause is a raw, unescaped '|' inside a cell -- for "
            r"example conditional-probability or set-builder notation written as 'P(A|B)' "
            r"instead of 'P(A \mid B)'. Escape an unavoidable literal pipe as '\|', or avoid it."
        )
VALID_WIDTH = re.compile(r"(?:0?\.\d+|1(?:\.0+)?)\\(?:textwidth|linewidth|columnwidth)")

# --- Pre-compile validation constants ---
# These support a last-line-of-defense pass over the fully assembled document,
# run once after every section is rendered and written but before invoking the
# LaTeX engine, so structural mistakes surface as a clear error instead of an
# opaque compiler failure (or, for dangling references, instead of a silently
# incomplete PDF full of "??").
REFERENCE_COMMAND = re.compile(r"\\(?:ref|Cref|cref|eqref|nameref|autoref|pageref)\*?\{(?P<label>[^{}]+)\}")
# Matches the `% <layout-anchor:{block_id}>` comment _render_section writes
# immediately before each block -- see locate_block_at_line.
_LAYOUT_ANCHOR_LINE = re.compile(r"^%\s*<layout-anchor:(?P<block_id>[A-Za-z0-9_-]+)>\s*$")
UNESCAPED_OPEN_BRACE = re.compile(r"(?<!\\)\{")
UNESCAPED_CLOSE_BRACE = re.compile(r"(?<!\\)\}")
# The argument of these commands is a label key, not prose: _sanitize_prose
# must copy it verbatim instead of escaping characters like '_' inside it,
# or the reference silently stops matching the label it was written to hit.
RAW_ARGUMENT_COMMANDS = {"ref", "Cref", "cref", "eqref", "nameref", "autoref", "pageref", "label"}
KNOWN_SECTION_LEVELS = {"chapter", "section", "subsection", "subsubsection"}
KNOWN_LAYOUT_POLICY_KEYS = {
    "insert_float_barrier_after_section",
    "maximum_full_width_figures_per_section",
    "allow_repeated_assets",
}
# "article" is already auto-downgraded to \section by _render_section_heading;
# these classes have no such downgrade path, so a "chapter" level here would
# reach the engine as a genuinely undefined \chapter command.
CHAPTERLESS_DOCUMENT_CLASSES_WITHOUT_DOWNGRADE = {"scrartcl"}


def _classify_composer_semantic_error(exc: Exception) -> str:
    """Map a _read_semantic_blocks() exception to a small, stable signal_type
    vocabulary. Covers both "the artifact isn't there / isn't parseable" and
    "the artifact's content doesn't match the blueprint contract" -- the
    composer's own independent re-check of what TextAgent already validated
    once, since a JSON artifact can outlive the process that wrote it."""
    if isinstance(exc, FileNotFoundError):
        return "missing_semantic_artifact"
    if isinstance(exc, ValueError):
        return "invalid_semantic_artifact"
    return "composer_semantic_check_error"


def _log_list_check(
    event_logger: Optional[EventLogger],
    *,
    verifier: str,
    errors: List[str],
    artifact_id: Optional[str],
    signal_type: str,
) -> None:
    """Log one pass/fail VerificationEvent for a composer check that returns a
    list of violation strings (empty list = pass). All of this module's
    validate_* methods return that shape."""
    if event_logger is None:
        return
    if errors:
        event_logger.log_verification(
            stage="composer",
            verifier=verifier,
            artifact_id=artifact_id,
            attempt=1,
            result="fail",
            signal_type=signal_type,
            message="; ".join(errors),
        )
    else:
        event_logger.log_verification(
            stage="composer",
            verifier=verifier,
            artifact_id=artifact_id,
            attempt=1,
            result="pass",
        )


# The same four checks TextAgent already gates behind its own
# ``enabled_validators`` (see DEFAULT_ENABLED_VALIDATORS in text_agent.py),
# re-run here as the composer's second line of defense right before
# rendering. Kept as an independently toggleable dict -- NOT mirrored
# automatically from TextAgent's setting -- so a verifier ablation run can
# study, say, "TextAgent's forbidden_environment check off, composer's on"
# as its own condition. Label-uniqueness checking has no code here because
# it stays always-on on the instance method (needs the document-wide label
# registry, not a per-code toggle). "markdown_emphasis" here is a pure
# residual CHECK, never a conversion -- TextAgent is where safe spans get
# mechanically converted; if one still reaches here unconverted (ablation
# disabled it there, or a stale pre-Stage-1 artifact), the composer must
# refuse loudly rather than silently rewrite content right before
# rendering, which would make the stored artifact, verification evidence,
# and PDF disagree about what this block actually contains.
DEFAULT_ENABLED_ASSEMBLER_VALIDATORS: Dict[str, bool] = {
    "forbidden_environment": True,
    "manual_label": True,
    "table_columns": True,
    "markdown_emphasis": True,
    # The five gate-level checks below: disabling one skips both running it
    # and logging its VerificationEvent, so ablation can measure what each
    # gate actually catches. Deliberately excluded from this dict: the
    # block id/order/type/asset_id consistency check inside
    # composer_semantic_block_validator (_validate_semantic_blocks) -- like
    # Planner's referential-integrity checks, disabling it would let a
    # blueprint/content mismatch reach the zip()-based renderer, which
    # assumes the two are already aligned and would misattribute content to
    # the wrong block instead of failing loudly.
    "layout_policy": True,
    "figure_asset_completeness": True,
    "figure_single_use": True,
    "reference": True,
    "document_frame": True,
}


class LatexAssembler:
    """Compose and structurally verify a document from its registered blueprint and assets."""

    def __init__(
        self,
        llm_client: Any = None,
        enabled_validators: Optional[Dict[str, bool]] = None,
    ):
        # Kept for constructor compatibility. Layout decisions are deterministic
        # once the planner has produced a DocumentBlueprint.
        self.llm = llm_client
        unknown = set(enabled_validators or {}) - set(DEFAULT_ENABLED_ASSEMBLER_VALIDATORS)
        if unknown:
            raise ValueError(
                f"Unknown enabled_validators code(s): {sorted(unknown)}. "
                f"Valid codes: {sorted(DEFAULT_ENABLED_ASSEMBLER_VALIDATORS)}."
            )
        self.enabled_validators: Dict[str, bool] = {
            **DEFAULT_ENABLED_ASSEMBLER_VALIDATORS,
            **(enabled_validators or {}),
        }

    def assemble(
        self,
        request_id: str,
        graph: DocumentGraph,
        output_dir: str = "outputs/generations",
        event_logger: Optional[EventLogger] = None,
    ) -> None:
        """Write a complete LaTeX project and structurally verify it.

        Raises ValueError/FileNotFoundError on any assembly or validation
        failure. Does not invoke a LaTeX engine -- call
        ``LatexCompiler.compile_pdf()`` afterward for that.
        """
        if graph.blueprint is None:
            raise ValueError("LaTeX assembly requires a DocumentBlueprint on the graph.")

        blueprint = graph.blueprint

        # Layout policy is decided entirely by the blueprint, independent of any
        # section content, so it can be checked before doing any file I/O.
        if self.enabled_validators.get("layout_policy", True):
            layout_errors = self.validate_layout_policy(blueprint)
            _log_list_check(
                event_logger, verifier="layout_policy_validator", errors=layout_errors,
                artifact_id=request_id, signal_type="layout_policy_violation",
            )
            if layout_errors:
                raise ValueError("Layout policy validation failed:\n- " + "\n- ".join(layout_errors))

        base_dir = Path(output_dir) / request_id
        sections_dir = base_dir / "sections"
        base_dir.mkdir(parents=True, exist_ok=True)
        sections_dir.mkdir(parents=True, exist_ok=True)

        if self.enabled_validators.get("figure_asset_completeness", True):
            missing_assets = [
                figure_id
                for figure_id, entry in graph.figure_registry.items()
                if entry.required and not entry.asset_path
            ]
            _log_list_check(
                event_logger, verifier="figure_asset_completeness_validator", errors=missing_assets,
                artifact_id=request_id, signal_type="missing_figure_asset",
            )
            if missing_assets:
                raise ValueError(
                    "Cannot compose document: required figure assets are missing: "
                    + ", ".join(sorted(missing_assets))
                )

        # A re-compose is permitted, but each final rendering pass begins with
        # fresh placement counters so the one-use constraint remains meaningful.
        graph.reset_figure_placements()
        self._rendered_labels: set[str] = set()
        self._write_preamble(base_dir, blueprint)
        self._write_front_matter(base_dir, blueprint)

        rendered_sections: Dict[str, str] = {}
        for section_layout in blueprint.sections:
            rendered_section = self._render_section(
                section_layout,
                blueprint,
                graph,
                sections_dir,
                event_logger=event_logger,
            )
            rendered_sections[section_layout.section_id] = rendered_section
            (sections_dir / f"{section_layout.section_id}.tex").write_text(
                rendered_section,
                encoding="utf-8",
            )

        self._write_main_tex(base_dir, blueprint)

        # Final pass over the fully assembled project, immediately before
        # compilation: figure bookkeeping vs. actual rendered content, dangling
        # cross-references, and the overall document frame's structural health.
        validation_errors: List[str] = []

        if self.enabled_validators.get("figure_single_use", True):
            figure_use_errors = self.validate_figure_single_use(graph, rendered_sections)
            _log_list_check(
                event_logger, verifier="figure_single_use_validator", errors=figure_use_errors,
                artifact_id=request_id, signal_type="figure_single_use_violation",
            )
            validation_errors += figure_use_errors

        if self.enabled_validators.get("reference", True):
            reference_errors = self.validate_references(rendered_sections)
            _log_list_check(
                event_logger, verifier="reference_validator", errors=reference_errors,
                artifact_id=request_id, signal_type="undefined_reference",
            )
            validation_errors += reference_errors

        if self.enabled_validators.get("document_frame", True):
            frame_errors = self.validate_document_frame(base_dir, blueprint)
            _log_list_check(
                event_logger, verifier="document_frame_validator", errors=frame_errors,
                artifact_id=request_id, signal_type="document_frame_incomplete",
            )
            validation_errors += frame_errors

        if validation_errors:
            raise ValueError(
                "Pre-compile validation failed:\n- " + "\n- ".join(validation_errors)
            )

    def validate_layout_policy(self, blueprint: DocumentBlueprint) -> List[str]:
        """Confirm the blueprint's layout policy and section levels are renderable.

        Catches a planner mistake up front: an unrecognized ``layout_policy`` key
        silently no-ops today (for example a typo of
        ``insert_float_barrier_after_section``), an unsupported section level is
        silently coerced to ``section`` by ``_render_section_heading``, and a
        ``chapter``-level section under a chapterless document class would
        otherwise only surface as an undefined ``\\chapter`` command deep inside
        the LaTeX engine's log. ``maximum_full_width_figures_per_section`` and
        ``allow_repeated_assets`` are checked here too, against the blueprint
        contract, before any file I/O happens.
        """
        errors: List[str] = []
        unknown_keys = set(blueprint.layout_policy) - KNOWN_LAYOUT_POLICY_KEYS
        if unknown_keys:
            errors.append(
                "Unrecognized layout_policy key(s): " + ", ".join(sorted(unknown_keys))
            )
        if "insert_float_barrier_after_section" in blueprint.layout_policy and not isinstance(
            blueprint.layout_policy["insert_float_barrier_after_section"], bool
        ):
            errors.append("layout_policy['insert_float_barrier_after_section'] must be a boolean.")

        allow_repeated = blueprint.layout_policy.get("allow_repeated_assets", False)
        if "allow_repeated_assets" in blueprint.layout_policy and not isinstance(allow_repeated, bool):
            errors.append("layout_policy['allow_repeated_assets'] must be a boolean.")
        elif allow_repeated:
            # DocumentGraph.register_figure_asset hard-caps every figure at one
            # placement (see state.py), and figure_slots is keyed one-to-one by
            # figure_id, so "repeated" placement isn't representable yet.
            errors.append(
                "layout_policy['allow_repeated_assets'] = true is not supported: "
                "every figure asset is limited to exactly one placement."
            )
        else:
            seen_assets: Dict[str, str] = {}
            for section in blueprint.sections:
                for block in section.blocks:
                    if not block.asset_id:
                        continue
                    if block.asset_id in seen_assets:
                        errors.append(
                            f"Asset '{block.asset_id}' is referenced by layout blocks "
                            f"'{seen_assets[block.asset_id]}' and '{block.id}', but "
                            "layout_policy['allow_repeated_assets'] is false."
                        )
                    else:
                        seen_assets[block.asset_id] = block.id

        if "maximum_full_width_figures_per_section" in blueprint.layout_policy:
            max_full_width = blueprint.layout_policy["maximum_full_width_figures_per_section"]
            if (
                not isinstance(max_full_width, int)
                or isinstance(max_full_width, bool)
                or max_full_width < 0
            ):
                errors.append(
                    "layout_policy['maximum_full_width_figures_per_section'] must be a "
                    "non-negative integer."
                )
            else:
                full_width_counts: Dict[str, int] = {}
                for placement in blueprint.figure_slots.values():
                    if placement.layout == FigureLayout.FLOAT:
                        full_width_counts[placement.owner_section] = (
                            full_width_counts.get(placement.owner_section, 0) + 1
                        )
                for section_id, count in full_width_counts.items():
                    if count > max_full_width:
                        errors.append(
                            f"Section '{section_id}' has {count} full-width figure(s), "
                            "exceeding layout_policy['maximum_full_width_figures_per_section'] "
                            f"= {max_full_width}."
                        )

        for section in blueprint.sections:
            level = section.level.lower()
            if level not in KNOWN_SECTION_LEVELS:
                errors.append(
                    f"Section '{section.section_id}' has unsupported level '{section.level}'; "
                    f"expected one of {sorted(KNOWN_SECTION_LEVELS)}."
                )
            elif (
                level == "chapter"
                and blueprint.document_class in CHAPTERLESS_DOCUMENT_CLASSES_WITHOUT_DOWNGRADE
            ):
                errors.append(
                    f"Section '{section.section_id}' uses level 'chapter', but document class "
                    f"'{blueprint.document_class}' has no \\chapter command and is not "
                    "auto-downgraded."
                )
        return errors

    def validate_figure_single_use(
        self,
        graph: DocumentGraph,
        rendered_sections: Dict[str, str],
    ) -> List[str]:
        """Confirm every figure keeps its single-use guarantee in the actual output.

        ``graph.validate_figure_usage`` checks the placement bookkeeping; this
        additionally counts the literal ``\\includegraphics`` occurrences in the
        rendered section text so a bookkeeping/rendering mismatch (a duplicated
        or vanished figure) cannot slip through undetected.
        """
        errors = list(graph.validate_figure_usage())
        assembled = "\n".join(rendered_sections.values())
        for figure_id, entry in graph.figure_registry.items():
            if not entry.asset_path:
                continue
            occurrences = assembled.count(f"{{{entry.asset_path}}}")
            expected = 1 if entry.placement_count == 1 else 0
            if occurrences != expected:
                errors.append(
                    f"Figure '{figure_id}' expected {expected} \\includegraphics occurrence(s) "
                    f"of '{entry.asset_path}' in the rendered sections but found {occurrences}."
                )
        return errors

    def validate_references(self, rendered_sections: Dict[str, str]) -> List[str]:
        """Confirm every \\ref-style command targets a label that was actually defined.

        LaTeX does not fail compilation on an undefined reference -- hyperref and
        cleveref just emit a "??" with a warning -- so this check is the only
        thing standing between a broken cross-reference and a silently
        incomplete PDF.
        """
        errors: List[str] = []
        for section_id, text in rendered_sections.items():
            for match in REFERENCE_COMMAND.finditer(text):
                label = match.group("label")
                if label not in self._rendered_labels:
                    errors.append(
                        f"Section '{section_id}' references undefined label '{label}'."
                    )
        return errors

    def validate_document_frame(self, base_dir: Path, blueprint: DocumentBlueprint) -> List[str]:
        """Confirm the assembled document frame is structurally complete.

        Runs after every frame and section file has been written, immediately
        before compilation, to catch a broken ``\\input`` target or a residual
        brace imbalance as a clear pre-compile error instead of an opaque LaTeX
        engine failure.
        """
        errors: List[str] = []
        frame_paths = {
            "preamble": base_dir / "preamble.tex",
            "frontmatter": base_dir / "frontmatter.tex",
            "main": base_dir / "main.tex",
        }
        for name, path in frame_paths.items():
            if not path.is_file():
                errors.append(f"Missing required frame file '{path}'.")
        if errors:
            # Nothing else here can be checked without the frame files.
            return errors

        preamble_text = frame_paths["preamble"].read_text(encoding="utf-8")
        if r"\documentclass" not in preamble_text:
            errors.append("preamble.tex is missing a \\documentclass declaration.")

        main_text = frame_paths["main"].read_text(encoding="utf-8")
        if main_text.count(r"\begin{document}") != 1 or main_text.count(r"\end{document}") != 1:
            errors.append("main.tex must contain exactly one \\begin{document}/\\end{document} pair.")

        assembled_parts = [preamble_text, frame_paths["frontmatter"].read_text(encoding="utf-8")]
        for match in re.finditer(r"\\input\{([^{}]+)\}", main_text):
            target = base_dir / f"{match.group(1)}.tex"
            if not target.is_file():
                errors.append(f"main.tex references missing input file '{target}'.")
            else:
                assembled_parts.append(target.read_text(encoding="utf-8"))

        assembled = "\n".join(assembled_parts)
        open_count = len(UNESCAPED_OPEN_BRACE.findall(assembled))
        close_count = len(UNESCAPED_CLOSE_BRACE.findall(assembled))
        if open_count != close_count:
            errors.append(
                "Assembled document has unbalanced braces "
                f"(found {open_count} unescaped '{{' vs {close_count} unescaped '}}')."
            )
        return errors

    def _render_section(
        self,
        layout: SectionLayout,
        blueprint: DocumentBlueprint,
        graph: DocumentGraph,
        sections_dir: Path,
        event_logger: Optional[EventLogger] = None,
    ) -> str:
        node = graph.nodes.get(layout.section_id)
        if node is None:
            raise ValueError(f"Blueprint refers to unknown section '{layout.section_id}'.")

        try:
            semantic_blocks = self._read_semantic_blocks(sections_dir, layout)
        except Exception as exc:
            if event_logger is not None:
                event_logger.log_verification(
                    stage="composer",
                    verifier="composer_semantic_block_validator",
                    artifact_id=layout.section_id,
                    section_id=layout.section_id,
                    attempt=1,
                    result="fail",
                    signal_type=_classify_composer_semantic_error(exc),
                    message=str(exc),
                )
            raise
        if event_logger is not None:
            event_logger.log_verification(
                stage="composer",
                verifier="composer_semantic_block_validator",
                artifact_id=layout.section_id,
                section_id=layout.section_id,
                attempt=1,
                result="pass",
            )
        figure_after = self._figures_by_anchor(layout.section_id, blueprint)
        rendered: List[str] = [self._render_section_heading(layout, blueprint)]

        for layout_block, content_block in zip(layout.blocks, semantic_blocks):
            # Written immediately before the block's own rendered LaTeX so a
            # compiler error's line number can be resolved back to the block
            # that produced it (see locate_block_at_line) -- the same
            # convention TextAgent._render_draft_with_anchors already uses
            # for its own draft artifact, applied here to the text that
            # actually gets compiled.
            rendered.append(f"% <layout-anchor:{layout_block.id}>")
            rendered.append(self._render_semantic_block(layout_block, content_block))

            for figure_id in figure_after.pop(layout_block.id, []):
                rendered.append(
                    self._render_figure(figure_id, layout.section_id, blueprint, graph, sections_dir)
                )

        # Slots without an explicit anchor belong at the end of their owner section.
        for figure_ids in figure_after.values():
            for figure_id in figure_ids:
                rendered.append(
                    self._render_figure(figure_id, layout.section_id, blueprint, graph, sections_dir)
                )

        if blueprint.layout_policy.get("insert_float_barrier_after_section", False):
            rendered.append(r"\FloatBarrier")
        return "\n\n".join(part for part in rendered if part.strip()) + "\n"

    @staticmethod
    def locate_block_at_line(section_tex_path: Path, line_number: int) -> Optional[str]:
        """Resolve a 1-indexed line number in a rendered ``sections/{id}.tex``
        file back to the semantic block whose content that line belongs to.

        Used by the compile-repair loop (``run_full_pipeline.
        DocGenPipeline._repair_and_recompile``) to narrow a tectonic error
        down to one block instead of redrafting the whole section. Returns
        ``None`` -- caller falls back to whole-section regeneration -- when
        the file doesn't exist, or the line falls before any anchor (e.g.
        inside the section heading itself).
        """
        if not section_tex_path.is_file():
            return None
        current_block_id: Optional[str] = None
        for index, line in enumerate(
            section_tex_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            match = _LAYOUT_ANCHOR_LINE.match(line)
            if match:
                current_block_id = match.group("block_id")
            if index >= line_number:
                break
        return current_block_id

    def _read_semantic_blocks(
        self,
        sections_dir: Path,
        layout: SectionLayout,
    ) -> List[ContentBlock]:
        """Load the TextAgent artifact; draft TeX is intentionally never read.

        ``*.draft.tex`` is a human-inspection artifact only.  Rendering from it
        would lose the type, planned block ID, and figure ownership required to
        construct a predictable LaTeX document.
        """
        blocks_path = sections_dir / f"{layout.section_id}.blocks.json"
        if not blocks_path.is_file():
            raise FileNotFoundError(
                f"Missing semantic block artifact '{blocks_path}'. "
                "Run TextAgent successfully before LaTeX integration."
            )

        try:
            payload = json.loads(blocks_path.read_text(encoding="utf-8"))
            output = TextAgentOutput.model_validate(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid semantic block artifact '{blocks_path}': {exc}") from exc

        if output.section_id != layout.section_id:
            raise ValueError(
                f"Semantic artifact '{blocks_path}' declares section '{output.section_id}', "
                f"not '{layout.section_id}'."
            )
        self._validate_semantic_blocks(layout, output.blocks, blocks_path)
        return output.blocks

    def _validate_semantic_blocks(
        self,
        layout: SectionLayout,
        blocks: Sequence[ContentBlock],
        blocks_path: Path,
    ) -> None:
        # blocks_path is always "<sections_dir>/<section_id>.blocks.json"
        # (see _read_semantic_blocks) -- derived here instead of adding a
        # separate parameter, so _validate_fragment's residual check below
        # can persist its own cost/provenance record (phase=
        # "assembler_residual_check") into the exact same sections_dir the
        # real generation path's own markdown_emphasis_conversion_*.json
        # records already live in, under the same node_id (section_id).
        sections_dir = blocks_path.parent
        node_id = layout.section_id
        expected_ids = [block.id for block in layout.blocks]
        actual_ids = [block.block_id for block in blocks]
        if actual_ids != expected_ids:
            raise ValueError(
                f"Semantic artifact '{blocks_path}' does not match blueprint block order. "
                f"Expected {expected_ids}, received {actual_ids}."
            )

        for layout_block, content_block in zip(layout.blocks, blocks):
            expected_type = SEMANTIC_TYPE_BY_LAYOUT_KIND[layout_block.kind]
            if content_block.type != expected_type:
                raise ValueError(
                    f"Block '{layout_block.id}' must be '{expected_type.value}', "
                    f"not '{content_block.type.value}'."
                )
            if not content_block.content or not content_block.content.strip():
                raise ValueError(f"Semantic block '{layout_block.id}' has no content.")
            if layout_block.kind == LayoutBlockKind.FIGURE:
                if content_block.asset_id != layout_block.asset_id:
                    raise ValueError(
                        f"Figure-reference block '{layout_block.id}' must reference "
                        f"asset '{layout_block.asset_id}'."
                    )
            elif content_block.asset_id:
                raise ValueError(
                    f"Only figure-reference blocks may declare asset_id; "
                    f"block '{layout_block.id}' is '{layout_block.kind.value}'."
                )
            self._validate_fragment(content_block, sections_dir, node_id)

    @staticmethod
    def _figures_by_anchor(
        section_id: str,
        blueprint: DocumentBlueprint,
    ) -> Dict[Optional[str], List[str]]:
        figures: Dict[Optional[str], List[str]] = {}
        for figure_id, placement in blueprint.figure_slots.items():
            if placement.owner_section == section_id:
                figures.setdefault(placement.anchor_after_block, []).append(figure_id)
        return figures

    def _render_section_heading(self, layout: SectionLayout, blueprint: DocumentBlueprint) -> str:
        level = layout.level.lower()
        if level not in {"chapter", "section", "subsection", "subsubsection"}:
            level = "section"
        if level == "chapter" and blueprint.document_class == "article":
            level = "section"
        label = self._safe_label(f"sec:{layout.section_id}", f"section '{layout.section_id}'")
        return rf"\{level}{{{self._escape_latex(layout.title)}}}" + "\n" + rf"\label{{{label}}}"

    def _render_semantic_block(self, layout: LayoutBlock, block: ContentBlock) -> str:
        """Render one validated semantic block using the planned presentation.

        TextAgent supplies meaning, not document-level LaTeX.  This separation
        lets the renderer apply typography consistently and avoids accidental
        nested floats, duplicate figures, or a raw section command from a model.
        """
        self._register_fragment_labels(block.content, f"block '{layout.id}'")
        if block.type == SemanticBlockType.PARAGRAPH:
            return self._sanitize_prose(block.content)
        if block.type == SemanticBlockType.LIST:
            return self._render_list(block.content)
        if block.type == SemanticBlockType.TABLE:
            return self._render_table(layout, block)
        if block.type == SemanticBlockType.NOTATION_TABLE:
            return self._render_table(layout, block, default_caption="Notation")
        if block.type == SemanticBlockType.EQUATION:
            return self._render_equation(layout, block)
        if block.type == SemanticBlockType.FIGURE_REFERENCE:
            return self._sanitize_prose(block.content)
        if block.type == SemanticBlockType.THEOREM:
            return self._render_theorem(layout, block)
        if block.type == SemanticBlockType.PROOF_SKETCH:
            return self._render_proof_sketch(block)

        title = self._escape_latex(block.title or layout.title or self._default_block_title(block.type))
        if block.type == SemanticBlockType.COMPARISON:
            return self._render_comparison(layout, title, block.content)

        body = self._sanitize_prose(block.content)

        style = {
            SemanticBlockType.KEY_TAKEAWAY: "colback=blue!4!white,colframe=blue!55!black",
            SemanticBlockType.DEFINITION: "colback=green!4!white,colframe=green!45!black",
            SemanticBlockType.WARNING: "colback=red!5!white,colframe=red!60!black",
            SemanticBlockType.CASE_STUDY: "colback=gray!5!white,colframe=black!55",
        }[block.type]

        # A tcolorbox has no LaTeX numbering of its own -- a \label{} placed
        # in one with no counter behind it would silently print whatever the
        # nearest *other* counter happens to be (usually the section number),
        # not a number unique to this definition. \refstepcounter{definition}
        # (declared in _write_preamble alongside the block-kind package scan
        # that turns it on) gives \label{} something real to capture, the
        # same way DEFINITION is the only tcolorbox-rendered kind
        # default_block_label() offers a label for at all.
        counter_prefix = ""
        if block.type == SemanticBlockType.DEFINITION:
            custom_title = block.title or layout.title
            title = f"Definition~\\thedefinition: {title}" if custom_title else "Definition~\\thedefinition"
            label = self._safe_label(
                default_block_label(LayoutBlockKind.DEFINITION, layout.id),
                f"definition block '{layout.id}'",
            )
            counter_prefix = f"\\refstepcounter{{definition}}\\label{{{label}}}\n"

        box = (
            counter_prefix
            + f"\\begin{{tcolorbox}}[{style},title={{{title}}},fonttitle=\\bfseries]\n"
            f"{body}\n"
            "\\end{tcolorbox}"
        )
        if layout.component == LatexComponent.MINIPAGE or block.type == SemanticBlockType.CASE_STUDY:
            return (
                "\\begin{center}\n\\begin{minipage}{0.92\\linewidth}\n"
                f"{box}\n"
                "\\end{minipage}\n\\end{center}"
            )
        return box

    @staticmethod
    def _default_block_title(block_type: SemanticBlockType) -> str:
        return {
            SemanticBlockType.KEY_TAKEAWAY: "Key takeaway",
            SemanticBlockType.DEFINITION: "Definition",
            SemanticBlockType.WARNING: "Important consideration",
            SemanticBlockType.CASE_STUDY: "Case study",
            SemanticBlockType.COMPARISON: "Comparison",
        }.get(block_type, "Details")

    def _validate_fragment(
        self,
        block: ContentBlock,
        sections_dir: Optional[Path] = None,
        node_id: Optional[str] = None,
    ) -> None:
        """Reject LaTeX that would escape the semantic-block rendering boundary.

        ``sections_dir``/``node_id`` are optional (default ``None``, same
        convention as ``event_logger`` elsewhere in this class) -- a direct
        unit-test caller that only wants the raise/no-raise behavior can
        still call ``_validate_fragment(block)`` with no change; the real
        ``_validate_semantic_blocks`` call site always supplies both, so the
        residual check's own cost is persisted (``phase=
        "assembler_residual_check"``) on the real composer path -- round-2
        review finding: this check's latency went completely unrecorded
        before, unlike TextAgent's generation-time conversion pass.
        """
        content = block.content.replace("\r\n", "\n")
        if self.enabled_validators.get("forbidden_environment", True):
            content = validate_fragment_environments(block.type, block.content, block.block_id)
        if self.enabled_validators.get("manual_label", True):
            validate_no_manual_labels(block.type, content, block.block_id)
        if (
            block.type in (SemanticBlockType.TABLE, SemanticBlockType.NOTATION_TABLE, SemanticBlockType.COMPARISON)
            and self.enabled_validators.get("table_columns", True)
        ):
            validate_markdown_table_columns(content, block.block_id)
        if self.enabled_validators.get("markdown_emphasis", True):
            start = time.perf_counter()
            residual = find_markdown_emphasis_matches(content)
            safe_count = sum(1 for m in residual if not m.ambiguous)
            ambiguous_count = len(residual) - safe_count
            try:
                if residual:
                    sample = residual[0]
                    raise ValueError(
                        f"Block '{block.block_id}' line {sample.line_number} still contains "
                        f"Markdown emphasis syntax ('{sample.marker}{sample.inner}{sample.marker}') "
                        "-- TextAgent's own markdown_emphasis check should have converted or "
                        "rejected this before the artifact reached the composer; converting it "
                        "here instead would make the stored artifact and the rendered PDF "
                        "disagree about this block's content."
                    )
                # title/caption are rendered through _escape_latex only
                # (never through LaTeX), so the same residual-check
                # discipline applies to the exact fields
                # _render_semantic_block/_render_table actually use for
                # final output, not only to `content`. Their own
                # find_markdown_emphasis_matches cost is folded into this
                # same latency window/record rather than persisting a
                # third and fourth record per block for two short strings.
                validate_plain_title(block.title, f"Block '{block.block_id}'")
                validate_plain_caption(block.caption, f"Block '{block.block_id}'")
            finally:
                latency_s = time.perf_counter() - start
                if sections_dir is not None and node_id is not None:
                    persist_markdown_emphasis_pass(
                        sections_dir,
                        node_id,
                        phase="assembler_residual_check",
                        block_id=block.block_id,
                        original_content=content,
                        converted_content=None,
                        safe_count=safe_count,
                        ambiguous_count=ambiguous_count,
                        latency_s=latency_s,
                    )
        for label_match in LABEL_TOKEN.finditer(content):
            self._validate_label(label_match.group("label"), f"block '{block.block_id}'")

    def _render_list(self, content: str) -> str:
        content = content.strip()
        if ENVIRONMENT_TOKEN.search(content):
            # Validation already restricted this to one of the supported list
            # environments.  Preserve explicitly authored nested lists.
            return content

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        numbered = bool(lines and re.match(r"\d+[.)]\s+", lines[0]))
        items: List[str] = []
        for line in lines:
            cleaned = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", line).strip()
            if cleaned:
                items.append(self._sanitize_prose(cleaned))
        if not items:
            items = [self._sanitize_prose(content)]
        environment = "enumerate" if numbered else "itemize"
        return "\n".join(
            [f"\\begin{{{environment}}}[leftmargin=*]"]
            + [f"\\item {item}" for item in items]
            + [f"\\end{{{environment}}}"]
        )

    def _render_table(self, layout: LayoutBlock, block: ContentBlock, default_caption: str = "Data summary") -> str:
        content = block.content.strip()
        if is_markdown_table(content):
            content = self._markdown_table_to_tabularx(content)

        # A specialist tool may return a complete table environment.  It has
        # already passed the environment validation, so retain its booktabs
        # structure instead of trying to reconstruct data from TeX source.
        if re.search(r"\\begin\s*\{(?:table|longtable)\}", content):
            return content

        if LABEL_TOKEN.search(content):
            raise ValueError(
                f"Table block '{layout.id}' may use labels only inside a complete "
                "table or longtable environment."
            )
        if not re.search(r"\\begin\s*\{(?:tabular|tabularx|array)\}", content):
            content = self._single_column_table(content)

        if re.search(r"\\begin\s*\{longtable\}", content):
            return content

        title = self._escape_latex(block.caption or block.title or layout.title or default_caption)
        label = self._safe_label(
            default_block_label(layout.kind, layout.id),
            f"table block '{layout.id}'",
        )
        float_options = self._safe_float_options(block.metadata.get("float_options", "htbp"))
        return (
            f"\\begin{{table}}[{float_options}]\n"
            "\\centering\n\\small\n"
            f"\\caption{{{title}}}\n"
            f"\\label{{{label}}}\n"
            f"{content}\n"
            "\\end{table}"
        )

    def _markdown_table_to_tabularx(self, content: str) -> str:
        """Render the leading Markdown table in ``content`` as a ``tabularx``.

        Only the leading contiguous block of lines is table data (see
        ``_split_leading_markdown_table``); anything after the first blank
        line is trailing prose the model wrote in the same block (e.g. a
        `comparison` block explaining its own table) and is appended below
        the table as an ordinary sanitized paragraph instead of being
        misparsed as a malformed row.
        """
        validate_markdown_table_columns(content, None)
        table_lines, remainder = _split_leading_markdown_table(content)
        raw_rows = [line for index, line in enumerate(table_lines) if index != 1]
        rows = [_split_markdown_row(row) for row in raw_rows]
        column_count = len(rows[0])
        column_spec = "@{}" + "l" + "X" * (column_count - 1) + "@{}"
        rendered = [f"\\begin{{tabularx}}{{\\linewidth}}{{{column_spec}}}", r"\toprule"]
        rendered.append(" & ".join(rf"\textbf{{{self._sanitize_prose(cell)}}}" for cell in rows[0]) + r" \\")
        rendered.append(r"\midrule")
        for row in rows[1:]:
            rendered.append(" & ".join(self._sanitize_prose(cell) for cell in row) + r" \\")
        rendered.extend([r"\bottomrule", r"\end{tabularx}"])
        if remainder:
            rendered.extend(["", self._sanitize_prose(remainder)])
        return "\n".join(rendered)

    def _single_column_table(self, content: str) -> str:
        safe_content = self._sanitize_prose(content).replace("\n\n", r"\\ ")
        return "\n".join(
            [r"\begin{tabularx}{\linewidth}{@{}X@{}}", r"\toprule", safe_content + r"\\", r"\bottomrule", r"\end{tabularx}"]
        )

    def _render_equation(self, layout: LayoutBlock, block: ContentBlock) -> str:
        """Render an EQUATION block, numbered and labeled by default.

        A model-supplied complete environment (``\\begin{align}...``) is the
        model's own explicit structural choice for a multi-line derivation,
        but ``validate_no_manual_labels`` now forbids it from writing its own
        ``\\label{}`` inside that environment (labels are composer-owned) --
        so the default label is inserted here, right after the opening
        ``\\begin{...}`` line, the same way the bare-formula case below gets
        one. Without this, ``DocumentGraph.label_registry`` would still
        promise ``eq:<block_id>`` exists (it doesn't distinguish bare from
        complete-environment equations), but nothing would ever actually
        emit it, leaving any cross-reference to it undefined at compile
        time. An already-``\\[...\\]``-wrapped body is left unlabeled by
        design: it's amsmath's *unnumbered* display math, so a label on it
        wouldn't refer to anything meaningful.
        """
        equation = block.content.strip()
        if ENVIRONMENT_TOKEN.search(equation):
            label = self._safe_label(
                default_block_label(LayoutBlockKind.EQUATION, layout.id),
                f"equation block '{layout.id}'",
            )
            begin_match = re.search(r"\\begin\{[A-Za-z*]+\}[^\n]*", equation)
            insert_at = begin_match.end()
            return (
                equation[:insert_at]
                + f"\n\\label{{{label}}}"
                + equation[insert_at:]
            )
        if equation.startswith(r"\[") and equation.endswith(r"\]"):
            return equation
        if equation.startswith("$$") and equation.endswith("$$"):
            equation = equation[2:-2].strip()
        elif equation.startswith("$") and equation.endswith("$"):
            equation = equation[1:-1].strip()
        if "$" in equation:
            raise ValueError(
                f"Equation block contains a stray '$' outside its single math body: {block.content!r}"
            )
        label = self._safe_label(
            default_block_label(LayoutBlockKind.EQUATION, layout.id),
            f"equation block '{layout.id}'",
        )
        return f"\\begin{{equation}}\n{equation}\n\\label{{{label}}}\n\\end{{equation}}"

    def _render_theorem(self, layout: LayoutBlock, block: ContentBlock) -> str:
        """Render a THEOREM block as a numbered amsthm ``theorem`` environment.

        The model supplies only the theorem statement (plain prose, validated
        the same way as definition/warning/case_study); the composer owns the
        environment itself, its numbering, and the optional named title --
        the same "content is semantic only" boundary used everywhere else in
        this file. See ``_write_preamble`` for the matching ``\\newtheorem``
        declaration and ``amsthm`` package inclusion.
        """
        body = self._sanitize_prose(block.content)
        title = block.title or layout.title
        opt_title = f"[{self._escape_latex(title)}]" if title else ""
        label = self._safe_label(
            default_block_label(LayoutBlockKind.THEOREM, layout.id),
            f"theorem block '{layout.id}'",
        )
        return f"\\begin{{theorem}}{opt_title}\n\\label{{{label}}}\n{body}\n\\end{{theorem}}"

    def _render_proof_sketch(self, block: ContentBlock) -> str:
        """Render a PROOF_SKETCH block as amsthm's built-in ``proof`` environment,
        renamed via its optional argument to make clear it is a sketch rather
        than a complete formal proof."""
        body = self._sanitize_prose(block.content)
        return f"\\begin{{proof}}[Proof Sketch]\n{body}\n\\end{{proof}}"

    def _render_comparison(self, layout: LayoutBlock, title: str, content: str) -> str:
        if is_markdown_table(content):
            body = self._markdown_table_to_tabularx(content)
        elif ENVIRONMENT_TOKEN.search(content):
            body = content.strip()
        else:
            body = self._sanitize_prose(content)
        if layout.component == LatexComponent.MINIPAGE:
            return (
                "\\begin{center}\n\\begin{minipage}{0.94\\linewidth}\n"
                f"\\textbf{{{title}}}\n\n{body}\n"
                "\\end{minipage}\n\\end{center}"
            )
        return (
            f"\\begin{{tcolorbox}}[colback=gray!4!white,colframe=black!45,title={{{title}}},fonttitle=\\bfseries]\n"
            f"{body}\n"
            "\\end{tcolorbox}"
        )

    @staticmethod
    def _sanitize_prose(content: str) -> str:
        """Escape text-mode special characters without disturbing TeX commands.

        TextAgent is prompted to supply escaped LaTeX.  This second pass makes
        ordinary prose (the common model output) safe while preserving commands,
        command arguments, and math delimiters.  Table and equation fragments
        intentionally bypass this method because ``&`` and ``_`` are meaningful
        syntax in those contexts.  ``~`` is left untouched: it is the standard
        LaTeX non-breaking space (as in ``Figure~\\ref{...}``), not a character
        needing escape.  The argument of a reference-style command such as
        ``\\ref``/``\\Cref``/``\\label`` is copied verbatim -- it is a label
        key, not prose, so escaping characters like '_' inside it would
        silently break the reference.

        ``$$...$$`` (display math) is treated the same as ``$...$``: a run of
        two ``$`` is one toggle, not two independent ones. Toggling twice was
        the bug -- ``$$`` flipped ``inline_math`` on and immediately back off,
        so everything up to the closing ``$$`` was treated as plain prose and
        had its ``^``/``_``/braces escaped into literal glyphs instead of
        being left as math.
        """
        output: List[str] = []
        text = content.replace("\r\n", "\n").strip()
        index = 0
        inline_math = False
        math_command_delimiter: Optional[str] = None
        latex_brace_depth = 0
        awaiting_command_argument = False
        pending_raw_argument = False
        while index < len(text):
            char = text[index]
            if char == "\\":
                if index + 1 < len(text) and text[index + 1].isalpha():
                    end = index + 2
                    while end < len(text) and text[end].isalpha():
                        end += 1
                    command = text[index:end]
                    output.append(command)
                    awaiting_command_argument = True
                    pending_raw_argument = command[1:] in RAW_ARGUMENT_COMMANDS
                    index = end
                    continue
                if index + 1 < len(text):
                    command = text[index : index + 2]
                    output.append(command)
                    if command in {r"\(", r"\["}:
                        inline_math = True
                        math_command_delimiter = command
                    elif command in {r"\)", r"\]"}:
                        inline_math = False
                        math_command_delimiter = None
                    awaiting_command_argument = False
                    pending_raw_argument = False
                    index += 2
                    continue
                output.append(r"\textbackslash{}")
                index += 1
                continue
            if char == "$":
                if index + 1 < len(text) and text[index + 1] == "$":
                    inline_math = not inline_math
                    output.append("$$")
                    awaiting_command_argument = False
                    pending_raw_argument = False
                    index += 2
                    continue
                inline_math = not inline_math
                output.append(char)
                awaiting_command_argument = False
                pending_raw_argument = False
            elif not inline_math and char == "{" and pending_raw_argument:
                close_index = text.find("}", index + 1)
                if close_index == -1:
                    raise ValueError(
                        "Unbalanced LaTeX group while sanitizing prose fragment "
                        f"(missing closing '}}' for a reference argument): {content!r}"
                    )
                output.append(text[index : close_index + 1])
                index = close_index + 1
                awaiting_command_argument = False
                pending_raw_argument = False
                continue
            elif not inline_math and char == "{":
                if awaiting_command_argument or latex_brace_depth:
                    output.append(char)
                    latex_brace_depth += 1
                else:
                    output.append(r"\{")
                awaiting_command_argument = False
            elif not inline_math and char == "}":
                if latex_brace_depth:
                    output.append(char)
                    latex_brace_depth -= 1
                else:
                    output.append(r"\}")
                awaiting_command_argument = False
            elif not inline_math and char in {"&", "%", "#", "_", "^"}:
                output.append(
                    {
                        "&": r"\&",
                        "%": r"\%",
                        "#": r"\#",
                        "_": r"\_",
                        "^": r"\textasciicircum{}",
                    }[char]
                )
                awaiting_command_argument = False
                pending_raw_argument = False
            else:
                output.append(char)
                if not char.isspace():
                    awaiting_command_argument = False
                    pending_raw_argument = False
            index += 1
        if latex_brace_depth:
            raise ValueError(
                f"Unbalanced LaTeX group while sanitizing prose fragment "
                f"(missing {latex_brace_depth} closing '}}'): {content!r}"
            )
        return "".join(output)

    def _safe_label(self, label: str, source: str) -> str:
        self._validate_label(label, source)
        self._register_label(label, source)
        return label

    @staticmethod
    def _validate_label(label: str, source: str) -> None:
        if not VALID_LABEL.fullmatch(label):
            raise ValueError(f"Invalid LaTeX label '{label}' from {source}.")

    def _register_fragment_labels(self, fragment: str, source: str) -> None:
        for label_match in LABEL_TOKEN.finditer(fragment):
            self._register_label(label_match.group("label"), source)

    def _register_label(self, label: str, source: str) -> None:
        self._validate_label(label, source)
        if label in self._rendered_labels:
            raise ValueError(f"Duplicate LaTeX label '{label}' from {source}.")
        self._rendered_labels.add(label)

    @staticmethod
    def _safe_float_options(value: Any) -> str:
        options = str(value).strip()
        # `H` requires an extra package and makes placement brittle; the
        # composer deliberately retains only standard LaTeX float positions.
        sanitized = "".join(char for char in options if char in "!htbp")
        return sanitized or "htbp"

    @staticmethod
    def _safe_width(value: Any, source: str) -> str:
        width = str(value).strip().replace(" ", "")
        if not VALID_WIDTH.fullmatch(width):
            raise ValueError(
                f"Invalid width '{value}' from {source}; use a fraction of "
                r"\textwidth, \linewidth, or \columnwidth."
            )
        return width

    def _render_figure(
        self,
        figure_id: str,
        section_id: str,
        blueprint: DocumentBlueprint,
        graph: DocumentGraph,
        sections_dir: Optional[Path] = None,
    ) -> str:
        placement = blueprint.figure_slots[figure_id]
        entry = graph.figure_registry.get(figure_id)
        if entry is None:
            raise ValueError(f"Figure '{figure_id}' is not registered.")
        path = (entry.asset_path or "").replace("\\", "/")
        if (
            not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not VALID_GRAPHICS_PATH.fullmatch(path)
        ):
            raise ValueError(f"Figure '{figure_id}' has an unsafe asset path '{path}'.")
        raw_caption = entry.caption or placement.caption
        if raw_caption and self.enabled_validators.get("markdown_emphasis", True):
            # A figure's caption is planner-authored, not TextAgent-authored,
            # but it is rendered through the exact same plain-text-only
            # _escape_latex as a block's own caption -- so it needs the same
            # rejection, not just ContentBlock.caption (see
            # validate_plain_caption/_validate_fragment above).
            # `sections_dir` is optional (default None, same convention as
            # _validate_fragment) -- a direct unit-test caller skips
            # persistence; the real _render_section call site always
            # supplies it, so this check's cost is recorded too (round-2
            # review finding -- see persist_markdown_emphasis_pass).
            start = time.perf_counter()
            residual = find_markdown_emphasis_matches(
                raw_caption.replace("\r\n", "\n"), apply_literal_math_mask=False
            )
            safe_count = sum(1 for m in residual if not m.ambiguous)
            ambiguous_count = len(residual) - safe_count
            try:
                validate_plain_caption(raw_caption, f"Figure '{figure_id}'")
            finally:
                latency_s = time.perf_counter() - start
                if sections_dir is not None:
                    persist_markdown_emphasis_pass(
                        sections_dir,
                        section_id,
                        phase="assembler_figure_caption_check",
                        block_id=figure_id,
                        original_content=raw_caption,
                        converted_content=None,
                        safe_count=safe_count,
                        ambiguous_count=ambiguous_count,
                        latency_s=latency_s,
                    )
        caption = self._escape_latex(raw_caption or figure_id)
        label = self._safe_label(
            entry.label or placement.label or f"fig:{figure_id}",
            f"figure '{figure_id}'",
        )
        width = self._safe_width(placement.width, f"figure '{figure_id}'")
        # Claim directly before emitting the sole \includegraphics command.
        graph.claim_figure_placement(figure_id, section_id)

        if placement.layout == FigureLayout.MINIPAGE:
            return (
                "\\begin{center}\n"
                f"\\begin{{minipage}}{{{width}}}\n"
                "\\centering\n"
                f"\\includegraphics[width=\\linewidth]{{{path}}}\n"
                f"\\captionof{{figure}}{{{caption}}}\n"
                f"\\label{{{label}}}\n"
                "\\end{minipage}\n"
                "\\end{center}"
            )
        return (
            f"\\begin{{figure}}[{self._safe_float_options(placement.float_options)}]\n"
            "\\centering\n"
            f"\\includegraphics[width={width}]{{{path}}}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\end{figure}"
        )

    def _write_preamble(self, base_dir: Path, blueprint: DocumentBlueprint) -> None:
        document_class = blueprint.document_class.strip()
        if document_class not in SAFE_DOCUMENT_CLASSES:
            raise ValueError(f"Unsupported LaTeX document class: '{document_class}'.")

        options = [option for option in blueprint.document.class_options if re.fullmatch(r"[A-Za-z0-9,=-]+", option)]
        option_text = f"[{','.join(options)}]" if options else ""
        packages = self._required_packages(blueprint)

        lines = [f"\\documentclass{option_text}{{{document_class}}}"]
        for package in PACKAGE_ORDER:
            if package in packages:
                lines.append(f"\\usepackage{{{package}}}")
        if "geometry" in packages:
            lines.append(r"\geometry{margin=1in}")
        if "hyperref" in packages:
            lines.append(r"\hypersetup{hidelinks}")
        if "amsthm" in packages:
            # THEOREM blocks render as \begin{theorem}...\end{theorem} (see
            # _render_theorem); PROOF_SKETCH blocks use amsthm's built-in
            # `proof` environment directly and need no separate declaration.
            lines.append(r"\newtheorem{theorem}{Theorem}[section]")
        if any(
            block.kind == LayoutBlockKind.DEFINITION
            for section in blueprint.sections
            for block in section.blocks
        ):
            # \newtheorem is a core LaTeX2e command, not amsthm-specific, so
            # this doesn't need "amsthm" in packages -- it exists purely to
            # give \refstepcounter{definition} (see _render_semantic_block) a
            # real, [section]-scoped counter to increment, so \label{def:...}
            # on a definition tcolorbox captures an actual "Definition N"
            # instead of silently reusing whatever the nearest other counter
            # (usually the section number) happens to be.
            lines.append(r"\newtheorem{definition}{Definition}[section]")
        if document_class in CHAPTER_FORCES_PAGE_BREAK_CLASSES and any(
            section.level.lower() == "chapter" for section in blueprint.sections
        ):
            # Strip the page break \chapter forces before it starts, so a new
            # chapter/section continues on the current page instead of always
            # opening a fresh one. Numbering and the heading itself are
            # untouched; only the \clearpage/\cleardoublepage call is removed.
            # See CHAPTER_PAGEBREAK_PATCH_TARGET: which macro actually holds
            # that call is class-family-specific, so a mismatch here would
            # silently fail (empty success/failure branches) and leave the
            # page break in place -- the failure branch below turns a future
            # mismatch into a visible compile-log warning instead of that.
            target = CHAPTER_PAGEBREAK_PATCH_TARGET[document_class]
            # \typeout fully expands its argument, so the patch target itself
            # (a control sequence) must never be interpolated in here directly
            # -- TeX would try to *call* it (e.g. \chapter expects an
            # argument) instead of just printing its name. Only the plain
            # document-class string, not any control sequence, goes in.
            failure_warning = (
                rf"\typeout{{document_generation_api WARNING: could not remove "
                rf"the automatic page break before chapter headings in "
                rf"{document_class}; chapter-level sections will still start "
                r"a new page.}"
            )
            lines.append(r"\makeatletter")
            lines.append(
                rf"\patchcmd{{{target}}}{{\if@openright\cleardoublepage\else\clearpage\fi}}"
                rf"{{}}{{}}{{{failure_warning}}}"
            )
            lines.append(r"\makeatother")
        (base_dir / "preamble.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _required_packages(self, blueprint: DocumentBlueprint) -> set[str]:
        packages = {package for package in blueprint.packages if package in SAFE_PACKAGES}
        # amsmath/amssymb are included unconditionally, not just when an
        # EQUATION-kind block is present: inline math (e.g. "$x \in \mathbb{R}$")
        # routinely appears inside paragraph/definition/table blocks too, and
        # \mathbb{}/\mathcal{}/\mathfrak{} specifically require amssymb (amsmath
        # alone does not define them) -- see the "Undefined control sequence"
        # incident this comment accompanies, where \mathbb{R} inside a plain
        # paragraph block failed to compile because amssymb was never loaded.
        packages.update({"graphicx", "hyperref", "cleveref", "amsmath", "amssymb"})
        for section in blueprint.sections:
            for block in section.blocks:
                if block.kind in {
                    LayoutBlockKind.KEY_TAKEAWAY,
                    LayoutBlockKind.DEFINITION,
                    LayoutBlockKind.WARNING,
                    LayoutBlockKind.CASE_STUDY,
                    LayoutBlockKind.COMPARISON,
                } or block.component == LatexComponent.TCOLORBOX:
                    packages.add("tcolorbox")
                if block.kind in {LayoutBlockKind.TABLE, LayoutBlockKind.COMPARISON, LayoutBlockKind.NOTATION_TABLE}:
                    packages.update({"booktabs", "tabularx"})
                if block.kind in {LayoutBlockKind.THEOREM, LayoutBlockKind.PROOF_SKETCH}:
                    packages.add("amsthm")
                if block.component == LatexComponent.TABULARX:
                    packages.add("tabularx")
                elif block.component == LatexComponent.LONGTABLE:
                    packages.add("longtable")
                if block.kind == LayoutBlockKind.LIST:
                    packages.add("enumitem")
        for placement in blueprint.figure_slots.values():
            if placement.layout == FigureLayout.MINIPAGE:
                packages.add("caption")
        if blueprint.layout_policy.get("insert_float_barrier_after_section", False):
            packages.add("placeins")
        if blueprint.document_class in CHAPTER_FORCES_PAGE_BREAK_CLASSES and any(
            section.level.lower() == "chapter" for section in blueprint.sections
        ):
            packages.add("etoolbox")
        return packages

    def _write_front_matter(self, base_dir: Path, blueprint: DocumentBlueprint) -> None:
        document = blueprint.document
        items = {item.value for item in document.include_front_matter}
        lines: List[str] = []
        if document.title and document.kind != DocumentKind.SECTION_DRAFT:
            title = self._escape_latex(document.title)
            if document.subtitle:
                title += r"\\[0.5em]\large " + self._escape_latex(document.subtitle)
            lines.extend(
                [
                    f"\\title{{{title}}}",
                    f"\\author{{{self._escape_latex(document.author or '')}}}",
                    f"\\date{{{document.date}}}",
                    r"\maketitle",
                ]
            )
        if "table_of_contents" in items:
            lines.append(r"\tableofcontents")
        if "list_of_figures" in items:
            lines.append(r"\listoffigures")
        if "list_of_tables" in items:
            lines.append(r"\listoftables")
        (base_dir / "frontmatter.tex").write_text("\n\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_main_tex(base_dir: Path, blueprint: DocumentBlueprint) -> None:
        lines = [r"\input{preamble}", r"\begin{document}", r"\input{frontmatter}"]
        lines.extend(rf"\input{{sections/{section.section_id}}}" for section in blueprint.sections)
        lines.append(r"\end{document}")
        (base_dir / "main.tex").write_text("\n\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _escape_latex(value: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(char, char) for char in value)

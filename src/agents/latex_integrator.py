"""Blueprint-driven LaTeX document composer and compiler."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
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
from src.evaluation.compile_log_parser import parse_compile_log
from src.evaluation.event_logger import EventLogger
from src.evaluation.schemas import CompileResult


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
        "matrix",
        "pmatrix",
        "bmatrix",
        "vmatrix",
        "Vmatrix",
        "smallmatrix",
    },
}
ENVIRONMENT_TOKEN = re.compile(r"\\(?P<action>begin|end)\s*\{(?P<name>[A-Za-z*]+)\}")


def validate_fragment_environments(block_type: SemanticBlockType, content: str, block_id: Optional[str]) -> str:
    """Reject a LaTeX command/environment a block's ``type`` isn't allowed to use.

    Pulled out of ``LatexIntegratorAgent._validate_fragment`` so TextAgent can
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

    allowed = ALLOWED_ENVIRONMENTS.get(block_type, set())
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


def validate_plain_title(title: Optional[str], source: str) -> None:
    """Reject a backslash (a LaTeX command) in a title-like field.

    Titles (document/section/block titles, table & figure captions) are
    rendered through ``LatexIntegratorAgent._escape_latex`` -- a blunt,
    command-blind escaper that assumes the string is plain text and turns
    every backslash into a literal, visible "\\textbackslash{}". A title
    that legitimately needs math notation (``\\pi``, ``\\theta``, ...) has no
    way to survive that; the fix is to catch it here, before rendering,
    rather than let it print as mangled text with no error at all.

    A bare ``$`` (e.g. a dollar figure in a finance-domain title like "$500
    Cost Overrun") is deliberately not flagged here: ``_escape_latex``
    already turns it into a literal, correctly-rendered dollar sign, so it
    isn't actually broken -- only a backslash command is.
    """
    if title and _TITLE_UNSAFE_CHAR.search(title):
        raise ValueError(
            f"{source} title {title!r} contains a LaTeX command (backslash), which "
            "renders as literal text, not a LaTeX command -- titles must be plain text."
        )


LABEL_TOKEN = re.compile(r"\\label\s*\{(?P<label>[^{}]+)\}")
VALID_LABEL = re.compile(r"[A-Za-z0-9:._-]+")
# A '|' not preceded by a backslash: the actual column delimiter in a Markdown
# table row, as opposed to a literal pipe character a model escaped as '\|'
# inside a cell (for example conditional-probability notation it failed to
# write as '\mid' -- see the table-formatting rules in text_agent.SYSTEM_PROMPT).
# An unescaped raw '|' inside a cell is the most common cause of a ragged
# table, since str.split("|") alone treats it as an extra column boundary.
UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
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


def _split_markdown_row(row: str) -> List[str]:
    """Split one Markdown table row on unescaped '|', restoring '\\|' to a
    literal '|' in each cell -- see UNESCAPED_PIPE for why this must not be
    a plain ``row.split("|")``."""
    return [cell.replace(r"\|", "|").strip() for cell in UNESCAPED_PIPE.split(row)]


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


class LatexIntegratorAgent:
    """Compose a document only from its registered blueprint and assets."""

    def __init__(self, llm_client: Any = None, engine: str = "tectonic", runs: int = 2):
        # Kept for constructor compatibility. Layout decisions are deterministic
        # once the planner has produced a DocumentBlueprint.
        self.llm = llm_client
        self.engine = engine
        self.runs = runs

    def integrate(
        self,
        request_id: str,
        graph: DocumentGraph,
        output_dir: str = "outputs/generations",
        event_logger: Optional[EventLogger] = None,
    ) -> bool:
        """Write a complete LaTeX project and compile it when an engine exists."""
        if graph.blueprint is None:
            raise ValueError("Latex integration requires a DocumentBlueprint on the graph.")

        blueprint = graph.blueprint

        # Layout policy is decided entirely by the blueprint, independent of any
        # section content, so it can be checked before doing any file I/O.
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
        figure_use_errors = self.validate_figure_single_use(graph, rendered_sections)
        _log_list_check(
            event_logger, verifier="figure_single_use_validator", errors=figure_use_errors,
            artifact_id=request_id, signal_type="figure_single_use_violation",
        )
        reference_errors = self.validate_references(rendered_sections)
        _log_list_check(
            event_logger, verifier="reference_validator", errors=reference_errors,
            artifact_id=request_id, signal_type="undefined_reference",
        )
        frame_errors = self.validate_document_frame(base_dir, blueprint)
        _log_list_check(
            event_logger, verifier="document_frame_validator", errors=frame_errors,
            artifact_id=request_id, signal_type="document_frame_incomplete",
        )
        validation_errors: List[str] = figure_use_errors + reference_errors + frame_errors
        if validation_errors:
            raise ValueError(
                "Pre-compile validation failed:\n- " + "\n- ".join(validation_errors)
            )

        return self.compile_pdf(request_id, output_dir, event_logger=event_logger)

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
            rendered.append(self._render_semantic_block(layout_block, content_block))

            for figure_id in figure_after.pop(layout_block.id, []):
                rendered.append(self._render_figure(figure_id, layout.section_id, blueprint, graph))

        # Slots without an explicit anchor belong at the end of their owner section.
        for figure_ids in figure_after.values():
            for figure_id in figure_ids:
                rendered.append(self._render_figure(figure_id, layout.section_id, blueprint, graph))

        if blueprint.layout_policy.get("insert_float_barrier_after_section", False):
            rendered.append(r"\FloatBarrier")
        return "\n\n".join(part for part in rendered if part.strip()) + "\n"

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
            self._validate_fragment(content_block)

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

    def _validate_fragment(self, block: ContentBlock) -> None:
        """Reject LaTeX that would escape the semantic-block rendering boundary."""
        content = validate_fragment_environments(block.type, block.content, block.block_id)
        validate_no_manual_labels(block.type, content, block.block_id)
        if block.type in (SemanticBlockType.TABLE, SemanticBlockType.NOTATION_TABLE, SemanticBlockType.COMPARISON):
            validate_markdown_table_columns(content, block.block_id)
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
        caption = self._escape_latex(entry.caption or placement.caption or figure_id)
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

    def _resolve_engine(self) -> tuple[Optional[str], List[str]]:
        if self.engine:
            explicit = shutil.which(self.engine)
            if explicit:
                return (explicit, []) if "tectonic" in Path(explicit).name else (explicit, ["-interaction=nonstopmode", "-halt-on-error"])
        tectonic = shutil.which("tectonic")
        if tectonic:
            return tectonic, []
        env_bin = os.environ.get("PDFLATEX_BIN")
        if env_bin:
            resolved = shutil.which(env_bin) or (env_bin if Path(env_bin).is_file() else None)
            if resolved:
                return str(resolved), ["-interaction=nonstopmode", "-halt-on-error"]
        pdflatex = shutil.which("pdflatex")
        if pdflatex:
            return pdflatex, ["-interaction=nonstopmode", "-halt-on-error"]
        return None, []

    def _finalize_compile_result(
        self,
        event_logger: Optional[EventLogger],
        base_dir: Path,
        *,
        compile_success: bool,
        engine: Optional[str],
        return_code: Optional[int],
        log_text: str,
        first_error_type_override: Optional[str] = None,
    ) -> None:
        """Build, persist (``compile_result.json``), and log one CompileResult --
        the single place every compile_pdf() return point converges through.
        A no-op when no event_logger is supplied, so a caller that never
        passes one (for example ``single_llm.py``'s direct ``compile_pdf``
        call) sees no new files and no behavior change.
        """
        if event_logger is None:
            return
        if engine:
            fatal_error_count, warning_count, first_error_type = parse_compile_log(log_text, engine)
        else:
            fatal_error_count, warning_count, first_error_type = 0, 0, None
        first_error_type = first_error_type_override or first_error_type
        result = CompileResult(
            compile_success=compile_success,
            engine=engine,
            return_code=return_code,
            fatal_error_count=fatal_error_count,
            warning_count=warning_count,
            first_error_type=first_error_type,
            pdf_exists=(base_dir / "main.pdf").exists(),
        )
        event_logger.compile_result = result
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            (base_dir / "compile_result.json").write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        event_logger.log_verification(
            stage="compile",
            verifier="latex_compiler",
            artifact_id=base_dir.name,
            attempt=1,
            result="pass" if compile_success else "fail",
            signal_type=None if compile_success else first_error_type,
            message=None if compile_success else (log_text[-2000:] or first_error_type_override),
        )

    def compile_pdf(
        self,
        request_id: str,
        output_dir: str = "outputs/generations",
        event_logger: Optional[EventLogger] = None,
    ) -> bool:
        base_dir = Path(output_dir) / request_id
        main_tex_path = base_dir / "main.tex"
        if not main_tex_path.exists():
            print(f"[Integrator Error] main.tex not found at {main_tex_path}")
            self._finalize_compile_result(
                event_logger, base_dir, compile_success=False, engine=None, return_code=None,
                log_text="", first_error_type_override="main_tex_missing",
            )
            return False

        engine_bin, mode_args = self._resolve_engine()
        if engine_bin is None:
            print("[Integrator Error] No LaTeX engine found (tried configured engine, tectonic, pdflatex).")
            self._finalize_compile_result(
                event_logger, base_dir, compile_success=False, engine=None, return_code=None,
                log_text="", first_error_type_override="no_latex_engine_found",
            )
            return False

        engine_name = Path(engine_bin).name
        runs = 1 if "tectonic" in engine_name else self.runs
        log_path = base_dir / "compile.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            for _ in range(runs):
                result = subprocess.run(
                    [engine_bin, *mode_args, "main.tex"],
                    cwd=base_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if result.returncode != 0:
                    print(f"[Integrator Warning] Compilation failed. Check {log_path}.")
                    self._finalize_compile_result(
                        event_logger, base_dir, compile_success=False, engine=engine_name,
                        return_code=result.returncode,
                        log_text=log_path.read_text(encoding="utf-8", errors="replace"),
                    )
                    return False
        pdf_exists = (base_dir / "main.pdf").exists()
        self._finalize_compile_result(
            event_logger, base_dir, compile_success=pdf_exists, engine=engine_name,
            return_code=result.returncode,
            log_text=log_path.read_text(encoding="utf-8", errors="replace"),
        )
        return pdf_exists

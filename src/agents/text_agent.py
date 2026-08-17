"""Semantic content generation aligned with a DocumentBlueprint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from scripts.llm_client import LLMClient
from src.agents.latex_assembler import (
    ENVIRONMENT_TOKEN,
    is_markdown_table,
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
from src.common.state import DocumentGraph, LayoutBlock, LayoutBlockKind, NodeStatus, SectionLayout
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
        # Covers _validate_blocks_against_layout's own raises (id/order
        # mismatch, not-a-JSON-array, ...) and SemanticBlockErrors (the
        # per-block content violations collected from validate_fragment_
        # environments/validate_no_manual_labels/validate_markdown_table_
        # columns/validate_plain_title/figure checks) -- the dominant
        # failure mode here.
        return "content_validation_error"
    return "generation_error"


class SemanticBlockError(ValueError):
    """One block's content violates a specific, named contract rule.

    ``code`` is a stable key into ``REMEDIATION_HINTS`` and into
    ``TextAgent.enabled_validators`` -- it is what lets a retry attempt (a)
    target a deterministic, rule-specific fix instead of a generic "try
    again", and (b) let an ablation run disable one named check without
    touching the others.
    """

    def __init__(self, code: str, block_id: Optional[str], message: str) -> None:
        super().__init__(message)
        self.code = code
        self.block_id = block_id


class SemanticBlockErrors(ValueError):
    """Aggregates every block-level failure from one validation pass.

    Also carries every ``ContentBlock`` that *did* parse (``blocks``, both
    passing and failing) so a caller can keep the blocks that passed instead
    of discarding a whole attempt over one bad block -- see
    ``TextAgent.process_node``'s progressive narrowing.
    """

    def __init__(self, blocks: List["ContentBlock"], errors: List[SemanticBlockError]) -> None:
        self.blocks = blocks
        self.errors = errors
        summary = "; ".join(f"[{error.block_id}:{error.code}] {error}" for error in errors)
        super().__init__(f"{len(errors)} block(s) failed validation: {summary}")


# Every entry here is a *content-quality* verifier that TextAgent runs on top
# of the structural contract (block id/order, required type, non-empty
# content -- always enforced, never toggleable, since the composer's
# single-sole-input contract depends on them and disabling them would break
# the pipeline rather than test anything). Meant to be flipped off per-code
# for verifier ablation experiments, e.g. TextAgent(enabled_validators=
# {"table_columns": False}) to measure how much that one check actually
# contributes to recovery/compile success.
DEFAULT_ENABLED_VALIDATORS: Dict[str, bool] = {
    "forbidden_environment": True,
    "manual_label": True,
    "table_columns": True,
    "plain_title": True,
    "figure_content": True,
    "unicode_math_symbol": True,
    "bare_math_notation": True,
    "table_shape": True,
    "equation_shape": True,
}

# A raw Unicode Greek letter or math symbol renders correctly by pure luck
# (most PDF viewers happen to have the glyph), but breaks the moment the
# document needs the symbol to behave like math (spacing, italics, no
# hyphenation, kerning against subscripts) -- and breaks outright under a
# strict LaTeX engine without full Unicode/fontspec support. Every key here
# is a character _validate_one_block rejects; every value is the LaTeX
# command REMEDIATION_HINTS tells the model to use instead.
UNICODE_MATH_SYMBOLS: Dict[str, str] = {
    "ε": r"\epsilon", "δ": r"\delta", "α": r"\alpha", "β": r"\beta", "γ": r"\gamma",
    "θ": r"\theta", "λ": r"\lambda", "μ": r"\mu", "π": r"\pi", "σ": r"\sigma",
    "φ": r"\phi", "ψ": r"\psi", "ω": r"\omega", "χ": r"\chi", "ξ": r"\xi",
    "ζ": r"\zeta", "η": r"\eta", "ι": r"\iota", "κ": r"\kappa", "ν": r"\nu",
    "ρ": r"\rho", "τ": r"\tau", "υ": r"\upsilon",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Ξ": r"\Xi",
    "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx", "∈": r"\in",
    "∉": r"\notin", "⊂": r"\subset", "⊆": r"\subseteq", "∪": r"\cup", "∩": r"\cap",
    "∀": r"\forall", "∃": r"\exists", "∞": r"\infty", "∂": r"\partial",
    "∇": r"\nabla", "∑": r"\sum", "∏": r"\prod", "√": r"\sqrt", "±": r"\pm",
    "×": r"\times", "÷": r"\div", "→": r"\to", "⇒": r"\Rightarrow", "·": r"\cdot",
}

# Command names that only mean something inside math mode -- used to detect a
# math expression left bare in prose (no surrounding $...$). Deliberately
# narrow (math-only macros, not '^'/'_'): those two characters have a
# legitimate bare-prose meaning too (a literal underscore in a filename, for
# instance) that LatexAssembler._sanitize_prose already escapes safely
# outside math mode, so flagging them here would false-positive on prose that
# was never broken.
MATH_ONLY_COMMANDS = frozenset({
    "epsilon", "varepsilon", "delta", "alpha", "beta", "gamma", "theta", "vartheta",
    "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "varsigma", "tau", "upsilon",
    "phi", "varphi", "psi", "omega", "chi", "zeta", "eta", "iota", "kappa",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "Omega",
    "leq", "geq", "neq", "approx", "sim", "equiv", "in", "notin", "subset", "subseteq",
    "cup", "cap", "forall", "exists", "infty", "partial", "nabla", "sum", "prod",
    "int", "sqrt", "frac", "to", "rightarrow", "leftarrow", "Rightarrow", "Leftarrow",
    "cdot", "times", "div", "pm", "mp", "mid", "mathbb", "mathcal", "binom",
    "lim", "sup", "inf",
})


def _validate_unicode_math_symbols(content: str, block_id: Optional[str]) -> None:
    found = sorted({char for char in content if char in UNICODE_MATH_SYMBOLS})
    if found:
        replacements = ", ".join(f"'{char}' -> '{UNICODE_MATH_SYMBOLS[char]}'" for char in found)
        raise ValueError(
            f"Block '{block_id}' uses raw Unicode math symbol(s) instead of LaTeX "
            f"commands: {replacements}."
        )


def _validate_bare_math_commands(content: str, block_id: Optional[str]) -> None:
    """Reject a math-only LaTeX command (e.g. ``\\epsilon``) written outside
    ``$...$``/``\\(...\\)``/``\\[...\\]``.

    Mirrors LatexAssembler._sanitize_prose's own $/\\(\\)/\\[\\] state
    tracking (including treating a ``$$`` run as one toggle, not two) so this
    check agrees with how the composer will actually interpret math mode --
    a command's own ``{...}`` argument is skipped rather than scanned, since
    it is typically an identifier (e.g. \\mathbb{R}'s argument), not prose.
    """
    text = content.replace("\r\n", "\n")
    length = len(text)
    index = 0
    in_math = False
    offenders: List[str] = []
    while index < length:
        char = text[index]
        if char == "\\":
            if index + 1 < length and text[index + 1].isalpha():
                end = index + 2
                while end < length and text[end].isalpha():
                    end += 1
                command_name = text[index + 1 : end]
                if not in_math and command_name in MATH_ONLY_COMMANDS:
                    offenders.append(f"\\{command_name}")
                index = end
                if index < length and text[index] == "{":
                    depth = 0
                    while index < length:
                        if text[index] == "{":
                            depth += 1
                        elif text[index] == "}":
                            depth -= 1
                            if depth == 0:
                                index += 1
                                break
                        index += 1
                continue
            if index + 1 < length and text[index : index + 2] in (r"\(", r"\["):
                in_math = True
                index += 2
                continue
            if index + 1 < length and text[index : index + 2] in (r"\)", r"\]"):
                in_math = False
                index += 2
                continue
            index += 2
            continue
        if char == "$":
            if index + 1 < length and text[index + 1] == "$":
                in_math = not in_math
                index += 2
                continue
            in_math = not in_math
            index += 1
            continue
        index += 1
    if offenders:
        unique = sorted(set(offenders))
        raise ValueError(
            f"Block '{block_id}' has math command(s) outside '$...$' (or "
            f"'\\(...\\)'/'\\[...\\]'): {', '.join(unique)}. Wrap the whole "
            "expression in math delimiters."
        )


# table_columns (above) only checks a table's *internal* correctness (every
# row has the header's column count) once it's already recognized as a
# table -- it silently no-ops on content that isn't table-shaped at all
# (see is_markdown_table). Nothing used to catch that case: a `table`/
# `notation_table` block containing plain prose passed every existing check
# and reached LatexAssembler, which silently wrapped the whole prose blob as
# a fake one-cell table (_single_column_table) instead of failing anywhere.
# Symmetrically, `bare_math_notation` is explicitly skipped for `equation`
# blocks (their content IS the math body, no $...$ wrapper needed), which
# left an equation block with literally no math notation at all -- plain
# English prose -- completely unchecked; LatexAssembler._render_equation
# then wrapped it verbatim, unescaped, inside \begin{equation}...\end{equation}.
# Deliberately narrow: chars that essentially never appear in ordinary
# English prose but are near-universal in even a bare (undelimited) formula
# like "E = mc^2" -- this must not reject a legitimate bare equation body
# just because it has no '$'.
_EQUATION_MATH_SIGNAL_CHARS = "=^_$\\"


def _validate_component_content_shape(
    kind: LayoutBlockKind, content: str, block_id: Optional[str]
) -> None:
    if kind in (LayoutBlockKind.TABLE, LayoutBlockKind.NOTATION_TABLE):
        if not (is_markdown_table(content) or ENVIRONMENT_TOKEN.search(content)):
            raise ValueError(
                f"Block '{block_id}' is a table/notation_table block but its content "
                "isn't table-shaped -- write a Markdown pipe table (with a header "
                "separator row) or a tabular/tabularx/longtable fragment, not prose."
            )
    elif kind == LayoutBlockKind.EQUATION:
        if not any(char in content for char in _EQUATION_MATH_SIGNAL_CHARS):
            raise ValueError(
                f"Block '{block_id}' is an equation block but its content has no math "
                "notation ('=', '^', '_', '$', or a LaTeX command) -- write an actual "
                "mathematical expression, not a prose description of one."
            )


# One deterministic, rule-specific remediation instruction per SemanticBlockError
# code. Looked up by _build_prompt_from_node and injected next to that block's
# own error message, instead of the one hard-coded, equation-shaped hint every
# failure used to get regardless of what actually went wrong.
REMEDIATION_HINTS: Dict[str, str] = {
    "type_mismatch": (
        "Set \"type\" to exactly the required_type listed for this block_id -- "
        "do not substitute a different (even related) type."
    ),
    "empty_content": '"content" must not be blank -- write the actual block content.',
    "forbidden_environment": (
        "Use only the LaTeX commands/environments this block type allows (see "
        "<Content rules by block type>). Prose-only types (paragraph, "
        "key_takeaway, warning, case_study, definition, theorem, proof_sketch) "
        "may not contain a numbered/labeled display-math or box environment, "
        "even if the instructions ask for something \"numbered\" or \"labeled\" -- "
        "write it as inline math ($...$) instead; a numbered, cross-referenceable "
        "version belongs in a dedicated `equation` block only. Exception: matrix "
        "notation (matrix, pmatrix, bmatrix, vmatrix, Vmatrix, smallmatrix) is "
        "allowed directly in any block type's content, inside $...$ -- it is not "
        "numbered/labeled, so it does not need a dedicated `equation` block."
    ),
    "manual_label": (
        "Do not write your own \\label{...}. Cite one of the exact strings from "
        "\"Available cross-reference labels\" instead of inventing a new label."
    ),
    "table_columns": (
        "Every row of the Markdown table, including the header separator row, "
        "must have the same number of '|'-delimited columns as the header. A "
        "raw '|' inside a cell is read as an extra column -- use '\\mid' for a "
        "conditional-probability or set-builder bar, or '\\|' for a literal pipe."
    ),
    "table_shape": (
        "Rewrite \"content\" as an actual table: a Markdown pipe table with a "
        "header row, a '|---|---|' separator row, and data rows -- or a "
        "tabular/tabularx/longtable fragment. Do not describe the table in prose."
    ),
    "equation_shape": (
        "Rewrite \"content\" as an actual mathematical expression (using '=', "
        "'^', '_', LaTeX commands like '\\frac{}{}', etc.), not a prose "
        "description of one. No '$...$' wrapper is needed -- the composer adds it."
    ),
    "plain_title": (
        '"title" is plain text only -- remove the backslash/LaTeX command from '
        'it. Put math notation in "content" instead, inside $...$.'
    ),
    "figure_asset_mismatch": (
        "This figure_reference block's \"asset_id\" must be exactly the "
        "asset_id given for this block_id in the request -- do not invent or "
        "guess a different one."
    ),
    "figure_forbidden_content": (
        "A figure_reference block must not contain \\begin{figure} or "
        "\\includegraphics -- reference the figure in a plain sentence only; "
        "the Composer places the image."
    ),
    "unicode_math_symbol": (
        "Replace every raw Unicode Greek letter/math symbol named in the error "
        "with its LaTeX command (e.g. 'ε' -> '\\epsilon', 'δ' -> '\\delta'), "
        "written inside $...$."
    ),
    "bare_math_notation": (
        "Wrap the math command(s) named in the error -- and the full expression "
        "they belong to -- in '$...$' (for example '$\\epsilon > 0$', not "
        "'\\epsilon > 0')."
    ),
    "compile_error": (
        "This exact block failed to compile as LaTeX with the error shown above. "
        "Rewrite \"content\" to fix that specific syntax problem (for example, "
        "close every '$...$'/environment you open, and never leave a math command "
        "outside '$...$') while preserving the block's meaning -- do not change "
        "what it says, only the broken LaTeX syntax."
    ),
}

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

<Math notation>
- Never write a raw Unicode Greek letter or math symbol (e.g. ε, δ, α, θ, λ,
  π, σ, ≤, ≥, ∈, →, ∞). Always use the equivalent LaTeX command instead
  (`\epsilon`, `\delta`, `\alpha`, `\theta`, `\lambda`, `\pi`, `\sigma`,
  `\leq`, `\geq`, `\in`, `\to`, `\infty`, ...).
- Every math expression must be wrapped in delimiters: `$...$` for inline
  math in prose, or the block's own math body for an `equation` block. Never
  write a LaTeX math command or expression as bare text outside `$...$`
  (wrong: "for every \epsilon > 0"; right: "for every $\epsilon > 0$").
- Matrix notation (`matrix`, `pmatrix`, `bmatrix`, `vmatrix`, `Vmatrix`,
  `smallmatrix`) is allowed inside `$...$` in any block type's content, not
  only `equation` -- it is not numbered/labeled, so it never needs its own
  dedicated `equation` block.

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

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        enabled_validators: Optional[Dict[str, bool]] = None,
    ) -> None:
        self.llm = llm_client or LLMClient()
        unknown = set(enabled_validators or {}) - set(DEFAULT_ENABLED_VALIDATORS)
        if unknown:
            raise ValueError(
                f"Unknown enabled_validators code(s): {sorted(unknown)}. "
                f"Valid codes: {sorted(DEFAULT_ENABLED_VALIDATORS)}."
            )
        self.enabled_validators: Dict[str, bool] = {
            **DEFAULT_ENABLED_VALIDATORS,
            **(enabled_validators or {}),
        }

    def process_node(
        self,
        node_id: str,
        graph: DocumentGraph,
        request_id: str,
        output_dir: str = "outputs/generations",
        max_attempts: int = 3,
        event_logger: Optional[EventLogger] = None,
        usage_sink: Optional[list] = None,
        outer_iteration: Optional[int] = None,
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

        Each attempt only asks the model to (re)generate whatever is still
        ``pending`` -- every other node's block starts out pending, but once
        a block passes validation it is accepted and never regenerated
        again. A structural failure (bad JSON, wrong ids/order/count for
        what was actually requested) can't be pinned on one block, so it
        leaves the whole pending set unchanged for a full retry; a
        ``SemanticBlockErrors`` failure narrows pending down to exactly the
        blocks that are still wrong, keeps the ones that passed, and gives
        the next attempt a per-block, rule-specific fix instruction instead
        of one generic notice for the whole section.
        """
        def _log_precondition_failure(exc: Exception) -> None:
            # node-lookup/layout-lookup failures happen before any LLM
            # attempt -- there is nothing to retry (no attempt tracker
            # exists yet, and resampling TextAgent's own LLM can't fix a
            # blueprint that has no layout blocks for this node), so this
            # can't go through AttemptTracker's retry-with-feedback pairing.
            # Without this, such a failure was invisible to the evaluation
            # layer entirely -- it propagated straight past every checked_
            # count/triggered_count in Verification Trigger Rate. Logged
            # with failure_id=None, the same convention used for other
            # failures with no recovery path (see EventLogger.new_failure_id).
            if event_logger is not None:
                event_logger.log_verification(
                    stage="text_agent",
                    verifier="text_semantic_block_validator",
                    artifact_id=node_id,
                    section_id=node_id,
                    attempt=0,
                    outer_iteration=outer_iteration,
                    result="fail",
                    signal_type=_classify_generation_error(exc),
                    message=str(exc),
                    producer_model=self.llm.model_name,
                    failure_id=None,
                )

        node = graph.nodes.get(node_id)
        if node is None:
            error = ValueError(f"Node '{node_id}' not found in graph.")
            _log_precondition_failure(error)
            raise error

        try:
            layout = self._section_layout_for_node(node_id, graph)
        except Exception as exc:
            _log_precondition_failure(exc)
            graph.update_node_status(node_id, NodeStatus.ERROR, error=str(exc))
            raise

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
            outer_iteration=outer_iteration,
        )

        layout_blocks = layout.blocks
        return self._run_generation_loop(
            node,
            graph,
            node_id,
            request_id,
            sections_dir,
            layout_blocks,
            pending_ids={block.id for block in layout_blocks},
            accepted={},
            block_errors={},
            max_attempts=max_attempts,
            tracker=tracker,
            usage_sink=usage_sink,
        )

    def regenerate_blocks(
        self,
        node_id: str,
        block_id: str,
        error_message: str,
        graph: DocumentGraph,
        request_id: str,
        output_dir: str = "outputs/generations",
        max_attempts: int = 2,
        event_logger: Optional[EventLogger] = None,
        usage_sink: Optional[list] = None,
        outer_iteration: Optional[int] = None,
    ) -> Optional[TextAgentOutput]:
        """Regenerate exactly one block of an already-drafted section.

        Called by the compile-repair loop (``run_full_pipeline.
        DocGenPipeline._repair_and_recompile``) after a compile failure is
        attributed to one rendered block via its
        ``% <layout-anchor:...>`` comment (see
        ``LatexAssembler.locate_block_at_line``), instead of redrafting the
        whole section the way ``process_node`` does. Every other block is
        loaded from the section's persisted ``.blocks.json`` and kept
        exactly as-is; only ``block_id`` re-enters ``_run_generation_loop``,
        pre-seeded as already-failing (via a ``SemanticBlockError`` carrying
        the compiler's own message) so the model sees that message on its
        first attempt instead of starting blind. Raises if the section has
        never been drafted yet -- there is nothing to regenerate *from*.
        """
        node = graph.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node '{node_id}' not found in graph.")
        layout = self._section_layout_for_node(node_id, graph)
        if block_id not in {block.id for block in layout.blocks}:
            raise ValueError(f"Block '{block_id}' is not part of section '{node_id}'.")

        sections_dir = Path(output_dir) / request_id / "sections"
        persisted = self._read_persisted_output(sections_dir, node_id)
        existing = {block.block_id: block for block in persisted.blocks}
        if block_id not in existing:
            raise ValueError(
                f"Block '{block_id}' has no persisted content for section '{node_id}' yet."
            )

        graph.update_node_status(node_id, NodeStatus.RUNNING)
        tracker = AttemptTracker(
            event_logger,
            stage="text_agent",
            verifier="text_semantic_block_validator",
            recovery_action="retry_with_feedback",
            artifact_id=node_id,
            section_id=node_id,
            block_id=block_id,
            producer_model=self.llm.model_name,
            outer_iteration=outer_iteration,
        )

        accepted = {bid: block for bid, block in existing.items() if bid != block_id}
        block_errors = {block_id: SemanticBlockError("compile_error", block_id, error_message)}
        return self._run_generation_loop(
            node,
            graph,
            node_id,
            request_id,
            sections_dir,
            layout.blocks,
            pending_ids={block_id},
            accepted=accepted,
            block_errors=block_errors,
            max_attempts=max_attempts,
            tracker=tracker,
            usage_sink=usage_sink,
            initial_used_tools=persisted.used_tools,
        )

    def _run_generation_loop(
        self,
        node: Any,
        graph: DocumentGraph,
        node_id: str,
        request_id: str,
        sections_dir: Path,
        layout_blocks: List[LayoutBlock],
        pending_ids: set,
        accepted: Dict[str, ContentBlock],
        block_errors: Dict[str, SemanticBlockError],
        max_attempts: int,
        tracker: AttemptTracker,
        usage_sink: Optional[list],
        initial_used_tools: Optional[List[ToolResponse]] = None,
    ) -> Optional[TextAgentOutput]:
        """Shared attempt loop behind both ``process_node`` (fresh section
        draft, every block starts pending) and ``regenerate_blocks`` (an
        already-drafted section, only the compiler-flagged block pending,
        every other block pre-seeded into ``accepted`` from disk) -- narrows
        ``pending_ids`` down on a ``SemanticBlockErrors`` failure exactly the
        same way either way.
        """
        last_error: Optional[str] = None
        used_tools: List[ToolResponse] = list(initial_used_tools) if initial_used_tools else []

        for attempt in range(1, max_attempts + 1):
            pending_layout_blocks = [block for block in layout_blocks if block.id in pending_ids]
            try:
                raw_response = self.llm.generate(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=self._build_prompt_from_node(
                        node,
                        graph,
                        pending_layout_blocks,
                        accepted_blocks=accepted,
                        block_errors=block_errors,
                        previous_error=last_error,
                    ),
                    temperature=0.3,
                    max_new_tokens=4096,
                    usage_sink=usage_sink,
                )
                data = json.loads(self.llm.extract_json_block(raw_response))
                new_blocks = self._validate_blocks_against_layout(
                    data.get("content_blocks"), pending_layout_blocks
                )
                used_tools.extend(self._execute_tools(data.get("tool_requests", [])))
                for block in new_blocks:
                    accepted[block.block_id] = block
                pending_ids = set()
                block_errors = {}
            except SemanticBlockErrors as exc:
                failing_ids = {error.block_id for error in exc.errors}
                block_errors = {error.block_id: error for error in exc.errors}
                for block in exc.blocks:
                    if block.block_id not in failing_ids:
                        accepted[block.block_id] = block
                pending_ids = failing_ids
                last_error = str(exc)
                tracker.record_failure(attempt, _classify_generation_error(exc), str(exc))
                print(
                    f"[TextAgent] Attempt {attempt}/{max_attempts} for '{node_id}': "
                    f"{len(failing_ids)} block(s) still failing: {sorted(failing_ids)}"
                )
            except Exception as exc:
                # Can't be attributed to a specific block (bad JSON, wrong
                # ids/order/count for what was requested, tool failure, ...),
                # so the whole pending set is retried unchanged next attempt.
                last_error = str(exc)
                block_errors = {}
                tracker.record_failure(attempt, _classify_generation_error(exc), str(exc))
                print(f"[TextAgent] Attempt {attempt}/{max_attempts} failed for '{node_id}': {exc}")

            if not pending_ids:
                ordered_blocks = [accepted[block.id] for block in layout_blocks]
                output = TextAgentOutput(
                    request_id=request_id,
                    section_id=node_id,
                    blocks=ordered_blocks,
                    used_tools=used_tools,
                )
                self._write_semantic_artifacts(sections_dir, node_id, output)

                draft_content = self._render_draft_with_anchors(ordered_blocks)
                graph.update_node_status(node_id, NodeStatus.DRAFTED, content=draft_content)
                tracker.record_success(attempt, resulting_artifact_id=f"{node_id}.blocks.json")
                print(f"[TextAgent] Saved semantic blocks for section: {node_id}")
                return output

        tracker.record_exhausted()
        graph.update_node_status(node_id, NodeStatus.ERROR, error=last_error)
        print(f"[TextAgent] Semantic block generation failed for '{node_id}' after {max_attempts} attempts: {last_error}")
        return None

    @staticmethod
    def _section_layout_for_node(node_id: str, graph: DocumentGraph) -> SectionLayout:
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
        return layout

    @staticmethod
    def _read_persisted_output(sections_dir: Path, node_id: str) -> TextAgentOutput:
        blocks_path = sections_dir / f"{node_id}.blocks.json"
        if not blocks_path.is_file():
            raise FileNotFoundError(
                f"No drafted output found for '{node_id}' at '{blocks_path}' -- "
                "regenerate_blocks() requires the section to already be drafted."
            )
        return TextAgentOutput.model_validate(json.loads(blocks_path.read_text(encoding="utf-8")))

    def _build_prompt_from_node(
        self,
        node: Any,
        graph: DocumentGraph,
        layout_blocks: List[LayoutBlock],
        accepted_blocks: Optional[Dict[str, ContentBlock]] = None,
        block_errors: Optional[Dict[str, SemanticBlockError]] = None,
        previous_error: Optional[str] = None,
    ) -> str:
        accepted_blocks = accepted_blocks or {}
        block_errors = block_errors or {}

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

        if block_errors:
            # Attributable failures: each block gets its own error text plus
            # the deterministic fix for exactly that rule, instead of one
            # generic notice covering every possible cause at once.
            hint_lines = [
                f"- Block '{block_id}': {error}\n"
                f"  How to fix: {REMEDIATION_HINTS.get(error.code, 'Fix this specific problem.')}"
                for block_id, error in block_errors.items()
            ]
            retry_notice = (
                "The block(s) below were rejected in your previous attempt. "
                "Regenerate ONLY these blocks (listed in 'Required semantic "
                "blocks' below), fixing exactly the problem described for "
                "each -- do not change anything else about them:\n"
                + "\n".join(hint_lines)
                + "\n\n"
            )
        elif previous_error:
            # Structural failure: couldn't be pinned on one block, so this is
            # the same whole-set retry as before, with one general hint.
            retry_notice = (
                f"Your previous attempt for this section was rejected: {previous_error}\n"
                "Fix this specific problem in your next response. If a block's "
                "required_type only allows prose (paragraph, key_takeaway, "
                "definition, warning, case_study, theorem, proof_sketch), do "
                "not wrap a value in a numbered/labeled display-equation "
                "environment there even if the instructions ask for one to be "
                '"numbered" or "labeled" -- write it as inline math ($...$) '
                "instead; a numbered, cross-referenceable version belongs in "
                "a dedicated `equation` block only.\n\n"
            )
        else:
            retry_notice = ""

        accepted_summary = ""
        if accepted_blocks:
            accepted_lines = [
                f"- '{block_id}' ({block.type.value}): {block.content.strip()[:80]}..."
                for block_id, block in accepted_blocks.items()
            ]
            accepted_summary = (
                "\nAlready-approved blocks in this section (already finished "
                "and written to the document -- for context only, do not "
                "regenerate these):\n" + "\n".join(accepted_lines) + "\n"
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
{accepted_summary}
Required semantic blocks, in this exact order:
{json.dumps(planned_blocks, ensure_ascii=False, indent=2)}

Available cross-reference labels (cite with \\Cref{{...}}/\\ref{{...}} using
the exact "label" string; never invent or guess one not listed here -- if
what you want to cite isn't here, it doesn't exist and can't be cited):
{json.dumps(citable_labels, ensure_ascii=False, indent=2)}

Figures owned by this section; refer to these only when relevant:
{json.dumps(figure_context, ensure_ascii=False, indent=2)}
""".strip()

    def _validate_blocks_against_layout(
        self,
        raw_blocks: Any,
        layout_blocks: List[LayoutBlock],
    ) -> List[ContentBlock]:
        """Validate one attempt's blocks against the requested layout blocks.

        ``layout_blocks`` may be the whole section or a narrowed pending
        subset (see ``process_node``'s progressive narrowing) -- either way,
        the returned blocks must match it exactly in id and order. That
        structural check fails the whole attempt immediately (there's no
        single block to blame for a wrong id, count, or order); every
        block's *content* is checked regardless of any earlier block's
        result and every failure is collected, not just the first, so a
        caller can narrow the next attempt to exactly what's still wrong in
        one pass instead of discovering violations one at a time.
        """
        if not isinstance(raw_blocks, list):
            raise ValueError("content_blocks must be a JSON array.")

        blocks = [ContentBlock.model_validate(raw_block) for raw_block in raw_blocks]
        expected_ids = [block.id for block in layout_blocks]
        actual_ids = [block.block_id for block in blocks]
        if actual_ids != expected_ids:
            raise ValueError(
                "Semantic block IDs must match the requested blocks in order. "
                f"Expected {expected_ids}, received {actual_ids}."
            )

        errors: List[SemanticBlockError] = []
        for content_block, layout_block in zip(blocks, layout_blocks):
            errors.extend(self._validate_one_block(content_block, layout_block))
        if errors:
            raise SemanticBlockErrors(blocks, errors)
        return blocks

    def _validate_one_block(
        self,
        content_block: ContentBlock,
        layout_block: LayoutBlock,
    ) -> List[SemanticBlockError]:
        """Check one block's content, returning every violation found.

        Structural contract checks (required type, non-empty content) always
        run -- the composer's single-sole-input contract depends on them.
        Everything else here is a content-quality verifier gated by
        ``self.enabled_validators``, so a verifier ablation run can disable
        one named check without touching the others.
        """
        block_id = layout_block.id
        errors: List[SemanticBlockError] = []

        expected_type = SEMANTIC_TYPE_BY_LAYOUT_KIND[layout_block.kind]
        if content_block.type != expected_type:
            errors.append(
                SemanticBlockError(
                    "type_mismatch",
                    block_id,
                    f"Block '{block_id}' must use type '{expected_type.value}', "
                    f"not '{content_block.type.value}'.",
                )
            )
            # A wrong type makes every type-specific check below meaningless
            # (e.g. checking the wrong type's environment allowlist) --
            # report just this and let the next attempt fix it first.
            return errors
        if not content_block.content.strip():
            errors.append(
                SemanticBlockError("empty_content", block_id, f"Block '{block_id}' has empty content.")
            )
            return errors

        # Same environment/command whitelist the composer enforces in
        # LatexAssembler._validate_fragment, run here too so a
        # violation raises inside process_node's try/except -- which
        # marks the node ERROR and lets the pipeline's iteration loop
        # retry it with a fresh sample -- instead of surfacing for the
        # first time during Phase 3 integration, where nothing retries
        # and the whole request dies.
        if self.enabled_validators.get("forbidden_environment", True):
            try:
                validate_fragment_environments(content_block.type, content_block.content, block_id)
            except ValueError as exc:
                errors.append(SemanticBlockError("forbidden_environment", block_id, str(exc)))

        # Same reasoning: both used to be composer-only (Phase 3), where
        # a violation surfaces only after every other section is already
        # drafted and nothing retries -- a manually-written \label{}
        # colliding with another independently-drafted section's guess
        # at the "same" label (the eq:energy_balance incident), or an
        # unescaped '|' inside a table cell (e.g. 'E[Y|X=x]' instead of
        # '\mid'), used to kill the whole document there instead of
        # retrying just this block.
        if self.enabled_validators.get("manual_label", True):
            try:
                validate_no_manual_labels(content_block.type, content_block.content, block_id)
            except ValueError as exc:
                errors.append(SemanticBlockError("manual_label", block_id, str(exc)))

        if (
            layout_block.kind in (LayoutBlockKind.TABLE, LayoutBlockKind.NOTATION_TABLE, LayoutBlockKind.COMPARISON)
            and self.enabled_validators.get("table_columns", True)
        ):
            try:
                validate_markdown_table_columns(content_block.content, block_id)
            except ValueError as exc:
                errors.append(SemanticBlockError("table_columns", block_id, str(exc)))

        # Content-shape checks, as opposed to table_columns' internal-
        # correctness check above: does this block even look like the thing
        # its kind claims to be. COMPARISON is deliberately excluded from the
        # table-shape check -- LatexAssembler._render_comparison already
        # treats plain prose as an equally legitimate rendering for that
        # kind, unlike TABLE/NOTATION_TABLE.
        if (
            layout_block.kind in (LayoutBlockKind.TABLE, LayoutBlockKind.NOTATION_TABLE)
            and self.enabled_validators.get("table_shape", True)
        ):
            try:
                _validate_component_content_shape(layout_block.kind, content_block.content, block_id)
            except ValueError as exc:
                errors.append(SemanticBlockError("table_shape", block_id, str(exc)))

        if (
            layout_block.kind == LayoutBlockKind.EQUATION
            and self.enabled_validators.get("equation_shape", True)
        ):
            try:
                _validate_component_content_shape(layout_block.kind, content_block.content, block_id)
            except ValueError as exc:
                errors.append(SemanticBlockError("equation_shape", block_id, str(exc)))

        if self.enabled_validators.get("plain_title", True):
            try:
                validate_plain_title(content_block.title, f"Block '{block_id}'")
            except ValueError as exc:
                errors.append(SemanticBlockError("plain_title", block_id, str(exc)))

        # Applies to every block type, including equation -- a raw Unicode
        # symbol is wrong LaTeX regardless of whether it sits inside a math
        # delimiter or not.
        if self.enabled_validators.get("unicode_math_symbol", True):
            try:
                _validate_unicode_math_symbols(content_block.content, block_id)
            except ValueError as exc:
                errors.append(SemanticBlockError("unicode_math_symbol", block_id, str(exc)))

        # equation is exempt: its whole "content" IS a math body (see
        # <Content rules by block type>), so it never needs its own $...$
        # wrapper -- the composer's _render_equation supplies the delimiters.
        if (
            layout_block.kind != LayoutBlockKind.EQUATION
            and self.enabled_validators.get("bare_math_notation", True)
        ):
            try:
                _validate_bare_math_commands(content_block.content, block_id)
            except ValueError as exc:
                errors.append(SemanticBlockError("bare_math_notation", block_id, str(exc)))

        if layout_block.kind == LayoutBlockKind.FIGURE and self.enabled_validators.get("figure_content", True):
            if content_block.asset_id != layout_block.asset_id:
                errors.append(
                    SemanticBlockError(
                        "figure_asset_mismatch",
                        block_id,
                        f"Figure reference block '{block_id}' must use asset_id '{layout_block.asset_id}'.",
                    )
                )
            forbidden = ("\\begin{figure}", "\\includegraphics")
            if any(token in content_block.content for token in forbidden):
                errors.append(
                    SemanticBlockError(
                        "figure_forbidden_content",
                        block_id,
                        f"Figure reference block '{block_id}' must not render a figure.",
                    )
                )
        return errors

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

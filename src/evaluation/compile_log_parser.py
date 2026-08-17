"""Parse a LaTeX compiler's log into (fatal_error_count, warning_count, first_error_type).

Built from real logs already on disk in this repo, not guessed. Tectonic
(the default engine -- see ``LatexCompiler._resolve_engine``) writes
lowercase ``note:``/``warning:``/``error:`` prefixed lines, which is a
different format from classic pdflatex output (``"! ..."`` fatal errors,
``"LaTeX Warning:"``/``"Package ... Warning:"`` warnings). The existing
``src/evaluation/latex_robustness/latex_robustness.py::parse_latex_log`` is
pdflatex-shaped only and would silently under-count against a real tectonic
log -- this is an independent parser, not a reuse of that one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.evaluation.schemas import RecoveryAction

# Tectonic emits these as their own "error:"/"warning:" lines, but they are
# terminal meta-commentary about the run as a whole, not distinct diagnostics
# -- counting them would double-count the one real error/warning they refer
# to. Confirmed against real logs in this repo: a 1-error/3-warning compile
# produces 2 raw "error:" lines and 4 raw "warning:" lines without this
# exclusion.
_TECTONIC_BOILERPLATE_SUBSTRINGS = (
    "halted on ",
    "warnings were issued by the tex engine",
)

_TECTONIC_ERROR_LINE = re.compile(r"^error:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_TECTONIC_WARNING_LINE = re.compile(r"^warning:\s*(.*)$", re.IGNORECASE | re.MULTILINE)

# Classic pdflatex/TeX log conventions. Never exercised by a real log in this
# repo (tectonic is preferred whenever both are installed -- see
# _resolve_engine), reachable only via the PDFLATEX_BIN fallback.
_PDFLATEX_ERROR_LINE = re.compile(r"^! (.*)$", re.MULTILINE)
_PDFLATEX_WARNING_LINE = re.compile(
    r"^(?:LaTeX(?: Font)? Warning: |Package \S+ Warning: |Overfull \\hbox|Underfull \\hbox)(.*)$",
    re.MULTILINE,
)

# Ordered (first match wins) message -> canonical first_error_type mapping,
# shared by both engines since the underlying TeX diagnostics read the same
# regardless of which binary produced them.
_ERROR_TYPE_RULES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"missing \$", re.IGNORECASE), "missing_math_delimiter"),
    (re.compile(r"undefined control sequence", re.IGNORECASE), "undefined_control_sequence"),
    (re.compile(r"undefined\s*$", re.IGNORECASE), "undefined_reference_or_command"),
    (re.compile(r"missing \\begin\{document\}", re.IGNORECASE), "missing_begin_document"),
    (re.compile(r"(?:runaway|missing) .*\\end", re.IGNORECASE), "unbalanced_environment"),
    (re.compile(r"file .* not found", re.IGNORECASE), "missing_file"),
    (re.compile(r"latex error", re.IGNORECASE), "latex_error"),
    (re.compile(r"package \S+ error", re.IGNORECASE), "package_error"),
    (re.compile(r"missing", re.IGNORECASE), "missing_delimiter_or_argument"),
)


def classify_error_type(message: str) -> str:
    """Map a raw error line to a canonical, small-vocabulary ``first_error_type``."""
    for pattern, error_type in _ERROR_TYPE_RULES:
        if pattern.search(message):
            return error_type
    return "other"


# package_error/missing_file are composer-owned bookkeeping (package set in
# _required_packages, figure asset paths in the figure registry) -- not model
# content and not a broken environment, so DETERMINISTIC_REPAIR rather than
# INFRASTRUCTURE_FAILURE. missing_begin_document means the assembled document
# frame itself is malformed, which LatexAssembler.validate_document_frame
# should already have caught before compilation -- reaching the engine at all
# means something structural broke, not the content.
_REGENERATE_CONTENT_ERROR_TYPES = frozenset({
    "missing_math_delimiter", "undefined_control_sequence",
    "undefined_reference_or_command", "unbalanced_environment",
    "missing_delimiter_or_argument", "latex_error",
})
_DETERMINISTIC_REPAIR_ERROR_TYPES = frozenset({"package_error", "missing_file"})
_INFRASTRUCTURE_FAILURE_ERROR_TYPES = frozenset({
    "missing_begin_document", "main_tex_missing", "no_latex_engine_found",
})


def classify_recovery_action(error_type: Optional[str]) -> Optional[RecoveryAction]:
    """Map a ``first_error_type`` to the kind of recovery it would need.

    ``None`` only when there's no error_type at all (e.g. compile succeeded);
    an unrecognized non-None error_type (``other``, or a future addition to
    ``classify_error_type``) falls to ``UNKNOWN`` rather than being silently
    miscategorized into one of the other three buckets.
    """
    if error_type is None:
        return None
    if error_type in _REGENERATE_CONTENT_ERROR_TYPES:
        return RecoveryAction.REGENERATE_CONTENT
    if error_type in _DETERMINISTIC_REPAIR_ERROR_TYPES:
        return RecoveryAction.DETERMINISTIC_REPAIR
    if error_type in _INFRASTRUCTURE_FAILURE_ERROR_TYPES:
        return RecoveryAction.INFRASTRUCTURE_FAILURE
    return RecoveryAction.UNKNOWN


@dataclass(frozen=True)
class CompileErrorLocation:
    """One fatal error, resolved back to the section file tectonic was
    reading when it fired.

    ``section_id`` needs no extra lookup to become a graph node id:
    ``LatexAssembler._write_main_tex`` ``\\input``s every section from its
    own physical file (``sections/{section_id}.tex``), so tectonic's own
    path component in the log line already *is* the section_id.
    """

    section_id: str
    line: int
    message: str
    error_type: str


# Same path tectonic reports every diagnostic against: the physical file it
# was reading when the diagnostic fired. Only fatal ``error:`` lines are
# matched (not ``warning:``) -- a warning (e.g. the missing-character/
# overfull-hbox lines seen in real logs) is cosmetic, not a reason to spend
# an LLM call rewriting a block. Never observed for pdflatex (see the module
# docstring): its log instead nests file names as parenthesis-delimited
# push/pop markers, which needs a stack to resolve correctly and isn't
# attempted here -- extract_error_locations() simply returns no locations
# for a pdflatex log.
_TECTONIC_LOCATED_ERROR = re.compile(
    r"^error:\s*sections/(?P<section_id>[A-Za-z0-9_-]+):(?P<line>\d+):\s*(?P<message>.*)$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_error_locations(log_text: str, engine_name: str) -> List[CompileErrorLocation]:
    """Return every fatal error tectonic attributed to a specific section
    file, in log order. Empty for a pdflatex log, or for a tectonic log
    whose fatal error has no section-file location (e.g. a preamble-level
    failure) -- callers fall back to their non-located behavior either way.
    """
    if "tectonic" not in engine_name.lower():
        return []
    locations: List[CompileErrorLocation] = []
    for match in _TECTONIC_LOCATED_ERROR.finditer(log_text):
        message = match.group("message").strip()
        if any(s in message.lower() for s in _TECTONIC_BOILERPLATE_SUBSTRINGS):
            continue
        locations.append(
            CompileErrorLocation(
                section_id=match.group("section_id"),
                line=int(match.group("line")),
                message=message,
                error_type=classify_error_type(message),
            )
        )
    return locations


def parse_compile_log(log_text: str, engine_name: str) -> Tuple[int, int, Optional[str]]:
    """Return ``(fatal_error_count, warning_count, first_error_type)`` for one compile.log."""
    if "tectonic" in engine_name.lower():
        error_lines = [
            m.group(1).strip()
            for m in _TECTONIC_ERROR_LINE.finditer(log_text)
            if not any(s in m.group(1).lower() for s in _TECTONIC_BOILERPLATE_SUBSTRINGS)
        ]
        warning_lines = [
            m.group(1).strip()
            for m in _TECTONIC_WARNING_LINE.finditer(log_text)
            if not any(s in m.group(1).lower() for s in _TECTONIC_BOILERPLATE_SUBSTRINGS)
        ]
    else:
        error_lines = [m.group(1).strip() for m in _PDFLATEX_ERROR_LINE.finditer(log_text)]
        warning_lines = [m.group(1).strip() for m in _PDFLATEX_WARNING_LINE.finditer(log_text)]

    first_error_type = classify_error_type(error_lines[0]) if error_lines else None
    return len(error_lines), len(warning_lines), first_error_type

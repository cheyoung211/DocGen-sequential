"""Coverage for parse_compile_log against real tectonic logs already on disk
in this repo, plus a synthetic pdflatex-shaped log (no real pdflatex log
exists anywhere in this repo -- tectonic is preferred whenever installed)."""

from __future__ import annotations

import unittest

from src.evaluation.compile_log_parser import parse_compile_log

# Verbatim from outputs/generations_08101010/math_calculus_0015/sample_000/compile.log
TECTONIC_SUCCESS_LOG = """\
note: Running TeX ...
warning: sections/sec_common_misconceptions:1: Overfull \\hbox (5.96147pt too wide) in paragraph at lines 1--1
note: Rerunning TeX because "main.toc" changed ...
warning: sections/sec_common_misconceptions:1: Overfull \\hbox (5.96147pt too wide) in paragraph at lines 1--1
note: Rerunning TeX because "main.toc" changed ...
warning: sections/sec_common_misconceptions:1: Overfull \\hbox (5.96147pt too wide) in paragraph at lines 1--1
warning: warnings were issued by the TeX engine; use --print and/or --keep-logs for details.
note: Running xdvipdfmx ...
note: Writing `main.pdf` (63.38 KiB)
note: Skipped writing 3 intermediate files (use --keep-intermediates to keep them)
"""

# Verbatim from outputs/generations_08101010/math_probability_theory_0020/sample_001/compile.log
TECTONIC_FAILURE_LOG = """\
note: Running TeX ...
error: sections/sec_formal_definition:14: Missing $ inserted
error: halted on potentially-recoverable error as specified
"""

# Synthetic -- explicitly not sourced from a real artifact in this repo.
PDFLATEX_FAILURE_LOG = """\
This is pdfTeX, Version 3.14159265-2.6-1.40.21 (TeX Live 2020)
! Undefined control sequence.
l.42 \\nonexistentcommand
LaTeX Warning: Reference `fig:missing' on page 3 undefined on input line 55.
Overfull \\hbox (12.0pt too wide) in paragraph at lines 60--61
! LaTeX Error: File `nonexistent.sty' not found.
"""


class ParseCompileLogTest(unittest.TestCase):
    def test_tectonic_success_log_has_no_fatal_errors_and_excludes_boilerplate_warning(self) -> None:
        fatal, warnings, first_error_type = parse_compile_log(TECTONIC_SUCCESS_LOG, "tectonic")
        self.assertEqual(fatal, 0)
        self.assertEqual(warnings, 3)
        self.assertIsNone(first_error_type)

    def test_tectonic_failure_log_excludes_halted_on_boilerplate(self) -> None:
        fatal, warnings, first_error_type = parse_compile_log(TECTONIC_FAILURE_LOG, "tectonic")
        self.assertEqual(fatal, 1)
        self.assertEqual(warnings, 0)
        self.assertEqual(first_error_type, "missing_math_delimiter")

    def test_pdflatex_failure_log_counts_both_fatal_errors_and_one_warning_type(self) -> None:
        fatal, warnings, first_error_type = parse_compile_log(PDFLATEX_FAILURE_LOG, "pdflatex")
        self.assertEqual(fatal, 2)
        self.assertEqual(warnings, 2)
        self.assertEqual(first_error_type, "undefined_control_sequence")


if __name__ == "__main__":
    unittest.main()

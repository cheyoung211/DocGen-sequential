# eval_mechanical_stability.py

from __future__ import annotations
import subprocess
import re
import os
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
import json
from typing import List


# -------------------------------------------------------
# 1. DataClass: Mechanical Evaluation Result
# -------------------------------------------------------
@dataclass
class MechanicalEvalResult:
    file: str
    compile_success: bool
    error_count: int
    warning_count: int
    error_types: List[str]
    structure_ok: bool
    missing_files: List[str]


# -------------------------------------------------------
# Helper: Resolve LaTeX engines
# -------------------------------------------------------
def resolve_pdflatex_bin() -> str | None:
    # 1) explicit override
    env_bin = os.environ.get("PDFLATEX_BIN")
    if env_bin:
        resolved = shutil.which(env_bin) or (env_bin if Path(env_bin).is_file() else None)
        if resolved:
            return str(resolved)

    # 2) PATH lookup
    resolved = shutil.which("pdflatex")
    if resolved:
        return resolved

    # 3) common TeX Live installation paths
    common_roots = [
        Path("/usr/local/texlive"),
        Path("/usr/share/texlive"),
        Path.home() / "texlive",
    ]
    for root in common_roots:
        if not root.exists():
            continue
        for candidate in root.glob("**/pdflatex"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def run_pdflatex(tex_path: Path) -> tuple[bool, str]:
    """
    Compile a LaTeX file.
    Priority:
      1) tectonic (same strategy as test/compile_test.py)
      2) pdflatex fallback
    Returns:
        (compile_success, log_text)
    """
    tex_dir = tex_path.parent
    tex_file = tex_path.name

    # 1) tectonic 우선 사용
    tectonic_bin = shutil.which("tectonic")
    if tectonic_bin is not None:
        proc = subprocess.run(
            [tectonic_bin, tex_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tex_dir,
            text=True,
        )
        log_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode == 0, log_text

    # 2) pdflatex fallback
    resolved_bin = resolve_pdflatex_bin()
    if resolved_bin is None:
        msg = (
            "[MechanicalEval] LaTeX engine not found: 'tectonic' and 'pdflatex'.\n"
            "Install tectonic (`conda install -c conda-forge tectonic`) or TeX Live,\n"
            "or set PDFLATEX_BIN to an absolute path.\n"
            f"Current PATH: {os.environ.get('PATH', '')}\n"
        )
        return False, msg

    compile_cmd = [
        resolved_bin,
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_file,
    ]

    proc = subprocess.run(
        compile_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=tex_dir,
        text=True,
    )

    log_text = proc.stdout or ""
    return proc.returncode == 0, log_text


# -------------------------------------------------------
# Helper: Parse LaTeX errors/warnings
# -------------------------------------------------------
def parse_latex_log(log_text: str) -> tuple[int, int, List[str]]:
    """
    Count errors and warnings from LaTeX log text.
    Return:
        (error_count, warning_count, error_types)
    """

    # Basic patterns
    error_patterns = [
        r"! Undefined control sequence",
        r"! Missing ",
        r"! LaTeX Error",
        r"! Package .* Error",
        r"^l\.\d+",  # line-based errors
    ]

    warning_patterns = [
        r"Warning:",
        r"LaTeX Warning",
        r"Package .* Warning",
    ]

    error_count = 0
    warning_count = 0
    error_types = []

    # Check error patterns
    for pat in error_patterns:
        matches = re.findall(pat, log_text, flags=re.MULTILINE)
        if matches:
            error_count += len(matches)
            error_types.extend([pat] * len(matches))

    # Check warning patterns
    for pat in warning_patterns:
        matches = re.findall(pat, log_text, flags=re.MULTILINE)
        warning_count += len(matches)

    return error_count, warning_count, error_types


# -------------------------------------------------------
# Helper: Simple Structural Integrity Check
# -------------------------------------------------------
def check_structure(source: str) -> tuple[bool, List[str]]:
    """
    Check structural correctness:
        - balanced \begin{} and \end{}
        - missing figures/tables
        - common LaTeX mistakes
    """

    missing_files = []

    # 1) Environment matching
    begin_env = re.findall(r"\\begin\{(.+?)\}", source)
    end_env = re.findall(r"\\end\{(.+?)\}", source)

    if sorted(begin_env) != sorted(end_env):
        structure_ok = False
    else:
        structure_ok = True

    # 2) Missing included graphics
    include_graphics = re.findall(r"\\includegraphics.*\{(.+?)\}", source)
    for f in include_graphics:
        if not Path(f).exists():
            missing_files.append(f)
            structure_ok = False

    return structure_ok, missing_files


# -------------------------------------------------------
# Main Evaluation Function
# -------------------------------------------------------
def evaluate_tex(tex_file: Path) -> MechanicalEvalResult:
    tex_source = tex_file.read_text(encoding="utf-8", errors="ignore")

    success, log_text = run_pdflatex(tex_file)
    error_count, warning_count, error_types = parse_latex_log(log_text)
    structure_ok, missing_files = check_structure(tex_source)

    return MechanicalEvalResult(
        file=str(tex_file),
        compile_success=success,
        error_count=error_count,
        warning_count=warning_count,
        error_types=error_types,
        structure_ok=structure_ok,
        missing_files=missing_files,
    )


def split_factor_sample(relative_dir: Path) -> tuple[str | None, str]:
    """Recover (factor, sample_id) from a sample directory relative to the batch dir.

    Supports both the factor/sample_id layout and the flatter sample_id-only
    layout (no stress-factor grouping).
    """
    path_parts = relative_dir.parts
    if len(path_parts) >= 2:
        return path_parts[0], path_parts[1]
    if len(path_parts) == 1:
        return None, path_parts[0]
    return None, "unknown"


def discover_sample_dirs(base_dir: Path) -> list[Path]:
    """Directories with evidence that generation was attempted (a plan.json or
    meta.json was written), regardless of whether it produced a compilable
    main.tex. A directory with neither marker never started generating and is
    excluded entirely rather than reported as a failure.
    """
    marker_files = list(base_dir.glob("**/plan.json")) + list(base_dir.glob("**/meta.json"))
    return sorted({f.parent for f in marker_files})


def run_batch_evaluation(args):
    base_dir = Path(args.dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {base_dir}")

    results = []
    sample_dirs = discover_sample_dirs(base_dir)
    print(f"[Eval] Starting batch evaluation for: {base_dir.name} ({len(sample_dirs)} attempted samples)")

    for sample_dir in sample_dirs:
        relative_path = sample_dir.relative_to(base_dir)
        factor_name, sample_id = split_factor_sample(relative_path)
        tex_file = sample_dir / "main.tex"

        if not tex_file.is_file():
            print(f"  - {relative_path}: generation attempted but main.tex is missing")
            results.append({
                "file": str(tex_file),
                "factor": factor_name,
                "sample_id": sample_id,
                "rel_path": str(relative_path),
                "generation_failed": True,
                "failure_reason": "main.tex not found",
                "compile_success": False,
                "error_count": None,
                "warning_count": None,
                "error_types": [],
                "structure_ok": None,
                "missing_files": [],
            })
            continue

        print(f"  - Evaluating: {relative_path}")
        res = evaluate_tex(tex_file)

        res_dict = asdict(res)
        res_dict.update({
            "factor": factor_name,
            "sample_id": sample_id,
            "rel_path": str(relative_path),
            "generation_failed": False,
        })
        results.append(res_dict)

    out_dir = Path.cwd() / "evaluation" / "latex_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{base_dir.name}_{Path(args.out).name}"
    out_path = out_dir / file_name

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    failed = sum(1 for r in results if r.get("generation_failed"))
    print(f"[Eval] {failed}/{len(results)} attempted samples never produced a main.tex.")
    print(f"[Eval] Done. {len(results)} samples evaluated. Results saved to {out_path}")

# -------------------------------------------------------
# CLI for batch evaluation
# -------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="outputs/batch_test_07281829/basic",
        help="Directory containing generated report directories.",
    )
    parser.add_argument(
        "--out",
        default="mechanical_eval.json",
        help="Output JSON file for evaluation results."
    )

    args = parser.parse_args()
    run_batch_evaluation(args)

    

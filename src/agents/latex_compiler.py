"""Invoke a LaTeX engine against an already-assembled project and classify the result."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, List, Optional

from src.evaluation.compile_log_parser import classify_recovery_action, parse_compile_log
from src.evaluation.event_logger import EventLogger
from src.evaluation.schemas import CompileResult


class LatexCompiler:
    """Compile an already-assembled LaTeX project and record the outcome."""

    def __init__(
        self,
        llm_client: Any = None,
        engine: str = "tectonic",
        runs: int = 2,
    ):
        # Kept for constructor compatibility. Compilation itself needs no LLM.
        self.llm = llm_client
        self.engine = engine
        self.runs = runs

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
            recovery_action=classify_recovery_action(first_error_type),
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
            print(f"[Compiler Error] main.tex not found at {main_tex_path}")
            self._finalize_compile_result(
                event_logger, base_dir, compile_success=False, engine=None, return_code=None,
                log_text="", first_error_type_override="main_tex_missing",
            )
            return False

        engine_bin, mode_args = self._resolve_engine()
        if engine_bin is None:
            print("[Compiler Error] No LaTeX engine found (tried configured engine, tectonic, pdflatex).")
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
                    print(f"[Compiler Warning] Compilation failed. Check {log_path}.")
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

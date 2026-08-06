# document_generation/helpers/latex_writer.py
# document_generation/helpers/latex_writer.py

from __future__ import annotations
from pathlib import Path
from typing import List
from scripts.schemas import ContentBlock


def escape_latex_text(text: str) -> str:
    """
    매우 간단한 LaTeX 특수문자 escape.
    - 수식($$...$$) 안까지 완벽히 구분하진 않지만,
    일반 문장에서는 &, %, #, _ 같은 것들을 안전하게 만들어줌.
    """
    LATEX_SPECIALS = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }
    for ch, rep in LATEX_SPECIALS.items():
        text = text.replace(ch, rep)
    return text

def blocks_to_latex(blocks: List[ContentBlock]) -> str:
    """
    ContentBlock 목록을 실제 LaTeX 텍스트로 변환.
    paragraph → 문단
    table_ref → 레퍼런스 문장
    figure_ref → 레퍼런스 문장
    """
    lines: List[str] = []


    for blk in blocks:

        if blk.type == "paragraph":
            text = (blk.text or "").strip()
            if text:
                text = escape_latex_text(text)
                lines.append(text + "\n")

        elif blk.type == "subsection":
            title = (blk.title or blk.text or "").strip()
            if title:
                lines.append(f"\\subsection*{{{escape_latex_text(title)}}}\n")

        elif blk.type == "itemize":
            items = blk.items or []
            clean_items = [escape_latex_text((it or "").strip()) for it in items if (it or "").strip()]
            if clean_items:
                lines.append("\\begin{itemize}\n")
                for it in clean_items:
                    lines.append(f"\\item {it}\n")
                lines.append("\\end{itemize}\n")

        elif blk.type == "enumerate":
            items = blk.items or []
            clean_items = [escape_latex_text((it or "").strip()) for it in items if (it or "").strip()]
            if clean_items:
                lines.append("\\begin{enumerate}\n")
                for it in clean_items:
                    lines.append(f"\\item {it}\n")
                lines.append("\\end{enumerate}\n")

        elif blk.type == "key_point":
            text = (blk.text or "").strip()
            if text:
                text = escape_latex_text(text)
                lines.append("\\begin{quote}\n")
                lines.append(f"\\textbf{{Key Point.}} {text}\n")
                lines.append("\\end{quote}\n")

        elif blk.type == "table_ref":
            ref = blk.ref_text or (
                f"Table~\\ref{{tab:{blk.table_id}}}" if blk.table_id else ""
            )
            ref = ref.strip()
            if ref:
                # \ref{} 같은 LaTeX 참조 문자열은 escape하면 깨질 수 있으므로 raw 유지
                if "\\ref{" not in ref:
                    ref = escape_latex_text(ref)
                lines.append(ref + "\n")

        elif blk.type == "figure_ref":
            ref = blk.ref_text or (
                f"Figure~\\ref{{fig:{blk.figure_id}}}" if blk.figure_id else ""
            )
            ref = ref.strip()
            if ref:
                if "\\ref{" not in ref:
                    ref = escape_latex_text(ref)
                lines.append(ref + "\n")

        elif blk.type == "table":
            headers = blk.table_headers or []
            rows = blk.table_rows or []
            if headers and rows:
                ncol = len(headers)
                colspec = " | ".join(["l"] * ncol)
                caption = (blk.table_caption or "").strip()
                lines.append("\\begin{table}[htbp]\n")
                lines.append("\\centering\n")
                if caption:
                    lines.append(f"\\caption{{{escape_latex_text(caption)}}}\n")
                lines.append(f"\\begin{{tabular}}{{{colspec}}}\n")
                lines.append("\\hline\n")
                lines.append(" & ".join(escape_latex_text(h.strip()) for h in headers) + " \\\\\n")
                lines.append("\\hline\n")
                for row in rows:
                    row_cells = [escape_latex_text(str(c).strip()) for c in row[:ncol]]
                    if len(row_cells) < ncol:
                        row_cells.extend([""] * (ncol - len(row_cells)))
                    lines.append(" & ".join(row_cells) + " \\\\\n")
                lines.append("\\hline\n")
                lines.append("\\end{tabular}\n")
                lines.append("\\end{table}\n")

        elif blk.type == "equation":
            # 이건 수식이니까 $$...$$로 감싸고 escape는 하지 않음
            eq = (blk.text or "").strip()
            if not (eq.startswith("$$") and eq.endswith("$$")):
                eq = f"$$\n{eq}\n$$"
            lines.append(eq + "\n")

        else:
            lines.append(f"% [Unknown block type: {blk.type}]\n")

    return "\n".join(lines)


def save_section_tex(
    section_id: str,
    title: str,
    blocks: List[ContentBlock],
    out_dir: str | Path,
) -> Path:
    """
    섹션을 tex 파일로 저장.
    """
    out_path = Path(out_dir) / f"{section_id}.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content = []

    # section heading 추가
    content.append(f"\\section{{{title}}}\n\\label{{sec:{section_id}}}\n")

    # 본문
    body = blocks_to_latex(blocks)
    content.append(body)

    full = "\n".join(content)
    out_path.write_text(full, encoding="utf-8")
    return out_path

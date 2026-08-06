# eval_multimodal_alignment.py

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
import math

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# -------------------------------
# Dataclass for per-figure result
# -------------------------------
@dataclass
class FigureAlignmentResult:
    doc_dir: str
    figure_id: str
    image_path: str
    caption: str
    clip_score: float


@dataclass
class TableAlignmentResult:
    doc_dir: str
    table_id: str
    table_path: str
    table_caption: str
    best_section: str
    alignment_score: float


# -------------------------------
# 1. LaTeX 파서: figure/table parsing
# -------------------------------
FIGURE_ENV_PATTERN = re.compile(
    r"\\begin\{figure\}.*?\\includegraphics(?:\[.*?\])?\{(?P<path>.+?)\}.*?"
    r"\\caption\{(?P<caption>.+?)\}.*?\\label\{(?P<label>.+?)\}.*?\\end\{figure\}",
    re.DOTALL,
)

TABLE_INPUT_PATTERN = re.compile(r"\\input\{(?P<path>tables/[^}]+)\}")
TABLE_CAPTION_PATTERN = re.compile(r"\\caption\{(?P<caption>.+?)\}", re.DOTALL)
SECTION_INPUT_PATTERN = re.compile(r"\\input\{(?P<path>sections/[^}]+)\}")
ANY_INPUT_PATTERN = re.compile(r"\\input\{(?P<path>[^{}]+)\}")
# The current pipeline never emits a standalone tables/*.tex \input; tables are
# rendered inline inside each section by TextAgent's data tools instead.
TABLE_ENV_PATTERN = re.compile(r"\\begin\{(?:table|longtable)\}.*?\\end\{(?:table|longtable)\}", re.DOTALL)


def extract_figures_from_tex(tex_source: str) -> List[Dict[str, str]]:
    """
    main.tex (또는 다른 tex) 안에서 figure 환경을 찾아서
    image path, caption, label(figure id) 를 추출한다.
    """
    figures: List[Dict[str, str]] = []

    for m in FIGURE_ENV_PATTERN.finditer(tex_source):
        img_path = m.group("path").strip()
        caption = m.group("caption").strip()
        label = m.group("label").strip()
        figures.append(
            {
                "image_path": img_path,
                "caption": caption,
                "label": label,
            }
        )

    return figures


def _resolve_input_tex_path(doc_dir: Path, input_path_no_ext: str) -> Path:
    # \input{tables/foo} 형태를 tables/foo.tex로 보정
    p = doc_dir / input_path_no_ext
    if p.suffix != ".tex":
        p = p.with_suffix(".tex")
    return p


def _clean_latex_for_text(source: str) -> str:
    # 간단한 정리: 명령어/중괄호 제거 및 공백 normalize
    text = re.sub(r"%.*", " ", source)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
    text = re.sub(r"[{}\\\\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _bow_cosine(a: str, b: str) -> float:
    ta = Counter(_tokenize(a))
    tb = Counter(_tokenize(b))
    if not ta or not tb:
        return 0.0
    common = set(ta.keys()) & set(tb.keys())
    dot = sum(ta[t] * tb[t] for t in common)
    na = math.sqrt(sum(v * v for v in ta.values()))
    nb = math.sqrt(sum(v * v for v in tb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def extract_tables_from_doc(doc_dir: Path, main_tex_source: str) -> List[Dict[str, str]]:
    tables: List[Dict[str, str]] = []

    # Legacy layout: a standalone tables/*.tex \input from main.tex.
    for m in TABLE_INPUT_PATTERN.finditer(main_tex_source):
        rel = m.group("path").strip()
        table_path = _resolve_input_tex_path(doc_dir, rel)
        if not table_path.is_file():
            continue
        src = table_path.read_text(encoding="utf-8", errors="ignore")
        cap_match = TABLE_CAPTION_PATTERN.search(src)
        caption = cap_match.group("caption").strip() if cap_match else ""
        tables.append(
            {
                "table_id": table_path.stem,
                "table_path": str(table_path),
                "caption": caption,
                "source": src,
            }
        )

    # Current layout: table/longtable environments rendered inline inside a
    # section file, so look for them there instead of a dedicated \input.
    for section in extract_sections_from_doc(doc_dir, main_tex_source):
        for idx, table_match in enumerate(TABLE_ENV_PATTERN.finditer(section["source"]), start=1):
            src = table_match.group(0)
            cap_match = TABLE_CAPTION_PATTERN.search(src)
            caption = cap_match.group("caption").strip() if cap_match else ""
            tables.append(
                {
                    "table_id": f"{section['section_id']}_table_{idx}",
                    "table_path": section["path"],
                    "caption": caption,
                    "source": src,
                }
            )

    return tables


def extract_sections_from_doc(doc_dir: Path, main_tex_source: str) -> List[Dict[str, str]]:
    sections: List[Dict[str, str]] = []
    for m in SECTION_INPUT_PATTERN.finditer(main_tex_source):
        rel = m.group("path").strip()
        sec_path = _resolve_input_tex_path(doc_dir, rel)
        if not sec_path.is_file():
            continue
        src = sec_path.read_text(encoding="utf-8", errors="ignore")
        sections.append(
            {
                "section_id": sec_path.stem,
                "path": str(sec_path),
                "source": src,
            }
        )
    return sections


def expand_tex_source(doc_dir: Path, main_tex_source: str) -> str:
    """Inline every \\input{...} target referenced from main.tex.

    Figures (and, for older layouts, tables) live inside per-section files
    that main.tex only references via \\input, so parsers that scan main.tex
    alone never see them.
    """
    parts = [main_tex_source]
    for m in ANY_INPUT_PATTERN.finditer(main_tex_source):
        included_path = _resolve_input_tex_path(doc_dir, m.group("path").strip())
        if included_path.is_file():
            parts.append(included_path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


# -------------------------------
# 2. CLIP 기반 이미지–텍스트 유사도 계산
# -------------------------------
class CLIPAligner:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[CLIPAligner] Loading CLIP model: {model_name} on {device}")
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def image_text_score(self, image: Image.Image, text: str) -> float:
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: (1, 1)
        score = logits_per_image.item()
        return float(score)


_CLIP_ALIGNER_CACHE: Dict[str, "CLIPAligner"] = {}


def get_clip_aligner(model_name: str) -> "CLIPAligner":
    """Load each CLIP model at most once per batch run instead of per sample."""
    if model_name not in _CLIP_ALIGNER_CACHE:
        _CLIP_ALIGNER_CACHE[model_name] = CLIPAligner(model_name=model_name)
    return _CLIP_ALIGNER_CACHE[model_name]


# -------------------------------
# 3. 단일 문서 디렉토리 평가
# -------------------------------
def evaluate_doc_dir(doc_dir: Path, expanded_source: str, clip_aligner: CLIPAligner) -> List[FigureAlignmentResult]:
    """
    하나의 문서 디렉토리(예: outputs/req_XXX/)에 대해:
      - main.tex(+ \\input된 섹션 파일들) 읽고 figure env 파싱
      - figure image + caption 쌍에 대해 CLIP score 계산
    """
    results: List[FigureAlignmentResult] = []

    figures = extract_figures_from_tex(expanded_source)
    if not figures:
        return results

    for fig in figures:
        rel_img_path = fig["image_path"]
        caption = fig["caption"]
        label = fig["label"]

        # main.tex 기준 상대 경로 → 절대 경로
        img_path = (doc_dir / rel_img_path).resolve()
        if not img_path.is_file():
            print(f"[EvalMM] Missing image file for {label}: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[EvalMM] Failed to open image {img_path}: {e}")
            continue

        score = clip_aligner.image_text_score(img, caption)

        results.append(
            FigureAlignmentResult(
                doc_dir=str(doc_dir),
                figure_id=label,
                image_path=str(img_path),
                caption=caption,
                clip_score=score,
            )
        )

    return results


# -------------------------------
# 4. figure-caption alignment summary (단일 문서)
# -------------------------------
def evaluate_figure_alignment(doc_dir: Path, main_tex_source: str, clip_model_name: str) -> Dict:
    expanded_source = expand_tex_source(doc_dir, main_tex_source)
    figures = extract_figures_from_tex(expanded_source)
    if not figures:
        return {
            "figure_alignment_available": False,
            "message": "No figures found in this document.",
            "num_figures": 0,
            "average_clip_score": None,
            "results": [],
        }

    clip_aligner = get_clip_aligner(clip_model_name)
    fig_results = evaluate_doc_dir(doc_dir, expanded_source, clip_aligner)
    if not fig_results:
        return {
            "figure_alignment_available": False,
            "message": "Figures are declared, but no valid image-caption pairs were evaluable.",
            "num_figures": 0,
            "average_clip_score": None,
            "results": [],
        }

    avg_fig = float(sum(r.clip_score for r in fig_results) / len(fig_results))
    return {
        "figure_alignment_available": True,
        "message": "Figure-caption alignment evaluated.",
        "num_figures": len(fig_results),
        "average_clip_score": avg_fig,
        "results": [asdict(r) for r in fig_results],
    }


# -------------------------------
# 5. table-text alignment (단일 문서)
# -------------------------------
def evaluate_table_text_alignment(doc_dir: Path, main_tex_source: str) -> Dict:
    tables = extract_tables_from_doc(doc_dir, main_tex_source)
    sections = extract_sections_from_doc(doc_dir, main_tex_source)

    if not tables:
        return {
            "table_alignment_available": False,
            "message": "No tables found in this document.",
            "num_tables": 0,
            "average_table_text_alignment_score": None,
            "results": [],
        }
    if not sections:
        return {
            "table_alignment_available": False,
            "message": "Tables exist, but no section tex files were found.",
            "num_tables": len(tables),
            "average_table_text_alignment_score": None,
            "results": [],
        }

    sec_texts = [
        {
            "section_id": s["section_id"],
            "text": _clean_latex_for_text(s["source"]),
        }
        for s in sections
    ]

    results: List[TableAlignmentResult] = []
    for tbl in tables:
        table_text = _clean_latex_for_text((tbl["caption"] + " " + tbl["source"]).strip())
        best_score = -1.0
        best_section = ""
        for sec in sec_texts:
            score = _bow_cosine(table_text, sec["text"])
            if score > best_score:
                best_score = score
                best_section = sec["section_id"]
        results.append(
            TableAlignmentResult(
                doc_dir=str(doc_dir),
                table_id=tbl["table_id"],
                table_path=tbl["table_path"],
                table_caption=tbl["caption"],
                best_section=best_section,
                alignment_score=float(max(best_score, 0.0)),
            )
        )

    avg = float(sum(r.alignment_score for r in results) / len(results)) if results else None
    return {
        "table_alignment_available": True,
        "message": "Table-text alignment evaluated.",
        "num_tables": len(results),
        "average_table_text_alignment_score": avg,
        "results": [asdict(r) for r in results],
    }

def discover_sample_dirs(base_dir: Path) -> List[Path]:
    """Directories with evidence that generation was attempted (a plan.json or
    meta.json was written), regardless of whether it produced a main.tex.
    A directory with neither marker never started generating and is excluded
    entirely rather than reported as a failure.
    """
    marker_files = list(base_dir.glob("**/plan.json")) + list(base_dir.glob("**/meta.json"))
    return sorted({f.parent for f in marker_files})


def run_multimodal_batch_evaluation(args):
    # 1. 입력 경로 설정 (예: run_001)
    base_dir = Path(args.doc_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {base_dir}")

    all_results = []

    # 2. 생성 시도 흔적이 있는 디렉토리만 탐색 (factor/sample 2단 구조와, factor
    # 구분이 없는 평탄한 구조를 모두 지원)
    sample_dirs = discover_sample_dirs(base_dir)
    print(f"[EvalMM] Found {len(sample_dirs)} attempted samples in {base_dir.name}")

    for current_doc_dir in sample_dirs:
        # 경로 분석 (factor, sample_id 추출)
        relative_path = current_doc_dir.relative_to(base_dir)
        path_parts = relative_path.parts

        if len(path_parts) >= 2:
            factor_name, sample_id = path_parts[0], path_parts[1]
        elif len(path_parts) == 1:
            factor_name, sample_id = None, path_parts[0]
        else:
            factor_name, sample_id = None, "unknown"

        main_tex = current_doc_dir / "main.tex"
        if not main_tex.is_file():
            print(f"  - {relative_path}: generation attempted but main.tex is missing")
            all_results.append({
                "factor": factor_name,
                "sample_id": sample_id,
                "doc_dir": str(current_doc_dir),
                "generation_failed": True,
                "failure_reason": "main.tex not found",
                "figure_alignment": None,
                "table_text_alignment": None,
            })
            continue

        print(f"  - Evaluating: {relative_path}")

        try:
            # 3. 파일 읽기 및 개별 평가 실행
            tex_source = main_tex.read_text(encoding="utf-8", errors="ignore")

            figure_summary = evaluate_figure_alignment(current_doc_dir, tex_source, args.clip_model)
            table_summary = evaluate_table_text_alignment(current_doc_dir, tex_source)

            # 4. 결과 객체 생성 및 메타데이터 추가
            summary = {
                "factor": factor_name,
                "sample_id": sample_id,
                "doc_dir": str(current_doc_dir),
                "generation_failed": False,
                "figure_alignment": figure_summary,
                "table_text_alignment": table_summary,
            }
            all_results.append(summary)

        except Exception as e:
            print(f"  [Error] Failed to evaluate {current_doc_dir}: {e}")

    # 5. 통합 결과 저장
    out_dir = Path.cwd() / "evaluation" / "multimodal_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = f"{base_dir.name}_{Path(args.out).name}"
    out_path = out_dir / file_name

    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # 6. 요약 정보 출력 (첫 번째 결과 기준 또는 전체 평균 등)
    failed = sum(1 for r in all_results if r.get("generation_failed"))
    print(f"\n[EvalMM] Batch Evaluation Done. Total: {len(all_results)} samples ({failed} never produced a main.tex).")
    print(f"[EvalMM] Results saved to {out_path}")

# -------------------------------
# 6. CLI entry
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multimodal alignment for a single generated document."
    )
    parser.add_argument(
        "--doc-dir",
        default="outputs/batch_test_07281829/basic",
        help="Batch output directory containing one or more sample subdirectories.",
    )
    parser.add_argument(
        "--out",
        default="multimodal_eval.json",
        help="Output JSON file to save evaluation results.",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model name (Hugging Face identifier).",
    )

    args = parser.parse_args()

    run_multimodal_batch_evaluation(args)

    # doc_dir = Path(args.doc_dir)
    # main_tex = doc_dir / "main.tex"
    # if not main_tex.is_file():
    #     raise FileNotFoundError(f"main.tex not found: {main_tex}")

    # tex_source = main_tex.read_text(encoding="utf-8", errors="ignore")
    # figure_summary = evaluate_figure_alignment(doc_dir, tex_source, args.clip_model)
    # table_summary = evaluate_table_text_alignment(doc_dir, tex_source)

    # summary = {
    #     "doc_dir": str(doc_dir),
    #     "figure_alignment": figure_summary,
    #     "table_text_alignment": table_summary,
    # }

    # out_dir = Path.cwd() / "evaluation" / "multimodal_alignment" 
    # out_dir.mkdir(parents=True, exist_ok=True)
    # out_path = out_dir / Path(args.out).name
    # out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # print(f"[EvalMM] Done. Results saved to {out_path}")
    # print(f"[EvalMM] Figure alignment: {summary['figure_alignment']['message']}")
    # print(f"[EvalMM] Avg figure CLIP score: {summary['figure_alignment']['average_clip_score']}")
    # print(f"[EvalMM] Table alignment: {summary['table_text_alignment']['message']}")


if __name__ == "__main__":
    main()

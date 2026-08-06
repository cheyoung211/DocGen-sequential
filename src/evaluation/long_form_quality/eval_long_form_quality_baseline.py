from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

TOPIC_KEYWORDS = {
    "history": ["history", "historical", "origin", "evolution", "development"],
    "types": ["types", "kind", "variety", "acoustic", "electric", "classical"],
    "structure": ["structure", "parts", "component", "body", "neck", "string"],
    "usage": ["use", "usage", "how to", "play", "beginner", "technique"],
}

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def strip_latex(source: str) -> str:
    """LaTeX 문법을 제거하고 순수 텍스트만 추출합니다."""
    text = re.sub(r"%.*", " ", source)
    # 섹션 태그 등 주요 명령어 제거 (단, 섹션 제목 추출을 위해 별도 처리 후 호출 권장)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
    text = re.sub(r"[{}\\\\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_sections_from_main(tex_content: str) -> dict[str, str]:
    """main.tex 파일에서 section을 기준으로 텍스트를 분할합니다."""
    # \section{Title} 패턴을 찾습니다.
    sections = re.split(r"\\section\*?\{([^{}]+)\}", tex_content)
    
    section_texts = {}
    if len(sections) > 1:
        # split 결과: [intro_text, title1, content1, title2, content2, ...]
        intro = sections[0].strip()
        if intro:
            section_texts["Introduction/Preamble"] = strip_latex(intro)
            
        for i in range(1, len(sections), 2):
            title = sections[i].strip()
            content = sections[i+1].strip() if i+1 < len(sections) else ""
            section_texts[title] = strip_latex(content)
    else:
        # section 구분이 없는 경우 전체를 하나로 처리
        section_texts["Full Document"] = strip_latex(tex_content)
        
    return section_texts

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def cosine_text(a: str, b: str) -> float:
    ca = Counter(tokenize(a))
    cb = Counter(tokenize(b))
    if not ca or not cb: return 0.0
    common = set(ca) & set(cb)
    dot = sum(ca[k] * cb[k] for k in common)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return float(dot / (na * nb)) if na * nb > 0 else 0.0

def infer_required_topics(instruction: str, section_titles: list[str]) -> dict[str, list[str]]:
    text = instruction.lower().strip()
    required: dict[str, list[str]] = {}
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in text for kw in kws):
            required[topic] = kws
    
    if not required:
        # fallback: main.tex에서 추출한 섹션 제목 기반
        for title in section_titles:
            title_lower = title.lower()
            for topic, kws in TOPIC_KEYWORDS.items():
                if any(kw in title_lower for kw in kws):
                    required[topic] = kws
    return required

def section_coherence(section_texts: dict[str, str]) -> dict[str, Any]:
    titles = list(section_texts.keys())
    pairs = []
    sims = []
    for i in range(len(titles) - 1):
        a = section_texts[titles[i]]
        b = section_texts[titles[i+1]]
        sim = cosine_text(a, b)
        sims.append(sim)
        pairs.append({"from": titles[i], "to": titles[i+1], "similarity": sim})
    score = float(sum(sims) / len(sims)) if sims else None
    return {"score": score, "pairwise": pairs}

def content_coverage(required_topics: dict[str, list[str]], corpus_text: str) -> dict[str, Any]:
    covered = 0
    total = len(required_topics)
    details = []
    lower = corpus_text.lower()
    for topic, kws in required_topics.items():
        hit_keywords = [kw for kw in kws if kw in lower]
        ok = len(hit_keywords) > 0
        if ok: covered += 1
        details.append({"topic": topic, "covered": ok, "matched_keywords": hit_keywords})
    return {"score": float(covered / total) if total > 0 else None, "details": details}

def length_adequacy(total_words: int, target_min: int, target_max: int) -> dict[str, Any]:
    if target_min <= total_words <= target_max:
        score = 1.0
    elif total_words < target_min:
        score = max(0.0, float(total_words / max(target_min, 1)))
    else:
        score = max(0.0, float(target_max / max(total_words, 1)))
    return {"score": score, "total_words": total_words, "within_target": target_min <= total_words <= target_max}

def evaluate_doc(doc_dir: Path, target_min: int, target_max: int) -> dict[str, Any]:
    # 1. Load Metadata (Handle Single LLM naming)
    metadata_path = doc_dir / "meta.json"  ########### meta -> metadata
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.json not found in {doc_dir}")
    # 'instruction_given' 대신 'instruction' 사용
    instruction = read_json(metadata_path).get("instruction", "")

    # 2. Load main.tex (Instead of plan.json + sections/)
    main_tex_path = doc_dir / "main.tex"
    if not main_tex_path.is_file():
        raise FileNotFoundError(f"main.tex not found in {doc_dir}")
    raw_tex = main_tex_path.read_text(encoding="utf-8", errors="ignore")
    
    # 3. Parse Sections & Corpus
    section_texts = parse_sections_from_main(raw_tex)
    corpus = "\n".join(section_texts.values())
    total_words = len(tokenize(corpus))
    
    # 4. Run Evaluations
    required_topics = infer_required_topics(instruction, list(section_texts.keys()))
    coherence_result = section_coherence(section_texts)
    coverage_result = content_coverage(required_topics, corpus)
    length_result = length_adequacy(total_words, target_min, target_max)

    return {
        "doc_dir": str(doc_dir),
        "core_metric_summary": {
            "section_coherence_score": coherence_result["score"],
            "length_adequacy_score": length_result["score"],
            "content_coverage_score": coverage_result["score"],
        },
        "details": {
            "section_coherence": coherence_result,
            "content_coverage": coverage_result,
            "length_adequacy": length_result,
            "detected_sections": list(section_texts.keys())
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate single LLM long-form output.")
    parser.add_argument("--doc-dir", required=True, help="Path to the single LLM output directory.")
    parser.add_argument("--target-min-words", type=int, default=1200)
    parser.add_argument("--target-max-words", type=int, default=2500)
    parser.add_argument("--out", default="single_llm_quality_eval.json")
    args = parser.parse_args()

    try:
        summary = evaluate_doc(
            doc_dir=Path(args.doc_dir),
            target_min=args.target_min_words,
            target_max=args.target_max_words,
        )

        out_dir = Path.cwd() / "evaluation" / "long_form_quality"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / args.out
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[SingleEval] Saved: {out_path}")
        print(f"[SingleEval] Coherence: {summary['core_metric_summary']['section_coherence_score']:.2f}")
    except Exception as e:
        print(f"[Error] {e}")

if __name__ == "__main__":
    main()
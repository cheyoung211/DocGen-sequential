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

BEGINNER_TERMS = [
    "beginner",
    "first-time",
    "first time",
    "basic",
    "easy",
    "step-by-step",
    "practice",
    "tips",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_outline(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Section id/title pairs, from either the legacy ``doc_outline`` field or
    the current planner's ``blueprint.sections`` (which carries the same
    ``section_id``/``title`` shape)."""
    outline = plan.get("doc_outline")
    if outline:
        return outline
    return plan.get("blueprint", {}).get("sections", [])


def strip_latex(source: str) -> str:
    text = re.sub(r"%.*", " ", source)
    text = re.sub(r"\\(section|subsection)\*?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\label\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
    text = re.sub(r"[{}\\\\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def normalize_name(text: str) -> str:
    return " ".join(tokenize(text))


def cosine_text(a: str, b: str) -> float:
    ca = Counter(tokenize(a))
    cb = Counter(tokenize(b))
    if not ca or not cb:
        return 0.0
    common = set(ca) & set(cb)
    dot = sum(ca[k] * cb[k] for k in common)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def infer_required_topics(instruction: str, plan: dict[str, Any]) -> dict[str, list[str]]:
    text = instruction.lower().strip()
    required: dict[str, list[str]] = {}

    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in text for kw in kws):
            required[topic] = kws

    if not required:
        # fallback: plan section title 기반으로 topic 추정
        for sec in get_outline(plan):
            title = f"{sec.get('section_id', '')} {sec.get('title', '')}".lower()
            for topic, kws in TOPIC_KEYWORDS.items():
                if any(kw in title for kw in kws):
                    required[topic] = kws
    return required


def extract_constraints(instruction: str) -> dict[str, Any]:
    t = instruction.lower()
    return {
        "history_brief_required": bool(re.search(r"history.*(brief|short|concise)", t)),
        "beginner_oriented_required": ("beginner" in t) or ("first time" in t) or ("first-time" in t),
    }


def find_history_section(section_texts: dict[str, str]) -> str | None:
    for sec_id in section_texts:
        if "history" in sec_id.lower():
            return sec_id
    return None


def count_words(text: str) -> int:
    return len(tokenize(text))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def section_coherence(section_texts: dict[str, str], section_order: list[str]) -> dict[str, Any]:
    pairs = []
    sims = []
    for i in range(len(section_order) - 1):
        a_id = section_order[i]
        b_id = section_order[i + 1]
        a = section_texts.get(a_id, "")
        b = section_texts.get(b_id, "")
        sim = cosine_text(a, b)
        sims.append(sim)
        pairs.append({"from": a_id, "to": b_id, "similarity": sim})

    score = float(sum(sims) / len(sims)) if sims else None
    return {"score": score, "pairwise": pairs}


def topic_consistency(section_texts: dict[str, str], plan: dict[str, Any]) -> dict[str, Any]:
    per_section = []
    vals = []
    for sec in get_outline(plan):
        sec_id = sec.get("section_id", "")
        title = sec.get("title", "")
        text = section_texts.get(sec_id, "")
        sim = cosine_text(title, text)
        vals.append(sim)
        per_section.append({"section_id": sec_id, "title": title, "consistency": sim})
    score = float(sum(vals) / len(vals)) if vals else None
    return {"score": score, "per_section": per_section}


def content_coverage(required_topics: dict[str, list[str]], corpus_text: str) -> dict[str, Any]:
    covered = 0
    total = len(required_topics)
    details = []
    lower = corpus_text.lower()
    for topic, kws in required_topics.items():
        hit_keywords = [kw for kw in kws if kw in lower]
        ok = len(hit_keywords) > 0
        if ok:
            covered += 1
        details.append({"topic": topic, "covered": ok, "matched_keywords": hit_keywords})
    score = float(covered / total) if total > 0 else None
    return {"score": score, "covered_topics": covered, "total_topics": total, "details": details}


def global_fact_consistency(plan: dict[str, Any], corpus_text: str) -> dict[str, Any]:
    facts = plan.get("global_facts", {}) or {}
    if not isinstance(facts, dict) or not facts:
        return {"score": None, "message": "No global_facts in plan.", "details": []}

    details = []
    consistent = 0
    total = 0
    lower = corpus_text.lower()
    for k, v in facts.items():
        v_str = str(v).strip()
        if not v_str:
            continue
        total += 1
        ok = v_str.lower() in lower
        if ok:
            consistent += 1
        details.append({"fact_key": str(k), "fact_value": v_str, "consistent": ok})

    score = float(consistent / total) if total > 0 else None
    return {
        "score": score,
        "consistent_references": consistent,
        "total_global_facts": total,
        "details": details,
    }


def length_adequacy(section_texts: dict[str, str], target_min: int, target_max: int) -> dict[str, Any]:
    per_section = {k: count_words(v) for k, v in section_texts.items()}
    total_words = sum(per_section.values())

    if target_min <= total_words <= target_max:
        score = 1.0
    elif total_words < target_min:
        score = max(0.0, float(total_words / max(target_min, 1)))
    else:
        score = max(0.0, float(target_max / max(total_words, 1)))

    return {
        "score": score,
        "total_words": total_words,
        "target_range": [target_min, target_max],
        "within_target": target_min <= total_words <= target_max,
        "words_per_section": per_section,
    }


def required_section_coverage(required_sections: list[str], plan: dict[str, Any], section_texts: dict[str, str]) -> dict[str, Any]:
    required = [s.strip() for s in required_sections if s and s.strip()]
    if not required:
        return {
            "score": None,
            "message": "No required sections provided.",
            "required_sections": [],
            "matched_sections": [],
            "missing_sections": [],
        }

    available_names: list[str] = []
    for sec in get_outline(plan):
        sec_id = str(sec.get("section_id", "")).strip()
        title = str(sec.get("title", "")).strip()
        if sec_id:
            available_names.append(sec_id)
        if title:
            available_names.append(title)
    available_names.extend(list(section_texts.keys()))

    normalized_available = [normalize_name(x) for x in available_names if x]
    matched: list[str] = []
    missing: list[str] = []

    for req in required:
        nreq = normalize_name(req)
        ok = any((nreq == cand) or (nreq in cand) or (cand in nreq) for cand in normalized_available)
        if ok:
            matched.append(req)
        else:
            missing.append(req)

    return {
        "score": float(len(matched) / len(required)),
        "required_sections": required,
        "matched_sections": matched,
        "missing_sections": missing,
        "matched_count": len(matched),
        "required_count": len(required),
    }


def load_sections(doc_dir: Path, plan: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    sec_dir = doc_dir / "sections"
    section_order = []
    section_texts: dict[str, str] = {}

    for sec in get_outline(plan):
        sec_id = sec.get("section_id", "")
        if not sec_id:
            continue
        path = sec_dir / f"{sec_id}.tex"
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        section_order.append(sec_id)
        section_texts[sec_id] = strip_latex(raw)

    # fallback: plan 기준 파일이 비어 있으면 실제 파일 전수 사용
    if not section_texts and sec_dir.is_dir():
        for p in sorted(sec_dir.glob("*.tex")):
            section_order.append(p.stem)
            section_texts[p.stem] = strip_latex(p.read_text(encoding="utf-8", errors="ignore"))

    return section_order, section_texts


def evaluate_doc(
    doc_dir: Path,
    # instruction: str,
    target_min: int,
    target_max: int,
    # required_sections: list[str] | None = None,
) -> dict[str, Any]:
    plan_path = doc_dir / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"plan.json not found: {plan_path}")
    plan = read_json(plan_path)

    # metadata.json is only written by older runs; the current pipeline keeps
    # the planner's restatement of the request in plan.json's global_context.
    metadata_path = doc_dir / "metadata.json"
    if metadata_path.is_file():
        instruction = read_json(metadata_path).get("instruction_given", "")
    else:
        instruction = plan.get("global_context", "")

    section_order, section_texts = load_sections(doc_dir, plan)
    if not section_texts:
        raise FileNotFoundError(f"No section tex files found under: {doc_dir / 'sections'}")

    corpus = "\n".join(section_texts[sid] for sid in section_order if sid in section_texts)
    required_topics = infer_required_topics(instruction, plan)

    # instruction_result = instruction_compliance(instruction, required_topics, section_texts, corpus)
    coherence_result = section_coherence(section_texts, section_order)
    topic_result = topic_consistency(section_texts, plan)
    coverage_result = content_coverage(required_topics, corpus)
    length_result = length_adequacy(section_texts, target_min=target_min, target_max=target_max)
    fact_result = global_fact_consistency(plan, corpus)
    # section_req_result = required_section_coverage(required_sections or [], plan, section_texts)
    section_req_result = required_section_coverage([], plan, section_texts)


    return {
        "doc_dir": str(doc_dir),
        "core_metric_summary": {
            # "instruction_compliance_score": instruction_result["score"],
            "section_coherence_score": coherence_result["score"],
            "topic_consistency_score": topic_result["score"],
            "length_adequacy_score": length_result["score"],
            "content_coverage_score": coverage_result["score"],
            "required_section_coverage_score": section_req_result["score"],
        },
        "automatic_evaluation_methods": {
            "section_coherence": coherence_result,
            "global_fact_consistency": fact_result,
            "content_coverage": coverage_result,
            "length_adequacy": length_result,
            "required_section_coverage": section_req_result,
        },
    }


def discover_sample_dirs(base_dir: Path) -> list[Path]:
    """Directories with evidence that generation was attempted (a plan.json or
    meta.json was written), regardless of whether it produced section files.
    A directory with neither marker never started generating and is excluded
    entirely rather than reported as a failure.
    """
    marker_files = list(base_dir.glob("**/plan.json")) + list(base_dir.glob("**/meta.json"))
    return sorted({f.parent for f in marker_files})


def run_longform_batch_evaluation(args):
    # 1. 입력 경로 설정 (예: run_001)
    base_dir = Path(args.doc_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {base_dir}")

    all_results = []

    # 2. 하위 디렉토리 탐색 (생성 시도 흔적이 있는 디렉토리만; factor/sample 2단
    # 구조와 factor 구분이 없는 평탄한 구조를 모두 지원한다.)
    sample_dirs = discover_sample_dirs(base_dir)

    print(f"[LongFormEval] Found {len(sample_dirs)} attempted samples in {base_dir.name}")

    for doc_path in sample_dirs:
        relative_path = doc_path.relative_to(base_dir)
        path_parts = relative_path.parts

        if len(path_parts) >= 2:
            factor_name, sample_id = path_parts[0], path_parts[1]
        elif len(path_parts) == 1:
            factor_name, sample_id = None, path_parts[0]
        else:
            factor_name, sample_id = None, "unknown"

        print(f"  - Evaluating: {relative_path}")

        try:
            # 3. 개별 샘플 평가 실행
            summary = evaluate_doc(
                doc_dir=doc_path,
                target_min=args.target_min_words,
                target_max=args.target_max_words,
            )

            # 4. 결과에 메타데이터 추가 및 리스트에 저장
            summary["factor"] = factor_name
            summary["sample_id"] = sample_id
            summary["generation_failed"] = False
            all_results.append(summary)

        except FileNotFoundError as e:
            # plan.json은 있었지만(=시도는 확인됨) 이후 단계 산출물이 없는 경우
            print(f"  [GenerationFailed] {relative_path}: {e}")
            all_results.append({
                "doc_dir": str(doc_path),
                "factor": factor_name,
                "sample_id": sample_id,
                "generation_failed": True,
                "failure_reason": str(e),
                "core_metric_summary": None,
                "automatic_evaluation_methods": None,
            })
        except Exception as e:
            print(f"  [Error] Failed to evaluate {doc_path}: {e}")

    # 5. 최종 결과 통합 저장
    out_dir = Path.cwd() / "evaluation" / "long_form_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = f"{base_dir.name}_{Path(args.out).name}"
    out_path = out_dir / file_name

    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    failed = sum(1 for r in all_results if r.get("generation_failed"))
    print(f"[LongFormEval] {failed}/{len(all_results)} attempted samples failed to produce evaluable output.")
    print(f"[LongFormEval] Saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate long-form quality for one generated output directory.")
    parser.add_argument(
        "--doc-dir",
        default="outputs/batch_test_07281829/basic",
        help="Path to a batch output directory containing one or more sample subdirectories.",
    )
    # parser.add_argument(
    #     "--instruction",
    #     default="",
    #     help="Original user instruction text. If omitted, topic coverage falls back to plan section titles.",
    # )
    # parser.add_argument(
    #     "--instruction-file",
    #     default=None,
    #     help="Optional file containing instruction text. Used when --instruction is empty.",
    # )
    parser.add_argument("--target-min-words", type=int, default=1200)
    parser.add_argument("--target-max-words", type=int, default=2500)
    parser.add_argument(
        "--required-sections",
        default="",
        help="Comma-separated required section names (ids or titles).",
    )
    # parser.add_argument(
    #     "--required-sections-file",
    #     default=None,
    #     help="Optional text file with one required section name per line.",
    # )
    parser.add_argument("--out", default="long_form_quality_eval.json")
    args = parser.parse_args()

    # instruction = args.instruction.strip()
    # if not instruction and args.instruction_file:
    #     instruction = Path(args.instruction_file).read_text(encoding="utf-8", errors="ignore").strip()

    # required_sections: list[str] = []
    # if args.required_sections.strip():
    #     required_sections.extend([x.strip() for x in args.required_sections.split(",") if x.strip()])
    # if args.required_sections_file:
    #     required_sections.extend(
    #         [
    #             line.strip()
    #             for line in Path(args.required_sections_file).read_text(encoding="utf-8", errors="ignore").splitlines()
    #             if line.strip()
    #         ]
    #     )

    # summary = evaluate_doc(
    #     doc_dir=Path(args.doc_dir),
    #     # instruction=instruction,
    #     target_min=args.target_min_words,
    #     target_max=args.target_max_words,
    #     # required_sections=required_sections,
    # )
    run_longform_batch_evaluation(args)

    # out_dir = Path.cwd() / "evaluation" / "long_form_quality"
    # out_dir.mkdir(parents=True, exist_ok=True)
    # out_path = out_dir / Path(args.out).name
    # out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # print(f"[LongFormEval] Saved: {out_path}")
    # # print(f"[LongFormEval] Core metrics: {summary['core_metric_summary']}")


if __name__ == "__main__":
    main()

"""Summarize what one generation batch is made of: per-domain averages
(section/block/LLM-call/outer-iteration counts) and the compiler-engine
split (tectonic vs. the pdflatex fallback), plus whether that engine
supports compile-error location attribution.

Reads only files already on disk under a batch's output directory
(``run_result.json``, ``plan.json``, ``sections/*.blocks.json``) -- no
re-running of the pipeline.

Usage:
    python3 scripts/batch_composition_summary.py --dir outputs/<batch_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Factor-id prefix -> domain label. Matches dataset2/benchmark's 4 domains
# (30 factors each, 120 total) -- see src/dataset/source_adapters/manual_seed.py.
DOMAIN_PREFIXES = {
    "fin_econ_": "fin_econ",
    "math_": "math",
    "sci_eng_": "sci_eng",
    "stats_ml_": "stats_ml",
}


def domain_of(factor: str) -> str:
    for prefix, label in DOMAIN_PREFIXES.items():
        if factor.startswith(prefix):
            return label
    return "unknown"


def find_sample_dirs(batch_dir: Path) -> List[Path]:
    """Sample dirs with a run_result.json (regardless of whether the run
    ever produced a plan.json -- e.g. a total planner failure)."""
    return sorted({p.parent for p in batch_dir.glob("*/*/run_result.json")})


def section_and_block_counts(sample_dir: Path) -> Optional[Dict[str, int]]:
    """Ground-truth section/block counts from what was actually rendered
    on disk, not the planned blueprint (a section that never finished
    drafting has no sections/<id>.tex, so it correctly doesn't count)."""
    sections_dir = sample_dir / "sections"
    if not sections_dir.is_dir():
        return None
    tex_files = [p for p in sections_dir.glob("*.tex") if not p.name.endswith(".draft.tex")]
    block_count = 0
    for blocks_json in sections_dir.glob("*.blocks.json"):
        try:
            data = json.loads(blocks_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        block_count += len(data.get("blocks", []))
    return {"sections": len(tex_files), "blocks": block_count}


def outer_iteration_count(run_result: dict) -> int:
    """Max outer_iteration seen across this run's verification events
    (the Phase-2 TextAgent loop); 1 if the run never iterated at all."""
    values = [
        ev.get("outer_iteration")
        for ev in run_result.get("verification_events", [])
        if ev.get("outer_iteration") is not None
    ]
    return max(values) if values else 1


def llm_call_count(run_result: dict) -> int:
    """Actual LLM API call count (token_usage.call_count), not the
    generation-attempt proxy -- see TokenUsage/RunResult in
    src/evaluation/schemas.py."""
    token_usage = run_result.get("token_usage") or {}
    return token_usage.get("call_count", 0)


def engine_supports_location_attribution(engine_name: Optional[str]) -> bool:
    """Mirrors src/evaluation/compile_log_parser.py::extract_error_locations,
    which returns [] outright for any non-tectonic engine name -- pdflatex
    logs have no per-section-file location markers to parse."""
    return bool(engine_name) and "tectonic" in engine_name.lower()


def summarize(batch_dir: Path) -> None:
    sample_dirs = find_sample_dirs(batch_dir)
    if not sample_dirs:
        print(f"No run_result.json found under {batch_dir}", file=sys.stderr)
        sys.exit(1)

    per_domain: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"sections": [], "blocks": [], "llm_calls": [], "outer_iterations": []}
    )
    engine_counts: Dict[str, int] = defaultdict(int)

    for sample_dir in sample_dirs:
        run_result = json.loads((sample_dir / "run_result.json").read_text(encoding="utf-8"))
        factor = run_result.get("benchmark_id") or sample_dir.parent.name
        domain = domain_of(factor)

        # A run that never got past the planner has no sections/ dir at all
        # (see e.g. all_verifiers_08201151's 7 total-planner-failure
        # factors) -- counted as 0 sections/blocks, not excluded, so every
        # domain's average has the same denominator as its "문서 수" column.
        counts = section_and_block_counts(sample_dir) or {"sections": 0, "blocks": 0}
        per_domain[domain]["sections"].append(counts["sections"])
        per_domain[domain]["blocks"].append(counts["blocks"])
        per_domain[domain]["llm_calls"].append(llm_call_count(run_result))
        per_domain[domain]["outer_iterations"].append(outer_iteration_count(run_result))

        compile_result = run_result.get("compile_result") or {}
        engine = compile_result.get("engine")
        if engine:
            engine_counts[engine] += 1
        else:
            engine_counts["(no compile attempt)"] += 1

    # --- Table 1: per-domain composition ---
    print("=== 분야별 batch 구성 ===")
    header = f"{'분야':<10}{'문서 수':>8}{'평균 section 수':>16}{'평균 block 수':>14}{'평균 LLM 호출 수':>18}{'평균 outer iteration 수':>24}"
    print(header)
    total_docs = 0
    all_sections: List[float] = []
    all_blocks: List[float] = []
    all_calls: List[float] = []
    all_outer: List[float] = []
    for domain in sorted(per_domain):
        stats = per_domain[domain]
        n_docs = len(stats["outer_iterations"])
        total_docs += n_docs
        all_sections += stats["sections"]
        all_blocks += stats["blocks"]
        all_calls += stats["llm_calls"]
        all_outer += stats["outer_iterations"]
        print(
            f"{domain:<10}{n_docs:>8}"
            f"{mean(stats['sections']):>16.2f}"
            f"{mean(stats['blocks']):>14.2f}"
            f"{mean(stats['llm_calls']):>18.2f}"
            f"{mean(stats['outer_iterations']):>24.2f}"
        )
    print(
        f"{'전체':<10}{total_docs:>8}"
        f"{mean(all_sections):>16.2f}"
        f"{mean(all_blocks):>14.2f}"
        f"{mean(all_calls):>18.2f}"
        f"{mean(all_outer):>24.2f}"
    )

    # --- Table 2: compiler engine split ---
    print()
    print("=== Compiler engine 구성 ===")
    total_runs = sum(engine_counts.values())
    print(f"{'Compiler engine':<24}{'실행 수':>8}{'비율':>10}{'오류 위치 attribution 가능 여부':>28}")
    for engine, count in sorted(engine_counts.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * count / total_runs if total_runs else 0.0
        attribution = "가능" if engine_supports_location_attribution(engine) else "불가능"
        print(f"{engine:<24}{count:>8}{pct:>9.1f}%{attribution:>28}")
    print(f"{'전체':<24}{total_runs:>8}{100.0:>9.1f}%{'—':>28}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Batch output directory (e.g. outputs/baseline_run_08201340)")
    args = parser.parse_args()
    summarize(Path(args.dir))


if __name__ == "__main__":
    main()

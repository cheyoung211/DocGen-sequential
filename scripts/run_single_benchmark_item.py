"""Run the full multi-stage DocGen pipeline on a single item from the new
LaTeX-focused benchmark (data/benchmark/benchmark_v1.jsonl), so the pipeline
can be sanity-checked against it and the output inspected directly.

Usage:
    python3 scripts/run_single_benchmark_item.py
    python3 scripts/run_single_benchmark_item.py --sample-id math_real_analysis_0001
    python3 scripts/run_single_benchmark_item.py --sample-id fin_econ_valuation_concepts_0021 --image-model sdxl
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from run_full_pipeline import DocGenPipeline, build_user_query  # noqa: E402
from llm_client import DEFAULT_MODEL  # noqa: E402


def load_item(dataset_path: Path, sample_id: Optional[str]) -> Dict[str, Any]:
    with dataset_path.open("r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if not items:
        raise SystemExit(f"No items found in {dataset_path}")
    if sample_id is None:
        return items[0]
    for item in items:
        if item.get("sample_id") == sample_id:
            return item
    raise SystemExit(f"sample_id {sample_id!r} not found in {dataset_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one benchmark_v1 item through the full DocGen pipeline.")
    parser.add_argument("--dataset", default="data/benchmark/benchmark_v1.jsonl", help="Path to the benchmark JSONL.")
    parser.add_argument("--sample-id", default=None, help="sample_id to run (default: first item in the file).")
    parser.add_argument("--out-dir", default="outputs/single_item_check", help="Where to write the generated document.")
    parser.add_argument("--image-model", default="flux", choices=["sdxl", "flux"])
    parser.add_argument("--iterations", type=int, default=5, help="Max text-agent iterations (see DocGenPipeline.run).")
    parser.add_argument(
        "--keep-previous-run", action="store_true",
        help="Do not wipe this sample_id's output directory before running. "
             "Off by default: re-running the same sample_id otherwise leaves stale "
             "sections/*.tex from a prior (possibly differently-planned) attempt "
             "mixed in with the new ones, which is confusing to inspect.",
    )
    args = parser.parse_args()

    item = load_item(REPO_ROOT / args.dataset, args.sample_id)
    print(f"[single-item check] sample_id={item['sample_id']!r} title={item['title']!r} "
          f"domain={item['domain']} difficulty={item['difficulty']}")

    result_dir = Path(args.out_dir) / item["sample_id"]
    if result_dir.exists() and not args.keep_previous_run:
        print(f"[single-item check] Removing previous run at {result_dir} (pass --keep-previous-run to disable).")
        shutil.rmtree(result_dir)

    pipeline = DocGenPipeline(
        planner_model=DEFAULT_MODEL,
        text_model=DEFAULT_MODEL,
        image_model=args.image_model,
        integrator_model=DEFAULT_MODEL,
        output_dir=args.out_dir,
    )
    pipeline.run(
        user_query=build_user_query(item),
        request_id=item["sample_id"],
        iterations=args.iterations,
    )

    # DocGenPipeline.run() never raises or returns a success flag on a
    # compilation failure -- it only prints a warning line -- so check the
    # actual compiled artifact here and fail loudly and unambiguously instead
    # of letting a broken document look like a normal finish.
    pdf_path = result_dir / "main.pdf"
    log_path = result_dir / "compile.log"
    print(f"\n[single-item check] Inspect the result at: {result_dir}")
    print(f"  - {result_dir / 'plan.json'}   (planner output / DocumentBlueprint)")
    print(f"  - {result_dir / 'main.tex'}     (composed LaTeX)")
    if pdf_path.exists():
        print(f"  - {pdf_path}     (compiled PDF)")
        print("\n[single-item check] RESULT: PASS (main.pdf was produced).")
    else:
        print(f"\n[single-item check] RESULT: FAIL -- no main.pdf was produced.")
        if log_path.exists():
            tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-20:]
            print(f"[single-item check] Last lines of {log_path}:")
            for line in tail:
                print(f"    {line}")
        else:
            print(f"[single-item check] No {log_path} either -- compilation likely never ran (no LaTeX engine found?).")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

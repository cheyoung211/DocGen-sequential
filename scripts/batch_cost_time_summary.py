"""Compare wall-clock time and LLM token cost between two generation
batches (e.g. a no-verifier baseline vs. an all-verifiers run).

Reads only ``run_result.json``'s ``elapsed_time`` (per-request wall clock,
set in scripts/run_full_pipeline.py::_build_run_result) and ``token_usage``
(real API call counts/tokens, not the generation-attempt proxy) -- no
re-running of the pipeline.

Usage:
    python3 scripts/batch_cost_time_summary.py \\
        --baseline outputs/baseline_run_08201340 \\
        --treatment outputs/all_verifiers_08201151 \\
        [--baseline-label "no verifier"] [--treatment-label "all verifiers"] \\
        [--input-price-per-1m 0.15] [--output-price-per-1m 0.60]

Default prices are gpt-4o-mini's published per-1M-token rates at time of
writing -- pass --input-price-per-1m/--output-price-per-1m to override if
they've since changed or you're pricing a different model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def load_runs(batch_dir: Path) -> List[dict]:
    runs = []
    for f in sorted(batch_dir.glob("*/*/run_result.json")):
        runs.append(json.loads(f.read_text(encoding="utf-8")))
    return runs


def summarize(runs: List[dict], input_price: float, output_price: float) -> Dict[str, float]:
    elapsed = [r["elapsed_time"] for r in runs if r.get("elapsed_time") is not None]
    input_tokens = [
        (r.get("token_usage") or {}).get("total_input_tokens", 0) for r in runs
    ]
    output_tokens = [
        (r.get("token_usage") or {}).get("total_output_tokens", 0) for r in runs
    ]
    calls = [(r.get("token_usage") or {}).get("call_count", 0) for r in runs]

    total_input = sum(input_tokens)
    total_output = sum(output_tokens)
    total_cost = total_input / 1_000_000 * input_price + total_output / 1_000_000 * output_price

    return {
        "n_docs": len(runs),
        "total_elapsed_sec": sum(elapsed),
        "avg_elapsed_sec": mean(elapsed) if elapsed else float("nan"),
        "avg_llm_calls": mean(calls) if calls else float("nan"),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "avg_input_tokens": mean(input_tokens) if input_tokens else float("nan"),
        "avg_output_tokens": mean(output_tokens) if output_tokens else float("nan"),
        "total_cost_usd": total_cost,
        "avg_cost_usd": total_cost / len(runs) if runs else float("nan"),
    }


def print_comparison(
    baseline_stats: Dict[str, float],
    treatment_stats: Dict[str, float],
    baseline_label: str,
    treatment_label: str,
) -> None:
    rows = [
        ("문서 수", "n_docs", "{:.0f}"),
        ("총 소요 시간 (초)", "total_elapsed_sec", "{:.1f}"),
        ("문서당 평균 소요 시간 (초)", "avg_elapsed_sec", "{:.2f}"),
        ("문서당 평균 LLM 호출 수", "avg_llm_calls", "{:.2f}"),
        ("총 input 토큰", "total_input_tokens", "{:.0f}"),
        ("총 output 토큰", "total_output_tokens", "{:.0f}"),
        ("문서당 평균 input 토큰", "avg_input_tokens", "{:.1f}"),
        ("문서당 평균 output 토큰", "avg_output_tokens", "{:.1f}"),
        ("총 예상 비용 (USD)", "total_cost_usd", "{:.4f}"),
        ("문서당 평균 예상 비용 (USD)", "avg_cost_usd", "{:.5f}"),
    ]
    label_w = 30
    col_w = 18
    print(
        f"{'지표':<{label_w}}{baseline_label:>{col_w}} {treatment_label:>{col_w}} "
        f"{'차이(treat-base)':>{col_w}} {'배율':>8}"
    )
    for label, key, fmt in rows:
        b = baseline_stats[key]
        t = treatment_stats[key]
        diff = t - b
        ratio = (t / b) if b else float("nan")
        print(
            f"{label:<{label_w}}"
            f"{fmt.format(b):>{col_w}} "
            f"{fmt.format(t):>{col_w}} "
            f"{fmt.format(diff):>{col_w}} "
            f"{ratio:>7.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="Baseline batch output dir")
    parser.add_argument("--treatment", required=True, help="Treatment (e.g. all-verifiers) batch output dir")
    parser.add_argument("--baseline-label", default="no verifier")
    parser.add_argument("--treatment-label", default="all verifiers")
    parser.add_argument("--input-price-per-1m", type=float, default=0.15, help="USD per 1M input tokens")
    parser.add_argument("--output-price-per-1m", type=float, default=0.60, help="USD per 1M output tokens")
    args = parser.parse_args()

    baseline_runs = load_runs(Path(args.baseline))
    treatment_runs = load_runs(Path(args.treatment))
    if not baseline_runs or not treatment_runs:
        print("No run_result.json found in one of the given directories.", file=sys.stderr)
        sys.exit(1)

    baseline_stats = summarize(baseline_runs, args.input_price_per_1m, args.output_price_per_1m)
    treatment_stats = summarize(treatment_runs, args.input_price_per_1m, args.output_price_per_1m)

    print(
        f"(비용은 gpt-4o-mini 기준 input ${args.input_price_per_1m}/1M, "
        f"output ${args.output_price_per_1m}/1M 가정 -- 실제 계약된 단가로 --input-price-per-1m/"
        f"--output-price-per-1m 조정 권장)"
    )
    print()
    print_comparison(baseline_stats, treatment_stats, args.baseline_label, args.treatment_label)


if __name__ == "__main__":
    main()

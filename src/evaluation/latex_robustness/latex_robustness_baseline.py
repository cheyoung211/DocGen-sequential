from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json

from latex_robustness import evaluate_tex


def collect_main_tex_files(base_dir: Path) -> list[Path]:
    return sorted(base_dir.glob("**/main.tex"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="baseline/outputs_baseline",
        help="Directory containing baseline .tex files to evaluate.",
    )
    parser.add_argument(
        "--out",
        default="baseline_mechanical_eval.json",
        help="Output file name (saved under evaluation/eval_result).",
    )
    args = parser.parse_args()

    tex_dir = Path(args.dir)
    if not tex_dir.exists():
        raise FileNotFoundError(f"Baseline output directory not found: {tex_dir}")

    main_tex_files = collect_main_tex_files(tex_dir)
    if not main_tex_files:
        raise FileNotFoundError(f"No main.tex found under: {tex_dir}")

    results = []
    for tex_file in main_tex_files:
        # print(f"[Baseline Eval] Checking {tex_file}")
        res = evaluate_tex(tex_file)
        results.append(asdict(res))

    out_dir = Path.cwd() / "evaluation" / "latex_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(args.out).name
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Baseline Eval] Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()

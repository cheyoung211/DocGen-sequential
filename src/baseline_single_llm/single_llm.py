import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llm_client import DEFAULT_MODEL, LLMClient
from src.agents.latex_compiler import LatexCompiler


def build_system_prompt() -> str:
    return (
        "You are an expert technical writer and LaTeX author.\n"
        "Your task is to write a COMPLETE self-contained LaTeX document "
        "in English that strictly follows the user's instructions.\n\n"
        "Output REQUIREMENTS:\n"
        "- Output ONLY valid LaTeX source code, no markdown fences, no explanations.\n"
        "- Use a single main .tex file that can be compiled with pdflatex.\n"
        "- Use \\documentclass{article} (or another standard class) and include preamble, "
        "title, sections, and content.\n"
        "- If the user mentions figures or tables, insert appropriate LaTeX environments "
        "(figure, table, tabular), but DO NOT include external image files; "
        "instead, you may use placeholder comments.\n"
        "- Ensure the length and depth match the user's specific constraints.\n"
    )


def build_user_prompt(instruction: str, title: str, overview: str) -> str:
    return "\n".join(
        [
            "### User Instructions:",
            instruction.strip(),
            "",
            "### Document Overview & Topic:",
            f"Title: {title}",
            f"Context: {overview}",
            "",
            "Please now write a single complete LaTeX document that satisfies all the instructions and incorporates the overview content effectively.",
            "Remember: output ONLY raw LaTeX code (no markdown, no ``` fences, no explanations).",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-path", default="stress_sample.jsonl", help="Path to JSONL dataset")
    parser.add_argument("-o", "--out-dir", default="outputs_baseline_batch", help="Output directory")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    llm = LLMClient(model_name=args.model_name, max_new_tokens=args.max_new_tokens)
    print(f"Using OpenAI Responses API model: {args.model_name}")

    # Same engine ("tectonic", falling back to pdflatex) and 2-pass run count as
    # DocGenPipeline's compiler, so compile success is comparable across baselines.
    compiler = LatexCompiler()

    root_dir = Path(args.out_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_path, "r", encoding="utf-8") as file:
        samples = [json.loads(line) for line in file]

    for index, item in enumerate(tqdm(samples, desc="Processing")):
        # New-schema benchmark items (data/benchmark/benchmark_v1.jsonl) already carry
        # a single self-contained instruction; use it directly instead of the
        # legacy WikiHow-schema instruction/input.title/input.overview fields below.
        natural_language_instruction = item.get("natural_language_instruction")
        if natural_language_instruction:
            instruction = natural_language_instruction
            title = item.get("title", f"Untitled_{index}")
            overview = item.get("overview", "")
        else:
            instruction = item.get("instruction", "")
            input_data = item.get("input", {})
            title = input_data.get("title", f"Untitled_{index}")
            overview = input_data.get("overview", "")
        stress_factor = item.get("sample_id") or item.get("stress_factor", "unknown")

        try:
            latex_out = llm.generate(
                build_system_prompt(),
                build_user_prompt(instruction, title, overview),
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )

            request_id = f"{stress_factor}/sample_{index:03d}"
            sample_dir = root_dir / request_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "main.tex").write_text(latex_out, encoding="utf-8")
            (sample_dir / "meta.json").write_text(
                json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            if compiler.compile_pdf(request_id, output_dir=str(root_dir)):
                print(f"[{request_id}] PDF compiled successfully.")
            else:
                print(f"[{request_id}] PDF compilation failed; see compile.log.")
        except Exception as exc:
            print(f"Error at index {index}: {exc}")

    print(f"Done! Results are in {root_dir}")


if __name__ == "__main__":
    main()

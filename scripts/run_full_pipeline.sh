# Gemini 3 Flash is used for every language-model agent.
# Set GEMINI_API_KEY (or OPENAI_API_KEY for gpt-* models) in .env before running this script.
# Image generation is off by default (--image-model none); pass "sdxl" or
# "flux" explicitly to re-enable it.
python3 scripts/run_full_pipeline.py \
  --query "Create the report of international stock market of 2026. Include global events that affected to the market and the moves of market index." \
  --image-model none \
  --planner-model gpt-5-mini

# Short-form (dataset2) dev set -- see README "Short-form variant (dataset2/)".
# Use data/benchmark/dev_debug.jsonl instead for the original long-form set.
python3 scripts/run_full_pipeline.py \
  --dataset "dataset2/benchmark/benchmark_v2.jsonl" \
  --image-model none \
  --planner-model gpt-4o-mini \
  --out-dir "outputs/all_verifiers/" \
  --seed 42

python3 scripts/run_full_pipeline.py \
  --dataset "dataset2/benchmark/benchmark_v2.jsonl" \
  --image-model none \
  --planner-model gpt-4o-mini \
  --out-dir "outputs/baseline_run/" \
  --seed 42 \
  --disable-text-validator bare_math_notation \
  --disable-text-validator equation_shape \
  --disable-text-validator figure_content \
  --disable-text-validator forbidden_environment \
  --disable-text-validator manual_label \
  --disable-text-validator plain_title \
  --disable-text-validator table_columns \
  --disable-text-validator table_shape \
  --disable-text-validator unicode_math_symbol \
  --disable-planner-validator component_kind_pairing \
  --disable-planner-validator leading_section_number \
  --disable-planner-validator plain_title \
  --disable-planner-validator required_sections \
  --disable-assembler-validator document_frame \
  --disable-assembler-validator figure_asset_completeness \
  --disable-assembler-validator figure_single_use \
  --disable-assembler-validator forbidden_environment \
  --disable-assembler-validator layout_policy \
  --disable-assembler-validator manual_label \
  --disable-assembler-validator reference \
  --disable-assembler-validator table_columns

# Local Qwen3.6-35B-A3B via vLLM, on data/benchmark/dev_debug.jsonl (12 items,
# "약 10개" smoke-test subset -- use benchmark_v1.jsonl/benchmark_v2.jsonl for
# the full 120-item set instead).
#
# Prerequisite: the vLLM server must already be running --
#   /data1/home/chelsy211/qwen_vllm/start_server.sh
#   curl http://localhost:8911/v1/models   # sanity check it's up
# (see memory "Qwen3.6 local vLLM setup" for troubleshooting). This still
# calls scripts/run_full_pipeline.py with python3 like every block above --
# run it from the `agents` conda env, not base.
python3 scripts/run_full_pipeline.py \
  --dataset "data/benchmark/dev_debug.jsonl" \
  --image-model none \
  --planner-model qwen3.6-35b-a3b \
  --text-model qwen3.6-35b-a3b \
  --integrator-model qwen3.6-35b-a3b \
  --out-dir "outputs/qwen_smoke_test/" #\
  # --seed 42

  # 가능 모델: gemini-*-preview / gemini-*.* (Google) / gpt-* (OpenAI) / qwen3.6-35b-a3b (local vLLM, GPU 3)


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
  --out-dir "outputs/baseline_run" \
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

  # 가능 모델: gemini-*-preview / gemini-*.* (Google) or gpt-* (OpenAI)


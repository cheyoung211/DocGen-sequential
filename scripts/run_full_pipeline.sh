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
  --dataset "dataset2/benchmark/dev_debug.jsonl" \
  --image-model none \
  --planner-model gpt-5-mini 

  # 가능 모델: gemini-*-preview / gemini-*.* (Google) or gpt-* (OpenAI)


# Gemini 3 Flash is used for every language-model agent.
# Set GEMINI_API_KEY (or OPENAI_API_KEY for gpt-* models) in .env before running this script.
python3 scripts/run_full_pipeline.py \
  --query "Create the report of international stock market of 2026. Include global events that affected to the market and the moves of market index." \
  --image-model "sdxl" \
  --planner-model gpt-5-mini

python3 scripts/run_full_pipeline.py \
  --dataset "data/benchmark/dev_debug.jsonl" \
  --image-model "sdxl" \
  --planner-model gpt-5-mini

  # 가능 모델: gemini-*-preview / gemini-*.* (Google) or gpt-* (OpenAI)


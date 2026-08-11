#!/usr/bin/env bash
# Foundation evaluation layer: generate a batch of documents, then compute
# Metric 1 (Verification Trigger Rate), Metric 2 (Recovery Success Rate),
# Metric 6 (Compile Success Rate), and Metric 7 (Contract Satisfaction Rate,
# hard/deterministic categories) over that batch.
#
# Usage:
#   scripts/run_evaluation.sh [DATASET_JSONL] [OUT_DIR_PREFIX] [MODEL]
##   -> scripts/run_evaluation.sh dataset2/benchmark/dev_debug.jsonl outputs/eval_run gpt-4o-mini
#   scripts/run_evaluation.sh --eval-only BATCH_DIR [DATASET_JSONL]
##   -> scripts/run_evaluation.sh --eval-only outputs/foundation_validate_08111246 dataset2/benchmark/benchmark_v2.jsonl
#
# Examples:
#   scripts/run_evaluation.sh
#   scripts/run_evaluation.sh dataset2/benchmark/dev_debug.jsonl outputs/eval_run gpt-4o-mini
#   scripts/run_evaluation.sh --eval-only outputs/foundation_validate_08111246 dataset2/benchmark/benchmark_v2.jsonl
#
# Requires:
#   - OPENAI_API_KEY (or GEMINI_API_KEY) in .env for the generation step.
#   - A LaTeX engine on PATH (tectonic preferred: `conda install -c
#     conda-forge tectonic`) for real Compile Success Rate numbers --
#     without one, every compile_result reports
#     first_error_type="no_latex_engine_found".
#
# The --eval-only form skips generation and only (re)computes metrics over
# an existing batch directory -- useful for re-running the aggregator after
# an evaluation-layer code change, or for a batch generated in a previous
# session.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--eval-only" ]]; then
  BATCH_DIR="${2:?Usage: run_evaluation.sh --eval-only BATCH_DIR [DATASET_JSONL]}"
  DATASET="${3:-dataset2/benchmark/dev_debug.jsonl}"
else
  DATASET="${1:-dataset2/benchmark/dev_debug.jsonl}"
  OUT_DIR_PREFIX="${2:-outputs/eval_run}"
  MODEL="${3:-gpt-4o-mini}"

  echo "[run_evaluation] Generating: dataset=$DATASET model=$MODEL"
  python3 scripts/run_full_pipeline.py \
    --dataset "$DATASET" \
    --image-model none \
    --planner-model "$MODEL" --text-model "$MODEL" --integrator-model "$MODEL" \
    --iterations 3 \
    --out-dir "$OUT_DIR_PREFIX"

  # run_full_pipeline.py stamps the real output dir as "${OUT_DIR_PREFIX}_MMDDHHMM".
  BATCH_DIR=$(ls -dt "${OUT_DIR_PREFIX}"_* | head -n 1)
fi

echo "[run_evaluation] Evaluating batch: $BATCH_DIR"
python3 src/evaluation/aggregate.py \
  --dir "$BATCH_DIR" \
  --benchmark "$DATASET" \
  --out foundation_metrics_eval.json

SUMMARY_PATH="evaluation/foundation_metrics/$(basename "$BATCH_DIR")_foundation_metrics_eval.json"
echo "[run_evaluation] Full summary: $SUMMARY_PATH"

python3 -c "
import json
d = json.load(open('$SUMMARY_PATH'))
print(json.dumps({
    'sample_count': d['sample_count'],
    'verification_trigger_rate_overall': d['verification_trigger_rate']['event_level']['overall'],
    'recovery_success_rate': d['recovery_success_rate']['recovery_success_rate'],
    'detected_failures_without_recovery_path': d['recovery_success_rate']['detected_failures_without_recovery_path'],
    'detected_failures_without_recovery_attempt': d['recovery_success_rate']['detected_failures_without_recovery_attempt'],
    'compile_success_rate': d['compile_success_rate']['compile_success_rate'],
    'contract_satisfaction_overall': d['contract_satisfaction_rate']['overall'],
}, indent=2, ensure_ascii=False))
"

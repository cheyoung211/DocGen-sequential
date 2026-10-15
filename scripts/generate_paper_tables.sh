#!/usr/bin/env bash
# Regenerate Table 4 (trajectory groups) and Table 5 (document-level quality
# + generation cost) from two pipeline output directories -- one all-verifiers
# batch, one no-verifier batch. Thin wrapper around
# src/evaluation/paper_tables.py; see that module's docstring for the full
# classification/rendering logic.
#
# Usage:
#   scripts/generate_paper_tables.sh ALL_VERIFIERS_DIR NO_VERIFIER_DIR [BENCHMARK_JSONL] [OUT_DIR]
#
# Example:
#   scripts/generate_paper_tables.sh \
#     outputs/qwen_all_verifiers_10151200 \
#     outputs/qwen_no_verifier_10151300 \
#     dataset2/benchmark/benchmark_v2.jsonl

set -euo pipefail
cd "$(dirname "$0")/.."

ALL_VERIFIERS_DIR="${1:?Usage: scripts/generate_paper_tables.sh ALL_VERIFIERS_DIR NO_VERIFIER_DIR [BENCHMARK_JSONL] [OUT_DIR]}"
NO_VERIFIER_DIR="${2:?Usage: scripts/generate_paper_tables.sh ALL_VERIFIERS_DIR NO_VERIFIER_DIR [BENCHMARK_JSONL] [OUT_DIR]}"
BENCHMARK="${3:-dataset2/benchmark/benchmark_v2.jsonl}"
OUT_DIR="${4:-evaluation/paper_tables}"

python3 -m src.evaluation.paper_tables \
  --condition "All verifiers=${ALL_VERIFIERS_DIR}" \
  --condition "No verifier=${NO_VERIFIER_DIR}" \
  --no-verifier-condition "No verifier" \
  --all-verifiers-condition "All verifiers" \
  --benchmark "$BENCHMARK" \
  --out-dir "$OUT_DIR"

# agent system evaluation
  python src/evaluation/long_form_quality/eval_long_form_quality.py \
  --doc-dir outputs/batch_test/run_20260415_151151 \
  --out longform_eval.json

# baseline single llm evaluation
  python src/evaluation/long_form_quality/eval_long_form_quality_baseline.py \
  --doc-dir outputs/outputs_single_llm/run_20260325_132956/constraint_binding/sample_001 \
  --out long_form_quality_eval_single.json

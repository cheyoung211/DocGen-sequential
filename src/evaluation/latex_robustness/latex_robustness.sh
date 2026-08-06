# agent system evaluation
python src/evaluation/latex_robustness/latex_robustness.py \
    --dir ./outputs/batch_test_07281829/basic \
    --out latex_eval.json

# baseline single llm evaluation
python src/evaluation/latex_robustness/latex_robustness_baseline.py\
    --dir outputs/outputs_single_llm/run_20260415_151151/constraint_binding/sample_001 \
    --out latex_eval_single.json

#!/bin/bash
# Variance Study: Run the same 100 instances 2 more times

cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH
mkdir -p variance_study

echo "STEP 1: Sample 100 instances from Run 1"
python3 variance_study/sample_from_existing.py \
    --run1-results baselines/results/deepseek-v4-flash_llm/jobs_results_diff.jsonl \
    --num-instances 100 \
    --output variance_study/sampled_ids.txt \
    --seed 42

SAMPLED_IDS=$(cat variance_study/sampled_ids.txt)
echo "Sampled IDs: $SAMPLED_IDS"
echo ""

echo "============================================================"
echo "RUN 2: Generate patches and evaluate"
echo "============================================================"

# Generate patches for Run 2
python3 baselines/main.py --model deepseek-v4-flash --analyzer llm --num-instances 100

# Evaluate Run 2 patches
python3 run_benchmark.py --ids "$SAMPLED_IDS" --skip-upstream-sync

# Save Run 2 results
cp results/jobs_ids_diff.jsonl variance_study/run_2_results.jsonl
echo "Run 2 complete"
echo ""

echo "============================================================"
echo "RUN 3: Generate patches and evaluate"
echo "============================================================"

# Generate patches for Run 3
python3 baselines/main.py --model deepseek-v4-flash --analyzer llm --num-instances 100

# Evaluate Run 3 patches
python3 run_benchmark.py --ids "$SAMPLED_IDS" --skip-upstream-sync

# Save Run 3 results
cp results/jobs_ids_diff.jsonl variance_study/run_3_results.jsonl
echo "Run 3 complete"
echo ""

echo "============================================================"
echo "Calculate variance"
echo "============================================================"
python3 variance_study/calculate_variance.py

echo "DONE! Check variance_study/ for results"

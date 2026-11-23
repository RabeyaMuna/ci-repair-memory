#!/bin/bash
# Variance Study: Run first 100 instances 2 more times

cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH
mkdir -p variance_study

echo "========================================"
echo "STEP 1: Extract first 100 from Run 1"
echo "========================================"
python3 -c "
import pandas as pd
import json

# Get first 100 IDs from dataset
df = pd.read_parquet('dataset/lca_dataset.parquet')
ids = [str(i) for i in df['id'].iloc[0:100].tolist()]
with open('variance_study/first_100_ids.txt', 'w') as f:
    f.write(','.join(ids))

# Extract Run 1 results for these 100
results = []
with open('baselines/results/deepseek-v4-flash_llm/jobs_results_diff.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            if str(r.get('id', '')) in ids:
                results.append(r)

with open('variance_study/run_1_results.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

print(f'✓ Extracted {len(results)} results for first 100 instances')
"

echo ""
echo "========================================"
echo "RUN 2: Generate patches"
echo "========================================"
python3 baselines/main.py \
    --model deepseek-v4-flash \
    --analyzer llm \
    --num-instances 100 \
    --run-name run2

echo ""
echo "RUN 2: Evaluate patches"
python3 run_benchmark.py \
    --ids "$(cat variance_study/first_100_ids.txt)" \
    --skip-upstream-sync

cp results/jobs_ids_diff.jsonl variance_study/run_2_results.jsonl
echo "✓ Run 2 complete"

echo ""
echo "========================================"
echo "RUN 3: Generate patches"
echo "========================================"
python3 baselines/main.py \
    --model deepseek-v4-flash \
    --analyzer llm \
    --num-instances 100 \
    --run-name run3

echo ""
echo "RUN 3: Evaluate patches"
python3 run_benchmark.py \
    --ids "$(cat variance_study/first_100_ids.txt)" \
    --skip-upstream-sync

cp results/jobs_ids_diff.jsonl variance_study/run_3_results.jsonl
echo "✓ Run 3 complete"

echo ""
echo "========================================"
echo "Calculate variance statistics"
echo "========================================"
python3 variance_study/calculate_variance.py

echo ""
echo "========================================"
echo "RESULTS"
echo "========================================"
echo "Generated patches saved in:"
echo "  baselines/results/deepseek-v4-flash_llm_run2/"
echo "  baselines/results/deepseek-v4-flash_llm_run3/"
echo ""
echo "Evaluation results:"
echo "  variance_study/run_1_results.jsonl"
echo "  variance_study/run_2_results.jsonl"
echo "  variance_study/run_3_results.jsonl"
echo ""
echo "Statistics and tables:"
echo "  variance_study/variance_statistics.json"
echo "  variance_study/variance_table.tex"
echo "  variance_study/paper_text.txt"

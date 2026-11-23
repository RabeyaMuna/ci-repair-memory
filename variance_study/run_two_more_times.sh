#!/bin/bash
# Variance Study: Run 2 more times on the same instances

# Configuration
MODEL="deepseek-v4-flash"
ANALYZER="llm"
NUM_INSTANCES=100
RUN1_RESULTS="baselines/results/deepseek-v4-flash_llm/jobs_results_diff.jsonl"

echo "======================================================================"
echo "VARIANCE STUDY: Running 2 more times on same 100 instances"
echo "======================================================================"
echo "Model: $MODEL"
echo "Analyzer: $ANALYZER"
echo "Instances: $NUM_INSTANCES"
echo "Run 1 results: $RUN1_RESULTS"
echo "======================================================================"
echo ""

# Step 1: Sample 100 instances from Run 1
echo "[STEP 1] Sampling $NUM_INSTANCES instances from Run 1..."
python3 variance_study/sample_from_existing.py \
    --run1-results "$RUN1_RESULTS" \
    --num-instances $NUM_INSTANCES \
    --output variance_study/sampled_ids.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to sample instances"
    exit 1
fi

# Read sampled IDs
SAMPLED_IDS=$(cat variance_study/sampled_ids.txt)
echo "[STEP 1] Sampled IDs: $SAMPLED_IDS"
echo ""

# Step 2: Run 2 (Generate patches + Evaluate)
echo "======================================================================"
echo "RUN 2: Generate patches and evaluate"
echo "======================================================================"
echo ""

echo "[RUN 2] Running baseline pipeline..."
python3 baselines/main.py \
    --model "$MODEL" \
    --analyzer "$ANALYZER" \
    --num-instances $NUM_INSTANCES

if [ $? -ne 0 ]; then
    echo "[ERROR] Run 2 baseline pipeline failed"
    exit 1
fi

echo ""
echo "[RUN 2] Evaluating patches..."
python3 run_benchmark.py \
    --ids "$SAMPLED_IDS" \
    --skip-upstream-sync

if [ $? -ne 0 ]; then
    echo "[ERROR] Run 2 evaluation failed"
    exit 1
fi

# Save Run 2 results
cp results/jobs_ids_diff.jsonl variance_study/run_2_results.jsonl
echo "[RUN 2] Results saved to variance_study/run_2_results.jsonl"
echo ""

# Step 3: Run 3 (Generate patches + Evaluate)
echo "======================================================================"
echo "RUN 3: Generate patches and evaluate"
echo "======================================================================"
echo ""

echo "[RUN 3] Running baseline pipeline..."
python3 baselines/main.py \
    --model "$MODEL" \
    --analyzer "$ANALYZER" \
    --num-instances $NUM_INSTANCES

if [ $? -ne 0 ]; then
    echo "[ERROR] Run 3 baseline pipeline failed"
    exit 1
fi

echo ""
echo "[RUN 3] Evaluating patches..."
python3 run_benchmark.py \
    --ids "$SAMPLED_IDS" \
    --skip-upstream-sync

if [ $? -ne 0 ]; then
    echo "[ERROR] Run 3 evaluation failed"
    exit 1
fi

# Save Run 3 results
cp results/jobs_ids_diff.jsonl variance_study/run_3_results.jsonl
echo "[RUN 3] Results saved to variance_study/run_3_results.jsonl"
echo ""

# Step 4: Analyze variance
echo "======================================================================"
echo "ANALYZING VARIANCE"
echo "======================================================================"
echo ""

python3 variance_study/analyze_variance.py \
    --run1 "$RUN1_RESULTS" \
    --run2 variance_study/run_2_results.jsonl \
    --run3 variance_study/run_3_results.jsonl \
    --sampled-ids variance_study/sampled_ids.txt

echo ""
echo "======================================================================"
echo "VARIANCE STUDY COMPLETE"
echo "======================================================================"
echo ""
echo "Results saved in variance_study/"
echo "  - run_1_results.jsonl (from existing)"
echo "  - run_2_results.jsonl (new)"
echo "  - run_3_results.jsonl (new)"
echo "  - variance_statistics.json"
echo "  - variance_table.tex (for paper)"
echo ""

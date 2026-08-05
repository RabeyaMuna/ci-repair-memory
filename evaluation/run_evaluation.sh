#!/bin/bash
# Quick evaluation script for CI-REPAIR-BENCH predictions

echo "Running CI-REPAIR-BENCH Evaluation..."
echo "======================================"

# Run evaluation
python3 evaluate_predictions.py \
  --preds results/preds.json \
  --dataset dataset/lca_dataset.parquet \
  --output results/evaluation_results.json \
  --csv results/evaluation_per_instance.csv \
  --max-k 15

echo ""
echo "Files generated:"
echo "  - results/evaluation_results.json"
echo "  - results/evaluation_per_instance.csv"
echo "  - results/evaluation_summary.md"
echo ""
echo "To view detailed summary:"
echo "  cat results/evaluation_summary.md"

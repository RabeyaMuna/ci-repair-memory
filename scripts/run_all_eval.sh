#!/usr/bin/env bash
# Run FL evaluation for all strategies and print combined tables.
set -e

PARQUET="$(python scripts/dataset_source.py)"
RESULTS="baselines/results"
SCRIPT="scripts/overall_fl_evaluator.py"

echo "=========================================="
echo "  Running FL evaluation for all strategies"
echo "=========================================="

python $SCRIPT eval --results-dir $RESULTS/deepseek-chat_llm  --parquet $PARQUET --label "Deepseek-Chat"  --strategy LLM
python $SCRIPT eval --results-dir $RESULTS/deepseek-chat_bm25 --parquet $PARQUET --label "Deepseek-Chat"  --strategy BM25
python $SCRIPT eval --results-dir $RESULTS/deepseek-coder_llm  --parquet $PARQUET --label "Deepseek-Coder" --strategy LLM
python $SCRIPT eval --results-dir $RESULTS/deepseek-coder_bm25 --parquet $PARQUET --label "Deepseek-Coder" --strategy BM25
python $SCRIPT eval --results-dir $RESULTS/gpt-4o-mini_llm  --parquet $PARQUET --label "GPT-4o-mini"   --strategy LLM
python $SCRIPT eval --results-dir $RESULTS/gpt-4o-mini_bm25 --parquet $PARQUET --label "GPT-4o-mini"   --strategy BM25
python $SCRIPT eval --results-dir $RESULTS/gpt-5-mini_llm   --parquet $PARQUET --label "GPT-5-mini"    --strategy LLM
python $SCRIPT eval --results-dir $RESULTS/gpt-5-mini_bm25  --parquet $PARQUET --label "GPT-5-mini"    --strategy BM25

echo ""
echo "=========================================="
echo "  Printing paper tables"
echo "=========================================="

python $SCRIPT tables \
  --reports \
    $RESULTS/deepseek-chat_llm/fl_full_report.json \
    $RESULTS/deepseek-chat_bm25/fl_full_report.json \
    $RESULTS/deepseek-coder_llm/fl_full_report.json \
    $RESULTS/deepseek-coder_bm25/fl_full_report.json \
    $RESULTS/gpt-4o-mini_llm/fl_full_report.json \
    $RESULTS/gpt-4o-mini_bm25/fl_full_report.json \
    $RESULTS/gpt-5-mini_llm/fl_full_report.json \
    $RESULTS/gpt-5-mini_bm25/fl_full_report.json \
  --latex paper/rq_tables.tex

echo ""
echo "Done. Reports saved in each results directory."
echo "LaTeX written to paper/rq_tables.tex"

# Memory Ablation Cost Analysis

This guide explains how to calculate token usage and costs for different memory configurations in your paper.

## Required Data

You need `token_report.json` files from each of your experimental runs:

1. **Baseline (No Memory)**: `baselines/results/MiniMax-M2.5_llm_baseline/token_report.json`
2. **Backward (L1 only)**: `baselines/results/MiniMax-M2.5_llm_memory_L1/token_report.json`
3. **Forward (L1+L2)**: `baselines/results/MiniMax-M2.5_llm_memory_L1L2/token_report.json`
4. **Bidirectional (L1+L2+L3)**: `baselines/results/MiniMax-M2.5_llm_memory/token_report.json`

## How to Generate token_report.json

If you don't have these files yet, they are generated automatically when you run the baseline experiments:

```bash
# Run baseline (no memory)
baselines/.venv/bin/python baselines/main.py --model-key MiniMax-M2.5

# Run L1 memory ablation
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \
  --ablation-levels L1 \
  --model-key MiniMax-M2.5

# Run L1+L2 memory ablation
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \
  --ablation-levels L1+L2 \
  --model-key MiniMax-M2.5

# Run L1+L2+L3 memory ablation (bidirectional)
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \
  --ablation-levels L1+L2+L3 \
  --model-key MiniMax-M2.5
```

Each run automatically creates a `token_report.json` that includes:
- Total input/output tokens for all LLM calls
- Cost calculations based on model pricing
- Per-task breakdown for detailed analysis

## Running the Analysis

Once you have the token_report.json files:

```bash
# Basic analysis (prints to console)
python scripts/analysis/calculate_memory_ablation_costs.py

# Save detailed JSON results
python scripts/analysis/calculate_memory_ablation_costs.py \
  --output results/memory_cost_analysis.json

# Generate LaTeX table for your paper
python scripts/analysis/calculate_memory_ablation_costs.py \
  --latex results/memory_cost_table.tex

# Specify custom results directory
python scripts/analysis/calculate_memory_ablation_costs.py \
  --results-dir /path/to/your/results
```

## Output Format

The script produces:

### Console Output
```
====================================================================================================
  MEMORY ABLATION STUDY — TOKEN USAGE AND COST ANALYSIS
====================================================================================================

Configuration                  Instances    Tokens (K/inst.)     Cost ($/inst.)      
----------------------------------------------------------------------------------------------------
Baseline (No Memory)           410          45.23                $0.0245
Backward (L1 only)             410          52.18                $0.0283
Forward (L1+L2)                410          58.91                $0.0319
Bidirectional (L1+L2+L3)       410          63.45                $0.0344
----------------------------------------------------------------------------------------------------

Differences from Baseline:
Configuration                  Token Δ (K)          Cost Δ ($)           Cost Δ (%)          
----------------------------------------------------------------------------------------------------
Backward (L1 only)             +6.95                +$0.0038             +15.5%
Forward (L1+L2)                +13.68               +$0.0074             +30.2%
Bidirectional (L1+L2+L3)       +18.22               +$0.0099             +40.4%

====================================================================================================
```

### JSON Output (--output)
Detailed breakdown including total tokens, costs, and per-instance averages for all configurations.

### LaTeX Table (--latex)
Ready-to-use LaTeX table for your paper:

```latex
\begin{table}[t]
\centering
\caption{Token Usage and Cost Analysis for Memory Ablation Study}
\label{tab:memory-cost}
\begin{tabular}{lrrr}
\toprule
Configuration & Instances & Tokens (K/inst.) & Cost (\$/inst.) \\
\midrule
Baseline & 410 & 45.23 & 0.0245 \\
L1 & 410 & 52.18 & 0.0283 \\
L1+L2 & 410 & 58.91 & 0.0319 \\
L1+L2+L3 & 410 & 63.45 & 0.0344 \\
\bottomrule
\end{tabular}
\end{table}
```

## If You Have Data in a Different Location

If your token_report.json files are elsewhere, you can either:

1. Use the `--results-dir` flag to point to them
2. Edit the `MEMORY_CONFIGS` list in `calculate_memory_ablation_costs.py` to match your directory structure

## Understanding the Numbers

- **Tokens (K/inst.)**: Average total tokens (input + output) per instance in thousands
- **Cost ($/inst.)**: Average cost per instance in USD based on model pricing
- **Token Δ**: Difference in token usage compared to baseline
- **Cost Δ**: Difference in cost compared to baseline (both absolute $ and percentage)

## Model Pricing

The costs are calculated based on the pricing in `baselines/utilities/token_tracker.py`:

- **MiniMax-M2.5**: $0.15/1M input tokens, $1.15/1M output tokens
- **GPT-4o-mini**: $0.15/1M input tokens, $0.60/1M output tokens
- **GPT-5-mini**: $1.10/1M input tokens, $4.40/1M output tokens

To update pricing or add models, edit `MODEL_PRICING` in `token_tracker.py`.

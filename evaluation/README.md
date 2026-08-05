# Evaluation Directory

This directory contains all evaluation scripts, metrics computation, result analysis, and benchmark outputs for CI-Repair-Bench.

## Directory Structure

```
evaluation/
├── README.md                      # This file
│
├── Evaluation Scripts
│   ├── evaluate_predictions.py   # Main evaluation script for patch predictions
│   ├── evaluate_subset.py        # Evaluate on dataset subsets
│   ├── evaluate_topk_overall.py  # Top-k localization evaluation
│   ├── run_evaluation.sh         # Shell script to run evaluations
│   ├── verify_metrics.py         # Metric validation and verification
│   └── explain_exact_match.py    # Exact match analysis
│
├── Analysis & Comparison
│   ├── compare_evaluations.py    # Compare multiple evaluation runs
│   ├── generate_results_table.py # Generate results tables
│   └── visualize_results.py      # Visualization utilities
│
└── results/                       # Evaluation outputs and results
    ├── evaluation_results.json   # Main evaluation results
    ├── preds.json                # Predictions from repair pipeline
    ├── preds-1.json              # Alternative prediction runs
    └── jobs_*_diff.jsonl         # CI job execution diffs
```

## Evaluation Scripts

### Main Evaluation

#### `evaluate_predictions.py`
Main evaluation script that validates generated patches against benchmark instances.

**Usage:**
```bash
python evaluation/evaluate_predictions.py \
    --predictions results/preds.json \
    --output results/evaluation_results.json
```

**Key Features:**
- Loads predictions from JSON
- Applies patches to benchmark instances
- Re-executes CI workflows via GitHub Actions
- Computes Pass@1 metrics
- Generates detailed evaluation reports

#### `run_evaluation.sh`
Shell script for running complete evaluation pipeline.

**Usage:**
```bash
./evaluation/run_evaluation.sh
```

**Pipeline Steps:**
1. Loads configuration
2. Validates predictions format
3. Runs CI re-execution
4. Computes metrics
5. Generates reports

### Subset Evaluation

#### `evaluate_subset.py`
Evaluate repair performance on specific dataset subsets (by repository, error type, etc.).

**Usage:**
```bash
# Evaluate on specific repositories
python evaluation/evaluate_subset.py \
    --repos agno,axolotl,litellm

# Evaluate on specific error types
python evaluation/evaluate_subset.py \
    --error-types "Code Formatting,Code Linting"
```

**Key Features:**
- Filter instances by repository
- Filter by error type
- Filter by commit date range
- Generate subset-specific metrics

### Localization Evaluation

#### `evaluate_topk_overall.py`
Evaluate fault localization performance using Top-k metrics.

**Usage:**
```bash
python evaluation/evaluate_topk_overall.py \
    --predictions results/preds.json
```

**Metrics Computed:**
- **Top-1, Top-3, Top-5**: Recall at rank k
- **MAP (Mean Average Precision)**: Overall ranking quality
- Per-repository localization accuracy
- Per-error-type localization accuracy

**Output:**
```json
{
  "top_1": 45.33,
  "top_3": 53.97,
  "top_5": 58.55,
  "map": 42.53
}
```

## Analysis & Comparison

### `compare_evaluations.py`
Compare results across multiple evaluation runs (different models, configurations, etc.).

**Usage:**
```bash
python evaluation/compare_evaluations.py \
    --baseline results/gpt4o_mini_results.json \
    --comparison results/gpt5_mini_results.json \
    --output results/comparison.json
```

**Features:**
- Side-by-side metric comparison
- Statistical significance testing
- Per-category performance delta
- Improvement/regression detection

### `generate_results_table.py`
Generate formatted result tables for papers and reports.

**Usage:**
```bash
python evaluation/generate_results_table.py \
    --results results/evaluation_results.json \
    --format latex  # or 'markdown', 'csv'
```

**Output Formats:**
- LaTeX tables for papers
- Markdown tables for documentation
- CSV for data analysis

### `visualize_results.py`
Create visualizations of evaluation results.

**Usage:**
```bash
python evaluation/visualize_results.py \
    --results results/evaluation_results.json \
    --output-dir results/plots/
```

**Visualizations:**
- Pass@1 by error type (bar chart)
- Localization accuracy (heatmap)
- Repository-level performance (scatter plot)
- Model comparison (grouped bar chart)

## Verification & Analysis

### `verify_metrics.py`
Validate metric computations and check for data consistency.

**Usage:**
```bash
python evaluation/verify_metrics.py \
    --results results/evaluation_results.json
```

**Checks Performed:**
- Metric value ranges (0-100%)
- Count consistency (applied patches ≤ total instances)
- Per-category sum matches overall totals
- No missing required fields

### `explain_exact_match.py`
Analyze exact match cases where generated patches match ground truth.

**Usage:**
```bash
python evaluation/explain_exact_match.py \
    --predictions results/preds.json
```

**Analysis:**
- Count exact matches
- Categorize by error type
- Identify common patterns
- Flag potential data leakage

## Results Directory

### Output Files

#### `evaluation_results.json`
Main evaluation output containing all metrics.

**Schema:**
```json
{
  "overall": {
    "total_instances": 567,
    "applied_patches": 455,
    "pass_at_1": 18.9,
    "model": "gpt-5-mini",
    "timestamp": "2026-08-04T19:15:00Z"
  },
  "per_error_type": {
    "Code Formatting": {"count": 121, "success": 43, "rate": 35.5},
    "Code Linting": {"count": 208, "success": 37, "rate": 17.8},
    ...
  },
  "per_repository": {
    "agno": {"count": 84, "success": 15, "rate": 17.9},
    ...
  },
  "localization": {
    "top_1": 45.33,
    "top_3": 53.97,
    "top_5": 58.55,
    "map": 42.53
  }
}
```

#### `preds.json`
Predictions from the repair pipeline.

**Schema:**
```json
[
  {
    "instance_id": 1,
    "repo_owner": "huggingface",
    "repo_name": "diffusers",
    "sha_fail": "2c06ffa4...f214ff9",
    "patch": "diff --git a/... unified diff format ...",
    "model_name": "gpt-5-mini",
    "localized_files": ["examples/community/ip_adapter_face_id.py"],
    "repair_strategy": "tool_based",
    "timestamp": "2026-08-04T10:30:00Z"
  },
  ...
]
```

#### `jobs_*_diff.jsonl`
CI job execution status differences (JSONL format).

**Files:**
- `jobs_success_diff.jsonl` - Successfully passing CI jobs
- `jobs_failure_diff.jsonl` - Failed CI jobs
- `jobs_timeout_diff.jsonl` - Timed-out jobs
- `jobs_error_diff.jsonl` - Jobs with execution errors
- `jobs_cancelled_diff.jsonl` - Cancelled jobs
- `jobs_awaiting_diff.jsonl` - Jobs still running
- `jobs_invalid_diff.jsonl` - Invalid job configurations

**Entry Format:**
```json
{"instance_id": 1, "status": "success", "workflow_url": "https://github.com/...", "run_id": 12345678}
```

## Evaluation Workflow

### Complete Evaluation Pipeline

```bash
# Step 1: Run benchmark to generate predictions
python run_benchmark.py --model gpt-5-mini --output evaluation/results/preds.json

# Step 2: Evaluate predictions via CI re-execution
python evaluation/evaluate_predictions.py \
    --predictions evaluation/results/preds.json \
    --output evaluation/results/evaluation_results.json

# Step 3: Compute localization metrics
python evaluation/evaluate_topk_overall.py \
    --predictions evaluation/results/preds.json

# Step 4: Verify metrics consistency
python evaluation/verify_metrics.py \
    --results evaluation/results/evaluation_results.json

# Step 5: Generate results table
python evaluation/generate_results_table.py \
    --results evaluation/results/evaluation_results.json \
    --format latex

# Step 6: Create visualizations
python evaluation/visualize_results.py \
    --results evaluation/results/evaluation_results.json \
    --output-dir evaluation/results/plots/
```

### Or use the shell script:
```bash
./evaluation/run_evaluation.sh
```

## Metrics Reference

### Pass@1
Percentage of instances successfully repaired on first attempt after full CI validation.

**Formula:**
```
Pass@1 = (Successful CI Passes / Total Instances) × 100
```

### Top-k Localization
Percentage of instances where at least one ground-truth repair-relevant file appears in the top-k ranked predictions.

**Formula:**
```
Top-k = (Instances with GT file in top-k / Total Instances) × 100
```

### Mean Average Precision (MAP)
Summarizes overall ranking quality across all instances.

**Formula:**
```
MAP = (1/N) × Σ AveragePrecision(instance_i)
```

## Error Type Categories

As defined in [Section 3.3 of paper](../../EMSE_CI_Repair.pdf):

1. **Code Formatting** - Spacing, indentation, import order violations
2. **Code Linting** - Static analysis rule violations
3. **Syntax Error** - Code that fails to parse
4. **Runtime Error** - Uncaught exceptions during execution
5. **Test Failure** - Test executes but produces wrong outcome
6. **Assertion Error** - Test fails on assert statement
7. **Type Checking** - Static type checker violations
8. **Dependency Issues** - Version conflicts, incompatibilities
9. **Package Install Error** - Physical installation failures
10. **Configuration Error** - Invalid project settings
11. **Environment Error** - Runner/execution context issues
12. **Doc/Docstring** - Documentation build/formatting failures

## Best Practices

### Running Evaluations

1. **Always use full CI validation**: Don't skip CI re-execution for faster results
2. **Save predictions before evaluation**: Keep predictions separate from results
3. **Version control results**: Tag results with model name, timestamp, config
4. **Document configuration**: Record all hyperparameters, prompts, settings
5. **Verify metrics**: Run `verify_metrics.py` after each evaluation

### Comparing Models

1. **Use identical dataset splits**: Ensure same instances across runs
2. **Control random seeds**: Set fixed seeds for reproducibility
3. **Document differences**: Note any config/prompt changes between runs
4. **Statistical significance**: Use `compare_evaluations.py` for proper testing
5. **Report confidence intervals**: Include uncertainty in results

### Result Interpretation

1. **Check applied patch count**: Low application rate indicates generation issues
2. **Analyze per-error-type**: Identify which failure categories are solvable
3. **Inspect failure logs**: Use `jobs_failure_diff.jsonl` for error analysis
4. **Validate localization**: Check if low repair correlates with poor localization
5. **Review exact matches**: Use `explain_exact_match.py` to check for leakage

## Troubleshooting

### Common Issues

**Low Pass@1 but high patch application rate**
- Patches apply but fail CI validation
- Check `jobs_failure_diff.jsonl` for CI error patterns
- May indicate over-aggressive or under-precise repairs

**Low patch application rate**
- Generated patches don't apply cleanly
- File paths may be incorrect
- Patches may be malformed (syntax errors in diff)

**Metrics don't sum correctly**
- Run `verify_metrics.py` to identify inconsistencies
- Check for duplicate instance IDs
- Verify error type labels are consistent

**Missing results for some instances**
- Check for timeout in CI execution
- Verify all instances have required metadata
- Check `jobs_error_diff.jsonl` and `jobs_timeout_diff.jsonl`

## Integration with Main Benchmark

The evaluation scripts integrate with the main benchmark pipeline:

```python
# In main benchmark script
from evaluation.evaluate_predictions import evaluate_patches
from evaluation.verify_metrics import verify_results

# Generate patches
predictions = repair_pipeline.run(benchmark_instances)

# Evaluate via CI
results = evaluate_patches(predictions, ci_config)

# Verify consistency
verify_results(results)
```

## References

- See main [README.md](../README.md) for overall project structure
- See [ARCHITECTURE.md](../ARCHITECTURE.md) for system design
- See paper Section 3.1 for evaluation methodology

---

**Last Updated:** August 4, 2026

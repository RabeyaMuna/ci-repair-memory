# CI-REPAIR-BENCH

A benchmark for evaluating CI/CD failure repair in GitHub Actions workflows.

## Overview

CI-REPAIR-BENCH provides a curated dataset of real-world CI failures and tools to evaluate automated repair approaches across multiple dimensions:

- **File Localization**: How accurately can models identify which files need changes?
- **CI Success**: Do the proposed patches actually fix the failing workflows?
- **Multi-Level Success**: Granular analysis at step and workflow levels

## Dataset

- **567 issues** from real GitHub repositories
- **Original CI failures** with logs and workflow definitions  
- **Successful fixes** (sha_fail → sha_success)
- **Changed files** for ground truth evaluation

Location: `dataset/lca_dataset.parquet`

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy and edit config:
```bash
cp config.yaml.example config.yaml
# Add your GitHub token
```

### 3. Generate Predictions

Your model should output predictions to `results/preds.json`:

```json
{
  "issue_id": {
    "predicted_files": ["path/to/file1.py", "path/to/file2.py"],
    "patch": "diff content..."
  }
}
```

### 4. Run CI Validation

Push patches and fetch validation results:

```bash
# Push patches to GitHub (creates workflow runs)
python run_benchmark.py

# Fetch validation results from GitHub API
python scripts/analysis/calculate_success_rate.py
```

### 5. Generate Dataset Overview (Optional)

Get comprehensive dataset statistics:

```bash
python scripts/analysis/dataset_overview.py --output results/dataset_overview.json
```

This generates:
- Diff statistics (lines added/deleted per issue)
- Validation steps overview
- Failure type distribution
- Language and repository statistics

### 6. Evaluate Results

```bash
python evaluate.py \
  --preds results/preds.json \
  --ci-results results/success_rate_evaluation.json \
  --output results/evaluation_summary.json
```

## Dataset Statistics

The dataset contains:
- **567 issues** from **145 unique repositories**
- **~133K lines** of code changes (82K added, 51K deleted)
- **20,638 validation steps** across 2,486 jobs
- **12 failure types** including linting, dependencies, formatting, tests, and runtime errors

Run `python scripts/analysis/dataset_overview.py` for detailed statistics.

## Evaluation Metrics

### File Localization
- **Top-K Accuracy** (k=1,3,5,10,15): Percentage of issues where at least one ground truth file appears in top-K predictions
- **Precision**: |Predicted ∩ Ground Truth| / |Predicted|
- **Exact Match**: Predicted file set exactly matches ground truth

### CI Success
- **Overall CI Success**: Percentage of workflows that pass after applying the patch
- **L1 (Step-Level)**: Of originally failed steps, percentage that now pass
- **L3 (Workflow-Level)**: Overall workflow pass/fail status

## Project Structure

```
CI-REPAIR-BENCH/
├── dataset/
│   ├── lca_dataset.parquet          # Main dataset
│   └── jobs_*.jsonl                 # Workflow run data
├── results/
│   ├── preds.json                   # Your model's predictions
│   ├── success_rate_evaluation.json # CI validation results
│   └── evaluation_summary.json      # Final metrics
├── scripts/analysis/
│   ├── calculate_success_rate.py    # L1/L3 evaluation
│   ├── fetch_validation_steps.py    # Fetch from GitHub API
│   ├── dataset_overview.py          # Dataset statistics
│   └── generate_plots.py            # Visualization
├── evaluate.py                      # Unified evaluation script
├── run_benchmark.py                 # Push patches to GitHub
└── README.md
```

## Example Results

```
📁 File Localization:
   Exact Match: 15.2%
   Precision: 42.8%
   Top-1: 38.5%
   Top-5: 61.3%

🔧 CI Success:
   Overall: 14.3%
   L1 Step Success: 26.3%
   L3 Workflow Pass: 14.3%
```

## Citation

If you use CI-REPAIR-BENCH in your research, please cite:

```bibtex
@inproceedings{ci-repair-bench-2024,
  title={CI-REPAIR-BENCH: A Benchmark for CI/CD Failure Repair},
  author={Your Name},
  booktitle={Proceedings of...},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or PR.

# CI-Repair-Bench Evaluation

Two focused evaluation scripts for CI repair analysis:

## 1. Top-K & Success Rate Evaluation
**File**: `topk_success_eval.py`

**Metrics**:
- **Top-K Accuracy** (k=1 to 15): Does ANY ground truth file appear in top K predictions?
- **Exact Match**: Do predicted files exactly match ground truth files?
- **Precision**: Percentage of predicted files that are correct
- **Success Rate (Pass@1)**: CI workflow pass rate

**Usage**:
```bash
python evaluation/topk_success_eval.py --model diff
```

**Output**: `results/topk_success_eval_{model}.json`

---

## 2. Multi-Level Success Rate Evaluation
**File**: `../scripts/analysis/multilevel_eval.py`

**Metrics**:
- **Level 1**: Did originally failed jobs pass?
- **Level 2**: Step-level pass rate across all runtime jobs
- **Level 3**: Did entire workflow pass?

**Usage**:
```bash
python scripts/analysis/multilevel_eval.py --model diff
```

**Output**:
- `results/jobs_results_{model}_multilevel.jsonl` - Detailed results
- `results/jobs_results_{model}_multilevel_summary.json` - Summary stats

---

## Quick Start

```bash
# Run both evaluations
python evaluation/topk_success_eval.py --model diff
python scripts/analysis/multilevel_eval.py --model diff
```

## File Requirements

- `results/preds.json` - Model predictions
- `results/jobs_ids_{model}.jsonl` - Pushed jobs
- `results/jobs_results_{model}.jsonl` - CI results
- `dataset/lca_dataset.parquet` - Ground truth dataset
- `dataset/failed_jobs_all.json` - Original failed jobs (for multilevel eval)

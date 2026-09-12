# CI-REPAIR-BENCH Analysis Scripts

## Overview

Two separate evaluation scripts:

### 1. File Localization (`evaluate_file_localization.py`)
- Top-K Accuracy (Top-1, Top-3, Top-5, Top-10, Top-15)
- Precision
- Exact Match
- **Evaluates only against pushed/validated IDs**

### 2. CI Success Rate (`calculate_success_rate.py`)
- **L1 (Step-Level)**: Did the originally failed steps (from CI logs) now pass?
- **L3 (Workflow-Level)**: Did the entire workflow pass after applying the patch?

## Prerequisites

Set your GitHub token:
```bash
export GITHUB_TOKEN=your_github_token_here
```

Or add it to `.env` file in the project root:
```
GITHUB_TOKEN=your_github_token_here
```

## Workflow

### Step 1: Evaluate File Localization

This evaluates predicted files against ground truth (only for pushed IDs).

```bash
python scripts/analysis/evaluate_file_localization.py
```

**Output**: `results/file_localization_metrics.json`

**What it calculates**:
- Top-K Accuracy: Is the ground truth file in top K predictions?
- Precision: Percentage of predicted files that are correct
- Exact Match: Do predictions exactly match ground truth?

**Time**: < 1 second

### Step 2: Fetch Step-Level Metadata (for CI Success Rate)

This fetches detailed step-by-step pass/fail information from GitHub Actions for all pushed validation runs.

```bash
python scripts/analysis/fetch_step_metadata.py
```

**Output**: `results/validation_steps_current.json`

**What it does**:
- Reads all job results from `results/jobs_*_diff.jsonl`
- For each job, fetches detailed workflow run data from GitHub Actions API
- Extracts step-level pass/fail information
- Saves complete step metadata

**Time**: ~1-2 minutes for 408 jobs (depends on API rate limits)

### Step 3: Calculate CI Success Rates

Calculates L1 and L3 success metrics based on the fetched metadata.

```bash
python scripts/analysis/calculate_success_rate.py
```

**Output**: `results/success_rate_evaluation.json`

**What it calculates**:

**L1 - Step-Level Success**:
- Compares originally failed steps (from dataset logs) with current step status
- Counts how many issues have their failed steps fixed
- Categories:
  - **Fully Fixed**: All originally failed steps now pass
  - **Partially Fixed**: Some failed steps now pass
  - **Not Fixed**: No failed steps pass

**L3 - Workflow-Level Success**:
- Overall workflow pass/fail status
- Simple: did the workflow pass after the patch?

### Step 4: Run Complete Analysis Suite

Run all evaluations in sequence:

```bash
# Quick (file localization only)
python analyze_results.py

# Detailed (file localization + CI success rates)
python analyze_results.py --detailed
```

## Metrics Explained

### File Localization (Top-K, Precision, Exact Match)

Now evaluated **only against the 408 pushed IDs**, not the full 565 dataset.

- **Top-K**: Is the ground truth file in the top K predictions?
- **Precision**: What percentage of predicted files are correct?
- **Exact Match**: Do predicted files exactly match ground truth?

### L1 - Step-Level Success

Shows if the agent can solve **visible problems** (steps that failed in logs).

**Example**:
```
Original failure: Job "build (3.9)" → Step "Codestyle" failed
After patch: Check if step "Codestyle" now passes

Result: 
  - If it passes → Step fixed ✓
  - If it still fails → Not fixed ✗
```

### L3 - Workflow-Level Success

Overall workflow health after applying the patch.

**Example**:
```
After patch:
  - Workflow passes → Success ✓
  - Workflow fails → Failure ✗
```

## Output Files

### validation_steps_current.json
```json
[
  {
    "id": "63",
    "repo_name": "errbot",
    "run_id": "34076841420",
    "url": "https://github.com/...",
    "overall_conclusion": "success",
    "step_details": {
      "all_steps": ["build (3.9): Set up job", ...],
      "failed_steps": [],
      "passed_steps": ["build (3.9): Set up job", ...],
      "total_steps": 45,
      "failed_steps_count": 0,
      "passed_steps_count": 45
    }
  }
]
```

### success_rate_evaluation.json
```json
{
  "summary": {
    "total_issues": 565,
    "evaluated_issues": 408,
    "coverage_rate": 72.21,
    "step_level": {
      "fully_fixed": 116,
      "partially_fixed": 0,
      "not_fixed": 292,
      "success_rate": 28.43
    },
    "workflow_level": {
      "passed": 116,
      "failed": 292,
      "pass_rate": 28.43
    }
  },
  "results": [...]
}
```

## Troubleshooting

### GitHub API Rate Limits

If you hit rate limits during fetch:
```bash
# Check your rate limit status
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/rate_limit
```

The script includes built-in rate limit handling and will retry with backoff.

### Missing Step Data

If some jobs don't have step data:
- The run might be too old (deleted from GitHub)
- The run URL might be invalid
- API permissions might be insufficient

Check the script output for specific warnings.

## Quick Reference

### File Localization Only
```bash
python scripts/analysis/evaluate_file_localization.py
```

### CI Success Rates Only
```bash
# First fetch step metadata (one-time, ~1-2 minutes)
python scripts/analysis/fetch_step_metadata.py

# Then calculate success rates
python scripts/analysis/calculate_success_rate.py
```

### Complete Pipeline
```bash
# File localization + CI success rates
python scripts/analysis/evaluate_file_localization.py && \
python scripts/analysis/fetch_step_metadata.py && \
python scripts/analysis/calculate_success_rate.py

# Or use the master script
python analyze_results.py --detailed
```

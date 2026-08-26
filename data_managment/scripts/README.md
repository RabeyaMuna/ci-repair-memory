# Dataset Management Scripts

## 📊 Dataset Enrichment (Main Scripts)

### ⭐ enrich_and_structure_dataset.py
**The main comprehensive script for dataset enrichment.**

```bash
python enrich_and_structure_dataset.py
```

**What it does:**
- Filters out unnecessary CI lifecycle steps
- Extracts validation jobs and steps
- Creates aggregated summaries (overall_jobs, failed_jobs)
- Enriches dataset with all validation information
- Generates all output files in one run

**Outputs:**
- `results/filtered_validation/commit_validation_jobs_steps.json`
- `results/filtered_validation/overall_jobs_by_issue.json`
- `results/filtered_validation/failed_jobs_by_issue.json`
- `results/filtered_validation/instance_validation_summary.json`
- `results/enriched_dataset.json` (final)

### 📈 analyze_jobs_summaries.py
**Analysis script for enriched dataset.**

```bash
python analyze_jobs_summaries.py
```

**Shows:**
- Overall vs failed jobs comparison
- Top instances by failure rate
- Most common failed job/step names
- Execution multipliers

### 📝 example_usage.py
**Example code showing how to use the enriched dataset.**

```bash
python example_usage.py
```

**Demonstrates:**
- Basic statistics
- Job conclusion breakdown
- Failed jobs by repo
- Step analysis
- Commit type comparison
- Instance-level analysis

---

## 🔧 CI Management Scripts

### fetch_and_trigger_metadata.py
Fetches metadata and triggers CI workflows for benchmark instances.

### fetch_logs_for_failed.py
Fetches logs for failed jobs.

### fetch_triggered_results.py
Fetches results from triggered CI workflows.

### monitor_ci_health.py
Monitors CI health and job status.

### setup_benchmark_branches.py
Sets up benchmark branches in repositories.

### workflow_manager.py
Manages GitHub Actions workflows.

### update_logs.py
Updates logs for jobs.

### check_missing_ids.py
Checks for missing instance IDs.

### run_approach_b.sh
Shell script for running approach B workflow.

---

## 🚀 Quick Start

For dataset enrichment, you only need:

```bash
# 1. Run the main enrichment script
python enrich_and_structure_dataset.py

# 2. (Optional) Analyze the results
python analyze_jobs_summaries.py

# 3. (Optional) See usage examples
python example_usage.py
```

That's it! One command enriches everything.

---

## 📚 Documentation

- **[../DATASET_ENRICHMENT_GUIDE.md](../DATASET_ENRICHMENT_GUIDE.md)** - Quick start guide
- **[../results/ENRICHMENT_SUMMARY.md](../results/ENRICHMENT_SUMMARY.md)** - Complete documentation
- **[../results/JOBS_SUMMARIES_EXPLANATION.md](../results/JOBS_SUMMARIES_EXPLANATION.md)** - Understanding counts

---

## 🗂️ Script Organization

```
scripts/
├── enrich_and_structure_dataset.py  ⭐ Main enrichment script
├── analyze_jobs_summaries.py        📈 Analysis
├── example_usage.py                 📝 Examples
│
├── fetch_and_trigger_metadata.py    🔧 CI management
├── fetch_logs_for_failed.py
├── fetch_triggered_results.py
├── monitor_ci_health.py
├── setup_benchmark_branches.py
├── workflow_manager.py
├── update_logs.py
├── check_missing_ids.py
└── run_approach_b.sh
```

---

## ✨ Notes

- **All old/duplicate enrichment scripts have been removed**
- The comprehensive script `enrich_and_structure_dataset.py` replaces:
  - `extract_validation_jobs_and_steps.py` (removed)
  - `enrich_dataset_with_validation.py` (removed)
  - `create_jobs_steps_flat_view.py` (removed)
- CI management scripts remain for workflow/metadata management

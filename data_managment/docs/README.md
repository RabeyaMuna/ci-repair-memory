# CI Benchmark Data Management

**Comprehensive system for managing CI benchmark data, monitoring workflow health, and maintaining ground truth.**

---

## 🎯 Overview

This system provides:

1. **Permanent Benchmark Branches** - Persistent test infrastructure
2. **Automated Metadata Collection** - Failed jobs and steps
3. **CI Workflow Monitoring** - Health checks and change detection
4. **Dataset Enrichment** - Add metadata to benchmark dataset
5. **Ground Truth Validation** - Detect when workflows change

---

## 📁 Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design.

```
data_managment/
├── workflow_manager.py          # 🎮 Main orchestrator
├── setup_benchmark_branches.py  # 🌿 Create permanent branches
├── fetch_commit_metadata.py     # 📥 Fetch jobs/steps metadata
├── trigger_ci_for_commits.py    # 🚀 Trigger missing CI runs
├── monitor_ci_health.py         # 🔍 Monitor CI health
├── update_failed_logs.py        # 💾 Update dataset
└── results/                     # 📊 All outputs
    ├── benchmark_branches.json
    ├── commit_job_metadata.json
    ├── missing_metadata_ids.json
    └── ci_workflow_health.json
```

---

## 🚀 Quick Start

### 1. Initial Setup (One-time)

```bash
cd data_managment

# Create permanent branches for all benchmark issues
python workflow_manager.py setup

# Fetch metadata for all commits
python workflow_manager.py fetch-metadata

# Update dataset with metadata
python workflow_manager.py update-dataset
```

### 2. Regular Operations

```bash
# Check current status
python workflow_manager.py status

# Monitor CI health (weekly/monthly)
python workflow_manager.py monitor

# Update metadata if needed
python workflow_manager.py fetch-metadata
```

### 3. Full Pipeline

```bash
# Run everything at once
python workflow_manager.py run-all
```

---

## 📖 Detailed Usage

### Workflow Manager (Recommended)

The `workflow_manager.py` orchestrates all operations:

```bash
python workflow_manager.py <command>

Commands:
  setup            Create permanent benchmark branches (one-time)
  fetch-metadata   Fetch commit metadata for all issues
  trigger-missing  Trigger CI for commits without metadata
  update-dataset   Update dataset with collected metadata
  monitor          Monitor CI workflow health
  status           Show current status
  run-all          Run complete pipeline
```

### Individual Scripts

#### 1. Setup Benchmark Branches

```bash
python setup_benchmark_branches.py
```

**What it does:**
- Creates permanent branches for each benchmark issue
- Branch naming: `benchmark_{owner}_{repo}_issue_{id}`
- Pushes to your fork: `RabeyaMuna/{repo}`
- Enables persistent CI testing

**Output:** `results/benchmark_branches.json`

---

#### 2. Fetch Commit Metadata

```bash
python fetch_commit_metadata.py
```

**What it does:**
- Fetches jobs and failed steps for all commits
- From `sha_fail` to `sha_success` for each issue
- Includes detailed step information for failed jobs

**Output:**
- `results/commit_job_metadata.json` - Complete metadata
- `results/missing_metadata_ids.json` - IDs needing CI trigger

**Structure:**
```json
{
  "id": "67",
  "repo": "RabeyaMuna/cowrie",
  "commits": [{
    "commit": "c34ad270...",
    "metadata": [{
      "job_name": "build (pypy-3.10)",
      "conclusion": "failure",
      "failed_steps": [{
        "name": "Install dependencies",
        "number": 5,
        "status": "completed",
        "conclusion": "failure"
      }],
      "all_steps": [...],
      "total_steps": 10,
      "failed_steps_count": 1
    }]
  }],
  "overall_failed_jobs": [...]
}
```

---

#### 3. Trigger CI for Missing Commits

```bash
python trigger_ci_for_commits.py
```

**What it does:**
- Reads `missing_metadata_ids.json`
- Creates/uses permanent branches
- Pushes to trigger CI on your fork
- Waits for CI completion
- Fetches new metadata

**Configuration:**
```python
BENCHMARK_OWNER = "RabeyaMuna"  # Your fork
WAIT_FOR_CI = True              # Wait for completion
MAX_WAIT_TIME = 1800            # 30 minutes max
```

---

#### 4. Monitor CI Health

```bash
python monitor_ci_health.py

# Quick check (sample of branches)
python monitor_ci_health.py --quick
```

**What it does:**
- Checks all benchmark branches
- Compares with baseline
- Detects changes:
  - Job count changes
  - Failure pattern changes
  - New/removed jobs
  - Workflow modifications

**Output:** `results/ci_workflow_health.json`

**Report includes:**
- ✓ OK - matches baseline
- ⚠️ Warning - non-critical changes
- ❌ Error - critical changes (ground truth may need update)

---

#### 5. Update Dataset

```bash
python update_failed_logs.py
```

**What it does:**
- Reads `commit_job_metadata.json`
- Extracts failed jobs and steps
- Updates `dataset/lca_dataset.parquet`
- Preserves existing structure

---

## 📊 Data Files

### Input
- `dataset/lca_dataset.parquet` - Main benchmark dataset (567 issues)

### Output

| File | Purpose | Updated |
|------|---------|---------|
| `benchmark_branches.json` | Permanent branch tracking | Once |
| `commit_job_metadata.json` | Complete metadata | As needed |
| `missing_metadata_ids.json` | IDs needing trigger | Auto-generated |
| `ci_workflow_health.json` | Health monitoring | Periodic |
| `ground_truth_changes.json` | Detected changes | When monitoring |

---

## 🔄 Workflows

### Initial Data Collection

```bash
# 1. Setup (one-time)
python workflow_manager.py setup

# 2. Fetch metadata
python workflow_manager.py fetch-metadata

# 3. If missing metadata found, trigger CI
python workflow_manager.py trigger-missing

# 4. Re-fetch after triggering
python workflow_manager.py fetch-metadata

# 5. Update dataset
python workflow_manager.py update-dataset
```

### Regular Maintenance (Weekly/Monthly)

```bash
# 1. Check health
python workflow_manager.py monitor

# 2. If changes detected, investigate
cat results/ci_workflow_health.json | jq '.summary'

# 3. Update metadata if needed
python workflow_manager.py fetch-metadata

# 4. Update dataset if ground truth changed
python workflow_manager.py update-dataset
```

### Before Benchmark Evaluation

```bash
# Quick health check
python monitor_ci_health.py --quick

# Ensure metadata is current
python workflow_manager.py status
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` in project root:

```bash
# Required
GH_TOKEN=your_github_personal_access_token

# Optional
BENCHMARK_OWNER=RabeyaMuna  # Your GitHub username
```

### Script Configuration

Edit directly in scripts:

**`trigger_ci_for_commits.py`:**
```python
WAIT_FOR_CI = True          # Wait for CI completion
CHECK_INTERVAL = 30         # Check every 30 seconds
MAX_WAIT_TIME = 1800        # 30 min timeout
```

**`fetch_commit_metadata.py`:**
```python
REQUEST_DELAY = 0.5         # Seconds between API requests
```

---

## 🎯 Use Cases

### 1. Initial Benchmark Setup
**Goal:** Collect metadata for all 567 issues

```bash
python workflow_manager.py run-all
```

### 2. Detect Workflow Changes
**Goal:** Monitor if CI workflows have been modified

```bash
python monitor_ci_health.py
```

### 3. Update Ground Truth
**Goal:** Add new metadata after detecting changes

```bash
python workflow_manager.py fetch-metadata
python workflow_manager.py update-dataset
```

### 4. Validate Benchmark Health
**Goal:** Ensure benchmarks still work before evaluation

```bash
python workflow_manager.py status
python monitor_ci_health.py --quick
```

---

## 📈 Status Checking

```bash
# Quick status
python workflow_manager.py status

# Detailed health report
python monitor_ci_health.py

# Check specific results
ls -lh results/
cat results/missing_metadata_ids.json | jq 'length'
cat results/ci_workflow_health.json | jq '.summary'
```

---

## 🐛 Troubleshooting

### No metadata found for commits

**Cause:** CI runs don't exist for those commits

**Solution:**
```bash
# Trigger CI for missing commits
python workflow_manager.py trigger-missing

# Re-fetch after CI completes
python workflow_manager.py fetch-metadata
```

### Rate limit errors

**Cause:** GitHub API rate limit exceeded

**Solution:**
- Ensure `GH_TOKEN` is set (5000 req/hr vs 60 req/hr)
- Increase `REQUEST_DELAY` in scripts
- Wait for rate limit reset

### Branch already exists

**Cause:** Branch created previously

**Solution:**
- This is normal! Branches are permanent
- Use `--force` flag to recreate (if needed)

### CI workflow changed

**Cause:** Upstream repo modified workflows

**Solution:**
1. Check health report for details
2. Decide if ground truth needs update
3. Re-fetch metadata if needed
4. Update dataset

---

## 🚨 Important Notes

### Permanent Branches
- Branches are **NOT deleted** after use
- Designed for long-term monitoring
- Can trigger CI repeatedly without recreation

### Fork Requirement
- All operations happen on YOUR fork (`RabeyaMuna/*`)
- Original repos are never modified
- Ensure fork exists before running

### Dataset Updates
- Updates are **additive** by default
- Original data is preserved
- Backup dataset before major updates

### API Rate Limits
- With token: 5000 requests/hour
- Without token: 60 requests/hour
- Scripts include rate limiting

---

## 📚 References

- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **GitHub API:** https://docs.github.com/en/rest
- **Benchmark Dataset:** `dataset/lca_dataset.parquet`

---

## 🔮 Future Enhancements

- [ ] Parallel processing for faster collection
- [ ] Incremental updates (only changed issues)
- [ ] Dashboard for health monitoring
- [ ] Automated scheduling (cron jobs)
- [ ] Notification system for changes
- [ ] Integration with evaluation pipeline

---

## ✅ Summary

**This system enables:**

1. ✅ **Persistent testing** via permanent branches
2. ✅ **Complete metadata** collection (jobs + steps)
3. ✅ **Automated monitoring** of CI health
4. ✅ **Change detection** for ground truth validation
5. ✅ **Dataset enrichment** with failure details
6. ✅ **Reusable infrastructure** without breaking changes

**Key Benefits:**

- No data loss between evaluations
- Automated workflow health monitoring
- Early detection of breaking changes
- Complete audit trail of CI failures
- Scalable to full benchmark (567 issues)

# Unified Fetch and Trigger Script

## ✅ What Was Created

### **`fetch_and_trigger_metadata.py`**
**Unified script that automatically:**
1. ✅ Checks if metadata exists
2. ✅ If exists → uses it
3. ✅ If missing → triggers CI → waits → fetches
4. ✅ Saves complete metadata + overall failed jobs/steps

**No manual intervention needed!**

---

## 🎯 Key Features

### 1. **Intelligent Strategy**
```
For each commit:
  ├─ Check if metadata exists
  ├─ If YES → fetch and use
  └─ If NO  → trigger CI on permanent branch
               ├─ Wait for CI completion
               └─ Fetch metadata
```

### 2. **Complete Output**
- ✅ `commit_job_metadata.json` - Complete metadata for ALL commits
- ✅ `failed_jobs_overall.json` - **NEW!** List of ALL failed jobs/steps from sha_fail to sha_success
- ✅ `missing_metadata_ids.json` - IDs that still need metadata

### 3. **Configuration-Driven**
Uses `config.yaml` and `.env`:
- ✅ GitHub token from `.env`
- ✅ All settings from `config.yaml`
- ✅ Easy to customize

---

## 📊 Output: `failed_jobs_overall.json`

**Complete list of ALL failed jobs and steps from sha_fail to sha_success:**

```json
[
  {
    "commit": "c34ad270...",
    "commit_order": 2,
    "job_name": "build (pypy-3.10)",
    "step_name": "Install dependencies",
    "step_number": 5,
    "conclusion": "failure"
  },
  {
    "commit": "1a2a90a4...",
    "commit_order": 3,
    "job_name": "build (pypy-3.10)",
    "step_name": "Run tests",
    "step_number": 7,
    "conclusion": "failure"
  }
]
```

**This is exactly what you need:**
- ✅ All failed jobs from failure to success
- ✅ Step-level details
- ✅ Commit order tracking
- ✅ Ready for analysis

---

## 🚀 Usage

### Simple Usage
```bash
cd data_managment
python fetch_and_trigger_metadata.py
```

That's it! The script will:
1. Load all 567 issues from dataset
2. For each issue, process all commits
3. Fetch existing metadata OR trigger CI
4. Save complete results

---

## ⚙️ Configuration

### `.env` (Credentials)
```bash
# GitHub Token (required)
GH_TOKEN=your_github_personal_access_token
```

### `config.yaml` (All Settings)
```yaml
github:
  benchmark_owner: "RabeyaMuna"
  rate_limit_delay: 0.5

ci_trigger:
  wait_for_completion: true
  check_interval: 30
  max_wait_time: 1800

data_collection:
  fetch_strategy: "auto"  # auto: fetch if available, trigger if missing
  checkpoint_interval: 10

paths:
  dataset: "../dataset/lca_dataset.parquet"
  results:
    metadata: "results/metadata"
    branches: "results/branches"

output:
  commit_metadata: "commit_job_metadata.json"
  missing_metadata: "missing_metadata_ids.json"
  failed_jobs_overall: "failed_jobs_overall.json"  # NEW!
```

---

## 📁 Organized Output Structure

```
results/
├── metadata/
│   ├── commit_job_metadata.json     # Complete metadata
│   ├── failed_jobs_overall.json     # ALL failed jobs/steps ⭐
│   └── missing_metadata_ids.json    # Still missing
│
├── branches/
│   └── benchmark_branches.json      # Permanent branches
│
├── health/
│   └── ci_workflow_health.json      # Health reports
│
└── logs/
    └── failed_job_logs.json         # Log content
```

---

## 🔄 How It Works

### Example: Issue #67 (RabeyaMuna/cowrie)

```
Commits: sha_fail → commit2 → commit3 → sha_success

Step 1: Check commit2
  ├─ API call: Check if CI ran
  ├─ Result: No metadata
  └─ Action: Trigger CI on permanent branch
      ├─ Push to branch
      ├─ Wait for CI (30s intervals)
      └─ Fetch metadata when complete

Step 2: Check commit3
  ├─ API call: Check if CI ran
  ├─ Result: Metadata exists!
  └─ Action: Fetch it

Step 3: Process results
  ├─ Collect all failed jobs
  ├─ Extract failed steps
  └─ Save to failed_jobs_overall.json
```

---

## 📊 Output Example

### `commit_job_metadata.json`
```json
{
  "id": "67",
  "commits": [
    {
      "commit": "c34ad270",
      "metadata": [
        {
          "job_name": "build (pypy-3.10)",
          "conclusion": "failure",
          "failed_steps": [
            {
              "name": "Install dependencies",
              "number": 5,
              "conclusion": "failure"
            }
          ],
          "total_steps": 10
        }
      ]
    }
  ]
}
```

### `failed_jobs_overall.json` (NEW!)
```json
[
  {
    "commit": "c34ad270",
    "commit_order": 2,
    "job_name": "build (pypy-3.10)",
    "step_name": "Install dependencies",
    "step_number": 5,
    "conclusion": "failure"
  }
]
```

**Perfect for:**
- ✅ Analysis of failure patterns
- ✅ Understanding what failed where
- ✅ Tracking failure progression
- ✅ Dataset enrichment

---

## 🆚 Old vs New Approach

### Before (Separate Scripts)
```bash
# 1. Fetch metadata
python fetch_commit_metadata.py
# → missing_metadata_ids.json

# 2. Trigger missing
python trigger_ci_for_commits.py
# → wait manually

# 3. Re-fetch
python fetch_commit_metadata.py
# → still might have missing

# Manual intervention required!
```

### Now (Unified Script)
```bash
# One command
python fetch_and_trigger_metadata.py
# → Automatically fetch OR trigger
# → Complete metadata
# → Overall failed jobs list
# → Ready to use!
```

---

## ⚡ Performance

### With Existing Metadata
- **Fast:** Just API calls to fetch
- **~2-3 hours** for 567 issues (with rate limiting)

### With Missing Metadata
- **Slower:** Triggers CI + waits
- **~30 min per commit** (depends on CI duration)
- **But automatic!** No manual intervention

### Smart Checkpoint System
- Saves every 10 issues
- Can resume if interrupted
- No data loss

---

## 🎯 Use Cases

### 1. Initial Data Collection
```bash
python fetch_and_trigger_metadata.py
```
**Result:** Complete metadata for all 567 issues

### 2. Update After Changes
```bash
# Edit config.yaml to process specific IDs
python fetch_and_trigger_metadata.py
```
**Result:** Only processes specified issues

### 3. Retry Failed Issues
```bash
# Script skips issues with complete metadata
python fetch_and_trigger_metadata.py
```
**Result:** Only processes missing/incomplete issues

---

## 📈 Progress Tracking

**During execution:**
```
================================================================================
Processing issue 67/567

[67] RabeyaMuna/cowrie
  SHA fail: 55f8e668, SHA success: 0747db16
  Found 3 commits
  [1/3] 55f8e668... No metadata - triggering...
    Triggering CI for 55f8e668...
    ✓ CI triggered
    Waiting for CI (max 1800s)...
    ✓ CI completed (120s)
  ✓ 7 jobs, 1 failed
  [2/3] c34ad270... ✓ 7 jobs, 1 failed
  [3/3] 1a2a90a4... ✓ 7 jobs, 1 failed
  Fetching success commit 0747db16... ✓ 7 jobs
  Summary: 4 commits, 2 failed steps
  Triggered CI for 1 commits

  💾 Checkpoint: 67 issues saved
```

---

## 🔧 Customization

### Trigger Settings
```yaml
ci_trigger:
  wait_for_completion: true   # Set false for faster (but may miss metadata)
  check_interval: 30          # How often to check CI status
  max_wait_time: 1800        # Max wait per commit (30 min)
```

### Processing Options
```yaml
data_collection:
  fetch_strategy: "auto"      # auto | fetch_only | trigger_only
  checkpoint_interval: 10     # Save every N issues
```

### Paths
```yaml
paths:
  results:
    metadata: "results/metadata"  # Where to save
```

---

## ✅ Benefits

1. **Fully Automated**
   - No manual intervention
   - Handles fetch and trigger seamlessly

2. **Complete Data**
   - All metadata collected
   - Failed jobs list generated
   - Missing IDs tracked

3. **Configuration-Driven**
   - Easy to customize
   - No code changes needed
   - Credentials in .env

4. **Organized Output**
   - Structured folders
   - Clear file purposes
   - Easy to find data

5. **Robust**
   - Checkpoint system
   - Error handling
   - Resume capability

---

## 📖 Related Files

- **Configuration:** [config.yaml](config.yaml)
- **Credentials:** `.env` (in project root)
- **Output Documentation:** [results/README.md](results/README.md)
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 🚀 Quick Start

1. **Setup credentials:**
   ```bash
   # Edit .env
   GH_TOKEN=your_token_here
   ```

2. **Customize config (optional):**
   ```bash
   # Edit config.yaml
   nano config.yaml
   ```

3. **Run:**
   ```bash
   python fetch_and_trigger_metadata.py
   ```

4. **Check results:**
   ```bash
   ls -lh results/metadata/
   ```

That's it! You have complete metadata with overall failed jobs/steps! 🎉

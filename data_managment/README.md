# CI Benchmark Data Management

**Complete system for managing CI benchmark data with organized folder structure.**

---

## 📁 Folder Structure

```
data_managment/
│
├── 📖 docs/                    # All documentation
│   ├── README.md                  # Complete user guide
│   ├── GETTING_STARTED.md         # Quick start guide
│   ├── UNIFIED_SCRIPT.md          # Unified script documentation
│   └── FOLDER_ORGANIZATION.md     # This folder structure explained
│
├── 🔧 scripts/                 # Main executable scripts
│   ├── run_approach_b.sh          # ⭐ One-command execution (Approach B)
│   ├── fetch_and_trigger_metadata.py  # Unified fetch+trigger
│   ├── setup_benchmark_branches.py    # Create permanent branches
│   ├── monitor_ci_health.py       # CI health monitoring
│   └── workflow_manager.py        # Orchestration tool
│
├── 🛠️  utils/                  # Utility scripts
│   ├── fetch_commit_metadata.py   # Fetch metadata only
│   ├── trigger_ci_for_commits.py  # Trigger CI only
│   ├── fetch_logs.py              # Fetch logs
│   └── update_failed_logs.py      # Update dataset
│
├── ⚙️  config/                 # Configuration files
│   └── config.yaml                # All settings (Approach B default)
│
├── 📊 results/                 # All outputs (organized)
│   ├── branches/                  # Benchmark branch tracking
│   ├── metadata/                  # Commit metadata + failed jobs
│   ├── health/                    # CI health reports
│   ├── logs/                      # Fetched logs
│   └── archives/                  # Historical backups
│
└── 🔬 failure_type/            # Failure classification (existing)
```

---

## 🚀 Quick Start (Approach B)

### Complete Data Collection

**Goal:** Get 100% complete metadata for all commits from sha_fail to sha_success.

**Strategy:** Fetch if exists, Trigger if missing.

### One Command:

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/data_managment

./scripts/run_approach_b.sh
```

That's it! Script will:
1. ✅ Setup benchmark branches (if needed)
2. ✅ For each commit: fetch OR trigger
3. ✅ Save complete metadata + overall failed jobs list
4. ✅ Checkpoint every 10 issues

**Time:** 10-20 hours for 567 issues

---

## 📋 Step-by-Step

### 1. Setup (Required)

**Add GitHub token to `.env`:**
```bash
# In project root
echo "GH_TOKEN=your_github_token" >> ../.env
```

### 2. Configure (Optional)

**Edit settings in `config/config.yaml`:**
```yaml
github:
  benchmark_owner: "RabeyaMuna"  # Your fork owner
  
ci_trigger:
  wait_for_completion: true      # Wait for CI
  max_wait_time: 1800           # 30 min max
  
data_collection:
  fetch_strategy: "complete"     # Approach B
  trigger_missing: true          # Trigger if no metadata
```

### 3. Run

**Option A: Automated (recommended):**
```bash
./scripts/run_approach_b.sh
```

**Option B: Step by step:**
```bash
# 1. Create branches (once)
python scripts/setup_benchmark_branches.py

# 2. Fetch + trigger
python scripts/fetch_and_trigger_metadata.py
```

**Option C: Workflow manager:**
```bash
python scripts/workflow_manager.py run-all
```

---

## 📊 Outputs

All results saved in organized `results/` folders:

### Primary Outputs

| File | Location | Description |
|------|----------|-------------|
| **Commit Metadata** | `results/metadata/commit_job_metadata.json` | Complete metadata for all commits |
| **Failed Jobs List** ⭐ | `results/metadata/failed_jobs_overall.json` | ALL failed jobs/steps from fail→success |
| **Missing IDs** | `results/metadata/missing_metadata_ids.json` | Issues still needing metadata |
| **Benchmark Branches** | `results/branches/benchmark_branches.json` | Permanent branch tracking |

### Example: Failed Jobs List

```json
[
  {
    "commit": "c34ad270...",
    "commit_order": 2,
    "job_name": "build (pypy-3.10)",
    "step_name": "Install dependencies",
    "step_number": 5,
    "conclusion": "failure"
  }
]
```

**Perfect for:**
- ✅ Understanding failure progression
- ✅ Adding to dataset
- ✅ Training models
- ✅ Analysis

---

## 📖 Documentation

### Start Here

1. **[README.md](docs/README.md)** - Complete user guide
2. **[GETTING_STARTED.md](docs/GETTING_STARTED.md)** - Quick start tutorial

### Detailed Guides

3. **[UNIFIED_SCRIPT.md](docs/UNIFIED_SCRIPT.md)** - Unified script documentation
4. **[FOLDER_ORGANIZATION.md](docs/FOLDER_ORGANIZATION.md)** - Folder structure details

### Results

5. **[results/README.md](results/README.md)** - Results folder documentation

---

## 🎯 Common Tasks

### Check Status
```bash
python scripts/workflow_manager.py status
```

### Monitor CI Health
```bash
python scripts/monitor_ci_health.py
```

### View Results
```bash
# Count issues
jq 'length' results/metadata/commit_job_metadata.json

# Count failed jobs
jq 'length' results/metadata/failed_jobs_overall.json

# View sample
jq '.[0]' results/metadata/failed_jobs_overall.json
```

### Update Dataset
```bash
python utils/update_failed_logs.py
```

---

## 🔄 Approach B Details

**What is Approach B?**

For EACH commit between sha_fail and sha_success:
1. Check if CI metadata exists
2. If **YES** → Fetch it (fast)
3. If **NO** → Trigger CI → Wait → Fetch (complete)

**Result:** 100% complete metadata

**Benefits:**
- ✅ Complete data for ALL commits
- ✅ Only triggers when needed (saves CI minutes)
- ✅ Automatic - no manual intervention
- ✅ Optimal balance of speed and completeness

---

## 🛠️ Scripts Overview

### Main Scripts (`scripts/`)

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `run_approach_b.sh` | One-command execution | **Start here!** ⭐ |
| `fetch_and_trigger_metadata.py` | Unified fetch+trigger | Approach B collection |
| `setup_benchmark_branches.py` | Create permanent branches | One-time setup |
| `monitor_ci_health.py` | Check CI health | Periodic monitoring |
| `workflow_manager.py` | Orchestrate operations | Alternative runner |

### Utility Scripts (`utils/`)

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `fetch_commit_metadata.py` | Fetch only (no trigger) | Quick data collection |
| `trigger_ci_for_commits.py` | Trigger only | Standalone triggering |
| `fetch_logs.py` | Fetch log content | Log collection |
| `update_failed_logs.py` | Update dataset | After collection |

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# Required
GH_TOKEN=your_github_personal_access_token

# Optional
BENCHMARK_OWNER=RabeyaMuna
```

### Settings (`config/config.yaml`)

```yaml
# Approach B configuration
data_collection:
  fetch_strategy: "complete"     # Fetch OR trigger
  trigger_missing: true          # Always trigger if missing

# Customize as needed
ci_trigger:
  wait_for_completion: true
  max_wait_time: 1800
  
github:
  benchmark_owner: "RabeyaMuna"
  rate_limit_delay: 0.5
```

---

## 📈 Expected Timeline

### Full Run (567 issues)

**Setup phase (one-time):**
- Create branches: 30-60 min

**Collection phase:**
- Commits with metadata: Fast (2-3 min/issue)
- Commits needing trigger: Slow (10-30 min/issue)
- **Total: 10-20 hours**

**Checkpoints:**
- Saves every 10 issues
- Can resume if interrupted

---

## 🔍 Verification

### After Completion

```bash
# Check completeness
jq 'length' results/metadata/commit_job_metadata.json

# View failed jobs
jq 'length' results/metadata/failed_jobs_overall.json

# Check missing
jq 'length' results/metadata/missing_metadata_ids.json

# Status summary
python scripts/workflow_manager.py status
```

---

## 🚨 Troubleshooting

### Common Issues

**Issue:** GitHub token not found
```bash
# Solution: Add to .env
echo "GH_TOKEN=your_token" >> ../.env
```

**Issue:** Rate limit errors
```yaml
# Solution: Increase delay in config/config.yaml
github:
  rate_limit_delay: 1.0  # Increase to 1 second
```

**Issue:** CI timeout
```yaml
# Solution: Increase timeout in config/config.yaml
ci_trigger:
  max_wait_time: 3600  # Increase to 60 minutes
```

---

## 📊 Folder Details

### `scripts/` - Main Scripts
Executable scripts for data collection and management.

### `utils/` - Utilities
Helper scripts for specific tasks (fetch-only, trigger-only, etc.)

### `config/` - Configuration
All settings in YAML format.

### `results/` - Outputs
All generated data organized by type:
- `branches/` - Branch tracking
- `metadata/` - Complete metadata
- `health/` - Health reports
- `logs/` - Log content
- `archives/` - Backups

### `docs/` - Documentation
Complete documentation for all features.

---

## 🎯 Use Cases

### 1. Initial Setup
```bash
./scripts/run_approach_b.sh
```

### 2. Monitor Health
```bash
python scripts/monitor_ci_health.py
```

### 3. Update Metadata
```bash
python scripts/fetch_and_trigger_metadata.py
```

### 4. Add to Dataset
```bash
python utils/update_failed_logs.py
```

---

## ✅ Summary

**You have a complete system for:**

1. ✅ **Permanent benchmark infrastructure**
   - Branches persist across runs
   - No data loss

2. ✅ **Complete metadata collection**
   - 100% data with Approach B
   - Failed jobs + steps

3. ✅ **Organized structure**
   - Clean folder organization
   - Easy to navigate

4. ✅ **One-command execution**
   - `./scripts/run_approach_b.sh`
   - Fully automated

5. ✅ **Configuration-driven**
   - `.env` for credentials
   - `config.yaml` for settings

---

## 🚀 Get Started

```bash
# 1. Add token
echo "GH_TOKEN=your_token" >> ../.env

# 2. Run
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/data_managment
./scripts/run_approach_b.sh

# 3. Wait ~10-20 hours

# 4. Get complete benchmark dataset! 🎉
```

---

## 📚 More Information

- **Complete Guide:** [docs/README.md](docs/README.md)
- **Quick Start:** [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- **Results Info:** [results/README.md](results/README.md)

---

**Need help?** Check the documentation in `docs/` folder.

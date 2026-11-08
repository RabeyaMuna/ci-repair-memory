# CI-Repair-Bench Monthly Maintenance Workflow

Complete step-by-step workflow for monthly health checks, manual fixes, and versioning.

---

## 📋 **Complete Workflow**

### **Phase 1: Health Check (Automated)**

#### **Step 1: Fetch Ground Truth & Run Health Check**

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH

# Run health check on all instances
python3 maintenance/run_health_check.py \
  --config config.yaml \
  --dataset data/ci_repair_bench_v1.parquet
```

**What this does:**
1. ✅ **Fetches ground truth** from dataset:
   - `id` - Instance identifier
   - `sha_fail` - Failing commit SHA  
   - `diff_fix` - Ground truth patch to apply

2. ✅ **Applies patch & pushes to GitHub**:
   - Clones repo at `sha_fail`
   - Applies `diff_fix` patch
   - Pushes to branch: `health_check_{id}_{timestamp}`
   - Triggers GitHub Actions workflow

3. ✅ **Polls for job results**:
   - Monitors workflow execution
   - Tracks execution time and queue delays
   - Records final status: `pass`/`fail`/`timeout`/`cancelled`

4. ✅ **Generates reports**:
   - `health_check_{timestamp}_full.json` - All instances
   - `health_check_{timestamp}_flagged.json` - Failed/timeout/cancelled only

**Output:** `maintenance/reports/health_check/`

---

### **Phase 2: Manual Inspection (Manual)**

#### **Step 2: Review Flagged Instances**

```bash
# View flagged instances summary
cat maintenance/reports/health_check/*_flagged.json | jq -r '
  "Total flagged: \(.total_flagged)\n",
  "Instances requiring attention:\n",
  (.flagged_instances[] | 
   "  ID: \(.id) | Repo: \(.repo)\n  Status: \(.status) | Message: \(.message)\n")'

# Or view full details
cat maintenance/reports/health_check/*_flagged.json | jq '.'
```

#### **Step 3: Manual Inspection & Fixes**

For each flagged instance, investigate and fix:

**Common issues:**
- ❌ **Failed**: Ground truth patch no longer works
  - Check if upstream dependencies changed
  - Check if GitHub Actions runner environment changed
  - Update `diff_fix` in dataset
  - **Check for flakiness** (see below)

- ⏱️ **Timeout**: Workflow took too long
  - Check workflow logs for hanging tests
  - May need to remove instance if permanently broken

- 🚫 **Cancelled**: Workflow was cancelled
  - Check GitHub Actions status
  - May be temporary GitHub outage

- ❓ **Invalid**: Repository deleted or inaccessible
  - Remove from dataset

#### **Optional: Detect Flaky Workflows**

If instances fail inconsistently, check for flakiness:

```bash
# Run multiple times on suspected flaky instances
python3 maintenance/run_health_check.py \
  --config config.yaml \
  --dataset data/ci_repair_bench_v1.parquet \
  --ids 42 67  # Suspected flaky instances

# Analyze for flakiness patterns
python3 maintenance/detect_flaky_workflows.py \
  --runs 3 \
  --threshold 0.5
```

**Flaky indicators:**
- Same commit has different outcomes across runs
- Execution time varies >50% of mean
- Success rate between 0-100% (not always pass/fail)

**How to fix:**
1. Open the flagged instance in your dataset file
2. Update the `diff_fix` patch if needed
3. Or mark for removal if invalid

---

### **Phase 3: Verification (Automated)**

#### **Step 4: Re-run Health Check on Fixed Instances**

```bash
# Test specific fixed instances
python3 maintenance/run_health_check.py \
  --config config.yaml \
  --dataset data/ci_repair_bench_v1.parquet \
  --ids 42 67 89

# Or re-run all instances
python3 maintenance/run_health_check.py \
  --config config.yaml \
  --dataset data/ci_repair_bench_v1.parquet
```

**Keep re-running until all instances pass!**

---

### **Phase 4: Dataset Comparison (Automated)**

#### **Step 5: Compare with Hugging Face Dataset**

Once **all instances pass**, compare your local dataset with the official Hugging Face version:

```bash
python3 maintenance/compare_with_huggingface.py \
  --local-dataset data/ci_repair_bench_v1.parquet \
  --hf-repo RabeyaMuna/ci-repair-bench \
  --hf-filename ci_repair_bench_v1.parquet
```

**What this does:**
1. Downloads official dataset from Hugging Face
2. Compares with your local dataset
3. Identifies:
   - ✨ **Added instances** (in local, not in HF)
   - 🗑️ **Removed instances** (in HF, not in local)
   - 🔄 **Modified instances** (different `diff_fix`, `sha_fail`, etc.)
4. Generates reports:
   - `hf_comparison_{timestamp}.json` - Full comparison data
   - `hf_comparison_{timestamp}.md` - Human-readable report

**Output:** `maintenance/reports/`

---

### **Phase 5: Versioning & Documentation (Automated)**

#### **Step 6: Create Versioned Release**

```bash
# Full monthly maintenance (includes versioning)
python3 maintenance/run_monthly_maintenance.py \
  --config config.yaml \
  --dataset data/ci_repair_bench_v1.parquet \
  --version-level patch

# With HuggingFace comparison (only if all instances passed)
python3 maintenance/run_monthly_maintenance.py \
  --config config.yaml \
  --dataset data/ci_repair_bench_v1.parquet \
  --version-level patch \
  --compare-hf
```

**What this does:**
1. Runs health check
2. Compares with previous month's results
3. Generates changelog from differences
4. Updates `VERSION.txt` and `CHANGELOG.md`
5. (Optional) Compares with Hugging Face if all passed
6. Creates maintenance reports

**Output:**
- `VERSION.txt` - Updated version number
- `CHANGELOG.md` - Updated with changes
- `maintenance/reports/maintenance_report_{YYYYMM}.json`
- `maintenance/reports/maintenance_summary_{YYYYMM}.md`
- (Optional) `maintenance/reports/hf_comparison_{timestamp}.{json,md}`

---

## 🔄 **Quick Reference Commands**

### **Complete Workflow (All-in-One)**

```bash
# 1. Initial health check
python3 maintenance/run_health_check.py \
  --config config.yaml \
  --dataset data/ci_repair_bench_v1.parquet

# 2. Review flagged instances
cat maintenance/reports/health_check/*_flagged.json | jq '.flagged_instances'

# 3. Manually fix instances in dataset...

# 4. Re-run health check until all pass
python3 maintenance/run_health_check.py \
  --config config.yaml \
  --dataset data/ci_repair_bench_v1.parquet

# 5. Once all pass, run full maintenance with HF comparison
python3 maintenance/run_monthly_maintenance.py \
  --config config.yaml \
  --dataset data/ci_repair_bench_v1.parquet \
  --version-level patch \
  --compare-hf
```

### **Standalone HuggingFace Comparison**

```bash
# Run comparison separately (after all instances pass)
python3 maintenance/compare_with_huggingface.py \
  --local-dataset data/ci_repair_bench_v1.parquet
```

### **Check Specific Instances Only**

```bash
# Test just a few instances
python3 maintenance/run_health_check.py \
  --config config.yaml \
  --dataset data/ci_repair_bench_v1.parquet \
  --ids 1 5 10 15 20
```

---

## 📊 **Report Files Generated**

| File | Description |
|------|-------------|
| `maintenance/reports/health_check/health_check_{timestamp}_full.json` | All health check results |
| `maintenance/reports/health_check/health_check_{timestamp}_flagged.json` | Failed/timeout/cancelled instances only |
| `maintenance/reports/maintenance_report_{YYYYMM}.json` | Monthly maintenance report with recommendations |
| `maintenance/reports/maintenance_summary_{YYYYMM}.md` | Human-readable maintenance summary |
| `maintenance/reports/hf_comparison_{timestamp}.json` | HuggingFace comparison data |
| `maintenance/reports/hf_comparison_{timestamp}.md` | HuggingFace comparison report |
| `VERSION.txt` | Current benchmark version |
| `CHANGELOG.md` | Version history and changes |

---

## ⚠️ **Important Notes**

1. **Always set `GITHUB_TOKEN`** before running:
   ```bash
   export GITHUB_TOKEN=your_github_token_here
   ```

2. **HuggingFace comparison runs ONLY when**:
   - All health check instances pass
   - `--compare-hf` flag is used
   - Or run separately after fixes

3. **Version levels**:
   - `patch` - Minor fixes, small updates (default)
   - `minor` - Instance additions/removals, workflow updates  
   - `major` - Breaking changes, format changes

4. **Dry run mode** (health check only, no version update):
   ```bash
   python3 maintenance/run_monthly_maintenance.py \
     --config config.yaml \
     --dataset data/ci_repair_bench_v1.parquet \
     --dry-run
   ```

---

## 🎯 **Decision Tree**

```
Start: Run health check
  ↓
All instances pass? 
  ├─ YES → Compare with HuggingFace → Generate version/changelog → DONE
  └─ NO → Review flagged instances
           ↓
         Manual inspection & fixes
           ↓
         Re-run health check
           ↓
         All pass now?
           ├─ YES → Compare with HuggingFace → Generate version/changelog → DONE
           └─ NO → Keep fixing and re-running
```

---

## 📞 **Troubleshooting**

### Health check fails with authentication error
```bash
# Make sure GITHUB_TOKEN is set
export GITHUB_TOKEN=your_token_here
```

### HuggingFace download fails
```bash
# Install huggingface_hub
pip install huggingface_hub
```

### All instances timing out
```bash
# Check GitHub Actions status
# https://www.githubstatus.com/
# May need to wait for GitHub service restoration
```

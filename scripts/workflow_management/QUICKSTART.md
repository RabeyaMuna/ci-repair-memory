# Quick Start: Dataset Metadata Enrichment

## TL;DR

Add metadata (total jobs, failed jobs, line changes) to your dataset:

```bash
# 1. Set your GitHub token
export GITHUB_TOKEN='your_token_here'

# 2. Run enrichment
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH
python scripts/workflow_management/enrich_metadata.py

# 3. Validate results
python scripts/workflow_management/validate_enrichment.py
```

**Output**: `dataset/lca_dataset_enriched.parquet` with 6 new metadata columns.

---

## Step-by-Step Guide

### Step 1: Get a GitHub Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (or just `public_repo` if all repos are public)
4. Generate and copy the token

```bash
export GITHUB_TOKEN='ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
```

To make it persistent, add to your `~/.zshrc` or `~/.bashrc`:
```bash
echo 'export GITHUB_TOKEN="ghp_your_token"' >> ~/.zshrc
source ~/.zshrc
```

### Step 2: Run the Enrichment Script

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH
python scripts/workflow_management/enrich_metadata.py
```

**Expected output:**
```
Loading dataset...
Loading failed jobs...
Loading jobs URLs...
Processing 567 records...
[1/567] Fetching total jobs for ID 44 (run 31636993972)...
  Processed 10/567 records
[11/567] Fetching total jobs for ID 67 (run 31637193148)...
  Processed 20/567 records
...

Saving enriched dataset to dataset/lca_dataset_enriched.parquet...

============================================================
SUMMARY STATISTICS
============================================================
Total records: 567

Line changes:
  Avg lines inserted: 12.45
  Avg lines deleted: 8.32
  Avg total changed: 20.77

Failed jobs:
  Avg failed jobs per issue: 1.85
  Max failed jobs: 12

Total jobs (matrix-expanded):
  Avg total jobs: 3.24
  Max total jobs: 48
============================================================

✅ Dataset enrichment complete!
```

**Time**: ~10-15 minutes for 567 issues (with GitHub API calls)

### Step 3: Validate the Enriched Dataset

```bash
python scripts/workflow_management/validate_enrichment.py
```

**Expected output:**
```
======================================================================
DATASET ENRICHMENT VALIDATION
======================================================================

📂 Loading enriched dataset from: dataset/lca_dataset_enriched.parquet
✓ Loaded 567 records

======================================================================
1. CHECKING REQUIRED COLUMNS
======================================================================
✓ total_jobs              - present
✓ num_failed_jobs         - present
✓ failed_jobs             - present
✓ lines_inserted          - present
✓ lines_deleted           - present
✓ total_lines_changed     - present

...

======================================================================
VALIDATION RESULT
======================================================================
✅ VALIDATION PASSED - Dataset enrichment successful!

You can now use the enriched dataset:
  /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset_enriched.parquet
```

### Step 4: Use the Enriched Dataset

```python
import pandas as pd

# Load enriched dataset
df = pd.read_parquet('dataset/lca_dataset_enriched.parquet')

# Check new columns
print(df.columns)
# Output includes: total_jobs, failed_jobs, num_failed_jobs, 
#                  lines_inserted, lines_deleted, total_lines_changed

# Example: Filter by complexity
simple_fixes = df[df['total_lines_changed'] < 20]
complex_fixes = df[df['total_lines_changed'] > 50]

# Example: Analyze failed jobs
most_common_failures = df['failed_jobs'].explode().value_counts()
print(most_common_failures.head(10))
```

---

## Troubleshooting

### Problem: Rate limit exceeded

**Symptoms:**
```
[API Error 403] https://api.github.com/...
Rate limit exceeded. Waiting 3600s...
```

**Solutions:**
1. Wait for rate limit reset (shown in script output)
2. Use a different GitHub token
3. Run with `FETCH_TOTAL_JOBS = False` in the script (skips API calls)

### Problem: Some total_jobs = 0

**Cause:** Workflow run was deleted or is from a different fork

**Solution:** This is expected behavior. The script handles it gracefully.

### Problem: GITHUB_TOKEN not found

**Symptoms:**
```
Warning: GITHUB_TOKEN not found in environment variables
Proceeding without fetching total jobs from GitHub API...
```

**Solution:**
```bash
export GITHUB_TOKEN='your_token_here'
# Re-run the enrichment script
```

### Problem: Missing repo_owner in dataset

**Symptoms:**
```
Warning: No repo_owner found for ID 44
```

**Solution:** The script tries to parse `repo_owner` from `repo_name`. If this fails, manually add a `repo_owner` column to your dataset.

---

## What Gets Added?

### New Columns

| Column | Description | Example |
|--------|-------------|---------|
| `total_jobs` | Total validation jobs (with matrix) | `6` |
| `num_failed_jobs` | Count of failed jobs | `2` |
| `failed_jobs` | List of failed job names | `["test (3.9)", "lint"]` |
| `lines_inserted` | Lines added in diff | `45` |
| `lines_deleted` | Lines removed in diff | `12` |
| `total_lines_changed` | Total changes | `57` |

### Example Record

**Before enrichment:**
```json
{
  "id": "44",
  "repo_name": "wandb",
  "diff": "diff --git a/file.py...",
  "changed_files": ["file.py"]
}
```

**After enrichment:**
```json
{
  "id": "44",
  "repo_name": "wandb",
  "diff": "diff --git a/file.py...",
  "changed_files": ["file.py"],
  
  "total_jobs": 1,
  "num_failed_jobs": 1,
  "failed_jobs": ["pre-commit"],
  "lines_inserted": 15,
  "lines_deleted": 8,
  "total_lines_changed": 23
}
```

---

## Integration with Your Workflow

### Option 1: Replace Original Dataset

```bash
# Backup original
cp dataset/lca_dataset.parquet dataset/lca_dataset_backup.parquet

# Replace with enriched
cp dataset/lca_dataset_enriched.parquet dataset/lca_dataset.parquet
```

### Option 2: Update run_benchmark.py

```python
# In run_benchmark.py, change:
dataset_info = os.path.join(config.get("base_dir"), "dataset", "lca_dataset.parquet")

# To:
dataset_info = os.path.join(config.get("base_dir"), "dataset", "lca_dataset_enriched.parquet")
```

### Option 3: Load Both (for comparison)

```python
df_original = pd.read_parquet("dataset/lca_dataset.parquet")
df_enriched = pd.read_parquet("dataset/lca_dataset_enriched.parquet")

# Compare
print(f"Original columns: {len(df_original.columns)}")
print(f"Enriched columns: {len(df_enriched.columns)}")
```

---

## Advanced Usage

### Run Without API Calls (Faster)

Edit `enrich_metadata.py` line 372:
```python
FETCH_TOTAL_JOBS = False  # Skip GitHub API calls
```

Then run normally. You'll still get diff stats and failed jobs.

### Custom Paths

Edit `enrich_metadata.py` lines 363-366:
```python
DATASET_PATH = "path/to/your/dataset.parquet"
FAILED_JOBS_PATH = "path/to/failed_jobs.json"
JOBS_IDS_PATH = "path/to/jobs_ids.jsonl"
OUTPUT_PATH = "path/to/output.parquet"
```

### Process Subset of Issues

Edit `enrich_metadata.py` to add filtering:
```python
# After loading dataset
df = pd.read_parquet(dataset_path)

# Filter to specific IDs
df = df[df['id'].isin(['44', '64', '63'])]

# Or filter by repo
df = df[df['repo_name'] == 'wandb']

# Continue with enrichment...
```

---

## Next Steps

After enrichment:

1. **Analyze metadata distribution**
   ```python
   df = pd.read_parquet('dataset/lca_dataset_enriched.parquet')
   
   print(df['total_jobs'].describe())
   print(df['num_failed_jobs'].value_counts())
   print(df['total_lines_changed'].describe())
   ```

2. **Create stratified splits**
   ```python
   # Split by complexity
   df['complexity'] = pd.cut(
       df['total_lines_changed'],
       bins=[0, 10, 50, float('inf')],
       labels=['simple', 'medium', 'complex']
   )
   ```

3. **Update evaluation scripts**
   ```python
   # Group results by metadata
   results.groupby('complexity').agg({
       'success_rate': 'mean',
       'num_failed_jobs': 'mean'
   })
   ```

4. **Visualize distributions**
   ```python
   import matplotlib.pyplot as plt
   
   df['total_lines_changed'].hist(bins=50)
   plt.xlabel('Total Lines Changed')
   plt.ylabel('Frequency')
   plt.title('Distribution of Change Sizes')
   plt.show()
   ```

---

## Files Created

- `enrich_metadata.py` - Main enrichment script
- `validate_enrichment.py` - Validation script
- `README_METADATA_ENRICHMENT.md` - Detailed documentation
- `METADATA_STRUCTURE.md` - Schema and examples
- `QUICKSTART.md` - This guide

All in: `/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/scripts/workflow_management/`

---

## Questions?

- Check `README_METADATA_ENRICHMENT.md` for detailed docs
- See `METADATA_STRUCTURE.md` for schema details
- Review script comments in `enrich_metadata.py`

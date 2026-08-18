# Complete Dataset Information Extraction

**Script**: `scripts/analysis/extract_dataset_info.py`

## What It Extracts

For **each issue**, extracts comprehensive information:

```json
{
  "id": "269",
  "sha_fail": "f57afa39e8c9dc4b57c95a021af8a588f6c8c822",
  "sha_success": "25ff649efe5018fc49fbf313eb05d204754309ae",
  "repo": "qutebrowser/qutebrowser",
  "workflow_name": "CI",
  "workflow_path": ".github/workflows/ci.yml",
  
  "ground_truth": {
    "files_changed": 7,
    "lines_inserted": 42,
    "lines_deleted": 80,
    "lines_modified": 19,
    "changed_files": ["file1.py", "file2.py", ...]
  },
  
  "validation_steps": {
    "total_steps": 45,
    "all_steps": ["linters (pylint)", "tests (py3.9)", ...],
    "passed_steps": ["linters (pylint)", ...],
    "failed_steps": ["tests (py3.9)", ...],
    "skipped_steps": [...],
    "cancelled_steps": [...]
  },
  
  "failed_jobs": {
    "count": 18,
    "jobs": ["tests (py3.9)", "tests (py3.10)", ...]
  }
}
```

## Usage

### Basic (Parse Workflow YAML)

```bash
python scripts/analysis/extract_dataset_info.py
```

Output: `dataset/dataset_complete_info.json`

### With GitHub API (Recommended - More Accurate)

```bash
# Set token
export GITHUB_TOKEN=your_token_here

# Run with API
python scripts/analysis/extract_dataset_info.py --use-api
```

**Why use API?**
- ✅ Accurate step names (matrix expanded)
- ✅ Passed/failed status per step
- ✅ Reflects actual execution
- ✅ No template variables

### Custom Output

```bash
python scripts/analysis/extract_dataset_info.py \
  --output my_output.json \
  --use-api \
  --token $GITHUB_TOKEN
```

## Data Sources

| Field | Source | Method |
|-------|--------|--------|
| `id`, `sha_fail`, `repo` | Dataset | Direct fields |
| `ground_truth.*` | Dataset `diff` field | Parse unified diff |
| `validation_steps.*` (--use-api) | GitHub API | Fetch from Actions API |
| `validation_steps.*` (default) | Dataset `workflow` | Parse YAML |
| `failed_jobs.*` | Dataset `logs` field | Extract step names |

## Output Statistics

```
📊 Overall Statistics
  Total issues:           567

📝 Ground Truth (from diffs)
  Total files changed:    5937
  Total lines inserted:   78085
  Total lines deleted:    50618
  Avg files per issue:    10.5
  Avg lines per issue:    227.0

✓ Validation Steps
  Total validation steps: 2477
  Avg steps per issue:    4.4
  
  From GitHub API:
    Passed steps:         1279
    Failed steps:         1198

✗ Failed Jobs (from CI logs)
  Total failed jobs:      1198
  Avg failed per issue:   2.1
```

## Use Cases

### 1. Feature Extraction for ML

```python
import json

# Load complete info
with open('dataset/dataset_complete_info.json') as f:
    data = json.load(f)

# Extract features
for issue in data:
    features = {
        'id': issue['id'],
        'num_files': issue['ground_truth']['files_changed'],
        'num_lines': issue['ground_truth']['lines_inserted'] + 
                     issue['ground_truth']['lines_deleted'],
        'total_steps': issue['validation_steps']['total_steps'],
        'failed_count': issue['failed_jobs']['count'],
        'failure_rate': issue['failed_jobs']['count'] / 
                       issue['validation_steps']['total_steps']
    }
```

### 2. Dataset Analysis

```python
# Analyze correlation
import pandas as pd

df = pd.DataFrame(data)
df['patch_size'] = df['ground_truth'].apply(
    lambda x: x['lines_inserted'] + x['lines_deleted']
)
df['failure_rate'] = df.apply(
    lambda row: row['failed_jobs']['count'] / 
                row['validation_steps']['total_steps'],
    axis=1
)

# Correlation between patch size and failure rate
correlation = df['patch_size'].corr(df['failure_rate'])
```

### 3. Memory Feature Storage

Later, you can store these as features in memory:

```python
# Save to memory feature file
memory_features = {
    issue['id']: {
        'ground_truth_stats': issue['ground_truth'],
        'validation_info': issue['validation_steps'],
        'failure_info': issue['failed_jobs']
    }
    for issue in data
}
```

## API Rate Limits

GitHub API rate limits:
- **Authenticated**: 5,000 requests/hour
- **Per repo**: ~2 requests per issue (runs + jobs)
- **For 567 issues**: ~1,134 requests (~13 minutes)

**Recommendation**: Run once and save results.

## Comparison: API vs YAML

| Aspect | YAML Parsing | GitHub API |
|--------|-------------|------------|
| Speed | Fast | Slower (~10-15 min) |
| Accuracy | ❌ Medium | ✅ High |
| Matrix jobs | ❌ Template vars | ✅ Expanded |
| Passed/failed | ❌ No | ✅ Yes |
| Offline | ✅ Yes | ❌ No |
| Token required | ❌ No | ✅ Yes |

## Next Steps

Once extracted, use this data for:

1. **Feature engineering** - ML model features
2. **Memory storage** - Store per-issue metadata
3. **Analysis** - Correlations, distributions
4. **Evaluation** - Compare predictions vs ground truth
5. **Filtering** - Select issues by characteristics

## Files Created

- `dataset/dataset_complete_info.json` - Complete info per issue
- One consolidated extraction (no scattered scripts)

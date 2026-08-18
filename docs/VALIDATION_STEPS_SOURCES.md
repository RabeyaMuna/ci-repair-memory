# Validation Steps - Data Sources

## Two Approaches to Get Validation Steps

### Approach 1: Parse Workflow YAML ❌ (Current - Less Accurate)

**Source**: `dataset.workflow` field (YAML content)

**Pros**:
- No API calls needed
- Works offline
- Fast

**Cons**:
- ❌ Matrix expansion can be inaccurate
- ❌ Doesn't account for conditional jobs (`if:` statements)
- ❌ Doesn't reflect what actually ran
- ❌ Template variables not resolved (`${{ matrix.python-version }}`)

**Example issue**:
```python
# Workflow defines:
"Python ${{ matrix.python-version }}"  # Template variable!

# Not the actual:
["Python 3.8", "Python 3.9", "Python 3.10", "Python 3.11"]
```

---

### Approach 2: Fetch from GitHub API ✅ (Recommended - Accurate)

**Source**: GitHub Actions API

**Endpoint**:
```
GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs
```

**Pros**:
- ✅ **Accurate**: Gets exactly what ran
- ✅ **Complete**: All jobs with their statuses
- ✅ **Expanded**: Matrix jobs already expanded
- ✅ **Real data**: Reflects actual execution

**Cons**:
- Requires API calls
- Needs authentication token
- Rate limits apply
- Slower (network requests)

---

## What GitHub API Returns

### For Each Job:
```json
{
  "id": 12345,
  "name": "tests (py39-pyqt63, ubuntu-22.04, 3.9)",  # Actual expanded name
  "status": "completed",
  "conclusion": "success",  // or "failure", "cancelled", etc.
  "started_at": "2024-01-01T00:00:00Z",
  "completed_at": "2024-01-01T00:10:00Z",
  "steps": [
    {
      "name": "Set up job",
      "status": "completed",
      "conclusion": "success"
    },
    {
      "name": "Run tests",
      "status": "completed",
      "conclusion": "failure"  // This step failed!
    }
  ]
}
```

### Complete Information:
1. **All validation steps** (from all jobs)
2. **Which steps passed** (conclusion: "success")
3. **Which steps failed** (conclusion: "failure")
4. **Which steps were skipped**
5. **Execution time** for each step

---

## Recommended Approach

### For Ground Truth Dataset Extraction:

Use **GitHub API** to get accurate validation steps:

```python
import requests

def fetch_validation_steps(repo_owner, repo_name, run_id, token):
    """Fetch actual validation steps from GitHub API."""
    
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }
    
    response = requests.get(url, headers=headers)
    jobs = response.json().get('jobs', [])
    
    all_steps = []
    failed_steps = []
    passed_steps = []
    
    for job in jobs:
        job_name = job['name']
        conclusion = job['conclusion']
        
        all_steps.append(job_name)
        
        if conclusion == 'success':
            passed_steps.append(job_name)
        elif conclusion == 'failure':
            failed_steps.append(job_name)
    
    return {
        'total_steps': len(all_steps),
        'all_steps': all_steps,
        'passed_steps': passed_steps,
        'failed_steps': failed_steps
    }
```

---

## Where to Get Run ID?

From the dataset's `sha_fail` commit, we can:

1. **Query GitHub API**:
   ```
   GET /repos/{owner}/{repo}/commits/{sha}/check-runs
   ```

2. **Get workflow runs**:
   ```
   GET /repos/{owner}/{repo}/actions/runs?head_sha={sha}
   ```

3. **Extract run_id** from the response

---

## Updated Extraction Schema

```python
{
  "id": "269",
  "sha_fail": "f57afa39e8c9...",
  "repo": "qutebrowser/qutebrowser",
  "workflow_name": "CI",
  
  # Ground truth patch
  "ground_truth": {
    "files_changed": 7,
    "lines_inserted": 42,
    "lines_deleted": 80,
    "changed_files": [...]
  },
  
  # Validation steps (from GitHub API - ACCURATE)
  "validation_steps": {
    "total_steps": 45,
    "all_steps": [
      "linters (pylint)",
      "linters (flake8)",
      "tests (py39-pyqt63, ubuntu-22.04, 3.9)",  # Expanded!
      "tests (py310-pyqt65, ubuntu-22.04, 3.10)"  # Expanded!
    ],
    "passed_steps": [
      "linters (pylint)",
      "linters (flake8)"
    ],
    "failed_steps": [
      "tests (py39-pyqt63, ubuntu-22.04, 3.9)",
      "tests (py310-pyqt65, ubuntu-22.04, 3.10)"
    ]
  }
}
```

---

## Implementation Plan

### Option A: Hybrid (Best)
1. Try to find existing run_id from `results/jobs_results_*.jsonl`
2. If found, use GitHub API
3. If not found, fallback to workflow YAML parsing

### Option B: API-only (Most Accurate)
1. Always fetch from GitHub API
2. Cache results to avoid re-fetching
3. Handle rate limits gracefully

### Option C: Use Existing Results Files
Since we already have:
- `results/jobs_ids_diff.jsonl` (pushed jobs with URLs)
- `results/jobs_results_diff.jsonl` (job results)

We can parse the URLs to get run_ids and fetch from API!

---

## Current State

**What we have**:
- ❌ Workflow YAML parsing (inaccurate for matrix jobs)
- ✅ Failed jobs from dataset logs (accurate)
- ✅ Ground truth patch stats (accurate)

**What we need**:
- ✅ Fetch validation steps from GitHub API for accuracy
- ✅ Get passed/failed status for each step
- ✅ Match with dataset logs to validate

---

## Recommendation

**Use GitHub API approach** because:
1. It's what `multilevel_eval.py` already does
2. It gives accurate, expanded job names  
3. It provides pass/fail status
4. The data is already being fetched for evaluation

We should integrate this into the extraction script!

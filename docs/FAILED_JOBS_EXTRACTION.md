# Failed Jobs Extraction Algorithm

## Overview

For each issue in the dataset, we extract information about which jobs/validation steps failed from the original CI run.

## Data Structure

### Source: `dataset/lca_dataset.parquet`

Each row has:
```python
{
  'id': '269',
  'repo_owner': 'qutebrowser',
  'repo_name': 'qutebrowser',
  'workflow_name': 'CI',
  'sha_fail': 'f57afa39e8c9...',
  'sha_success': '25ff649efe50...',
  'logs': [  # Array of failed job logs
    {
      'step_name': 'tests-docker (py, archlinux-webengine-unstable-qt6)',
      'log': '... actual log output ...'
    },
    {
      'step_name': 'tests (py311-pyqt67, ubuntu-22.04, 3.11)',
      'log': '... actual log output ...'
    },
    # ... more failed steps
  ]
}
```

### Output Files

1. **`dataset/failed_jobs_all.json`** - Failed job names (no logs)
   ```json
   [
     {
       "id": "269",
       "failed_jobs": [
         {"step_name": "tests-docker (py, archlinux-webengine-unstable-qt6)"},
         {"step_name": "tests (py311-pyqt67, ubuntu-22.04, 3.11)"}
       ]
     }
   ]
   ```

2. **`dataset/failed_job_logs.json`** - Failed job names WITH logs
   ```json
   [
     {
       "id": 269,  # Note: number, not string
       "failed_jobs": [
         {
           "step_name": "tests-docker (py, archlinux-webengine-unstable-qt6)",
           "log": "... full log output ..."
         }
       ]
     }
   ]
   ```

## Extraction Algorithm

### Step 1: Extract from Dataset

```python
import pandas as pd
import json

def extract_failed_jobs_from_dataset(dataset_path):
    """Extract failed jobs info from dataset."""
    
    # Load dataset
    df = pd.read_parquet(dataset_path)
    
    failed_jobs_all = []
    failed_job_logs = []
    
    for _, row in df.iterrows():
        issue_id = str(row['id'])
        logs = row.get('logs', [])
        
        # Extract failed job names (no logs)
        failed_jobs_all.append({
            'id': issue_id,  # String
            'failed_jobs': [
                {'step_name': log_entry['step_name']}
                for log_entry in logs
                if 'step_name' in log_entry
            ]
        })
        
        # Extract failed jobs with logs
        failed_job_logs.append({
            'id': int(issue_id),  # Number
            'failed_jobs': [
                {
                    'step_name': log_entry['step_name'],
                    'log': log_entry['log']
                }
                for log_entry in logs
                if 'step_name' in log_entry and 'log' in log_entry
            ]
        })
    
    return failed_jobs_all, failed_job_logs
```

### Step 2: Count Statistics

```python
def count_failed_jobs_stats(dataset_path):
    """Count job and step statistics per issue."""
    
    df = pd.read_parquet(dataset_path)
    
    stats = []
    for _, row in df.iterrows():
        issue_id = str(row['id'])
        logs = row.get('logs', [])
        
        # Extract unique job names (step_name might be "job/step" or just "job")
        step_names = [log['step_name'] for log in logs if 'step_name' in log]
        
        # Count total failed steps
        total_failed_steps = len(step_names)
        
        # Extract unique job names (before "/" if present)
        unique_jobs = set()
        for step_name in step_names:
            # If step_name is "tests (py3.9)" or "job_name / step_name"
            job_name = step_name.split('/')[0].strip() if '/' in step_name else step_name
            unique_jobs.add(job_name)
        
        stats.append({
            'id': issue_id,
            'total_failed_steps': total_failed_steps,
            'unique_failed_jobs': len(unique_jobs),
            'failed_job_names': list(unique_jobs),
            'failed_step_names': step_names
        })
    
    return stats
```

## Usage in Evaluation

### Multi-Level Evaluation (`multilevel_eval.py`)

```python
# Load original failed jobs
def load_original_failed_jobs(failed_jobs_file):
    """Load from failed_jobs_all.json or failed_job_logs.json."""
    with open(failed_jobs_file, 'r') as f:
        data = json.load(f)
    
    failed_jobs_map = {}
    for entry in data:
        entry_id = str(entry.get('id'))
        failed_jobs = entry.get('failed_jobs', [])
        
        # Extract job names
        job_names = []
        for job in failed_jobs:
            step_name = job.get('step_name', '')
            if step_name:
                # Extract base job name (before "/" if present)
                job_name = step_name.split('/')[0].strip()
                if job_name and job_name not in job_names:
                    job_names.append(job_name)
        
        if job_names:
            failed_jobs_map[entry_id] = job_names
    
    return failed_jobs_map

# Usage
failed_jobs_map = load_original_failed_jobs('dataset/failed_jobs_all.json')

# For ID 269
print(f"ID 269 failed jobs: {failed_jobs_map.get('269', [])}")
# Output: ['tests-docker (py, archlinux-webengine-unstable-qt6)', 'tests (py311-pyqt67, ubuntu-22.04, 3.11)', ...]
```

## Key Points

1. **Source of Truth**: `dataset/lca_dataset.parquet` → `logs` field
2. **Number of Failed Steps**: `len(row['logs'])` for each issue
3. **Failed Job Names**: Extract `step_name` from each log entry
4. **Job vs Step**: Some entries are job names, some are "job/step" format
5. **ID Format**: 
   - `failed_jobs_all.json`: String IDs
   - `failed_job_logs.json`: Integer IDs

## Example Output

For ID 269:
- **Total failed steps**: 18
- **Failed step names**:
  - `tests-docker (py, archlinux-webengine-unstable-qt6)`
  - `tests-docker (py-qt5, archlinux-webengine-unstable)`
  - `tests (py311-pyqt67, ubuntu-22.04, 3.11)`
  - ... (15 more)

## Verification Script

```bash
# Check extraction
python3 << 'EOF'
import pandas as pd

df = pd.read_parquet('dataset/lca_dataset.parquet')
row = df[df['id'] == '269'].iloc[0]

print(f"ID: {row['id']}")
print(f"Workflow: {row['workflow_name']}")
print(f"Failed steps count: {len(row['logs'])}")
print(f"\nFailed step names:")
for i, log in enumerate(row['logs'], 1):
    print(f"  {i}. {log['step_name']}")
EOF
```

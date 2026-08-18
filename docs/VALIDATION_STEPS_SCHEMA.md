# Validation Steps Schema & Extraction

## Complete Feature Set

For each issue, we should extract:

```python
{
  # Basic identifiers
  "id": "269",
  "sha_fail": "f57afa39e8c9dc4b57c95a021af8a588f6c8c822",
  "repo": "qutebrowser/qutebrowser",
  "workflow_name": "CI",
  
  # Overall validation steps (from workflow YAML)
  "total_validation_steps": 45,  # Total jobs/steps defined in workflow
  "all_validation_steps": [      # List of ALL step names in workflow
    "linters (pylint)",
    "linters (flake8)",
    "tests (py39-pyqt63, ubuntu-22.04, 3.9)",
    "tests (py310-pyqt65, ubuntu-22.04, 3.10)",
    # ... all other steps
  ],
  
  # Failed validation info (from CI logs)
  "failed_jobs_count": 18,       # Number of jobs that failed
  "failed_jobs": [               # List of failed job names
    "tests-docker (py, archlinux-webengine-unstable-qt6)",
    "tests (py311-pyqt67, ubuntu-22.04, 3.11)",
    # ... other failed jobs
  ],
  
  # Derived metrics
  "failure_rate": 0.40,          # failed_jobs_count / total_validation_steps
  "passed_jobs_count": 27,       # total_validation_steps - failed_jobs_count
}
```

## Feature Name Recommendations

### Option 1: Verbose (Clearer)
```python
{
  "id": str,
  "sha_fail": str,
  "total_validation_steps": int,           # Total steps in workflow
  "all_validation_steps": List[str],       # All step names
  "failed_jobs_count": int,                # Number that failed
  "failed_jobs": List[str],                # Failed job names
  "passed_jobs_count": int,                # Number that passed
  "failure_rate": float                    # Percentage failed
}
```

### Option 2: Concise (Shorter)
```python
{
  "id": str,
  "sha_fail": str,
  "total_steps": int,
  "all_steps": List[str],
  "failed_count": int,
  "failed_steps": List[str],
  "passed_count": int,
  "failure_rate": float
}
```

### Option 3: Namespaced (Organized)
```python
{
  "id": str,
  "sha_fail": str,
  "workflow": {
    "total_steps": int,
    "all_steps": List[str]
  },
  "ci_logs": {
    "failed_count": int,
    "failed_steps": List[str]
  },
  "metrics": {
    "passed_count": int,
    "failure_rate": float
  }
}
```

## Recommended: Option 1 (Verbose & Clear)

**Why?**
- Self-documenting field names
- No ambiguity about what each field means
- Easy to understand for new users
- Consistent with existing naming

## Extraction Algorithm

### Step 1: Extract from Dataset

```python
import pandas as pd
import yaml
from typing import Dict, List

def extract_validation_info(dataset_path: str) -> List[Dict]:
    """Extract complete validation information for each issue."""
    
    df = pd.read_parquet(dataset_path)
    validation_data = []
    
    for _, row in df.iterrows():
        issue_id = str(row['id'])
        
        # 1. Extract failed jobs from logs
        logs = row.get('logs', [])
        failed_jobs = [
            log['step_name'] 
            for log in logs 
            if isinstance(log, dict) and 'step_name' in log
        ]
        
        # 2. Extract all validation steps from workflow YAML
        workflow_yaml = row.get('workflow', '')
        all_steps = extract_all_steps_from_workflow(workflow_yaml)
        
        # 3. Calculate metrics
        total_steps = len(all_steps)
        failed_count = len(failed_jobs)
        passed_count = total_steps - failed_count
        failure_rate = failed_count / total_steps if total_steps > 0 else 0
        
        # 4. Compile complete info
        validation_info = {
            # Identifiers
            "id": issue_id,
            "sha_fail": row.get('sha_fail', ''),
            "repo": f"{row.get('repo_owner', '')}/{row.get('repo_name', '')}",
            "workflow_name": row.get('workflow_name', ''),
            
            # Overall validation steps (from workflow)
            "total_validation_steps": total_steps,
            "all_validation_steps": all_steps,
            
            # Failed validation (from CI logs)
            "failed_jobs_count": failed_count,
            "failed_jobs": failed_jobs,
            
            # Derived metrics
            "passed_jobs_count": passed_count,
            "failure_rate": round(failure_rate, 4)
        }
        
        validation_data.append(validation_info)
    
    return validation_data


def extract_all_steps_from_workflow(workflow_yaml: str) -> List[str]:
    """Extract all job/step names from workflow YAML."""
    
    if not workflow_yaml:
        return []
    
    try:
        workflow = yaml.safe_load(workflow_yaml)
        all_steps = []
        
        # Extract job names
        jobs = workflow.get('jobs', {})
        for job_id, job_config in jobs.items():
            # Base job name
            job_name = job_config.get('name', job_id)
            
            # Check for matrix strategy
            strategy = job_config.get('strategy', {})
            matrix = strategy.get('matrix', {})
            
            if matrix and 'include' in matrix:
                # Matrix jobs expand to multiple instances
                for item in matrix['include']:
                    # Generate step name from matrix values
                    matrix_name = generate_matrix_name(job_name, item)
                    all_steps.append(matrix_name)
            elif matrix:
                # Matrix with variables (need to expand)
                expanded = expand_matrix(job_name, matrix)
                all_steps.extend(expanded)
            else:
                # Single job
                all_steps.append(job_name)
        
        return all_steps
    
    except Exception as e:
        print(f"Error parsing workflow: {e}")
        return []


def generate_matrix_name(base_name: str, matrix_item: Dict) -> str:
    """Generate step name from matrix configuration."""
    
    # Common matrix variables
    matrix_vars = []
    for key in ['python-version', 'os', 'version', 'testenv']:
        if key in matrix_item:
            matrix_vars.append(str(matrix_item[key]))
    
    if matrix_vars:
        return f"{base_name} ({', '.join(matrix_vars)})"
    
    return base_name


def expand_matrix(base_name: str, matrix: Dict) -> List[str]:
    """Expand matrix strategy to individual step names."""
    
    # This is complex - need to handle:
    # - matrix.python-version: [3.9, 3.10, 3.11]
    # - matrix.os: [ubuntu-latest, windows-latest]
    # Generate all combinations
    
    import itertools
    
    matrix_keys = [k for k in matrix.keys() if k not in ['include', 'exclude']]
    if not matrix_keys:
        return [base_name]
    
    # Get all combinations
    matrix_values = [matrix[k] for k in matrix_keys]
    combinations = list(itertools.product(*matrix_values))
    
    expanded_names = []
    for combo in combinations:
        combo_str = ', '.join(str(v) for v in combo)
        expanded_names.append(f"{base_name} ({combo_str})")
    
    return expanded_names
```

### Step 2: Save to File

```python
def save_validation_data(validation_data: List[Dict], output_path: str):
    """Save complete validation data."""
    
    import json
    
    with open(output_path, 'w') as f:
        json.dump(validation_data, f, indent=2)
    
    print(f"Saved validation data: {output_path}")
    print(f"  Total issues: {len(validation_data)}")
    
    # Stats
    total_steps = sum(v['total_validation_steps'] for v in validation_data)
    total_failed = sum(v['failed_jobs_count'] for v in validation_data)
    
    print(f"  Total validation steps: {total_steps}")
    print(f"  Total failed: {total_failed}")
    print(f"  Overall failure rate: {total_failed/total_steps*100:.2f}%")
```

## Output File Structure

**Recommended**: `dataset/validation_steps_complete.json`

```json
[
  {
    "id": "269",
    "sha_fail": "f57afa39e8c9dc4b57c95a021af8a588f6c8c822",
    "repo": "qutebrowser/qutebrowser",
    "workflow_name": "CI",
    "total_validation_steps": 45,
    "all_validation_steps": [
      "linters (pylint)",
      "linters (flake8)",
      "linters (mypy-pyqt6)",
      "tests (py39-pyqt63, ubuntu-22.04, 3.9)",
      "tests (py310-pyqt65, ubuntu-22.04, 3.10)"
    ],
    "failed_jobs_count": 18,
    "failed_jobs": [
      "tests-docker (py, archlinux-webengine-unstable-qt6)",
      "tests (py311-pyqt67, ubuntu-22.04, 3.11)"
    ],
    "passed_jobs_count": 27,
    "failure_rate": 0.4
  }
]
```

## Usage in Evaluation

```python
# Load validation data
with open('dataset/validation_steps_complete.json') as f:
    validation_data = json.load(f)

# Get info for specific issue
issue_269 = next(v for v in validation_data if v['id'] == '269')

print(f"Issue {issue_269['id']}:")
print(f"  Total steps: {issue_269['total_validation_steps']}")
print(f"  Failed: {issue_269['failed_jobs_count']}")
print(f"  Passed: {issue_269['passed_jobs_count']}")
print(f"  Failure rate: {issue_269['failure_rate']*100:.1f}%")
```

## Summary

**Recommended Schema:**
- ✅ `id`, `sha_fail`, `repo`, `workflow_name` - Identifiers
- ✅ `total_validation_steps`, `all_validation_steps` - From workflow YAML
- ✅ `failed_jobs_count`, `failed_jobs` - From CI logs
- ✅ `passed_jobs_count`, `failure_rate` - Derived metrics

**Extraction Sources:**
1. **Workflow YAML** (`dataset.workflow`) → All validation steps
2. **CI Logs** (`dataset.logs`) → Failed jobs
3. **Calculate** → Passed jobs, failure rate

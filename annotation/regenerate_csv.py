#!/usr/bin/env python3
"""
Regenerate annotation CSV from local dataset with all necessary fields
"""

import os
import pandas as pd
import json
import numpy as np

def convert_to_json(obj):
    """Convert any object (including numpy arrays) to JSON string"""
    if isinstance(obj, dict):
        return json.dumps(obj)
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, (list, tuple)):
        return json.dumps(obj, default=str)
    elif isinstance(obj, np.ndarray):
        return json.dumps(obj.tolist(), default=str)
    else:
        try:
            return json.dumps(obj, default=str)
        except:
            return '[]'

# Load local dataset
dataset_path = "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet"
print(f"Loading dataset from: {dataset_path}")

if not os.path.exists(dataset_path):
    print(f"Error: Dataset not found at {dataset_path}")
    exit(1)

df = pd.read_parquet(dataset_path)
print(f"Loaded {len(df)} instances from dataset")

# Create annotation CSV with all necessary columns
annotation_data = []

for idx, row in df.iterrows():
    # Basic info
    task_id = row['id']
    repo_name = row['repo_name']
    repo_owner = row['repo_owner']

    # Workflow info
    workflow_name = row.get('workflow_name', '')
    workflow_content = row.get('workflow', '')

    # GitHub link - create compare URL to show diff
    sha_fail = row['sha_fail']
    sha_success = row.get('sha_success', '')

    # Create compare URL to show the diff between success and fail
    # Format: /compare/base...compare shows changes from base to compare
    # Use FULL commit SHA (not shortened) for GitHub to work properly
    if sha_success:
        # Show diff from success to fail (what broke it)
        github_link = f"https://github.com/{repo_owner}/{repo_name}/compare/{sha_success}...{sha_fail}"
    else:
        # Fallback: commit view shows the changes in that commit
        github_link = f"https://github.com/{repo_owner}/{repo_name}/commit/{sha_fail}"

    # Failed jobs (from dataset)
    failed_jobs = row.get('failed_jobs', [])
    failed_jobs_json = convert_to_json(failed_jobs)

    # Failure types from dataset
    failure_types = row.get('failure_types', [])
    failure_subtypes = row.get('failure_subtypes', [])

    # Convert to readable strings
    if isinstance(failure_types, (list, np.ndarray)):
        system_types = ', '.join([str(x) for x in failure_types])
    else:
        system_types = str(failure_types) if failure_types else ''

    if isinstance(failure_subtypes, (list, np.ndarray)):
        system_subtypes = ', '.join([str(x) for x in failure_subtypes])
    else:
        system_subtypes = str(failure_subtypes) if failure_subtypes else ''

    # Failure details as evidence
    failure_details = row.get('failure_details', '')

    # Ground truth diff (the actual fix from the dataset)
    ground_truth_diff = row.get('diff', '')
    if not ground_truth_diff or ground_truth_diff == '':
        ground_truth_diff = 'Ground truth diff not available'

    annotation_data.append({
        'ID': task_id,
        'Repository': f"{repo_owner}/{repo_name}",
        'Workflow_Name': workflow_name,
        'Workflow_Content': workflow_content,
        'GitHub_Compare_Link': github_link,
        'System_Generated_Failure_Types': system_types,
        'System_Generated_Sub_Types': system_subtypes,
        'System_Generated_Evidence': convert_to_json(failure_details),
        'failed_jobs': failed_jobs_json,
        'Ground_Truth_Diff': ground_truth_diff,
        # Empty columns for manual annotation
        'Human_Failure_Types': '',
        'Human_Sub_Types': '',
        'Human_Notes': ''
    })

# Create DataFrame
annotation_df = pd.DataFrame(annotation_data)

# Save for both annotators
for annotator in ['rabeya', 'nakhla']:
    output_csv = f"/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/annotation/{annotator}_annotations.csv"

    # Backup existing if it exists
    if os.path.exists(output_csv):
        backup_csv = output_csv.replace('.csv', '_old_backup.csv')
        pd.read_csv(output_csv).to_csv(backup_csv, index=False)
        print(f"✓ Backed up existing file to: {backup_csv}")

    # Save new file
    annotation_df.to_csv(output_csv, index=False)
    print(f"✓ Generated: {output_csv}")

# Print statistics
def count_jobs(json_str):
    try:
        data = json.loads(json_str)
        return len(data) if isinstance(data, (list, dict)) else 0
    except:
        return 0

job_counts = annotation_df['failed_jobs'].apply(count_jobs)

print(f"\n{'='*60}")
print(f"GENERATION COMPLETE")
print(f"{'='*60}")
print(f"Total instances: {len(annotation_df)}")
print(f"Instances with failed jobs: {(job_counts > 0).sum()}")
print(f"Average jobs per instance: {job_counts.mean():.2f}")
print(f"\nFiles ready for annotation:")
print(f"  - rabeya_annotations.csv")
print(f"  - nakhla_annotations.csv")

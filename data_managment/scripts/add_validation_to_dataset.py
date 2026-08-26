#!/usr/bin/env python3
"""
Add Validation Jobs/Steps Data to Dataset

This script merges the filtered validation data (overall jobs, failed jobs)
into the lca_dataset.parquet file, adding new columns:

NEW COLUMNS:
- overall_jobs: List of all unique jobs and steps for the instance
- total_steps: Count of all unique steps for the instance
- total_jobs: Count of all unique jobs for the instance
- failed_jobs: List of failed jobs
- total_failed_steps: Count of failed steps
- total_failed_jobs: Count of failed jobs


Source: data_managment/results/filtered_validation/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_MANAGEMENT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DATA_MANAGEMENT_DIR.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
VALIDATION_DIR = DATA_MANAGEMENT_DIR / "results" / "filtered_validation"

# Input files
OVERALL_JOBS_FILE = VALIDATION_DIR / "overall_jobs_by_issue.json"
FAILED_JOBS_FILE = VALIDATION_DIR / "failed_jobs_by_issue.json"
INSTANCE_SUMMARY_FILE = VALIDATION_DIR / "instance_validation_summary.json"

# Output
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "lca_dataset_with_validation.parquet"


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON file."""
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return []

    with open(file_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'instances' in data:
        return data['instances']
    return data


def create_validation_lookup(overall_jobs: List[Dict], failed_jobs: List[Dict], summaries: List[Dict]) -> Dict[str, Dict]:
    """Create lookup dictionary by issue ID."""

    lookup = {}

    # Process overall jobs
    for item in overall_jobs:
        issue_id = str(item.get('id', ''))
        lookup[issue_id] = {
            'overall_jobs': item.get('overall_jobs', []),
            'overall_jobs_count': item.get('no_of_jobs', 0),
            'overall_steps_count': item.get('no_of_steps', 0),
        }

    # Process failed jobs
    for item in failed_jobs:
        issue_id = str(item.get('id', ''))
        if issue_id not in lookup:
            lookup[issue_id] = {}

        lookup[issue_id].update({
            'failed_jobs': item.get('failed_jobs', []),
            'failed_jobs_count': item.get('no_of_failed_jobs', 0),
            'failed_steps_count': item.get('no_of_failed_steps', 0),
        })

    # Process summaries (for commits count)
    for item in summaries:
        issue_id = str(item.get('id', ''))
        if issue_id not in lookup:
            lookup[issue_id] = {}

        lookup[issue_id].update({
            'commits_count': item.get('completed_commits_count', 0),
            'total_validation_jobs': item.get('total_validation_jobs', 0),
            'total_validation_steps': item.get('total_validation_steps', 0),
        })

    return lookup


def add_validation_to_dataset(df: pd.DataFrame, validation_lookup: Dict[str, Dict]) -> pd.DataFrame:
    """Add validation columns to dataset."""

    print(f"Adding validation data to {len(df)} instances...")

    # Initialize new columns
    df['validation_overall_jobs'] = None
    df['validation_overall_jobs_count'] = 0
    df['validation_overall_steps_count'] = 0
    df['validation_failed_jobs'] = None
    df['validation_failed_jobs_count'] = 0
    df['validation_failed_steps_count'] = 0
    df['validation_commits_count'] = 0
    df['validation_total_jobs'] = 0
    df['validation_total_steps'] = 0

    matched = 0
    not_matched = 0

    for idx, row in df.iterrows():
        issue_id = str(row['id'])

        if issue_id in validation_lookup:
            val_data = validation_lookup[issue_id]

            df.at[idx, 'validation_overall_jobs'] = val_data.get('overall_jobs', [])
            df.at[idx, 'validation_overall_jobs_count'] = val_data.get('overall_jobs_count', 0)
            df.at[idx, 'validation_overall_steps_count'] = val_data.get('overall_steps_count', 0)
            df.at[idx, 'validation_failed_jobs'] = val_data.get('failed_jobs', [])
            df.at[idx, 'validation_failed_jobs_count'] = val_data.get('failed_jobs_count', 0)
            df.at[idx, 'validation_failed_steps_count'] = val_data.get('failed_steps_count', 0)
            df.at[idx, 'validation_commits_count'] = val_data.get('commits_count', 0)
            df.at[idx, 'validation_total_jobs'] = val_data.get('total_validation_jobs', 0)
            df.at[idx, 'validation_total_steps'] = val_data.get('total_validation_steps', 0)

            matched += 1
        else:
            not_matched += 1

    print(f"✓ Matched {matched} instances with validation data")
    print(f"✗ {not_matched} instances without validation data")

    return df


def main():
    """Main entry point."""

    print("="*70)
    print("ADD VALIDATION DATA TO DATASET")
    print("="*70)

    # Load dataset
    print(f"\n1. Loading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"   Loaded {len(df)} instances")

    # Load validation data
    print(f"\n2. Loading validation data...")
    overall_jobs = load_json_file(OVERALL_JOBS_FILE)
    print(f"   Overall jobs: {len(overall_jobs)} instances")

    failed_jobs = load_json_file(FAILED_JOBS_FILE)
    print(f"   Failed jobs: {len(failed_jobs)} instances")

    summaries = load_json_file(INSTANCE_SUMMARY_FILE)
    print(f"   Summaries: {len(summaries)} instances")

    # Create lookup
    print(f"\n3. Creating validation lookup...")
    validation_lookup = create_validation_lookup(overall_jobs, failed_jobs, summaries)
    print(f"   Lookup created for {len(validation_lookup)} instances")

    # Add to dataset
    print(f"\n4. Adding validation columns to dataset...")
    df_updated = add_validation_to_dataset(df, validation_lookup)

    # Show summary
    print(f"\n5. Summary of new columns:")
    print(f"   validation_overall_jobs_count: {df_updated['validation_overall_jobs_count'].sum():,.0f} total jobs")
    print(f"   validation_overall_steps_count: {df_updated['validation_overall_steps_count'].sum():,.0f} total steps")
    print(f"   validation_failed_jobs_count: {df_updated['validation_failed_jobs_count'].sum():,.0f} failed jobs")
    print(f"   validation_failed_steps_count: {df_updated['validation_failed_steps_count'].sum():,.0f} failed steps")
    print(f"   validation_commits_count: {df_updated['validation_commits_count'].sum():,.0f} total commits")

    # Save
    print(f"\n6. Saving to {OUTPUT_PATH}...")
    df_updated.to_parquet(OUTPUT_PATH, index=False)
    print(f"   ✓ Saved successfully!")

    # Show sample
    print(f"\n7. Sample data (first instance with validation):")
    sample = df_updated[df_updated['validation_commits_count'] > 0].iloc[0]
    print(f"   ID: {sample['id']}")
    print(f"   Repo: {sample['repo_owner']}/{sample['repo_name']}")
    print(f"   Commits: {sample['validation_commits_count']}")
    print(f"   Total Jobs: {sample['validation_total_jobs']}")
    print(f"   Total Steps: {sample['validation_total_steps']}")
    print(f"   Failed Jobs: {sample['validation_failed_jobs_count']}")
    print(f"   Failed Steps: {sample['validation_failed_steps_count']}")

    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"\nNew dataset saved to: {OUTPUT_PATH}")
    print(f"\nNew columns added:")
    print(f"  - validation_overall_jobs (list)")
    print(f"  - validation_overall_jobs_count (int)")
    print(f"  - validation_overall_steps_count (int)")
    print(f"  - validation_failed_jobs (list)")
    print(f"  - validation_failed_jobs_count (int)")
    print(f"  - validation_failed_steps_count (int)")
    print(f"  - validation_commits_count (int)")
    print(f"  - validation_total_jobs (int)")
    print(f"  - validation_total_steps (int)")
    print("\nYou can now use this dataset with complete validation data!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Add no_commits column to dataset showing commit count between sha_fail and sha_success
"""
import json
import os
import sys
from datetime import datetime

def count_commits_in_range(instance):
    """
    Count commits between sha_fail and sha_success.
    Uses the 'commits' field if available, otherwise returns None.
    """
    # Check if commits list exists
    commits = instance.get('commits', [])
    if commits:
        return len(commits)

    # Alternative: check if there's a commit count field
    if 'commit_count' in instance:
        return instance['commit_count']

    # If no commits data, return None
    return None


def add_commit_counts(input_file, output_file):
    """
    Add no_commits field to each instance in the dataset.
    """
    print(f"Loading dataset from: {input_file}")

    with open(input_file, 'r') as f:
        data = json.load(f)

    # Handle different dataset formats
    if isinstance(data, dict) and 'instances' in data:
        instances = data['instances']
        if isinstance(instances, dict):
            instances = list(instances.values())
        is_enriched_format = True
    elif isinstance(data, list):
        instances = data
        is_enriched_format = False
    else:
        print(f"Error: Unsupported dataset format")
        return

    print(f"Processing {len(instances)} instances...")

    # Statistics
    added_count = 0
    already_exists = 0
    no_data = 0

    # Add no_commits field
    for instance in instances:
        if 'no_commits' in instance:
            already_exists += 1
            continue

        commit_count = count_commits_in_range(instance)

        if commit_count is not None:
            instance['no_commits'] = commit_count
            added_count += 1
        else:
            # No commits data available - set to None or -1
            instance['no_commits'] = None
            no_data += 1

    # Update metadata if enriched format
    if is_enriched_format:
        if 'metadata' not in data:
            data['metadata'] = {}

        data['metadata']['no_commits_added'] = True
        data['metadata']['no_commits_added_at'] = datetime.now().isoformat()
        data['instances'] = instances
    else:
        data = instances

    # Save to output file
    print(f"\nSaving to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nDone!")
    print(f"  Added no_commits:        {added_count}")
    print(f"  Already existed:         {already_exists}")
    print(f"  No commit data:          {no_data}")
    print(f"  Total instances:         {len(instances)}")

    # Show sample
    print(f"\nSample instances with no_commits:")
    for i, instance in enumerate(instances[:5], 1):
        id_val = instance.get('id', '?')
        repo = instance.get('repo_name', 'unknown')
        no_commits = instance.get('no_commits', 'N/A')
        sha_fail = (instance.get('sha_fail', '') or '')[:12]
        sha_success = (instance.get('sha_success', '') or '')[:12]
        print(f"  {i}. ID {id_val} ({repo}): {no_commits} commits ({sha_fail}...{sha_success})")


def main():
    """Main function"""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Default paths
    input_file = os.path.join(base_dir, "data_managment/results/enriched_dataset.json")
    output_file = os.path.join(base_dir, "data_managment/results/enriched_dataset_with_counts.json")

    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print(f"\nUsage: python {sys.argv[0]} [input_file] [output_file]")
        print(f"Default input:  data_managment/results/enriched_dataset.json")
        print(f"Default output: data_managment/results/enriched_dataset_with_counts.json")
        sys.exit(1)

    add_commit_counts(input_file, output_file)
    print(f"\n✅ Dataset with no_commits field saved to: {output_file}")


if __name__ == '__main__':
    main()

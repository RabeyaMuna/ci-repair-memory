#!/usr/bin/env python3
"""
Export dataset with no_commits to CSV format
"""
import json
import csv
import os

def export_to_csv(json_file, csv_file):
    """Export dataset to CSV with key fields including no_commits"""

    print(f"Loading: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)

    instances = data.get('instances', data)
    if isinstance(instances, dict):
        instances = list(instances.values())

    print(f"Exporting {len(instances)} instances to CSV...")

    # Define columns to export
    columns = [
        'id',
        'repo_name',
        'repo_owner',
        'sha_fail',
        'sha_success',
        'no_commits',
        'workflow_name',
        'workflow_file',
    ]

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        for inst in instances:
            # Create a row with only the specified columns
            row = {col: inst.get(col, '') for col in columns}
            writer.writerow(row)

    print(f"✅ Exported to: {csv_file}")
    print(f"\nColumns: {', '.join(columns)}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))

    json_file = os.path.join(base_dir, "data_managment/results/enriched_dataset_with_counts.json")
    csv_file = os.path.join(base_dir, "data_managment/results/dataset_commit_counts.csv")

    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}")
        exit(1)

    export_to_csv(json_file, csv_file)

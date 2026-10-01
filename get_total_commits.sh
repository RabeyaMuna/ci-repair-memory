#!/bin/bash
# Quick script to get total commits used in dataset

cd "$(dirname "$0")"

python3 << 'EOF'
import json

# Load dataset
with open('data_managment/results/enriched_dataset_with_counts.json', 'r') as f:
    data = json.load(f)

instances = data.get('instances', data)
if isinstance(instances, dict):
    instances = list(instances.values())

# Calculate statistics
commit_counts = [inst.get('no_commits', 0) for inst in instances]
total_commits = sum(commit_counts)
total_instances = len(instances)
avg_commits = total_commits / total_instances if total_instances > 0 else 0
min_commits = min(commit_counts) if commit_counts else 0
max_commits = max(commit_counts) if commit_counts else 0
median_commits = sorted(commit_counts)[len(commit_counts)//2] if commit_counts else 0

print("=" * 60)
print("TOTAL COMMITS USED IN DATASET")
print("=" * 60)
print(f"Total commits (sum of all ranges): {total_commits:,}")
print(f"Total instances:                   {total_instances:,}")
print(f"Average commits per instance:      {avg_commits:.2f}")
print(f"Median commits:                    {median_commits}")
print(f"Min commits:                       {min_commits}")
print(f"Max commits:                       {max_commits}")
print("=" * 60)
EOF

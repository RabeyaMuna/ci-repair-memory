#!/usr/bin/env python3
"""
Detailed commit statistics calculator
"""
import json
import sys
from collections import Counter

def load_dataset(file_path):
    """Load dataset from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    instances = data.get('instances', data)
    if isinstance(instances, dict):
        instances = list(instances.values())

    return instances

def calculate_statistics(instances):
    """Calculate comprehensive commit statistics"""

    # Extract commit counts
    commit_counts = [inst.get('no_commits', 0) for inst in instances]

    # Basic statistics
    total_commits = sum(commit_counts)
    total_instances = len(commit_counts)
    avg_commits = total_commits / total_instances if total_instances > 0 else 0
    min_commits = min(commit_counts) if commit_counts else 0
    max_commits = max(commit_counts) if commit_counts else 0
    median_commits = sorted(commit_counts)[len(commit_counts)//2] if commit_counts else 0

    # Distribution
    counter = Counter(commit_counts)

    # Percentiles
    sorted_counts = sorted(commit_counts)
    p25 = sorted_counts[len(sorted_counts)//4] if sorted_counts else 0
    p75 = sorted_counts[3*len(sorted_counts)//4] if sorted_counts else 0
    p90 = sorted_counts[9*len(sorted_counts)//10] if sorted_counts else 0

    return {
        'total_commits': total_commits,
        'total_instances': total_instances,
        'avg_commits': avg_commits,
        'min_commits': min_commits,
        'max_commits': max_commits,
        'median_commits': median_commits,
        'p25': p25,
        'p75': p75,
        'p90': p90,
        'distribution': counter
    }

def print_statistics(stats):
    """Print formatted statistics"""

    print("=" * 70)
    print(" COMMIT STATISTICS SUMMARY")
    print("=" * 70)

    print("\n📊 OVERALL STATISTICS:")
    print("-" * 70)
    print(f"  Total commits (all ranges):        {stats['total_commits']:>10,}")
    print(f"  Total instances:                   {stats['total_instances']:>10,}")
    print(f"  Average commits per instance:      {stats['avg_commits']:>10.2f}")

    print("\n📈 DISTRIBUTION METRICS:")
    print("-" * 70)
    print(f"  Minimum commits:                   {stats['min_commits']:>10}")
    print(f"  25th percentile:                   {stats['p25']:>10}")
    print(f"  Median (50th percentile):          {stats['median_commits']:>10}")
    print(f"  75th percentile:                   {stats['p75']:>10}")
    print(f"  90th percentile:                   {stats['p90']:>10}")
    print(f"  Maximum commits:                   {stats['max_commits']:>10}")

    print("\n📋 TOP 10 MOST COMMON COMMIT COUNTS:")
    print("-" * 70)
    print(f"  {'Commits':<10} {'Instances':<12} {'Percentage':<12} {'Bar'}")
    print("-" * 70)

    total = stats['total_instances']
    for count, instances in sorted(stats['distribution'].items(),
                                   key=lambda x: x[1], reverse=True)[:10]:
        pct = instances / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {count:<10} {instances:<12} {pct:>6.1f}%       {bar}")

    print("\n" + "=" * 70)

def main():
    """Main function"""
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir,
                            "data_managment/results/enriched_dataset_with_counts.json")

    # Allow command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Loading dataset from: {file_path}\n")
    instances = load_dataset(file_path)
    stats = calculate_statistics(instances)
    print_statistics(stats)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Analyze which CI failure types were successfully vs unsuccessfully repaired.

ALGORITHM:
1. Load failure type classifications (from results/failure_classifications.json)
2. Load repair results (from baselines/results/{model}/jobs_results_diff.jsonl)
3. For each failure type:
   - Count total instances with that failure type
   - Count successful repairs (conclusion == 'success')
   - Count unsuccessful repairs (conclusion != 'success')
   - Calculate success rate
4. Sort by success rate to identify which failure types are hardest to fix
5. Output statistics and save detailed results

This helps identify:
- Which types of CI failures the model struggles with most
- Patterns in unsuccessful repairs
- Areas for model improvement
"""

import json
import os
from collections import defaultdict


def load_jsonl(filepath):
    """Load JSONL file into list of dictionaries."""
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def load_failure_classifications(filepath):
    """
    Load failure type classifications.

    Returns: dict mapping instance_id -> list of failure types
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    id_to_failures = {}

    # Handle list format: [{issue_id, failure_type}, ...]
    if 'classifications' in data and isinstance(data['classifications'], list):
        for item in data['classifications']:
            instance_id = str(item.get('issue_id', item.get('id', '')))
            failure_types = item.get('failure_type', item.get('failure_types', []))

            # Convert string to list
            if isinstance(failure_types, str):
                failure_types = [failure_types]

            if failure_types and instance_id:
                id_to_failures[instance_id] = failure_types

    return id_to_failures


def analyze_failure_type_success(results_dir, failure_classifications_path):
    """
    Analyze success rates by original failure type.

    Returns: dict of failure_type -> {total, successful, unsuccessful, success_rate, ...}
    """

    results_file = os.path.join(results_dir, "jobs_results_diff.jsonl")

    # Load job results
    all_jobs = load_jsonl(results_file)
    print(f"Loaded {len(all_jobs)} job results")

    # Load failure classifications
    id_to_failures = load_failure_classifications(failure_classifications_path)
    print(f"Loaded failure types for {len(id_to_failures)} instances")

    # Initialize stats tracking
    failure_type_stats = defaultdict(lambda: {
        'total': 0,
        'successful': 0,
        'unsuccessful': 0,
        'success_rate': 0.0,
        'successful_ids': [],
        'unsuccessful_ids': []
    })

    # Track instances not found in classifications
    missing_classifications = []

    # Analyze each job
    for job in all_jobs:
        job_id = str(job.get('id', ''))
        conclusion = job.get('conclusion', '')

        # Get failure types for this instance
        failure_types = id_to_failures.get(job_id, [])

        if not failure_types:
            missing_classifications.append(job_id)
            failure_types = ['Unknown']

        # Update stats for each failure type
        # (instances can have multiple failure types)
        for ft in failure_types:
            failure_type_stats[ft]['total'] += 1

            if conclusion == 'success':
                failure_type_stats[ft]['successful'] += 1
                failure_type_stats[ft]['successful_ids'].append(job_id)
            else:
                failure_type_stats[ft]['unsuccessful'] += 1
                failure_type_stats[ft]['unsuccessful_ids'].append(job_id)

    # Calculate success rates
    for ft, stats in failure_type_stats.items():
        if stats['total'] > 0:
            stats['success_rate'] = (stats['successful'] / stats['total']) * 100

    if missing_classifications:
        print(f"\nWARNING: {len(missing_classifications)} instances missing classifications")
        print(f"  Sample IDs: {missing_classifications[:5]}")

    return dict(failure_type_stats)


def print_results(stats):
    """Print formatted results table."""

    # Sort by success rate (ascending) to show hardest failure types first
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['success_rate'])

    print("\nFAILURE TYPE SUCCESS RATES (sorted by difficulty):")
    print("=" * 90)
    print(f"{'Failure Type':<40} {'Total':>6} {'Success':>7} {'Failed':>7} {'Success Rate':>12}")
    print("-" * 90)

    for ft, data in sorted_stats:
        print(f"{ft:<40} {data['total']:>6} {data['successful']:>7} "
              f"{data['unsuccessful']:>7} {data['success_rate']:>11.1f}%")

    # Overall statistics (note: sum may be > total due to multiple failure types per instance)
    total_instances = sum(data['total'] for data in stats.values())
    total_successful = sum(data['successful'] for data in stats.values())
    overall_success_rate = (total_successful / total_instances * 100) if total_instances > 0 else 0

    print("-" * 90)
    print(f"{'TOTAL (with overlap)':<40} {total_instances:>6} {total_successful:>7} "
          f"{total_instances - total_successful:>7} {overall_success_rate:>11.1f}%")


def identify_hardest_types(stats, threshold=20, min_instances=5):
    """Identify failure types that are hardest to fix."""
    hardest_types = [
        (ft, data) for ft, data in stats.items()
        if data['success_rate'] < threshold and data['total'] >= min_instances
    ]

    if hardest_types:
        print(f"\nHARDEST FAILURE TYPES (Success Rate < {threshold}%, n >= {min_instances}):")
        print("-" * 90)
        for ft, data in sorted(hardest_types, key=lambda x: x[1]['success_rate']):
            print(f"  {ft:<40} {data['success_rate']:>5.1f}% ({data['successful']}/{data['total']})")

    return hardest_types


def save_results(stats, output_file):
    """Save detailed results to JSON file."""

    # Convert to serializable format (limit ID lists to save space)
    output_data = {
        ft: {
            'total': data['total'],
            'successful': data['successful'],
            'unsuccessful': data['unsuccessful'],
            'success_rate': data['success_rate'],
            'successful_ids_sample': data['successful_ids'][:20],
            'unsuccessful_ids_sample': data['unsuccessful_ids'][:20]
        }
        for ft, data in stats.items()
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed analysis saved to: {output_file}")


def main():
    print("=" * 90)
    print("ANALYZING REPAIR SUCCESS RATES BY ORIGINAL FAILURE TYPE")
    print("=" * 90)
    print()

    results_dir = "baselines/results/gpt-5-mini_llm"
    classifications_file = "results/failure_classifications.json"

    # Verify files exist
    if not os.path.exists(classifications_file):
        print(f"ERROR: Classification file not found: {classifications_file}")
        return

    if not os.path.exists(os.path.join(results_dir, "jobs_results_diff.jsonl")):
        print(f"ERROR: Results file not found in {results_dir}")
        return

    # Run analysis
    stats = analyze_failure_type_success(results_dir, classifications_file)

    # Print results
    print_results(stats)

    # Identify hardest types
    hardest = identify_hardest_types(stats, threshold=20, min_instances=5)

    # Save detailed results
    output_file = os.path.join(results_dir, "failure_type_success_analysis.json")
    save_results(stats, output_file)

    print("\n" + "=" * 90)
    print("ALGORITHM SUMMARY")
    print("=" * 90)
    print("""
This analysis:
1. Loaded failure classifications from results/failure_classifications.json
2. Loaded repair results from jobs_results_diff.jsonl
3. For each failure type:
   - Counted total instances
   - Counted successful vs unsuccessful repairs
   - Calculated success rate
4. Sorted results by success rate to identify hardest types
5. Identified failure types with <20% success rate (n >= 5)

Key findings can help identify:
- Which CI failure types are hardest for the model to fix
- Potential areas for model improvement
- Patterns in unsuccessful repairs
    """)


if __name__ == "__main__":
    main()

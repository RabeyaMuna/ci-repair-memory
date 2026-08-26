#!/usr/bin/env python3
"""Analyze overall jobs and failed jobs summaries from enriched dataset."""

import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent.parent / "results"


def analyze_overall_vs_failed():
    """Compare overall jobs vs failed jobs."""
    print("=" * 80)
    print("Analysis: Overall Jobs vs Failed Jobs")
    print("=" * 80)

    with open(DATA_DIR / "enriched_dataset.json", "r") as f:
        data = json.load(f)

    # Aggregate statistics
    stats = {
        'total_instances': len(data['instances']),
        'instances_with_overall_jobs': 0,
        'instances_with_failed_jobs': 0,
        'total_overall_jobs': 0,
        'total_overall_steps': 0,
        'total_failed_jobs': 0,
        'total_failed_steps': 0,
        'total_validation_jobs': 0,
        'total_validation_steps': 0,
    }

    # Collect per-instance data
    instance_data = []

    for instance in data['instances']:
        instance_info = {
            'id': instance['id'],
            'repo': instance['repo'],
            'workflow': instance.get('workflow_name', ''),
        }

        # Overall jobs summary (unique jobs aggregated at instance level)
        if 'overall_jobs_summary' in instance:
            overall = instance['overall_jobs_summary']
            instance_info['has_overall_jobs'] = overall.get('has_overall_jobs', False)
            instance_info['no_of_jobs'] = overall.get('no_of_jobs', 0)
            instance_info['no_of_steps'] = overall.get('no_of_steps', 0)

            if overall.get('has_overall_jobs'):
                stats['instances_with_overall_jobs'] += 1
            stats['total_overall_jobs'] += instance_info['no_of_jobs']
            stats['total_overall_steps'] += instance_info['no_of_steps']

        # Failed jobs summary (unique failed jobs aggregated at instance level)
        if 'failed_jobs_summary' in instance:
            failed = instance['failed_jobs_summary']
            instance_info['has_failed_jobs'] = failed.get('has_failed_jobs', False)
            instance_info['no_of_failed_jobs'] = failed.get('no_of_failed_jobs', 0)
            instance_info['no_of_failed_steps'] = failed.get('no_of_failed_steps', 0)

            if failed.get('has_failed_jobs'):
                stats['instances_with_failed_jobs'] += 1
            stats['total_failed_jobs'] += instance_info['no_of_failed_jobs']
            stats['total_failed_steps'] += instance_info['no_of_failed_steps']

        # Validation summary (from validation data - includes all commits)
        if 'instance_validation_summary' in instance:
            val_summary = instance['instance_validation_summary']
            instance_info['validation_jobs'] = val_summary.get('total_validation_jobs', 0)
            instance_info['validation_steps'] = val_summary.get('total_validation_steps', 0)
            instance_info['validation_failed_jobs'] = val_summary.get('total_failed_jobs', 0)
            instance_info['validation_failed_steps'] = val_summary.get('total_failed_steps', 0)

            stats['total_validation_jobs'] += instance_info['validation_jobs']
            stats['total_validation_steps'] += instance_info['validation_steps']

        # Calculate failure rate
        if instance_info.get('no_of_jobs', 0) > 0:
            instance_info['failure_rate'] = instance_info.get('no_of_failed_jobs', 0) / instance_info['no_of_jobs']
        else:
            instance_info['failure_rate'] = 0.0

        instance_data.append(instance_info)

    # Print summary
    print("\n--- Summary Statistics ---")
    print(f"Total instances: {stats['total_instances']}")
    print(f"Instances with overall jobs: {stats['instances_with_overall_jobs']}")
    print(f"Instances with failed jobs: {stats['instances_with_failed_jobs']}")

    print("\n--- Overall Jobs (Unique per instance) ---")
    print(f"Total overall jobs: {stats['total_overall_jobs']}")
    print(f"Total overall steps: {stats['total_overall_steps']}")
    print(f"Avg jobs per instance: {stats['total_overall_jobs'] / stats['total_instances']:.1f}")
    print(f"Avg steps per instance: {stats['total_overall_steps'] / stats['total_instances']:.1f}")

    print("\n--- Failed Jobs (Unique per instance) ---")
    print(f"Total failed jobs: {stats['total_failed_jobs']}")
    print(f"Total failed steps: {stats['total_failed_steps']}")
    print(f"Failure rate (jobs): {stats['total_failed_jobs'] / stats['total_overall_jobs'] * 100:.1f}%")
    print(f"Failure rate (steps): {stats['total_failed_steps'] / stats['total_overall_steps'] * 100:.1f}%")

    print("\n--- Validation Jobs (All commits, all runs) ---")
    print(f"Total validation jobs: {stats['total_validation_jobs']}")
    print(f"Total validation steps: {stats['total_validation_steps']}")
    print(f"Avg validation jobs per instance: {stats['total_validation_jobs'] / stats['total_instances']:.1f}")

    print("\n--- Explanation ---")
    print("• Overall Jobs: Unique job names aggregated at instance level")
    print("  (e.g., if 'test' job appears in 3 commits, it counts as 1)")
    print("• Validation Jobs: All job executions across all commits")
    print("  (e.g., if 'test' job appears in 3 commits, it counts as 3)")
    print("• Failed Jobs: Unique job names that failed at least once")

    return instance_data, stats


def top_instances_by_failure_rate(instance_data):
    """Show instances with highest failure rates."""
    print("\n" + "=" * 80)
    print("Top 20 Instances by Failure Rate (Overall Jobs)")
    print("=" * 80)

    # Filter and sort by failure rate
    instances_with_jobs = [
        inst for inst in instance_data
        if inst.get('no_of_jobs', 0) > 0
    ]
    instances_with_jobs.sort(key=lambda x: x['failure_rate'], reverse=True)

    print(f"\n{'ID':<6} {'Repo':<35} {'Workflow':<25} {'Failed':<8} {'Total':<8} {'Rate':<8}")
    print("-" * 100)

    for inst in instances_with_jobs[:20]:
        print(
            f"{inst['id']:<6} "
            f"{inst['repo'][:34]:<35} "
            f"{inst['workflow'][:24]:<25} "
            f"{inst.get('no_of_failed_jobs', 0):<8} "
            f"{inst.get('no_of_jobs', 0):<8} "
            f"{inst['failure_rate']*100:>6.1f}%"
        )


def failed_job_names_analysis(instance_data):
    """Analyze which job names fail most often."""
    print("\n" + "=" * 80)
    print("Most Common Failed Job Names")
    print("=" * 80)

    with open(DATA_DIR / "enriched_dataset.json", "r") as f:
        data = json.load(f)

    failed_job_names = []

    for instance in data['instances']:
        if 'failed_jobs_summary' in instance:
            failed_jobs = instance['failed_jobs_summary'].get('failed_jobs', [])
            for job in failed_jobs:
                job_name = job.get('job_name', '')
                if job_name:
                    failed_job_names.append(job_name)

    job_name_counts = Counter(failed_job_names)

    print(f"\nTotal unique failed job instances: {len(failed_job_names)}")
    print(f"Unique job names: {len(job_name_counts)}")
    print("\nTop 20 failed job names:")
    print(f"{'Count':<8} Job Name")
    print("-" * 80)

    for job_name, count in job_name_counts.most_common(20):
        print(f"{count:<8} {job_name}")


def failed_step_names_analysis():
    """Analyze which step names fail most often."""
    print("\n" + "=" * 80)
    print("Most Common Failed Step Names")
    print("=" * 80)

    with open(DATA_DIR / "enriched_dataset.json", "r") as f:
        data = json.load(f)

    failed_step_names = []

    for instance in data['instances']:
        if 'failed_jobs_summary' in instance:
            failed_jobs = instance['failed_jobs_summary'].get('failed_jobs', [])
            for job in failed_jobs:
                step_names = job.get('step_names', [])
                failed_step_names.extend(step_names)

    step_name_counts = Counter(failed_step_names)

    print(f"\nTotal failed step instances: {len(failed_step_names)}")
    print(f"Unique step names: {len(step_name_counts)}")
    print("\nTop 20 failed step names:")
    print(f"{'Count':<8} Step Name")
    print("-" * 80)

    for step_name, count in step_name_counts.most_common(20):
        print(f"{count:<8} {step_name}")


def comparison_table():
    """Show comparison between overall and validation counts."""
    print("\n" + "=" * 80)
    print("Comparison: Overall Jobs (Unique) vs Validation Jobs (All Executions)")
    print("=" * 80)

    with open(DATA_DIR / "enriched_dataset.json", "r") as f:
        data = json.load(f)

    comparison_data = []

    for instance in data['instances']:
        row = {
            'id': instance['id'],
            'repo': instance['repo'],
            'overall_jobs': instance.get('overall_jobs_summary', {}).get('no_of_jobs', 0),
            'overall_steps': instance.get('overall_jobs_summary', {}).get('no_of_steps', 0),
            'validation_jobs': instance.get('instance_validation_summary', {}).get('total_validation_jobs', 0),
            'validation_steps': instance.get('instance_validation_summary', {}).get('total_validation_steps', 0),
        }

        if row['overall_jobs'] > 0:
            row['multiplier_jobs'] = row['validation_jobs'] / row['overall_jobs']
            row['multiplier_steps'] = row['validation_steps'] / row['overall_steps'] if row['overall_steps'] > 0 else 0
        else:
            row['multiplier_jobs'] = 0
            row['multiplier_steps'] = 0

        comparison_data.append(row)

    # Sort by multiplier (highest first)
    comparison_data.sort(key=lambda x: x['multiplier_jobs'], reverse=True)

    print(f"\n{'ID':<6} {'Repo':<30} {'Overall':<12} {'Validation':<12} {'Multiplier':<10}")
    print(f"{'':6} {'':30} {'Jobs/Steps':<12} {'Jobs/Steps':<12} {'(Jobs)':<10}")
    print("-" * 80)

    for row in comparison_data[:15]:
        print(
            f"{row['id']:<6} "
            f"{row['repo'][:29]:<30} "
            f"{row['overall_jobs']:>4}/{row['overall_steps']:<6} "
            f"{row['validation_jobs']:>5}/{row['validation_steps']:<6} "
            f"{row['multiplier_jobs']:>8.1f}x"
        )

    print("\nNote: Multiplier shows how many times each unique job was executed on average")


def main():
    """Run all analyses."""
    instance_data, stats = analyze_overall_vs_failed()
    top_instances_by_failure_rate(instance_data)
    failed_job_names_analysis(instance_data)
    failed_step_names_analysis()
    comparison_table()

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

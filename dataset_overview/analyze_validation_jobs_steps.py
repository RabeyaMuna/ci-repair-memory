#!/usr/bin/env python3
"""
Analyze Validation Jobs and Steps from CI Metadata

This script analyzes the complete validation data from sha_fail to sha_success
for each instance, showing:
1. Total validation jobs and steps per instance
2. Failed jobs and steps per instance
3. Complete commit-by-commit breakdown
4. Aggregated statistics for paper

Based on filtered_validation data extracted from all_instances_metadata.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = PROJECT_ROOT / "data_managment" / "results" / "filtered_validation"
DATASET_PATH = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
OUTPUT_DIR = PROJECT_ROOT / "dataset_overview"


def load_validation_data() -> Dict[str, Any]:
    """Load validation summary data."""
    summary_file = VALIDATION_DIR / "instance_validation_summary.json"

    if not summary_file.exists():
        raise FileNotFoundError(
            f"Validation summary not found: {summary_file}\n"
            f"Run: python data_managment/scripts/extract_validation_jobs_and_steps.py"
        )

    with open(summary_file, 'r') as f:
        data = json.load(f)

    return data


def analyze_validation_statistics(validation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze validation jobs and steps statistics."""

    instances = validation_data.get("instances", [])

    if not instances:
        print("Warning: No instances found in validation data")
        return {}

    print(f"Analyzing {len(instances)} instances with validation data...")

    # Collect per-instance metrics
    total_jobs_list = []
    total_steps_list = []
    failed_jobs_list = []
    failed_steps_list = []
    commits_per_instance = []

    # Track unique job and step names
    all_job_names = []
    all_step_names = []
    all_failed_job_names = []
    all_failed_step_names = []

    # Per-instance details
    instance_details = []

    for instance in instances:
        instance_id = instance.get("id", "unknown")
        repo = instance.get("repo", "unknown")

        # Counts
        total_jobs = instance.get("total_validation_jobs", 0)
        total_steps = instance.get("total_validation_steps", 0)
        failed_jobs = instance.get("overall_failed_jobs_count", 0)
        failed_steps = instance.get("overall_failed_steps_count", 0)
        commits_count = instance.get("completed_commits_count", 0)

        total_jobs_list.append(total_jobs)
        total_steps_list.append(total_steps)
        failed_jobs_list.append(failed_jobs)
        failed_steps_list.append(failed_steps)
        commits_per_instance.append(commits_count)

        # Names
        job_names = instance.get("unique_validation_job_names", [])
        step_names = instance.get("unique_validation_step_names", [])
        failed_job_names = instance.get("unique_failed_job_names", [])
        failed_step_names = instance.get("unique_failed_step_names", [])

        all_job_names.extend(job_names)
        all_step_names.extend(step_names)
        all_failed_job_names.extend(failed_job_names)
        all_failed_step_names.extend(failed_step_names)

        # Store details
        instance_details.append({
            "id": instance_id,
            "repo": repo,
            "total_jobs": total_jobs,
            "total_steps": total_steps,
            "failed_jobs": failed_jobs,
            "failed_steps": failed_steps,
            "commits": commits_count,
        })

    # Compute aggregates
    total_instances = len(instances)
    total_validation_jobs = sum(total_jobs_list)
    total_validation_steps = sum(total_steps_list)
    total_failed_jobs = sum(failed_jobs_list)
    total_failed_steps = sum(failed_steps_list)
    total_commits = sum(commits_per_instance)

    # Compute distributions
    jobs_per_instance_dist = pd.Series(total_jobs_list).describe()
    steps_per_instance_dist = pd.Series(total_steps_list).describe()
    failed_jobs_per_instance_dist = pd.Series(failed_jobs_list).describe()
    failed_steps_per_instance_dist = pd.Series(failed_steps_list).describe()

    # Count unique names
    unique_job_names = len(set(all_job_names))
    unique_step_names = len(set(all_step_names))
    unique_failed_job_names = len(set(all_failed_job_names))
    unique_failed_step_names = len(set(all_failed_step_names))

    # Top job names
    job_name_counts = Counter(all_job_names)
    step_name_counts = Counter(all_step_names)
    failed_job_name_counts = Counter(all_failed_job_names)
    failed_step_name_counts = Counter(all_failed_step_names)

    # Compute percentages
    failed_jobs_percentage = (total_failed_jobs / total_validation_jobs * 100) if total_validation_jobs > 0 else 0
    failed_steps_percentage = (total_failed_steps / total_validation_steps * 100) if total_validation_steps > 0 else 0

    return {
        "summary": {
            "total_instances_with_validation": total_instances,
            "total_validation_jobs": total_validation_jobs,
            "total_validation_steps": total_validation_steps,
            "total_failed_jobs": total_failed_jobs,
            "total_failed_steps": total_failed_steps,
            "total_commits_analyzed": total_commits,
            "avg_commits_per_instance": round(total_commits / total_instances, 2),
            "avg_jobs_per_instance": round(total_validation_jobs / total_instances, 2),
            "avg_steps_per_instance": round(total_validation_steps / total_instances, 2),
            "avg_failed_jobs_per_instance": round(total_failed_jobs / total_instances, 2),
            "avg_failed_steps_per_instance": round(total_failed_steps / total_instances, 2),
            "avg_steps_per_job": round(total_validation_steps / total_validation_jobs, 2) if total_validation_jobs > 0 else 0,
            "failed_jobs_percentage": round(failed_jobs_percentage, 2),
            "failed_steps_percentage": round(failed_steps_percentage, 2),
        },

        "distributions": {
            "jobs_per_instance": {
                "min": int(jobs_per_instance_dist["min"]),
                "25th": int(jobs_per_instance_dist["25%"]),
                "median": int(jobs_per_instance_dist["50%"]),
                "75th": int(jobs_per_instance_dist["75%"]),
                "max": int(jobs_per_instance_dist["max"]),
                "mean": round(jobs_per_instance_dist["mean"], 2),
                "std": round(jobs_per_instance_dist["std"], 2),
            },
            "steps_per_instance": {
                "min": int(steps_per_instance_dist["min"]),
                "25th": int(steps_per_instance_dist["25%"]),
                "median": int(steps_per_instance_dist["50%"]),
                "75th": int(steps_per_instance_dist["75%"]),
                "max": int(steps_per_instance_dist["max"]),
                "mean": round(steps_per_instance_dist["mean"], 2),
                "std": round(steps_per_instance_dist["std"], 2),
            },
            "failed_jobs_per_instance": {
                "min": int(failed_jobs_per_instance_dist["min"]),
                "25th": int(failed_jobs_per_instance_dist["25%"]),
                "median": int(failed_jobs_per_instance_dist["50%"]),
                "75th": int(failed_jobs_per_instance_dist["75%"]),
                "max": int(failed_jobs_per_instance_dist["max"]),
                "mean": round(failed_jobs_per_instance_dist["mean"], 2),
                "std": round(failed_jobs_per_instance_dist["std"], 2),
            },
            "failed_steps_per_instance": {
                "min": int(failed_steps_per_instance_dist["min"]),
                "25th": int(failed_steps_per_instance_dist["25%"]),
                "median": int(failed_steps_per_instance_dist["50%"]),
                "75th": int(failed_steps_per_instance_dist["75%"]),
                "max": int(failed_steps_per_instance_dist["max"]),
                "mean": round(failed_steps_per_instance_dist["mean"], 2),
                "std": round(failed_steps_per_instance_dist["std"], 2),
            },
        },

        "unique_names": {
            "unique_job_names": unique_job_names,
            "unique_step_names": unique_step_names,
            "unique_failed_job_names": unique_failed_job_names,
            "unique_failed_step_names": unique_failed_step_names,
        },

        "top_job_names": dict(job_name_counts.most_common(10)),
        "top_step_names": dict(step_name_counts.most_common(10)),
        "top_failed_job_names": dict(failed_job_name_counts.most_common(10)),
        "top_failed_step_names": dict(failed_step_name_counts.most_common(10)),

        "per_instance_details": instance_details[:20],  # Top 20 for inspection
    }


def print_summary(stats: Dict[str, Any]):
    """Print formatted summary."""

    summary = stats["summary"]
    dists = stats["distributions"]

    print("\n" + "="*70)
    print("VALIDATION JOBS AND STEPS ANALYSIS")
    print("="*70)

    print(f"\n📊 OVERALL STATISTICS (from fail→success commits)")
    print("-" * 70)
    print(f"Instances with validation data: {summary['total_instances_with_validation']}")
    print(f"Total commits analyzed: {summary['total_commits_analyzed']}")
    print(f"Avg commits per instance: {summary['avg_commits_per_instance']}")

    print(f"\n⚙️  VALIDATION JOBS AND STEPS")
    print("-" * 70)
    print(f"Total Validation Jobs: {summary['total_validation_jobs']:,}")
    print(f"Total Validation Steps: {summary['total_validation_steps']:,}")
    print(f"Avg Jobs per Instance: {summary['avg_jobs_per_instance']}")
    print(f"Avg Steps per Instance: {summary['avg_steps_per_instance']}")
    print(f"Avg Steps per Job: {summary['avg_steps_per_job']}")

    print(f"\n❌ FAILED JOBS AND STEPS")
    print("-" * 70)
    print(f"Total Failed Jobs: {summary['total_failed_jobs']:,}")
    print(f"Total Failed Steps: {summary['total_failed_steps']:,}")
    print(f"Avg Failed Jobs per Instance: {summary['avg_failed_jobs_per_instance']}")
    print(f"Avg Failed Steps per Instance: {summary['avg_failed_steps_per_instance']}")
    print(f"Failed Jobs Percentage: {summary['failed_jobs_percentage']}%")
    print(f"Failed Steps Percentage: {summary['failed_steps_percentage']}%")

    print(f"\n📈 DISTRIBUTIONS")
    print("-" * 70)
    print(f"Jobs per Instance: median={dists['jobs_per_instance']['median']}, "
          f"mean={dists['jobs_per_instance']['mean']}, "
          f"max={dists['jobs_per_instance']['max']}")
    print(f"Steps per Instance: median={dists['steps_per_instance']['median']}, "
          f"mean={dists['steps_per_instance']['mean']}, "
          f"max={dists['steps_per_instance']['max']}")
    print(f"Failed Jobs per Instance: median={dists['failed_jobs_per_instance']['median']}, "
          f"mean={dists['failed_jobs_per_instance']['mean']}, "
          f"max={dists['failed_jobs_per_instance']['max']}")
    print(f"Failed Steps per Instance: median={dists['failed_steps_per_instance']['median']}, "
          f"mean={dists['failed_steps_per_instance']['mean']}, "
          f"max={dists['failed_steps_per_instance']['max']}")

    print(f"\n🏷️  UNIQUE NAMES")
    print("-" * 70)
    print(f"Unique Job Names: {stats['unique_names']['unique_job_names']}")
    print(f"Unique Step Names: {stats['unique_names']['unique_step_names']}")
    print(f"Unique Failed Job Names: {stats['unique_names']['unique_failed_job_names']}")
    print(f"Unique Failed Step Names: {stats['unique_names']['unique_failed_step_names']}")

    print(f"\n🔝 TOP 5 FAILED JOB NAMES")
    print("-" * 70)
    for i, (name, count) in enumerate(list(stats['top_failed_job_names'].items())[:5], 1):
        print(f"  {i}. {name}: {count} failures")

    print(f"\n🔝 TOP 5 FAILED STEP NAMES")
    print("-" * 70)
    for i, (name, count) in enumerate(list(stats['top_failed_step_names'].items())[:5], 1):
        print(f"  {i}. {name}: {count} failures")

    print("\n" + "="*70)


def main():
    """Main entry point."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading validation data...")
    validation_data = load_validation_data()

    print("Analyzing validation statistics...")
    stats = analyze_validation_statistics(validation_data)

    # Save to file
    output_file = OUTPUT_DIR / "validation_jobs_steps_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ Analysis saved to: {output_file}")

    # Print summary
    print_summary(stats)

    # Print paper summary
    print("\n" + "="*70)
    print("FOR PAPER - KEY NUMBERS")
    print("="*70)
    summary = stats["summary"]
    print(f"""
Dataset Validation Metrics (from fail→success commits):

• Total Instances: {summary['total_instances_with_validation']}
• Total Commits: {summary['total_commits_analyzed']} ({summary['avg_commits_per_instance']} avg per instance)
• Total Validation Jobs: {summary['total_validation_jobs']:,} ({summary['avg_jobs_per_instance']} avg per instance)
• Total Validation Steps: {summary['total_validation_steps']:,} ({summary['avg_steps_per_instance']} avg per instance)
• Total Failed Jobs: {summary['total_failed_jobs']:,} ({summary['avg_failed_jobs_per_instance']} avg per instance)
• Total Failed Steps: {summary['total_failed_steps']:,} ({summary['avg_failed_steps_per_instance']} avg per instance)
• Failure Rate: {summary['failed_jobs_percentage']}% jobs, {summary['failed_steps_percentage']}% steps

This data represents the COMPLETE validation from sha_fail to sha_success,
tracking ALL jobs and steps across ALL commits in the repair trajectory.
    """)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate detailed dataset overview for paper - includes comprehensive
statistics on jobs, steps, failures, and code changes.

This is the COMPLETE version for academic papers with all metrics.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
OUTPUT_DIR = PROJECT_ROOT / "dataset_overview"


def safe_convert(value):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value


def extract_diff_stats(diff_text: str) -> Dict[str, int]:
    """Extract detailed diff statistics."""
    if not diff_text:
        return {
            "lines_added": 0,
            "lines_deleted": 0,
            "lines_changed": 0,
            "files_modified": 0,
        }

    lines = diff_text.split("\n")
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    deleted = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    files = len(re.findall(r"^diff --git", diff_text, re.MULTILINE))

    return {
        "lines_added": added,
        "lines_deleted": deleted,
        "lines_changed": added + deleted,
        "files_modified": files,
    }


def analyze_jobs_and_steps(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze overall jobs and steps across all instances."""

    # Overall totals
    total_jobs = df["total_jobs"].fillna(0).sum()
    total_steps = df["total_steps"].fillna(0).sum()

    # Filtered totals (if available)
    total_filtered_jobs = df["overall_jobs_count_filtered"].fillna(0).sum()
    total_filtered_steps = df["overall_steps_count_filtered"].fillna(0).sum()

    # Per-instance averages
    avg_jobs_per_instance = df["total_jobs"].mean()
    avg_steps_per_instance = df["total_steps"].mean()

    # Distribution
    jobs_per_instance = df["total_jobs"].fillna(0).astype(int).value_counts().to_dict()

    # Instances with data
    instances_with_jobs = (df["total_jobs"] > 0).sum()
    instances_with_filtered = df["overall_jobs_count_filtered"].notna().sum()

    return {
        "overall_jobs": {
            "total_jobs_all_instances": int(total_jobs),
            "total_steps_all_instances": int(total_steps),
            "instances_with_job_data": int(instances_with_jobs),
            "avg_jobs_per_instance": round(avg_jobs_per_instance, 2),
            "avg_steps_per_instance": round(avg_steps_per_instance, 2),
            "avg_steps_per_job": round(total_steps / total_jobs, 2) if total_jobs > 0 else 0,
        },
        "filtered_jobs": {
            "total_filtered_jobs": int(total_filtered_jobs),
            "total_filtered_steps": int(total_filtered_steps),
            "instances_with_filtered_data": int(instances_with_filtered),
            "avg_filtered_jobs_per_instance": round(total_filtered_jobs / instances_with_filtered, 2) if instances_with_filtered > 0 else 0,
            "avg_filtered_steps_per_instance": round(total_filtered_steps / instances_with_filtered, 2) if instances_with_filtered > 0 else 0,
        },
        "distribution": {
            "jobs_per_instance_distribution": {str(k): int(v) for k, v in sorted(jobs_per_instance.items())},
        }
    }


def analyze_failed_jobs_and_steps(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze failed jobs and steps in detail."""

    instances_with_failures = 0
    total_failed_jobs = 0
    total_failed_steps = 0
    failed_job_names = []
    failed_steps_list = []

    for idx, row in df.iterrows():
        failed_jobs = row.get("failed_jobs")
        if failed_jobs is not None and len(failed_jobs) > 0:
            instances_with_failures += 1

            if isinstance(failed_jobs, np.ndarray):
                failed_jobs = failed_jobs.tolist()

            for job_info in failed_jobs:
                if isinstance(job_info, dict):
                    total_failed_jobs += 1
                    job_name = job_info.get("job_name", "unknown")
                    failed_job_names.append(job_name)

                    steps = job_info.get("steps", [])
                    if isinstance(steps, np.ndarray):
                        steps = steps.tolist()

                    total_failed_steps += len(steps)
                    failed_steps_list.extend(steps)

    # Count unique failed job names and steps
    job_name_counts = Counter(failed_job_names)

    # Analyze common failure steps
    step_keywords = defaultdict(int)
    for step in failed_steps_list:
        step_lower = str(step).lower()
        # Extract keywords
        if "test" in step_lower:
            step_keywords["test"] += 1
        if "lint" in step_lower or "flake8" in step_lower or "pylint" in step_lower:
            step_keywords["linting"] += 1
        if "format" in step_lower or "black" in step_lower or "prettier" in step_lower:
            step_keywords["formatting"] += 1
        if "build" in step_lower or "compile" in step_lower:
            step_keywords["build"] += 1
        if "install" in step_lower or "dependencies" in step_lower:
            step_keywords["dependencies"] += 1

    return {
        "failed_jobs_overview": {
            "instances_with_failed_jobs": int(instances_with_failures),
            "total_failed_jobs": int(total_failed_jobs),
            "total_failed_steps": int(total_failed_steps),
            "avg_failed_jobs_per_instance": round(total_failed_jobs / instances_with_failures, 2) if instances_with_failures > 0 else 0,
            "avg_failed_steps_per_instance": round(total_failed_steps / instances_with_failures, 2) if instances_with_failures > 0 else 0,
            "avg_failed_steps_per_job": round(total_failed_steps / total_failed_jobs, 2) if total_failed_jobs > 0 else 0,
        },
        "top_failed_job_names": dict(job_name_counts.most_common(10)),
        "failed_step_categories": dict(step_keywords),
    }


def analyze_code_changes(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze code changes in detail."""

    all_stats = []
    for idx, row in df.iterrows():
        stats = extract_diff_stats(row.get("diff", ""))
        all_stats.append(stats)

    stats_df = pd.DataFrame(all_stats)

    # Aggregate statistics
    total_added = stats_df["lines_added"].sum()
    total_deleted = stats_df["lines_deleted"].sum()
    total_changed = stats_df["lines_changed"].sum()
    total_files = stats_df["files_modified"].sum()

    # Distributions
    lines_changed_dist = stats_df["lines_changed"].describe()
    files_modified_dist = stats_df["files_modified"].describe()

    # Size categories
    small_changes = (stats_df["lines_changed"] <= 10).sum()
    medium_changes = ((stats_df["lines_changed"] > 10) & (stats_df["lines_changed"] <= 100)).sum()
    large_changes = ((stats_df["lines_changed"] > 100) & (stats_df["lines_changed"] <= 500)).sum()
    very_large_changes = (stats_df["lines_changed"] > 500).sum()

    return {
        "aggregate": {
            "total_lines_added": int(total_added),
            "total_lines_deleted": int(total_deleted),
            "total_lines_changed": int(total_changed),
            "total_files_modified": int(total_files),
            "avg_lines_added_per_instance": round(total_added / len(df), 2),
            "avg_lines_deleted_per_instance": round(total_deleted / len(df), 2),
            "avg_lines_changed_per_instance": round(total_changed / len(df), 2),
            "avg_files_modified_per_instance": round(total_files / len(df), 2),
        },
        "distribution": {
            "lines_changed": {
                "min": int(lines_changed_dist["min"]),
                "25th_percentile": int(lines_changed_dist["25%"]),
                "median": int(lines_changed_dist["50%"]),
                "75th_percentile": int(lines_changed_dist["75%"]),
                "max": int(lines_changed_dist["max"]),
                "mean": round(lines_changed_dist["mean"], 2),
                "std": round(lines_changed_dist["std"], 2),
            },
            "files_modified": {
                "min": int(files_modified_dist["min"]),
                "25th_percentile": int(files_modified_dist["25%"]),
                "median": int(files_modified_dist["50%"]),
                "75th_percentile": int(files_modified_dist["75%"]),
                "max": int(files_modified_dist["max"]),
                "mean": round(files_modified_dist["mean"], 2),
                "std": round(files_modified_dist["std"], 2),
            },
        },
        "size_categories": {
            "small_changes_1_10_lines": int(small_changes),
            "medium_changes_11_100_lines": int(medium_changes),
            "large_changes_101_500_lines": int(large_changes),
            "very_large_changes_500plus_lines": int(very_large_changes),
        }
    }


def normalize_error_types(error_type: Any) -> List[str]:
    """Normalize error_type field to list of strings."""
    if error_type is None or (isinstance(error_type, float) and pd.isna(error_type)):
        return []
    if isinstance(error_type, np.ndarray):
        return [str(e).strip() for e in error_type.tolist() if str(e).strip()]
    if isinstance(error_type, list):
        return [str(e).strip() for e in error_type if str(e).strip()]
    return [str(error_type).strip()] if str(error_type).strip() else []


def analyze_repositories(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze repository distribution."""

    df = df.copy()
    df["full_repo"] = df["repo_owner"] + "/" + df["repo_name"]

    unique_repo_names = df["repo_name"].nunique()
    unique_full_repos = df["full_repo"].nunique()

    # Repository counts
    repo_name_counts = df["repo_name"].value_counts()
    full_repo_counts = df["full_repo"].value_counts()

    # Language distribution
    language_counts = df["language"].value_counts()

    # Repos with multiple instances (good for memory experiments)
    repos_with_multiple = (repo_name_counts > 1).sum()
    instances_in_multi_repos = repo_name_counts[repo_name_counts > 1].sum()

    return {
        "overall": {
            "unique_repo_names": int(unique_repo_names),
            "unique_owner_repo_pairs": int(unique_full_repos),
            "forks_or_duplicates": int(unique_full_repos - unique_repo_names),
            "repos_with_multiple_instances": int(repos_with_multiple),
            "instances_from_multi_instance_repos": int(instances_in_multi_repos),
            "percentage_in_multi_repos": round(instances_in_multi_repos / len(df) * 100, 2),
        },
        "top_10_by_repo_name": {k: int(v) for k, v in repo_name_counts.head(10).items()},
        "top_10_by_owner_repo": {k: int(v) for k, v in full_repo_counts.head(10).items()},
        "language_distribution": {k: int(v) for k, v in language_counts.items()},
    }


def analyze_failure_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze failure type distribution."""

    df = df.copy()
    df["error_list"] = df["error_type"].apply(normalize_error_types)
    df["num_error_types"] = df["error_list"].apply(len)

    # All error types
    all_errors = [e for errors in df["error_list"] for e in errors]
    error_counts = Counter(all_errors)

    # Multi-problem analysis
    single_problem = (df["num_error_types"] == 1).sum()
    multi_problem = (df["num_error_types"] > 1).sum()

    # Co-occurrence
    co_occurrence = defaultdict(int)
    for error_list in df[df["num_error_types"] > 1]["error_list"]:
        if len(error_list) > 1:
            sorted_errors = tuple(sorted(error_list))
            co_occurrence[sorted_errors] += 1

    return {
        "unique_error_types": len(error_counts),
        "total_error_occurrences": len(all_errors),
        "error_type_distribution": dict(error_counts),
        "top_10_error_types": dict(error_counts.most_common(10)),
        "multi_problem_stats": {
            "single_problem_instances": int(single_problem),
            "multi_problem_instances": int(multi_problem),
            "multi_problem_percentage": round(multi_problem / len(df) * 100, 2),
            "avg_errors_per_instance": round(df["num_error_types"].mean(), 2),
            "max_errors_in_instance": int(df["num_error_types"].max()),
        },
        "top_10_error_combinations": [
            {
                "combination": list(combo),
                "count": int(count)
            }
            for combo, count in sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    }


def generate_detailed_overview() -> Dict[str, Any]:
    """Generate comprehensive detailed overview for paper."""

    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"Dataset loaded: {len(df)} instances\n")

    print("Analyzing repositories...")
    repo_stats = analyze_repositories(df)

    print("Analyzing failure types...")
    failure_stats = analyze_failure_types(df)

    print("Analyzing jobs and steps...")
    jobs_stats = analyze_jobs_and_steps(df)

    print("Analyzing failed jobs and steps...")
    failed_stats = analyze_failed_jobs_and_steps(df)

    print("Analyzing code changes...")
    changes_stats = analyze_code_changes(df)

    overview = {
        "dataset_summary": {
            "total_instances": len(df),
            "data_collection_date": "2026",
            "benchmark_name": "CI-REPAIR-BENCH",
        },
        "repositories": repo_stats,
        "failure_types": failure_stats,
        "jobs_and_steps": jobs_stats,
        "failed_jobs_and_steps": failed_stats,
        "code_changes": changes_stats,
    }

    return overview


def print_paper_summary(stats: Dict[str, Any]):
    """Print formatted summary for paper."""

    print("\n" + "="*70)
    print("CI-REPAIR-BENCH DETAILED OVERVIEW (FOR PAPER)")
    print("="*70)

    # Dataset summary
    print("\n📊 DATASET SUMMARY")
    print("-" * 70)
    print(f"Total Instances: {stats['dataset_summary']['total_instances']}")

    # Repositories
    repo = stats['repositories']['overall']
    print(f"\n📁 REPOSITORIES")
    print("-" * 70)
    print(f"Unique Repository Names: {repo['unique_repo_names']}")
    print(f"Unique Owner/Repo Pairs: {repo['unique_owner_repo_pairs']}")
    print(f"  (includes {repo['forks_or_duplicates']} forks/duplicates)")
    print(f"Repos with Multiple Instances: {repo['repos_with_multiple_instances']}")
    print(f"  ({repo['percentage_in_multi_repos']}% of instances)")

    # Failure types
    ft = stats['failure_types']
    print(f"\n🔴 FAILURE TYPES")
    print("-" * 70)
    print(f"Unique Error Types: {ft['unique_error_types']}")
    print(f"Multi-Problem Instances: {ft['multi_problem_stats']['multi_problem_instances']} "
          f"({ft['multi_problem_stats']['multi_problem_percentage']}%)")
    print(f"Average Errors per Instance: {ft['multi_problem_stats']['avg_errors_per_instance']}")
    print(f"\nTop 5 Failure Types:")
    for i, (error, count) in enumerate(list(ft['top_10_error_types'].items())[:5], 1):
        pct = count / ft['total_error_occurrences'] * 100
        print(f"  {i}. {error}: {count} ({pct:.1f}%)")

    # Jobs and steps
    jobs = stats['jobs_and_steps']['overall_jobs']
    filtered = stats['jobs_and_steps']['filtered_jobs']
    print(f"\n⚙️  JOBS AND STEPS")
    print("-" * 70)
    print(f"Total Jobs (all instances): {jobs['total_jobs_all_instances']:,}")
    print(f"Total Steps (all instances): {jobs['total_steps_all_instances']:,}")
    print(f"Average Jobs per Instance: {jobs['avg_jobs_per_instance']}")
    print(f"Average Steps per Instance: {jobs['avg_steps_per_instance']}")
    print(f"Average Steps per Job: {jobs['avg_steps_per_job']}")
    print(f"\nFiltered Data:")
    print(f"  Total Filtered Jobs: {filtered['total_filtered_jobs']:,}")
    print(f"  Total Filtered Steps: {filtered['total_filtered_steps']:,}")

    # Failed jobs and steps
    failed = stats['failed_jobs_and_steps']['failed_jobs_overview']
    print(f"\n❌ FAILED JOBS AND STEPS")
    print("-" * 70)
    print(f"Instances with Failures: {failed['instances_with_failed_jobs']}")
    print(f"Total Failed Jobs: {failed['total_failed_jobs']:,}")
    print(f"Total Failed Steps: {failed['total_failed_steps']:,}")
    print(f"Average Failed Jobs per Instance: {failed['avg_failed_jobs_per_instance']}")
    print(f"Average Failed Steps per Instance: {failed['avg_failed_steps_per_instance']}")
    print(f"Average Failed Steps per Job: {failed['avg_failed_steps_per_job']}")

    # Code changes
    changes = stats['code_changes']['aggregate']
    dist = stats['code_changes']['distribution']
    sizes = stats['code_changes']['size_categories']
    print(f"\n📝 CODE CHANGES")
    print("-" * 70)
    print(f"Total Lines Added: {changes['total_lines_added']:,}")
    print(f"Total Lines Deleted: {changes['total_lines_deleted']:,}")
    print(f"Total Lines Changed: {changes['total_lines_changed']:,}")
    print(f"Total Files Modified: {changes['total_files_modified']:,}")
    print(f"\nAverages per Instance:")
    print(f"  Lines Changed: {changes['avg_lines_changed_per_instance']}")
    print(f"  Files Modified: {changes['avg_files_modified_per_instance']}")
    print(f"\nChange Size Distribution:")
    print(f"  Median Lines Changed: {dist['lines_changed']['median']}")
    print(f"  75th Percentile: {dist['lines_changed']['75th_percentile']}")
    print(f"  Max Lines Changed: {dist['lines_changed']['max']}")
    print(f"\nSize Categories:")
    print(f"  Small (1-10 lines): {sizes['small_changes_1_10_lines']}")
    print(f"  Medium (11-100 lines): {sizes['medium_changes_11_100_lines']}")
    print(f"  Large (101-500 lines): {sizes['large_changes_101_500_lines']}")
    print(f"  Very Large (500+ lines): {sizes['very_large_changes_500plus_lines']}")

    print("\n" + "="*70)


def main():
    """Main entry point."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    stats = generate_detailed_overview()

    # Save to JSON
    output_file = OUTPUT_DIR / "detailed_paper_statistics.json"
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ Detailed statistics saved to: {output_file}")

    # Print formatted summary
    print_paper_summary(stats)


if __name__ == "__main__":
    main()

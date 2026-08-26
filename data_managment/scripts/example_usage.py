#!/usr/bin/env python3
"""Example usage of the enriched dataset."""

import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent.parent / "results"


def example_1_basic_stats():
    """Example 1: Get basic statistics from enriched dataset."""
    print("=" * 60)
    print("Example 1: Basic Statistics")
    print("=" * 60)

    with open(DATA_DIR / "enriched_dataset.json", "r") as f:
        data = json.load(f)

    total_instances = len(data["instances"])
    total_jobs = 0
    total_steps = 0
    failed_jobs = 0

    for instance in data["instances"]:
        if summary := instance.get("instance_validation_summary"):
            total_jobs += summary.get("total_validation_jobs", 0)
            total_steps += summary.get("total_validation_steps", 0)
            failed_jobs += summary.get("total_failed_jobs", 0)

    print(f"Total instances: {total_instances}")
    print(f"Total validation jobs: {total_jobs}")
    print(f"Total validation steps: {total_steps}")
    print(f"Failed jobs: {failed_jobs} ({failed_jobs/total_jobs*100:.1f}%)")
    print()


def example_2_job_conclusions():
    """Example 2: Analyze job conclusions across all instances."""
    print("=" * 60)
    print("Example 2: Job Conclusion Breakdown")
    print("=" * 60)

    with open(DATA_DIR / "jobs_steps_flat_view.json", "r") as f:
        data = json.load(f)

    conclusions = Counter(job["conclusion"] for job in data["data"])

    print(f"Total jobs: {len(data['data'])}")
    print("\nBreakdown by conclusion:")
    for conclusion, count in conclusions.most_common():
        percentage = count / len(data["data"]) * 100
        print(f"  {conclusion:20s}: {count:6d} ({percentage:5.1f}%)")
    print()


def example_3_failed_jobs_by_repo():
    """Example 3: Find repos with most failed jobs."""
    print("=" * 60)
    print("Example 3: Top 10 Repos with Most Failed Jobs")
    print("=" * 60)

    with open(DATA_DIR / "jobs_steps_flat_view.json", "r") as f:
        data = json.load(f)

    failed_by_repo = Counter()
    total_by_repo = Counter()

    for job in data["data"]:
        repo = job["repo"]
        total_by_repo[repo] += 1
        if job["conclusion"] == "failure":
            failed_by_repo[repo] += 1

    print("Repo                              Failed  Total  Rate")
    print("-" * 60)
    for repo, failed_count in failed_by_repo.most_common(10):
        total = total_by_repo[repo]
        rate = failed_count / total * 100
        print(f"{repo:30s}  {failed_count:6d}  {total:5d}  {rate:5.1f}%")
    print()


def example_4_step_analysis():
    """Example 4: Analyze which steps fail most often."""
    print("=" * 60)
    print("Example 4: Top 10 Most Commonly Failed Steps")
    print("=" * 60)

    with open(DATA_DIR / "enriched_dataset.json", "r") as f:
        data = json.load(f)

    failed_steps = []

    for instance in data["instances"]:
        for commit in instance.get("commit_metadata", []):
            for failed_job in commit.get("failed_validation_jobs", []):
                for step_name in failed_job.get("failed_validation_step_names", []):
                    failed_steps.append(step_name)

    step_counts = Counter(failed_steps)

    print(f"Total failed steps: {len(failed_steps)}")
    print(f"Unique failed step names: {len(step_counts)}")
    print("\nTop 10 failed steps:")
    for step_name, count in step_counts.most_common(10):
        print(f"  {count:4d}  {step_name}")
    print()


def example_5_commit_type_comparison():
    """Example 5: Compare fail vs success commits."""
    print("=" * 60)
    print("Example 5: Fail vs Success Commit Comparison")
    print("=" * 60)

    with open(DATA_DIR / "jobs_steps_by_commit.json", "r") as f:
        data = json.load(f)

    fail_commits = [c for c in data["data"] if c["commit_type"] == "fail"]
    success_commits = [c for c in data["data"] if c["commit_type"] == "success"]

    def get_avg_jobs(commits):
        if not commits:
            return 0
        return sum(len(c.get("jobs", [])) for c in commits) / len(commits)

    def get_avg_failed_jobs(commits):
        if not commits:
            return 0
        total_failed = sum(
            (c.get("job_conclusion_counts") or {}).get("failure", 0) for c in commits
        )
        return total_failed / len(commits)

    print(f"Fail commits: {len(fail_commits)}")
    print(f"  Avg jobs per commit: {get_avg_jobs(fail_commits):.1f}")
    print(f"  Avg failed jobs per commit: {get_avg_failed_jobs(fail_commits):.1f}")
    print()

    print(f"Success commits: {len(success_commits)}")
    print(f"  Avg jobs per commit: {get_avg_jobs(success_commits):.1f}")
    print(f"  Avg failed jobs per commit: {get_avg_failed_jobs(success_commits):.1f}")
    print()


def example_6_instance_level():
    """Example 6: Instance-level analysis."""
    print("=" * 60)
    print("Example 6: Instance-Level Analysis")
    print("=" * 60)

    with open(DATA_DIR / "jobs_steps_by_instance.json", "r") as f:
        data = json.load(f)

    # Find instances with high failure rates
    instances_with_stats = []

    for instance in data["data"]:
        summary = instance.get("instance_validation_summary", {})
        total_jobs = summary.get("total_validation_jobs", 0)
        failed_jobs = summary.get("total_failed_jobs", 0)

        if total_jobs > 0:
            failure_rate = failed_jobs / total_jobs
            instances_with_stats.append(
                {
                    "id": instance["instance_id"],
                    "repo": instance["repo"],
                    "workflow": instance["workflow_name"],
                    "total_jobs": total_jobs,
                    "failed_jobs": failed_jobs,
                    "failure_rate": failure_rate,
                }
            )

    # Sort by failure rate (descending)
    instances_with_stats.sort(key=lambda x: x["failure_rate"], reverse=True)

    print("Top 10 instances by failure rate:")
    print("ID   Repo                          Workflow              Failed  Total  Rate")
    print("-" * 85)
    for inst in instances_with_stats[:10]:
        print(
            f"{inst['id']:4s} {inst['repo']:30s} {inst['workflow']:20s} "
            f"{inst['failed_jobs']:6d}  {inst['total_jobs']:5d}  {inst['failure_rate']*100:5.1f}%"
        )
    print()


def main():
    """Run all examples."""
    example_1_basic_stats()
    example_2_job_conclusions()
    example_3_failed_jobs_by_repo()
    example_4_step_analysis()
    example_5_commit_type_comparison()
    example_6_instance_level()

    print("=" * 60)
    print("For more examples, see ENRICHMENT_SUMMARY.md")
    print("=" * 60)


if __name__ == "__main__":
    main()

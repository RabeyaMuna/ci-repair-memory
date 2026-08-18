#!/usr/bin/env python3
"""
Validate pushed jobs to ensure data integrity.

This script checks:
1. Commit SHAs exist on GitHub
2. Branches exist
3. Workflow runs can be fetched
4. Data fields are complete

Use this to diagnose issues before running the benchmark.
"""

import os
import json
import requests
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()


def validate_commit_exists(owner, repo, commit_sha, token):
    """Check if a commit exists on GitHub."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.ok:
            return True, response.json()
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)


def validate_branch_exists(owner, repo, branch, token):
    """Check if a branch exists on GitHub."""
    url = f"https://api.github.com/repos/{owner}/{repo}/branches/{branch}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        return response.ok, response.status_code
    except Exception as e:
        return False, str(e)


def get_workflow_runs_for_commit(owner, repo, commit_sha, token):
    """Get workflow runs for a specific commit."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}/check-runs"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.ok:
            data = response.json()
            check_runs = data.get("check_runs", [])
            return len(check_runs), check_runs
        else:
            return 0, []
    except Exception as e:
        return 0, []


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("❌ GITHUB_TOKEN not found")
        return

    config = OmegaConf.load("config.yaml")
    benchmark_owner = config.get("benchmark_owner")

    if not benchmark_owner:
        print("❌ benchmark_owner not found in config.yaml")
        return

    jobs_file = "results/jobs_ids_diff.jsonl"
    if not os.path.exists(jobs_file):
        print(f"❌ File not found: {jobs_file}")
        return

    print("="*80)
    print("VALIDATE PUSHED JOBS")
    print("="*80)
    print(f"Benchmark owner: {benchmark_owner}\n")

    # Read all jobs
    jobs = []
    with open(jobs_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    jobs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not jobs:
        print("No jobs found")
        return

    print(f"Found {len(jobs)} jobs to validate\n")

    # Validation stats
    valid_commits = 0
    invalid_commits = 0
    valid_branches = 0
    invalid_branches = 0
    has_workflows = 0
    no_workflows = 0
    missing_fields = []

    # Validate each job
    for i, job in enumerate(jobs, 1):
        job_id = job.get("id", "?")
        repo = job.get("repo_name", "unknown")
        commit = job.get("commit", "")
        branch = job.get("branch_name", "")

        print(f"[{i}/{len(jobs)}] ID {job_id:4s} | {repo:20s}")

        # Check required fields
        required = ["repo_name", "commit", "id", "sha_original", "branch_name", "workflow"]
        missing = [f for f in required if not job.get(f)]
        if missing:
            print(f"  ⚠️  Missing fields: {', '.join(missing)}")
            missing_fields.append((job_id, missing))

        # Validate commit exists
        if commit:
            exists, info = validate_commit_exists(benchmark_owner, repo, commit, token)
            if exists:
                valid_commits += 1
                print(f"  ✓ Commit exists: {commit[:8]}")
            else:
                invalid_commits += 1
                print(f"  ✗ Commit NOT found: {commit[:8]} ({info})")

        # Validate branch exists
        if branch:
            exists, status = validate_branch_exists(benchmark_owner, repo, branch, token)
            if exists:
                valid_branches += 1
                print(f"  ✓ Branch exists: {branch}")
            else:
                invalid_branches += 1
                print(f"  ✗ Branch NOT found: {branch} ({status})")

        # Check for workflow runs
        if commit:
            count, runs = get_workflow_runs_for_commit(benchmark_owner, repo, commit, token)
            if count > 0:
                has_workflows += 1
                statuses = [r.get("status") for r in runs]
                conclusions = [r.get("conclusion") for r in runs]
                print(f"  ✓ {count} workflow run(s) found")
                print(f"    Statuses: {set(statuses)}")
                if any(conclusions):
                    print(f"    Conclusions: {set(c for c in conclusions if c)}")
            else:
                no_workflows += 1
                print(f"  ⚠️  No workflow runs found (may not have triggered yet)")

        print()

    # Summary
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"\nCommits:")
    print(f"  ✓ Valid:   {valid_commits}")
    print(f"  ✗ Invalid: {invalid_commits}")

    print(f"\nBranches:")
    print(f"  ✓ Valid:   {valid_branches}")
    print(f"  ✗ Invalid: {invalid_branches}")

    print(f"\nWorkflow Runs:")
    print(f"  ✓ Found:     {has_workflows}")
    print(f"  ⚠️  Not found: {no_workflows}")

    if missing_fields:
        print(f"\nMissing Fields ({len(missing_fields)} jobs):")
        for job_id, fields in missing_fields[:10]:
            print(f"  ID {job_id}: {', '.join(fields)}")
        if len(missing_fields) > 10:
            print(f"  ... and {len(missing_fields) - 10} more")

    print("\n" + "="*80)

    if invalid_commits > 0:
        print("⚠️  WARNING: Some commits don't exist on GitHub!")
        print("   This may indicate a bug in push_repo() or the commits were deleted.")
    elif no_workflows > 100:
        print("⚠️  WARNING: Many jobs have no workflow runs!")
        print("   Workflows may not have triggered due to branch restrictions.")
    else:
        print("✅ Validation complete!")


if __name__ == "__main__":
    main()

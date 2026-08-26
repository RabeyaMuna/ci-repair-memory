#!/usr/bin/env python3
"""
Trigger CI runs for commits that don't have metadata.

For each commit without metadata:
  1. Create a temporary branch pointing to that commit
  2. Push the branch to trigger CI
  3. Wait for CI to complete (optional)
  4. Clean up the branch

This allows us to gather metadata for commits that didn't have CI runs originally.
"""

import os
import json
import subprocess
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration
METADATA_FILE = Path(__file__).parent / "results" / "metadata" / "commit_job_metadata.json"
DATASET_FILE = Path(__file__).parent.parent / "dataset" / "lca_dataset.parquet"

# GitHub token
GITHUB_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")

# Benchmark fork owner - UPDATE THIS to your fork's owner
BENCHMARK_OWNER = os.environ.get("BENCHMARK_OWNER", "RabeyaMuna")  # Your GitHub username for forks

# Settings
WAIT_FOR_CI = True  # Wait for CI to complete before cleaning up
CHECK_INTERVAL = 30  # seconds between CI status checks
MAX_WAIT_TIME = 1800  # 30 minutes max wait per commit
BRANCH_PREFIX = "ci-trigger-test"


def get_github_headers():
    """Get GitHub API headers with auth token."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def run_git_command(cmd: List[str], cwd: str = None) -> tuple[bool, str]:
    """Run a git command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)


def check_ci_status(owner: str, repo: str, commit_sha: str) -> Optional[str]:
    """
    Check CI status for a commit.
    Returns: 'pending', 'success', 'failure', 'error', or None if no runs found.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}/check-runs"

    try:
        response = requests.get(url, headers=get_github_headers(), timeout=30)

        if response.status_code == 200:
            data = response.json()
            check_runs = data.get("check_runs", [])

            if not check_runs:
                return None

            statuses = [run.get("status") for run in check_runs]
            conclusions = [run.get("conclusion") for run in check_runs]

            # If any are still running, status is pending
            if "in_progress" in statuses or "queued" in statuses:
                return "pending"

            # If any failed, overall is failure
            if "failure" in conclusions:
                return "failure"

            # If all successful
            if all(c == "success" for c in conclusions if c is not None):
                return "success"

            # Other states
            if "cancelled" in conclusions:
                return "cancelled"

            return "unknown"
        else:
            return None
    except Exception as e:
        print(f"    [ERROR] Failed to check CI status: {e}")
        return None


def trigger_ci_for_commit(owner: str, repo: str, commit_sha: str, repo_path: str) -> bool:
    """
    Trigger CI for a commit by creating and pushing a temporary branch.

    Returns True if successful, False otherwise.
    """
    branch_name = f"{BRANCH_PREFIX}-{commit_sha[:8]}-{int(time.time())}"

    print(f"  Creating branch: {branch_name}")

    # Create branch at the commit
    success, output = run_git_command(
        ["git", "branch", branch_name, commit_sha],
        cwd=repo_path
    )

    if not success:
        print(f"    [ERROR] Failed to create branch: {output}")
        return False

    # Push the branch
    print(f"  Pushing branch to trigger CI...")
    success, output = run_git_command(
        ["git", "push", "origin", branch_name],
        cwd=repo_path
    )

    if not success:
        print(f"    [ERROR] Failed to push branch: {output}")
        # Clean up local branch
        run_git_command(["git", "branch", "-D", branch_name], cwd=repo_path)
        return False

    print(f"  ✓ Branch pushed successfully")

    # Wait for CI to complete if enabled
    if WAIT_FOR_CI:
        print(f"  Waiting for CI to complete (max {MAX_WAIT_TIME}s)...")
        start_time = time.time()

        while time.time() - start_time < MAX_WAIT_TIME:
            time.sleep(CHECK_INTERVAL)

            status = check_ci_status(owner, repo, commit_sha)
            elapsed = int(time.time() - start_time)

            if status is None:
                print(f"    [{elapsed}s] No CI runs found yet...")
            elif status == "pending":
                print(f"    [{elapsed}s] CI in progress...")
            elif status in ["success", "failure", "cancelled", "unknown"]:
                print(f"    [{elapsed}s] CI completed with status: {status}")
                break
            else:
                print(f"    [{elapsed}s] Status: {status}")

        if time.time() - start_time >= MAX_WAIT_TIME:
            print(f"    [WARN] Timeout waiting for CI after {MAX_WAIT_TIME}s")

    # Clean up: delete the branch
    print(f"  Cleaning up branch...")
    run_git_command(["git", "push", "origin", "--delete", branch_name], cwd=repo_path)
    run_git_command(["git", "branch", "-D", branch_name], cwd=repo_path)

    print(f"  ✓ Cleanup complete")
    return True


def clone_or_get_repo(owner: str, repo: str, base_dir: Path, fork_owner: str = None) -> Optional[str]:
    """
    Clone repository if not exists, or return path to existing repo.

    Args:
        owner: Original repo owner (for metadata reference)
        repo: Repository name
        base_dir: Base directory for clones
        fork_owner: If provided, clone from fork_owner/repo instead of owner/repo
    """
    # Use fork owner if provided, otherwise use original owner
    clone_owner = fork_owner if fork_owner else owner
    repo_path = base_dir / clone_owner / repo

    if repo_path.exists():
        print(f"  Using existing repo at: {repo_path}")
        # Fetch latest changes
        run_git_command(["git", "fetch", "--all"], cwd=str(repo_path))
        return str(repo_path)

    print(f"  Cloning {clone_owner}/{repo}...")
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    clone_url = f"https://github.com/{clone_owner}/{repo}.git"
    success, output = run_git_command(["git", "clone", clone_url, str(repo_path)])

    if not success:
        print(f"    [ERROR] Failed to clone: {output}")
        return None

    return str(repo_path)


def load_commits_without_metadata() -> List[Dict]:
    """
    Load commit metadata file and identify commits without metadata.

    Returns list of dicts with: issue_id, repo_owner, repo_name, commit_sha
    """
    if not METADATA_FILE.exists():
        print(f"[ERROR] Metadata file not found: {METADATA_FILE}")
        return []

    print(f"Loading metadata from: {METADATA_FILE}")
    with open(METADATA_FILE, "r") as f:
        data = json.load(f)

    commits_to_trigger = []

    for issue in data:
        issue_id = issue.get("id")
        repo = issue.get("repo", "")

        if "/" not in repo:
            continue

        owner, repo_name = repo.split("/", 1)

        # Check each commit in the issue
        for commit_entry in issue.get("commits", []):
            has_metadata = len(commit_entry.get("metadata", [])) > 0

            if not has_metadata:
                commits_to_trigger.append({
                    "issue_id": issue_id,
                    "repo_owner": owner,
                    "repo_name": repo_name,
                    "commit_sha": commit_entry.get("commit"),
                    "commit_order": commit_entry.get("order")
                })

    return commits_to_trigger


def main():
    """Main function to trigger CI for commits without metadata."""
    if not GITHUB_TOKEN:
        print("[ERROR] No GitHub token found. Set GH_TOKEN or GITHUB_TOKEN in .env")
        return

    print(f"✓ GitHub token loaded")

    # Load commits without metadata
    commits = load_commits_without_metadata()

    if not commits:
        print("\n✓ All commits have metadata!")
        return

    print(f"\nFound {len(commits)} commits without metadata")

    # Group by repository
    repos = {}
    for commit in commits:
        repo_key = f"{commit['repo_owner']}/{commit['repo_name']}"
        if repo_key not in repos:
            repos[repo_key] = []
        repos[repo_key].append(commit)

    print(f"Across {len(repos)} repositories")

    # Ask for confirmation
    print("\n" + "="*80)
    print("This will create temporary branches and trigger CI runs.")
    print(f"Estimated time: ~{len(commits) * (MAX_WAIT_TIME if WAIT_FOR_CI else 10) / 60:.0f} minutes")
    print("="*80)

    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Create temp directory for repos
    temp_dir = Path(__file__).parent.parent / "temp_repos"
    temp_dir.mkdir(exist_ok=True)

    # Process each repository
    total_triggered = 0
    total_failed = 0

    for repo_key, repo_commits in repos.items():
        print(f"\n{'='*80}")
        print(f"Repository: {repo_key}")
        print(f"Commits to trigger: {len(repo_commits)}")
        print(f"{'='*80}")

        owner, repo_name = repo_key.split("/", 1)

        # Clone or get repo - use BENCHMARK_OWNER fork if specified
        print(f"  Original repo: {owner}/{repo_name}")
        print(f"  Using fork: {BENCHMARK_OWNER}/{repo_name}")
        repo_path = clone_or_get_repo(owner, repo_name, temp_dir, fork_owner=BENCHMARK_OWNER)

        if not repo_path:
            print(f"[SKIP] Could not access repository")
            total_failed += len(repo_commits)
            continue

        # Trigger CI for each commit
        for i, commit in enumerate(repo_commits, 1):
            commit_sha = commit["commit_sha"]
            issue_id = commit["issue_id"]

            print(f"\n[{i}/{len(repo_commits)}] Issue {issue_id}, commit {commit_sha[:8]}")

            # Trigger on fork, but check status on fork (BENCHMARK_OWNER)
            success = trigger_ci_for_commit(BENCHMARK_OWNER, repo_name, commit_sha, repo_path)

            if success:
                total_triggered += 1
            else:
                total_failed += 1

            # Small delay between commits
            time.sleep(2)

    # Summary
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"  Total commits: {len(commits)}")
    print(f"  Successfully triggered: {total_triggered}")
    print(f"  Failed: {total_failed}")
    print(f"\nRun fetch_commit_metadata.py again to collect the new metadata.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

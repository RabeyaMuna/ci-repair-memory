#!/usr/bin/env python3
"""
Cancel all workflow runs for the benchmark.

This cancels all queued/in-progress workflow runs to clear the queue.
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()


def parse_url_to_run_info(url):
    """Extract owner, repo, run_id from GitHub Actions URL."""
    if not url:
        return None, None, None

    parts = url.strip("/").split("/")
    if "actions" in parts and "runs" in parts:
        try:
            idx = parts.index("actions")
            owner = parts[idx - 2]
            repo = parts[idx - 1]
            run_id = parts[idx + 2]
            return owner, repo, run_id
        except (ValueError, IndexError):
            pass

    return None, None, None


def get_all_runs_for_repo(owner, repo, token):
    """Get all workflow runs for a repo."""
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs?per_page=100"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.ok:
            return response.json().get("workflow_runs", [])
    except Exception as e:
        print(f"  ⚠️  Error fetching runs: {e}")

    return []


def cancel_run(owner, repo, run_id, token):
    """Cancel a specific workflow run."""
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/cancel"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    try:
        response = requests.post(url, headers=headers, timeout=20)
        return response.status_code == 202  # Accepted
    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("❌ GITHUB_TOKEN not found")
        return

    # Load config to get benchmark_owner
    from omegaconf import OmegaConf
    config = OmegaConf.load("config.yaml")
    benchmark_owner = config.get("benchmark_owner")

    if not benchmark_owner:
        print("❌ benchmark_owner not found in config.yaml")
        return

    # Read jobs_ids file
    jobs_file = "results/jobs_ids_diff.jsonl"
    if not os.path.exists(jobs_file):
        print(f"❌ File not found: {jobs_file}")
        return

    print("="*80)
    print("CANCEL ALL BENCHMARK WORKFLOW RUNS")
    print("="*80)
    print(f"Benchmark owner: {benchmark_owner}\n")

    # Collect all repos from jobs file
    repos_to_cancel = {}  # {repo_name: True}

    with open(jobs_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    repo_name = entry.get("repo_name")

                    if repo_name:
                        repos_to_cancel[repo_name] = True
                except json.JSONDecodeError:
                    pass

    if not repos_to_cancel:
        print("No repositories found to cancel runs for")
        return

    print(f"\nFound {len(repos_to_cancel)} repositories")

    # Ask for confirmation
    print("\n⚠️  This will cancel ALL workflow runs in these repos under {benchmark_owner}.")
    print(f"   Repos: {', '.join(list(repos_to_cancel.keys())[:10])}{' ...' if len(repos_to_cancel) > 10 else ''}")
    response = input("\nContinue? (yes/no): ").strip().lower()

    if response not in ("yes", "y"):
        print("Cancelled.")
        return

    print("\n🔄 Fetching and cancelling runs...\n")

    total_cancelled = 0
    total_failed = 0
    total_runs = 0

    for repo_name in repos_to_cancel.keys():
        print(f"📦 {benchmark_owner}/{repo_name}")

        # Get all runs for this repo
        runs = get_all_runs_for_repo(benchmark_owner, repo_name, token)

        if not runs:
            print(f"  No runs found")
            continue

        # Filter to only queued/in_progress runs
        active_runs = [r for r in runs if r.get("status") in ("queued", "in_progress", "waiting")]

        if not active_runs:
            print(f"  No active runs to cancel ({len(runs)} total runs)")
            continue

        print(f"  Found {len(active_runs)} active runs (out of {len(runs)} total)")
        total_runs += len(active_runs)

        for run in active_runs:
            run_id = run.get("id")
            status = run.get("status")
            print(f"    Cancelling run {run_id} ({status})...", end=" ", flush=True)

            success = cancel_run(benchmark_owner, repo_name, run_id, token)

            if success:
                print("✓")
                total_cancelled += 1
            else:
                print("✗")
                total_failed += 1

            time.sleep(0.3)  # Rate limit protection

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Repositories checked: {len(repos_to_cancel)}")
    print(f"  Active runs found:    {total_runs}")
    print(f"  ✓ Cancelled:          {total_cancelled}")
    print(f"  ✗ Failed:             {total_failed}")
    print("="*80)

    if total_cancelled > 0:
        print("\n✅ Done! All active runs have been cancelled.")
        print("You can now re-run the benchmark or investigate the runner issue.")
    else:
        print("\nℹ️  No active runs were cancelled.")


if __name__ == "__main__":
    main()

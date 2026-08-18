#!/usr/bin/env python3
"""
Enable workflows that were disabled due to inactivity.

GitHub automatically disables workflows on inactive forks.
This script re-enables them via the API.
"""

import os
import json
import requests
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()


def get_workflow_state(owner, repo, workflow_file, token):
    """Get the state of a workflow."""
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_file}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.ok:
            data = response.json()
            return data.get("state"), data.get("id")
        return None, None
    except Exception:
        return None, None


def enable_workflow(owner, repo, workflow_id, token):
    """Enable a disabled workflow."""
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_id}/enable"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    try:
        response = requests.put(url, headers=headers, timeout=20)
        return response.status_code == 204
    except Exception:
        return False


def main():
    import sys

    # Check for --yes flag
    auto_confirm = "--yes" in sys.argv or "-y" in sys.argv

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
    print("ENABLE DISABLED WORKFLOWS")
    print("="*80)
    print(f"Benchmark owner: {benchmark_owner}\n")

    # Read all jobs and collect unique workflow files
    workflows_to_check = {}  # {(repo, workflow_file): [job_ids]}

    with open(jobs_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    job = json.loads(line)
                    repo = job.get("repo_name")
                    workflow = job.get("workflow", "")
                    job_id = job.get("id")

                    if repo and workflow:
                        workflow_file = os.path.basename(workflow)
                        key = (repo, workflow_file)
                        if key not in workflows_to_check:
                            workflows_to_check[key] = []
                        workflows_to_check[key].append(job_id)
                except json.JSONDecodeError:
                    pass

    print(f"Found {len(workflows_to_check)} unique workflows to check\n")

    disabled_workflows = []
    enabled_workflows = []

    # Check each workflow
    for (repo, workflow_file), job_ids in workflows_to_check.items():
        state, workflow_id = get_workflow_state(benchmark_owner, repo, workflow_file, token)

        if state == "disabled_inactivity":
            disabled_workflows.append((repo, workflow_file, workflow_id, job_ids))
            print(f"⚠️  {repo:20s} | {workflow_file:30s} | DISABLED (affects {len(job_ids)} jobs)")
        elif state == "active":
            enabled_workflows.append((repo, workflow_file))
        elif state:
            print(f"   {repo:20s} | {workflow_file:30s} | {state}")

    if not disabled_workflows:
        print("\n✅ All workflows are enabled!")
        return

    print(f"\n{len(disabled_workflows)} workflows are disabled due to inactivity")
    print(f"{len(enabled_workflows)} workflows are active\n")

    if auto_confirm:
        print("Auto-confirming (--yes flag provided)")
        response = "yes"
    else:
        response = input(f"Enable {len(disabled_workflows)} disabled workflows? (yes/no): ").strip().lower()

    if response not in ("yes", "y"):
        print("Cancelled.")
        return

    print("\nEnabling workflows...\n")

    enabled_count = 0
    failed_count = 0

    for repo, workflow_file, workflow_id, job_ids in disabled_workflows:
        print(f"  {repo:20s} | {workflow_file:30s} ...", end=" ", flush=True)

        success = enable_workflow(benchmark_owner, repo, workflow_id, token)

        if success:
            print(f"✓ Enabled (affects IDs: {', '.join(job_ids[:5])}{', ...' if len(job_ids) > 5 else ''})")
            enabled_count += 1
        else:
            print(f"✗ Failed")
            failed_count += 1

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  ✓ Enabled: {enabled_count}")
    print(f"  ✗ Failed:  {failed_count}")
    print("="*80)

    if enabled_count > 0:
        print("\n✅ Workflows enabled!")
        print("\nNEXT STEPS:")
        print("1. Re-push the affected jobs to trigger workflows")
        print("2. Or manually trigger workflows via GitHub UI")
        print("3. Or wait and workflows will auto-trigger on next push")


if __name__ == "__main__":
    main()

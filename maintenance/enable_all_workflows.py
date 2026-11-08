#!/usr/bin/env python3
"""
Enable all workflows for repositories that are waiting
"""
import json
import requests
from pathlib import Path
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load("config.yaml")
token = config.get("GITHUB_TOKEN")
benchmark_owner = config.get("benchmark_owner", "RabeyaMuna")

if not token:
    print("[ERROR] GITHUB_TOKEN not found in config.yaml")
    exit(1)

headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github+json"
}

# Load jobs awaiting
jobs_file = Path("results/jobs_awaiting_diff.jsonl")
if not jobs_file.exists():
    print("[ERROR] jobs_awaiting_diff.jsonl not found")
    exit(1)

# Get unique repo names
repos = set()
with open(jobs_file) as f:
    for line in f:
        job = json.loads(line)
        repos.add(job["repo_name"])

print(f"Found {len(repos)} repositories with waiting jobs")
print(f"Enabling workflows for: {', '.join(sorted(repos))}\n")

enabled_repos = []
failed_repos = []

for i, repo_name in enumerate(sorted(repos), 1):
    print(f"[{i}/{len(repos)}] Processing {repo_name}...")

    # List all workflows
    list_url = f"https://api.github.com/repos/{benchmark_owner}/{repo_name}/actions/workflows"

    try:
        response = requests.get(list_url, headers=headers, timeout=20)
        if not response.ok:
            print(f"  [WARN] Could not list workflows: HTTP {response.status_code}")
            failed_repos.append(repo_name)
            continue

        workflows = response.json().get("workflows", [])

        if not workflows:
            print(f"  [INFO] No workflows found")
            continue

        enabled_count = 0
        for workflow in workflows:
            workflow_id = workflow.get("id")
            workflow_name = workflow.get("name", workflow.get("path", "unknown"))
            state = workflow.get("state")

            if state in ("disabled_inactivity", "disabled_fork", "disabled_manually"):
                print(f"    Enabling '{workflow_name}' (state: {state})...")
                enable_url = f"https://api.github.com/repos/{benchmark_owner}/{repo_name}/actions/workflows/{workflow_id}/enable"
                enable_response = requests.put(enable_url, headers=headers, timeout=20)

                if enable_response.status_code == 204:
                    print(f"      [OK] Enabled")
                    enabled_count += 1
                else:
                    print(f"      [ERROR] Failed: HTTP {enable_response.status_code}")

        if enabled_count > 0:
            print(f"  [OK] Enabled {enabled_count} workflow(s)")
            enabled_repos.append(repo_name)
        else:
            print(f"  [OK] All workflows already active")
            enabled_repos.append(repo_name)

    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        failed_repos.append(repo_name)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Successfully processed: {len(enabled_repos)}/{len(repos)}")
if failed_repos:
    print(f"Failed: {len(failed_repos)}")
    print(f"  {', '.join(failed_repos)}")
print("="*60)

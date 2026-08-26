#!/usr/bin/env python3
"""
Fetch logs for all failed instances from dataset.
"""
import os
import sys
import json
import requests
from pathlib import Path
from urllib.parse import urlparse

# Load credentials from .env
GITHUB_TOKEN = None
env_file = Path('/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/.env')
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('GITHUB_TOKEN=') and not line.startswith('#'):
                GITHUB_TOKEN = line.split('=', 1)[1].strip()
                break

if not GITHUB_TOKEN:
    GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')

if not GITHUB_TOKEN:
    print("ERROR: GITHUB_TOKEN not found in .env or environment")
    sys.exit(1)

HEADERS = {
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
    'Authorization': f'Bearer {GITHUB_TOKEN}'
}

# Input: Read failed workflow results from here
FAILED_RESULTS_FILE = '/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/results/jobs_failure_diff.jsonl'

# Output: Save failed logs to here
OUTPUT_FILE = '/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/data_managment/results/logs/failed_job_logs.json'

def fetch_job_logs(repo_owner: str, repo_name: str, job_id: int) -> str:
    """Fetch logs for a specific job."""
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/actions/jobs/{job_id}/logs'
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        return response.text
    return ''

def fetch_jobs_for_run(repo_owner: str, repo_name: str, run_id: int):
    """Fetch jobs for a workflow run."""
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs/{run_id}/jobs'
    response = requests.get(url, headers=HEADERS, params={'per_page': 100})

    if response.status_code == 200:
        return response.json().get('jobs', [])
    return []

def fetch_logs_for_failed_jobs(repo_owner: str, repo_name: str, run_id: int):
    """Fetch logs for failed jobs in a workflow run."""
    jobs = fetch_jobs_for_run(repo_owner, repo_name, run_id)
    failed_jobs = [j for j in jobs if j['conclusion'] == 'failure']

    if not failed_jobs:
        print(f"    ✗ No failed jobs")
        return []

    print(f"    ✓ Found {len(failed_jobs)} failed job(s), fetching logs...")

    logs_data = []
    for job in failed_jobs:
        print(f"      - {job['name']}", end=' ')

        job_log = fetch_job_logs(repo_owner, repo_name, job['id'])

        if job_log:
            print(f"✓ ({len(job_log)} chars)")

            # Add log for each failed step
            failed_steps = [s for s in job.get('steps', []) if s.get('conclusion') == 'failure']

            if failed_steps:
                for step in failed_steps:
                    logs_data.append({
                        'log': job_log,
                        'step_name': step['name']
                    })
            else:
                # If no specific failed step, use job name
                logs_data.append({
                    'log': job_log,
                    'step_name': job['name']
                })
        else:
            print(f"✗ (logs unavailable)")

    return logs_data

def process_instance(record):
    """Process one failed workflow result and fetch its failed-job logs."""
    workflow_url = record.get('url', '')
    path_parts = [part for part in urlparse(workflow_url).path.split('/') if part]
    if len(path_parts) < 5 or path_parts[2:4] != ['actions', 'runs']:
        print(f"\n  ID {record.get('id')}: invalid workflow URL: {workflow_url!r}")
        return None

    repo_owner, repo_name = path_parts[:2]
    run_id = path_parts[4]
    if not run_id.isdigit():
        print(f"\n  ID {record.get('id')}: invalid workflow run ID: {run_id!r}")
        return None

    print(f"\n  ID {record['id']}: {repo_owner}/{repo_name}")

    logs = fetch_logs_for_failed_jobs(
        repo_owner,
        repo_name,
        int(run_id)
    )

    if logs:
        print(f"    ✓ Collected {len(logs)} log(s)")
        return {
            'id': record['id'],
            'sha_fail': record['commit'],
            'logs': logs
        }
    else:
        print(f"    → No logs available")
        return None

def main():
    print(f"Loading failed workflow results from {FAILED_RESULTS_FILE}...")

    if not os.path.exists(FAILED_RESULTS_FILE):
        print(f"ERROR: {FAILED_RESULTS_FILE} not found")
        return

    failed_results = []
    with open(FAILED_RESULTS_FILE, 'r') as f:
        for line in f:
            if line.strip():
                failed_results.append(json.loads(line))

    print(f"Loaded {len(failed_results)} failed workflow results")
    print('='*80)

    # Process each instance
    results = []
    success_count = 0
    failed_count = 0

    for record in failed_results:
        result = process_instance(record)
        if result:
            results.append(result)
            success_count += 1
        else:
            failed_count += 1

    # Save results
    print('\n' + '='*80)
    print("Saving logs...")

    # Create directory if needed
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Collected logs for {success_count} instances")
    print(f"✗ Failed to fetch logs for {failed_count} instances")
    print(f"✓ Saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()

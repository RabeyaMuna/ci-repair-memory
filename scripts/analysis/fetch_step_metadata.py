#!/usr/bin/env python3
"""
Fetch Step-Level Metadata for Current Results
==============================================

Fetches detailed step-level pass/fail information from GitHub Actions
for the current validation runs.

Usage:
    python scripts/analysis/fetch_step_metadata.py
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Configuration
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
if not GITHUB_TOKEN:
    # Try to load from .env file
    env_file = Path(__file__).parents[2] / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('GITHUB_TOKEN='):
                    GITHUB_TOKEN = line.split('=', 1)[1].strip()
                    break

if not GITHUB_TOKEN:
    print("ERROR: GITHUB_TOKEN not found in environment or .env file")
    print("Please set GITHUB_TOKEN environment variable or add it to .env")
    sys.exit(1)

HEADERS = {
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
    'Authorization': f'Bearer {GITHUB_TOKEN}'
}


def fetch_jobs_for_run(repo_owner: str, repo_name: str, run_id: str) -> List[Dict]:
    """Fetch jobs and their steps for a workflow run."""
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs/{run_id}/jobs'

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code == 200:
            return response.json().get('jobs', [])
        elif response.status_code == 404:
            print(f"  Warning: Run {run_id} not found (404)")
            return []
        else:
            print(f"  Warning: API returned {response.status_code} for run {run_id}")
            return []
    except Exception as e:
        print(f"  Error fetching jobs for run {run_id}: {e}")
        return []


def process_jobs_to_step_details(jobs: List[Dict]) -> Dict:
    """Extract step-level pass/fail details from jobs."""
    all_steps = []
    failed_steps = []
    passed_steps = []

    job_details = []

    for job in jobs:
        job_name = job.get('name', '')
        job_conclusion = job.get('conclusion', '')

        steps = job.get('steps', [])

        for step in steps:
            step_name = step.get('name', '')
            step_conclusion = step.get('conclusion', '')

            # Full step identifier: "job_name: step_name"
            full_step_name = f"{job_name}: {step_name}"
            all_steps.append(full_step_name)

            if step_conclusion == 'failure':
                failed_steps.append(full_step_name)
            elif step_conclusion == 'success':
                passed_steps.append(full_step_name)

        # Job-level details
        job_details.append({
            'job_name': job_name,
            'job_id': job.get('id'),
            'conclusion': job_conclusion,
            'steps': [
                {
                    'name': s.get('name', ''),
                    'conclusion': s.get('conclusion', ''),
                    'status': s.get('status', '')
                }
                for s in steps
            ]
        })

    return {
        'all_steps': all_steps,
        'failed_steps': failed_steps,
        'passed_steps': passed_steps,
        'total_steps': len(all_steps),
        'failed_steps_count': len(failed_steps),
        'passed_steps_count': len(passed_steps),
        'jobs': job_details
    }


def extract_run_id_from_url(url: str) -> Optional[str]:
    """Extract run ID from GitHub Actions URL."""
    if not url:
        return None

    # URL format: https://github.com/owner/repo/actions/runs/12345
    parts = url.rstrip('/').split('/')
    if 'runs' in parts:
        run_idx = parts.index('runs')
        if run_idx + 1 < len(parts):
            return parts[run_idx + 1]

    return None


def main():
    print("="*80)
    print("FETCH STEP-LEVEL METADATA")
    print("="*80)
    print()

    # Paths
    base_dir = Path.cwd()
    results_dir = base_dir / 'results'
    output_file = results_dir / 'validation_steps_current.json'

    # Load all job results
    print(" Loading job results...")
    job_files = [
        'jobs_success_diff.jsonl',
        'jobs_failure_diff.jsonl',
        'jobs_awaiting_diff.jsonl',
        'jobs_cancelled_diff.jsonl',
        'jobs_error_diff.jsonl',
    ]

    all_jobs = []
    for filename in job_files:
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        all_jobs.append(json.loads(line))

    print(f"    Found {len(all_jobs)} jobs to process")

    # Fetch step-level metadata
    print("\n Fetching step-level metadata from GitHub Actions...")
    results = []

    for job in tqdm(all_jobs, desc="Fetching", unit="job"):
        issue_id = job.get('id')
        url = job.get('url', '')
        repo_name = job.get('repo_name', '')

        # Extract run ID from URL
        run_id = extract_run_id_from_url(url)

        if not run_id:
            print(f"  Warning: Could not extract run ID from URL for issue {issue_id}")
            continue

        # Extract owner from URL
        # URL format: https://github.com/owner/repo/actions/runs/12345
        try:
            parts = url.split('/')
            github_idx = parts.index('github.com')
            owner = parts[github_idx + 1]
            repo = parts[github_idx + 2]
        except (ValueError, IndexError):
            print(f"  Warning: Could not parse repository from URL for issue {issue_id}")
            continue

        # Fetch jobs and steps
        jobs_data = fetch_jobs_for_run(owner, repo, run_id)

        if not jobs_data:
            # No jobs data available
            step_details = {
                'all_steps': [],
                'failed_steps': [],
                'passed_steps': [],
                'total_steps': 0,
                'failed_steps_count': 0,
                'passed_steps_count': 0,
                'jobs': []
            }
        else:
            step_details = process_jobs_to_step_details(jobs_data)

        # Store result
        result_entry = {
            'id': issue_id,
            'repo_name': repo_name,
            'run_id': run_id,
            'url': url,
            'overall_conclusion': job.get('conclusion', ''),
            'step_details': step_details
        }
        results.append(result_entry)

    # Save results
    print(f"\n Saving results...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    {output_file}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total_processed = len(results)
    with_steps = sum(1 for r in results if r['step_details']['total_steps'] > 0)
    without_steps = total_processed - with_steps

    print(f"\n Total jobs processed: {total_processed}")
    print(f"   With step data: {with_steps}")
    print(f"   Without step data: {without_steps}")

    if with_steps > 0:
        total_steps = sum(r['step_details']['total_steps'] for r in results)
        total_failed = sum(r['step_details']['failed_steps_count'] for r in results)
        total_passed = sum(r['step_details']['passed_steps_count'] for r in results)

        print(f"\n Total steps across all jobs: {total_steps}")
        print(f"   Failed: {total_failed}")
        print(f"   Passed: {total_passed}")

    print("="*80)
    print("\n✓ Step-level metadata fetched successfully!")
    print(f"  Now run: python scripts/analysis/calculate_success_rate.py")

    return 0


if __name__ == "__main__":
    exit(main())

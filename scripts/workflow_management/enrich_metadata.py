"""
Enrich dataset with additional metadata:
- total_jobs: Total validation steps/jobs considering matrix
- failed_jobs: Names of failed job steps (from failed_jobs_all.json)
- lines_inserted: Number of lines added in the diff
- lines_deleted: Number of lines removed in the diff
- total_lines_changed: Total lines changed (inserted + deleted)
"""

import json
import os
import re
import requests
import time
from typing import Dict, List, Any
import pandas as pd
from pathlib import Path


def parse_diff_stats(diff: str) -> Dict[str, int]:
    """
    Parse a git diff to extract line change statistics.

    Args:
        diff: Git diff string

    Returns:
        Dictionary with 'lines_inserted', 'lines_deleted', 'total_lines_changed'
    """
    if not diff or not isinstance(diff, str):
        return {
            'lines_inserted': 0,
            'lines_deleted': 0,
            'total_lines_changed': 0
        }

    lines_inserted = 0
    lines_deleted = 0

    for line in diff.split('\n'):
        # Skip diff headers and file markers
        if line.startswith('+++') or line.startswith('---') or \
           line.startswith('@@') or line.startswith('diff --git'):
            continue

        # Count insertions (lines starting with +)
        if line.startswith('+'):
            lines_inserted += 1
        # Count deletions (lines starting with -)
        elif line.startswith('-'):
            lines_deleted += 1

    return {
        'lines_inserted': lines_inserted,
        'lines_deleted': lines_deleted,
        'total_lines_changed': lines_inserted + lines_deleted
    }


def get_total_jobs_from_github(
    repo_owner: str,
    repo_name: str,
    run_id: str,
    token: str,
    max_retries: int = 3
) -> int:
    """
    Fetch total number of jobs for a workflow run from GitHub API.

    Args:
        repo_owner: Repository owner
        repo_name: Repository name
        run_id: Workflow run ID (extracted from URL)
        token: GitHub API token
        max_retries: Maximum number of retry attempts

    Returns:
        Total number of jobs (considering matrix expansions)
    """
    if not run_id:
        return 0

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    jobs_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs/{run_id}/jobs"

    for attempt in range(max_retries):
        try:
            response = requests.get(
                jobs_url,
                headers=headers,
                params={"per_page": 100},  # GitHub API pagination
                timeout=30
            )

            if response.status_code == 404:
                print(f"  Run {run_id} not found (404) - may have been deleted")
                return 0

            if response.status_code == 403:
                print(f"  Rate limit or permission issue (403) for run {run_id}")
                # Check if rate limit
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = response.headers.get('X-RateLimit-Remaining')
                    if remaining == '0':
                        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                        wait_seconds = max(reset_time - time.time(), 60)
                        print(f"  Rate limit exceeded. Waiting {wait_seconds}s...")
                        time.sleep(wait_seconds)
                        continue
                return 0

            if not response.ok:
                print(f"  API error {response.status_code} for run {run_id}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return 0

            data = response.json()
            total_count = data.get("total_count", 0)

            # Handle pagination if there are more than 100 jobs
            jobs = data.get("jobs", [])
            while len(jobs) < total_count and "next" in response.links:
                response = requests.get(
                    response.links["next"]["url"],
                    headers=headers,
                    timeout=30
                )
                if response.ok:
                    jobs.extend(response.json().get("jobs", []))
                else:
                    break

            return total_count

        except requests.exceptions.RequestException as e:
            print(f"  Network error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return 0
        except Exception as e:
            print(f"  Unexpected error: {e}")
            return 0

    return 0


def extract_run_id_from_url(url: str) -> str:
    """
    Extract workflow run ID from GitHub Actions URL.

    Example URL: https://github.com/owner/repo/actions/runs/31636993972
    Returns: '31636993972'
    """
    if not url:
        return ""

    match = re.search(r'/actions/runs/(\d+)', url)
    if match:
        return match.group(1)
    return ""


def load_failed_jobs(failed_jobs_path: str) -> Dict[str, List[str]]:
    """
    Load failed jobs mapping from failed_jobs_all.json.

    Args:
        failed_jobs_path: Path to failed_jobs_all.json

    Returns:
        Dictionary mapping issue ID to list of failed job names
    """
    failed_jobs_map = {}

    if not os.path.exists(failed_jobs_path):
        print(f"Warning: {failed_jobs_path} not found")
        return failed_jobs_map

    with open(failed_jobs_path, 'r') as f:
        failed_jobs_data = json.load(f)

    for item in failed_jobs_data:
        issue_id = str(item.get('id'))
        failed_jobs = item.get('failed_jobs', [])
        failed_jobs_names = [job.get('step_name') for job in failed_jobs]
        failed_jobs_map[issue_id] = failed_jobs_names

    return failed_jobs_map


def enrich_dataset_metadata(
    dataset_path: str,
    failed_jobs_path: str,
    jobs_ids_path: str,
    output_path: str,
    github_token: str,
    fetch_total_jobs: bool = True
):
    """
    Enrich dataset with additional metadata.

    Args:
        dataset_path: Path to lca_dataset.parquet
        failed_jobs_path: Path to failed_jobs_all.json
        jobs_ids_path: Path to jobs_ids_diff.jsonl
        output_path: Path to save enriched dataset
        github_token: GitHub API token
        fetch_total_jobs: Whether to fetch total jobs from GitHub API
    """
    print("Loading dataset...")
    df = pd.read_parquet(dataset_path)

    print("Loading failed jobs...")
    failed_jobs_map = load_failed_jobs(failed_jobs_path)

    print("Loading jobs URLs...")
    jobs_urls = {}
    with open(jobs_ids_path, 'r') as f:
        for line in f:
            job_data = json.loads(line)
            issue_id = str(job_data.get('id'))
            jobs_urls[issue_id] = {
                'url': job_data.get('url', ''),
                'repo_name': job_data.get('repo_name', '')
            }

    print(f"Processing {len(df)} records...")

    # Add new columns
    df['total_jobs'] = 0
    df['failed_jobs'] = None
    df['lines_inserted'] = 0
    df['lines_deleted'] = 0
    df['total_lines_changed'] = 0
    df['num_failed_jobs'] = 0

    for idx, row in df.iterrows():
        issue_id = str(row['id'])

        # 1. Parse diff statistics
        diff_stats = parse_diff_stats(row.get('diff', ''))
        df.at[idx, 'lines_inserted'] = diff_stats['lines_inserted']
        df.at[idx, 'lines_deleted'] = diff_stats['lines_deleted']
        df.at[idx, 'total_lines_changed'] = diff_stats['total_lines_changed']

        # 2. Add failed jobs list
        failed_jobs = failed_jobs_map.get(issue_id, [])
        df.at[idx, 'failed_jobs'] = failed_jobs
        df.at[idx, 'num_failed_jobs'] = len(failed_jobs)

        # 3. Fetch total jobs from GitHub API (if enabled)
        if fetch_total_jobs and issue_id in jobs_urls:
            url = jobs_urls[issue_id]['url']
            repo_name = jobs_urls[issue_id]['repo_name']
            run_id = extract_run_id_from_url(url)

            if run_id:
                # Extract repo_owner from dataset
                repo_owner = row.get('repo_owner', '')

                if not repo_owner and '/' in repo_name:
                    # If repo_name contains owner, split it
                    parts = repo_name.split('/')
                    if len(parts) == 2:
                        repo_owner = parts[0]
                        repo_name = parts[1]

                if repo_owner:
                    print(f"[{idx+1}/{len(df)}] Fetching total jobs for ID {issue_id} (run {run_id})...")
                    total_jobs = get_total_jobs_from_github(
                        repo_owner=repo_owner,
                        repo_name=repo_name,
                        run_id=run_id,
                        token=github_token
                    )
                    df.at[idx, 'total_jobs'] = total_jobs

                    # Rate limiting: sleep briefly between requests
                    time.sleep(0.5)
                else:
                    print(f"  Warning: No repo_owner found for ID {issue_id}")

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} records")

    print(f"\nSaving enriched dataset to {output_path}...")
    df.to_parquet(output_path, index=False)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"\nLine changes:")
    print(f"  Avg lines inserted: {df['lines_inserted'].mean():.2f}")
    print(f"  Avg lines deleted: {df['lines_deleted'].mean():.2f}")
    print(f"  Avg total changed: {df['total_lines_changed'].mean():.2f}")
    print(f"\nFailed jobs:")
    print(f"  Avg failed jobs per issue: {df['num_failed_jobs'].mean():.2f}")
    print(f"  Max failed jobs: {df['num_failed_jobs'].max()}")

    if fetch_total_jobs:
        print(f"\nTotal jobs (matrix-expanded):")
        print(f"  Avg total jobs: {df['total_jobs'].mean():.2f}")
        print(f"  Max total jobs: {df['total_jobs'].max()}")

    print("="*60)


if __name__ == "__main__":
    # Configuration
    BASE_DIR = "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH"

    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "lca_dataset.parquet")
    FAILED_JOBS_PATH = os.path.join(BASE_DIR, "dataset", "failed_jobs_all.json")
    JOBS_IDS_PATH = os.path.join(BASE_DIR, "results", "jobs_ids_diff.jsonl")
    OUTPUT_PATH = os.path.join(BASE_DIR, "dataset", "lca_dataset_enriched.parquet")

    # Get GitHub token from environment or config.yaml
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

    if not GITHUB_TOKEN:
        # Try loading from config.yaml
        try:
            from omegaconf import OmegaConf
            config_path = os.path.join(BASE_DIR, "config.yaml")
            if os.path.exists(config_path):
                config = OmegaConf.load(config_path)
                GITHUB_TOKEN = config.get("token")
                if GITHUB_TOKEN:
                    print(f"✓ Using GitHub token from config.yaml")
        except ImportError:
            pass

    if not GITHUB_TOKEN:
        print("Warning: GITHUB_TOKEN not found in environment variables or config.yaml")
        print("Set it with: export GITHUB_TOKEN='your_token_here'")
        print("Proceeding without fetching total jobs from GitHub API...")
        FETCH_TOTAL_JOBS = False
    else:
        FETCH_TOTAL_JOBS = True

    enrich_dataset_metadata(
        dataset_path=DATASET_PATH,
        failed_jobs_path=FAILED_JOBS_PATH,
        jobs_ids_path=JOBS_IDS_PATH,
        output_path=OUTPUT_PATH,
        github_token=GITHUB_TOKEN,
        fetch_total_jobs=FETCH_TOTAL_JOBS
    )

    print("\n✅ Dataset enrichment complete!")

#!/usr/bin/env python3
"""
Fetch failed jobs and steps for each commit between sha_fail and sha_success.

For each issue:
  1. Get all commits from sha_fail to sha_success
  2. For each commit, fetch job metadata from GitHub API (if available)
  3. Include initial failure info from the dataset
  4. Save results showing:
     - How many commits were needed to solve the issue
     - What jobs and steps failed at each commit
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration
DATASET_FILE = Path(__file__).parent.parent / "dataset" / "lca_dataset.parquet"
OUTPUT_FILE = Path(__file__).parent / "results" / "metadata" / "commit_job_metadata.json"

# GitHub token from .env file or environment
GITHUB_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests


def get_github_headers():
    """Get GitHub API headers with auth token if available."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def get_commits_between(owner: str, repo: str, sha_fail: str, sha_success: str) -> List[str]:
    """
    Get all commits from sha_fail to sha_success using GitHub API.
    Returns list of commit SHAs in chronological order.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/compare/{sha_fail}...{sha_success}"

    try:
        response = requests.get(url, headers=get_github_headers(), timeout=30)

        if response.status_code == 200:
            data = response.json()
            commits = data.get("commits", [])
            # commits are already in chronological order
            commit_shas = [commit["sha"] for commit in commits]

            # Include sha_fail at the beginning (but not sha_success at the end, it's already there)
            if commit_shas and commit_shas[-1] == sha_success:
                all_commits = [sha_fail] + commit_shas[:-1]  # sha_success will be added separately
            else:
                all_commits = [sha_fail] + commit_shas

            return all_commits
        elif response.status_code == 404:
            print(f"    [WARN] Commits not found (404) - repo might be private or commits don't exist")
            return []
        elif response.status_code == 403:
            print(f"    [WARN] Rate limit or permission issue (403)")
            return []
        else:
            print(f"    [WARN] GitHub API returned {response.status_code}")
            return []
    except Exception as e:
        print(f"    [ERROR] Failed to fetch commits: {e}")
        return []


def get_job_steps(owner: str, repo: str, job_id: int) -> Optional[Dict]:
    """Fetch detailed job steps for a specific job."""
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/jobs/{job_id}"

    try:
        response = requests.get(url, headers=get_github_headers(), timeout=30)

        if response.status_code == 200:
            data = response.json()
            steps = data.get("steps", [])

            failed_steps = []
            all_steps = []

            for step in steps:
                step_info = {
                    "name": step.get("name"),
                    "number": step.get("number"),
                    "status": step.get("status"),
                    "conclusion": step.get("conclusion"),
                    "started_at": step.get("started_at"),
                    "completed_at": step.get("completed_at")
                }
                all_steps.append(step_info)

                if step.get("conclusion") == "failure":
                    failed_steps.append(step_info)

            return {
                "all_steps": all_steps,
                "failed_steps": failed_steps,
                "total_steps": len(all_steps),
                "failed_steps_count": len(failed_steps)
            }
        else:
            return None
    except Exception as e:
        print(f"        [ERROR] Failed to fetch job steps: {e}")
        return None


def get_job_metadata_for_commit(owner: str, repo: str, commit_sha: str) -> Dict:
    """
    Fetch job metadata for a specific commit using GitHub check-runs API.
    Returns dict with jobs, steps, and conclusion info.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}/check-runs"

    try:
        response = requests.get(url, headers=get_github_headers(), timeout=30)

        if response.status_code == 200:
            data = response.json()
            check_runs = data.get("check_runs", [])

            if not check_runs:
                return {
                    "has_metadata": False,
                    "commit_sha": commit_sha,
                    "reason": "no_check_runs",
                    "jobs": []
                }

            jobs_info = []
            failed_jobs = []
            conclusions = []
            statuses = []

            for run in check_runs:
                conclusion = run.get("conclusion")
                status = run.get("status")

                job_info = {
                    "id": run.get("id"),
                    "name": run.get("name"),
                    "status": status,
                    "conclusion": conclusion,
                    "html_url": run.get("html_url"),
                    "started_at": run.get("started_at"),
                    "completed_at": run.get("completed_at"),
                    "app_name": run.get("app", {}).get("name", "GitHub Actions")
                }

                jobs_info.append(job_info)
                conclusions.append(conclusion)
                statuses.append(status)

                # Track failed jobs and fetch their steps
                if conclusion == "failure":
                    failed_job = {
                        "job_name": run.get("name"),
                        "job_id": run.get("id"),
                        "html_url": run.get("html_url"),
                        "steps": None
                    }

                    # Fetch job details to get step information
                    job_details = get_job_steps(owner, repo, run.get("id"))
                    if job_details:
                        failed_job["steps"] = job_details.get("failed_steps", [])
                        failed_job["all_steps"] = job_details.get("all_steps", [])
                        failed_job["total_steps"] = job_details.get("total_steps", 0)
                        failed_job["failed_steps_count"] = job_details.get("failed_steps_count", 0)

                    failed_jobs.append(failed_job)

            # Determine overall conclusion
            if "in_progress" in statuses or "queued" in statuses:
                overall = "in_progress"
            elif None in conclusions:
                overall = "waiting"
            elif "failure" in conclusions:
                overall = "failure"
            elif all(c == "success" for c in conclusions if c is not None):
                overall = "success"
            elif "cancelled" in conclusions:
                overall = "cancelled"
            elif "timed_out" in conclusions:
                overall = "timeout"
            else:
                overall = "unknown"

            return {
                "has_metadata": True,
                "commit_sha": commit_sha,
                "jobs": jobs_info,
                "overall_conclusion": overall,
                "failed_jobs": failed_jobs,
                "total_jobs": len(jobs_info),
                "failed_jobs_count": len(failed_jobs),
                "success_jobs_count": sum(1 for c in conclusions if c == "success")
            }
        elif response.status_code == 404:
            return {
                "has_metadata": False,
                "commit_sha": commit_sha,
                "reason": "not_found_404"
            }
        elif response.status_code == 403:
            return {
                "has_metadata": False,
                "commit_sha": commit_sha,
                "reason": "rate_limit_or_forbidden_403"
            }
        else:
            return {
                "has_metadata": False,
                "commit_sha": commit_sha,
                "reason": f"api_error_{response.status_code}"
            }
    except Exception as e:
        return {
            "has_metadata": False,
            "commit_sha": commit_sha,
            "reason": "exception",
            "error": str(e)
        }


def convert_numpy_to_list(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj


def process_issue(row: pd.Series) -> Dict:
    """Process a single issue and fetch commit metadata."""
    issue_id = str(row['id'])
    owner = str(row['repo_owner'])
    repo = str(row['repo_name'])
    sha_fail = str(row['sha_fail'])
    sha_success = str(row['sha_success'])

    print(f"\n[{issue_id}] {owner}/{repo}")
    print(f"  SHA fail: {sha_fail[:8]}")
    print(f"  SHA success: {sha_success[:8]}")

    # Get initial failure information from dataset
    initial_failure_info = {
        "sha_fail": sha_fail,
        "workflow_name": str(row.get('workflow_name', '')),
        "workflow_path": str(row.get('workflow_path', '')),
        "error_type": str(row.get('error_type', '')),
        "commit_date": str(row.get('commit_date', '')),
        "total_jobs": int(row.get('total_jobs', 0)) if pd.notna(row.get('total_jobs')) else 0,
        "total_steps": int(row.get('total_steps', 0)) if pd.notna(row.get('total_steps')) else 0,
        "overall_jobs": convert_numpy_to_list(row.get('overall_jobs', [])),
        "failed_jobs": convert_numpy_to_list(row.get('failed_jobs', [])),
    }

    # Get all commits between sha_fail and sha_success
    print(f"  Fetching commits...")
    commits = get_commits_between(owner, repo, sha_fail, sha_success)

    if not commits:
        print(f"  [WARN] No commits found between {sha_fail[:8]} and {sha_success[:8]}")
        # Return minimal structure with just ID and empty arrays
        return {
            "id": issue_id,
            "sha_fail": sha_fail,
            "sha_success": sha_success,
            "repo": f"{owner}/{repo}",
            "workflow_path": initial_failure_info.get("workflow_path", ""),
            "commits": [],
            "overall_failed_jobs": []
        }

    print(f"  Found {len(commits)} commits from fail to success")

    # Fetch metadata for each commit (except sha_success which we'll handle separately)
    commits_metadata = []
    for i, commit_sha in enumerate(commits):
        print(f"  [{i+1}/{len(commits)}] {commit_sha[:8]}...", end=" ")

        metadata = get_job_metadata_for_commit(owner, repo, commit_sha)

        # Add position info
        metadata["commit_index"] = i
        metadata["is_fail_commit"] = (commit_sha == sha_fail)

        if metadata.get("has_metadata"):
            print(f"✓ [{metadata['overall_conclusion']}] {metadata['total_jobs']} jobs, {metadata['failed_jobs_count']} failed")
        else:
            reason = metadata.get('reason', 'unknown')
            print(f"✗ No metadata ({reason})")

            # If no metadata available, skip fetching details and just mark as empty
            if reason in ['no_check_runs', 'not_found_404']:
                metadata = {
                    "commit_sha": commit_sha,
                    "commit_index": i,
                    "is_fail_commit": (commit_sha == sha_fail),
                    "has_metadata": False,
                    "reason": reason,
                    "jobs": []
                }

        commits_metadata.append(metadata)

        import time
        time.sleep(REQUEST_DELAY)  # Rate limiting

    # Fetch metadata for sha_success separately
    print(f"  Fetching success commit {sha_success[:8]}...", end=" ")
    success_metadata = get_job_metadata_for_commit(owner, repo, sha_success)
    success_metadata["is_success_commit"] = True

    if success_metadata.get("has_metadata"):
        print(f"✓ [{success_metadata['overall_conclusion']}] {success_metadata['total_jobs']} jobs")
    else:
        print(f"✗ No metadata")

    # Build cleaner output structure
    commits_list = []
    overall_failed_jobs = []

    # Process all commits (including sha_fail commits)
    for idx, commit_meta in enumerate(commits_metadata):
        commit_sha = commit_meta.get("commit_sha")

        # Build metadata array for this commit
        metadata = []
        if commit_meta.get("has_metadata"):
            # Create a map of job_id to failed steps for quick lookup
            failed_steps_by_job = {}
            for failed_job in commit_meta.get("failed_jobs", []):
                job_id = failed_job.get("job_id")
                failed_steps_by_job[job_id] = {
                    "failed_steps": failed_job.get("steps", []),
                    "all_steps": failed_job.get("all_steps", []),
                    "total_steps": failed_job.get("total_steps", 0),
                    "failed_steps_count": failed_job.get("failed_steps_count", 0)
                }

            # Build job metadata with steps included
            for job in commit_meta.get("jobs", []):
                job_id = job.get("id")
                job_data = {
                    "job_name": job.get("name"),
                    "job_id": job_id,
                    "status": job.get("status"),
                    "conclusion": job.get("conclusion"),
                    "url": job.get("html_url")
                }

                # Add step information if this job failed
                if job_id in failed_steps_by_job:
                    step_info = failed_steps_by_job[job_id]
                    job_data["failed_steps"] = step_info["failed_steps"]
                    job_data["all_steps"] = step_info["all_steps"]
                    job_data["total_steps"] = step_info["total_steps"]
                    job_data["failed_steps_count"] = step_info["failed_steps_count"]

                metadata.append(job_data)

            # Collect failed jobs for overall list
            for failed_job in commit_meta.get("failed_jobs", []):
                job_name = failed_job.get("job_name")

                # Get failed step names
                steps = failed_job.get("steps", [])
                if steps:
                    for step in steps:
                        overall_failed_jobs.append({
                            "commit": commit_sha,
                            "commit_order": idx + 1,
                            "job_name": job_name,
                            "step_name": step.get("name"),
                            "step_number": step.get("number"),
                            "conclusion": step.get("conclusion")
                        })
                else:
                    # Job failed but no step details available
                    overall_failed_jobs.append({
                        "commit": commit_sha,
                        "commit_order": idx + 1,
                        "job_name": job_name,
                        "step_name": None,
                        "step_number": None,
                        "conclusion": "failure"
                    })

        commits_list.append({
            "order": idx + 1,
            "commit": commit_sha,
            "is_fail_commit": commit_meta.get("is_fail_commit", False),
            "metadata": metadata
        })

    # Add sha_success as final commit
    success_sha = sha_success
    success_metadata_list = []
    if success_metadata.get("has_metadata"):
        # Create a map of job_id to failed steps for quick lookup
        success_failed_steps_by_job = {}
        for failed_job in success_metadata.get("failed_jobs", []):
            job_id = failed_job.get("job_id")
            success_failed_steps_by_job[job_id] = {
                "failed_steps": failed_job.get("steps", []),
                "all_steps": failed_job.get("all_steps", []),
                "total_steps": failed_job.get("total_steps", 0),
                "failed_steps_count": failed_job.get("failed_steps_count", 0)
            }

        for job in success_metadata.get("jobs", []):
            job_id = job.get("id")
            job_data = {
                "job_name": job.get("name"),
                "job_id": job_id,
                "status": job.get("status"),
                "conclusion": job.get("conclusion"),
                "url": job.get("html_url")
            }

            # Add step information if this job failed
            if job_id in success_failed_steps_by_job:
                step_info = success_failed_steps_by_job[job_id]
                job_data["failed_steps"] = step_info["failed_steps"]
                job_data["all_steps"] = step_info["all_steps"]
                job_data["total_steps"] = step_info["total_steps"]
                job_data["failed_steps_count"] = step_info["failed_steps_count"]

            success_metadata_list.append(job_data)

    commits_list.append({
        "order": len(commits) + 1,
        "commit": success_sha,
        "is_success_commit": True,
        "metadata": success_metadata_list
    })

    result = {
        "id": issue_id,
        "sha_fail": sha_fail,
        "sha_success": sha_success,
        "repo": f"{owner}/{repo}",
        "workflow_path": initial_failure_info.get("workflow_path", ""),
        "commits": commits_list,
        "overall_failed_jobs": overall_failed_jobs
    }

    print(f"  Summary: {len(commits_list)} commits, {len(overall_failed_jobs)} failed job steps")

    return result


def main():
    """Main function to process all issues."""
    if not DATASET_FILE.exists():
        print(f"[ERROR] Dataset file not found: {DATASET_FILE}")
        return

    # Check for GitHub token
    if not GITHUB_TOKEN:
        print("[WARN] No GitHub token found in .env file or environment.")
        print("       Add GITHUB_TOKEN to .env file in the project root.")
        print("       Public API rate limit: 60 requests/hour")
        print("       Authenticated rate limit: 5000 requests/hour")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"✓ GitHub token loaded from .env file (ends with ...{GITHUB_TOKEN[-4:]})")

    # Load dataset
    print(f"\nLoading dataset from: {DATASET_FILE}")
    df = pd.read_parquet(DATASET_FILE)
    print(f"Found {len(df)} issues in dataset")

    # Optional: filter to specific IDs for testing
    # df = df[df['id'].isin(['1', '12', '13'])]  # Uncomment to test with specific IDs

    # Process each issue
    results = []
    start_time = datetime.now()

    for i, (idx, row) in enumerate(df.iterrows()):
        print(f"\n{'='*80}")
        print(f"Processing issue {i+1}/{len(df)}")

        try:
            result = process_issue(row)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] Failed to process issue: {e}")
            results.append({
                "id": str(row['id']),
                "error": str(e),
                "repo_owner": str(row['repo_owner']),
                "repo_name": str(row['repo_name']),
            })

        # Save intermediate results every 10 issues
        if (i + 1) % 10 == 0:
            temp_file = OUTPUT_FILE.with_suffix('.tmp.json')
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n  [CHECKPOINT] Saved intermediate results to {temp_file}")

    # Save final results
    print(f"\n{'='*80}")
    print(f"Saving final results to: {OUTPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = datetime.now() - start_time
    print(f"✓ Successfully saved metadata for {len(results)} issues")
    print(f"  Total time: {elapsed}")

    # Collect and save IDs with missing metadata
    missing_metadata_ids = []
    for issue in results:
        issue_id = issue.get("id")
        commits = issue.get("commits", [])

        # Check if any commits are missing metadata
        has_missing = False
        for commit in commits:
            if len(commit.get("metadata", [])) == 0:
                has_missing = True
                break

        if has_missing or len(commits) == 0:
            missing_metadata_ids.append({
                "id": issue_id,
                "repo": issue.get("repo", ""),
                "total_commits": len(commits),
                "commits_without_metadata": sum(1 for c in commits if len(c.get("metadata", [])) == 0)
            })

    # Save missing IDs to separate file
    missing_ids_file = OUTPUT_FILE.parent / "missing_metadata_ids.json"
    with open(missing_ids_file, "w", encoding="utf-8") as f:
        json.dump(missing_metadata_ids, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(missing_metadata_ids)} issues with missing metadata to: {missing_ids_file}")

    # Print summary
    total_commits = sum(r.get("total_commits_between", 0) for r in results)
    total_with_metadata = sum(r.get("summary", {}).get("commits_with_metadata", 0) for r in results)
    total_failures = sum(r.get("summary", {}).get("commits_with_failures", 0) for r in results)
    total_successes = sum(r.get("summary", {}).get("commits_with_success", 0) for r in results)

    print(f"\n{'='*80}")
    print("Overall Summary:")
    print(f"  Total issues processed: {len(results)}")
    print(f"  Total commits analyzed: {total_commits}")
    print(f"  Commits with metadata: {total_with_metadata}")
    print(f"  Commits with failures: {total_failures}")
    print(f"  Commits with success: {total_successes}")
    print(f"  Commits without metadata: {total_commits - total_with_metadata}")

    # Calculate average commits per issue
    if len(results) > 0:
        avg_commits = total_commits / len(results)
        print(f"\n  Average commits per issue: {avg_commits:.2f}")


if __name__ == "__main__":
    main()

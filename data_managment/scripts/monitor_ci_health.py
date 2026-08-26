#!/usr/bin/env python3
"""
Monitor CI Workflow Health

Validates that benchmark CI workflows are still functioning correctly:
- Workflows still exist
- Same failure patterns as baseline
- No unexpected changes
- Ground truth validation

Runs on permanent benchmark branches to detect:
- Workflow modifications
- Infrastructure changes
- Feature changes requiring ground truth updates
"""

import os
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration
BRANCHES_FILE = Path(__file__).parent / "results" / "branches" / "benchmark_branches.json"
BASELINE_FILE = Path(__file__).parent / "results" / "metadata" / "commit_job_metadata.json"
HEALTH_REPORT_FILE = Path(__file__).parent / "results" / "health" / "ci_workflow_health.json"

GITHUB_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
BENCHMARK_OWNER = os.environ.get("BENCHMARK_OWNER", "RabeyaMuna")


def get_github_headers():
    """Get GitHub API headers."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def check_branch_ci_status(owner: str, repo: str, branch_name: str) -> Optional[Dict]:
    """
    Check CI status for a branch.

    Returns dict with:
    - status: pending/success/failure/none
    - jobs: list of job info
    - workflows: list of workflows run
    """
    # Get latest commit on branch
    url = f"https://api.github.com/repos/{owner}/{repo}/branches/{branch_name}"

    try:
        response = requests.get(url, headers=get_github_headers(), timeout=30)

        if response.status_code != 200:
            return None

        branch_data = response.json()
        commit_sha = branch_data["commit"]["sha"]

        # Get check runs for this commit
        check_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}/check-runs"
        check_response = requests.get(check_url, headers=get_github_headers(), timeout=30)

        if check_response.status_code != 200:
            return None

        check_data = check_response.json()
        check_runs = check_data.get("check_runs", [])

        if not check_runs:
            return {
                "status": "none",
                "commit_sha": commit_sha,
                "jobs": [],
                "workflows": []
            }

        jobs = []
        workflows = set()

        for run in check_runs:
            jobs.append({
                "name": run.get("name"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "workflow": run.get("app", {}).get("name", "GitHub Actions")
            })
            workflows.add(run.get("app", {}).get("name", "GitHub Actions"))

        # Determine overall status
        statuses = [j["status"] for j in jobs]
        conclusions = [j["conclusion"] for j in jobs]

        if "in_progress" in statuses or "queued" in statuses:
            overall_status = "pending"
        elif "failure" in conclusions:
            overall_status = "failure"
        elif all(c == "success" for c in conclusions if c):
            overall_status = "success"
        else:
            overall_status = "unknown"

        return {
            "status": overall_status,
            "commit_sha": commit_sha,
            "jobs": jobs,
            "workflows": list(workflows),
            "total_jobs": len(jobs),
            "failed_jobs": sum(1 for j in jobs if j["conclusion"] == "failure")
        }

    except Exception as e:
        print(f"    Error checking CI: {e}")
        return None


def compare_with_baseline(issue_id: str, current: Dict, baseline: Dict) -> Dict:
    """
    Compare current CI status with baseline.

    Returns dict with:
    - status: ok/warning/error
    - changes: list of detected changes
    - details: comparison details
    """
    changes = []

    # Compare job counts
    baseline_jobs = baseline.get("total_jobs", 0)
    current_jobs = current.get("total_jobs", 0)

    if baseline_jobs != current_jobs:
        changes.append({
            "type": "job_count_changed",
            "severity": "warning",
            "message": f"Job count changed: {baseline_jobs} → {current_jobs}"
        })

    # Compare failure counts
    baseline_failures = baseline.get("failed_jobs_count", 0)
    current_failures = current.get("failed_jobs", 0)

    if baseline_failures != current_failures:
        changes.append({
            "type": "failure_count_changed",
            "severity": "error",
            "message": f"Failure count changed: {baseline_failures} → {current_failures}"
        })

    # Compare job names
    baseline_job_names = set(baseline.get("jobs", []))
    current_job_names = set(j["name"] for j in current.get("jobs", []))

    added_jobs = current_job_names - baseline_job_names
    removed_jobs = baseline_job_names - current_job_names

    if added_jobs:
        changes.append({
            "type": "jobs_added",
            "severity": "warning",
            "message": f"New jobs: {', '.join(added_jobs)}"
        })

    if removed_jobs:
        changes.append({
            "type": "jobs_removed",
            "severity": "warning",
            "message": f"Removed jobs: {', '.join(removed_jobs)}"
        })

    # Determine overall status
    if not changes:
        status = "ok"
    elif any(c["severity"] == "error" for c in changes):
        status = "error"
    else:
        status = "warning"

    return {
        "status": status,
        "changes": changes,
        "baseline": {
            "jobs": baseline_jobs,
            "failures": baseline_failures
        },
        "current": {
            "jobs": current_jobs,
            "failures": current_failures
        }
    }


def monitor_all_branches(quick: bool = False) -> Dict:
    """
    Monitor all benchmark branches.

    Args:
        quick: If True, only check a sample of branches

    Returns health report dict
    """
    print(f"\n{'='*80}")
    print("🔍 CI Workflow Health Monitoring")
    print(f"{'='*80}\n")

    # Load branches
    if not BRANCHES_FILE.exists():
        print("❌ No benchmark branches found. Run setup first.")
        return {}

    with open(BRANCHES_FILE) as f:
        branches = json.load(f)

    print(f"Found {len(branches)} benchmark branches")

    # Load baseline if available
    baseline_data = {}
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            baseline_list = json.load(f)
            baseline_data = {item["id"]: item for item in baseline_list}
        print(f"Loaded baseline for {len(baseline_data)} issues")
    else:
        print("⚠️  No baseline found - will only report current status")

    # Filter to successful branches
    active_branches = [b for b in branches if b.get("status") in ["created", "existing"]]
    print(f"Active branches: {len(active_branches)}")

    if quick:
        import random
        active_branches = random.sample(active_branches, min(10, len(active_branches)))
        print(f"Quick mode: checking {len(active_branches)} branches\n")

    # Monitor each branch
    results = []
    ok_count = 0
    warning_count = 0
    error_count = 0

    for i, branch in enumerate(active_branches, 1):
        issue_id = branch["id"]
        repo = branch["repo"]
        branch_name = branch["branch_name"]

        print(f"[{i}/{len(active_branches)}] Issue {issue_id}: {repo}/{branch_name[:30]}...")

        # Check current CI status
        current_status = check_branch_ci_status(BENCHMARK_OWNER, repo, branch_name)

        if not current_status:
            print(f"  ❌ Failed to check CI status")
            results.append({
                "id": issue_id,
                "status": "error",
                "error": "Failed to fetch CI status"
            })
            error_count += 1
            continue

        # Compare with baseline
        if issue_id in baseline_data:
            comparison = compare_with_baseline(
                issue_id,
                current_status,
                baseline_data[issue_id]
            )

            if comparison["status"] == "ok":
                print(f"  ✓ OK - matches baseline")
                ok_count += 1
            elif comparison["status"] == "warning":
                print(f"  ⚠️  Warning - {len(comparison['changes'])} changes")
                for change in comparison["changes"]:
                    print(f"      {change['message']}")
                warning_count += 1
            else:
                print(f"  ❌ Error - critical changes detected")
                for change in comparison["changes"]:
                    print(f"      {change['message']}")
                error_count += 1

            results.append({
                "id": issue_id,
                "repo": repo,
                "branch_name": branch_name,
                **comparison,
                "checked_at": datetime.now().isoformat()
            })
        else:
            # No baseline - just report current status
            print(f"  ℹ️  Status: {current_status['status']} ({current_status['total_jobs']} jobs)")
            results.append({
                "id": issue_id,
                "repo": repo,
                "branch_name": branch_name,
                "status": "no_baseline",
                "current": current_status,
                "checked_at": datetime.now().isoformat()
            })

        # Rate limiting
        time.sleep(0.5)

    # Generate report
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_checked": len(active_branches),
        "summary": {
            "ok": ok_count,
            "warning": warning_count,
            "error": error_count
        },
        "results": results
    }

    # Save report
    HEALTH_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HEALTH_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*80}")
    print("📊 Summary")
    print(f"{'='*80}")
    print(f"  ✓ OK: {ok_count}")
    print(f"  ⚠️  Warnings: {warning_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"\n✓ Report saved to: {HEALTH_REPORT_FILE}")
    print(f"{'='*80}\n")

    return report


def main():
    """Main entry point."""
    import sys

    if not GITHUB_TOKEN:
        print("❌ No GitHub token found. Set GH_TOKEN in .env")
        return

    quick = "--quick" in sys.argv

    monitor_all_branches(quick=quick)


if __name__ == "__main__":
    main()

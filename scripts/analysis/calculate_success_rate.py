#!/usr/bin/env python3
"""
Calculate success rate for each issue by comparing:
- Original failed jobs/steps (from dataset)
- Validation results after patch (fetch from GitHub API)

Success rate = (originally_failed_steps_that_now_pass / total_originally_failed_steps) * 100
"""

import json
import os
import pandas as pd
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional


def load_workflow_runs(results_file: Path) -> Dict[str, Dict]:
    """Load workflow run URLs from jobs_results_diff.jsonl."""
    runs = {}
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                issue_id = str(data.get('id'))
                runs[issue_id] = {
                    'url': data.get('url'),
                    'conclusion': data.get('conclusion'),
                    'run_id': data['url'].split('/')[-1] if data.get('url') else None,
                    'repo_name': data.get('repo_name')
                }
    return runs


def fetch_workflow_jobs_from_api(run_url: str, github_token: str, benchmark_owner: str) -> Optional[Dict]:
    """
    Fetch workflow run details from GitHub API.

    Returns dict with jobs and their step status.
    """
    if not run_url:
        return None

    # Extract repo and run_id from URL
    # URL format: https://github.com/{owner}/{repo}/actions/runs/{run_id}
    parts = run_url.rstrip('/').split('/')
    if len(parts) < 7:
        return None

    repo_name = parts[-4]  # e.g., 'wandb'
    run_id = parts[-1]

    # GitHub API endpoint
    api_url = f"https://api.github.com/repos/{benchmark_owner}/{repo_name}/actions/runs/{run_id}/jobs"

    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Parse jobs and steps
        jobs_data = {}
        for job in data.get('jobs', []):
            job_name = job.get('name')
            job_conclusion = job.get('conclusion')

            steps_data = []
            failed_steps = []

            for step in job.get('steps', []):
                step_name = step.get('name')
                step_conclusion = step.get('conclusion')

                steps_data.append({
                    'name': step_name,
                    'conclusion': step_conclusion
                })

                # Track failed steps
                if step_conclusion in ['failure', 'cancelled', 'timed_out']:
                    failed_steps.append(step_name)

            jobs_data[job_name] = {
                'conclusion': job_conclusion,
                'all_steps': [s['name'] for s in steps_data],
                'failed_steps': failed_steps,
                'steps_details': steps_data
            }

        return jobs_data

    except requests.exceptions.RequestException as e:
        print(f"   [ERROR] API call failed: {e}")
        return None
    except Exception as e:
        print(f"   [ERROR] Failed to parse API response: {e}")
        return None


def calculate_issue_success_rate(issue_row: pd.Series, workflow_run: Dict, github_token: str, benchmark_owner: str) -> Dict:
    """
    Calculate success rate for one issue.

    Args:
        issue_row: Row from dataset with original failed_jobs
        validation_data: Validation results after patch

    Returns:
        Dict with success rate and details
    """
    issue_id = str(issue_row['id'])

    # Get ORIGINAL failed jobs/steps (before patch)
    original_failed_jobs = issue_row.get('failed_jobs', [])
    if original_failed_jobs is None or (hasattr(original_failed_jobs, '__len__') and len(original_failed_jobs) == 0):
        return {
            "id": issue_id,
            "status": "no_original_failures",
            "success_rate": 0,
            "message": "No original failure data"
        }

    # Fetch validation results from API (after patch)
    if not workflow_run:
        return {
            "id": issue_id,
            "status": "no_workflow_run",
            "success_rate": 0,
            "message": "No workflow run data"
        }

    print(f"   [API] Fetching {issue_id}...", end='', flush=True)
    jobs_from_api = fetch_workflow_jobs_from_api(
        workflow_run['url'],
        github_token,
        benchmark_owner
    )
    print(" ✓")

    if not jobs_from_api:
        return {
            "id": issue_id,
            "status": "api_failed",
            "success_rate": 0,
            "message": "Failed to fetch from API"
        }

    overall_conclusion = workflow_run.get('conclusion', 'unknown')

    # Build lookup for validation (from API response)
    # jobs_from_api: {job_name: {'failed_steps': [...], 'all_steps': [...]}}
    validation_failed_dict = {
        job_name: job_data['failed_steps']
        for job_name, job_data in jobs_from_api.items()
    }

    # For each originally failed STEP, check if it now passes
    total_originally_failed_steps = 0
    now_passing_steps = 0
    job_details = []

    for orig_job in original_failed_jobs:
        if not isinstance(orig_job, dict):
            continue

        job_name = str(orig_job.get('job_name', ''))
        orig_failed_step_names = [str(s) for s in orig_job.get('steps', [])]

        if not orig_failed_step_names:
            continue

        total_originally_failed_steps += len(orig_failed_step_names)

        # Get validation failed steps for this job
        validation_failed_steps = validation_failed_dict.get(job_name, [])

        # Count how many originally failed steps are NOT in validation failed list
        # (meaning they now pass)
        job_now_passing = 0
        still_failing = []

        for step_name in orig_failed_step_names:
            if step_name not in validation_failed_steps:
                job_now_passing += 1
            else:
                still_failing.append(step_name)

        now_passing_steps += job_now_passing

        job_details.append({
            "job_name": job_name,
            "originally_failed_steps": len(orig_failed_step_names),
            "now_passing": job_now_passing,
            "still_failing": len(still_failing),
            "still_failing_steps": still_failing,
            "job_success_rate": round((job_now_passing / len(orig_failed_step_names) * 100), 2) if orig_failed_step_names else 0
        })

    # Calculate overall success rate
    if total_originally_failed_steps == 0:
        success_rate = 0
        status = "no_failures"
    else:
        success_rate = (now_passing_steps / total_originally_failed_steps) * 100

        if now_passing_steps == total_originally_failed_steps:
            status = "fully_fixed"
        elif now_passing_steps > 0:
            status = "partially_fixed"
        else:
            status = "not_fixed"

    return {
        "id": issue_id,
        "repo": f"{issue_row.get('repo_owner', '')}/{issue_row.get('repo_name', '')}",
        "status": status,
        "workflow_conclusion": overall_conclusion,
        "success_rate": round(success_rate, 2),
        "total_originally_failed_steps": total_originally_failed_steps,
        "now_passing_steps": now_passing_steps,
        "still_failing_steps": total_originally_failed_steps - now_passing_steps,
        "job_details": job_details
    }


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent.parent.parent

    dataset_path = base_dir / "dataset" / "lca_dataset.parquet"
    results_file = base_dir / "results" / "jobs_results_diff.jsonl"
    output_path = base_dir / "results" / "success_rate_evaluation.json"

    # Load config
    config_path = base_dir / "config.yaml"
    if config_path.exists():
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)
        github_token = config.get("github_token") or config.get("GITHUB_TOKEN")
        benchmark_owner = config.get("benchmark_owner", "RabeyaMuna")
    else:
        github_token = os.environ.get('GITHUB_TOKEN')
        benchmark_owner = os.environ.get("BENCHMARK_OWNER", "RabeyaMuna")

    if not github_token:
        print("[ERROR] GitHub token not found in config.yaml or GITHUB_TOKEN env var!")
        return 1

    print("="*80)
    print("SUCCESS RATE EVALUATION (with GitHub API)")
    print("="*80)
    print()
    print("Comparing:")
    print("  • Original failed jobs/steps (from dataset)")
    print("  • Validation results after patch (fetch from GitHub API)")
    print()

    # Load data
    print("📂 Loading dataset...")
    df = pd.read_parquet(dataset_path)
    print(f"   Loaded {len(df)} issues")

    print("\n📂 Loading workflow runs...")
    workflow_runs = load_workflow_runs(results_file)
    print(f"   Loaded {len(workflow_runs)} workflow runs")

    # Filter to only issues with workflow runs
    df_with_runs = df[df['id'].astype(str).isin(workflow_runs.keys())].copy()
    print(f"   Filtered to {len(df_with_runs)} issues with workflow runs")

    # Process each issue
    print("\n🔄 Calculating success rates (fetching from API)...\n")

    results = []
    for idx, (_, row) in enumerate(df_with_runs.iterrows(), 1):
        issue_id = str(row['id'])
        workflow_run = workflow_runs.get(issue_id)

        result = calculate_issue_success_rate(row, workflow_run, github_token, benchmark_owner)
        results.append(result)

        # Rate limiting
        time.sleep(0.5)  # Be nice to GitHub API

        # Print progress
        status_symbol = {
            "fully_fixed": "✅",
            "partially_fixed": "🟡",
            "not_fixed": "❌",
            "no_validation": "⚠️",
            "no_original_failures": "➖"
        }.get(result['status'], "❓")

        passed = result.get('now_passing_steps', 0)
        total = result.get('total_originally_failed_steps', 0)

        print(
            f"[{idx:3d}/{len(df_with_runs)}] {status_symbol} ID {result['id']:4s}  "
            f"Success: {result['success_rate']:5.1f}%  "
            f"({passed}/{total} originally failed steps now pass)  "
            f"{result['status']}"
        )

    # Calculate statistics
    print("\n📊 Computing statistics...")

    valid_results = [r for r in results if r['status'] not in ['no_validation', 'no_original_failures']]

    fully_fixed = sum(1 for r in valid_results if r['status'] == 'fully_fixed')
    partially_fixed = sum(1 for r in valid_results if r['status'] == 'partially_fixed')
    not_fixed = sum(1 for r in valid_results if r['status'] == 'not_fixed')

    avg_success_rate = sum(r['success_rate'] for r in valid_results) / len(valid_results) if valid_results else 0

    # Workflow-level stats
    workflow_passed = sum(1 for r in results if r.get('workflow_conclusion') == 'success')
    workflow_failed = sum(1 for r in results if r.get('workflow_conclusion') == 'failure')

    stats = {
        "total_issues": len(results),
        "with_validation": len(valid_results),
        "step_level": {
            "fully_fixed": fully_fixed,
            "partially_fixed": partially_fixed,
            "not_fixed": not_fixed,
            "average_success_rate": round(avg_success_rate, 2)
        },
        "workflow_level": {
            "passed": workflow_passed,
            "failed": workflow_failed,
            "pass_rate": round((workflow_passed / len(results) * 100), 2) if results else 0
        }
    }

    # Save results
    print("\n💾 Saving results...")
    output = {
        "summary": stats,
        "results": results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"   ✓ {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total issues: {stats['total_issues']}")
    print(f"With validation data: {stats['with_validation']}")
    print()
    print(f"Step-level Success Rate:")
    print(f"  ✅ Fully fixed:     {stats['step_level']['fully_fixed']}")
    print(f"  🟡 Partially fixed: {stats['step_level']['partially_fixed']}")
    print(f"  ❌ Not fixed:       {stats['step_level']['not_fixed']}")
    print(f"  📊 Average success rate: {stats['step_level']['average_success_rate']}%")
    print()
    print(f"Workflow-level (L3):")
    print(f"  ✅ Passed: {stats['workflow_level']['passed']} ({stats['workflow_level']['pass_rate']}%)")
    print(f"  ❌ Failed: {stats['workflow_level']['failed']}")
    print("="*80)

    return 0


if __name__ == "__main__":
    exit(main())

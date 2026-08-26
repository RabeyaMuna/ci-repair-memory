#!/usr/bin/env python3
"""
Comprehensive script to filter, enrich, and structure the CI-REPAIR-BENCH dataset.

This script:
1. Filters out unnecessary CI lifecycle steps
2. Extracts validation jobs and steps
3. Creates aggregated summaries (overall_jobs, failed_jobs)
4. Enriches the dataset with all validation information
5. Generates multiple output formats for different use cases

Replaces multiple separate scripts with one unified pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    requests = None


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_MANAGEMENT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = DATA_MANAGEMENT_DIR / "results" / "all_instances_metadata.json"
DEFAULT_OUTPUT_DIR = DATA_MANAGEMENT_DIR / "results" / "filtered_validation"
SCHEMA_VERSION = "1.0"
FILTER_VERSION = "1.0"

# Step filtering rules
EXACT_EXCLUDED_STEPS = {
    "complete job": "runner_lifecycle",
    "stop containers": "runner_lifecycle",
    "thank you message": "notification",
    "codecov": "report_upload",
}

STEP_RULES = (
    ("generated_post_action", re.compile(r"^post(?:\s|$)", re.IGNORECASE)),
    (
        "cleanup_or_teardown",
        re.compile(
            r"(?:^|[\s_:/-])(?:cleanup|clean-up|tear\s*down)(?:$|[\s_:/-])",
            re.IGNORECASE,
        ),
    ),
    (
        "artifact_or_report_upload",
        re.compile(
            r"\b(?:upload|send)\b.*\b(?:artifact|coverage|report|result|log)s?\b"
            r"|\b(?:codecov|coveralls)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "publish_or_deploy",
        re.compile(
            r"\b(?:publish|deploy|push)\b.*\b(?:package|image|release|docs?|site)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "notification_or_comment",
        re.compile(
            r"\b(?:notify|notification|comment)\b|\bpost\b.*\bsummary\b",
            re.IGNORECASE,
        ),
    ),
)

ADMIN_JOB_RULE = re.compile(
    r"^(?:cleanup|clean-up|tear\s*down|upload[-_ ](?:artifact|report)s?|"
    r"notify|notification)(?:$|[\s_:/-])",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Input all_instances_metadata.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for all generated files")
    parser.add_argument("--fetch-steps", action="store_true",
                        help="Fetch full step details from GitHub API")
    parser.add_argument("--github-token", type=str,
                        help="GitHub token for API (or set GITHUB_TOKEN env)")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_exclude_step(step_name: str) -> tuple[bool, str | None]:
    """Check if a step should be excluded and return the reason."""
    normalized = step_name.strip().lower()

    # Check exact matches
    if normalized in EXACT_EXCLUDED_STEPS:
        return True, EXACT_EXCLUDED_STEPS[normalized]

    # Check pattern matches
    for reason, pattern in STEP_RULES:
        if pattern.search(step_name):
            return True, reason

    return False, None


def should_exclude_job(job_name: str) -> bool:
    """Check if a job should be excluded entirely."""
    return bool(ADMIN_JOB_RULE.search(job_name))


def filter_steps(step_names: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Filter steps and return validation steps and excluded steps."""
    validation_steps = []
    excluded_steps = []

    for step_name in step_names:
        should_exclude, reason = should_exclude_step(step_name)
        if should_exclude:
            excluded_steps.append({"name": step_name, "reason": reason})
        else:
            validation_steps.append(step_name)

    return validation_steps, excluded_steps


def count_jobs_by_conclusion(jobs: list[dict[str, Any]]) -> dict[str, int]:
    """Count jobs by their conclusion status."""
    conclusions = Counter(job.get('conclusion', 'unknown') for job in jobs)
    return {
        'total': len(jobs),
        'success': conclusions.get('success', 0),
        'failure': conclusions.get('failure', 0),
        'cancelled': conclusions.get('cancelled', 0),
        'skipped': conclusions.get('skipped', 0),
        'neutral': conclusions.get('neutral', 0),
        'timed_out': conclusions.get('timed_out', 0),
        'action_required': conclusions.get('action_required', 0),
        'unknown': conclusions.get('unknown', 0) + conclusions.get(None, 0),
    }


def fetch_job_steps_from_github(
    repo: str,
    run_id: int | str,
    job_id: int | str,
    github_token: str | None
) -> list[dict[str, Any]] | None:
    """Fetch full step details for a job from GitHub API."""
    if not requests or not github_token:
        return None

    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json',
    }

    url = f'https://api.github.com/repos/{repo}/actions/jobs/{job_id}'

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        job_data = response.json()

        steps = job_data.get('steps', [])
        structured_steps = []

        for step in steps:
            structured_steps.append({
                'name': step.get('name'),
                'number': step.get('number'),
                'status': step.get('status'),
                'conclusion': step.get('conclusion'),
                'started_at': step.get('started_at'),
                'completed_at': step.get('completed_at'),
            })

        return structured_steps

    except Exception as e:
        print(f"    Warning: Failed to fetch steps for job {job_id}: {e}")
        return None


def process_commit(commit: dict[str, Any], fetch_steps: bool, github_token: str | None, repo: str) -> dict[str, Any]:
    """Process a single commit and filter its jobs/steps."""
    processed = {
        'sha': commit.get('original_commit', ''),
        'commit_type': commit.get('commit_type', ''),
        'action': commit.get('action', ''),
        'commit_message': commit.get('commit_message', ''),
        'commit_author': commit.get('commit_author', ''),
        'commit_date': commit.get('commit_date', ''),
        'workflow_run_id': commit.get('workflow_run_id'),
        'workflow_conclusion': commit.get('workflow_conclusion', ''),
        'run_html_url': commit.get('run_html_url', ''),
    }

    # Build lookup for failed steps from original data
    original_failed_jobs = commit.get('failed_jobs', [])
    failed_steps_by_job = {}
    for failed_job in original_failed_jobs:
        job_name = failed_job.get('job_name', '')
        failed_step_names = failed_job.get('step_names', [])
        failed_steps_by_job[job_name] = failed_step_names

    # Process jobs
    validation_jobs = []
    excluded_jobs_count = 0
    source_jobs = commit.get('overall_jobs', [])
    failed_validation_jobs = []

    for job in source_jobs:
        job_name = job.get('job_name', '')

        # Check if job should be excluded
        if should_exclude_job(job_name):
            excluded_jobs_count += 1
            continue

        # Filter steps
        step_names = job.get('step_names', [])
        validation_step_names, excluded_steps = filter_steps(step_names)

        # Fetch detailed steps if requested
        steps_data = []
        steps_source = 'validation_names'
        if fetch_steps and github_token:
            job_id = job.get('job_id')
            run_id = processed['workflow_run_id']
            if job_id and run_id:
                api_steps = fetch_job_steps_from_github(repo, run_id, job_id, github_token)
                if api_steps:
                    # Filter API steps
                    filtered_api_steps = [
                        step for step in api_steps
                        if step['name'] in validation_step_names
                    ]
                    steps_data = filtered_api_steps
                    steps_source = 'api'
                    time.sleep(0.1)  # Rate limiting

        # If no API data, create minimal step structure
        if not steps_data:
            steps_data = [
                {
                    'name': name,
                    'number': idx + 1,
                    'conclusion': None,
                    'status': None,
                    'started_at': None,
                    'completed_at': None,
                }
                for idx, name in enumerate(validation_step_names)
            ]

        validation_job = {
            'job_id': job.get('job_id'),
            'job_name': job_name,
            'status': job.get('status'),
            'conclusion': job.get('conclusion'),
            'runner_name': job.get('runner_name'),
            'runner_group_name': job.get('runner_group_name'),
            'labels': job.get('labels', []),
            'started_at': job.get('started_at'),
            'completed_at': job.get('completed_at'),
            'html_url': job.get('html_url'),
            'step_data_available': bool(steps_data),
            'validation_steps_count': len(validation_step_names),
            'validation_step_names': validation_step_names,
            'excluded_steps': excluded_steps,
            'steps': steps_data,
            'steps_data_source': steps_source,
        }

        validation_jobs.append(validation_job)

        # Check if job failed
        if job.get('conclusion') == 'failure':
            # Get actual failed step names from original data
            original_failed_step_names = failed_steps_by_job.get(job_name, [])

            # Filter to keep only validation steps (exclude lifecycle steps)
            failed_validation_step_names = [
                step for step in original_failed_step_names
                if step in validation_step_names
            ]

            # If no failed steps found after filtering, use at least 1 (the job failed somehow)
            if not failed_validation_step_names:
                # Fallback: assume first validation step failed
                failed_validation_step_names = validation_step_names[:1] if validation_step_names else []

            if failed_validation_step_names:  # Only add if there are failed steps
                failed_validation_jobs.append({
                    'job_id': job.get('job_id'),
                    'job_name': job_name,
                    'runner_name': job.get('runner_name'),
                    'failure_evidence': 'failed_steps',
                    'failed_validation_steps_count': len(failed_validation_step_names),
                    'failed_validation_step_names': failed_validation_step_names,
                })

    processed.update({
        'source_jobs_count': len(source_jobs),
        'validation_jobs_count': len(validation_jobs),
        'excluded_jobs_count': excluded_jobs_count,
        'source_steps_count': sum(len(j.get('step_names', [])) for j in source_jobs),
        'validation_steps_count': sum(j['validation_steps_count'] for j in validation_jobs),
        'excluded_steps_count': sum(len(j['excluded_steps']) for j in validation_jobs),
        'failed_validation_jobs_count': len(failed_validation_jobs),
        'failed_validation_steps_count': sum(j['failed_validation_steps_count'] for j in failed_validation_jobs),
        'validation_jobs': validation_jobs,
        'failed_validation_jobs': failed_validation_jobs,
    })

    return processed


def create_overall_jobs_summary(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create aggregated overall jobs summary per instance."""
    summaries = []

    for instance in instances:
        instance_id = instance['id']
        unique_jobs = {}  # job_name -> {job_name, step_names}

        for commit in instance.get('commits', []):
            for job in commit.get('validation_jobs', []):
                job_name = job['job_name']
                if job_name not in unique_jobs:
                    unique_jobs[job_name] = {
                        'job_name': job_name,
                        'step_names': job['validation_step_names'],
                    }

        summaries.append({
            'id': instance_id,
            'repo': instance.get('repo', ''),
            'workflow_name': instance.get('workflow_name', ''),
            'workflow_file': instance.get('workflow_file', ''),
            'has_overall_jobs': len(unique_jobs) > 0,
            'overall_jobs': list(unique_jobs.values()),
            'no_of_jobs': len(unique_jobs),
            'no_of_steps': sum(len(j['step_names']) for j in unique_jobs.values()),
        })

    return summaries


def create_failed_jobs_summary(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create aggregated failed jobs summary per instance."""
    summaries = []

    for instance in instances:
        instance_id = instance['id']
        unique_failed_jobs = {}  # job_name -> {job_name, step_names}

        for commit in instance.get('commits', []):
            for failed_job in commit.get('failed_validation_jobs', []):
                job_name = failed_job['job_name']
                if job_name not in unique_failed_jobs:
                    unique_failed_jobs[job_name] = {
                        'job_name': job_name,
                        'step_names': failed_job.get('failed_validation_step_names', []),
                    }

        summaries.append({
            'id': instance_id,
            'repo': instance.get('repo', ''),
            'workflow_name': instance.get('workflow_name', ''),
            'workflow_file': instance.get('workflow_file', ''),
            'has_failed_jobs': len(unique_failed_jobs) > 0,
            'failed_jobs': list(unique_failed_jobs.values()),
            'no_of_failed_jobs': len(unique_failed_jobs),
            'no_of_failed_steps': sum(len(j['step_names']) for j in unique_failed_jobs.values()),
        })

    return summaries


def create_instance_validation_summary(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create instance-level validation summary."""
    summaries = []

    for instance in instances:
        total_jobs = sum(c.get('validation_jobs_count', 0) for c in instance['commits'])
        total_steps = sum(c.get('validation_steps_count', 0) for c in instance['commits'])
        total_failed_jobs = sum(c.get('failed_validation_jobs_count', 0) for c in instance['commits'])
        total_failed_steps = sum(c.get('failed_validation_steps_count', 0) for c in instance['commits'])

        # Aggregate job conclusions
        all_jobs = []
        for commit in instance['commits']:
            all_jobs.extend(commit.get('validation_jobs', []))

        job_conclusions = count_jobs_by_conclusion(all_jobs)

        summaries.append({
            'id': instance['id'],
            'repo': instance['repo'],
            'workflow_name': instance.get('workflow_name', ''),
            'workflow_file': instance.get('workflow_file', ''),
            'sha_fail': instance.get('sha_fail', ''),
            'sha_success': instance.get('sha_success', ''),
            'source_commits_count': len(instance['commits']),
            'completed_commits_count': len(instance['commits']),
            'total_validation_jobs': total_jobs,
            'total_validation_steps': total_steps,
            'total_failed_jobs': total_failed_jobs,
            'total_failed_steps': total_failed_steps,
            'job_conclusion_counts': job_conclusions,
        })

    return summaries


def enrich_and_structure_instance(
    instance: dict[str, Any],
    overall_jobs_data: dict[str, Any],
    failed_jobs_data: dict[str, Any],
    validation_summary: dict[str, Any],
) -> dict[str, Any]:
    """Enrich an instance with all aggregated data."""
    enriched = instance.copy()

    # Add overall jobs summary
    enriched['overall_jobs_summary'] = {
        'has_overall_jobs': overall_jobs_data.get('has_overall_jobs', False),
        'overall_jobs': overall_jobs_data.get('overall_jobs', []),
        'no_of_jobs': overall_jobs_data.get('no_of_jobs', 0),
        'no_of_steps': overall_jobs_data.get('no_of_steps', 0),
    }

    # Add failed jobs summary
    enriched['failed_jobs_summary'] = {
        'has_failed_jobs': failed_jobs_data.get('has_failed_jobs', False),
        'failed_jobs': failed_jobs_data.get('failed_jobs', []),
        'no_of_failed_jobs': failed_jobs_data.get('no_of_failed_jobs', 0),
        'no_of_failed_steps': failed_jobs_data.get('no_of_failed_steps', 0),
    }

    # Add instance validation summary
    enriched['instance_validation_summary'] = {
        'total_commits': validation_summary.get('source_commits_count', 0),
        'total_validation_jobs': validation_summary.get('total_validation_jobs', 0),
        'total_validation_steps': validation_summary.get('total_validation_steps', 0),
        'total_failed_jobs': validation_summary.get('total_failed_jobs', 0),
        'total_failed_steps': validation_summary.get('total_failed_steps', 0),
        'job_conclusion_counts': validation_summary.get('job_conclusion_counts', {}),
    }

    # Enrich commits with validation data
    enriched_commits = []
    for commit in instance.get('commits', []):
        enriched_commit = commit.copy()

        # Add validation stats
        enriched_commit['validation_stats'] = {
            'total_validation_jobs': commit.get('validation_jobs_count', 0),
            'total_validation_steps': commit.get('validation_steps_count', 0),
            'failed_validation_jobs': commit.get('failed_validation_jobs_count', 0),
            'failed_validation_steps': commit.get('failed_validation_steps_count', 0),
            'excluded_jobs': commit.get('excluded_jobs_count', 0),
            'excluded_steps': commit.get('excluded_steps_count', 0),
            'source_jobs_count': commit.get('source_jobs_count', 0),
            'source_steps_count': commit.get('source_steps_count', 0),
        }

        # Add job conclusion counts
        validation_jobs = commit.get('validation_jobs', [])
        enriched_commit['job_conclusion_counts'] = count_jobs_by_conclusion(validation_jobs)

        enriched_commits.append(enriched_commit)

    enriched['commits'] = enriched_commits

    return enriched


def main() -> None:
    args = parse_args()

    # Setup GitHub token
    github_token = args.github_token or os.environ.get('GITHUB_TOKEN')
    if args.fetch_steps and not github_token:
        print("Warning: --fetch-steps requires GitHub token")
        args.fetch_steps = False

    if args.fetch_steps and not requests:
        print("Warning: requests library not installed")
        args.fetch_steps = False

    print(f"Loading dataset from: {args.input}")
    with args.input.open('r') as f:
        main_data = json.load(f)

    source_sha256 = sha256_file(args.input)
    print(f"Source SHA-256: {source_sha256}")
    print(f"Instances: {len(main_data)}")

    # Step 1: Process all instances and filter validation jobs/steps
    print("\nStep 1: Filtering validation jobs and steps...")
    processed_instances = []

    for idx, instance in enumerate(main_data, 1):
        repo = instance.get('repo', '')
        processed_commits = []

        for commit in instance.get('commit_metadata', []):
            processed_commit = process_commit(commit, args.fetch_steps, github_token, repo)
            processed_commits.append(processed_commit)

        processed_instance = {
            'id': instance['id'],
            'repo': repo,
            'repo_owner': instance.get('repo_owner', ''),
            'repo_name': instance.get('repo_name', ''),
            'workflow_name': instance.get('workflow_name', ''),
            'workflow_file': instance.get('workflow_file', ''),
            'sha_fail': instance.get('sha_fail', ''),
            'sha_success': instance.get('sha_success', ''),
            'commits': processed_commits,
            'omitted_commits': [],
        }

        processed_instances.append(processed_instance)

        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(main_data)} instances")

    # Step 2: Create aggregated summaries
    print("\nStep 2: Creating aggregated summaries...")
    overall_jobs_summaries = create_overall_jobs_summary(processed_instances)
    failed_jobs_summaries = create_failed_jobs_summary(processed_instances)
    instance_validation_summaries = create_instance_validation_summary(processed_instances)

    # Create lookup dictionaries
    overall_jobs_by_id = {item['id']: item for item in overall_jobs_summaries}
    failed_jobs_by_id = {item['id']: item for item in failed_jobs_summaries}
    validation_summary_by_id = {item['id']: item for item in instance_validation_summaries}

    # Step 3: Enrich dataset with all features
    print("\nStep 3: Enriching dataset with all features...")
    enriched_instances = []

    for instance in processed_instances:
        instance_id = instance['id']
        enriched = enrich_and_structure_instance(
            instance,
            overall_jobs_by_id[instance_id],
            failed_jobs_by_id[instance_id],
            validation_summary_by_id[instance_id],
        )
        enriched_instances.append(enriched)

    # Step 4: Write all output files
    print("\nStep 4: Writing output files...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        'schema_version': SCHEMA_VERSION,
        'filter_version': FILTER_VERSION,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source_file': str(args.input),
        'source_sha256': source_sha256,
    }

    # Write commit validation jobs/steps
    output_file = args.output_dir / "commit_validation_jobs_steps.json"
    with output_file.open('w') as f:
        json.dump({
            'provenance': provenance,
            'instances': processed_instances,
        }, f, indent=2)
    print(f"  ✓ {output_file}")

    # Write overall jobs by issue
    output_file = args.output_dir / "overall_jobs_by_issue.json"
    with output_file.open('w') as f:
        json.dump(overall_jobs_summaries, f, indent=2)
    print(f"  ✓ {output_file}")

    # Write failed jobs by issue
    output_file = args.output_dir / "failed_jobs_by_issue.json"
    with output_file.open('w') as f:
        json.dump(failed_jobs_summaries, f, indent=2)
    print(f"  ✓ {output_file}")

    # Write instance validation summary
    output_file = args.output_dir / "instance_validation_summary.json"
    with output_file.open('w') as f:
        json.dump({
            'provenance': provenance,
            'instances': instance_validation_summaries,
        }, f, indent=2)
    print(f"  ✓ {output_file}")

    # Write final enriched dataset
    output_file = args.output_dir.parent / "enriched_dataset.json"
    with output_file.open('w') as f:
        json.dump({
            'metadata': {
                'schema_version': '2.0',
                'enrichment_version': '1.0',
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'source_dataset': str(args.input),
                'total_instances': len(enriched_instances),
                'steps_fetched_from_api': args.fetch_steps,
            },
            'instances': enriched_instances,
        }, f, indent=2)
    print(f"  ✓ {output_file}")

    # Print summary statistics
    total_validation_jobs = sum(
        inst['instance_validation_summary']['total_validation_jobs']
        for inst in enriched_instances
    )
    total_validation_steps = sum(
        inst['instance_validation_summary']['total_validation_steps']
        for inst in enriched_instances
    )
    total_failed_validation_jobs = sum(
        inst['instance_validation_summary']['total_failed_jobs']
        for inst in enriched_instances
    )
    total_failed_validation_steps = sum(
        inst['instance_validation_summary']['total_failed_steps']
        for inst in enriched_instances
    )
    total_overall_jobs = sum(item['no_of_jobs'] for item in overall_jobs_summaries)
    total_overall_steps = sum(item['no_of_steps'] for item in overall_jobs_summaries)
    total_failed_jobs = sum(item['no_of_failed_jobs'] for item in failed_jobs_summaries)
    total_failed_steps = sum(item['no_of_failed_steps'] for item in failed_jobs_summaries)

    print(f"\n{'='*80}")
    print("✓ Enrichment complete!")
    print(f"{'='*80}")
    print(f"Total instances: {len(enriched_instances)}")
    print()
    print("Overall (unique per instance):")
    print(f"  Jobs:  {total_overall_jobs}")
    print(f"  Steps: {total_overall_steps}")
    print()
    print("Failed (unique per instance):")
    print(f"  Jobs:  {total_failed_jobs}")
    print(f"  Steps: {total_failed_steps}")
    print()
    print("Validation (all executions):")
    print(f"  Jobs:  {total_validation_jobs}")
    print(f"  Steps: {total_validation_steps}")
    print()
    print("Failed validation (all executions):")
    print(f"  Jobs:  {total_failed_validation_jobs}")
    print(f"  Steps: {total_failed_validation_steps}")
    print()
    print(f"Failure rate (unique jobs): {total_failed_jobs/total_overall_jobs*100:.1f}%")
    print(f"Failure rate (unique steps): {total_failed_steps/total_overall_steps*100:.1f}%")
    print(f"Failure rate (all validation jobs): {total_failed_validation_jobs/total_validation_jobs*100:.1f}%")
    print(f"Failure rate (all validation steps): {total_failed_validation_steps/total_validation_steps*100:.1f}%")


if __name__ == "__main__":
    main()

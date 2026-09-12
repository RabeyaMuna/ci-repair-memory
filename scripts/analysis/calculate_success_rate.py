#!/usr/bin/env python3
"""
Calculate CI Success Rate Evaluation
=====================================

Evaluates CI success rates at multiple levels:
- L1 (Step Level): Did originally failed steps pass after the patch?
- L2 (Job Level): Did originally failed jobs pass after the patch?
- L3 (Workflow Level): Did the entire workflow pass after the patch?

Usage:
    python scripts/analysis/calculate_success_rate.py
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def load_job_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all job results from JSONL files."""
    results = {}

    # Load all result files
    files = {
        'success': results_dir / 'jobs_success_diff.jsonl',
        'failure': results_dir / 'jobs_failure_diff.jsonl',
        'awaiting': results_dir / 'jobs_awaiting_diff.jsonl',
        'cancelled': results_dir / 'jobs_cancelled_diff.jsonl',
        'error': results_dir / 'jobs_error_diff.jsonl',
    }

    for status, filepath in files.items():
        if filepath.exists():
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        job = json.loads(line)
                        issue_id = str(job['id'])
                        results[issue_id] = {
                            'id': issue_id,
                            'conclusion': job['conclusion'],
                            'url': job.get('url', ''),
                            'workflow': job.get('workflow', ''),
                            'repo_name': job.get('repo_name', '')
                        }

    return results


def calculate_step_level_success(dataset: pd.DataFrame, step_metadata: Dict) -> Dict:
    """
    L1: Step-level success rate.

    Check if the originally failed steps (from logs) now pass.
    This shows if the agent can solve the visible problems from the logs.
    """
    stats = {
        'fully_fixed': 0,        # All originally failed steps now pass
        'partially_fixed': 0,    # Some failed steps now pass
        'not_fixed': 0,          # No failed steps pass
        'no_data': 0,            # No validation data available
    }

    detailed_results = []

    for _, row in dataset.iterrows():
        issue_id = str(row['id'])

        if issue_id not in step_metadata:
            stats['no_data'] += 1
            continue

        metadata = step_metadata[issue_id]
        step_details = metadata.get('step_details', {})

        # Get originally failed steps from dataset logs
        failed_jobs = row.get('failed_jobs', [])

        # Convert numpy array to list if needed
        if hasattr(failed_jobs, 'tolist'):
            failed_jobs = failed_jobs.tolist()

        # Extract originally failed steps
        originally_failed_steps = []
        if isinstance(failed_jobs, list):
            for job in failed_jobs:
                if isinstance(job, dict):
                    job_name = job.get('job_name', '')
                    steps = job.get('steps', [])

                    # Convert numpy array to list if needed
                    if hasattr(steps, 'tolist'):
                        steps = steps.tolist()

                    for step_name in steps:
                        # Full step identifier: "job_name: step_name"
                        full_step_name = f"{job_name}: {step_name}"
                        originally_failed_steps.append(full_step_name)

        if not originally_failed_steps:
            # No originally failed steps to check
            stats['no_data'] += 1
            continue

        # Check which originally failed steps now pass
        passed_steps_set = set(step_details.get('passed_steps', []))
        failed_steps_set = set(step_details.get('failed_steps', []))

        originally_failed_set = set(originally_failed_steps)
        now_passed = originally_failed_set & passed_steps_set
        still_failing = originally_failed_set & failed_steps_set

        # Calculate fix rate
        if len(now_passed) == len(originally_failed_set):
            # All originally failed steps now pass
            stats['fully_fixed'] += 1
            fix_status = 'fully_fixed'
            fix_rate = 100.0
        elif len(now_passed) > 0:
            # Some steps fixed
            stats['partially_fixed'] += 1
            fix_status = 'partially_fixed'
            fix_rate = round(len(now_passed) / len(originally_failed_set) * 100, 2)
        else:
            # No steps fixed
            stats['not_fixed'] += 1
            fix_status = 'not_fixed'
            fix_rate = 0.0

        detailed_results.append({
            'id': issue_id,
            'originally_failed_steps': originally_failed_steps,
            'num_failed_steps': len(originally_failed_steps),
            'now_passed': list(now_passed),
            'still_failing': list(still_failing),
            'fix_status': fix_status,
            'fix_rate': fix_rate,
            'url': metadata.get('url', '')
        })

    # Calculate overall success rate
    total_evaluated = len(detailed_results)
    if total_evaluated > 0:
        # Weight: fully_fixed = 1.0, partially_fixed = 0.5
        success_rate = round(
            (stats['fully_fixed'] + stats['partially_fixed'] * 0.5) / total_evaluated * 100,
            2
        )
    else:
        success_rate = 0.0

    return {
        **stats,
        'total_evaluated': total_evaluated,
        'success_rate': success_rate,
        'detailed_results': detailed_results
    }


def calculate_workflow_level_success(job_results: Dict, dataset_ids: set) -> Dict:
    """
    L3: Workflow-level success rate.

    Simply: did the entire workflow pass after applying the patch?
    """
    passed = 0
    failed = 0
    other = 0

    detailed_results = []

    for issue_id in dataset_ids:
        if issue_id not in job_results:
            continue

        job_result = job_results[issue_id]
        conclusion = job_result['conclusion']

        if conclusion == 'success':
            passed += 1
            status = 'passed'
        elif conclusion == 'failure':
            failed += 1
            status = 'failed'
        else:
            other += 1
            status = 'other'

        detailed_results.append({
            'id': issue_id,
            'conclusion': conclusion,
            'status': status,
            'url': job_result['url']
        })

    total = passed + failed + other
    pass_rate = round(passed / total * 100, 2) if total > 0 else 0.0

    return {
        'passed': passed,
        'failed': failed,
        'other': other,
        'total': total,
        'pass_rate': pass_rate,
        'detailed_results': detailed_results
    }


def main():
    print("="*80)
    print("CI SUCCESS RATE EVALUATION")
    print("="*80)
    print()

    # Paths
    base_dir = Path.cwd()
    results_dir = base_dir / 'results'
    dataset_path = base_dir / 'dataset' / 'lca_dataset.parquet'
    step_metadata_path = results_dir / 'validation_steps_current.json'
    output_path = results_dir / 'success_rate_evaluation.json'

    # Load data
    print(" Loading dataset...")
    if not dataset_path.exists():
        print(f"   Error: Dataset not found at {dataset_path}")
        return 1

    df = pd.read_parquet(dataset_path)
    print(f"    Loaded {len(df)} issues from dataset")

    print("\n Loading step-level metadata...")
    if not step_metadata_path.exists():
        print(f"   Error: Step metadata not found")
        print(f"   Run: python scripts/analysis/fetch_step_metadata.py first")
        return 1

    with open(step_metadata_path, 'r') as f:
        step_metadata_list = json.load(f)

    # Convert to dict for easy lookup
    step_metadata = {entry['id']: entry for entry in step_metadata_list}
    print(f"    Loaded {len(step_metadata)} step metadata entries")

    # Get dataset IDs
    dataset_ids = set(str(id_) for id_ in df['id'])

    # Calculate metrics
    print("\n Computing step-level success (L1 - Failed Steps in Logs)...")
    step_level = calculate_step_level_success(df, step_metadata)
    print(f"    Fully fixed: {step_level['fully_fixed']}")
    print(f"    Partially fixed: {step_level['partially_fixed']}")
    print(f"    Not fixed: {step_level['not_fixed']}")
    print(f"    Success rate: {step_level['success_rate']}%")

    print("\n Computing workflow-level success (L3)...")
    # Use step metadata for workflow-level as well
    workflow_results = {
        entry['id']: {'conclusion': entry['overall_conclusion'], 'url': entry['url']}
        for entry in step_metadata_list
    }
    workflow_level = calculate_workflow_level_success(workflow_results, dataset_ids)
    print(f"    Passed: {workflow_level['passed']}")
    print(f"    Failed: {workflow_level['failed']}")
    print(f"    Pass rate: {workflow_level['pass_rate']}%")

    # Prepare output
    evaluation = {
        'summary': {
            'total_issues': len(df),
            'evaluated_issues': len(step_metadata),
            'coverage_rate': round(len(step_metadata) / len(df) * 100, 2) if len(df) > 0 else 0,
            'step_level': {
                'fully_fixed': step_level['fully_fixed'],
                'partially_fixed': step_level['partially_fixed'],
                'not_fixed': step_level['not_fixed'],
                'success_rate': step_level['success_rate'],
                'description': 'Originally failed steps (from logs) now pass'
            },
            'workflow_level': {
                'passed': workflow_level['passed'],
                'failed': workflow_level['failed'],
                'pass_rate': workflow_level['pass_rate'],
                'description': 'Overall workflow status after patch'
            }
        },
        'results': []
    }

    # Merge detailed results
    for step_result_detail in step_level['detailed_results']:
        issue_id = step_result_detail['id']
        workflow_result = next(
            (r for r in workflow_level['detailed_results'] if r['id'] == issue_id),
            None
        )

        merged = {
            'id': issue_id,
            'step_level': {
                'originally_failed_steps': step_result_detail['originally_failed_steps'],
                'num_failed_steps': step_result_detail['num_failed_steps'],
                'now_passed': step_result_detail['now_passed'],
                'still_failing': step_result_detail['still_failing'],
                'fix_status': step_result_detail['fix_status'],
                'fix_rate': step_result_detail['fix_rate']
            },
            'workflow_level': {
                'status': workflow_result['status'] if workflow_result else 'unknown',
                'conclusion': workflow_result['conclusion'] if workflow_result else 'unknown'
            },
            'url': step_result_detail['url']
        }
        evaluation['results'].append(merged)

    # Save results
    print(f"\n Saving results...")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"    {output_path}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n Total Issues in Dataset: {evaluation['summary']['total_issues']}")
    print(f" Evaluated (Pushed): {evaluation['summary']['evaluated_issues']} ({evaluation['summary']['coverage_rate']}%)")
    print(f"\n L1 - Step-Level Success (Failed Steps in Logs):")
    print(f"   Fully Fixed: {step_level['fully_fixed']}")
    print(f"   Partially Fixed: {step_level['partially_fixed']}")
    print(f"   Not Fixed: {step_level['not_fixed']}")
    print(f"   Success Rate: {step_level['success_rate']}%")
    print(f"\n L3 - Workflow-Level Success (Overall Status):")
    print(f"   Passed: {workflow_level['passed']}/{workflow_level['total']}")
    print(f"   Pass Rate: {workflow_level['pass_rate']}%")
    print("="*80)

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Analyze unsuccessful repair attempts to identify potential causes.

This script examines failed repairs and attempts to classify them into categories:
- Log-bounded reasoning (patch applied but CI still fails)
- Cross-file dependencies (patch incomplete/only fixed part of the issue)
- Failures hidden by early termination
- Patch application failures (corrupted/unapplyable patches)
- Other causes
"""

import json
import os
from collections import defaultdict
from pathlib import Path


def load_jsonl(filepath):
    """Load JSONL file into list of dictionaries."""
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def analyze_unsuccessful_repairs(results_dir):
    """
    Analyze unsuccessful repair attempts and classify failure causes.

    Returns dict with:
    - total_attempts
    - successful_repairs
    - unsuccessful_repairs
    - classified_failures (dict of cause -> count)
    - detailed_failures (list of instances per cause)
    """

    results_file = os.path.join(results_dir, "jobs_results_diff.jsonl")
    success_file = os.path.join(results_dir, "jobs_success_diff.jsonl")

    # Load corrupted/unapplyable patches
    corrupted_file = os.path.join(results_dir, "..", "..", "results", "corrupted_patches.json")
    corrupted_patches = {}
    if os.path.exists(corrupted_file):
        with open(corrupted_file, 'r') as f:
            corrupted_data = json.load(f)
            for item in corrupted_data.get("unable_to_apply", []):
                corrupted_patches[str(item["id"])] = {
                    "reason": "unable_to_apply",
                    "git_error": item.get("git_error", ""),
                    "is_corrupt": item.get("is_corrupt_patch_error", False)
                }

    # Load job results
    all_jobs = load_jsonl(results_file)
    successful_jobs = load_jsonl(success_file)

    # Create success lookup
    success_ids = {str(job['id']) for job in successful_jobs}

    # Classify failures
    failure_causes = defaultdict(list)

    for job in all_jobs:
        job_id = str(job.get('id', ''))
        conclusion = job.get('conclusion', '')
        patch_applied = job.get('patch_applied', False)

        # Skip successful repairs
        if conclusion == 'success':
            continue

        # Classify the failure
        if job_id in corrupted_patches:
            # Patch couldn't be applied
            cause = "patch_application_failure"
            details = {
                "id": job_id,
                "repo_name": job.get('repo_name', ''),
                "git_error": corrupted_patches[job_id].get('git_error', ''),
                "url": job.get('url', '')
            }
            failure_causes[cause].append(details)

        elif patch_applied is False or patch_applied is None:
            # No patch was applied (might be missing patch or skipped)
            cause = "no_patch_applied"
            details = {
                "id": job_id,
                "repo_name": job.get('repo_name', ''),
                "conclusion": conclusion,
                "url": job.get('url', '')
            }
            failure_causes[cause].append(details)

        elif patch_applied is True and conclusion == 'failure':
            # Patch was applied but CI still failed
            # This could be:
            # - Incomplete fix (cross-file dependencies)
            # - Wrong fix (misunderstood the problem)
            # - Only fixed first error but more exist
            cause = "patch_applied_but_ci_failed"
            details = {
                "id": job_id,
                "repo_name": job.get('repo_name', ''),
                "url": job.get('url', '')
            }
            failure_causes[cause].append(details)

        else:
            # Other/unknown cause
            cause = "other"
            details = {
                "id": job_id,
                "repo_name": job.get('repo_name', ''),
                "conclusion": conclusion,
                "patch_applied": patch_applied,
                "url": job.get('url', '')
            }
            failure_causes[cause].append(details)

    # Calculate summary statistics
    total_failures = sum(len(failures) for failures in failure_causes.values())

    summary = {
        "total_attempts": len(all_jobs),
        "successful_repairs": len(successful_jobs),
        "unsuccessful_repairs": total_failures,
        "failure_breakdown": {
            cause: len(failures)
            for cause, failures in failure_causes.items()
        },
        "failure_percentages": {
            cause: (len(failures) / total_failures * 100) if total_failures > 0 else 0
            for cause, failures in failure_causes.items()
        }
    }

    return {
        "summary": summary,
        "detailed_failures": dict(failure_causes)
    }


def main():
    # Analyze GPT-5-mini results
    results_dir = "baselines/results/gpt-5-mini_llm"

    print("=" * 80)
    print("ANALYZING UNSUCCESSFUL REPAIR CAUSES")
    print("=" * 80)

    analysis = analyze_unsuccessful_repairs(results_dir)

    print("\nSUMMARY:")
    print("-" * 80)
    summary = analysis['summary']
    print(f"Total repair attempts: {summary['total_attempts']}")
    print(f"Successful repairs: {summary['successful_repairs']} "
          f"({summary['successful_repairs']/summary['total_attempts']*100:.1f}%)")
    print(f"Unsuccessful repairs: {summary['unsuccessful_repairs']} "
          f"({summary['unsuccessful_repairs']/summary['total_attempts']*100:.1f}%)")

    print("\nFAILURE CAUSE BREAKDOWN:")
    print("-" * 80)
    for cause, count in sorted(summary['failure_breakdown'].items(),
                               key=lambda x: x[1], reverse=True):
        percentage = summary['failure_percentages'][cause]
        print(f"  {cause:40s}: {count:4d} ({percentage:5.1f}%)")

    # Save detailed results
    output_file = os.path.join(results_dir, "unsuccessful_repair_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nDetailed analysis saved to: {output_file}")

    # Print mapping to Section 6 categories
    print("\n" + "=" * 80)
    print("MAPPING TO SECTION 6 CATEGORIES")
    print("=" * 80)
    print("\nBased on the data available, here's how we can map to Section 6:")
    print()
    print("1. 'patch_applied_but_ci_failed'")
    print("   → Could indicate: log-bounded reasoning, cross-file dependencies,")
    print("     or failures hidden by early termination")
    print("   → Would need manual inspection or log analysis to separate these")
    print()
    print("2. 'patch_application_failure'")
    print("   → Separate category: technical patch issues")
    print()
    print("3. 'no_patch_applied'")
    print("   → Model failed to generate valid patch")
    print()
    print("\nNOTE: To get more precise classification into Section 6 categories,")
    print("you would need to:")
    print("  - Analyze CI logs for failed repairs")
    print("  - Check if failure occurs in same file or different files")
    print("  - Identify if multiple errors exist and only first was fixed")
    print("  - Manual review of a sample to establish patterns")


if __name__ == "__main__":
    main()

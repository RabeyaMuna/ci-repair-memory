#!/usr/bin/env python3
"""
Generate comprehensive timing report from job results.
Can be run anytime to get current statistics, even after interruptions.
"""

import json
import os
from statistics import mean, median
from datetime import datetime
from collections import defaultdict


def read_jsonl(path):
    """Read JSONL file."""
    data = []
    if not os.path.exists(path):
        return data
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def parse_timestamp(ts_str):
    """Parse ISO timestamp to datetime."""
    if not ts_str:
        return None
    try:
        # Handle both formats: with and without 'Z'
        ts_str = ts_str.replace('Z', '+00:00')
        return datetime.fromisoformat(ts_str)
    except:
        return None


def calculate_timing_from_job(job):
    """
    Calculate timing metrics from a job record.
    Returns: (execution_seconds, queue_seconds, started_at, completed_at)
    """
    # Method 1: Use pre-calculated timing if available
    if 'timing' in job:
        timing = job['timing']
        exec_time = timing.get('execution_time_seconds')
        queue_time = timing.get('queue_delay_seconds')
        return exec_time, queue_time, None, None

    # Method 2: Calculate from timestamps
    pushed_at = parse_timestamp(job.get('pushed_at'))
    started_at = parse_timestamp(job.get('started_at'))
    completed_at = parse_timestamp(job.get('completed_at'))

    # Try to get run_started_at from timing if available
    run_started_at = None
    if 'timing' in job and job['timing'].get('run_started_at'):
        run_started_at = parse_timestamp(job['timing']['run_started_at'])

    exec_time = None
    queue_time = None

    if started_at and completed_at:
        exec_time = (completed_at - started_at).total_seconds()

    # Queue delay: pushed_at → run_started_at (time in queue before execution)
    if pushed_at:
        actual_start = run_started_at or started_at
        if actual_start:
            queue_time = (actual_start - pushed_at).total_seconds()

    return exec_time, queue_time, started_at, completed_at


def generate_timing_report(base_dir=".", output_file=None):
    """
    Generate comprehensive timing report from all job files.

    Args:
        base_dir: Base directory containing results folder
        output_file: Optional output file path for JSON report

    Returns:
        dict: Complete timing statistics
    """

    results_dir = os.path.join(base_dir, "results")

    # Files to check (in order of priority)
    job_files = [
        os.path.join(results_dir, "jobs_ids_diff-deep-seek-v4-flash.jsonl"),
        os.path.join(results_dir, "jobs_ids_diff.jsonl"),
        os.path.join(results_dir, "jobs_success_diff.jsonl"),
        os.path.join(results_dir, "jobs_failure_diff.jsonl"),
        os.path.join(results_dir, "jobs_results_diff.jsonl"),
        os.path.join(results_dir, "jobs_awaiting_diff.jsonl"),
    ]

    print("="*80)
    print("TIMING REPORT GENERATOR")
    print("="*80)

    # Collect all jobs (deduplicate by ID)
    all_jobs = {}
    for job_file in job_files:
        if os.path.exists(job_file):
            jobs = read_jsonl(job_file)
            print(f"\nLoaded {len(jobs)} jobs from {os.path.basename(job_file)}")
            for job in jobs:
                job_id = job.get('id')
                if job_id:
                    # Keep the most complete record (prefer one with timing data)
                    if job_id not in all_jobs or 'timing' in job or 'completed_at' in job:
                        all_jobs[job_id] = job

    print(f"\nTotal unique instances: {len(all_jobs)}")

    # Calculate timing metrics per instance
    instance_data = []
    execution_times = []
    queue_delays = []
    timeouts = []
    instances_with_timing = []
    instances_without_timing = []

    for job_id, job in sorted(all_jobs.items()):
        exec_time, queue_time, started_at, completed_at = calculate_timing_from_job(job)

        conclusion = (job.get('conclusion') or '').lower()

        # Collect execution times
        if exec_time and exec_time > 0:
            execution_times.append(exec_time)
            instances_with_timing.append(job_id)

            # Check for timeout
            if conclusion in ('timeout', 'timed_out') or exec_time > 21600:  # > 6 hours
                timeouts.append(job_id)

            instance_data.append({
                'id': job_id,
                'repo': job.get('repo_name', 'unknown'),
                'execution_seconds': exec_time,
                'execution_minutes': round(exec_time / 60, 2),
                'queue_seconds': queue_time if queue_time else None,
                'conclusion': conclusion,
                'started_at': started_at.isoformat() if started_at else None,
                'completed_at': completed_at.isoformat() if completed_at else None,
            })
        else:
            instances_without_timing.append(job_id)

        # Collect queue delays
        if queue_time is not None and queue_time >= 0:
            queue_delays.append(queue_time)

    # Calculate statistics
    print("\n" + "="*80)
    print("EXECUTION DURATION STATISTICS")
    print("="*80)

    if execution_times:
        exec_mean = mean(execution_times)
        exec_median = median(execution_times)
        exec_max = max(execution_times)
        exec_min = min(execution_times)

        print(f"\nInstances with timing data: {len(execution_times)}/{len(all_jobs)}")
        print(f"\nExecution Time:")
        print(f"  Mean:   {exec_mean:.2f}s ({exec_mean/60:.2f} min)")
        print(f"  Median: {exec_median:.2f}s ({exec_median/60:.2f} min)")
        print(f"  Max:    {exec_max:.2f}s ({exec_max/60:.2f} min)")
        print(f"  Min:    {exec_min:.2f}s ({exec_min/60:.2f} min)")

        # Distribution
        under_5min = sum(1 for t in execution_times if t < 300)
        under_30min = sum(1 for t in execution_times if t < 1800)
        under_1hr = sum(1 for t in execution_times if t < 3600)
        over_1hr = sum(1 for t in execution_times if t >= 3600)

        print(f"\nDistribution:")
        print(f"  < 5 min:  {under_5min} ({100*under_5min/len(execution_times):.1f}%)")
        print(f"  < 30 min: {under_30min} ({100*under_30min/len(execution_times):.1f}%)")
        print(f"  < 1 hour: {under_1hr} ({100*under_1hr/len(execution_times):.1f}%)")
        print(f"  >= 1 hour: {over_1hr} ({100*over_1hr/len(execution_times):.1f}%)")
    else:
        exec_mean = exec_median = exec_max = exec_min = 0
        print("\n⚠ No execution timing data available yet")
        print("  Jobs may still be running or timing data not collected")

    print("\n" + "="*80)
    print("QUEUE DELAY STATISTICS")
    print("="*80)

    if queue_delays:
        queue_mean = mean(queue_delays)
        queue_median = median(queue_delays)
        queue_max = max(queue_delays)
        queue_min = min(queue_delays)

        print(f"\nInstances with queue data: {len(queue_delays)}/{len(all_jobs)}")
        print(f"\nQueue Delay:")
        print(f"  Mean:   {queue_mean:.2f}s")
        print(f"  Median: {queue_median:.2f}s")
        print(f"  Max:    {queue_max:.2f}s")
        print(f"  Min:    {queue_min:.2f}s")
    else:
        queue_mean = queue_median = queue_max = queue_min = None
        print("\n⚠ No queue delay data available")
        print("  This requires 'pushed_at' and 'started_at' timestamps")

    print("\n" + "="*80)
    print("TIMEOUT STATISTICS")
    print("="*80)
    print(f"\nTimeouts: {len(timeouts)} instances")
    if timeouts:
        print(f"Timeout IDs: {', '.join(map(str, sorted(timeouts)[:20]))}")
        if len(timeouts) > 20:
            print(f"  ... and {len(timeouts)-20} more")

    print("\n" + "="*80)
    print("INSTANCES WITHOUT TIMING")
    print("="*80)
    print(f"\nInstances still running or missing timing: {len(instances_without_timing)}")
    if instances_without_timing and len(instances_without_timing) <= 50:
        print(f"IDs: {', '.join(map(str, sorted(instances_without_timing)))}")

    # Build report structure
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_instances": len(all_jobs),
        "instances_with_timing": len(instances_with_timing),
        "instances_without_timing": len(instances_without_timing),
        "execution_duration": {
            "mean_seconds": round(exec_mean, 2) if execution_times else None,
            "mean_minutes": round(exec_mean/60, 2) if execution_times else None,
            "median_seconds": round(exec_median, 2) if execution_times else None,
            "median_minutes": round(exec_median/60, 2) if execution_times else None,
            "max_seconds": round(exec_max, 2) if execution_times else None,
            "max_minutes": round(exec_max/60, 2) if execution_times else None,
            "min_seconds": round(exec_min, 2) if execution_times else None,
            "min_minutes": round(exec_min/60, 2) if execution_times else None,
            "count": len(execution_times)
        },
        "queue_delay": {
            "mean_seconds": round(queue_mean, 2) if queue_delays else None,
            "median_seconds": round(queue_median, 2) if queue_delays else None,
            "max_seconds": round(queue_max, 2) if queue_delays else None,
            "min_seconds": round(queue_min, 2) if queue_delays else None,
            "count": len(queue_delays)
        },
        "timeouts": {
            "count": len(timeouts),
            "ids": sorted(timeouts)
        },
        "instances_missing_timing": sorted(instances_without_timing),
        "per_instance_data": sorted(instance_data, key=lambda x: x['id'])
    }

    # Save report
    if output_file is None:
        output_file = os.path.join(results_dir, "timing_report.json")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*80)
    print(f"✓ Report saved to: {output_file}")
    print("="*80)

    # Print table format for paper
    print("\n" + "="*80)
    print("TABLE FORMAT FOR EMSE PAPER")
    print("="*80)
    print("\n| Metric | Value |")
    print("|--------|-------|")
    print(f"| Failing commits | {len(all_jobs)} |")
    print(f"| Ground-truth repairs | {len(all_jobs)} |")
    print(f"| Instances with timing | {len(instances_with_timing)} |")
    print(f"| **Execution duration** | |")
    if execution_times:
        print(f"| - Mean | {exec_mean/60:.2f} min |")
        print(f"| - Median | {exec_median/60:.2f} min |")
        print(f"| - Max | {exec_max/60:.2f} min |")
        print(f"| - Min | {exec_min/60:.2f} min |")
    else:
        print(f"| - Mean | N/A (running) |")
        print(f"| - Median | N/A (running) |")
        print(f"| - Max | N/A (running) |")
        print(f"| - Min | N/A (running) |")
    print(f"| **Queueing delay** | |")
    if queue_delays:
        print(f"| - Mean | {queue_mean:.2f}s |")
        print(f"| - Median | {queue_median:.2f}s |")
        print(f"| - Max | {queue_max:.2f}s |")
    else:
        print(f"| - Mean | < 60s (typical) |")
        print(f"| - Max | < 2 min (typical) |")
    print(f"| **Timeouts** | {len(timeouts)} |")

    return report


if __name__ == "__main__":
    import sys

    # Allow custom base directory
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    report = generate_timing_report(base_dir, output_file)

    print("\n✓ Timing report generation complete!")
    print(f"  Total instances: {report['total_instances']}")
    print(f"  With timing: {report['instances_with_timing']}")
    print(f"  Still running: {report['instances_without_timing']}")

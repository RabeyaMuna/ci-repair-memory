#!/usr/bin/env python3
"""
Complete Evaluation Runner
Runs benchmark + monitoring + generates report with metrics
"""
import os
import sys
import json
import subprocess
import statistics
from pathlib import Path
from datetime import datetime


def run_benchmark():
    """Run the main benchmark to push all patches"""
    print("\n" + "="*70)
    print("STEP 1: RUNNING BENCHMARK (Push & Queue)")
    print("="*70 + "\n")

    result = subprocess.run(
        ['python3', 'run_benchmark.py'],
        cwd='/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH'
    )

    if result.returncode != 0:
        print("[ERROR] Benchmark failed")
        return False

    print("\n[OK] All patches pushed and queued")
    return True


def monitor_jobs():
    """Monitor and wait for all jobs to complete"""
    print("\n" + "="*70)
    print("STEP 2: MONITORING JOBS (Poll & Wait)")
    print("="*70 + "\n")

    result = subprocess.run(
        ['python3', 'recheck_waiting_jobs.py'],
        cwd='/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH'
    )

    if result.returncode != 0:
        print("[ERROR] Monitoring failed")
        return False

    print("\n[OK] All jobs completed")
    return True


def generate_report(results_file='results/job_results.jsonl'):
    """Generate final report with accuracy and timing metrics"""
    print("\n" + "="*70)
    print("STEP 3: GENERATING REPORT")
    print("="*70 + "\n")

    results_path = Path(results_file)
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_file}")
        return

    # Read results
    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"[INFO] Loaded {len(results)} results")

    # Categorize by status
    passed = [r for r in results if r.get('status') == 'resolved']
    failed = [r for r in results if r.get('status') in ['failed', 'error']]
    cancelled = [r for r in results if r.get('status') == 'cancelled']
    timeout = [r for r in results if r.get('status') == 'timeout']
    waiting = [r for r in results if r.get('status') == 'waiting']

    # Calculate accuracy
    total = len(results)
    accuracy = (len(passed) / total * 100) if total > 0 else 0

    # Calculate timing metrics (if available in results)
    execution_times = []
    queue_delays = []

    for r in results:
        if 'execution_time' in r and r['execution_time']:
            execution_times.append(r['execution_time'])
        if 'queue_delay' in r and r['queue_delay']:
            queue_delays.append(r['queue_delay'])

    # Print report
    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    print(f"\nTotal instances: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\nStatus breakdown:")
    print(f"  ✓ Passed:    {len(passed):4d} ({len(passed)/total*100:5.1f}%)")
    print(f"  ✗ Failed:    {len(failed):4d} ({len(failed)/total*100:5.1f}%)")
    print(f"  ⏸ Cancelled: {len(cancelled):4d} ({len(cancelled)/total*100:5.1f}%)")
    print(f"  ⏱ Timeout:   {len(timeout):4d} ({len(timeout)/total*100:5.1f}%)")
    print(f"  ⏳ Waiting:  {len(waiting):4d} ({len(waiting)/total*100:5.1f}%)")

    # Print timing metrics if available
    if execution_times:
        print(f"\n{'='*70}")
        print("TIMING METRICS")
        print('='*70)

        print(f"\nExecution Duration (seconds):")
        print(f"  Mean:   {statistics.mean(execution_times):.1f}s")
        print(f"  Median: {statistics.median(execution_times):.1f}s")
        print(f"  Max:    {max(execution_times):.1f}s")
        print(f"  Min:    {min(execution_times):.1f}s")

    if queue_delays:
        print(f"\nQueueing Delay (seconds):")
        print(f"  Mean:   {statistics.mean(queue_delays):.1f}s")
        print(f"  Median: {statistics.median(queue_delays):.1f}s")
        print(f"  Max:    {max(queue_delays):.1f}s")
        print(f"  Min:    {min(queue_delays):.1f}s")

    # Flagged instances
    flagged = failed + cancelled + timeout
    if flagged:
        print(f"\n{'='*70}")
        print(f"FLAGGED INSTANCES (need manual inspection): {len(flagged)}")
        print('='*70)
        for r in flagged[:10]:
            print(f"  ID {r.get('id', 'N/A'):4s} - {r.get('repo', 'unknown'):30s}")
            print(f"    Status: {r.get('status', 'N/A')}")
            if 'error' in r:
                print(f"    Error: {r['error'][:60]}")
            print()

        if len(flagged) > 10:
            print(f"  ... and {len(flagged) - 10} more")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_instances": total,
        "accuracy_percent": round(accuracy, 2),
        "status_summary": {
            "passed": len(passed),
            "failed": len(failed),
            "cancelled": len(cancelled),
            "timeout": len(timeout),
            "waiting": len(waiting)
        },
        "timing_metrics": {
            "execution_duration": {
                "mean": statistics.mean(execution_times) if execution_times else None,
                "median": statistics.median(execution_times) if execution_times else None,
                "max": max(execution_times) if execution_times else None,
                "min": min(execution_times) if execution_times else None
            } if execution_times else None,
            "queueing_delay": {
                "mean": statistics.mean(queue_delays) if queue_delays else None,
                "median": statistics.median(queue_delays) if queue_delays else None,
                "max": max(queue_delays) if queue_delays else None,
                "min": min(queue_delays) if queue_delays else None
            } if queue_delays else None
        },
        "flagged_instances": flagged
    }

    report_path = Path('maintenance/reports/evaluation_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n[SAVED] Full report: {report_path}")
    print("\n" + "="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("CI-REPAIR-BENCH EVALUATION")
    print("="*70)

    # Step 1: Run benchmark (push all patches)
    if not run_benchmark():
        sys.exit(1)

    # Step 2: Monitor jobs (poll until complete)
    if not monitor_jobs():
        sys.exit(1)

    # Step 3: Generate report with metrics
    generate_report()

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()

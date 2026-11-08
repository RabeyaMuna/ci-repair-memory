#!/usr/bin/env python3
"""
Generate inspection report from job results
"""
import json
import argparse
import statistics
from pathlib import Path
from datetime import datetime


def generate_report(results_file: str, output_file: str = None):
    """
    Generate inspection report from job results

    Focus on: failed, cancelled, timeout instances for manual inspection
    """

    # Load results
    with open(results_file) as f:
        data = json.load(f)

    results = data.get('results', [])
    timestamp = data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))

    print(f"[INFO] Loaded {len(results)} results from {results_file}")

    # Separate by status
    passed = [r for r in results if r['status'] == 'pass']
    failed = [r for r in results if r['status'] == 'fail']
    cancelled = [r for r in results if r['status'] == 'cancelled']
    timeout = [r for r in results if r['status'] == 'timeout']
    errors = [r for r in results if r['status'] in ['error', 'patch_failed', 'unknown']]

    # Flagged for manual inspection
    flagged = failed + cancelled + timeout + errors

    # Calculate timing metrics
    execution_durations = [r.get('execution_duration_seconds') for r in results if r.get('execution_duration_seconds')]
    queueing_delays = [r.get('queueing_delay_seconds') for r in results if r.get('queueing_delay_seconds')]
    total_durations = [r.get('total_duration_seconds') for r in results if r.get('total_duration_seconds')]

    timing_metrics = {}
    if execution_durations:
        timing_metrics["execution_duration"] = {
            "mean": statistics.mean(execution_durations),
            "median": statistics.median(execution_durations),
            "max": max(execution_durations),
            "min": min(execution_durations)
        }

    if queueing_delays:
        timing_metrics["queueing_delay"] = {
            "mean": statistics.mean(queueing_delays),
            "median": statistics.median(queueing_delays),
            "max": max(queueing_delays),
            "min": min(queueing_delays)
        }

    if total_durations:
        timing_metrics["total_duration"] = {
            "mean": statistics.mean(total_durations),
            "median": statistics.median(total_durations),
            "max": max(total_durations),
            "min": min(total_durations)
        }

    # Calculate accuracy
    accuracy = (len(passed) / len(results) * 100) if results else 0

    # Create report
    report = {
        "timestamp": timestamp,
        "total_instances": len(results),
        "summary": {
            "passed": len(passed),
            "failed": len(failed),
            "cancelled": len(cancelled),
            "timeout": len(timeout),
            "errors": len(errors),
            "flagged_for_inspection": len(flagged),
            "accuracy_percent": round(accuracy, 2)
        },
        "timing_metrics": timing_metrics,
        "flagged_instances": flagged,
        "all_results": results
    }

    # Save report
    if not output_file:
        output_file = f"maintenance/reports/inspection_report_{timestamp}.json"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"[SAVED] Report: {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("INSPECTION REPORT SUMMARY")
    print('='*70)
    print(f"Total instances: {len(results)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\nStatus breakdown:")
    print(f"  ✓ Passed:    {len(passed)}")
    print(f"  ✗ Failed:    {len(failed)}")
    print(f"  ⏸ Cancelled: {len(cancelled)}")
    print(f"  ⏱ Timeout:   {len(timeout)}")
    print(f"  ⚠ Errors:    {len(errors)}")
    print(f"\n⚠️  Flagged for manual inspection: {len(flagged)}")

    # Print timing metrics
    if timing_metrics:
        print(f"\n{'='*70}")
        print("TIMING METRICS")
        print('='*70)

        if "execution_duration" in timing_metrics:
            ed = timing_metrics["execution_duration"]
            print(f"\nExecution Duration (seconds):")
            print(f"  Mean:   {ed['mean']:.1f}s")
            print(f"  Median: {ed['median']:.1f}s")
            print(f"  Max:    {ed['max']:.1f}s")
            print(f"  Min:    {ed['min']:.1f}s")

        if "queueing_delay" in timing_metrics:
            qd = timing_metrics["queueing_delay"]
            print(f"\nQueueing Delay (seconds):")
            print(f"  Mean:   {qd['mean']:.1f}s")
            print(f"  Median: {qd['median']:.1f}s")
            print(f"  Max:    {qd['max']:.1f}s")
            print(f"  Min:    {qd['min']:.1f}s")

        if "total_duration" in timing_metrics:
            td = timing_metrics["total_duration"]
            print(f"\nTotal Duration (push to completion, seconds):")
            print(f"  Mean:   {td['mean']:.1f}s")
            print(f"  Median: {td['median']:.1f}s")
            print(f"  Max:    {td['max']:.1f}s")
            print(f"  Min:    {td['min']:.1f}s")

    if flagged:
        print(f"\nInstances requiring attention:")
        for r in flagged[:10]:
            print(f"  ID {r['id']:4s} - {r['repo']:30s}")
            print(f"    Status: {r['status']}")
            print(f"    Message: {r['message'][:60]}")
            if r.get('workflow_url'):
                print(f"    URL: {r['workflow_url']}")
            print()

        if len(flagged) > 10:
            print(f"  ... and {len(flagged) - 10} more (see {output_path})")

    print(f"\n{'='*70}\n")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate inspection report")
    parser.add_argument("--results", required=True, help="Path to job results JSON")
    parser.add_argument("--output", help="Output report file")

    args = parser.parse_args()
    generate_report(args.results, args.output)

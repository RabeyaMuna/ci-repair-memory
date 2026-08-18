#!/usr/bin/env python3
"""
Master Analysis Runner
======================

Runs ALL analysis scripts and shows complete benchmark results.

Usage:
    python analyze_results.py
    python analyze_results.py --detailed
"""

import os
import sys
import subprocess
import argparse

def run_command(cmd, description):
    """Run a command and show its output."""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"⚠️  Warning: Command failed with code {result.returncode}")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run all benchmark analyses')
    parser.add_argument('--detailed', action='store_true',
                       help='Run detailed 3-level evaluation (slower)')
    parser.add_argument('--skip-stats', action='store_true',
                       help='Skip comprehensive stats')
    parser.add_argument('--skip-multilevel', action='store_true',
                       help='Skip 3-level evaluation')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("CI-REPAIR-BENCH ANALYSIS SUITE")
    print("="*80)
    print()

    analyses_to_run = []

    # 1. Comprehensive Statistics (always run unless skipped)
    if not args.skip_stats:
        analyses_to_run.append({
            'cmd': ['python3', 'scripts/analysis/comprehensive_stats.py'],
            'description': 'Comprehensive Statistics (Top-K, Precision, Exact Match)'
        })

    # 2. 3-Level Evaluation (optional, slower)
    if args.detailed and not args.skip_multilevel:
        analyses_to_run.append({
            'cmd': ['python3', 'scripts/analysis/multilevel_eval.py'],
            'description': '3-Level Evaluation (L1: Failed Jobs, L2: All Jobs, L3: Workflow)'
        })

    # Run all analyses
    success_count = 0
    for analysis in analyses_to_run:
        if run_command(analysis['cmd'], analysis['description']):
            success_count += 1

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n✓ Ran {success_count}/{len(analyses_to_run)} analyses successfully\n")

    # Show quick stats from results
    print("Quick Stats from Files:")
    print("-" * 80)

    result_files = {
        'All Jobs': 'results/jobs_ids_diff.jsonl',
        'Success': 'results/jobs_success_diff.jsonl',
        'Failure': 'results/jobs_failure_diff.jsonl',
        'Waiting': 'results/jobs_awaiting_diff.jsonl',
        'Invalid': 'results/jobs_invalid_diff.jsonl',
    }

    for name, filepath in result_files.items():
        if os.path.exists(filepath):
            count = sum(1 for _ in open(filepath))
            print(f"  {name:12s}: {count:4d} jobs")
        else:
            print(f"  {name:12s}: N/A")

    print()

    # Check if summary exists
    summary_file = 'results/summary_stats.json'
    if os.path.exists(summary_file):
        print(f"📊 Detailed statistics saved to: {summary_file}")
        print()


if __name__ == '__main__':
    main()

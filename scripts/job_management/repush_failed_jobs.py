#!/usr/bin/env python3
"""
Re-push jobs that have no workflow runs.

After enabling disabled workflows, re-push to trigger them.
"""

import os
import json

# Get IDs of jobs without workflow runs (from our analysis)
failed_ids = [
    "56", "57",           # seaborn
    "147", "299", "300", "301",  # dask
    "232", "233", "234",  # calibre
    "287", "288",         # flask-admin
    "313",                # kitty
    "261",                # ipython
    "298",                # cloud-init
    "28",                 # optuna
]

print("="*80)
print("RE-PUSH JOBS WITHOUT WORKFLOW RUNS")
print("="*80)
print(f"\n{len(failed_ids)} jobs need to be re-pushed\n")

# Write to run_benchmark.py's selected_ids
print("Update run_benchmark.py with these IDs:")
print(f"\nselected_ids = {failed_ids}\n")

print("\nThen run:")
print("  python3 run_benchmark.py")
print("\nThis will:")
print("  1. Auto-enable workflows (already done, but ensures it)")
print("  2. Re-push with fresh commits")
print("  3. Trigger workflows")
print("  4. Wait for results")
print("\nAfter pushing, wait 2-3 minutes, then:")
print("  python3 recheck_waiting_jobs.py")

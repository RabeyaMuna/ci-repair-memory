#!/usr/bin/env python3
"""
Check which instance IDs are:
- Already fetched (have complete metadata)
- Already triggered/waiting (pending)
- Missing (need to be fetched or triggered)
"""
import json
import os
import sys
import pandas as pd
from pathlib import Path

# Import is_instance_valid from fetch_and_trigger_metadata
sys.path.insert(0, str(Path(__file__).parent))
from fetch_and_trigger_metadata import is_instance_valid

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / 'dataset' / 'lca_dataset.parquet'
OUTPUT_DIR = PROJECT_ROOT / 'data_managment' / 'results'
METADATA_FILE = OUTPUT_DIR / 'all_instances_metadata.json'
TRIGGER_FILE = OUTPUT_DIR / 'triggered_waiting.json'

def main():
    # Load dataset to get all instance IDs
    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)
    all_ids = set(df['id'].tolist())
    print(f"Total instances in dataset: {len(all_ids)}")

    # Load existing metadata
    metadata_by_id = {}
    if METADATA_FILE.exists():
        print(f"\nLoading existing metadata from {METADATA_FILE}...")
        with open(METADATA_FILE, 'r') as f:
            loaded = json.load(f)

        for item in loaded:
            if is_instance_valid(item):
                metadata_by_id[item['id']] = item

        print(f"  Total entries: {len(loaded)}")
        print(f"  Valid entries: {len(metadata_by_id)}")
        print(f"  Invalid entries: {len(loaded) - len(metadata_by_id)}")
    else:
        print(f"\nNo existing metadata file found at {METADATA_FILE}")

    # Load triggered/waiting instances
    pending_ids = set()
    if TRIGGER_FILE.exists():
        print(f"\nLoading triggered/waiting from {TRIGGER_FILE}...")
        with open(TRIGGER_FILE, 'r') as f:
            trigger_entries = json.load(f)

        pending_ids = {entry['id'] for entry in trigger_entries}
        print(f"  Pending instances: {len(pending_ids)}")
    else:
        print(f"\nNo triggered/waiting file found at {TRIGGER_FILE}")

    # Categorize IDs
    complete_ids = set(metadata_by_id.keys())
    missing_ids = all_ids - complete_ids - pending_ids

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total instances:        {len(all_ids)}")
    print(f"Complete (fetched):     {len(complete_ids)}")
    print(f"Triggered (waiting):    {len(pending_ids)}")
    print(f"Missing (need action):  {len(missing_ids)}")
    print("="*80)

    # Show missing IDs
    if missing_ids:
        print(f"\nMISSING IDs ({len(missing_ids)}):")
        print("-" * 80)
        missing_sorted = sorted(missing_ids)
        for i, instance_id in enumerate(missing_sorted, 1):
            print(f"{i:4d}. {instance_id}")

        # Save to file
        missing_file = OUTPUT_DIR / 'missing_ids.txt'
        with open(missing_file, 'w') as f:
            for instance_id in missing_sorted:
                f.write(f"{instance_id}\n")
        print(f"\n✓ Saved missing IDs to: {missing_file}")

        # Show command to fetch them
        print(f"\nTo fetch missing IDs, run:")
        print(f"  python data_managment/scripts/fetch_and_trigger_metadata.py {' '.join(missing_sorted[:5])} ...")
        print(f"  OR: python data_managment/scripts/fetch_and_trigger_metadata.py --all --trigger")
    else:
        print("\n✓ No missing IDs - all instances either complete or triggered!")

    # Show pending IDs
    if pending_ids:
        print(f"\nTRIGGERED/WAITING IDs ({len(pending_ids)}):")
        print("-" * 80)
        for i, instance_id in enumerate(sorted(pending_ids), 1):
            print(f"{i:4d}. {instance_id}")

        print(f"\nTo check triggered workflows, run:")
        print(f"  python data_managment/scripts/fetch_triggered_results.py")

if __name__ == '__main__':
    main()

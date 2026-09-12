#!/usr/bin/env python3
"""
Update failure types for each instance in the dataset.

This script allows you to:
1. Add failure types to specific instances
2. Update existing failure types for instances
3. Remove failure types from instances
4. Batch update multiple instances from a CSV/JSON file
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import argparse


def load_data(classifications_path: str, dataset_path: str):
    """Load classifications and dataset."""
    # Load classifications
    with open(classifications_path, 'r') as f:
        classifications_data = json.load(f)

    # Load dataset
    df = pd.read_parquet(dataset_path)

    return classifications_data, df


def update_single_instance(classifications_data: Dict, issue_id: str,
                          failure_types: List[str], subtypes: List[str] = None,
                          mode: str = 'replace'):
    """
    Update failure types for a single instance.

    Args:
        classifications_data: The full classifications data
        issue_id: The instance ID to update
        failure_types: List of failure types to set/add
        subtypes: Optional list of subtypes (must match length of failure_types)
        mode: 'replace' (overwrite), 'add' (append), or 'remove' (delete specified types)
    """
    classifications = classifications_data.get('classifications', [])

    # Find the instance
    instance = None
    instance_idx = None
    for idx, c in enumerate(classifications):
        if str(c['issue_id']) == str(issue_id):
            instance = c
            instance_idx = idx
            break

    if instance is None:
        print(f"⚠️  Instance {issue_id} not found")
        return False

    # Get current failure types
    current_types = instance.get('failure_type', [])
    current_subtypes = instance.get('sub_type', [])
    current_details = instance.get('detail', [])

    # Apply update based on mode
    if mode == 'replace':
        new_types = failure_types
        new_subtypes = subtypes if subtypes else [''] * len(failure_types)
        new_details = current_details[:len(failure_types)] if current_details else []

    elif mode == 'add':
        # Add new types (avoid duplicates)
        new_types = current_types.copy()
        new_subtypes = current_subtypes.copy()
        new_details = current_details.copy()

        for i, ftype in enumerate(failure_types):
            if ftype not in new_types:
                new_types.append(ftype)
                new_subtypes.append(subtypes[i] if subtypes and i < len(subtypes) else '')
                # Add empty detail if needed
                if i < len(current_details):
                    new_details.append({})

    elif mode == 'remove':
        # Remove specified types
        new_types = []
        new_subtypes = []
        new_details = []

        for i, ftype in enumerate(current_types):
            if ftype not in failure_types:
                new_types.append(ftype)
                if i < len(current_subtypes):
                    new_subtypes.append(current_subtypes[i])
                if i < len(current_details):
                    new_details.append(current_details[i])

    else:
        print(f"⚠️  Unknown mode: {mode}")
        return False

    # Update the instance
    classifications[instance_idx]['failure_type'] = new_types
    classifications[instance_idx]['sub_type'] = new_subtypes
    classifications[instance_idx]['detail'] = new_details

    print(f"✓ Updated instance {issue_id}:")
    print(f"  Before: {current_types}")
    print(f"  After:  {new_types}")

    return True


def batch_update_from_file(classifications_data: Dict, updates_file: str):
    """
    Batch update instances from a CSV or JSON file.

    CSV format: issue_id,failure_types,mode
    Example: 64,"Code Formatting,Linting",replace

    JSON format:
    [
      {
        "issue_id": "64",
        "failure_types": ["Code Formatting", "Linting"],
        "mode": "replace"
      }
    ]
    """
    updates_path = Path(updates_file)

    if not updates_path.exists():
        print(f"⚠️  File not found: {updates_file}")
        return 0

    updates_count = 0

    if updates_path.suffix == '.csv':
        # Load CSV
        df = pd.read_csv(updates_file)

        for _, row in df.iterrows():
            issue_id = str(row['issue_id'])
            failure_types = str(row['failure_types']).split(',')
            failure_types = [ft.strip() for ft in failure_types]
            mode = row.get('mode', 'replace')
            subtypes = str(row.get('subtypes', '')).split(',') if 'subtypes' in row else None

            if update_single_instance(classifications_data, issue_id, failure_types, subtypes, mode):
                updates_count += 1

    elif updates_path.suffix == '.json':
        # Load JSON
        with open(updates_file, 'r') as f:
            updates = json.load(f)

        for update in updates:
            issue_id = str(update['issue_id'])
            failure_types = update['failure_types']
            mode = update.get('mode', 'replace')
            subtypes = update.get('subtypes', None)

            if update_single_instance(classifications_data, issue_id, failure_types, subtypes, mode):
                updates_count += 1

    else:
        print(f"⚠️  Unsupported file format: {updates_path.suffix}")
        return 0

    return updates_count


def save_data(classifications_data: Dict, df: pd.DataFrame,
              classifications_path: str, dataset_path: str, backup: bool = True):
    """Save updated classifications and dataset."""

    # Create backups if requested
    if backup:
        backup_classifications = classifications_path.replace('.json', '_backup.json')
        backup_dataset = dataset_path.replace('.parquet', '_backup.parquet')

        with open(classifications_path, 'r') as f_in:
            with open(backup_classifications, 'w') as f_out:
                f_out.write(f_in.read())

        df_original = pd.read_parquet(dataset_path)
        df_original.to_parquet(backup_dataset, index=False)

        print(f"\n💾 Backups created:")
        print(f"   {backup_classifications}")
        print(f"   {backup_dataset}")

    # Save classifications
    with open(classifications_path, 'w') as f:
        json.dump(classifications_data, f, indent=2)

    # Update dataset with new failure types
    classification_map = {c['issue_id']: c for c in classifications_data['classifications']}

    df['failure_types'] = df['id'].astype(str).map(
        lambda x: classification_map.get(x, {}).get('failure_type', [])
    )
    df['failure_subtypes'] = df['id'].astype(str).map(
        lambda x: classification_map.get(x, {}).get('sub_type', [])
    )
    df['num_failure_types'] = df['failure_types'].apply(len)

    # Save dataset
    df.to_parquet(dataset_path, index=False)

    print(f"\n✓ Saved:")
    print(f"   {classifications_path}")
    print(f"   {dataset_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Update failure types for instances in the dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replace failure types for a single instance
  python3 scripts/update_instance_failure_types.py \\
    --issue-id 64 \\
    --failure-types "Code Formatting,Linting" \\
    --mode replace

  # Add failure types to an instance
  python3 scripts/update_instance_failure_types.py \\
    --issue-id 64 \\
    --failure-types "Test Failure" \\
    --mode add

  # Remove failure types from an instance
  python3 scripts/update_instance_failure_types.py \\
    --issue-id 64 \\
    --failure-types "Code Formatting" \\
    --mode remove

  # Batch update from CSV file
  python3 scripts/update_instance_failure_types.py \\
    --batch updates.csv \\
    --backup
        """
    )

    parser.add_argument('--classifications', type=str,
                       default='results/failure_classifications_cleaned.json',
                       help='Path to classifications JSON')
    parser.add_argument('--dataset', type=str,
                       default='dataset/lca_dataset.parquet',
                       help='Path to dataset')
    parser.add_argument('--issue-id', type=str,
                       help='Issue ID to update (for single instance update)')
    parser.add_argument('--failure-types', type=str,
                       help='Comma-separated list of failure types')
    parser.add_argument('--subtypes', type=str,
                       help='Comma-separated list of subtypes (optional)')
    parser.add_argument('--mode', type=str, choices=['replace', 'add', 'remove'],
                       default='replace',
                       help='Update mode: replace, add, or remove')
    parser.add_argument('--batch', type=str,
                       help='CSV or JSON file with batch updates')
    parser.add_argument('--backup', action='store_true',
                       help='Create backup before updating')

    args = parser.parse_args()

    print("="*80)
    print("UPDATE INSTANCE FAILURE TYPES")
    print("="*80)
    print()

    # Load data
    print(f"📂 Loading data...")
    classifications_data, df = load_data(args.classifications, args.dataset)
    print(f"   Loaded {len(classifications_data['classifications'])} classifications")
    print(f"   Loaded {len(df)} instances from dataset")

    # Perform updates
    updates_count = 0

    if args.batch:
        # Batch update
        print(f"\n🔄 Batch updating from {args.batch}...")
        updates_count = batch_update_from_file(classifications_data, args.batch)
        print(f"\n✓ Updated {updates_count} instances")

    elif args.issue_id and args.failure_types:
        # Single instance update
        print(f"\n🔄 Updating instance {args.issue_id}...")
        failure_types = [ft.strip() for ft in args.failure_types.split(',')]
        subtypes = [st.strip() for st in args.subtypes.split(',')] if args.subtypes else None

        if update_single_instance(classifications_data, args.issue_id,
                                 failure_types, subtypes, args.mode):
            updates_count = 1

    else:
        print("⚠️  Please provide either --batch or (--issue-id and --failure-types)")
        return 1

    # Save if updates were made
    if updates_count > 0:
        save_data(classifications_data, df, args.classifications, args.dataset, args.backup)
        print(f"\n🎉 Successfully updated {updates_count} instance(s)")
    else:
        print("\n⚠️  No instances were updated")

    print("="*80)
    return 0


if __name__ == "__main__":
    exit(main())

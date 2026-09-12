#!/usr/bin/env python3
"""
Clean failure classifications by removing duplicate failure types and update the main dataset.

This script:
1. Loads failure_classifications.json
2. Removes duplicate failure types for each issue (while preserving order)
3. Updates the main dataset with unique failure types
4. Saves both cleaned classifications and updated dataset
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def remove_duplicates_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates from list while preserving order of first occurrence."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def clean_classification(classification: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a single classification by removing duplicate failure types.

    Only keeps unique failure types (preserving first occurrence order).
    Example: ['Code Formatting', 'Linting', 'Code Formatting'] → ['Code Formatting', 'Linting']
    """
    failure_types = classification.get('failure_type', [])
    sub_types = classification.get('sub_type', [])
    details = classification.get('detail', [])

    # Track seen failure types and keep first occurrence
    seen_types = set()
    cleaned_failure_types = []
    cleaned_sub_types = []
    cleaned_details = []

    # Iterate through all items
    max_len = max(len(failure_types), len(sub_types), len(details)) if failure_types or sub_types or details else 0

    for i in range(max_len):
        # Get values (use empty string if index out of range)
        ftype = failure_types[i] if i < len(failure_types) else ''
        stype = sub_types[i] if i < len(sub_types) else ''
        detail = details[i] if i < len(details) else {}

        # Skip empty failure types
        if not ftype:
            continue

        # Only add if we haven't seen this failure_type before
        if ftype not in seen_types:
            seen_types.add(ftype)
            cleaned_failure_types.append(ftype)
            cleaned_sub_types.append(stype)
            if detail:
                cleaned_details.append(detail)

    return {
        'issue_id': classification['issue_id'],
        'failure_type': cleaned_failure_types,
        'sub_type': cleaned_sub_types,
        'detail': cleaned_details
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Clean and update failure types in dataset')
    parser.add_argument('--classifications', type=str,
                        default='results/failure_classifications.json',
                        help='Path to failure classifications JSON')
    parser.add_argument('--dataset', type=str,
                        default='dataset/lca_dataset.parquet',
                        help='Path to main dataset')
    parser.add_argument('--output-classifications', type=str,
                        default='results/failure_classifications_cleaned.json',
                        help='Output path for cleaned classifications')
    parser.add_argument('--output-dataset', type=str,
                        default='dataset/lca_dataset.parquet',
                        help='Output path for updated dataset (defaults to overwrite original)')
    parser.add_argument('--backup', action='store_true',
                        help='Create backup of original files before overwriting')
    args = parser.parse_args()

    print("="*80)
    print("CLEAN AND UPDATE FAILURE TYPES")
    print("="*80)
    print()

    # Load classifications
    print(f"📂 Loading classifications from {args.classifications}...")
    with open(args.classifications, 'r') as f:
        data = json.load(f)

    classifications = data.get('classifications', [])
    print(f"   Loaded {len(classifications)} classifications")

    # Find and report duplicates before cleaning
    print("\n🔍 Analyzing duplicates...")
    issues_with_duplicates = []
    total_duplicates = 0

    for c in classifications:
        failure_types = c.get('failure_type', [])
        unique_types = list(set(failure_types))

        if len(failure_types) != len(unique_types):
            num_duplicates = len(failure_types) - len(unique_types)
            total_duplicates += num_duplicates
            issues_with_duplicates.append({
                'id': c['issue_id'],
                'before': failure_types,
                'num_duplicates': num_duplicates
            })

    print(f"   Found {len(issues_with_duplicates)} issues with duplicate failure types")
    print(f"   Total duplicate entries to remove: {total_duplicates}")

    if issues_with_duplicates:
        print("\n   Sample issues with duplicates:")
        for item in issues_with_duplicates[:5]:
            print(f"     ID {item['id']}: {item['before']}")
            print(f"       ({item['num_duplicates']} duplicates)")

    # Clean classifications
    print("\n🧹 Cleaning classifications...")
    cleaned_classifications = []

    for c in classifications:
        cleaned = clean_classification(c)
        cleaned_classifications.append(cleaned)

    # Update the data dictionary with cleaned classifications
    data['classifications'] = cleaned_classifications

    # Recalculate unique types and subtypes
    all_types = []
    all_subtypes = []
    for c in cleaned_classifications:
        all_types.extend(c['failure_type'])
        all_subtypes.extend(c['sub_type'])

    data['unique_failure_types'] = sorted(set(all_types))
    data['unique_failure_subtypes'] = sorted(set(all_subtypes))
    data['total_classified'] = len(cleaned_classifications)

    print(f"   ✓ Cleaned {len(cleaned_classifications)} classifications")
    print(f"   ✓ Unique failure types: {len(data['unique_failure_types'])}")
    print(f"   ✓ Unique failure subtypes: {len(data['unique_failure_subtypes'])}")

    # Verify duplicates are removed
    print("\n✅ Verifying cleanup...")
    remaining_duplicates = 0
    for c in cleaned_classifications:
        failure_types = c['failure_type']
        if len(failure_types) != len(set(failure_types)):
            remaining_duplicates += 1

    if remaining_duplicates == 0:
        print("   ✓ No duplicates found in cleaned data!")
    else:
        print(f"   ⚠️  Still found {remaining_duplicates} issues with duplicates")

    # Save cleaned classifications
    print(f"\n💾 Saving cleaned classifications to {args.output_classifications}...")
    Path(args.output_classifications).parent.mkdir(exist_ok=True, parents=True)

    with open(args.output_classifications, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"   ✓ Saved cleaned classifications")

    # Load and update main dataset
    print(f"\n📂 Loading main dataset from {args.dataset}...")
    df = pd.read_parquet(args.dataset)
    print(f"   Loaded dataset with {len(df)} issues")

    # Create backup if requested
    if args.backup and args.output_dataset == args.dataset:
        backup_path = args.dataset.replace('.parquet', '_backup.parquet')
        print(f"\n💾 Creating backup at {backup_path}...")
        df.to_parquet(backup_path, index=False)
        print(f"   ✓ Backup created")

    # Create classification lookup
    print("\n🔄 Adding failure types to dataset...")
    classification_map = {c['issue_id']: c for c in cleaned_classifications}

    # Add columns to dataframe
    df['failure_types'] = df['id'].astype(str).map(
        lambda x: classification_map.get(x, {}).get('failure_type', [])
    )
    df['failure_subtypes'] = df['id'].astype(str).map(
        lambda x: classification_map.get(x, {}).get('sub_type', [])
    )
    df['num_failure_types'] = df['failure_types'].apply(len)

    # Statistics
    print("\n📊 Dataset statistics:")
    print(f"   Total issues: {len(df)}")
    print(f"   Issues with failure types: {(df['num_failure_types'] > 0).sum()}")
    print(f"   Issues with multiple failure types: {(df['num_failure_types'] > 1).sum()}")
    print(f"   Average failure types per issue: {df['num_failure_types'].mean():.2f}")

    # Show distribution
    print("\n   Failure type distribution:")
    type_counts = {}
    for types_list in df['failure_types']:
        for ftype in types_list:
            type_counts[ftype] = type_counts.get(ftype, 0) + 1

    for ftype, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"     {ftype}: {count}")

    # Save updated dataset
    print(f"\n💾 Saving updated dataset to {args.output_dataset}...")
    df.to_parquet(args.output_dataset, index=False)
    print(f"   ✓ Dataset updated successfully")

    # Summary
    print("\n" + "="*80)
    print("✅ SUMMARY")
    print("="*80)
    print(f"Issues with duplicates found: {len(issues_with_duplicates)}")
    print(f"Duplicate entries removed: {total_duplicates}")
    print(f"Cleaned classifications saved: {args.output_classifications}")
    print(f"Updated dataset saved: {args.output_dataset}")

    if args.backup and args.output_dataset == args.dataset:
        print(f"Backup created: {backup_path}")

    print("\n🎉 All done! Dataset now has unique failure types.")
    print("="*80)

    return 0


if __name__ == "__main__":
    exit(main())

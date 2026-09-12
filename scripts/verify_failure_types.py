#!/usr/bin/env python3
"""Quick script to verify failure types in the dataset."""

import pandas as pd
import sys

def verify_failure_types(dataset_path='dataset/lca_dataset.parquet'):
    """Verify failure types are properly set in dataset."""
    df = pd.read_parquet(dataset_path)

    print("="*80)
    print("FAILURE TYPES VERIFICATION")
    print("="*80)

    # Check if columns exist
    required_cols = ['failure_types', 'failure_subtypes', 'num_failure_types']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"\n❌ MISSING COLUMNS: {missing_cols}")
        print("   Run: python3 scripts/clean_and_update_failure_types.py --backup")
        return False

    print(f"\n✅ All required columns present")

    # Statistics
    print(f"\n📊 STATISTICS:")
    print(f"   Total instances: {len(df)}")
    print(f"   With failure types: {(df['num_failure_types'] > 0).sum()}")
    print(f"   With multiple types: {(df['num_failure_types'] > 1).sum()}")
    print(f"   Average types/instance: {df['num_failure_types'].mean():.2f}")

    # Type distribution
    print(f"\n📈 TOP FAILURE TYPES:")
    type_counts = {}
    for types_list in df['failure_types']:
        for ftype in types_list:
            type_counts[ftype] = type_counts.get(ftype, 0) + 1

    for i, (ftype, count) in enumerate(sorted(type_counts.items(), key=lambda x: -x[1])[:10], 1):
        pct = count / len(df) * 100
        print(f"   {i:2d}. {ftype:25s}: {count:3d} ({pct:5.1f}%)")

    # Sample instances
    print(f"\n📋 SAMPLE INSTANCES:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"\n   Instance {row['id']}:")
        for j, (ftype, subtype) in enumerate(zip(row['failure_types'], row['failure_subtypes']), 1):
            print(f"     {j}. {ftype}")
            if subtype:
                print(f"        └─ {subtype}")

    print("\n" + "="*80)
    print("✅ VERIFICATION COMPLETE")
    print("="*80)

    return True

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'dataset/lca_dataset.parquet'
    verify_failure_types(dataset)

#!/usr/bin/env python3
"""
Normalize all failure types to the 12 canonical types from the taxonomy.

Fixes the dataset by mapping all variant types to their canonical forms:
- Import Error → Dependency Issues
- Build Failure → Configuration Error
- Code Quality → Linting
- etc.
"""

import json
import os
import shutil
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict
import argparse


# The 12 canonical failure types from taxonomy
CANONICAL_TAXONOMY = {
    "Code Formatting": "Issues with code style, indentation, line length, etc.",
    "Linting": "Code quality issues detected by linters (unused imports, undefined vars, etc.)",
    "Syntax Error": "Python syntax errors, invalid code structure",
    "Runtime Error": "Errors that occur during execution (AttributeError, KeyError, etc.)",
    "Test Failure": "Unit/integration tests that fail",
    "Assertion Error": "Failed assertions in tests",
    "Type Checking": "Type annotation errors, mypy issues",
    "Dependency Issues": "Missing dependencies, version conflicts",
    "Package Install Error": "Failed package installation, missing requirements",
    "Configuration Error": "Invalid config files, wrong settings",
    "Environment Error": "Platform-specific issues, missing env vars",
    "Doc/Docstring": "Documentation format issues, missing docstrings"
}

# Mapping from all possible types to canonical types
TYPE_MAPPING = {
    # Already canonical (keep as-is)
    "Code Formatting": "Code Formatting",
    "Linting": "Linting",
    "Syntax Error": "Syntax Error",
    "Runtime Error": "Runtime Error",
    "Test Failure": "Test Failure",
    "Assertion Error": "Assertion Error",
    "Type Checking": "Type Checking",
    "Dependency Issues": "Dependency Issues",
    "Package Install Error": "Package Install Error",
    "Configuration Error": "Configuration Error",
    "Environment Error": "Environment Error",
    "Doc/Docstring": "Doc/Docstring",

    # Variants that need mapping
    "Import Error": "Dependency Issues",
    "Import Chain": "Dependency Issues",
    "Build Error": "Configuration Error",
    "Build Failure": "Configuration Error",
    "Build/Compilation Error": "Configuration Error",
    "CI Failure": "Configuration Error",
    "Code Quality": "Linting",
    "Code Style": "Code Formatting",
    "Runtime Warning": "Runtime Error",
    "Documentation": "Doc/Docstring",
    "Warning": "Linting",
    "Infrastructure Error": "Environment Error",
    "Correctness": "Test Failure"
}


def normalize_failure_types(classifications: list) -> tuple:
    """
    Normalize all failure types to canonical 12 types.

    Returns:
        (normalized_classifications, mapping_stats)
    """
    print("\n🔄 Normalizing failure types to 12 canonical types...")

    normalized = []
    mapping_stats = Counter()

    for c in classifications:
        failure_types = c.get('failure_type', [])
        sub_types = c.get('sub_type', [])
        details = c.get('detail', [])

        # Normalize each type
        normalized_types = []
        normalized_subtypes = []
        normalized_details = []
        seen = set()  # Track to avoid duplicates after normalization

        for i, ftype in enumerate(failure_types):
            # Map to canonical type
            canonical = TYPE_MAPPING.get(ftype, ftype)

            # Track mapping
            if ftype != canonical:
                mapping_stats[f"{ftype} → {canonical}"] += 1

            # Only add if not duplicate after normalization
            if canonical not in seen:
                normalized_types.append(canonical)
                seen.add(canonical)

                # Keep corresponding subtype and detail
                if i < len(sub_types):
                    normalized_subtypes.append(sub_types[i])
                else:
                    normalized_subtypes.append('')

                if i < len(details):
                    normalized_details.append(details[i])
                else:
                    normalized_details.append({})

        # Create normalized classification
        normalized_c = {
            'issue_id': c['issue_id'],
            'failure_type': normalized_types,
            'sub_type': normalized_subtypes,
            'detail': normalized_details
        }
        normalized.append(normalized_c)

    print(f"   ✓ Normalized {len(normalized)} classifications")

    # Report mappings
    if mapping_stats:
        print("\n   Mappings applied:")
        for mapping, count in mapping_stats.most_common():
            print(f"     {mapping}: {count} instances")

    return normalized, dict(mapping_stats)


def calculate_statistics(classifications: list) -> dict:
    """Calculate occurrence and co-occurrence statistics."""
    print("\n📊 Calculating statistics for 12 canonical types...")

    total_instances = len(classifications)
    type_counts = Counter()

    # Count occurrences
    for c in classifications:
        for ftype in c.get('failure_type', []):
            type_counts[ftype] += 1

    # Calculate co-occurrence
    cooccurrence = defaultdict(lambda: defaultdict(int))

    for c in classifications:
        failure_types = c.get('failure_type', [])

        for type1 in failure_types:
            for type2 in failure_types:
                cooccurrence[type1][type2] += 1

    # Create pairs (excluding self-pairs, avoiding duplicates)
    pairs = []
    canonical_types = sorted(CANONICAL_TAXONOMY.keys())

    for type1 in canonical_types:
        for type2 in canonical_types:
            if type1 < type2:
                count = cooccurrence[type1].get(type2, 0)
                if count > 0:
                    pairs.append({
                        'type1': type1,
                        'type2': type2,
                        'count': count,
                        'percentage': (count / total_instances) * 100
                    })

    pairs.sort(key=lambda x: x['count'], reverse=True)

    # Occurrence statistics
    occurrence_stats = []
    for ftype in canonical_types:
        count = type_counts.get(ftype, 0)
        if count > 0:
            occurrence_stats.append({
                'failure_type': ftype,
                'count': count,
                'percentage': (count / total_instances) * 100
            })

    occurrence_stats.sort(key=lambda x: x['count'], reverse=True)

    print(f"   ✓ Statistics calculated")

    return {
        'total_instances': total_instances,
        'unique_types': len([x for x in occurrence_stats if x['count'] > 0]),
        'occurrence': occurrence_stats,
        'type_counts': dict(type_counts),
        'cooccurrence_matrix': {k: dict(v) for k, v in cooccurrence.items()},
        'cooccurrence_pairs': pairs
    }


def generate_report(stats: dict, mapping_stats: dict, output_path: str):
    """Generate comprehensive report."""
    print("\n📝 Generating report...")

    lines = []

    # Header
    lines.append("=" * 100)
    lines.append("NORMALIZED FAILURE TYPE STATISTICS - 12 CANONICAL TYPES")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Total instances: {stats['total_instances']}")
    lines.append(f"Canonical failure types: {stats['unique_types']}")
    lines.append("")

    # Mappings applied
    if mapping_stats:
        lines.append("=" * 100)
        lines.append("MAPPINGS APPLIED")
        lines.append("=" * 100)
        lines.append("")
        for mapping, count in sorted(mapping_stats.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {mapping}: {count} instances")
        lines.append("")

    # Occurrence
    lines.append("=" * 100)
    lines.append("1. FAILURE TYPE OCCURRENCE (12 Canonical Types)")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'Rank':<6} {'Failure Type':<35} {'Count':<10} {'% of Instances':<15}")
    lines.append("-" * 100)

    for rank, item in enumerate(stats['occurrence'], 1):
        lines.append(f"{rank:<6} {item['failure_type']:<35} "
                    f"{item['count']:<10} {item['percentage']:>6.1f}%")

    lines.append("")

    # Co-occurrence
    lines.append("=" * 100)
    lines.append("2. FAILURE TYPE CO-OCCURRENCE")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'Rank':<6} {'Type 1':<30} {'Type 2':<30} {'Count':<10} {'% of Instances':<15}")
    lines.append("-" * 100)

    for rank, pair in enumerate(stats['cooccurrence_pairs'], 1):
        lines.append(f"{rank:<6} {pair['type1']:<30} {pair['type2']:<30} "
                    f"{pair['count']:<10} {pair['percentage']:>6.1f}%")

    lines.append("")
    lines.append("=" * 100)
    lines.append("END OF REPORT")
    lines.append("=" * 100)

    # Save
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"   ✓ Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Normalize all failure types to 12 canonical types'
    )
    parser.add_argument('--classifications', type=str,
                       default='results/failure_classifications_cleaned.json',
                       help='Input classifications JSON')
    parser.add_argument('--dataset', type=str,
                       default='dataset/lca_dataset.parquet',
                       help='Input dataset')
    parser.add_argument('--output-classifications', type=str,
                       default='results/failure_classifications_12types.json',
                       help='Output normalized classifications')
    parser.add_argument('--output-dataset', type=str,
                       default=None,
                       help='Optional alternate dataset output; defaults to updating --dataset in place')
    parser.add_argument('--output-stats', type=str,
                       default='results/failure_stats_12types.json',
                       help='Output statistics JSON')
    parser.add_argument('--output-report', type=str,
                       default='results/failure_report_12types.txt',
                       help='Output text report')
    parser.add_argument('--update-original', action='store_true',
                       help='Also overwrite the input classifications JSON')
    parser.add_argument('--no-dataset-backup', action='store_true',
                       help='Skip the timestamped backup when updating the dataset in place')

    args = parser.parse_args()

    print("=" * 100)
    print("NORMALIZE TO 12 CANONICAL FAILURE TYPES")
    print("=" * 100)

    # Load data
    print("\n📂 Loading data...")
    with open(args.classifications, 'r') as f:
        data = json.load(f)

    df = pd.read_parquet(args.dataset)

    print(f"   ✓ Loaded {len(data['classifications'])} classifications")
    print(f"   ✓ Loaded {len(df)} instances from dataset")

    # Show current state
    current_types = set()
    for c in data['classifications']:
        current_types.update(c.get('failure_type', []))

    print(f"\n   Current unique types: {len(current_types)}")
    print(f"   Target canonical types: {len(CANONICAL_TAXONOMY)}")

    # Normalize
    normalized_classifications, mapping_stats = normalize_failure_types(data['classifications'])

    # Verify
    final_types = set()
    for c in normalized_classifications:
        final_types.update(c.get('failure_type', []))

    non_canonical = final_types - set(CANONICAL_TAXONOMY.keys())
    if non_canonical:
        print(f"\n   ⚠️  WARNING: Found non-canonical types: {non_canonical}")
    else:
        print(f"\n   ✅ All types are now canonical ({len(final_types)} types)")

    # Calculate statistics
    stats = calculate_statistics(normalized_classifications)

    # Save normalized classifications
    print(f"\n💾 Saving normalized classifications...")

    output_class_path = args.classifications if args.update_original else args.output_classifications

    normalized_data = {
        'total_classified': len(normalized_classifications),
        'taxonomy': CANONICAL_TAXONOMY,
        'unique_failure_types': sorted(final_types),
        'classifications': normalized_classifications
    }

    Path(output_class_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_class_path, 'w') as f:
        json.dump(normalized_data, f, indent=2)

    print(f"   ✓ Saved to {output_class_path}")

    # Update dataset
    print(f"\n💾 Updating dataset...")

    output_dataset_path = args.output_dataset or args.dataset

    classification_map = {c['issue_id']: c for c in normalized_classifications}

    df['failure_types'] = df['id'].astype(str).map(
        lambda x: classification_map.get(x, {}).get('failure_type', [])
    )
    # Keep the legacy paper-facing field synchronized with the canonical labels.
    df['error_type'] = df['failure_types'].apply(list)
    df['failure_subtypes'] = df['id'].astype(str).map(
        lambda x: classification_map.get(x, {}).get('sub_type', [])
    )
    df['failure_details'] = df['id'].astype(str).map(
        lambda x: classification_map.get(x, {}).get('detail', [])
    )
    df['num_failure_types'] = df['failure_types'].apply(len)

    dataset_input = Path(args.dataset).resolve()
    dataset_output = Path(output_dataset_path).resolve()
    backup_path = None
    if dataset_input == dataset_output:
        if not args.no_dataset_backup:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            backup_path = dataset_input.with_name(
                f'{dataset_input.stem}.before-12type-normalization-{timestamp}{dataset_input.suffix}'
            )
            shutil.copy2(dataset_input, backup_path)
        temporary_path = dataset_output.with_suffix(dataset_output.suffix + '.tmp')
        df.to_parquet(temporary_path, index=False)
        # Confirm the artifact is readable before atomically replacing the source.
        pd.read_parquet(temporary_path, columns=['id'])
        os.replace(temporary_path, dataset_output)
    else:
        dataset_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dataset_output, index=False)
    print(f"   ✓ Saved to {output_dataset_path}")
    if backup_path:
        print(f"   ✓ Backup saved to {backup_path}")

    # Save statistics
    print(f"\n💾 Saving statistics...")

    all_stats = {
        'canonical_taxonomy': CANONICAL_TAXONOMY,
        'mapping_applied': mapping_stats,
        'statistics': stats
    }

    Path(args.output_stats).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output_stats, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"   ✓ Saved to {args.output_stats}")

    # Generate report
    generate_report(stats, mapping_stats, args.output_report)

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total instances: {stats['total_instances']}")
    print(f"Canonical failure types: {stats['unique_types']}")
    print()

    print("Top 5 failure types by occurrence:")
    for rank, item in enumerate(stats['occurrence'][:5], 1):
        print(f"  {rank}. {item['failure_type']}: {item['count']} instances ({item['percentage']:.1f}%)")

    print()
    print("Top 5 co-occurring pairs:")
    for rank, pair in enumerate(stats['cooccurrence_pairs'][:5], 1):
        print(f"  {rank}. {pair['type1']} + {pair['type2']}: "
              f"{pair['count']} instances ({pair['percentage']:.1f}%)")

    print()
    print(f"✅ Dataset updated with 12 canonical types: {output_dataset_path}")
    print(f"📊 Normalized classifications: {output_class_path}")
    print(f"📊 Statistics: {args.output_stats}")
    print(f"📊 Report: {args.output_report}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    exit(main())

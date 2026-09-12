#!/usr/bin/env python3
"""
Analyze failure type occurrence and co-occurrence across the overall dataset.

This script calculates:
1. Occurrence count for each unique failure type
2. Co-occurrence matrix showing which types appear together
3. Detailed statistics about failure type combinations
"""

import json
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
import argparse


def load_data(classifications_path: str):
    """Load classifications data."""
    with open(classifications_path, 'r') as f:
        data = json.load(f)
    return data['classifications']


def calculate_occurrence(classifications: list) -> dict:
    """Calculate occurrence count for each failure type."""
    print("\n📊 Calculating failure type occurrences...")

    total_instances = len(classifications)
    type_counts = Counter()

    # Count each failure type across all instances
    for c in classifications:
        for ftype in c.get('failure_type', []):
            type_counts[ftype] += 1

    # Create detailed statistics
    occurrence_stats = []
    for ftype, count in type_counts.most_common():
        occurrence_stats.append({
            'failure_type': ftype,
            'count': count,
            'percentage': (count / total_instances) * 100
        })

    print(f"   ✓ Found {len(type_counts)} unique failure types")

    return {
        'total_instances': total_instances,
        'unique_types': len(type_counts),
        'occurrence': occurrence_stats
    }


def calculate_cooccurrence(classifications: list) -> dict:
    """Calculate co-occurrence matrix for failure types."""
    print("\n🔗 Calculating co-occurrence matrix...")

    total_instances = len(classifications)

    # Build co-occurrence matrix
    cooccurrence_matrix = defaultdict(lambda: defaultdict(int))

    for c in classifications:
        failure_types = c.get('failure_type', [])

        # Count pairs
        for type1 in failure_types:
            for type2 in failure_types:
                cooccurrence_matrix[type1][type2] += 1

    # Get all unique types
    all_types = sorted(cooccurrence_matrix.keys())

    # Create pairs list (excluding self-pairs, avoiding duplicates A-B and B-A)
    pairs = []
    for type1 in all_types:
        for type2 in all_types:
            if type1 < type2:  # Only one direction
                count = cooccurrence_matrix[type1].get(type2, 0)
                if count > 0:
                    pairs.append({
                        'type1': type1,
                        'type2': type2,
                        'count': count,
                        'percentage': (count / total_instances) * 100
                    })

    # Sort by count
    pairs.sort(key=lambda x: x['count'], reverse=True)

    # Convert matrix to regular dict for JSON
    matrix_dict = {k: dict(v) for k, v in cooccurrence_matrix.items()}

    print(f"   ✓ Calculated {len(pairs)} co-occurring pairs")

    return {
        'matrix': matrix_dict,
        'pairs': pairs,
        'total_pairs': len(pairs)
    }


def generate_report(occurrence_stats: dict, cooccurrence_stats: dict, output_path: str):
    """Generate a detailed text report."""
    print(f"\n📝 Generating report...")

    lines = []

    # Header
    lines.append("=" * 100)
    lines.append("FAILURE TYPE OCCURRENCE AND CO-OCCURRENCE ANALYSIS")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Total instances in dataset: {occurrence_stats['total_instances']}")
    lines.append(f"Unique failure types: {occurrence_stats['unique_types']}")
    lines.append("")

    # Section 1: Individual Occurrence
    lines.append("=" * 100)
    lines.append("SECTION 1: INDIVIDUAL FAILURE TYPE OCCURRENCE")
    lines.append("=" * 100)
    lines.append("")
    lines.append("How many times each failure type appears across all instances:")
    lines.append("")
    lines.append(f"{'Rank':<6} {'Failure Type':<35} {'Count':<10} {'% of Instances':<15}")
    lines.append("-" * 100)

    for rank, item in enumerate(occurrence_stats['occurrence'], 1):
        lines.append(f"{rank:<6} {item['failure_type']:<35} "
                    f"{item['count']:<10} {item['percentage']:>6.1f}%")

    lines.append("")
    lines.append(f"Note: Percentages can exceed 100% because one instance can have multiple failure types.")
    lines.append("")

    # Section 2: Co-occurrence
    lines.append("=" * 100)
    lines.append("SECTION 2: FAILURE TYPE CO-OCCURRENCE")
    lines.append("=" * 100)
    lines.append("")
    lines.append("How many instances have BOTH Type 1 AND Type 2 together:")
    lines.append("")
    lines.append(f"{'Rank':<6} {'Type 1':<30} {'Type 2':<30} {'Count':<10} {'% of Instances':<15}")
    lines.append("-" * 100)

    for rank, pair in enumerate(cooccurrence_stats['pairs'], 1):
        lines.append(f"{rank:<6} {pair['type1']:<30} {pair['type2']:<30} "
                    f"{pair['count']:<10} {pair['percentage']:>6.1f}%")

    lines.append("")
    lines.append(f"Total co-occurring pairs: {cooccurrence_stats['total_pairs']}")
    lines.append("")

    # Section 3: Top Co-occurrences Summary
    lines.append("=" * 100)
    lines.append("SECTION 3: TOP 20 CO-OCCURRING PAIRS")
    lines.append("=" * 100)
    lines.append("")

    for rank, pair in enumerate(cooccurrence_stats['pairs'][:20], 1):
        lines.append(f"{rank:2}. {pair['type1']} + {pair['type2']}")
        lines.append(f"    → {pair['count']} instances ({pair['percentage']:.1f}% of dataset)")
        lines.append("")

    # Section 4: Co-occurrence Matrix
    lines.append("=" * 100)
    lines.append("SECTION 4: FULL CO-OCCURRENCE MATRIX")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Each cell shows: Number of instances where both types appear together")
    lines.append("")

    # Get all types for matrix
    all_types = sorted(cooccurrence_stats['matrix'].keys())

    # Matrix header (abbreviated type names)
    header = "Type".ljust(20)
    type_abbrev = {}
    for i, ftype in enumerate(all_types, 1):
        abbrev = f"T{i}"
        type_abbrev[ftype] = abbrev
        header += f"{abbrev:>6}"

    lines.append(header)
    lines.append("-" * 100)

    # Matrix rows
    for type1 in all_types:
        row = f"{type_abbrev[type1]} {type1[:15]}".ljust(20)
        for type2 in all_types:
            count = cooccurrence_stats['matrix'].get(type1, {}).get(type2, 0)
            row += f"{count:>6}"
        lines.append(row)

    lines.append("")
    lines.append("Legend:")
    for ftype, abbrev in sorted(type_abbrev.items(), key=lambda x: int(x[1][1:])):
        lines.append(f"  {abbrev}: {ftype}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("END OF REPORT")
    lines.append("=" * 100)

    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"   ✓ Report saved to {output_path}")


def save_json(occurrence_stats: dict, cooccurrence_stats: dict, output_path: str):
    """Save statistics as JSON."""
    all_stats = {
        'occurrence': occurrence_stats,
        'cooccurrence': cooccurrence_stats
    }

    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"   ✓ JSON saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze failure type occurrence and co-occurrence'
    )
    parser.add_argument('--classifications', type=str,
                       default='results/failure_classifications_cleaned.json',
                       help='Path to classifications JSON')
    parser.add_argument('--output-json', type=str,
                       default='results/failure_occurrence_stats.json',
                       help='Output JSON file')
    parser.add_argument('--output-report', type=str,
                       default='results/failure_occurrence_report.txt',
                       help='Output text report')

    args = parser.parse_args()

    print("=" * 100)
    print("FAILURE TYPE OCCURRENCE AND CO-OCCURRENCE ANALYSIS")
    print("=" * 100)

    # Load data
    print("\n📂 Loading classifications...")
    classifications = load_data(args.classifications)
    print(f"   ✓ Loaded {len(classifications)} classifications")

    # Calculate statistics
    occurrence_stats = calculate_occurrence(classifications)
    cooccurrence_stats = calculate_cooccurrence(classifications)

    # Save JSON
    print("\n💾 Saving results...")
    Path(args.output_json).parent.mkdir(exist_ok=True, parents=True)
    save_json(occurrence_stats, cooccurrence_stats, args.output_json)

    # Generate report
    generate_report(occurrence_stats, cooccurrence_stats, args.output_report)

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total instances: {occurrence_stats['total_instances']}")
    print(f"Unique failure types: {occurrence_stats['unique_types']}")
    print()
    print("Top 5 failure types by occurrence:")
    for rank, item in enumerate(occurrence_stats['occurrence'][:5], 1):
        print(f"  {rank}. {item['failure_type']}: {item['count']} instances ({item['percentage']:.1f}%)")

    print()
    print("Top 5 co-occurring pairs:")
    for rank, pair in enumerate(cooccurrence_stats['pairs'][:5], 1):
        print(f"  {rank}. {pair['type1']} + {pair['type2']}: "
              f"{pair['count']} instances ({pair['percentage']:.1f}%)")

    print()
    print(f"📊 Full report: {args.output_report}")
    print(f"📊 JSON data: {args.output_json}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    exit(main())

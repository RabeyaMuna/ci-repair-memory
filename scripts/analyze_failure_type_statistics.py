#!/usr/bin/env python3
"""
Generate comprehensive statistics about failure types across the entire dataset.

This script analyzes:
1. Overall failure type distribution
2. Failure type co-occurrences
3. Repository-level statistics
4. Temporal trends (if date information is available)
5. Failure type complexity (single vs multiple types)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import argparse


def load_data(classifications_path: str, dataset_path: str):
    """Load classifications and dataset."""
    print("📂 Loading data...")

    with open(classifications_path, 'r') as f:
        classifications_data = json.load(f)

    df = pd.read_parquet(dataset_path)

    print(f"   ✓ Loaded {len(classifications_data['classifications'])} classifications")
    print(f"   ✓ Loaded {len(df)} instances from dataset")

    return classifications_data, df


def calculate_basic_statistics(classifications: List[Dict]) -> Dict:
    """Calculate basic statistics about failure types."""
    print("\n📊 Calculating basic statistics...")

    total_instances = len(classifications)
    failure_type_counts = Counter()
    failure_subtype_counts = Counter()
    instances_per_num_types = Counter()

    for c in classifications:
        failure_types = c.get('failure_type', [])
        subtypes = c.get('sub_type', [])

        # Count failure types
        for ftype in failure_types:
            failure_type_counts[ftype] += 1

        # Count subtypes
        for stype in subtypes:
            if stype:  # Skip empty subtypes
                failure_subtype_counts[stype] += 1

        # Count instances by number of failure types
        num_types = len(failure_types)
        instances_per_num_types[num_types] += 1

    stats = {
        'total_instances': total_instances,
        'total_failure_types': len(failure_type_counts),
        'total_subtypes': len(failure_subtype_counts),
        'failure_type_counts': dict(failure_type_counts),
        'failure_subtype_counts': dict(failure_subtype_counts),
        'instances_per_num_types': dict(instances_per_num_types),
        'instances_with_single_type': instances_per_num_types.get(1, 0),
        'instances_with_multiple_types': sum(v for k, v in instances_per_num_types.items() if k > 1),
        'max_types_per_instance': max(instances_per_num_types.keys()) if instances_per_num_types else 0,
        'avg_types_per_instance': sum(k * v for k, v in instances_per_num_types.items()) / total_instances
    }

    print("   ✓ Basic statistics calculated")
    return stats


def calculate_cooccurrence_matrix(classifications: List[Dict]) -> Dict:
    """Calculate failure type co-occurrence matrix."""
    print("\n🔗 Calculating co-occurrence matrix...")

    # Get all unique failure types
    all_types = set()
    for c in classifications:
        all_types.update(c.get('failure_type', []))

    all_types = sorted(all_types)

    # Initialize co-occurrence matrix
    cooccurrence = defaultdict(lambda: defaultdict(int))

    # Count co-occurrences
    for c in classifications:
        failure_types = c.get('failure_type', [])

        # For each pair of failure types in the same instance
        for i, type1 in enumerate(failure_types):
            for type2 in failure_types[i:]:  # Including self-pairs
                cooccurrence[type1][type2] += 1
                if type1 != type2:
                    cooccurrence[type2][type1] += 1

    # Convert to regular dict for JSON serialization
    cooccurrence_dict = {k: dict(v) for k, v in cooccurrence.items()}

    # Find most common pairs
    pairs = []
    for type1 in all_types:
        for type2 in all_types:
            if type1 < type2:  # Avoid duplicates (A,B) and (B,A)
                count = cooccurrence[type1].get(type2, 0)
                if count > 0:
                    pairs.append((type1, type2, count))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print("   ✓ Co-occurrence matrix calculated")
    return {
        'matrix': cooccurrence_dict,
        'top_pairs': pairs[:20]  # Top 20 pairs
    }


def calculate_repository_statistics(df: pd.DataFrame) -> Dict:
    """Calculate statistics by repository."""
    print("\n🏢 Calculating repository statistics...")

    if 'repo_name' not in df.columns or 'failure_types' not in df.columns:
        print("   ⚠️  Repository or failure_types column not found")
        return {}

    repo_stats = {}

    for repo in df['repo_name'].unique():
        repo_df = df[df['repo_name'] == repo]

        # Count failure types in this repo
        failure_type_counts = Counter()
        for types_list in repo_df['failure_types']:
            if isinstance(types_list, (list, np.ndarray)):
                for ftype in types_list:
                    failure_type_counts[ftype] += 1

        repo_stats[repo] = {
            'total_instances': len(repo_df),
            'avg_types_per_instance': repo_df['num_failure_types'].mean() if 'num_failure_types' in repo_df.columns else 0,
            'most_common_type': failure_type_counts.most_common(1)[0] if failure_type_counts else ('None', 0),
            'unique_failure_types': len(failure_type_counts),
            'failure_type_distribution': dict(failure_type_counts.most_common(10))
        }

    print(f"   ✓ Statistics for {len(repo_stats)} repositories")
    return repo_stats


def calculate_temporal_statistics(df: pd.DataFrame) -> Dict:
    """Calculate temporal trends if date information is available."""
    print("\n📅 Calculating temporal statistics...")

    if 'commit_date' not in df.columns or 'failure_types' not in df.columns:
        print("   ⚠️  commit_date or failure_types column not found")
        return {}

    # Convert to datetime
    df['commit_date_parsed'] = pd.to_datetime(df['commit_date'], errors='coerce')

    # Filter out invalid dates
    df_with_dates = df[df['commit_date_parsed'].notna()].copy()

    if len(df_with_dates) == 0:
        print("   ⚠️  No valid dates found")
        return {}

    # Group by month
    df_with_dates['month'] = df_with_dates['commit_date_parsed'].dt.to_period('M')

    monthly_stats = {}
    for month, month_df in df_with_dates.groupby('month'):
        failure_type_counts = Counter()
        for types_list in month_df['failure_types']:
            if isinstance(types_list, (list, np.ndarray)):
                for ftype in types_list:
                    failure_type_counts[ftype] += 1

        monthly_stats[str(month)] = {
            'total_instances': len(month_df),
            'avg_types_per_instance': month_df['num_failure_types'].mean() if 'num_failure_types' in month_df.columns else 0,
            'top_3_types': dict(failure_type_counts.most_common(3))
        }

    print(f"   ✓ Temporal statistics for {len(monthly_stats)} months")
    return monthly_stats


def generate_summary_report(stats: Dict, cooccurrence: Dict, repo_stats: Dict,
                           temporal_stats: Dict, output_dir: str):
    """Generate a human-readable summary report."""
    print("\n📝 Generating summary report...")

    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("FAILURE TYPE STATISTICS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Basic Statistics
    report_lines.append("## OVERALL STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total instances: {stats['total_instances']}")
    report_lines.append(f"Unique failure types: {stats['total_failure_types']}")
    report_lines.append(f"Unique subtypes: {stats['total_subtypes']}")
    report_lines.append(f"Instances with single failure type: {stats['instances_with_single_type']} ({stats['instances_with_single_type']/stats['total_instances']*100:.1f}%)")
    report_lines.append(f"Instances with multiple failure types: {stats['instances_with_multiple_types']} ({stats['instances_with_multiple_types']/stats['total_instances']*100:.1f}%)")
    report_lines.append(f"Average failure types per instance: {stats['avg_types_per_instance']:.2f}")
    report_lines.append(f"Maximum failure types in one instance: {stats['max_types_per_instance']}")
    report_lines.append("")

    # Failure Type Distribution
    report_lines.append("## FAILURE TYPE DISTRIBUTION")
    report_lines.append("-" * 80)
    sorted_types = sorted(stats['failure_type_counts'].items(), key=lambda x: x[1], reverse=True)

    report_lines.append(f"{'Rank':<6} {'Failure Type':<30} {'Count':<10} {'Percentage':<10}")
    report_lines.append("-" * 80)

    for rank, (ftype, count) in enumerate(sorted_types, 1):
        percentage = count / stats['total_instances'] * 100
        report_lines.append(f"{rank:<6} {ftype:<30} {count:<10} {percentage:>6.1f}%")

    report_lines.append("")

    # Complexity Distribution
    report_lines.append("## COMPLEXITY DISTRIBUTION (Number of Failure Types per Instance)")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Num Types':<15} {'Count':<10} {'Percentage':<10}")
    report_lines.append("-" * 80)

    sorted_complexity = sorted(stats['instances_per_num_types'].items())
    for num_types, count in sorted_complexity:
        percentage = count / stats['total_instances'] * 100
        report_lines.append(f"{num_types:<15} {count:<10} {percentage:>6.1f}%")

    report_lines.append("")

    # Co-occurrence
    report_lines.append("## TOP FAILURE TYPE CO-OCCURRENCES")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Rank':<6} {'Type 1':<25} {'Type 2':<25} {'Count':<10}")
    report_lines.append("-" * 80)

    for rank, (type1, type2, count) in enumerate(cooccurrence['top_pairs'][:15], 1):
        report_lines.append(f"{rank:<6} {type1:<25} {type2:<25} {count:<10}")

    report_lines.append("")

    # Top Subtypes
    report_lines.append("## TOP 20 FAILURE SUBTYPES")
    report_lines.append("-" * 80)
    sorted_subtypes = sorted(stats['failure_subtype_counts'].items(), key=lambda x: x[1], reverse=True)[:20]

    report_lines.append(f"{'Rank':<6} {'Subtype':<50} {'Count':<10}")
    report_lines.append("-" * 80)

    for rank, (subtype, count) in enumerate(sorted_subtypes, 1):
        subtype_short = (subtype[:47] + '...') if len(subtype) > 50 else subtype
        report_lines.append(f"{rank:<6} {subtype_short:<50} {count:<10}")

    report_lines.append("")

    # Repository Statistics (top 10)
    if repo_stats:
        report_lines.append("## TOP 10 REPOSITORIES BY INSTANCE COUNT")
        report_lines.append("-" * 80)
        sorted_repos = sorted(repo_stats.items(), key=lambda x: x[1]['total_instances'], reverse=True)[:10]

        report_lines.append(f"{'Rank':<6} {'Repository':<30} {'Instances':<12} {'Avg Types':<12}")
        report_lines.append("-" * 80)

        for rank, (repo, repo_data) in enumerate(sorted_repos, 1):
            repo_short = (repo[:27] + '...') if len(repo) > 30 else repo
            report_lines.append(f"{rank:<6} {repo_short:<30} {repo_data['total_instances']:<12} {repo_data['avg_types_per_instance']:<12.2f}")

        report_lines.append("")

    # Footer
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Save report
    report_path = Path(output_dir) / 'failure_type_statistics_report.txt'
    report_path.parent.mkdir(exist_ok=True, parents=True)

    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"   ✓ Report saved to {report_path}")

    return report_path


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive failure type statistics')
    parser.add_argument('--classifications', type=str,
                       default='results/failure_classifications_cleaned.json',
                       help='Path to classifications JSON')
    parser.add_argument('--dataset', type=str,
                       default='dataset/lca_dataset.parquet',
                       help='Path to dataset')
    parser.add_argument('--output-dir', type=str,
                       default='results',
                       help='Output directory for statistics')
    parser.add_argument('--format', type=str, choices=['json', 'txt', 'both'],
                       default='both',
                       help='Output format: json, txt, or both')

    args = parser.parse_args()

    print("=" * 80)
    print("FAILURE TYPE STATISTICS ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    classifications_data, df = load_data(args.classifications, args.dataset)
    classifications = classifications_data.get('classifications', [])

    # Calculate statistics
    stats = calculate_basic_statistics(classifications)
    cooccurrence = calculate_cooccurrence_matrix(classifications)
    repo_stats = calculate_repository_statistics(df)
    temporal_stats = calculate_temporal_statistics(df)

    # Compile all statistics
    all_stats = {
        'basic_statistics': stats,
        'cooccurrence': cooccurrence,
        'repository_statistics': repo_stats,
        'temporal_statistics': temporal_stats,
        'metadata': {
            'total_instances': len(classifications),
            'dataset_path': args.dataset,
            'classifications_path': args.classifications
        }
    }

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save JSON
    if args.format in ['json', 'both']:
        json_path = output_dir / 'failure_type_statistics.json'
        with open(json_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n💾 JSON statistics saved to {json_path}")

    # Generate and save text report
    if args.format in ['txt', 'both']:
        report_path = generate_summary_report(stats, cooccurrence, repo_stats,
                                             temporal_stats, str(output_dir))

        # Print summary to console
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total instances: {stats['total_instances']}")
        print(f"Unique failure types: {stats['total_failure_types']}")
        print(f"Average types per instance: {stats['avg_types_per_instance']:.2f}")
        print(f"\nTop 5 failure types:")
        sorted_types = sorted(stats['failure_type_counts'].items(), key=lambda x: x[1], reverse=True)
        for rank, (ftype, count) in enumerate(sorted_types[:5], 1):
            percentage = count / stats['total_instances'] * 100
            print(f"  {rank}. {ftype}: {count} ({percentage:.1f}%)")

        print(f"\n📊 Full report: {report_path}")

    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())

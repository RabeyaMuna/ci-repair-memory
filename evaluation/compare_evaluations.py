#!/usr/bin/env python3
"""
Compare different evaluation runs side-by-side.
Useful for tracking performance improvements across different models or approaches.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_result(path: Path) -> Dict:
    """Load evaluation result from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def print_comparison_table(results: List[Dict], labels: List[str]):
    """Print side-by-side comparison of multiple evaluation results."""

    print("\n" + "=" * 120)
    print("EVALUATION COMPARISON")
    print("=" * 120)

    # Header
    header = f"{'Metric':<30}"
    for label in labels:
        header += f" | {label:>20}"
    print(header)
    print("-" * 120)

    # Get aggregates
    aggs = [r['aggregate'] for r in results]

    # Evaluated instances
    row = f"{'Evaluated Instances':<30}"
    for agg in aggs:
        n = agg.get('evaluated_instances', agg.get('with_predictions', agg.get('total_instances', 0)))
        row += f" | {n:>20}"
    print(row)

    print("-" * 120)

    # Exact Match
    row = f"{'Exact Match Rate':<30}"
    for agg in aggs:
        rate = agg.get('exact_match_rate', 0)
        count = agg.get('exact_match', 0)
        n = agg.get('evaluated_instances', agg.get('with_predictions', 1))
        row += f" | {rate:>6.2f}% ({count:>2}/{n:<2})"
    print(row)

    print("-" * 120)

    # Top-K metrics
    for k in [1, 2, 3, 4, 5, 10, 15]:
        key = f'top_{k}'
        row = f"{f'Top-{k} Accuracy':<30}"
        for agg in aggs:
            rate = agg.get(f'{key}_rate', 0)
            count = agg.get(key, 0)
            n = agg.get('evaluated_instances', agg.get('with_predictions', 1))
            row += f" | {rate:>6.2f}% ({count:>2}/{n:<2})"
        print(row)

    print("-" * 120)

    # Precision
    row = f"{'Mean Precision':<30}"
    for agg in aggs:
        prec = agg.get('mean_precision', 0)
        row += f" | {prec:>20.4f}"
    print(row)

    print("=" * 120)


def print_improvement_analysis(baseline: Dict, current: Dict, baseline_label: str, current_label: str):
    """Print improvement analysis comparing current to baseline."""

    print("\n" + "=" * 80)
    print(f"IMPROVEMENT ANALYSIS: {current_label} vs {baseline_label}")
    print("=" * 80)

    base_agg = baseline['aggregate']
    curr_agg = current['aggregate']

    metrics = [
        ('exact_match_rate', 'Exact Match', '%'),
        ('top_1_rate', 'Top-1 Accuracy', '%'),
        ('top_3_rate', 'Top-3 Accuracy', '%'),
        ('top_5_rate', 'Top-5 Accuracy', '%'),
        ('top_10_rate', 'Top-10 Accuracy', '%'),
        ('mean_precision', 'Mean Precision', 'abs'),
    ]

    print(f"\n{'Metric':<25} {'Baseline':>12} {'Current':>12} {'Δ':>12} {'Δ%':>12}")
    print("-" * 80)

    for key, name, unit in metrics:
        base_val = base_agg.get(key, 0)
        curr_val = curr_agg.get(key, 0)
        delta = curr_val - base_val

        if base_val != 0:
            delta_pct = (delta / base_val) * 100
        else:
            delta_pct = 0 if delta == 0 else float('inf')

        # Format
        if unit == '%':
            base_str = f"{base_val:.2f}%"
            curr_str = f"{curr_val:.2f}%"
            delta_str = f"{delta:+.2f}%"
        else:
            base_str = f"{base_val:.4f}"
            curr_str = f"{curr_val:.4f}"
            delta_str = f"{delta:+.4f}"

        if delta_pct == float('inf'):
            delta_pct_str = "N/A"
        else:
            delta_pct_str = f"{delta_pct:+.1f}%"

        # Color code (emoji)
        if delta > 0:
            symbol = "📈"
        elif delta < 0:
            symbol = "📉"
        else:
            symbol = "➡️"

        print(f"{name:<25} {base_str:>12} {curr_str:>12} {delta_str:>12} {delta_pct_str:>10} {symbol}")

    print("=" * 80)


def print_detailed_comparison(results: List[Dict], labels: List[str]):
    """Print detailed per-instance comparison."""

    print("\n" + "=" * 100)
    print("PER-INSTANCE COMPARISON")
    print("=" * 100)

    # Get all instance IDs across all results
    all_ids = set()
    for result in results:
        for inst in result['per_instance']:
            all_ids.add(inst['id'])

    sorted_ids = sorted(all_ids, key=lambda x: int(x) if x.isdigit() else x)

    # Print header
    print(f"\n{'ID':<6}", end="")
    for label in labels:
        print(f" | {label + ' (T1/T3/T5/Prec)':<35}", end="")
    print()
    print("-" * 100)

    # Print each instance
    for inst_id in sorted_ids:
        print(f"{inst_id:<6}", end="")

        for result in results:
            # Find instance in this result
            inst_data = None
            for inst in result['per_instance']:
                if inst['id'] == inst_id:
                    inst_data = inst
                    break

            if inst_data:
                t1 = "✓" if inst_data.get('top_1') else "✗"
                t3 = "✓" if inst_data.get('top_3') else "✗"
                t5 = "✓" if inst_data.get('top_5') else "✗"
                prec = inst_data.get('precision', 0)
                info = f"{t1}/{t3}/{t5} P={prec:.3f}"
            else:
                info = "N/A"

            print(f" | {info:<35}", end="")

        print()

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple evaluation results side-by-side"
    )
    parser.add_argument(
        'results',
        nargs='+',
        help='Paths to evaluation result JSON files'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        help='Labels for each result (optional, defaults to filenames)'
    )
    parser.add_argument(
        '--baseline',
        type=int,
        default=0,
        help='Index of baseline result for improvement analysis (default: 0)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed per-instance comparison'
    )

    args = parser.parse_args()

    # Load results
    results = []
    for path_str in args.results:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        results.append(load_result(path))

    if not results:
        print("Error: No valid result files found")
        return

    # Generate labels
    if args.labels:
        labels = args.labels[:len(results)]
    else:
        labels = [Path(p).stem for p in args.results[:len(results)]]

    # Ensure we have enough labels
    while len(labels) < len(results):
        labels.append(f"Result {len(labels) + 1}")

    # Print comparison
    print_comparison_table(results, labels)

    # Print improvement analysis if we have multiple results
    if len(results) > 1 and args.baseline < len(results):
        for i in range(len(results)):
            if i != args.baseline:
                print_improvement_analysis(
                    results[args.baseline],
                    results[i],
                    labels[args.baseline],
                    labels[i]
                )

    # Print detailed comparison if requested
    if args.detailed:
        print_detailed_comparison(results, labels)


if __name__ == "__main__":
    main()

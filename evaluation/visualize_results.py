#!/usr/bin/env python3
"""
Visualize evaluation results with detailed breakdowns.
"""

import json
import pandas as pd
from pathlib import Path


def load_results(path='results/evaluation_results.json'):
    with open(path, 'r') as f:
        return json.load(f)


def print_topk_comparison(results):
    """Print Top-K comparison table."""
    df = pd.DataFrame(results['per_instance'])
    df_with_preds = df[df['has_prediction'] == True]

    print("\n" + "="*90)
    print("TOP-K ACCURACY COMPARISON")
    print("="*90)
    print(f"\n{'K':<8} {'Hits':<8} {'Rate (with preds)':<25} {'Overall Rate':<20}")
    print("-" * 90)

    n_with_preds = len(df_with_preds)
    n_total = results['aggregate']['total_instances']

    for k in [1, 2, 3, 4, 5, 10, 15]:
        col = f'top_{k}'
        hits = df_with_preds[col].sum()
        rate_with_preds = hits / n_with_preds * 100 if n_with_preds > 0 else 0
        overall_rate = results['aggregate'][f'{col}_rate']

        bar_with_preds = '█' * int(rate_with_preds / 5) + '░' * (20 - int(rate_with_preds / 5))
        bar_overall = '█' * int(overall_rate / 5) + '░' * (20 - int(overall_rate / 5))

        print(f"Top-{k:<3} {hits:<8} {rate_with_preds:>6.2f}% {bar_with_preds}   {overall_rate:>6.2f}% {bar_overall}")


def print_precision_breakdown(results):
    """Print precision breakdown by category."""
    df = pd.DataFrame(results['per_instance'])
    df_with_preds = df[df['has_prediction'] == True]

    print("\n" + "="*90)
    print("PRECISION BREAKDOWN")
    print("="*90)

    categories = [
        ("Perfect (1.0)", lambda x: x == 1.0),
        ("Very High (0.8-1.0)", lambda x: 0.8 <= x < 1.0),
        ("High (0.5-0.8)", lambda x: 0.5 <= x < 0.8),
        ("Medium (0.3-0.5)", lambda x: 0.3 <= x < 0.5),
        ("Low (0.1-0.3)", lambda x: 0.1 <= x < 0.3),
        ("Very Low (0-0.1)", lambda x: 0.0 < x < 0.1),
        ("Zero (0.0)", lambda x: x == 0.0),
    ]

    print(f"\n{'Category':<25} {'Count':<8} {'Percentage':<15} {'Visualization':<30}")
    print("-" * 90)

    total = len(df_with_preds)
    for category, condition in categories:
        count = df_with_preds['precision'].apply(condition).sum()
        pct = count / total * 100 if total > 0 else 0
        bar = '█' * int(count * 2) + '░' * (total * 2 - int(count * 2))
        print(f"{category:<25} {count:<8} {pct:>6.2f}%         {bar}")


def print_instance_details(results, sort_by='precision'):
    """Print detailed instance information."""
    df = pd.DataFrame(results['per_instance'])
    df_with_preds = df[df['has_prediction'] == True].copy()

    print("\n" + "="*120)
    print(f"INSTANCE DETAILS (sorted by {sort_by})")
    print("="*120)

    df_sorted = df_with_preds.sort_values(by=sort_by, ascending=False)

    print(f"\n{'ID':<6} {'Pred':<6} {'GT':<6} {'Match':<7} {'Prec':<8} {'EM':<4} {'T1':<4} {'T3':<4} {'T5':<4} {'T10':<5}")
    print("-" * 120)

    for _, row in df_sorted.iterrows():
        em_mark = '✓' if row['exact_match'] else '✗'
        t1_mark = '✓' if row['top_1'] else '✗'
        t3_mark = '✓' if row['top_3'] else '✗'
        t5_mark = '✓' if row['top_5'] else '✗'
        t10_mark = '✓' if row['top_10'] else '✗'

        print(f"{row['id']:<6} {row['num_predicted']:<6} {row['num_ground_truth']:<6} "
              f"{row['num_matched']:<7} {row['precision']:<8.4f} "
              f"{em_mark:<4} {t1_mark:<4} {t3_mark:<4} {t5_mark:<4} {t10_mark:<5}")


def print_success_vs_failure(results):
    """Compare successful vs failed predictions."""
    df = pd.DataFrame(results['per_instance'])
    df_with_preds = df[df['has_prediction'] == True]

    print("\n" + "="*90)
    print("SUCCESS vs FAILURE ANALYSIS")
    print("="*90)

    # Top-1 hits vs misses
    top1_hits = df_with_preds[df_with_preds['top_1'] == True]
    top1_misses = df_with_preds[df_with_preds['top_1'] == False]

    print(f"\nTop-1 HITS (n={len(top1_hits)}):")
    print(f"  Mean Precision:     {top1_hits['precision'].mean():.4f}")
    print(f"  Mean Pred Files:    {top1_hits['num_predicted'].mean():.1f}")
    print(f"  Mean GT Files:      {top1_hits['num_ground_truth'].mean():.1f}")
    print(f"  Mean Matched Files: {top1_hits['num_matched'].mean():.1f}")

    print(f"\nTop-1 MISSES (n={len(top1_misses)}):")
    print(f"  Mean Precision:     {top1_misses['precision'].mean():.4f}")
    print(f"  Mean Pred Files:    {top1_misses['num_predicted'].mean():.1f}")
    print(f"  Mean GT Files:      {top1_misses['num_ground_truth'].mean():.1f}")
    print(f"  Mean Matched Files: {top1_misses['num_matched'].mean():.1f}")


def main():
    print("\n" + "="*90)
    print("CI-REPAIR-BENCH EVALUATION VISUALIZATION")
    print("="*90)

    results = load_results()

    # Overall stats
    agg = results['aggregate']
    print(f"\nTotal Instances:        {agg['total_instances']}")
    print(f"With Predictions:       {agg['with_predictions']} ({agg['success_rate']:.2f}%)")
    print(f"Mean Precision:         {agg['mean_precision']:.4f}")

    # Various visualizations
    print_topk_comparison(results)
    print_precision_breakdown(results)
    print_success_vs_failure(results)
    print_instance_details(results, sort_by='precision')

    print("\n" + "="*90)
    print("LEGEND:")
    print("  Pred  = Number of predicted files")
    print("  GT    = Number of ground truth files")
    print("  Match = Number of matched files")
    print("  Prec  = Precision score")
    print("  EM    = Exact Match")
    print("  T1-T10 = Top-1 through Top-10 accuracy")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()

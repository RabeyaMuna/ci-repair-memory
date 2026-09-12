#!/usr/bin/env python3
"""
Verify predictions against ground truth from dataset
Computes Top-K accuracy, Exact Match, and Precision metrics
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


def extract_files_from_diff(diff_text: str) -> List[str]:
    """Extract changed file paths from diff text."""
    if not diff_text:
        return []

    files = []
    # Match "diff --git a/path b/path"
    for line in diff_text.split('\n'):
        if line.startswith('diff --git'):
            # Extract file path from "diff --git a/file b/file"
            match = re.search(r' b/(.+?)(?:\s|$)', line)
            if match:
                files.append(match.group(1))

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files


def load_predictions(preds_file: Path) -> Dict:
    """Load predictions from JSON file."""
    with open(preds_file, 'r') as f:
        preds = json.load(f)

    print(f"✓ Loaded {len(preds)} predictions from {preds_file}")

    # Extract predicted files from diffs or use predicted_files field
    predictions_with_files = {}
    has_predicted_files_field = 0
    extracted_from_diff = 0

    for issue_id, pred_data in preds.items():
        # Check if predicted_files field exists
        predicted = pred_data.get('predicted_files', [])

        if predicted:
            has_predicted_files_field += 1
        elif 'diff' in pred_data:
            predicted = extract_files_from_diff(pred_data['diff'])
            extracted_from_diff += 1

        if predicted:
            predictions_with_files[issue_id] = predicted

    print(f"  - {has_predicted_files_field} predictions had 'predicted_files' field")
    print(f"  - {extracted_from_diff} predictions extracted from diff")
    print(f"  - {len(predictions_with_files)} total predictions with files")

    return predictions_with_files


def load_ground_truth(dataset_file: Path, pushed_ids: Set[str] = None) -> Dict[str, Set[str]]:
    """Load ground truth from dataset parquet file."""
    df = pd.read_parquet(dataset_file)

    print(f"\n✓ Loaded dataset from {dataset_file}")
    print(f"  - Total rows: {len(df)}")

    gt_files = {}
    for _, row in df.iterrows():
        issue_id = str(row['id'])

        # Only include IDs that were pushed/validated if filter is provided
        if pushed_ids is not None and issue_id not in pushed_ids:
            continue

        changed_files = row.get('changed_files', [])
        if changed_files is not None and len(changed_files) > 0:
            gt_files[issue_id] = set(changed_files)

    print(f"  - {len(gt_files)} issues with ground truth files")
    if pushed_ids:
        print(f"  - Filtered to {len(pushed_ids)} pushed IDs")

    return gt_files


def get_pushed_ids(results_dir: Path) -> Set[str]:
    """Get IDs that were actually pushed/validated from CI results."""
    success_rate_file = results_dir / 'success_rate_evaluation.json'

    if not success_rate_file.exists():
        return None

    with open(success_rate_file, 'r') as f:
        data = json.load(f)
        pushed_ids = {str(r['id']) for r in data.get('results', [])}

    print(f"\n✓ Loaded {len(pushed_ids)} pushed IDs from {success_rate_file}")
    return pushed_ids


def compute_metrics(predictions: Dict[str, List[str]],
                   ground_truth: Dict[str, Set[str]],
                   k_values: List[int] = [1, 3, 5, 10, 15, 20]) -> Dict:
    """Compute all metrics by comparing predictions against ground truth."""

    print(f"\n{'='*80}")
    print("COMPUTING METRICS")
    print(f"{'='*80}")

    # Initialize counters
    total = 0
    exact_matches = 0
    precision_sum = 0.0
    topk_hits = {k: 0 for k in k_values}

    # For detailed analysis
    missing_gt = []
    missing_pred = []
    empty_predictions = []

    # Track per-instance results for detailed output
    results = []

    for issue_id in ground_truth.keys():
        if issue_id not in predictions:
            missing_pred.append(issue_id)
            continue

        total += 1
        predicted = predictions[issue_id]
        gt = ground_truth[issue_id]

        if not predicted:
            empty_predictions.append(issue_id)
            continue

        if not gt:
            continue

        pred_set = set(predicted)

        # Exact match: all predicted files match exactly with ground truth
        is_exact_match = pred_set == gt
        if is_exact_match:
            exact_matches += 1

        # Precision: fraction of predicted files that are correct
        intersection = pred_set & gt
        precision = len(intersection) / len(pred_set) if pred_set else 0
        precision_sum += precision

        # Top-K accuracy: if any of top-k predictions is in ground truth
        for k in k_values:
            top_k_files = set(predicted[:k])
            if top_k_files & gt:  # If intersection is non-empty
                topk_hits[k] += 1

        # Store result for detailed analysis
        results.append({
            'id': issue_id,
            'predicted': predicted,
            'ground_truth': list(gt),
            'exact_match': is_exact_match,
            'precision': precision,
            'top_1_correct': predicted[0] in gt if predicted else False,
            'num_predicted': len(predicted),
            'num_ground_truth': len(gt),
            'num_correct': len(intersection)
        })

    # Compute final metrics
    exact_match_count = exact_matches
    exact_match_rate = (exact_matches / total * 100) if total > 0 else 0
    avg_precision = (precision_sum / total * 100) if total > 0 else 0

    topk_accuracy = {
        k: {
            'count': hits,
            'rate': round((hits / total * 100), 2) if total > 0 else 0
        }
        for k, hits in topk_hits.items()
    }

    metrics = {
        'total_issues': total,
        'exact_match': {
            'count': exact_match_count,
            'rate': round(exact_match_rate, 2)
        },
        'precision': {
            'average': round(avg_precision, 2)
        },
        'top_k_accuracy': {f'top_{k}': v['rate'] for k, v in topk_accuracy.items()},
        'top_k_counts': {f'top_{k}': v['count'] for k, v in topk_accuracy.items()},
        'diagnostics': {
            'missing_predictions': len(missing_pred),
            'empty_predictions': len(empty_predictions),
            'missing_ground_truth': len(missing_gt)
        }
    }

    return metrics, results


def print_detailed_report(metrics: Dict, results: List[Dict], show_examples: bool = True):
    """Print comprehensive metrics report."""

    print(f"\n{'='*80}")
    print("GROUND TRUTH VERIFICATION REPORT")
    print(f"{'='*80}")

    print(f"\n📊 TOTAL EVALUATED: {metrics['total_issues']} issues")

    # Diagnostics
    diag = metrics['diagnostics']
    if any(diag.values()):
        print(f"\n⚠️  DIAGNOSTICS:")
        if diag['missing_predictions']:
            print(f"  - {diag['missing_predictions']} issues missing predictions")
        if diag['empty_predictions']:
            print(f"  - {diag['empty_predictions']} issues with empty predictions")
        if diag['missing_ground_truth']:
            print(f"  - {diag['missing_ground_truth']} issues missing ground truth")

    # Exact Match
    print(f"\n{'-'*80}")
    print("EXACT MATCH")
    print(f"{'-'*80}")
    em = metrics['exact_match']
    print(f"  Count: {em['count']} / {metrics['total_issues']}")
    print(f"  Rate:  {em['rate']:.2f}%")

    # Precision
    print(f"\n{'-'*80}")
    print("PRECISION")
    print(f"{'-'*80}")
    print(f"  Average: {metrics['precision']['average']:.2f}%")

    # Top-K Accuracy
    print(f"\n{'-'*80}")
    print("TOP-K ACCURACY")
    print(f"{'-'*80}")
    print(f"  {'K':<8} {'Accuracy':<12} {'Count':<15}")
    print(f"  {'-'*8} {'-'*12} {'-'*15}")

    for k_name, accuracy in sorted(metrics['top_k_accuracy'].items(),
                                   key=lambda x: int(x[0].split('_')[1])):
        k = k_name.split('_')[1]
        count_key = f'top_{k}'
        count = metrics['top_k_counts'][count_key]
        print(f"  {k:<8} {accuracy:>6.2f}%      {count:>4} / {metrics['total_issues']}")

    # Analysis
    print(f"\n{'-'*80}")
    print("INSIGHTS")
    print(f"{'-'*80}")

    top_k_acc = metrics['top_k_accuracy']
    top1 = top_k_acc['top_1']
    top3 = top_k_acc['top_3']
    top5 = top_k_acc['top_5']
    top10 = top_k_acc['top_10']

    print(f"  • Top-1 → Top-3: +{top3 - top1:.2f} percentage points")
    print(f"  • Top-3 → Top-5: +{top5 - top3:.2f} percentage points")
    print(f"  • Top-5 → Top-10: +{top10 - top5:.2f} percentage points")

    # Distribution analysis
    if results:
        precisions = [r['precision'] for r in results]
        num_predicted = [r['num_predicted'] for r in results]
        num_gt = [r['num_ground_truth'] for r in results]

        print(f"\n  Distribution:")
        print(f"    - Avg predicted files per issue: {sum(num_predicted)/len(num_predicted):.2f}")
        print(f"    - Avg ground truth files per issue: {sum(num_gt)/len(num_gt):.2f}")
        print(f"    - Precision distribution:")
        print(f"      - 100% precision: {sum(1 for p in precisions if p == 1.0)} issues")
        print(f"      - 50-99% precision: {sum(1 for p in precisions if 0.5 <= p < 1.0)} issues")
        print(f"      - 1-49% precision: {sum(1 for p in precisions if 0 < p < 0.5)} issues")
        print(f"      - 0% precision: {sum(1 for p in precisions if p == 0)} issues")

    # Show examples
    if show_examples and results:
        print(f"\n{'-'*80}")
        print("EXAMPLE CASES")
        print(f"{'-'*80}")

        # Perfect predictions
        perfect = [r for r in results if r['exact_match']][:3]
        if perfect:
            print(f"\n  ✓ Perfect Matches (showing {len(perfect)}):")
            for r in perfect:
                print(f"    ID {r['id']}: {len(r['predicted'])} files predicted, all correct")
                print(f"      Predicted: {r['predicted'][:3]}")

        # Good top-1
        good_top1 = [r for r in results if r['top_1_correct'] and not r['exact_match']][:3]
        if good_top1:
            print(f"\n  ✓ Correct Top-1 (but not exact match) (showing {len(good_top1)}):")
            for r in good_top1:
                print(f"    ID {r['id']}: Top file correct, precision={r['precision']:.2%}")
                print(f"      Top-1: {r['predicted'][0]}")
                print(f"      Ground truth: {r['ground_truth']}")

        # Failed cases
        failed = [r for r in results if not r['top_1_correct']][:3]
        if failed:
            print(f"\n  ✗ Top-1 Misses (showing {len(failed)}):")
            for r in failed:
                print(f"    ID {r['id']}: precision={r['precision']:.2%}")
                print(f"      Predicted: {r['predicted'][:3]}")
                print(f"      Ground truth: {r['ground_truth']}")

    print(f"\n{'='*80}\n")


def main():
    # File paths
    base_dir = Path('/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH')
    preds_file = base_dir / 'results/preds.json'
    dataset_file = base_dir / 'dataset/lca_dataset.parquet'
    results_dir = base_dir / 'results'

    print(f"{'='*80}")
    print("GROUND TRUTH VERIFICATION")
    print(f"{'='*80}")

    # Load data
    predictions = load_predictions(preds_file)

    # Get pushed IDs (optional filter)
    pushed_ids = get_pushed_ids(results_dir)

    # Load ground truth
    ground_truth = load_ground_truth(dataset_file, pushed_ids)

    # Compute metrics
    metrics, results = compute_metrics(predictions, ground_truth)

    # Print report
    print_detailed_report(metrics, results, show_examples=True)

    # Save detailed results
    output_file = base_dir / 'results/ground_truth_verification.json'
    with open(output_file, 'w') as f:
        json.dump({
            'metrics': metrics,
            'sample_results': results[:20]  # Save first 20 for inspection
        }, f, indent=2)

    print(f"✓ Saved detailed results to: {output_file}")

    # Compare with existing metrics file
    existing_metrics_file = results_dir / 'file_localization_metrics.json'
    if existing_metrics_file.exists():
        with open(existing_metrics_file, 'r') as f:
            existing = json.load(f)

        print(f"\n{'='*80}")
        print("COMPARISON WITH EXISTING METRICS FILE")
        print(f"{'='*80}")
        print(f"\nExisting metrics file: {existing_metrics_file}")
        print(f"\n  {'Metric':<30} {'Existing':<15} {'Computed':<15} {'Match':<10}")
        print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*10}")

        print(f"  {'Total Issues':<30} {existing['total_issues']:<15} {metrics['total_issues']:<15} {'✓' if existing['total_issues'] == metrics['total_issues'] else '✗'}")
        print(f"  {'Exact Match Rate':<30} {existing['exact_match']['rate']:<15} {metrics['exact_match']['rate']:<15} {'✓' if abs(existing['exact_match']['rate'] - metrics['exact_match']['rate']) < 0.01 else '✗'}")
        print(f"  {'Avg Precision':<30} {existing['precision']['average']:<15} {metrics['precision']['average']:<15} {'✓' if abs(existing['precision']['average'] - metrics['precision']['average']) < 0.01 else '✗'}")

        for k in [1, 3, 5, 10, 15]:
            key = f'top_{k}'
            if key in existing['top_k_accuracy'] and key in metrics['top_k_accuracy']:
                existing_val = existing['top_k_accuracy'][key]
                computed_val = metrics['top_k_accuracy'][key]
                match = '✓' if abs(existing_val - computed_val) < 0.01 else '✗'
                print(f"  {f'Top-{k} Accuracy':<30} {existing_val:<15} {computed_val:<15} {match}")

        print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()

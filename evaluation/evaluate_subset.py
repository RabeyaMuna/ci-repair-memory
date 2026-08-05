#!/usr/bin/env python3
"""
Evaluation script for CI-REPAIR-BENCH predictions - SUBSET ANALYSIS.

Analyzes ONLY the instances that have predictions (not the full dataset).
This shows the true performance of the approach on evaluated instances.

Usage:
    python evaluate_subset.py --preds results/preds.json --dataset dataset/lca_dataset.parquet
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


def normalize_path(path: str) -> str:
    """Normalize file paths for comparison."""
    return path.strip().replace("\\", "/").lstrip("/")


def extract_files_from_diff(diff_text: str) -> List[str]:
    """
    Extract file paths from unified diff.
    Parses 'diff --git a/... b/...' headers and returns the target files (b/ side).
    """
    files = []
    pattern = re.compile(r"^diff --git a/.+ b/(.+)$", re.MULTILINE)
    for match in pattern.finditer(diff_text):
        file_path = match.group(1).strip()
        if file_path:
            files.append(normalize_path(file_path))
    return files


def load_ground_truth(dataset_path: Path, instance_ids: Set[str]) -> Dict[str, Set[str]]:
    """
    Load ground truth ONLY for specified instance IDs.
    Returns: {id -> set of ground truth file paths}
    """
    df = pd.read_parquet(dataset_path)
    ground_truth = {}

    for _, row in df.iterrows():
        instance_id = str(row['id'])

        # Only load GT for instances we have predictions for
        if instance_id not in instance_ids:
            continue

        diff = row.get('diff', '')

        if diff:
            gt_files = extract_files_from_diff(diff)
            ground_truth[instance_id] = set(gt_files)
        else:
            ground_truth[instance_id] = set()

    print(f"Loaded ground truth for {len(ground_truth)} evaluated instances")
    return ground_truth


def load_predictions(preds_path: Path) -> Dict[str, List[str]]:
    """
    Load predictions from preds.json.
    Returns: {id -> list of predicted file paths (ranked)}
    """
    with open(preds_path, 'r') as f:
        preds_data = json.load(f)

    predictions = {}

    for instance_id, data in preds_data.items():
        diff = data.get('diff', '')

        if diff:
            # Extract files from predicted diff (in order they appear)
            pred_files = extract_files_from_diff(diff)
            if pred_files:  # Only include if there are actual predictions
                predictions[instance_id] = pred_files

    print(f"Loaded predictions for {len(predictions)} instances")
    return predictions


def calculate_metrics(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    max_k: int = 15
) -> Dict:
    """
    Calculate evaluation metrics for EVALUATED INSTANCES ONLY.

    Metrics calculated over instances with predictions:
    - Exact Match: predicted set == ground truth set
    - Top-K (1 to max_k): any ground truth file in top K predictions
    - Precision: |predicted ∩ GT| / |predicted|
    """

    results = {
        'per_instance': [],
        'aggregate': {
            'evaluated_instances': 0,  # Instances with predictions
            'exact_match': 0,
            'total_precision': 0.0,
        }
    }

    # Initialize top-k counters
    for k in range(1, max_k + 1):
        results['aggregate'][f'top_{k}'] = 0

    # Iterate only over instances with predictions
    for instance_id in sorted(predictions.keys(), key=lambda x: int(x) if x.isdigit() else x):
        pred_files = predictions[instance_id]
        gt_files = ground_truth.get(instance_id, set())
        pred_set = set(pred_files)

        results['aggregate']['evaluated_instances'] += 1

        # Exact Match
        exact_match = (pred_set == gt_files) and bool(gt_files)
        if exact_match:
            results['aggregate']['exact_match'] += 1

        # Top-K accuracy
        top_k_hits = {}
        for k in range(1, max_k + 1):
            hit = any(pred_file in gt_files for pred_file in pred_files[:k])
            top_k_hits[f'top_{k}'] = hit
            if hit:
                results['aggregate'][f'top_{k}'] += 1

        # Precision
        if pred_set:
            matched_files = pred_set & gt_files
            precision = len(matched_files) / len(pred_set)
        else:
            precision = 0.0

        results['aggregate']['total_precision'] += precision

        # Store per-instance result
        instance_result = {
            'id': instance_id,
            'num_predicted': len(pred_files),
            'num_ground_truth': len(gt_files),
            'num_matched': len(pred_set & gt_files),
            'exact_match': exact_match,
            'precision': round(precision, 4),
            'predicted_files': pred_files,
            'ground_truth_files': sorted(list(gt_files)),
        }
        instance_result.update(top_k_hits)

        results['per_instance'].append(instance_result)

    # Calculate aggregate percentages (based on evaluated instances only)
    n_eval = results['aggregate']['evaluated_instances']
    if n_eval > 0:
        results['aggregate']['exact_match_rate'] = round(
            results['aggregate']['exact_match'] / n_eval * 100, 2
        )
        results['aggregate']['mean_precision'] = round(
            results['aggregate']['total_precision'] / n_eval, 4
        )

        for k in range(1, max_k + 1):
            key = f'top_{k}'
            count = results['aggregate'][key]
            results['aggregate'][f'{key}_rate'] = round(count / n_eval * 100, 2)

    return results


def print_summary(results: Dict, max_k: int = 15):
    """Print evaluation summary for evaluated subset."""
    agg = results['aggregate']
    n_eval = agg['evaluated_instances']

    print("\n" + "=" * 80)
    print("SUBSET EVALUATION RESULTS (Evaluated Instances Only)")
    print("=" * 80)
    print(f"\nEvaluated Instances:  {n_eval}")
    print(f"(Analyzing performance on instances where predictions were generated)")

    print(f"\n{'─' * 80}")
    print("EXACT MATCH")
    print(f"{'─' * 80}")
    print(f"Exact Match Rate:     {agg['exact_match_rate']:.2f}% ({agg['exact_match']}/{n_eval})")

    print(f"\n{'─' * 80}")
    print("TOP-K ACCURACY (File-Level)")
    print(f"{'─' * 80}")

    # Print all k values
    for start_k in range(1, max_k + 1, 5):
        end_k = min(start_k + 4, max_k)
        line_parts = []
        for k in range(start_k, end_k + 1):
            key = f'top_{k}'
            count = agg[key]
            rate = agg[f'{key}_rate']
            line_parts.append(f"Top-{k:2d}: {rate:6.2f}% ({count}/{n_eval})")
        print("  " + "    ".join(line_parts))

    print(f"\n{'─' * 80}")
    print("PRECISION")
    print(f"{'─' * 80}")
    print(f"Mean Precision:       {agg['mean_precision']:.4f}")

    # Statistical summary
    precisions = [inst['precision'] for inst in results['per_instance']]
    print(f"\nPrecision Distribution:")
    print(f"  Min:        {min(precisions):.4f}")
    print(f"  Q1 (25%):   {sorted(precisions)[len(precisions)//4]:.4f}")
    print(f"  Median:     {sorted(precisions)[len(precisions)//2]:.4f}")
    print(f"  Q3 (75%):   {sorted(precisions)[3*len(precisions)//4]:.4f}")
    print(f"  Max:        {max(precisions):.4f}")

    # Precision categories
    perfect = sum(1 for p in precisions if p == 1.0)
    high = sum(1 for p in precisions if p >= 0.5)
    medium = sum(1 for p in precisions if 0.3 <= p < 0.5)
    low = sum(1 for p in precisions if 0 < p < 0.3)
    zero = sum(1 for p in precisions if p == 0.0)

    print(f"\nPrecision Categories:")
    print(f"  Perfect (1.0):        {perfect:2d} ({perfect/n_eval*100:5.1f}%)")
    print(f"  High (≥0.5):          {high:2d} ({high/n_eval*100:5.1f}%)")
    print(f"  Medium (0.3-0.5):     {medium:2d} ({medium/n_eval*100:5.1f}%)")
    print(f"  Low (0-0.3):          {low:2d} ({low/n_eval*100:5.1f}%)")
    print(f"  Zero (0.0):           {zero:2d} ({zero/n_eval*100:5.1f}%)")

    print(f"\n{'=' * 80}\n")


def save_results(results: Dict, output_path: Path):
    """Save detailed results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {output_path}")


def save_summary_csv(results: Dict, output_path: Path):
    """Save per-instance results to CSV."""
    df = pd.DataFrame(results['per_instance'])
    df.to_csv(output_path, index=False)
    print(f"Per-instance CSV saved to: {output_path}")


def generate_summary_table(results: Dict) -> str:
    """Generate a compact summary table."""
    agg = results['aggregate']
    n = agg['evaluated_instances']

    table = f"""
QUICK SUMMARY (Evaluated Instances: {n})
{'=' * 60}

Metric                    | Count      | Rate (%)
------------------------- | ---------- | ----------
Exact Match               | {agg['exact_match']:10d} | {agg['exact_match_rate']:9.2f}%
Top-1 Accuracy            | {agg['top_1']:10d} | {agg['top_1_rate']:9.2f}%
Top-3 Accuracy            | {agg['top_3']:10d} | {agg['top_3_rate']:9.2f}%
Top-5 Accuracy            | {agg['top_5']:10d} | {agg['top_5_rate']:9.2f}%
Top-10 Accuracy           | {agg['top_10']:10d} | {agg['top_10_rate']:9.2f}%
Top-15 Accuracy           | {agg['top_15']:10d} | {agg['top_15_rate']:9.2f}%
Mean Precision            | {agg['mean_precision']:10.4f} | -

Key Insights:
- {agg['top_1']} out of {n} instances ({agg['top_1_rate']:.1f}%) have a GT file in rank 1
- {agg['top_3']} out of {n} instances ({agg['top_3_rate']:.1f}%) have a GT file in top 3
- Improvement from Top-1 to Top-3: +{agg['top_3_rate'] - agg['top_1_rate']:.1f}%
- Improvement from Top-3 to Top-5: +{agg['top_5_rate'] - agg['top_3_rate']:.1f}%
"""
    return table


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CI-REPAIR-BENCH predictions (SUBSET: evaluated instances only)"
    )
    parser.add_argument(
        '--preds',
        type=str,
        default='results/preds.json',
        help='Path to predictions JSON file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset/lca_dataset.parquet',
        help='Path to ground truth dataset (parquet)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/evaluation_subset.json',
        help='Path to save detailed results JSON'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='results/evaluation_subset.csv',
        help='Path to save per-instance CSV'
    )
    parser.add_argument(
        '--max-k',
        type=int,
        default=15,
        help='Maximum K for Top-K accuracy (default: 15)'
    )

    args = parser.parse_args()

    # Resolve paths
    preds_path = Path(args.preds)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    csv_path = Path(args.csv)

    # Check paths exist
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"\n{'=' * 80}")
    print("SUBSET EVALUATION - Analyzing Evaluated Instances Only")
    print(f"{'=' * 80}")
    print(f"\nPredictions file: {preds_path}")
    print(f"Dataset file:     {dataset_path}")
    print(f"Maximum K:        {args.max_k}\n")

    # Load predictions first to know which instances to load GT for
    predictions = load_predictions(preds_path)

    # Load ground truth only for instances we have predictions for
    instance_ids = set(predictions.keys())
    ground_truth = load_ground_truth(dataset_path, instance_ids)

    # Calculate metrics
    print("\nCalculating metrics...")
    results = calculate_metrics(predictions, ground_truth, max_k=args.max_k)

    # Print summary
    print_summary(results, max_k=args.max_k)

    # Generate and print quick summary
    summary_table = generate_summary_table(results)
    print(summary_table)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output_path)
    save_summary_csv(results, csv_path)

    # Save summary table
    summary_path = output_path.parent / 'evaluation_subset_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary_table)
    print(f"Summary table saved to: {summary_path}")

    print("\n✓ Subset evaluation complete!")


if __name__ == "__main__":
    main()

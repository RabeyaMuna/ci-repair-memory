#!/usr/bin/env python3
"""
Evaluation script for CI-REPAIR-BENCH predictions.

Calculates metrics:
- Exact Match: predicted files exactly match ground truth files
- Top-K (1-15): ground truth file appears in top K predictions
- Precision: |predicted ∩ ground_truth| / |predicted|
- Success Rate: % of instances with valid predictions

Usage:
    python evaluate_predictions.py --preds results/preds.json --dataset dataset/lca_dataset.parquet
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

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


def load_ground_truth(dataset_path: Path) -> Dict[str, Set[str]]:
    """
    Load ground truth from dataset parquet file.
    Returns: {id -> set of ground truth file paths}
    """
    df = pd.read_parquet(dataset_path)
    ground_truth = {}

    for _, row in df.iterrows():
        instance_id = str(row['id'])
        diff = row.get('diff', '')

        if diff:
            gt_files = extract_files_from_diff(diff)
            ground_truth[instance_id] = set(gt_files)
        else:
            ground_truth[instance_id] = set()

    print(f"Loaded ground truth for {len(ground_truth)} instances")
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
            predictions[instance_id] = pred_files
        else:
            predictions[instance_id] = []

    print(f"Loaded predictions for {len(predictions)} instances")
    return predictions


def calculate_metrics(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    max_k: int = 15
) -> Dict:
    """
    Calculate evaluation metrics.

    Metrics:
    - Exact Match: predicted set == ground truth set
    - Top-K (1 to max_k): any ground truth file in top K predictions
    - Precision: |predicted ∩ GT| / |predicted|
    - Success Rate: % with predictions
    """

    results = {
        'per_instance': [],
        'aggregate': {
            'total_instances': 0,
            'with_predictions': 0,
            'with_ground_truth': 0,
            'exact_match': 0,
            'total_precision': 0.0,
        }
    }

    # Initialize top-k counters
    for k in range(1, max_k + 1):
        results['aggregate'][f'top_{k}'] = 0

    # Get all instance IDs from ground truth
    all_ids = sorted(ground_truth.keys(), key=lambda x: int(x) if x.isdigit() else x)

    for instance_id in all_ids:
        gt_files = ground_truth.get(instance_id, set())
        pred_files = predictions.get(instance_id, [])
        pred_set = set(pred_files)

        # Skip if no ground truth
        if not gt_files:
            continue

        results['aggregate']['total_instances'] += 1
        results['aggregate']['with_ground_truth'] += 1

        has_prediction = len(pred_files) > 0
        if has_prediction:
            results['aggregate']['with_predictions'] += 1

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
            'has_prediction': has_prediction,
            'predicted_files': pred_files,
            'ground_truth_files': sorted(list(gt_files)),
        }
        instance_result.update(top_k_hits)

        results['per_instance'].append(instance_result)

    # Calculate aggregate percentages
    n_total = results['aggregate']['total_instances']
    if n_total > 0:
        results['aggregate']['success_rate'] = round(
            results['aggregate']['with_predictions'] / n_total * 100, 2
        )
        results['aggregate']['exact_match_rate'] = round(
            results['aggregate']['exact_match'] / n_total * 100, 2
        )
        results['aggregate']['mean_precision'] = round(
            results['aggregate']['total_precision'] / n_total, 4
        )

        for k in range(1, max_k + 1):
            key = f'top_{k}'
            count = results['aggregate'][key]
            results['aggregate'][f'{key}_rate'] = round(count / n_total * 100, 2)

    return results


def print_summary(results: Dict, max_k: int = 15):
    """Print evaluation summary."""
    agg = results['aggregate']
    n_total = agg['total_instances']

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nTotal Instances:      {n_total}")
    print(f"With Predictions:     {agg['with_predictions']} ({agg['success_rate']:.2f}%)")
    print(f"With Ground Truth:    {agg['with_ground_truth']}")

    print(f"\n{'─' * 80}")
    print("EXACT MATCH")
    print(f"{'─' * 80}")
    print(f"Exact Match Rate:     {agg['exact_match_rate']:.2f}% ({agg['exact_match']}/{n_total})")

    print(f"\n{'─' * 80}")
    print("TOP-K ACCURACY (File-Level)")
    print(f"{'─' * 80}")

    # Print in groups of 5
    for start_k in range(1, max_k + 1, 5):
        end_k = min(start_k + 4, max_k)
        line_parts = []
        for k in range(start_k, end_k + 1):
            key = f'top_{k}'
            count = agg[key]
            rate = agg[f'{key}_rate']
            line_parts.append(f"Top-{k:2d}: {rate:6.2f}% ({count}/{n_total})")
        print("  " + "    ".join(line_parts))

    print(f"\n{'─' * 80}")
    print("PRECISION")
    print(f"{'─' * 80}")
    print(f"Mean Precision:       {agg['mean_precision']:.4f}")

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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CI-REPAIR-BENCH predictions against ground truth"
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
        default='results/evaluation_results.json',
        help='Path to save detailed results JSON'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='results/evaluation_per_instance.csv',
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

    print(f"\nLoading predictions from: {preds_path}")
    print(f"Loading ground truth from: {dataset_path}")
    print(f"Maximum K: {args.max_k}\n")

    # Load data
    predictions = load_predictions(preds_path)
    ground_truth = load_ground_truth(dataset_path)

    # Calculate metrics
    print("\nCalculating metrics...")
    results = calculate_metrics(predictions, ground_truth, max_k=args.max_k)

    # Print summary
    print_summary(results, max_k=args.max_k)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output_path)
    save_summary_csv(results, csv_path)

    print("Evaluation complete!")


if __name__ == "__main__":
    main()

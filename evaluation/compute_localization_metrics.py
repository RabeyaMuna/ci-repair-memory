#!/usr/bin/env python3
"""
Compute Localization Metrics for CI Repair.

Metrics:
- Exact Match: Entire predicted file set must exactly match ground-truth file set
- Precision: |Predicted ∩ Ground Truth| / |Predicted|
- Recall: |Predicted ∩ Ground Truth| / |Ground Truth|
- F1: 2 * (Precision * Recall) / (Precision + Recall)
- Top-K Recall: At least one GT file appears in top-K predictions

Usage:
    python evaluation/compute_localization_metrics.py \
        --preds results/preds.json \
        --dataset dataset/lca_dataset.parquet \
        --output evaluation/results/localization_metrics.json
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
    """Extract file paths from unified diff (maintains order)."""
    files = []
    pattern = re.compile(r"^diff --git a/.+ b/(.+)$", re.MULTILINE)
    for match in pattern.finditer(diff_text):
        file_path = match.group(1).strip()
        if file_path:
            files.append(normalize_path(file_path))
    return files


def load_predictions(preds_path: Path) -> Dict[str, List[str]]:
    """Load predictions from JSON file."""
    with open(preds_path, 'r') as f:
        data = json.load(f)

    predictions = {}
    for instance_id, instance_data in data.items():
        # Try both 'diff' and 'patch' keys
        patch = instance_data.get('diff', '') or instance_data.get('patch', '')
        if patch:
            pred_files = extract_files_from_diff(patch)
            predictions[str(instance_id)] = pred_files

    return predictions


def load_ground_truth(dataset_path: Path, instance_ids: Set[str]) -> Dict[str, Set[str]]:
    """Load ground truth for specified instance IDs."""
    df = pd.read_parquet(dataset_path)
    ground_truth = {}

    for _, row in df.iterrows():
        instance_id = str(row['id'])
        if instance_id not in instance_ids:
            continue

        # Extract ground truth files from diff/patch
        patch = row.get('diff', '') or row.get('ground_truth_patch', '') or row.get('patch', '')
        if patch:
            gt_files = extract_files_from_diff(patch)
            ground_truth[instance_id] = set(normalize_path(f) for f in gt_files)

    return ground_truth


def compute_metrics(
    predicted_files: List[str],
    ground_truth_files: Set[str]
) -> Dict[str, float]:
    """
    Compute Precision, Recall, F1, and Exact Match for a single instance.

    Args:
        predicted_files: List of predicted files (maintains order for Top-K)
        ground_truth_files: Set of ground truth files

    Returns:
        Dictionary with metrics
    """
    pred_set = set(predicted_files)
    gt_set = ground_truth_files

    # Intersection
    intersection = pred_set & gt_set

    # Precision = |Predicted ∩ Ground Truth| / |Predicted|
    precision = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0.0

    # Recall = |Predicted ∩ Ground Truth| / |Ground Truth|
    recall = len(intersection) / len(gt_set) if len(gt_set) > 0 else 0.0

    # F1 Score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Exact Match: Entire predicted set == ground truth set
    exact_match = 1.0 if pred_set == gt_set else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match': exact_match,
        'num_predicted': len(pred_set),
        'num_ground_truth': len(gt_set),
        'num_correct': len(intersection)
    }


def compute_topk_recall(
    predicted_files: List[str],
    ground_truth_files: Set[str],
    k_values: List[int]
) -> Dict[str, bool]:
    """
    Compute Top-K Recall: Does at least one GT file appear in top-K predictions?

    Args:
        predicted_files: List of predicted files (ordered)
        ground_truth_files: Set of ground truth files
        k_values: List of K values to evaluate

    Returns:
        Dictionary mapping k -> boolean (True if at least one GT in top-k)
    """
    topk_recall = {}

    for k in k_values:
        # Take top-k predictions
        top_k_preds = set(predicted_files[:k])
        # Check if at least one GT file is in top-k
        has_gt_in_topk = len(top_k_preds & ground_truth_files) > 0
        topk_recall[f'top_{k}'] = has_gt_in_topk

    return topk_recall


def main():
    parser = argparse.ArgumentParser(
        description="Compute Localization Metrics for CI Repair"
    )
    parser.add_argument(
        '--preds',
        type=Path,
        required=True,
        help='Path to predictions JSON file'
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Path to dataset parquet file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Path to save results JSON (optional)'
    )
    parser.add_argument(
        '--k-values',
        type=str,
        default='1,3,5,10,15',
        help='Comma-separated K values for Top-K Recall (default: 1,3,5,10,15)'
    )

    args = parser.parse_args()

    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(',')]

    print(f"\nLoading predictions from: {args.preds}")
    predictions = load_predictions(args.preds)

    print(f"Loading ground truth from: {args.dataset}")
    instance_ids = set(predictions.keys())
    ground_truth = load_ground_truth(args.dataset, instance_ids)

    print(f"\nEvaluating {len(predictions)} instances...")
    print(f"Computing metrics with K values: {k_values}\n")

    # Compute per-instance metrics
    per_instance_results = []
    aggregate_metrics = {
        'exact_match': [],
        'precision': [],
        'recall': [],
        'f1': [],
    }

    for k in k_values:
        aggregate_metrics[f'top_{k}_recall'] = []

    for instance_id in sorted(predictions.keys(), key=lambda x: int(x)):
        pred_files = predictions[instance_id]
        gt_files = ground_truth.get(instance_id, set())

        if not gt_files:
            print(f"Warning: No ground truth for instance {instance_id}, skipping...")
            continue

        # Compute basic metrics
        metrics = compute_metrics(pred_files, gt_files)

        # Compute Top-K Recall
        topk_recall = compute_topk_recall(pred_files, gt_files, k_values)

        # Store results
        instance_result = {
            'instance_id': instance_id,
            **metrics,
            **topk_recall,
            'predicted_files': pred_files,
            'ground_truth_files': sorted(list(gt_files))
        }
        per_instance_results.append(instance_result)

        # Aggregate
        aggregate_metrics['exact_match'].append(metrics['exact_match'])
        aggregate_metrics['precision'].append(metrics['precision'])
        aggregate_metrics['recall'].append(metrics['recall'])
        aggregate_metrics['f1'].append(metrics['f1'])

        for k in k_values:
            aggregate_metrics[f'top_{k}_recall'].append(1.0 if topk_recall[f'top_{k}'] else 0.0)

    # Compute aggregate statistics
    n = len(per_instance_results)
    aggregate_stats = {
        'total_instances': n,
        'exact_match': {
            'count': sum(aggregate_metrics['exact_match']),
            'percentage': (sum(aggregate_metrics['exact_match']) / n * 100) if n > 0 else 0.0
        },
        'precision': {
            'mean': sum(aggregate_metrics['precision']) / n if n > 0 else 0.0,
            'median': sorted(aggregate_metrics['precision'])[n // 2] if n > 0 else 0.0
        },
        'recall': {
            'mean': sum(aggregate_metrics['recall']) / n if n > 0 else 0.0,
            'median': sorted(aggregate_metrics['recall'])[n // 2] if n > 0 else 0.0
        },
        'f1': {
            'mean': sum(aggregate_metrics['f1']) / n if n > 0 else 0.0,
            'median': sorted(aggregate_metrics['f1'])[n // 2] if n > 0 else 0.0
        }
    }

    # Top-K Recall percentages
    for k in k_values:
        count = sum(aggregate_metrics[f'top_{k}_recall'])
        aggregate_stats[f'top_{k}_recall'] = {
            'count': int(count),
            'percentage': (count / n * 100) if n > 0 else 0.0
        }

    # Print results
    print("=" * 100)
    print("LOCALIZATION METRICS EVALUATION")
    print("=" * 100)
    print(f"\nTotal Instances: {n}\n")

    print("-" * 100)
    print("EXACT MATCH")
    print("-" * 100)
    print(f"  Instances with exact match: {aggregate_stats['exact_match']['count']}/{n}")
    print(f"  Percentage: {aggregate_stats['exact_match']['percentage']:.2f}%\n")

    print("-" * 100)
    print("PRECISION (|Predicted ∩ Ground Truth| / |Predicted|)")
    print("-" * 100)
    print(f"  Mean: {aggregate_stats['precision']['mean']:.4f}")
    print(f"  Median: {aggregate_stats['precision']['median']:.4f}\n")

    print("-" * 100)
    print("RECALL (|Predicted ∩ Ground Truth| / |Ground Truth|)")
    print("-" * 100)
    print(f"  Mean: {aggregate_stats['recall']['mean']:.4f}")
    print(f"  Median: {aggregate_stats['recall']['median']:.4f}\n")

    print("-" * 100)
    print("F1 SCORE")
    print("-" * 100)
    print(f"  Mean: {aggregate_stats['f1']['mean']:.4f}")
    print(f"  Median: {aggregate_stats['f1']['median']:.4f}\n")

    print("-" * 100)
    print("TOP-K RECALL (At least one GT file in top-K predictions)")
    print("-" * 100)
    for k in k_values:
        stats = aggregate_stats[f'top_{k}_recall']
        print(f"  Top-{k:<2}: {stats['count']}/{n} instances ({stats['percentage']:.2f}%)")
    print()

    print("=" * 100)

    # Prepare output
    results = {
        'summary': aggregate_stats,
        'per_instance': per_instance_results
    }

    # Save if output path provided
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")

    print("\n✓ Evaluation complete!\n")


if __name__ == '__main__':
    main()

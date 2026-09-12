#!/usr/bin/env python3
"""
File Localization Evaluation
=============================

Computes file localization metrics against ground truth:
- Top-K Accuracy (Top-1, Top-3, Top-5, Top-10, Top-15)
- Precision
- Exact Match

Evaluates only against pushed/validated IDs (not full dataset).

Usage:
    python scripts/analysis/evaluate_file_localization.py
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def extract_files_from_diff(diff_text: str) -> list:
    """Extract changed file paths from diff text."""
    if not diff_text:
        return []

    import re
    files = []
    # Match "diff --git a/path b/path"
    for line in diff_text.split('\n'):
        if line.startswith('diff --git'):
            # Extract file path from "diff --git a/file b/file"
            # Use space before b/ to match the second occurrence (destination path)
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


def get_pushed_ids(results_dir: Path) -> Optional[set]:
    """Get IDs that were actually pushed/validated from CI results."""
    success_rate_file = results_dir / 'success_rate_evaluation.json'

    if not success_rate_file.exists():
        return None

    with open(success_rate_file, 'r') as f:
        data = json.load(f)
        return {str(r['id']) for r in data.get('results', [])}


def compute_file_localization_metrics(preds_file: Path, dataset_file: Path, pushed_ids: Optional[set] = None) -> Dict:
    """Compute file localization metrics: Top-K accuracy, Precision, Exact Match.

    Args:
        preds_file: Path to predictions JSON
        dataset_file: Path to dataset parquet
        pushed_ids: Set of IDs that were actually pushed/validated. If provided,
                   only evaluate against these IDs (not the full dataset).
    """
    with open(preds_file, 'r') as f:
        preds = json.load(f)

    df = pd.read_parquet(dataset_file)

    gt_files = {}
    for _, row in df.iterrows():
        issue_id = str(row['id'])

        # Only include IDs that were pushed/validated if filter is provided
        if pushed_ids is not None and issue_id not in pushed_ids:
            continue

        changed_files = row.get('changed_files', [])
        if changed_files is not None and len(changed_files) > 0:
            gt_files[issue_id] = set(changed_files)

    total = 0
    exact_matches = 0
    precision_sum = 0
    topk_hits = {k: 0 for k in [1, 3, 5, 10, 15]}

    for issue_id, pred_data in preds.items():
        if issue_id not in gt_files:
            continue

        total += 1

        # Get predicted files - either from predicted_files or extract from diff
        predicted = pred_data.get('predicted_files', [])
        if not predicted and 'diff' in pred_data:
            predicted = extract_files_from_diff(pred_data['diff'])

        ground_truth = gt_files[issue_id]

        if not predicted or not ground_truth:
            continue

        pred_set = set(predicted)

        if pred_set == ground_truth:
            exact_matches += 1

        if len(pred_set) > 0:
            intersection = pred_set & ground_truth
            precision = len(intersection) / len(pred_set)
            precision_sum += precision

        for k in topk_hits.keys():
            top_k_files = set(predicted[:k])
            if top_k_files & ground_truth:
                topk_hits[k] += 1

    exact_match_rate = (exact_matches / total * 100) if total > 0 else 0
    avg_precision = (precision_sum / total * 100) if total > 0 else 0

    topk_accuracy = {
        f"top_{k}": round((hits / total * 100), 2) if total > 0 else 0
        for k, hits in topk_hits.items()
    }

    return {
        "total_issues": total,
        "exact_match": {"count": exact_matches, "rate": round(exact_match_rate, 2)},
        "precision": {"average": round(avg_precision, 2)},
        "top_k_accuracy": topk_accuracy
    }


def main():
    parser = argparse.ArgumentParser(description='File Localization Evaluation')
    parser.add_argument('--preds', type=str, default='results/preds.json',
                       help='Path to predictions JSON file')
    parser.add_argument('--dataset', type=str, default='dataset/lca_dataset.parquet',
                       help='Path to dataset parquet file')
    parser.add_argument('--output', type=str, default='results/file_localization_metrics.json',
                       help='Output path for metrics JSON')
    args = parser.parse_args()

    print("="*80)
    print("FILE LOCALIZATION EVALUATION")
    print("="*80)
    print()

    # Check if predictions file exists
    if not Path(args.preds).exists():
        print(f" Error: Predictions file not found: {args.preds}")
        return 1

    # Get pushed IDs (if available)
    results_dir = Path(args.output).parent
    pushed_ids = get_pushed_ids(results_dir)

    if pushed_ids:
        print(f" Evaluating against {len(pushed_ids)} pushed IDs (not full dataset)")
    else:
        print(f" Note: Evaluating against all predictions (no CI validation data)")

    # Compute metrics
    print(" Computing file localization metrics...")
    metrics = compute_file_localization_metrics(
        Path(args.preds),
        Path(args.dataset),
        pushed_ids=pushed_ids
    )
    print(f"    Evaluated {metrics['total_issues']} issues")

    # Save results
    print(f"\n Saving...")
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    {args.output}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n Total Evaluated: {metrics['total_issues']}")
    print(f"\n Exact Match: {metrics['exact_match']['rate']}%")
    print(f" Precision: {metrics['precision']['average']}%")
    print(f"\n Top-K Accuracy:")
    print(f"   Top-1:  {metrics['top_k_accuracy']['top_1']}%")
    print(f"   Top-3:  {metrics['top_k_accuracy']['top_3']}%")
    print(f"   Top-5:  {metrics['top_k_accuracy']['top_5']}%")
    print(f"   Top-10: {metrics['top_k_accuracy']['top_10']}%")
    print(f"   Top-15: {metrics['top_k_accuracy']['top_15']}%")
    print("="*80)

    return 0


if __name__ == "__main__":
    exit(main())

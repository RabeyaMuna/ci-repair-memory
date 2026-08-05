#!/usr/bin/env python3
"""
Top-K Overall Accuracy Evaluation.

Different from per-instance Top-K:
- Per-instance: Does ANY GT file appear in top K? (binary per instance)
- Overall: Of all K positions across all instances, what % are GT files?

Example: If 16 instances have predictions
- Per-instance Top-1: Count how many instances have GT in position 1
- Overall Top-1: Of 16 position-1 predictions, how many are GT files?

Usage:
    python evaluate_topk_overall.py --preds results/preds.json --dataset dataset/lca_dataset.parquet
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
    """Extract file paths from unified diff (maintains order)."""
    files = []
    pattern = re.compile(r"^diff --git a/.+ b/(.+)$", re.MULTILINE)
    for match in pattern.finditer(diff_text):
        file_path = match.group(1).strip()
        if file_path:
            files.append(normalize_path(file_path))
    return files


def load_ground_truth(dataset_path: Path, instance_ids: Set[str]) -> Dict[str, Set[str]]:
    """Load ground truth only for specified instance IDs."""
    df = pd.read_parquet(dataset_path)
    ground_truth = {}

    for _, row in df.iterrows():
        instance_id = str(row['id'])
        if instance_id not in instance_ids:
            continue

        diff = row.get('diff', '')
        if diff:
            gt_files = extract_files_from_diff(diff)
            ground_truth[instance_id] = set(gt_files)
        else:
            ground_truth[instance_id] = set()

    return ground_truth


def load_predictions(preds_path: Path) -> Dict[str, List[str]]:
    """Load predictions from preds.json."""
    with open(preds_path, 'r') as f:
        preds_data = json.load(f)

    predictions = {}
    for instance_id, data in preds_data.items():
        diff = data.get('diff', '')
        if diff:
            pred_files = extract_files_from_diff(diff)
            if pred_files:
                predictions[instance_id] = pred_files

    return predictions


def calculate_topk_overall(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    max_k: int = 15
) -> Dict:
    """
    Calculate Top-K Overall Accuracy.

    For each position k (1 to max_k):
    - Count how many instances have a prediction at position k
    - Count how many of those predictions are GT files
    - Accuracy = correct / total
    """

    results = {
        'per_position': {},
        'per_instance': [],
        'summary': {
            'total_instances': len(predictions),
        }
    }

    # Initialize position counters
    for k in range(1, max_k + 1):
        results['per_position'][f'pos_{k}'] = {
            'total': 0,
            'correct': 0,
            'accuracy': 0.0
        }

    # Analyze each instance
    for instance_id in sorted(predictions.keys(), key=lambda x: int(x) if x.isdigit() else x):
        pred_files = predictions[instance_id]
        gt_files = ground_truth.get(instance_id, set())

        instance_result = {
            'id': instance_id,
            'num_predicted': len(pred_files),
            'num_ground_truth': len(gt_files),
            'predictions': pred_files[:max_k],  # Only store up to max_k
            'position_correctness': {}
        }

        # Check each position
        for k in range(1, min(len(pred_files) + 1, max_k + 1)):
            pred_at_k = pred_files[k - 1]  # k-1 because list is 0-indexed
            is_correct = pred_at_k in gt_files

            # Update position stats
            pos_key = f'pos_{k}'
            results['per_position'][pos_key]['total'] += 1
            if is_correct:
                results['per_position'][pos_key]['correct'] += 1

            # Store per-instance
            instance_result['position_correctness'][f'pos_{k}'] = is_correct

        results['per_instance'].append(instance_result)

    # Calculate accuracies for each position
    for k in range(1, max_k + 1):
        pos_key = f'pos_{k}'
        pos_data = results['per_position'][pos_key]
        if pos_data['total'] > 0:
            pos_data['accuracy'] = pos_data['correct'] / pos_data['total'] * 100

    return results


def calculate_cumulative_topk(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    max_k: int = 15
) -> Dict:
    """
    Calculate cumulative Top-K accuracy.

    Top-K cumulative: Of all predictions in positions 1 to K across all instances,
    what percentage are GT files?
    """

    cumulative = {}

    for target_k in range(1, max_k + 1):
        total_preds = 0
        correct_preds = 0

        for instance_id in predictions:
            pred_files = predictions[instance_id]
            gt_files = ground_truth.get(instance_id, set())

            # Consider positions 1 to target_k
            for k in range(1, min(len(pred_files) + 1, target_k + 1)):
                pred_at_k = pred_files[k - 1]
                total_preds += 1
                if pred_at_k in gt_files:
                    correct_preds += 1

        accuracy = (correct_preds / total_preds * 100) if total_preds > 0 else 0.0

        cumulative[f'top_{target_k}'] = {
            'total_predictions': total_preds,
            'correct_predictions': correct_preds,
            'accuracy': accuracy
        }

    return cumulative


def print_summary(results: Dict, cumulative: Dict, max_k: int = 15):
    """Print evaluation summary."""

    n_instances = results['summary']['total_instances']

    print("\n" + "=" * 100)
    print("TOP-K OVERALL ACCURACY EVALUATION")
    print("=" * 100)
    print(f"\nTotal Instances: {n_instances}")
    print("\nThis analysis shows: Of all predictions at position K, what % are ground truth files?")

    # Per-position accuracy
    print(f"\n{'─' * 100}")
    print("POSITION-WISE ACCURACY (Each position independently)")
    print(f"{'─' * 100}")
    print(f"\n{'Position':<12} {'Total Preds':<15} {'Correct':<12} {'Accuracy':<15} {'Visualization':<30}")
    print("-" * 100)

    for k in range(1, max_k + 1):
        pos_key = f'pos_{k}'
        pos_data = results['per_position'][pos_key]

        if pos_data['total'] == 0:
            continue

        bar = '█' * int(pos_data['accuracy'] / 5) + '░' * (20 - int(pos_data['accuracy'] / 5))

        print(f"Position {k:<2}   {pos_data['total']:<15} {pos_data['correct']:<12} "
              f"{pos_data['accuracy']:>6.2f}%        {bar}")

    # Cumulative Top-K
    print(f"\n{'─' * 100}")
    print("CUMULATIVE TOP-K ACCURACY (All predictions from position 1 to K)")
    print(f"{'─' * 100}")
    print(f"\n{'Top-K':<12} {'Total Preds':<15} {'Correct':<12} {'Accuracy':<15} {'Visualization':<30}")
    print("-" * 100)

    for k in [1, 2, 3, 4, 5, 10, 15]:
        if k > max_k:
            break

        key = f'top_{k}'
        if key not in cumulative:
            continue

        data = cumulative[key]
        bar = '█' * int(data['accuracy'] / 5) + '░' * (20 - int(data['accuracy'] / 5))

        print(f"Top-{k:<2}        {data['total_predictions']:<15} {data['correct_predictions']:<12} "
              f"{data['accuracy']:>6.2f}%        {bar}")

    print("\n" + "=" * 100)


def print_detailed_per_instance(results: Dict):
    """Print detailed per-instance position correctness."""

    print("\n" + "=" * 120)
    print("PER-INSTANCE POSITION CORRECTNESS")
    print("=" * 120)
    print(f"\n{'ID':<6} {'#Pred':<7} {'#GT':<7} {'Pos1':<6} {'Pos2':<6} {'Pos3':<6} {'Pos4':<6} {'Pos5':<6} "
          f"{'Pos10':<7} {'Pos15':<7}")
    print("-" * 120)

    for inst in results['per_instance']:
        row = f"{inst['id']:<6} {inst['num_predicted']:<7} {inst['num_ground_truth']:<7}"

        for k in [1, 2, 3, 4, 5, 10, 15]:
            pos_key = f'pos_{k}'
            if pos_key in inst['position_correctness']:
                mark = '✓' if inst['position_correctness'][pos_key] else '✗'
            else:
                mark = '-'

            col_width = 7 if k >= 10 else 6
            row += f" {mark:<{col_width}}"

        print(row)

    print("=" * 120)


def save_results(results: Dict, cumulative: Dict, output_path: Path):
    """Save results to JSON."""
    output = {
        'per_position': results['per_position'],
        'cumulative_topk': cumulative,
        'per_instance': results['per_instance'],
        'summary': results['summary']
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Top-K Overall Accuracy Evaluation"
    )
    parser.add_argument('--preds', default='results/preds.json')
    parser.add_argument('--dataset', default='dataset/lca_dataset.parquet')
    parser.add_argument('--output', default='results/topk_overall_evaluation.json')
    parser.add_argument('--max-k', type=int, default=15)
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed per-instance analysis')

    args = parser.parse_args()

    print(f"\nLoading predictions from: {args.preds}")
    predictions = load_predictions(Path(args.preds))

    print(f"Loading ground truth from: {args.dataset}")
    instance_ids = set(predictions.keys())
    ground_truth = load_ground_truth(Path(args.dataset), instance_ids)

    print(f"\nCalculating Top-K overall accuracy (K=1 to {args.max_k})...")
    results = calculate_topk_overall(predictions, ground_truth, args.max_k)
    cumulative = calculate_cumulative_topk(predictions, ground_truth, args.max_k)

    print_summary(results, cumulative, args.max_k)

    if args.detailed:
        print_detailed_per_instance(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, cumulative, output_path)
    print(f"\nResults saved to: {output_path}")
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()

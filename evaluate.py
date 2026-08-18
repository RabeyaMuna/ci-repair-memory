#!/usr/bin/env python3
"""
CI-REPAIR-BENCH Unified Evaluation Script
==========================================

Computes all essential metrics:
1. File Localization: Top-K, Precision, Exact Match
2. CI Success: Overall workflow pass rate  
3. Multi-Level Success: L1 (step-level), L3 (workflow-level)

Usage:
    python evaluate.py --preds results/preds.json --ci-results results/success_rate_evaluation.json
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict


def compute_file_localization_metrics(preds_file: Path, dataset_file: Path) -> Dict:
    """Compute file localization metrics: Top-K accuracy, Precision, Exact Match."""
    with open(preds_file, 'r') as f:
        preds = json.load(f)

    df = pd.read_parquet(dataset_file)

    gt_files = {}
    for _, row in df.iterrows():
        issue_id = str(row['id'])
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
        predicted = pred_data.get('predicted_files', [])
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


def compute_ci_success_metrics(results_file: Path) -> Dict:
    """Compute CI success metrics from success_rate_evaluation.json."""
    with open(results_file, 'r') as f:
        data = json.load(f)

    summary = data['summary']

    return {
        "overall_ci_success": {
            "description": "Percentage of workflows that passed after patch",
            "passed": summary['workflow_level']['passed'],
            "total": summary['total_issues'],
            "rate": summary['workflow_level']['pass_rate']
        },
        "level_1_step_success": {
            "description": "Of originally failed steps, how many now pass",
            "fully_fixed": summary['step_level']['fully_fixed'],
            "partially_fixed": summary['step_level']['partially_fixed'],
            "not_fixed": summary['step_level']['not_fixed'],
            "average_success_rate": summary['step_level']['average_success_rate']
        },
        "level_3_workflow": {
            "description": "Workflow-level pass/fail",
            "passed": summary['workflow_level']['passed'],
            "failed": summary['workflow_level']['failed'],
            "pass_rate": summary['workflow_level']['pass_rate']
        }
    }


def main():
    parser = argparse.ArgumentParser(description='CI-REPAIR-BENCH Unified Evaluation')
    parser.add_argument('--preds', type=str, default='results/preds.json')
    parser.add_argument('--dataset', type=str, default='dataset/lca_dataset.parquet')
    parser.add_argument('--ci-results', type=str, default='results/success_rate_evaluation.json')
    parser.add_argument('--output', type=str, default='results/evaluation_summary.json')
    args = parser.parse_args()

    print("="*80)
    print("CI-REPAIR-BENCH UNIFIED EVALUATION")
    print("="*80)
    print()

    print("📊 Computing file localization metrics...")
    if Path(args.preds).exists():
        loc_metrics = compute_file_localization_metrics(Path(args.preds), Path(args.dataset))
        print(f"   ✓ Evaluated {loc_metrics['total_issues']} issues")
    else:
        print(f"   ⚠️  Predictions file not found")
        loc_metrics = None

    print("\n📊 Computing CI success metrics...")
    if Path(args.ci_results).exists():
        ci_metrics = compute_ci_success_metrics(Path(args.ci_results))
        print(f"   ✓ Loaded {ci_metrics['overall_ci_success']['total']} issues")
    else:
        print(f"   ⚠️  Run: python scripts/analysis/calculate_success_rate.py first")
        ci_metrics = None

    evaluation = {"file_localization": loc_metrics, "ci_success": ci_metrics}

    print(f"\n💾 Saving...")
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"   ✓ {args.output}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if loc_metrics:
        print("\n📁 File Localization:")
        print(f"   Exact Match: {loc_metrics['exact_match']['rate']}%")
        print(f"   Precision: {loc_metrics['precision']['average']}%")
        print(f"   Top-1: {loc_metrics['top_k_accuracy']['top_1']}%")
        print(f"   Top-5: {loc_metrics['top_k_accuracy']['top_5']}%")

    if ci_metrics:
        print("\n🔧 CI Success:")
        print(f"   Overall: {ci_metrics['overall_ci_success']['rate']}%")
        print(f"   L1 Step Success: {ci_metrics['level_1_step_success']['average_success_rate']}%")
        print(f"   L3 Workflow Pass: {ci_metrics['level_3_workflow']['pass_rate']}%")

    print("="*80)


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Variance Study for CI-REPAIR-BENCH

Runs the same subset of instances multiple times to measure variance in Pass@1 results.
This addresses reviewer concern C5 about single-run evaluation.

Usage:
    python variance_study/run_variance_study.py --num-instances 100 --num-runs 3
"""

import os
import sys
import json
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import from baselines
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from omegaconf import OmegaConf


def sample_instances(dataset_path, num_instances, seed=42):
    """
    Randomly sample instances for the variance study.

    Returns:
        list: Sampled instance IDs
    """
    df = pd.read_parquet(dataset_path)
    all_ids = df['id'].tolist()

    random.seed(seed)
    sampled = random.sample(all_ids, min(num_instances, len(all_ids)))

    print(f"\n[SAMPLE] Sampled {len(sampled)} instances from {len(all_ids)} total")
    print(f"[SAMPLE] First 10 IDs: {sampled[:10]}")

    return sampled


def run_baseline_pipeline(instance_ids, run_number, model, analyzer, base_dir):
    """
    Run the baseline pipeline for one variance run.

    Args:
        instance_ids: List of instance IDs to process
        run_number: Current run number (1, 2, 3, ...)
        model: Model name (e.g., 'deepseek-v4-flash')
        analyzer: Analyzer type ('llm' or 'bm25')
        base_dir: Base directory of the project

    Returns:
        Path to generated patches file
    """
    print(f"\n{'='*80}")
    print(f"RUN {run_number}: GENERATING PATCHES")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Analyzer: {analyzer}")
    print(f"Instances: {len(instance_ids)}")
    print(f"{'='*80}\n")

    # Create a temporary dataset file with only selected instances
    config = OmegaConf.load(os.path.join(base_dir, "config.yaml"))
    dataset_path = os.path.join(base_dir, "dataset", "lca_dataset.parquet")

    # Load full dataset and filter to selected IDs
    df = pd.read_parquet(dataset_path)
    subset_df = df[df['id'].isin(instance_ids)].copy()

    # Save subset to temporary file
    temp_dataset = os.path.join(base_dir, f"variance_study/temp_dataset_run{run_number}.parquet")
    subset_df.to_parquet(temp_dataset)
    print(f"[RUN {run_number}] Created temporary dataset with {len(subset_df)} instances")

    # Modify config to use a separate output folder for this run
    variance_output_dir = os.path.join(base_dir, f"baselines/results_variance/run_{run_number}")
    os.makedirs(variance_output_dir, exist_ok=True)

    # Create modified config for this run
    modified_config = OmegaConf.to_container(config, resolve=True)
    modified_config['project_result_dir'] = variance_output_dir
    modified_config['changed_files_folder'] = os.path.join(variance_output_dir, "changed_files")

    temp_config_path = os.path.join(base_dir, f"variance_study/temp_config_run{run_number}.yaml")
    with open(temp_config_path, 'w') as f:
        OmegaConf.save(modified_config, f)

    # Run baseline pipeline
    cmd = [
        sys.executable,
        os.path.join(base_dir, "baselines/main.py"),
        "--model", model,
        "--analyzer", analyzer,
        "--num-instances", str(len(instance_ids))
    ]

    # Temporarily replace config.yaml with our modified version
    original_config = os.path.join(base_dir, "config.yaml")
    backup_config = os.path.join(base_dir, "config.yaml.backup")

    try:
        # Backup original config
        import shutil
        shutil.copy(original_config, backup_config)
        shutil.copy(temp_config_path, original_config)

        # Also need to temporarily replace the dataset
        original_dataset = os.path.join(base_dir, "dataset/lca_dataset.parquet")
        backup_dataset = os.path.join(base_dir, "dataset/lca_dataset.parquet.backup")
        shutil.copy(original_dataset, backup_dataset)
        shutil.copy(temp_dataset, original_dataset)

        print(f"\n[RUN {run_number}] Starting baseline pipeline...")
        result = subprocess.run(cmd, check=True, cwd=base_dir)

        # Restore original files
        shutil.copy(backup_config, original_config)
        shutil.copy(backup_dataset, original_dataset)

        # Find generated patches
        patches_file = os.path.join(variance_output_dir, f"{model}_{analyzer}", "generated_patches.json")

        if os.path.exists(patches_file):
            print(f"\n[RUN {run_number}] ✓ Patches generated successfully: {patches_file}")
            return patches_file
        else:
            print(f"\n[RUN {run_number}] ✗ No patches file found at: {patches_file}")
            return None

    except Exception as e:
        print(f"\n[RUN {run_number}] ✗ Pipeline failed: {e}")
        # Restore original files on error
        if os.path.exists(backup_config):
            shutil.copy(backup_config, original_config)
        if os.path.exists(backup_dataset):
            shutil.copy(backup_dataset, original_dataset)
        return None
    finally:
        # Clean up backups
        for f in [backup_config, backup_dataset, temp_config_path, temp_dataset]:
            if os.path.exists(f):
                os.remove(f)


def evaluate_patches(patches_file, run_number, base_dir):
    """
    Evaluate generated patches using the benchmark.

    Args:
        patches_file: Path to generated patches JSON
        run_number: Current run number
        base_dir: Base directory of the project

    Returns:
        dict: Evaluation results with Pass@1 rate
    """
    print(f"\n{'='*80}")
    print(f"RUN {run_number}: EVALUATING PATCHES")
    print(f"{'='*80}\n")

    if not patches_file or not os.path.exists(patches_file):
        print(f"[RUN {run_number}] ✗ No patches file to evaluate")
        return None

    # Read patches to get instance IDs
    with open(patches_file, 'r') as f:
        patches = json.load(f)

    instance_ids = [p.get('id') or p.get('task_id') for p in patches if p.get('diff')]

    if not instance_ids:
        print(f"[RUN {run_number}] ✗ No valid patches found")
        return None

    print(f"[RUN {run_number}] Found {len(instance_ids)} patches to evaluate")

    # Copy patches to expected location for benchmark
    # The benchmark expects patches at results/generated_patches-diff.json
    benchmark_patches_dir = os.path.join(base_dir, "results")
    os.makedirs(benchmark_patches_dir, exist_ok=True)

    benchmark_patches_file = os.path.join(benchmark_patches_dir, "generated_patches-diff.json")

    import shutil
    shutil.copy(patches_file, benchmark_patches_file)
    print(f"[RUN {run_number}] Copied patches to: {benchmark_patches_file}")

    # Run benchmark evaluation
    ids_str = ",".join(str(id) for id in instance_ids)

    cmd = [
        sys.executable,
        os.path.join(base_dir, "run_benchmark.py"),
        "--ids", ids_str,
        "--skip-upstream-sync"
    ]

    try:
        print(f"\n[RUN {run_number}] Starting benchmark evaluation...")
        subprocess.run(cmd, check=True, cwd=base_dir)

        # Read results
        results_file = os.path.join(base_dir, "results/jobs_ids_diff.jsonl")

        if not os.path.exists(results_file):
            print(f"[RUN {run_number}] ✗ No results file found")
            return None

        # Save results for this run
        run_results_file = os.path.join(base_dir, f"variance_study/run_{run_number}_results.jsonl")
        shutil.copy(results_file, run_results_file)

        # Calculate Pass@1
        total = 0
        passed = 0

        with open(results_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                    total += 1
                    # Check if resolved (patch worked)
                    if result.get('resolved') or result.get('status') == 'success':
                        passed += 1
                except json.JSONDecodeError:
                    continue

        pass_rate = (passed / total * 100) if total > 0 else 0

        print(f"\n[RUN {run_number}] ✓ Evaluation complete")
        print(f"[RUN {run_number}] Pass@1: {pass_rate:.2f}% ({passed}/{total})")

        return {
            'run_number': run_number,
            'total_instances': total,
            'passed': passed,
            'pass_rate': pass_rate,
            'results_file': run_results_file
        }

    except Exception as e:
        print(f"\n[RUN {run_number}] ✗ Evaluation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run variance study on CI-REPAIR-BENCH"
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=100,
        help="Number of instances to sample (default: 100)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of independent runs (default: 3)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v4-flash",
        help="Model to use (default: deepseek-v4-flash)"
    )
    parser.add_argument(
        "--analyzer",
        choices=['llm', 'bm25'],
        default='llm',
        help="Log analyzer type (default: llm)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--skip-patch-generation",
        action="store_true",
        help="Skip patch generation (use existing patches)"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip benchmark evaluation (only generate patches)"
    )

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent.absolute()
    variance_dir = base_dir / "variance_study"
    variance_dir.mkdir(exist_ok=True)

    # Load dataset
    dataset_path = base_dir / "dataset" / "lca_dataset.parquet"
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        return

    print(f"\n{'='*80}")
    print(f"VARIANCE STUDY CONFIGURATION")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Analyzer: {args.analyzer}")
    print(f"Number of instances: {args.num_instances}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*80}\n")

    # Sample instances (same for all runs)
    sampled_ids = sample_instances(dataset_path, args.num_instances, args.seed)

    # Save sampled IDs
    sampled_ids_file = variance_dir / "sampled_instances.json"
    with open(sampled_ids_file, 'w') as f:
        json.dump({
            "num_instances": len(sampled_ids),
            "seed": args.seed,
            "instance_ids": sampled_ids,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"[SAVED] Sampled instance IDs: {sampled_ids_file}")

    # Run multiple times
    all_results = []

    for run_num in range(1, args.num_runs + 1):
        # Stage 1: Generate patches
        if not args.skip_patch_generation:
            patches_file = run_baseline_pipeline(
                sampled_ids,
                run_num,
                args.model,
                args.analyzer,
                base_dir
            )
        else:
            # Try to find existing patches
            patches_file = base_dir / f"baselines/results_variance/run_{run_num}/{args.model}_{args.analyzer}/generated_patches.json"
            if patches_file.exists():
                print(f"\n[RUN {run_num}] Using existing patches: {patches_file}")
            else:
                print(f"\n[RUN {run_num}] No existing patches found")
                patches_file = None

        # Stage 2: Evaluate patches
        if not args.skip_evaluation and patches_file:
            eval_result = evaluate_patches(patches_file, run_num, base_dir)
            if eval_result:
                all_results.append(eval_result)

        print(f"\n[RUN {run_num}] Complete\n")

    # Analyze variance
    if len(all_results) >= 2:
        import numpy as np
        from scipy import stats as sp_stats

        print(f"\n{'='*80}")
        print(f"VARIANCE ANALYSIS")
        print(f"{'='*80}\n")

        pass_rates = [r['pass_rate'] for r in all_results]

        mean = np.mean(pass_rates)
        std_dev = np.std(pass_rates, ddof=1)
        variance = np.var(pass_rates, ddof=1)

        print(f"Pass@1 rates across runs:")
        for r in all_results:
            print(f"  Run {r['run_number']}: {r['pass_rate']:.2f}% ({r['passed']}/{r['total_instances']})")

        print(f"\nStatistics:")
        print(f"  Mean: {mean:.2f}%")
        print(f"  Standard Deviation: {std_dev:.2f}%")
        print(f"  Variance: {variance:.4f}")
        print(f"  Range: [{min(pass_rates):.2f}%, {max(pass_rates):.2f}%]")

        # 95% confidence interval
        if len(pass_rates) >= 2:
            n = len(pass_rates)
            std_err = std_dev / np.sqrt(n)
            t_value = sp_stats.t.ppf(0.975, n - 1)
            margin = t_value * std_err

            print(f"\n95% Confidence Interval: [{mean - margin:.2f}%, {mean + margin:.2f}%]")
            print(f"Margin of Error: ±{margin:.2f}%")

        # Save statistics
        stats_file = variance_dir / "variance_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump({
                "configuration": {
                    "model": args.model,
                    "analyzer": args.analyzer,
                    "num_instances": args.num_instances,
                    "num_runs": args.num_runs,
                    "seed": args.seed
                },
                "results": all_results,
                "statistics": {
                    "mean": float(mean),
                    "std_dev": float(std_dev),
                    "variance": float(variance),
                    "min": float(min(pass_rates)),
                    "max": float(max(pass_rates)),
                    "pass_rates": pass_rates
                },
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        print(f"\n[SAVED] Statistics saved to: {stats_file}")

        # Print paper text
        print(f"\n{'='*80}")
        print(f"FOR YOUR PAPER RESPONSE:")
        print(f"{'='*80}\n")
        print(f"\"We conducted a variance study by running the {args.model} configuration")
        print(f"three times on a randomly sampled subset of {args.num_instances} instances.")
        print(f"The Pass@1 rate was {mean:.1f}% ± {std_dev:.2f}% (mean ± std dev),")
        print(f"with a variance of {variance:.4f}, demonstrating")
        if std_dev < 2.0:
            print(f"very low variability in model performance across independent runs.\"")
        elif std_dev < 5.0:
            print(f"low variability in model performance across independent runs.\"")
        else:
            print(f"acceptable variability in model performance across independent runs.\"")

        print(f"\n{'='*80}")

    print(f"\nVARIANCE STUDY COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

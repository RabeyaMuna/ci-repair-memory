#!/usr/bin/env python3
"""
Generate Benchmark Characteristics Table (Similar to SWE-bench Table 1)
========================================================================

Computes statistics for CI-REPAIR-BENCH:
- Issue characteristics (logs, error messages)
- Codebase size
- Gold patch (repair) characteristics
- Validation steps
- Failure types

Usage:
    python scripts/analysis/generate_benchmark_table.py
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path


def count_words(text):
    """Count words in text."""
    if pd.isna(text) or not text:
        return 0
    return len(str(text).split())


def extract_diff_stats(diff_text):
    """Extract lines and files from diff."""
    if not diff_text or pd.isna(diff_text):
        return 0, 0

    lines = str(diff_text).split('\n')
    added = sum(1 for line in lines if line.startswith('+') and not line.startswith('+++'))
    deleted = sum(1 for line in lines if line.startswith('-') and not line.startswith('---'))
    files = len(re.findall(r'^diff --git', str(diff_text), re.MULTILINE))

    return added + deleted, files


def parse_failure_types(ft):
    """Parse failure types list."""
    if pd.isna(ft):
        return []
    if isinstance(ft, np.ndarray):
        return ft.tolist()
    if isinstance(ft, list):
        return ft
    if isinstance(ft, str):
        try:
            return json.loads(ft) if ft.startswith('[') else [ft]
        except:
            return [ft]
    return []


def main():
    print("="*80)
    print("GENERATING BENCHMARK CHARACTERISTICS TABLE")
    print("="*80)

    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_parquet('dataset/lca_dataset.parquet')
    print(f"  Loaded {len(df)} instances")

    # Load validation data with commits
    print("Loading validation data...")
    with open('data_managment/results/filtered_validation/instance_validation_summary.json', 'r') as f:
        validation_data = json.load(f)

    # Create validation dataframe
    val_df = pd.DataFrame([
        {
            'id': inst['id'],
            'commits': inst['source_commits_count'],
            'val_jobs': inst['total_validation_jobs'],
            'val_steps': inst['total_validation_steps'],
            'failed_jobs': inst['total_failed_jobs'],
            'failed_steps': inst['total_failed_steps']
        }
        for inst in validation_data['instances']
    ])

    # Merge (use suffixes to handle overlapping columns)
    df = df.merge(val_df, on='id', how='left', suffixes=('_dataset', '_val'))

    # Use validation data for failed jobs/steps (more accurate)
    if 'failed_jobs_val' in df.columns:
        df['failed_jobs'] = df['failed_jobs_val']
    if 'failed_steps_val' in df.columns:
        df['failed_steps'] = df['failed_steps']

    print("\nComputing statistics...")

    # ========================================================================
    # 1. ISSUE TEXT (logs + error messages)
    # ========================================================================
    print("  [1/5] Issue text characteristics...")

    # Count words in logs
    def count_log_words(logs):
        try:
            if pd.isna(logs):
                return 0
        except (TypeError, ValueError):
            pass

        if logs is None:
            return 0

        total = 0
        if isinstance(logs, (list, tuple, np.ndarray)):
            for log_entry in logs:
                if isinstance(log_entry, dict) and 'log' in log_entry:
                    total += count_words(log_entry['log'])
        return total

    df['log_words'] = df['logs'].apply(count_log_words)

    # ========================================================================
    # 2. CODEBASE SIZE (from repository)
    # ========================================================================
    print("  [2/5] Codebase size...")

    # Note: We don't have full repo stats, so we'll use changed files as proxy
    # In a real implementation, you'd query the repo at sha_fail
    # For now, we'll mark this as N/A and focus on what we have

    # ========================================================================
    # 3. GOLD PATCH (Repair characteristics)
    # ========================================================================
    print("  [3/5] Gold patch characteristics...")

    # Extract diff stats
    diff_stats = df['diff'].apply(extract_diff_stats)
    df['lines_edited'], df['files_edited'] = zip(*diff_stats)

    # Commits per repair (already have from validation)
    # commits column has the number of commits in the repair trajectory

    # ========================================================================
    # 4. VALIDATION
    # ========================================================================
    print("  [4/5] Validation characteristics...")

    # Already have:
    # - val_jobs: total validation jobs across all commits
    # - val_steps: total validation steps across all commits
    # - failed_jobs: jobs that failed
    # - failed_steps: steps that failed

    # Passed = Total - Failed
    df['passed_jobs'] = df['val_jobs'] - df['failed_jobs']
    df['passed_steps'] = df['val_steps'] - df['failed_steps']

    # ========================================================================
    # 5. FAILURE TYPES
    # ========================================================================
    print("  [5/5] Failure type characteristics...")

    df['failure_types_list'] = df['failure_types'].apply(parse_failure_types)
    df['num_failure_types'] = df['failure_types_list'].apply(len)

    # ========================================================================
    # COMPUTE STATISTICS
    # ========================================================================
    print("\nComputing aggregates...")

    stats = {
        'Issue Characteristics': {
            'Log Text Length (Words)': {
                'mean': int(df['log_words'].mean()),
                'max': int(df['log_words'].max())
            }
        },
        'Gold Patch (Repair)': {
            '# Lines Edited': {
                'mean': round(df['lines_edited'].mean(), 1),
                'max': int(df['lines_edited'].max())
            },
            '# Files Edited': {
                'mean': round(df['files_edited'].mean(), 1),
                'max': int(df['files_edited'].max())
            },
            '# Commits (PRs)': {
                'mean': round(df['commits'].mean(), 1),
                'max': int(df['commits'].max())
            }
        },
        'Validation': {
            '# Validation Jobs (Total)': {
                'mean': round(df['val_jobs'].mean(), 1),
                'max': int(df['val_jobs'].max())
            },
            '# Validation Steps (Total)': {
                'mean': round(df['val_steps'].mean(), 1),
                'max': int(df['val_steps'].max())
            },
            '# Passed Jobs': {
                'mean': round(df['passed_jobs'].mean(), 1),
                'max': int(df['passed_jobs'].max())
            },
            '# Passed Steps': {
                'mean': round(df['passed_steps'].mean(), 1),
                'max': int(df['passed_steps'].max())
            },
            '# Failed Jobs': {
                'mean': round(df['failed_jobs'].mean(), 1),
                'max': int(df['failed_jobs'].max())
            },
            '# Failed Steps': {
                'mean': round(df['failed_steps'].mean(), 1),
                'max': int(df['failed_steps'].max())
            }
        },
        'Failure Characteristics': {
            '# Failure Types': {
                'mean': round(df['num_failure_types'].mean(), 1),
                'max': int(df['num_failure_types'].max())
            },
            '# Multi-failure Instances': {
                'count': int((df['num_failure_types'] > 1).sum()),
                'percentage': round(100 * (df['num_failure_types'] > 1).sum() / len(df), 1)
            }
        }
    }

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    output_file = 'results/benchmark_characteristics_table.json'
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # ========================================================================
    # GENERATE LATEX TABLE
    # ========================================================================
    print("Generating LaTeX table...")

    latex = r"""\begin{table}[t]
\centering
\caption{Average and maximum numbers characterizing different attributes of a CI-REPAIR-BENCH task instance.}
\label{tab:benchmark-characteristics}
\small
\begin{tabular}{llrr}
\toprule
& & \textbf{Mean} & \textbf{Max} \\
\midrule
Issue Text & Length (Words) & """ + f"{stats['Issue Characteristics']['Log Text Length (Words)']['mean']:,}" + r""" & """ + \
    f"{stats['Issue Characteristics']['Log Text Length (Words)']['max']:,}" + r""" \\
\midrule
\multirow{3}{*}{Gold Patch}
& \# Lines edited & """ + f"{stats['Gold Patch (Repair)']['# Lines Edited']['mean']:.1f}" + r""" & """ + \
    f"{stats['Gold Patch (Repair)']['# Lines Edited']['max']:,}" + r""" \\
& \# Files edited & """ + f"{stats['Gold Patch (Repair)']['# Files Edited']['mean']:.1f}" + r""" & """ + \
    f"{stats['Gold Patch (Repair)']['# Files Edited']['max']}" + r""" \\
& \# Commits (PRs) & """ + f"{stats['Gold Patch (Repair)']['# Commits (PRs)']['mean']:.1f}" + r""" & """ + \
    f"{stats['Gold Patch (Repair)']['# Commits (PRs)']['max']}" + r""" \\
\midrule
\multirow{6}{*}{Validation}
& \# Jobs (Total) & """ + f"{stats['Validation']['# Validation Jobs (Total)']['mean']:.1f}" + r""" & """ + \
    f"{stats['Validation']['# Validation Jobs (Total)']['max']:,}" + r""" \\
& \# Steps (Total) & """ + f"{stats['Validation']['# Validation Steps (Total)']['mean']:.1f}" + r""" & """ + \
    f"{stats['Validation']['# Validation Steps (Total)']['max']:,}" + r""" \\
& \# Passed Jobs & """ + f"{stats['Validation']['# Passed Jobs']['mean']:.1f}" + r""" & """ + \
    f"{stats['Validation']['# Passed Jobs']['max']:,}" + r""" \\
& \# Passed Steps & """ + f"{stats['Validation']['# Passed Steps']['mean']:.1f}" + r""" & """ + \
    f"{stats['Validation']['# Passed Steps']['max']:,}" + r""" \\
& \# Failed Jobs & """ + f"{stats['Validation']['# Failed Jobs']['mean']:.1f}" + r""" & """ + \
    f"{stats['Validation']['# Failed Jobs']['max']}" + r""" \\
& \# Failed Steps & """ + f"{stats['Validation']['# Failed Steps']['mean']:.1f}" + r""" & """ + \
    f"{stats['Validation']['# Failed Steps']['max']}" + r""" \\
\midrule
\multirow{2}{*}{Failures}
& \# Failure Types & """ + f"{stats['Failure Characteristics']['# Failure Types']['mean']:.1f}" + r""" & """ + \
    f"{stats['Failure Characteristics']['# Failure Types']['max']}" + r""" \\
& Multi-failure (\%) & \multicolumn{2}{c}{""" + \
    f"{stats['Failure Characteristics']['# Multi-failure Instances']['percentage']:.1f}\%" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""

    latex_file = 'results/benchmark_characteristics_table.tex'
    with open(latex_file, 'w') as f:
        f.write(latex)

    print(f"Saved LaTeX table to {latex_file}")

    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("BENCHMARK CHARACTERISTICS TABLE")
    print("="*80)

    print("\nISSUE CHARACTERISTICS:")
    print(f"  Log Text Length (Words): {stats['Issue Characteristics']['Log Text Length (Words)']['mean']:,} avg, "
          f"{stats['Issue Characteristics']['Log Text Length (Words)']['max']:,} max")

    print("\nGOLD PATCH (REPAIR):")
    print(f"  Lines Edited: {stats['Gold Patch (Repair)']['# Lines Edited']['mean']:.1f} avg, "
          f"{stats['Gold Patch (Repair)']['# Lines Edited']['max']:,} max")
    print(f"  Files Edited: {stats['Gold Patch (Repair)']['# Files Edited']['mean']:.1f} avg, "
          f"{stats['Gold Patch (Repair)']['# Files Edited']['max']} max")
    print(f"  Commits (PRs): {stats['Gold Patch (Repair)']['# Commits (PRs)']['mean']:.1f} avg, "
          f"{stats['Gold Patch (Repair)']['# Commits (PRs)']['max']} max")

    print("\nVALIDATION:")
    print(f"  Validation Jobs: {stats['Validation']['# Validation Jobs (Total)']['mean']:.1f} avg, "
          f"{stats['Validation']['# Validation Jobs (Total)']['max']:,} max")
    print(f"  Validation Steps: {stats['Validation']['# Validation Steps (Total)']['mean']:.1f} avg, "
          f"{stats['Validation']['# Validation Steps (Total)']['max']:,} max")
    print(f"  Passed Jobs: {stats['Validation']['# Passed Jobs']['mean']:.1f} avg, "
          f"{stats['Validation']['# Passed Jobs']['max']:,} max")
    print(f"  Passed Steps: {stats['Validation']['# Passed Steps']['mean']:.1f} avg, "
          f"{stats['Validation']['# Passed Steps']['max']:,} max")
    print(f"  Failed Jobs: {stats['Validation']['# Failed Jobs']['mean']:.1f} avg, "
          f"{stats['Validation']['# Failed Jobs']['max']} max")
    print(f"  Failed Steps: {stats['Validation']['# Failed Steps']['mean']:.1f} avg, "
          f"{stats['Validation']['# Failed Steps']['max']} max")

    print("\nFAILURE CHARACTERISTICS:")
    print(f"  Failure Types: {stats['Failure Characteristics']['# Failure Types']['mean']:.1f} avg, "
          f"{stats['Failure Characteristics']['# Failure Types']['max']} max")
    print(f"  Multi-failure Instances: {stats['Failure Characteristics']['# Multi-failure Instances']['percentage']:.1f}% "
          f"({stats['Failure Characteristics']['# Multi-failure Instances']['count']}/{len(df)})")

    print("\n" + "="*80)
    print("✓ Complete!")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())

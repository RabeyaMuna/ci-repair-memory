#!/usr/bin/env python3
"""
Benchmark Recurrence Analysis for CI-REPAIR-BENCH

Computes TWO similarity measures for the paper:
1. Structural Similarity (Jaccard): failure type, CI tools, packages, changed files
2. Lexical Similarity (TF-IDF + Cosine): failure + workflow + repair text

Table metrics:
- Overall nearest neighbor
- Within repository
- Cross repository
- Historical predecessor (chronologically earlier instances only)

This is OFFLINE benchmark characterization - includes ground-truth repairs.
NOT what's available at retrieval time (that's for MemRepair evaluation).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
OUTPUT_DIR = PROJECT_ROOT / "dataset_overview"


# ============================================================================
# STRUCTURAL SIMILARITY (Jaccard)
# ============================================================================

def extract_structural_features(row: Dict[str, Any], use_full_paths: bool = False) -> Dict[str, Set[str]]:
    """
    Extract structural features:
    - failure_types: error categories
    - tools: CI tools (pytest, npm, etc.)
    - packages: mentioned packages
    - changed_files: full paths (if use_full_paths=True) or basenames only

    Args:
        row: Instance data
        use_full_paths: If True, use full file paths; if False, use basenames only
                       Should be True for within-repository, False for cross-repository
    """

    # Failure types
    error_type = row.get("error_type", [])
    if isinstance(error_type, np.ndarray):
        failure_types = {str(e).strip().lower() for e in error_type.tolist()}
    elif isinstance(error_type, list):
        failure_types = {str(e).strip().lower() for e in error_type}
    else:
        failure_types = {str(error_type).strip().lower()} if error_type else set()

    # Tools from logs and workflow
    logs = row.get("logs", "")
    workflow = row.get("workflow", "")

    logs_text = ""
    if isinstance(logs, np.ndarray):
        for item in logs:
            if isinstance(item, dict) and 'log' in item:
                logs_text += str(item['log']) + " "
    elif isinstance(logs, str):
        logs_text = logs

    workflow_text = str(workflow) if workflow else ""
    text = (logs_text + " " + workflow_text).lower()

    # Extract tools
    tool_patterns = [
        r'\b(pytest|npm|pip|poetry|cargo|gradle|maven|tox|flake8|mypy|black|isort)\b',
        r'\b(eslint|prettier|jest|mocha|phpunit|composer|bundler|rspec)\b',
        r'\b(rubocop|pylint|bandit|coverage|unittest)\b',
    ]

    tools = set()
    for pattern in tool_patterns:
        tools.update(re.findall(pattern, text))

    # Extract packages (filter to common ones)
    package_pattern = r'\b([a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*)\b'
    all_packages = set(re.findall(package_pattern, text[:5000]))
    common_packages = {'numpy', 'pandas', 'flask', 'django', 'react', 'express', 'pytest', 'jest'}
    packages = all_packages & common_packages

    # Changed files
    changed_files = row.get("changed_files", [])
    if isinstance(changed_files, np.ndarray):
        files = changed_files.tolist()
    elif isinstance(changed_files, list):
        files = changed_files
    else:
        files = [changed_files] if changed_files else []

    # Use full paths or basenames consistently
    if use_full_paths:
        # Within-repository: full paths are meaningful
        file_set = {str(f).strip().lower().replace("\\", "/") for f in files if f}
    else:
        # Cross-repository: only basenames are comparable
        file_set = {os.path.basename(str(f).strip().replace("\\", "/")).lower() for f in files if f}

    return {
        "failure_types": failure_types,
        "tools": tools,
        "packages": packages,
        "changed_files": file_set,
    }


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity = |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_structural_similarity(row_a: Dict, row_b: Dict) -> float:
    """
    Structural similarity = average Jaccard across features.

    Logic:
    - Within-repository: Both instances use FULL file paths
    - Cross-repository: Both instances use BASENAMES only

    This ensures consistent, symmetric comparison.
    """

    # Determine if same repository
    repo_a = f"{row_a.get('repo_owner')}/{row_a.get('repo_name')}"
    repo_b = f"{row_b.get('repo_owner')}/{row_b.get('repo_name')}"
    same_repo = (repo_a == repo_b)

    # Extract features with consistent file handling for BOTH instances
    feat_a = extract_structural_features(row_a, use_full_paths=same_repo)
    feat_b = extract_structural_features(row_b, use_full_paths=same_repo)

    # Compute average Jaccard across all features
    sims = [
        jaccard_similarity(feat_a["failure_types"], feat_b["failure_types"]),
        jaccard_similarity(feat_a["tools"], feat_b["tools"]),
        jaccard_similarity(feat_a["changed_files"], feat_b["changed_files"]),
    ]

    return sum(sims) / len(sims) if sims else 0.0


# ============================================================================
# LEXICAL SIMILARITY (TF-IDF)
# ============================================================================

def build_instance_text(row: Dict[str, Any], include_repair: bool = True) -> str:
    """
    Build text representation:
    - Failure type
    - Logs excerpt
    - Workflow excerpt
    - Ground-truth repair (offline benchmark characterization)
    """

    parts = []

    # Failure type
    error_type = row.get("error_type", [])
    if isinstance(error_type, np.ndarray):
        parts.append("Failure: " + ", ".join(str(e) for e in error_type.tolist()))
    elif error_type:
        parts.append(f"Failure: {error_type}")

    # Logs
    logs = row.get("logs", "")
    logs_text = ""
    if isinstance(logs, np.ndarray):
        for item in logs:
            if isinstance(item, dict) and 'log' in item:
                logs_text += str(item['log']) + " "
    elif isinstance(logs, str):
        logs_text = logs

    parts.append(f"Logs: {logs_text[-10000:]}")  # Last 10k chars where errors appear

    # Workflow
    workflow = row.get("workflow", "")
    if workflow:
        parts.append(f"Workflow: {str(workflow)[:1000]}")

    # Ground-truth repair (for offline analysis)
    if include_repair:
        diff = row.get("diff", "")
        if diff:
            parts.append(f"Repair: {str(diff)[:2000]}")

        changed_files = row.get("changed_files", [])
        if isinstance(changed_files, np.ndarray):
            files_str = ", ".join(str(f) for f in changed_files.tolist())
        elif isinstance(changed_files, list):
            files_str = ", ".join(str(f) for f in changed_files)
        else:
            files_str = str(changed_files) if changed_files else ""

        if files_str:
            parts.append(f"Files: {files_str}")

    return " ".join(parts)


# ============================================================================
# PAPER METRICS
# ============================================================================

def compute_paper_metrics(df_results: pd.DataFrame, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute metrics for Table~\ref{tab:benchmark_similarity}:
    - Overall nearest neighbor
    - Within repository
    - Cross repository
    - Historical predecessor (only earlier instances)
    """

    # Create ID to date mapping
    id_to_date = {str(row['id']): row.get('commit_date', '') for _, row in df.iterrows()}
    id_to_repo = {str(row['id']): row['repo_name'] for _, row in df.iterrows()}

    metrics = {}

    # 1. Overall nearest neighbor (max similarity for each instance)
    print("Computing overall nearest neighbor...")
    nearest_jaccard = []
    nearest_tfidf = []

    for instance_id in df['id'].unique():
        pairs = df_results[(df_results['id_a'] == instance_id) | (df_results['id_b'] == instance_id)]
        if len(pairs) > 0:
            nearest_jaccard.append(pairs['jaccard'].max())
            nearest_tfidf.append(pairs['tfidf_cosine'].max())

    metrics['overall_nearest_neighbor'] = {
        'jaccard': round(np.mean(nearest_jaccard), 3) if nearest_jaccard else 0.0,
        'tfidf_cosine': round(np.mean(nearest_tfidf), 3) if nearest_tfidf else 0.0,
    }

    # 2. Within repository
    print("Computing within repository...")
    within_repo = df_results[df_results['same_repo']]
    metrics['within_repository'] = {
        'jaccard': round(within_repo['jaccard'].mean(), 3),
        'tfidf_cosine': round(within_repo['tfidf_cosine'].mean(), 3),
        'pairs': len(within_repo),
    }

    # 3. Cross repository
    print("Computing cross repository...")
    cross_repo = df_results[~df_results['same_repo']]
    metrics['cross_repository'] = {
        'jaccard': round(cross_repo['jaccard'].mean(), 3),
        'tfidf_cosine': round(cross_repo['tfidf_cosine'].mean(), 3),
        'pairs': len(cross_repo),
    }

    # 4. Historical predecessor (only earlier chronologically)
    print("Computing historical predecessor...")
    historical_jaccard = []
    historical_tfidf = []

    for instance_id in df['id'].unique():
        instance_date = id_to_date.get(str(instance_id), '')
        if not instance_date:
            continue

        # Find pairs where other instance is earlier
        pairs_a = df_results[df_results['id_a'] == instance_id].copy()
        pairs_a['other_id'] = pairs_a['id_b']
        pairs_a['other_date'] = pairs_a['id_b'].map(id_to_date)

        pairs_b = df_results[df_results['id_b'] == instance_id].copy()
        pairs_b['other_id'] = pairs_b['id_a']
        pairs_b['other_date'] = pairs_b['id_a'].map(id_to_date)

        all_pairs = pd.concat([pairs_a, pairs_b])

        # Filter to only earlier instances
        earlier = all_pairs[all_pairs['other_date'] < instance_date]

        if len(earlier) > 0:
            historical_jaccard.append(earlier['jaccard'].max())
            historical_tfidf.append(earlier['tfidf_cosine'].max())

    metrics['historical_predecessor'] = {
        'jaccard': round(np.mean(historical_jaccard), 3) if historical_jaccard else 0.0,
        'tfidf_cosine': round(np.mean(historical_tfidf), 3) if historical_tfidf else 0.0,
        'instances_with_predecessor': len(historical_jaccard),
    }

    # 5. Per-repository statistics
    print("Computing per-repository statistics...")
    repo_stats = {}

    for repo in df['repo_name'].unique():
        repo_ids = set(df[df['repo_name'] == repo]['id'].astype(str))

        # Pairs where both are from this repo
        repo_pairs = df_results[
            df_results['id_a'].isin(repo_ids) &
            df_results['id_b'].isin(repo_ids)
        ]

        if len(repo_pairs) > 0:
            repo_stats[repo] = {
                'instances': len(repo_ids),
                'pairs': len(repo_pairs),
                'mean_jaccard': round(repo_pairs['jaccard'].mean(), 3),
                'mean_tfidf': round(repo_pairs['tfidf_cosine'].mean(), 3),
            }

    # Top 10 repos by Jaccard similarity
    top_repos = sorted(repo_stats.items(), key=lambda x: x[1]['mean_jaccard'], reverse=True)[:10]
    metrics['top_10_repositories'] = {repo: stats for repo, stats in top_repos}

    return metrics


def print_latex_table(metrics: Dict[str, Any]):
    """Print LaTeX table for the paper."""

    print("\n" + "="*70)
    print("TABLE FOR PAPER (Table~\\ref{tab:benchmark_similarity})")
    print("="*70)

    print("\n\\begin{table}[t]")
    print("\\caption{Recurrence of similar CI repair instances. Values report mean")
    print("         nearest-neighbor similarity across benchmark instances.}")
    print("\\label{tab:benchmark_similarity}")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("\\textbf{Comparison} &")
    print("\\textbf{Jaccard} &")
    print("\\textbf{TF--IDF Cosine} \\\\")
    print("\\hline")
    print(f"Overall nearest neighbor  & {metrics['overall_nearest_neighbor']['jaccard']:.3f} & {metrics['overall_nearest_neighbor']['tfidf_cosine']:.3f} \\\\")
    print(f"Within repository         & {metrics['within_repository']['jaccard']:.3f} & {metrics['within_repository']['tfidf_cosine']:.3f} \\\\")
    print(f"Cross repository          & {metrics['cross_repository']['jaccard']:.3f} & {metrics['cross_repository']['tfidf_cosine']:.3f} \\\\")
    print(f"Historical predecessor    & {metrics['historical_predecessor']['jaccard']:.3f} & {metrics['historical_predecessor']['tfidf_cosine']:.3f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def print_repository_analysis(metrics: Dict[str, Any]):
    """Print per-repository recurrence analysis."""

    print("\n" + "="*70)
    print("PER-REPOSITORY RECURRENCE (Top 10 by Jaccard)")
    print("="*70)
    print(f"\n{'Repository':<30} {'Instances':>10} {'Pairs':>10} {'Jaccard':>10} {'TF-IDF':>10}")
    print("-" * 70)

    for repo, stats in metrics['top_10_repositories'].items():
        print(f"{repo:<30} {stats['instances']:>10} {stats['pairs']:>10} {stats['mean_jaccard']:>10.3f} {stats['mean_tfidf']:>10.3f}")


def verify_feature_extraction(df: pd.DataFrame):
    """Verify that within-repo uses full paths and cross-repo uses basenames."""

    print("\n" + "="*70)
    print("VERIFICATION: Within-Repo vs Cross-Repo File Handling")
    print("="*70)

    rows = df.to_dict('records')

    # Find one within-repo pair
    within_repo_example = None
    for i in range(min(50, len(rows))):
        for j in range(i + 1, min(50, len(rows))):
            if rows[i]['repo_name'] == rows[j]['repo_name']:
                within_repo_example = (rows[i], rows[j])
                break
        if within_repo_example:
            break

    # Find one cross-repo pair
    cross_repo_example = None
    for i in range(min(50, len(rows))):
        for j in range(i + 1, min(50, len(rows))):
            if rows[i]['repo_name'] != rows[j]['repo_name']:
                cross_repo_example = (rows[i], rows[j])
                break
        if cross_repo_example:
            break

    # Show within-repo example
    if within_repo_example:
        row_a, row_b = within_repo_example
        print(f"\n✓ WITHIN-REPOSITORY EXAMPLE:")
        print(f"  Repo: {row_a['repo_owner']}/{row_a['repo_name']}")
        print(f"  Instance A (ID: {row_a['id']})")
        print(f"  Instance B (ID: {row_b['id']})")

        feat_a = extract_structural_features(row_a, use_full_paths=True)
        feat_b = extract_structural_features(row_b, use_full_paths=True)

        print(f"\n  Changed files (A) - FULL PATHS:")
        for f in list(feat_a['changed_files'])[:3]:
            print(f"    - {f}")

        print(f"\n  Changed files (B) - FULL PATHS:")
        for f in list(feat_b['changed_files'])[:3]:
            print(f"    - {f}")

        jaccard = jaccard_similarity(feat_a['changed_files'], feat_b['changed_files'])
        print(f"\n  File Jaccard similarity: {jaccard:.4f}")

    # Show cross-repo example
    if cross_repo_example:
        row_a, row_b = cross_repo_example
        print(f"\n✓ CROSS-REPOSITORY EXAMPLE:")
        print(f"  Repo A: {row_a['repo_owner']}/{row_a['repo_name']} (ID: {row_a['id']})")
        print(f"  Repo B: {row_b['repo_owner']}/{row_b['repo_name']} (ID: {row_b['id']})")

        feat_a = extract_structural_features(row_a, use_full_paths=False)
        feat_b = extract_structural_features(row_b, use_full_paths=False)

        print(f"\n  Changed files (A) - BASENAMES ONLY:")
        for f in list(feat_a['changed_files'])[:3]:
            print(f"    - {f}")

        print(f"\n  Changed files (B) - BASENAMES ONLY:")
        for f in list(feat_b['changed_files'])[:3]:
            print(f"    - {f}")

        jaccard = jaccard_similarity(feat_a['changed_files'], feat_b['changed_files'])
        print(f"\n  File Jaccard similarity: {jaccard:.4f}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark recurrence analysis for paper")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Sample size (default: full dataset)")
    parser.add_argument("--verify", action="store_true",
                        help="Show verification examples before running full analysis")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*70)
    print("BENCHMARK RECURRENCE ANALYSIS")
    print("="*70)

    # Load dataset
    print(f"\nLoading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"Loaded {len(df)} instances")

    # Sample if requested
    if args.sample_size and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=42)
        print(f"Sampled to {len(df)} instances")

    # Verify feature extraction logic
    if args.verify or (args.sample_size and args.sample_size <= 100):
        verify_feature_extraction(df)

    # Build text representations (includes ground-truth repair)
    print("\nBuilding text representations (with ground-truth repair)...")
    texts = [build_instance_text(row, include_repair=True) for _, row in df.iterrows()]

    # Fit TF-IDF
    print("Computing TF-IDF vectors...")
    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix = tfidf.fit_transform(texts)

    # Compute pairwise similarities
    print(f"Computing pairwise similarities for {len(df)} instances...")
    print(f"Total pairs: {len(df) * (len(df) - 1) // 2:,}")

    rows = df.to_dict('records')
    results = []

    total = len(rows) * (len(rows) - 1) // 2
    computed = 0

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            row_a = rows[i]
            row_b = rows[j]

            # Structural (Jaccard)
            jaccard_sim = compute_structural_similarity(row_a, row_b)

            # Lexical (TF-IDF)
            tfidf_sim = sklearn_cosine(tfidf_matrix[i:i+1], tfidf_matrix[j:j+1])[0][0]

            results.append({
                "id_a": str(row_a["id"]),
                "id_b": str(row_b["id"]),
                "repo_a": row_a['repo_name'],
                "repo_b": row_b['repo_name'],
                "same_repo": row_a["repo_name"] == row_b["repo_name"],
                "jaccard": round(float(jaccard_sim), 4),
                "tfidf_cosine": round(float(tfidf_sim), 4),
            })

            computed += 1
            if computed % 1000 == 0:
                print(f"  Progress: {computed:,}/{total:,} ({100*computed/total:.1f}%)")

    df_results = pd.DataFrame(results)

    # Compute paper metrics
    print("\nComputing paper metrics...")
    paper_metrics = compute_paper_metrics(df_results, df)

    # Save results
    output_file = OUTPUT_DIR / "benchmark_recurrence_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(paper_metrics, f, indent=2)

    csv_file = OUTPUT_DIR / "recurrence_pairs.csv"
    df_results.to_csv(csv_file, index=False)

    print(f"\n✓ Results saved to: {output_file}")
    print(f"✓ Pairs saved to: {csv_file}")

    # Print outputs
    print_latex_table(paper_metrics)
    print_repository_analysis(paper_metrics)

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nMetrics:")
    print(f"  Same-repo pairs: {paper_metrics['within_repository']['pairs']:,}")
    print(f"  Cross-repo pairs: {paper_metrics['cross_repository']['pairs']:,}")
    print(f"  Instances with historical precedent: {paper_metrics['historical_predecessor']['instances_with_predecessor']:,}")
    print("\nThis is OFFLINE benchmark characterization (includes ground-truth repairs).")
    print("NOT what's available at retrieval time (that's for MemRepair evaluation).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Recurrence Analysis Using Structured CI Context

Pipeline:
    Raw CI Evidence → Structured CI Context (LLM) → Similarity Analysis

This script separates:
1. CI failure evidence (from structured extraction)
2. Developer repair (from ground-truth patches)

This distinction is important because:
- affected_files (from CI) = failure localization
- changed_files (from patch) = actual repair
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
STRUCTURED_CI_PATH = PROJECT_ROOT / "log_details.json"
OUTPUT_DIR = PROJECT_ROOT / "dataset_overview"


# ============================================================================
# STRUCTURED CI CONTEXT (Extracted via LLM)
# ============================================================================

def load_structured_ci_contexts() -> Dict[str, Dict[str, Any]]:
    """
    Load LLM-extracted structured CI contexts.

    Schema:
        error_context: High-level failure explanation
        failure_signals: Specific error messages (tool + error + location)
        relevant_files: Files mentioned in CI failure evidence
        error_types: Categorized failure types
        failed_job: Job/step/command information

    This is the CI failure evidence, separate from developer repair.
    """
    with open(STRUCTURED_CI_PATH) as f:
        contexts = json.load(f)

    # Index by ID for lookup
    return {str(ctx['id']): ctx for ctx in contexts}


def extract_ci_structural_features(
    ci_context: Dict[str, Any],
    use_full_paths: bool = False
) -> Dict[str, Set[str]]:
    """
    Extract structural features from CI failure evidence.

    Features:
    - failure_categories: Top-level error categories
    - failure_subcategories: Specific error types
    - tools: CI tools that detected failures
    - validation_commands: Commands that failed
    - affected_files: Files mentioned in CI failure evidence

    Args:
        ci_context: Structured CI context from LLM extraction
        use_full_paths: If True, use full paths; if False, basenames only
    """

    # Failure categories and subcategories
    error_types = ci_context.get("error_types", [])
    categories = {et.get("category", "").strip().lower() for et in error_types if et.get("category")}
    subcategories = {et.get("subcategory", "").strip().lower() for et in error_types if et.get("subcategory")}

    # Tools that detected failures
    relevant_files = ci_context.get("relevant_files", [])
    tools = {rf.get("failed_tool", "").strip().lower() for rf in relevant_files if rf.get("failed_tool")}

    # Validation commands
    failed_jobs = ci_context.get("failed_job", [])
    commands = {fj.get("command", "").strip().lower() for fj in failed_jobs if fj.get("command")}

    # Affected files (from CI failure evidence, not developer patch)
    if use_full_paths:
        affected_files = {
            rf.get("file", "").strip().lower().replace("\\", "/")
            for rf in relevant_files if rf.get("file")
        }
    else:
        # Cross-repo: only basenames
        affected_files = {
            Path(rf.get("file", "")).name.strip().lower()
            for rf in relevant_files if rf.get("file")
        }

    return {
        "failure_categories": categories,
        "failure_subcategories": subcategories,
        "tools": tools,
        "validation_commands": commands,
        "affected_files": affected_files,
    }


def extract_repair_structural_features(
    row: Dict[str, Any],
    use_full_paths: bool = False
) -> Dict[str, Set[str]]:
    """
    Extract structural features from developer repair (ground-truth patch).

    Features:
    - changed_files: Files actually modified by developer

    This is SEPARATE from affected_files in CI evidence because:
    - affected_files = failure localization (from CI)
    - changed_files = actual repair (from developer patch)
    """

    changed_files = row.get("changed_files", [])
    if isinstance(changed_files, np.ndarray):
        files = changed_files.tolist()
    elif isinstance(changed_files, list):
        files = changed_files
    else:
        files = [changed_files] if changed_files else []

    if use_full_paths:
        file_set = {str(f).strip().lower().replace("\\", "/") for f in files if f}
    else:
        file_set = {Path(str(f)).name.strip().lower() for f in files if f}

    return {
        "changed_files": file_set,
    }


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity = |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_structural_similarity(
    row_a: Dict,
    row_b: Dict,
    ci_contexts: Dict[str, Dict]
) -> float:
    """
    Structural similarity using normalized CI contexts.

    Combines:
    1. CI failure evidence (from structured extraction)
    2. Developer repair (from ground-truth patches)
    """

    # Determine if same repository
    repo_a = f"{row_a.get('repo_owner')}/{row_a.get('repo_name')}"
    repo_b = f"{row_b.get('repo_owner')}/{row_b.get('repo_name')}"
    same_repo = (repo_a == repo_b)

    # Get structured CI contexts
    ci_a = ci_contexts.get(str(row_a['id']), {})
    ci_b = ci_contexts.get(str(row_b['id']), {})

    # Extract CI failure features
    ci_feat_a = extract_ci_structural_features(ci_a, use_full_paths=same_repo)
    ci_feat_b = extract_ci_structural_features(ci_b, use_full_paths=same_repo)

    # Extract developer repair features
    repair_feat_a = extract_repair_structural_features(row_a, use_full_paths=same_repo)
    repair_feat_b = extract_repair_structural_features(row_b, use_full_paths=same_repo)

    # Compute Jaccard across all features
    sims = [
        # CI failure evidence
        jaccard_similarity(ci_feat_a["failure_categories"], ci_feat_b["failure_categories"]),
        jaccard_similarity(ci_feat_a["failure_subcategories"], ci_feat_b["failure_subcategories"]),
        jaccard_similarity(ci_feat_a["tools"], ci_feat_b["tools"]),
        jaccard_similarity(ci_feat_a["validation_commands"], ci_feat_b["validation_commands"]),
        jaccard_similarity(ci_feat_a["affected_files"], ci_feat_b["affected_files"]),
        # Developer repair
        jaccard_similarity(repair_feat_a["changed_files"], repair_feat_b["changed_files"]),
    ]

    return sum(sims) / len(sims) if sims else 0.0


# ============================================================================
# LEXICAL SIMILARITY (TF-IDF)
# ============================================================================

def build_instance_text(
    row: Dict[str, Any],
    ci_context: Dict[str, Any],
    include_repair: bool = True
) -> str:
    """
    Build text representation from structured CI context + developer repair.

    Components:
    1. CI failure evidence (from structured extraction):
       - Error context (high-level summary)
       - Failure signals (specific errors)
       - Workflow information

    2. Developer repair (ground-truth):
       - Diff patch
       - Changed files
    """

    parts = []

    # === CI FAILURE EVIDENCE ===

    # High-level error context
    error_context = ci_context.get("error_context", [])
    if error_context:
        parts.append("Context: " + " ".join(str(ec) for ec in error_context))

    # Failure signals (last 10k chars where errors appear)
    failure_signals = ci_context.get("failure_signals", [])
    if failure_signals:
        signals_text = " ".join(str(fs) for fs in failure_signals)
        parts.append(f"Signals: {signals_text[-10000:]}")

    # Error types
    error_types = ci_context.get("error_types", [])
    for et in error_types:
        if et.get("category"):
            parts.append(f"Category: {et['category']}")
        if et.get("subcategory"):
            parts.append(f"Type: {et['subcategory']}")

    # Failed job/command info
    failed_jobs = ci_context.get("failed_job", [])
    for fj in failed_jobs:
        if fj.get("command"):
            parts.append(f"Command: {fj['command']}")

    # Workflow (from original dataset)
    workflow = row.get("workflow", "")
    if workflow:
        parts.append(f"Workflow: {str(workflow)[:1000]}")

    # === DEVELOPER REPAIR (GROUND-TRUTH) ===

    if include_repair:
        # Repair patch
        diff = row.get("diff", "")
        if diff:
            parts.append(f"Repair: {str(diff)[:2000]}")

        # Changed files
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
    Compute metrics for paper Table.

    Metrics:
    - Overall nearest neighbor (max similarity for each instance)
    - Within repository (same repo pairs)
    - Cross repository (different repo pairs)
    - Historical predecessor (chronologically earlier only)
    """

    # Create ID to date mapping
    id_to_date = {str(row['id']): row.get('commit_date', '') for _, row in df.iterrows()}

    metrics = {}

    # 1. Overall nearest neighbor
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

    # 4. Historical predecessor
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

    top_repos = sorted(repo_stats.items(), key=lambda x: x[1]['mean_jaccard'], reverse=True)[:10]
    metrics['top_10_repositories'] = {repo: stats for repo, stats in top_repos}

    return metrics


def print_latex_table(metrics: Dict[str, Any]):
    """Print LaTeX table for paper."""

    print("\n" + "="*70)
    print("LATEX TABLE (using structured CI contexts)")
    print("="*70)

    print("\n\\begin{table}[t]")
    print("\\caption{Recurrence of similar CI repair instances. Values report mean")
    print("         nearest-neighbor similarity across benchmark instances.}")
    print("\\label{tab:benchmark_similarity}")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("\\textbf{Comparison} &")
    print("\\textbf{Structural (Jaccard)} &")
    print("\\textbf{Lexical (Cosine)} \\\\")
    print("\\hline")
    print(f"Overall nearest neighbor  & {metrics['overall_nearest_neighbor']['jaccard']:.3f} & {metrics['overall_nearest_neighbor']['tfidf_cosine']:.3f} \\\\")
    print(f"Within repository         & {metrics['within_repository']['jaccard']:.3f} & {metrics['within_repository']['tfidf_cosine']:.3f} \\\\")
    print(f"Cross repository          & {metrics['cross_repository']['jaccard']:.3f} & {metrics['cross_repository']['tfidf_cosine']:.3f} \\\\")
    print(f"Historical predecessor    & {metrics['historical_predecessor']['jaccard']:.3f} & {metrics['historical_predecessor']['tfidf_cosine']:.3f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Recurrence analysis using structured CI contexts")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Sample size (default: full dataset)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*70)
    print("RECURRENCE ANALYSIS (Structured CI Context Pipeline)")
    print("="*70)

    # Load structured CI contexts
    print(f"\nLoading structured CI contexts from {STRUCTURED_CI_PATH}...")
    ci_contexts = load_structured_ci_contexts()
    print(f"Loaded {len(ci_contexts)} structured CI contexts")

    # Load dataset
    print(f"\nLoading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"Loaded {len(df)} instances")

    # Filter to instances with structured CI contexts
    df = df[df['id'].astype(str).isin(ci_contexts.keys())]
    print(f"Filtered to {len(df)} instances with structured CI contexts")

    # Sample if requested
    if args.sample_size and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=42)
        print(f"Sampled to {len(df)} instances")

    # Build text representations
    print("\nBuilding text representations (structured CI context + ground-truth repair)...")
    rows = df.to_dict('records')
    texts = [
        build_instance_text(row, ci_contexts.get(str(row['id']), {}), include_repair=True)
        for row in rows
    ]

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
    print(f"Total pairs: {len(rows) * (len(rows) - 1) // 2:,}")

    results = []
    total = len(rows) * (len(rows) - 1) // 2
    computed = 0

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            row_a = rows[i]
            row_b = rows[j]

            # Structural (Jaccard on normalized CI context)
            jaccard_sim = compute_structural_similarity(row_a, row_b, ci_contexts)

            # Lexical (TF-IDF cosine)
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
    output_file = OUTPUT_DIR / "recurrence_analysis_structured.json"
    with open(output_file, 'w') as f:
        json.dump(paper_metrics, f, indent=2)

    csv_file = OUTPUT_DIR / "recurrence_pairs_structured.csv"
    df_results.to_csv(csv_file, index=False)

    print(f"\n✓ Results saved to: {output_file}")
    print(f"✓ Pairs saved to: {csv_file}")

    # Print outputs
    print_latex_table(paper_metrics)

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nMetrics:")
    print(f"  Instances analyzed: {len(df):,}")
    print(f"  Same-repo pairs: {paper_metrics['within_repository']['pairs']:,}")
    print(f"  Cross-repo pairs: {paper_metrics['cross_repository']['pairs']:,}")
    print(f"  Instances with historical precedent: {paper_metrics['historical_predecessor']['instances_with_predecessor']:,}")

    print("\n📋 Pipeline:")
    print("  Raw CI Evidence → Structured CI Context (LLM) → Similarity Analysis")
    print("\n🔍 Key distinction:")
    print("  - affected_files (from CI) = failure localization")
    print("  - changed_files (from patch) = actual repair")


if __name__ == "__main__":
    main()

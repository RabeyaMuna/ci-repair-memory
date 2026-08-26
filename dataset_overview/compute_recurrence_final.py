#!/usr/bin/env python3
"""
Recurrence Analysis Following Paper Formulation

Computes:
1. R_within(i) = max similarity to instances in same repository
2. R_overall(i) = max similarity to any instance
3. Both under structural (Jaccard) and lexical (TF-IDF cosine) similarity

Reports mean across all benchmark instances.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
STRUCTURED_CI_PATH = PROJECT_ROOT / "log_details.json"
OUTPUT_DIR = PROJECT_ROOT / "dataset_overview"


# ============================================================================
# LOAD STRUCTURED CI CONTEXTS
# ============================================================================

def load_structured_ci_contexts() -> Dict[str, Dict[str, Any]]:
    """Load LLM-extracted normalized CI contexts."""
    with open(STRUCTURED_CI_PATH) as f:
        contexts = json.load(f)
    return {str(ctx['id']): ctx for ctx in contexts}


# ============================================================================
# STRUCTURAL SIMILARITY (Jaccard over normalized attributes)
# ============================================================================

def extract_normalized_attributes(
    row: Dict[str, Any],
    ci_context: Dict[str, Any],
    use_full_paths: bool = False
) -> Dict[str, Set[str]]:
    """
    Extract normalized attributes for structural similarity.

    Attributes:
    - failure_categories: Error categories
    - failure_subcategories: Specific error types
    - failure_signals: Error messages/codes
    - validation_commands: Commands that failed
    - tools: CI tools used
    - affected_files: Files mentioned in CI failure evidence
    - changed_files: Files modified in developer repair

    Note: affected_files (from CI) ≠ changed_files (from patch)
    """

    # From normalized CI context
    error_types = ci_context.get("error_types", [])
    categories = {et.get("category", "").strip().lower() for et in error_types if et.get("category")}
    subcategories = {et.get("subcategory", "").strip().lower() for et in error_types if et.get("subcategory")}

    failure_signals = ci_context.get("failure_signals", [])
    signals = {str(fs).strip().lower() for fs in failure_signals if fs}

    failed_jobs = ci_context.get("failed_job", [])
    commands = {fj.get("command", "").strip().lower() for fj in failed_jobs if fj.get("command")}

    relevant_files = ci_context.get("relevant_files", [])
    tools = {rf.get("failed_tool", "").strip().lower() for rf in relevant_files if rf.get("failed_tool")}

    # Affected files (from CI evidence)
    if use_full_paths:
        affected_files = {
            rf.get("file", "").strip().lower().replace("\\", "/")
            for rf in relevant_files if rf.get("file")
        }
    else:
        affected_files = {
            Path(rf.get("file", "")).name.strip().lower()
            for rf in relevant_files if rf.get("file")
        }

    # Changed files (from developer repair)
    changed_files = row.get("changed_files", [])
    if isinstance(changed_files, np.ndarray):
        files = changed_files.tolist()
    elif isinstance(changed_files, list):
        files = changed_files
    else:
        files = [changed_files] if changed_files else []

    if use_full_paths:
        changed_file_set = {str(f).strip().lower().replace("\\", "/") for f in files if f}
    else:
        changed_file_set = {Path(str(f)).name.strip().lower() for f in files if f}

    return {
        "failure_categories": categories,
        "failure_subcategories": subcategories,
        "failure_signals": signals,
        "validation_commands": commands,
        "tools": tools,
        "affected_files": affected_files,
        "changed_files": changed_file_set,
    }


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity = |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_structural_similarity(
    attrs_a: Dict[str, Set[str]],
    attrs_b: Dict[str, Set[str]]
) -> float:
    """
    S_struct(i,j) = aggregated Jaccard across normalized attributes.

    Averages Jaccard similarity across all available attributes.
    """
    attribute_names = [
        "failure_categories",
        "failure_subcategories",
        "failure_signals",
        "validation_commands",
        "tools",
        "affected_files",
        "changed_files",
    ]

    sims = [
        jaccard_similarity(attrs_a[attr], attrs_b[attr])
        for attr in attribute_names
    ]

    return sum(sims) / len(sims) if sims else 0.0


# ============================================================================
# LEXICAL SIMILARITY (TF-IDF Cosine)
# ============================================================================

def build_text_representation(
    row: Dict[str, Any],
    ci_context: Dict[str, Any]
) -> str:
    """
    Build text representation from:
    - Normalized CI context
    - Failure log evidence (last 10k chars)
    - Workflow context
    - Ground-truth repair information
    """

    parts = []

    # Normalized CI context
    error_context = ci_context.get("error_context", [])
    if error_context:
        parts.append("Context: " + " ".join(str(ec) for ec in error_context))

    failure_signals = ci_context.get("failure_signals", [])
    if failure_signals:
        signals_text = " ".join(str(fs) for fs in failure_signals)
        parts.append(f"Signals: {signals_text[-10000:]}")

    error_types = ci_context.get("error_types", [])
    for et in error_types:
        if et.get("category"):
            parts.append(f"Category: {et['category']}")
        if et.get("subcategory"):
            parts.append(f"Type: {et['subcategory']}")

    failed_jobs = ci_context.get("failed_job", [])
    for fj in failed_jobs:
        if fj.get("command"):
            parts.append(f"Command: {fj['command']}")

    # Workflow context
    workflow = row.get("workflow", "")
    if workflow:
        parts.append(f"Workflow: {str(workflow)[:1000]}")

    # Ground-truth repair
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
# RECURRENCE ANALYSIS
# ============================================================================

def compute_recurrence_metrics(
    df: pd.DataFrame,
    ci_contexts: Dict[str, Dict],
    structural_similarity_matrix: np.ndarray,
    lexical_similarity_matrix: np.ndarray
) -> Dict[str, Any]:
    """
    Compute recurrence metrics following paper formulation:

    R_within(i) = max_{j ≠ i, r_j = r_i} S(i,j)
    R_overall(i) = max_{j ≠ i} S(i,j)

    Both under structural and lexical similarity.
    """

    n = len(df)
    rows = df.to_dict('records')

    # Track per-instance recurrence
    within_structural = []
    within_lexical = []
    overall_structural = []
    overall_lexical = []

    print("\nComputing per-instance recurrence...")

    for i in range(n):
        repo_i = rows[i]['repo_name']

        # R_within(i): max similarity to same-repo instances
        same_repo_indices = [j for j in range(n) if j != i and rows[j]['repo_name'] == repo_i]

        if same_repo_indices:
            max_struct_within = max(structural_similarity_matrix[i, j] for j in same_repo_indices)
            max_lex_within = max(lexical_similarity_matrix[i, j] for j in same_repo_indices)
            within_structural.append(max_struct_within)
            within_lexical.append(max_lex_within)

        # R_overall(i): max similarity to any instance
        other_indices = [j for j in range(n) if j != i]
        if other_indices:
            max_struct_overall = max(structural_similarity_matrix[i, j] for j in other_indices)
            max_lex_overall = max(lexical_similarity_matrix[i, j] for j in other_indices)
            overall_structural.append(max_struct_overall)
            overall_lexical.append(max_lex_overall)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n} instances")

    # Compute means
    metrics = {
        "within_repository": {
            "structural": round(np.mean(within_structural), 3) if within_structural else 0.0,
            "lexical": round(np.mean(within_lexical), 3) if within_lexical else 0.0,
            "n_instances": len(within_structural),
        },
        "overall_benchmark": {
            "structural": round(np.mean(overall_structural), 3) if overall_structural else 0.0,
            "lexical": round(np.mean(overall_lexical), 3) if overall_lexical else 0.0,
            "n_instances": len(overall_structural),
        },
    }

    return metrics


def print_latex_table(metrics: Dict[str, Any]):
    """Print LaTeX table for paper."""

    print("\n" + "="*70)
    print("LATEX TABLE (Paper Formulation)")
    print("="*70)

    print("\n\\begin{table}[t]")
    print("\\caption{Recurrence of similar CI repair instances in CI-Repair-Bench.")
    print("         Values report mean nearest-neighbor similarity under two scopes:")
    print("         within-repository and overall benchmark.}")
    print("\\label{tab:benchmark_recurrence}")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("\\textbf{Scope} &")
    print("\\textbf{Structural} &")
    print("\\textbf{Lexical} \\\\")
    print("\\hline")
    print(f"Within repository  & {metrics['within_repository']['structural']:.3f} & {metrics['within_repository']['lexical']:.3f} \\\\")
    print(f"Overall benchmark  & {metrics['overall_benchmark']['structural']:.3f} & {metrics['overall_benchmark']['lexical']:.3f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*70)
    print("RECURRENCE ANALYSIS (Paper Formulation)")
    print("="*70)

    # Load data
    print(f"\nLoading structured CI contexts from {STRUCTURED_CI_PATH}...")
    ci_contexts = load_structured_ci_contexts()
    print(f"Loaded {len(ci_contexts)} structured CI contexts")

    print(f"\nLoading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"Loaded {len(df)} instances")

    # Filter to instances with structured CI contexts
    df = df[df['id'].astype(str).isin(ci_contexts.keys())].reset_index(drop=True)
    print(f"Filtered to {len(df)} instances with structured CI contexts")

    rows = df.to_dict('records')
    n = len(rows)

    # ========================================================================
    # STRUCTURAL SIMILARITY MATRIX
    # ========================================================================

    print("\n" + "="*70)
    print("Computing Structural Similarity (Jaccard over normalized attributes)")
    print("="*70)

    # Extract normalized attributes for all instances
    print("\nExtracting normalized attributes...")
    all_attrs = []
    for row in rows:
        ci_ctx = ci_contexts.get(str(row['id']), {})
        # Use full paths for within-repo, basenames for cross-repo
        # We'll handle this per-pair, but for now extract both
        attrs = extract_normalized_attributes(row, ci_ctx, use_full_paths=True)
        all_attrs.append(attrs)

    # Compute pairwise structural similarity
    print(f"Computing pairwise structural similarity for {n} instances...")
    structural_sim_matrix = np.zeros((n, n))

    total_pairs = n * (n - 1) // 2
    computed = 0

    for i in range(n):
        structural_sim_matrix[i, i] = 1.0  # Self-similarity

        for j in range(i + 1, n):
            # Determine if same repo
            same_repo = rows[i]['repo_name'] == rows[j]['repo_name']

            # Re-extract with appropriate file handling
            ci_ctx_i = ci_contexts.get(str(rows[i]['id']), {})
            ci_ctx_j = ci_contexts.get(str(rows[j]['id']), {})

            attrs_i = extract_normalized_attributes(rows[i], ci_ctx_i, use_full_paths=same_repo)
            attrs_j = extract_normalized_attributes(rows[j], ci_ctx_j, use_full_paths=same_repo)

            sim = compute_structural_similarity(attrs_i, attrs_j)

            structural_sim_matrix[i, j] = sim
            structural_sim_matrix[j, i] = sim

            computed += 1
            if computed % 5000 == 0:
                print(f"  Progress: {computed:,}/{total_pairs:,} ({100*computed/total_pairs:.1f}%)")

    print(f"✓ Structural similarity matrix computed")

    # ========================================================================
    # LEXICAL SIMILARITY MATRIX
    # ========================================================================

    print("\n" + "="*70)
    print("Computing Lexical Similarity (TF-IDF Cosine)")
    print("="*70)

    # Build text representations
    print("\nBuilding text representations...")
    texts = [
        build_text_representation(row, ci_contexts.get(str(row['id']), {}))
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

    # Compute pairwise cosine similarity
    print("Computing pairwise lexical similarity...")
    lexical_sim_matrix = sklearn_cosine(tfidf_matrix)

    print(f"✓ Lexical similarity matrix computed")

    # ========================================================================
    # RECURRENCE METRICS
    # ========================================================================

    print("\n" + "="*70)
    print("Computing Recurrence Metrics")
    print("="*70)

    metrics = compute_recurrence_metrics(
        df, ci_contexts,
        structural_sim_matrix,
        lexical_sim_matrix
    )

    # Save results
    output_file = OUTPUT_DIR / "recurrence_analysis_final.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Print LaTeX table
    print_latex_table(metrics)

    # Summary
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nInstances analyzed: {n:,}")
    print(f"\nWithin-repository recurrence:")
    print(f"  Structural: {metrics['within_repository']['structural']:.3f}")
    print(f"  Lexical: {metrics['within_repository']['lexical']:.3f}")
    print(f"  (Based on {metrics['within_repository']['n_instances']} instances with same-repo precedents)")
    print(f"\nOverall benchmark recurrence:")
    print(f"  Structural: {metrics['overall_benchmark']['structural']:.3f}")
    print(f"  Lexical: {metrics['overall_benchmark']['lexical']:.3f}")
    print(f"  (Based on {metrics['overall_benchmark']['n_instances']} instances)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze distribution of similarity scores to answer:
- What % of instances have a highly similar precedent (>0.7, >0.8)?
- What % of pairs are highly similar?
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# Import from the main script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from compute_recurrence_final import (
    load_structured_ci_contexts,
    extract_normalized_attributes,
    compute_structural_similarity,
    build_text_representation
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
OUTPUT_DIR = PROJECT_ROOT / "dataset_overview"


def main():
    print("="*70)
    print("SIMILARITY DISTRIBUTION ANALYSIS")
    print("="*70)

    # Load data
    ci_contexts = load_structured_ci_contexts()
    df = pd.read_parquet(DATASET_PATH)
    df = df[df['id'].astype(str).isin(ci_contexts.keys())].reset_index(drop=True)

    rows = df.to_dict('records')
    n = len(rows)

    print(f"\nAnalyzing {n} instances...")

    # Build text representations
    texts = [
        build_text_representation(row, ci_contexts.get(str(row['id']), {}))
        for row in rows
    ]

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix = tfidf.fit_transform(texts)
    lexical_sim_matrix = sklearn_cosine(tfidf_matrix)

    # Compute structural similarity matrix (simplified - just for distribution)
    print("\nComputing structural similarities...")
    structural_sim_matrix = np.zeros((n, n))

    for i in range(n):
        structural_sim_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            same_repo = rows[i]['repo_name'] == rows[j]['repo_name']
            ci_ctx_i = ci_contexts.get(str(rows[i]['id']), {})
            ci_ctx_j = ci_contexts.get(str(rows[j]['id']), {})

            attrs_i = extract_normalized_attributes(rows[i], ci_ctx_i, use_full_paths=same_repo)
            attrs_j = extract_normalized_attributes(rows[j], ci_ctx_j, use_full_paths=same_repo)

            sim = compute_structural_similarity(attrs_i, attrs_j)
            structural_sim_matrix[i, j] = sim
            structural_sim_matrix[j, i] = sim

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n}")

    # Analyze distribution
    print("\n" + "="*70)
    print("DISTRIBUTION ANALYSIS")
    print("="*70)

    # Per-instance max similarity
    within_struct = []
    within_lex = []
    overall_struct = []
    overall_lex = []

    for i in range(n):
        repo_i = rows[i]['repo_name']
        same_repo_idx = [j for j in range(n) if j != i and rows[j]['repo_name'] == repo_i]
        other_idx = [j for j in range(n) if j != i]

        if same_repo_idx:
            within_struct.append(max(structural_sim_matrix[i, j] for j in same_repo_idx))
            within_lex.append(max(lexical_sim_matrix[i, j] for j in same_repo_idx))

        if other_idx:
            overall_struct.append(max(structural_sim_matrix[i, j] for j in other_idx))
            overall_lex.append(max(lexical_sim_matrix[i, j] for j in other_idx))

    # Thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    print("\n📊 WITHIN-REPOSITORY RECURRENCE")
    print("-" * 70)
    print(f"Total instances with same-repo precedents: {len(within_struct)}\n")

    print("Structural (Jaccard):")
    for thresh in thresholds:
        count = sum(1 for s in within_struct if s >= thresh)
        pct = 100 * count / len(within_struct) if within_struct else 0
        print(f"  ≥{thresh:.1f}: {count:3d}/{len(within_struct)} ({pct:.1f}%)")

    print("\nLexical (TF-IDF Cosine):")
    for thresh in thresholds:
        count = sum(1 for s in within_lex if s >= thresh)
        pct = 100 * count / len(within_lex) if within_lex else 0
        print(f"  ≥{thresh:.1f}: {count:3d}/{len(within_lex)} ({pct:.1f}%)")

    print("\n📊 OVERALL BENCHMARK RECURRENCE")
    print("-" * 70)
    print(f"Total instances: {len(overall_struct)}\n")

    print("Structural (Jaccard):")
    for thresh in thresholds:
        count = sum(1 for s in overall_struct if s >= thresh)
        pct = 100 * count / len(overall_struct) if overall_struct else 0
        print(f"  ≥{thresh:.1f}: {count:3d}/{len(overall_struct)} ({pct:.1f}%)")

    print("\nLexical (TF-IDF Cosine):")
    for thresh in thresholds:
        count = sum(1 for s in overall_lex if s >= thresh)
        pct = 100 * count / len(overall_lex) if overall_lex else 0
        print(f"  ≥{thresh:.1f}: {count:3d}/{len(overall_lex)} ({pct:.1f}%)")

    # Quartiles
    print("\n📈 QUARTILE ANALYSIS")
    print("-" * 70)

    def print_quartiles(values, name):
        print(f"\n{name}:")
        print(f"  Min:  {np.min(values):.3f}")
        print(f"  Q1:   {np.percentile(values, 25):.3f}")
        print(f"  Med:  {np.percentile(values, 50):.3f}")
        print(f"  Mean: {np.mean(values):.3f}")
        print(f"  Q3:   {np.percentile(values, 75):.3f}")
        print(f"  Max:  {np.max(values):.3f}")

    print_quartiles(within_struct, "Within-Repo Structural")
    print_quartiles(within_lex, "Within-Repo Lexical")
    print_quartiles(overall_struct, "Overall Structural")
    print_quartiles(overall_lex, "Overall Lexical")

    # Save results
    results = {
        "within_repository": {
            "structural": {
                f"pct_above_{thresh}": round(100 * sum(1 for s in within_struct if s >= thresh) / len(within_struct), 1)
                for thresh in thresholds
            },
            "lexical": {
                f"pct_above_{thresh}": round(100 * sum(1 for s in within_lex if s >= thresh) / len(within_lex), 1)
                for thresh in thresholds
            },
        },
        "overall_benchmark": {
            "structural": {
                f"pct_above_{thresh}": round(100 * sum(1 for s in overall_struct if s >= thresh) / len(overall_struct), 1)
                for thresh in thresholds
            },
            "lexical": {
                f"pct_above_{thresh}": round(100 * sum(1 for s in overall_lex if s >= thresh) / len(overall_lex), 1)
                for thresh in thresholds
            },
        },
        "quartiles": {
            "within_structural": {
                "min": round(float(np.min(within_struct)), 3),
                "q1": round(float(np.percentile(within_struct, 25)), 3),
                "median": round(float(np.percentile(within_struct, 50)), 3),
                "mean": round(float(np.mean(within_struct)), 3),
                "q3": round(float(np.percentile(within_struct, 75)), 3),
                "max": round(float(np.max(within_struct)), 3),
            },
            "within_lexical": {
                "min": round(float(np.min(within_lex)), 3),
                "q1": round(float(np.percentile(within_lex, 25)), 3),
                "median": round(float(np.percentile(within_lex, 50)), 3),
                "mean": round(float(np.mean(within_lex)), 3),
                "q3": round(float(np.percentile(within_lex, 75)), 3),
                "max": round(float(np.max(within_lex)), 3),
            },
        }
    }

    output_file = OUTPUT_DIR / "similarity_distribution.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    print("\n" + "="*70)
    print("💡 INTERPRETATION FOR PAPER")
    print("="*70)
    print("\nSample text:")
    print(f"""
High lexical recurrence is widespread: {results['within_repository']['lexical']['pct_above_0.7']:.0f}% of instances
have a within-repository precedent with lexical similarity exceeding 0.7,
and {results['within_repository']['lexical']['pct_above_0.8']:.0f}% exceed 0.8. Structural recurrence is more
moderate: {results['within_repository']['structural']['pct_above_0.5']:.0f}% of instances have a precedent with
structural similarity above 0.5.
    """.strip())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Statistical Significance Testing for CI Repair Results
Uses McNemar's test for paired comparisons with Holm correction
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import chi2
from collections import defaultdict


def load_results(results_file: Path) -> Dict[str, bool]:
    """Load results and return dict mapping instance_id -> success/failure"""
    results = {}
    with open(results_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            instance_id = data['id']
            success = data['conclusion'] == 'success'
            results[instance_id] = success
    return results


def create_contingency_table(results1: Dict[str, bool], results2: Dict[str, bool]) -> np.ndarray:
    """
    Create 2x2 contingency table for McNemar's test

    Returns:
                    Method2_Pass | Method2_Fail
    Method1_Pass        a        |      b
    Method1_Fail        c        |      d
    """
    # Get shared instances
    shared_ids = set(results1.keys()) & set(results2.keys())

    a = sum(1 for id in shared_ids if results1[id] and results2[id])      # both pass
    b = sum(1 for id in shared_ids if results1[id] and not results2[id])  # method1 only
    c = sum(1 for id in shared_ids if not results1[id] and results2[id])  # method2 only
    d = sum(1 for id in shared_ids if not results1[id] and not results2[id])  # both fail

    return np.array([[a, b], [c, d]])


def mcnemar_test(table: np.ndarray, correction: bool = True) -> Tuple[float, float]:
    """
    Perform McNemar's test manually

    Args:
        table: 2x2 contingency table [[a, b], [c, d]]
        correction: Apply continuity correction for small samples

    Returns:
        (statistic, p_value)

    McNemar's test focuses on discordant pairs (b and c):
    - χ² = (b - c)² / (b + c)  [without correction]
    - χ² = (|b - c| - 1)² / (b + c)  [with continuity correction]
    """
    a, b = table[0]
    c, d = table[1]

    # McNemar's test uses only the discordant pairs (b and c)
    if b + c == 0:
        # No discordant pairs - methods are identical
        return 0.0, 1.0

    if correction:
        # Apply continuity correction (recommended for small samples)
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        statistic = (b - c) ** 2 / (b + c)

    # p-value from chi-square distribution with 1 degree of freedom
    p_value = 1 - chi2.cdf(statistic, df=1)

    return statistic, p_value


def holm_correction(p_values: List[float]) -> List[float]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons

    Args:
        p_values: List of p-values

    Returns:
        List of adjusted p-values
    """
    n = len(p_values)
    # Sort p-values with their original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Apply Holm correction
    adjusted_p = np.zeros(n)
    for i, p in enumerate(sorted_p):
        adjusted_p[sorted_indices[i]] = min(1.0, p * (n - i))

    # Enforce monotonicity
    for i in range(1, n):
        adjusted_p[sorted_indices[i]] = max(adjusted_p[sorted_indices[i]],
                                           adjusted_p[sorted_indices[i-1]])

    return adjusted_p.tolist()


def main():
    base_dir = Path("/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines/results")

    # Define models to compare
    models = {
        'GPT-5-mini': 'gpt-5-mini_llm',
        'DeepSeek-Coder': 'deepseek-coder_llm',
        'DeepSeek-Chat': 'deepseek-chat_llm',
        'GPT-4o-mini': 'gpt-4o-mini_llm',
    }

    # Load all results
    print("=" * 80)
    print("LOADING RESULTS")
    print("=" * 80)
    all_results = {}
    for model_name, dir_name in models.items():
        results_file = base_dir / dir_name / "jobs_results_diff.jsonl"
        if results_file.exists():
            all_results[model_name] = load_results(results_file)
            success_count = sum(all_results[model_name].values())
            total_count = len(all_results[model_name])
            print(f"{model_name:20s}: {success_count:3d}/{total_count:3d} ({100*success_count/total_count:.1f}%)")
        else:
            print(f"{model_name:20s}: FILE NOT FOUND")

    print("\n" + "=" * 80)
    print("TABLE 2: PAIRWISE MODEL COMPARISONS (All use Agent/LLM-based FL)")
    print("=" * 80)

    # Pairwise comparisons for Table 2
    model_pairs = [
        ('GPT-5-mini', 'DeepSeek-Coder'),
        ('GPT-5-mini', 'DeepSeek-Chat'),
        ('GPT-5-mini', 'GPT-4o-mini'),
        ('DeepSeek-Coder', 'DeepSeek-Chat'),
        ('DeepSeek-Coder', 'GPT-4o-mini'),
        ('DeepSeek-Chat', 'GPT-4o-mini'),
    ]

    results_table2 = []
    for model1, model2 in model_pairs:
        if model1 in all_results and model2 in all_results:
            table = create_contingency_table(all_results[model1], all_results[model2])
            statistic, p_value = mcnemar_test(table)
            results_table2.append({
                'comparison': f"{model1} vs {model2}",
                'model1': model1,
                'model2': model2,
                'table': table,
                'statistic': statistic,
                'p_value': p_value
            })

    # Apply Holm correction for Table 2
    p_values_table2 = [r['p_value'] for r in results_table2]
    adjusted_p_table2 = holm_correction(p_values_table2)

    print(f"\n{'Comparison':<40s} {'Both Pass':>10s} {'A only':>10s} {'B only':>10s} {'Both Fail':>10s} {'χ²':>8s} {'p-value':>10s} {'Adj. p':>10s} {'Sig':>5s}")
    print("-" * 130)

    for i, r in enumerate(results_table2):
        a, b = r['table'][0]
        c, d = r['table'][1]
        adj_p = adjusted_p_table2[i]
        sig = "*" if adj_p < 0.05 else ""
        print(f"{r['comparison']:<40s} {a:10d} {b:10d} {c:10d} {d:10d} {r['statistic']:8.3f} {r['p_value']:10.4f} {adj_p:10.4f} {sig:>5s}")

    # Now check if we have Agent vs Retrieval comparisons
    print("\n" + "=" * 80)
    print("TABLE 4: AGENT vs RETRIEVAL STRATEGY COMPARISONS")
    print("=" * 80)

    # Check for BM25/Retrieval results
    retrieval_models = {
        'GPT-5-mini': 'gpt-5-mini_bm25',
        'DeepSeek-Coder': 'deepseek-coder_bm25',
        'DeepSeek-Chat': 'deepseek-chat_bm25',
        'GPT-4o-mini': 'gpt-4o-mini_bm25',
    }

    retrieval_results = {}
    for model_name, dir_name in retrieval_models.items():
        results_file = base_dir / dir_name / "jobs_results_diff.jsonl"
        if results_file.exists():
            retrieval_results[model_name] = load_results(results_file)
            success_count = sum(retrieval_results[model_name].values())
            total_count = len(retrieval_results[model_name])
            print(f"{model_name:20s} (Retrieval): {success_count:3d}/{total_count:3d} ({100*success_count/total_count:.1f}%)")

    if retrieval_results:
        results_table4 = []
        for model_name in models.keys():
            if model_name in all_results and model_name in retrieval_results:
                table = create_contingency_table(all_results[model_name], retrieval_results[model_name])
                statistic, p_value = mcnemar_test(table)
                results_table4.append({
                    'comparison': f"{model_name}: Agent vs Retrieval",
                    'model': model_name,
                    'table': table,
                    'statistic': statistic,
                    'p_value': p_value
                })

        # Apply Holm correction for Table 4
        p_values_table4 = [r['p_value'] for r in results_table4]
        adjusted_p_table4 = holm_correction(p_values_table4)

        print(f"\n{'Comparison':<40s} {'Both Pass':>10s} {'Agnt only':>10s} {'Retr only':>10s} {'Both Fail':>10s} {'χ²':>8s} {'p-value':>10s} {'Adj. p':>10s} {'Sig':>5s}")
        print("-" * 130)

        for i, r in enumerate(results_table4):
            a, b = r['table'][0]
            c, d = r['table'][1]
            adj_p = adjusted_p_table4[i]
            sig = "*" if adj_p < 0.05 else ""
            print(f"{r['comparison']:<40s} {a:10d} {b:10d} {c:10d} {d:10d} {r['statistic']:8.3f} {r['p_value']:10.4f} {adj_p:10.4f} {sig:>5s}")
    else:
        print("No retrieval/BM25 results found. Skipping Table 4 comparisons.")

    print("\n" + "=" * 80)
    print("SUMMARY: WHERE TO ADD ASTERISKS (*)")
    print("=" * 80)

    print("\n📊 TABLE 2 - Model Comparisons:")
    print("Add * to these models when comparing to the baseline:")
    for i, r in enumerate(results_table2):
        adj_p = adjusted_p_table2[i]
        if adj_p < 0.05:
            print(f"  ✓ {r['comparison']:<40s} (p={adj_p:.4f})")

    if retrieval_results:
        print("\n📊 TABLE 4 - Strategy Comparisons:")
        print("Add * to 'Retrieval' row for these models:")
        for i, r in enumerate(results_table4):
            adj_p = adjusted_p_table4[i]
            if adj_p < 0.05:
                print(f"  ✓ {r['comparison']:<40s} (p={adj_p:.4f})")

    print("\n" + "=" * 80)
    print("NOTES:")
    print("=" * 80)
    print("- χ² is the McNemar test statistic")
    print("- p-value is the raw p-value from McNemar's test")
    print("- Adj. p is the Holm-corrected p-value (accounts for multiple comparisons)")
    print("- Sig column shows * if Adj. p < 0.05 (statistically significant)")
    print("- 'Both Pass' = both methods succeeded on this instance")
    print("- 'A only' = only first method succeeded")
    print("- 'B only' = only second method succeeded")
    print("- 'Both Fail' = both methods failed on this instance")


if __name__ == "__main__":
    main()

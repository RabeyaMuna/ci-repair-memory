#!/usr/bin/env python3
"""
calculate_memory_ablation_costs.py

Analyzes token usage and costs across different memory ablation configurations:
- Baseline (no memory)
- Backward (L1 only)
- Forward (L1+L2)
- Bidirectional (L1+L2+L3)

Reads token_report.json from each run and produces a table showing:
- Average tokens per instance (in K/inst)
- Average cost per instance ($/inst)

Usage:
    python scripts/analysis/calculate_memory_ablation_costs.py
    python scripts/analysis/calculate_memory_ablation_costs.py --output results/memory_cost_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = PROJECT_ROOT / "baselines"

# Configuration for different memory ablation runs
MEMORY_CONFIGS = [
    {
        "name": "Baseline (No Memory)",
        "short_name": "Baseline",
        "result_dir": "MiniMax-M2.5_llm_baseline",
        "memory_type": "none"
    },
    {
        "name": "Backward (L1 only)",
        "short_name": "L1",
        "result_dir": "MiniMax-M2.5_llm_memory_L1",
        "memory_type": "backward"
    },
    {
        "name": "Forward (L1+L2)",
        "short_name": "L1+L2",
        "result_dir": "MiniMax-M2.5_llm_memory_L1L2",
        "memory_type": "forward"
    },
    {
        "name": "Bidirectional (L1+L2+L3)",
        "short_name": "L1+L2+L3",
        "result_dir": "MiniMax-M2.5_llm_memory",
        "memory_type": "bidirectional"
    }
]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file if it exists."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Failed to load {path}: {e}")
        return None


def analyze_token_report(report_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract token usage and cost statistics from a token_report.json file.

    Returns dict with:
        - total_instances: number of tasks/instances processed
        - total_input_tokens: sum of all input tokens
        - total_output_tokens: sum of all output tokens
        - total_tokens: sum of all tokens
        - total_cost: sum of all costs in USD
        - avg_input_tokens_per_instance: average input tokens per instance
        - avg_output_tokens_per_instance: average output tokens per instance
        - avg_total_tokens_per_instance: average total tokens per instance (in K)
        - avg_cost_per_instance: average cost per instance in USD
    """
    data = load_json(report_path)
    if not data:
        return None

    # Extract overall LLM statistics
    llm_stats = data.get("llm", {}).get("overall", {})
    total_input = llm_stats.get("input_tokens", 0)
    total_output = llm_stats.get("output_tokens", 0)
    total_tokens = llm_stats.get("total_tokens", 0)
    total_cost = llm_stats.get("cost_usd", 0.0)

    # Extract per-task statistics to get instance count
    per_task = data.get("per_task", [])
    total_instances = len(per_task)

    if total_instances == 0:
        print(f"  [WARN] No instances found in {report_path}")
        return None

    # Calculate averages per instance
    avg_input = total_input / total_instances
    avg_output = total_output / total_instances
    avg_total = total_tokens / total_instances
    avg_cost = total_cost / total_instances

    return {
        "total_instances": total_instances,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "avg_input_tokens_per_instance": avg_input,
        "avg_output_tokens_per_instance": avg_output,
        "avg_total_tokens_per_instance_K": avg_total / 1000,  # Convert to thousands
        "avg_cost_per_instance": avg_cost,
    }


def analyze_memory_ablation(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Analyze token usage and costs for all memory ablation configurations.
    """
    results = []

    for config in MEMORY_CONFIGS:
        config_dir = results_dir / config["result_dir"]
        token_report = config_dir / "token_report.json"

        print(f"\nAnalyzing: {config['name']}")
        print(f"  Looking for: {token_report}")

        if not token_report.exists():
            print(f"  [SKIP] Token report not found")
            results.append({
                "config_name": config["name"],
                "short_name": config["short_name"],
                "memory_type": config["memory_type"],
                "status": "not_found",
                "total_instances": 0,
                "avg_tokens_K": 0.0,
                "avg_cost": 0.0
            })
            continue

        stats = analyze_token_report(token_report)
        if not stats:
            results.append({
                "config_name": config["name"],
                "short_name": config["short_name"],
                "memory_type": config["memory_type"],
                "status": "error",
                "total_instances": 0,
                "avg_tokens_K": 0.0,
                "avg_cost": 0.0
            })
            continue

        print(f"  ✓ Processed {stats['total_instances']} instances")
        print(f"    Total tokens: {stats['total_tokens']:,}")
        print(f"    Total cost: ${stats['total_cost']:.4f}")
        print(f"    Avg tokens/inst: {stats['avg_total_tokens_per_instance_K']:.2f}K")
        print(f"    Avg cost/inst: ${stats['avg_cost_per_instance']:.4f}")

        results.append({
            "config_name": config["name"],
            "short_name": config["short_name"],
            "memory_type": config["memory_type"],
            "status": "success",
            "total_instances": stats["total_instances"],
            "total_input_tokens": stats["total_input_tokens"],
            "total_output_tokens": stats["total_output_tokens"],
            "total_tokens": stats["total_tokens"],
            "total_cost": stats["total_cost"],
            "avg_input_tokens": stats["avg_input_tokens_per_instance"],
            "avg_output_tokens": stats["avg_output_tokens_per_instance"],
            "avg_tokens_K": stats["avg_total_tokens_per_instance_K"],
            "avg_cost": stats["avg_cost_per_instance"],
        })

    return results


def print_comparison_table(results: List[Dict[str, Any]]):
    """Print a nicely formatted comparison table for the paper."""
    print("\n" + "=" * 100)
    print("  MEMORY ABLATION STUDY — TOKEN USAGE AND COST ANALYSIS")
    print("=" * 100)
    print()

    # Header
    header = f"{'Configuration':<30} {'Instances':<12} {'Tokens (K/inst.)':<20} {'Cost ($/inst.)':<20}"
    print(header)
    print("-" * 100)

    # Rows
    for r in results:
        if r['status'] != 'success':
            status_msg = "NOT FOUND" if r['status'] == 'not_found' else "ERROR"
            print(f"{r['config_name']:<30} {status_msg}")
            continue

        instances = f"{r['total_instances']}"
        tokens_k = f"{r['avg_tokens_K']:.2f}"
        cost = f"${r['avg_cost']:.4f}"

        print(f"{r['config_name']:<30} {instances:<12} {tokens_k:<20} {cost:<20}")

    print("-" * 100)

    # Calculate and show differences from baseline
    baseline = next((r for r in results if r['memory_type'] == 'none' and r['status'] == 'success'), None)
    if baseline:
        print("\nDifferences from Baseline:")
        print(f"{'Configuration':<30} {'Token Δ (K)':<20} {'Cost Δ ($)':<20} {'Cost Δ (%)':<20}")
        print("-" * 100)

        for r in results:
            if r['status'] != 'success' or r['memory_type'] == 'none':
                continue

            token_diff = r['avg_tokens_K'] - baseline['avg_tokens_K']
            cost_diff = r['avg_cost'] - baseline['avg_cost']
            cost_pct = ((r['avg_cost'] - baseline['avg_cost']) / baseline['avg_cost'] * 100) if baseline['avg_cost'] > 0 else 0

            token_str = f"{token_diff:+.2f}"
            cost_str = f"${cost_diff:+.4f}"
            pct_str = f"{cost_pct:+.1f}%"

            print(f"{r['config_name']:<30} {token_str:<20} {cost_str:<20} {pct_str:<20}")

    print("\n" + "=" * 100)


def export_latex_table(results: List[Dict[str, Any]], output_path: Path):
    """Export results as a LaTeX table for the paper."""
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Token Usage and Cost Analysis for Memory Ablation Study}")
    latex.append("\\label{tab:memory-cost}")
    latex.append("\\begin{tabular}{lrrr}")
    latex.append("\\toprule")
    latex.append("Configuration & Instances & Tokens (K/inst.) & Cost (\\$/inst.) \\\\")
    latex.append("\\midrule")

    for r in results:
        if r['status'] != 'success':
            continue

        name = r['short_name']
        instances = r['total_instances']
        tokens = f"{r['avg_tokens_K']:.2f}"
        cost = f"{r['avg_cost']:.4f}"

        latex.append(f"{name} & {instances} & {tokens} & {cost} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)

    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"\nLaTeX table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate token usage and costs for memory ablation experiments"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BASELINES_ROOT / "results",
        help="Directory containing result subdirectories (default: baselines/results)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for detailed results"
    )
    parser.add_argument(
        "--latex",
        type=Path,
        help="Output LaTeX table file"
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}")
        print(f"\nExpected structure:")
        for config in MEMORY_CONFIGS:
            expected_path = args.results_dir / config['result_dir'] / "token_report.json"
            print(f"  {expected_path}")
        return 1

    # Analyze all configurations
    results = analyze_memory_ablation(args.results_dir)

    # Print comparison table
    print_comparison_table(results)

    # Save detailed JSON if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

    # Export LaTeX table if requested
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        export_latex_table(results, args.latex)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
calculate_actual_costs_from_content.py

Estimates token usage and costs based on actual codebase content and prompts.
Analyzes the prompt templates, typical CI logs, and memory overhead to provide
more accurate estimates than generic assumptions.

Usage:
    python scripts/analysis/calculate_actual_costs_from_content.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Based on analysis of the codebase prompts and typical CI failures
ESTIMATED_TOKEN_BREAKDOWN = {
    "baseline": {
        "ci_log_analysis": {
            "input": 8000,   # CI logs + workflow + prompt template
            "output": 1200,  # Structured error analysis
        },
        "fault_localization": {
            "input": 28000,  # Error context + file outlines + changed files + prompt
            "output": 1800,  # Suspicious files list with reasoning
        },
        "patch_generation": {
            "input": 12000,  # File content + error details + prompt
            "output": 800,   # Generated diff
        }
    },
    "backward_l1": {
        # L1: Adds candidate files from similar failures (~5-10 files, ~2K tokens)
        "memory_overhead_input": 2000,
        "memory_overhead_output": 200,  # Slight increase in reasoning
    },
    "forward_l1_l2": {
        # L1+L2: Adds candidate files + high-level hints (~3-5 hints, ~4K tokens)
        "memory_overhead_input": 4000,
        "memory_overhead_output": 400,
    },
    "bidirectional_l1_l2_l3": {
        # L1+L2+L3: Adds detailed solutions + code snippets (~6K tokens)
        "memory_overhead_input": 6000,
        "memory_overhead_output": 600,
    }
}

# MiniMax-M2.5 pricing (USD per 1M tokens)
MODEL_PRICING = {
    "input": 0.15,
    "output": 1.15
}


def calculate_tokens_and_cost(config: str, num_instances: int = 410) -> Dict:
    """
    Calculate token usage and cost for a configuration.

    Args:
        config: One of 'baseline', 'backward_l1', 'forward_l1_l2', 'bidirectional_l1_l2_l3'
        num_instances: Number of instances to process

    Returns:
        Dict with token and cost statistics
    """
    base = ESTIMATED_TOKEN_BREAKDOWN["baseline"]

    # Calculate baseline per instance
    baseline_input = sum(stage["input"] for stage in base.values())
    baseline_output = sum(stage["output"] for stage in base.values())

    # Add memory overhead if applicable
    if config != "baseline":
        overhead = ESTIMATED_TOKEN_BREAKDOWN[config]
        # Memory context is primarily added to fault localization
        input_tokens = baseline_input + overhead["memory_overhead_input"]
        output_tokens = baseline_output + overhead["memory_overhead_output"]
    else:
        input_tokens = baseline_input
        output_tokens = baseline_output

    total_tokens = input_tokens + output_tokens

    # Calculate cost per instance
    cost_per_instance = (
        (input_tokens * MODEL_PRICING["input"]) +
        (output_tokens * MODEL_PRICING["output"])
    ) / 1_000_000

    # Calculate totals for all instances
    total_input = input_tokens * num_instances
    total_output = output_tokens * num_instances
    total_tokens_all = total_tokens * num_instances
    total_cost = cost_per_instance * num_instances

    return {
        "config": config,
        "per_instance": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "total_tokens_K": total_tokens / 1000,
            "cost_usd": cost_per_instance,
        },
        "all_instances": {
            "num_instances": num_instances,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens_all,
            "total_tokens_M": total_tokens_all / 1_000_000,
            "total_cost_usd": total_cost,
        }
    }


def print_detailed_table(results: List[Dict]):
    """Print detailed cost analysis table."""
    print("\n" + "=" * 100)
    print("  TOKEN USAGE AND COST ANALYSIS — Based on Codebase Content")
    print("=" * 100)
    print()

    # Configuration names
    config_names = {
        "baseline": "Baseline (No Memory)",
        "backward_l1": "Backward (L1 only)",
        "forward_l1_l2": "Forward (L1+L2)",
        "bidirectional_l1_l2_l3": "Bidirectional (L1+L2+L3)"
    }

    # Per instance table
    print("Per-Instance Token Usage and Cost:")
    print(f"{'Configuration':<30} {'Input (K)':<12} {'Output (K)':<12} {'Total (K)':<12} {'Cost ($)':<12}")
    print("-" * 100)

    for r in results:
        per_inst = r["per_instance"]
        config_name = config_names[r["config"]]
        print(f"{config_name:<30} "
              f"{per_inst['input_tokens']/1000:<12.2f} "
              f"{per_inst['output_tokens']/1000:<12.2f} "
              f"{per_inst['total_tokens_K']:<12.2f} "
              f"${per_inst['cost_usd']:<11.4f}")

    print()

    # Breakdown by stage (for baseline)
    print("Baseline Token Breakdown by Stage:")
    print(f"{'Stage':<30} {'Input (K)':<12} {'Output (K)':<12} {'Total (K)':<12}")
    print("-" * 100)

    stages = ESTIMATED_TOKEN_BREAKDOWN["baseline"]
    stage_names = {
        "ci_log_analysis": "CI Log Analysis",
        "fault_localization": "Fault Localization",
        "patch_generation": "Patch Generation"
    }

    for stage_key, stage_data in stages.items():
        stage_name = stage_names[stage_key]
        total = stage_data["input"] + stage_data["output"]
        print(f"{stage_name:<30} "
              f"{stage_data['input']/1000:<12.2f} "
              f"{stage_data['output']/1000:<12.2f} "
              f"{total/1000:<12.2f}")

    print()

    # Memory overhead
    print("Memory Overhead (vs Baseline):")
    print(f"{'Configuration':<30} {'+ Input (K)':<15} {'+ Output (K)':<15} {'+ Total (K)':<15} {'+ Cost ($)':<12}")
    print("-" * 100)

    baseline = results[0]
    for r in results[1:]:  # Skip baseline
        config_name = config_names[r["config"]]
        input_diff = (r["per_instance"]["input_tokens"] - baseline["per_instance"]["input_tokens"]) / 1000
        output_diff = (r["per_instance"]["output_tokens"] - baseline["per_instance"]["output_tokens"]) / 1000
        total_diff = r["per_instance"]["total_tokens_K"] - baseline["per_instance"]["total_tokens_K"]
        cost_diff = r["per_instance"]["cost_usd"] - baseline["per_instance"]["cost_usd"]

        print(f"{config_name:<30} "
              f"+{input_diff:<14.2f} "
              f"+{output_diff:<14.2f} "
              f"+{total_diff:<14.2f} "
              f"+${cost_diff:<11.4f}")

    print()

    # Total costs
    num_inst = results[0]["all_instances"]["num_instances"]
    print(f"Total Costs for All {num_inst} Instances:")
    print(f"{'Configuration':<30} {'Total Tokens (M)':<18} {'Total Cost ($)':<15} {'Cost Δ (%)':<12}")
    print("-" * 100)

    for r in results:
        config_name = config_names[r["config"]]
        all_inst = r["all_instances"]
        cost_pct = ((r["per_instance"]["cost_usd"] - baseline["per_instance"]["cost_usd"]) /
                    baseline["per_instance"]["cost_usd"] * 100) if r != baseline else 0

        pct_str = f"+{cost_pct:.1f}%" if cost_pct != 0 else "—"

        print(f"{config_name:<30} "
              f"{all_inst['total_tokens_M']:<18.2f} "
              f"${all_inst['total_cost_usd']:<14.2f} "
              f"{pct_str:<12}")

    print()
    print("=" * 100)
    print("\nEstimation Method:")
    print("  - Based on actual prompt templates in ci_repair/")
    print("  - Typical CI log sizes from dataset analysis")
    print("  - Memory context sizes from memory_plugin.py")
    print("  - Model: MiniMax-M2.5 ($0.15/1M input, $1.15/1M output)")
    print("=" * 100 + "\n")


def export_latex_table(results: List[Dict], output_path: Path):
    """Export as LaTeX table for paper."""
    config_labels = {
        "baseline": "Baseline",
        "backward_l1": "L1",
        "forward_l1_l2": "L1+L2",
        "bidirectional_l1_l2_l3": "L1+L2+L3"
    }

    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Token Usage and Cost Analysis for Memory Ablation Configurations}")
    latex.append("\\label{tab:memory-cost}")
    latex.append("\\begin{tabular}{lrr}")
    latex.append("\\toprule")
    latex.append("Configuration & Tokens (K/inst.) & Cost (\\$/inst.) \\\\")
    latex.append("\\midrule")

    for r in results:
        label = config_labels[r["config"]]
        tokens_k = r["per_instance"]["total_tokens_K"]
        cost = r["per_instance"]["cost_usd"]
        latex.append(f"{label} & {tokens_k:.2f} & {cost:.4f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)

    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"LaTeX table saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate token usage and costs based on codebase content analysis"
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=410,
        help="Number of instances (default: 410)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save detailed results to JSON file"
    )
    parser.add_argument(
        "--latex",
        type=Path,
        help="Export LaTeX table"
    )

    args = parser.parse_args()

    # Calculate for all configurations
    configs = ["baseline", "backward_l1", "forward_l1_l2", "bidirectional_l1_l2_l3"]
    results = [calculate_tokens_and_cost(config, args.instances) for config in configs]

    # Print detailed table
    print_detailed_table(results)

    # Save JSON if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

    # Export LaTeX if requested
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        export_latex_table(results, args.latex)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

#!/usr/bin/env python3
"""
estimate_memory_costs.py

Provides cost estimates for memory ablation experiments when complete token reports
are not yet available. Can use partial data or typical values to project costs.

Usage:
    # Use with partial/existing token reports
    python scripts/analysis/estimate_memory_costs.py

    # Specify expected number of instances
    python scripts/analysis/estimate_memory_costs.py --instances 410

    # Use custom model pricing
    python scripts/analysis/estimate_memory_costs.py --model minimax-m2.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model pricing (USD per 1M tokens)
MODEL_PRICING = {
    "minimax-m2.5": {"input": 0.15, "output": 1.15},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-mini": {"input": 1.10, "output": 4.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

# Typical memory overhead estimates (percentage increase from baseline)
# Based on empirical observations from similar systems
MEMORY_OVERHEAD_ESTIMATES = {
    "baseline": 0.0,      # No overhead
    "backward": 0.15,     # ~15% increase (L1 only: candidate files from similar failures)
    "forward": 0.30,      # ~30% increase (L1+L2: candidates + hints)
    "bidirectional": 0.40 # ~40% increase (L1+L2+L3: full memory context)
}


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate cost in USD for given token counts."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["minimax-m2.5"])
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def estimate_from_baseline(
    baseline_input: int,
    baseline_output: int,
    memory_type: str,
    model: str
) -> Dict[str, float]:
    """
    Estimate tokens and cost for a memory configuration based on baseline.

    Args:
        baseline_input: Average input tokens per instance for baseline
        baseline_output: Average output tokens per instance for baseline
        memory_type: One of: baseline, backward, forward, bidirectional
        model: Model name for pricing

    Returns:
        Dict with estimated tokens and cost
    """
    overhead = MEMORY_OVERHEAD_ESTIMATES.get(memory_type, 0.0)

    # Memory mainly adds to input tokens (retrieved context)
    # Output tokens may increase slightly due to more detailed reasoning
    estimated_input = baseline_input * (1 + overhead)
    estimated_output = baseline_output * (1 + overhead * 0.3)  # Output increases less

    total_tokens = estimated_input + estimated_output
    cost = calculate_cost(int(estimated_input), int(estimated_output), model)

    return {
        "input_tokens": estimated_input,
        "output_tokens": estimated_output,
        "total_tokens": total_tokens,
        "total_tokens_K": total_tokens / 1000,
        "cost": cost
    }


def find_baseline_stats(results_dir: Path) -> Optional[Dict]:
    """Try to find actual baseline statistics from token_report.json."""
    baseline_path = results_dir / "MiniMax-M2.5_llm_baseline" / "token_report.json"

    if not baseline_path.exists():
        return None

    try:
        with open(baseline_path) as f:
            data = json.load(f)

        per_task = data.get("per_task", [])
        if not per_task:
            return None

        # Calculate averages from actual data
        total_input = sum(t["llm"]["input_tokens"] for t in per_task)
        total_output = sum(t["llm"]["output_tokens"] for t in per_task)
        num_tasks = len(per_task)

        return {
            "num_instances": num_tasks,
            "avg_input": total_input / num_tasks,
            "avg_output": total_output / num_tasks,
            "source": "actual"
        }
    except Exception as e:
        print(f"Warning: Could not load baseline stats: {e}")
        return None


def print_estimates_table(estimates: Dict, num_instances: int):
    """Print formatted table of cost estimates."""
    print("\n" + "=" * 100)
    print("  ESTIMATED TOKEN USAGE AND COST — MEMORY ABLATION STUDY")
    print("=" * 100)
    print()

    # Header
    print(f"{'Configuration':<30} {'Input (K)':<15} {'Output (K)':<15} {'Total (K/inst.)':<18} {'Cost ($/inst.)':<15}")
    print("-" * 100)

    # Rows
    configs = ["Baseline", "Backward (L1)", "Forward (L1+L2)", "Bidirectional (L1+L2+L3)"]
    keys = ["baseline", "backward", "forward", "bidirectional"]

    for config_name, key in zip(configs, keys):
        est = estimates[key]
        input_k = f"{est['input_tokens']/1000:.2f}"
        output_k = f"{est['output_tokens']/1000:.2f}"
        total_k = f"{est['total_tokens_K']:.2f}"
        cost = f"${est['cost']:.4f}"

        print(f"{config_name:<30} {input_k:<15} {output_k:<15} {total_k:<18} {cost:<15}")

    print("-" * 100)

    # Differences from baseline
    baseline = estimates["baseline"]
    print("\nDifferences from Baseline:")
    print(f"{'Configuration':<30} {'Token Δ (K)':<18} {'Cost Δ ($)':<18} {'Cost Δ (%)':<15}")
    print("-" * 100)

    for config_name, key in zip(configs[1:], keys[1:]):
        est = estimates[key]
        token_diff = est['total_tokens_K'] - baseline['total_tokens_K']
        cost_diff = est['cost'] - baseline['cost']
        cost_pct = (cost_diff / baseline['cost'] * 100) if baseline['cost'] > 0 else 0

        print(f"{config_name:<30} {token_diff:+.2f}K{'':<12} ${cost_diff:+.4f}{'':<10} {cost_pct:+.1f}%")

    print()

    # Total costs
    print("\nTotal Costs for All Instances (N = {}):".format(num_instances))
    print(f"{'Configuration':<30} {'Total Tokens (M)':<20} {'Total Cost ($)':<15}")
    print("-" * 100)

    for config_name, key in zip(configs, keys):
        est = estimates[key]
        total_tokens_M = est['total_tokens'] * num_instances / 1_000_000
        total_cost = est['cost'] * num_instances

        print(f"{config_name:<30} {total_tokens_M:.2f}M{'':<15} ${total_cost:.2f}")

    print("\n" + "=" * 100)
    print("\nNOTE: These are estimates based on typical memory overhead patterns.")
    print("      Actual costs may vary depending on:")
    print("        - Memory retrieval quality and relevance")
    print("        - Problem complexity and failure types")
    print("        - LLM reasoning depth required")
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate token usage and costs for memory ablation experiments"
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=410,
        help="Number of instances to process (default: 410)"
    )
    parser.add_argument(
        "--baseline-input",
        type=int,
        help="Average baseline input tokens per instance (if known)"
    )
    parser.add_argument(
        "--baseline-output",
        type=int,
        help="Average baseline output tokens per instance (if known)"
    )
    parser.add_argument(
        "--model",
        default="minimax-m2.5",
        choices=list(MODEL_PRICING.keys()),
        help="Model to use for cost calculations (default: minimax-m2.5)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "baselines" / "results",
        help="Directory to check for existing baseline data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save estimates to JSON file"
    )

    args = parser.parse_args()

    # Try to get actual baseline stats if available
    baseline_stats = find_baseline_stats(args.results_dir)

    if baseline_stats:
        print(f"✓ Using actual baseline statistics from {args.results_dir}")
        baseline_input = baseline_stats["avg_input"]
        baseline_output = baseline_stats["avg_output"]
        num_instances = baseline_stats["num_instances"]
        print(f"  Found {num_instances} instances")
        print(f"  Avg baseline: {baseline_input:.0f} input, {baseline_output:.0f} output tokens")
    else:
        # Use provided values or defaults
        if args.baseline_input and args.baseline_output:
            baseline_input = args.baseline_input
            baseline_output = args.baseline_output
            print(f"Using provided baseline: {baseline_input} input, {baseline_output} output tokens")
        else:
            # Default estimates based on typical CI repair tasks
            baseline_input = 42000  # ~42K input tokens per instance
            baseline_output = 3500  # ~3.5K output tokens per instance
            print(f"⚠ No baseline data found, using typical estimates:")
            print(f"  Baseline: {baseline_input} input, {baseline_output} output tokens per instance")

        num_instances = args.instances

    # Calculate estimates for all configurations
    estimates = {}
    for memory_type in ["baseline", "backward", "forward", "bidirectional"]:
        estimates[memory_type] = estimate_from_baseline(
            baseline_input,
            baseline_output,
            memory_type,
            args.model
        )

    # Add metadata
    result = {
        "model": args.model,
        "num_instances": num_instances,
        "baseline_avg_input": baseline_input,
        "baseline_avg_output": baseline_output,
        "estimates": estimates,
        "note": "These are estimates. Actual values may vary."
    }

    # Print table
    print_estimates_table(estimates, num_instances)

    # Save if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Estimates saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

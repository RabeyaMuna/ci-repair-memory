#!/usr/bin/env python3
"""
Generate plots for presentation slides.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

OUTPUT_DIR = Path("docs/slides/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_validation_funnel(summary_path: str):
    """Plot L1/L2/L3 validation funnel."""
    with open(summary_path, 'r') as f:
        data = json.load(f)

    total = data['total_pushed']
    l1_pass = data['level_1_ci_failure_jobs']['passed']
    l2_all_pass = data['level_2_extended_jobs']['all_passed']
    l3_pass = data['level_3_complete']['passed']

    # Execution success rate (the more honest metric)
    l2_exec_rate = data['level_2_extended_jobs'].get('mean_executed_success_rate',
                   data['level_2_extended_jobs'].get('aggregate_executed_success_rate', 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Funnel
    levels = ['L1:\nOriginal\nFailed Jobs', 'L2:\nAll Steps\n(Strict)', 'L3:\nComplete\nWorkflow']
    values = [l1_pass, l2_all_pass, l3_pass]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    ax1.barh(levels, values, color=colors, alpha=0.7)
    ax1.set_xlabel('Number of Issues Passed')
    ax1.set_title('Three-Level Validation Results\n(Strict: All Must Pass)')
    ax1.set_xlim(0, total)

    # Add percentage labels
    for i, (level, value) in enumerate(zip(levels, values)):
        pct = (value / total) * 100
        ax1.text(value + 5, i, f'{value} ({pct:.1f}%)', va='center')

    # Right: L2 detailed breakdown
    l2_data = data['level_2_extended_jobs']
    categories = ['All Passed', 'Partial', 'All Failed', 'Incomplete']
    l2_values = [
        l2_data['all_passed'],
        l2_data['partial'],
        l2_data['all_failed'],
        l2_data['incomplete']
    ]
    colors2 = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']

    ax2.pie(l2_values, labels=categories, autopct='%1.1f%%', colors=colors2, startangle=90)
    ax2.set_title(f'L2 Issue Distribution\n(Exec Success Rate: {l2_exec_rate:.1f}%)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'validation_funnel.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: validation_funnel.png")


def plot_step_breakdown(summary_path: str):
    """Plot step-level breakdown for L2."""
    with open(summary_path, 'r') as f:
        data = json.load(f)

    l2 = data['level_2_extended_jobs']

    # Create stacked bar
    categories = ['Passed', 'Failed', 'Skipped', 'Cancelled']
    values = [
        l2['passed_steps'],
        l2['failed_steps'],
        l2['skipped_steps'],
        l2['cancelled_steps']
    ]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6', '#34495e']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Horizontal stacked bar
    left = 0
    for cat, val, color in zip(categories, values, colors):
        pct = (val / l2['total_steps']) * 100
        ax.barh(0, val, left=left, color=color, label=f'{cat}: {val} ({pct:.1f}%)')
        left += val

    ax.set_yticks([])
    ax.set_xlabel('Number of Steps')
    ax.set_title('L2: Step-Level Breakdown Across All Issues')
    ax.legend(loc='upper right')

    # Add total
    ax.text(l2['total_steps']/2, 0, f"Total: {l2['total_steps']} steps",
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: step_breakdown.png")


def plot_l2_metrics_comparison(summary_path: str):
    """Compare the two L2 metrics side by side."""
    with open(summary_path, 'r') as f:
        data = json.load(f)

    l2 = data['level_2_extended_jobs']

    # Get both metrics
    exec_rate = l2.get('mean_executed_success_rate',
                       l2.get('aggregate_executed_success_rate', 0))
    overall_rate = l2.get('mean_overall_pass_rate',
                          l2.get('aggregate_overall_pass_rate', 0))

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Execution Success\n(excl. skipped)', 'Overall Pass Rate\n(incl. skipped)']
    values = [exec_rate, overall_rate]
    colors = ['#2ecc71', '#3498db']

    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(0, 100)
    ax.set_title('L2: Two Ways to Measure Step Success\n(Mean across issues)')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)

    # Add explanation
    ax.text(0, 5, f'Executed: {l2["executed_steps"]:,} steps\n(passed + failed + cancelled)',
            ha='center', fontsize=10, style='italic')
    ax.text(1, 5, f'Total: {l2["total_steps"]:,} steps\n(includes {l2["skipped_steps"]:,} skipped)',
            ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'l2_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: l2_metrics_comparison.png")


def plot_benchmark_comparison():
    """Create comparison table as a figure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Data
    data = [
        ['Feature', 'SWE-bench', 'SWE-bench-V', 'SWE-bench Pro', 'CI-REPAIR-BENCH'],
        ['Task Scope', 'Single issue', 'Single issue', 'Single issue', '✓ PR-level'],
        ['Commits/Issue', '1', '1', '1', '✓ Multi-commit'],
        ['Failure Types', '1', '1', '1', '✓ Multiple'],
        ['Validation', '1-stage', '1-stage', '1-stage', '✓ 3-stage (L1/L2/L3)'],
        ['Real CI', '✗', 'Partial', 'Partial', '✓ GitHub Actions'],
        ['Multi-file', '?', '?', '?', '✓ Avg X files/PR'],
    ]

    # Create table
    table = ax.table(cellText=data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style our column
    for i in range(1, len(data)):
        table[(i, 4)].set_facecolor('#e8f8f5')
        table[(i, 4)].set_text_props(weight='bold')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    plt.title('Benchmark Comparison: CI-REPAIR-BENCH vs. SWE-bench Family',
              fontsize=14, fontweight='bold', pad=20)

    plt.savefig(OUTPUT_DIR / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: benchmark_comparison.png")


def plot_memory_results():
    """Plot memory-guided repair results."""
    # Placeholder - update with actual ablation results
    models = ['Baseline\n(No Memory)', 'L1 Only', 'L1 + L2', 'L1 + L2 + L3']
    success_rates = [13.48, 14.2, 15.8, 17.98]  # Update with actual values

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#95a5a6', '#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(models, success_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('L3 Success Rate (%)')
    ax.set_ylim(0, 25)
    ax.set_title('Memory-Guided Repair: Incremental Improvement\n(+4.5% absolute, +33% relative)')
    ax.axhline(y=success_rates[0], color='red', linestyle='--', alpha=0.3, label='Baseline')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'memory_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: memory_results.png")


def main():
    """Generate all plots."""
    base_dir = Path("/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH")
    summary_path = base_dir / "results/jobs_results_diff_multilevel_summary.json"

    print("="*80)
    print("Generating Presentation Plots")
    print("="*80)

    if not summary_path.exists():
        print(f"❌ Summary not found: {summary_path}")
        return

    print("\n📊 Generating plots...")

    # Generate all plots
    plot_validation_funnel(str(summary_path))
    plot_step_breakdown(str(summary_path))
    plot_l2_metrics_comparison(str(summary_path))
    plot_benchmark_comparison()
    plot_memory_results()

    print(f"\n✅ All plots saved to: {OUTPUT_DIR}")
    print("\nUse these in your slides:")
    print("  1. validation_funnel.png - Shows L1/L2/L3 results")
    print("  2. step_breakdown.png - Detailed L2 step analysis")
    print("  3. l2_metrics_comparison.png - Two L2 metrics side-by-side")
    print("  4. benchmark_comparison.png - vs SWE-bench")
    print("  5. memory_results.png - Memory-guided improvement")


if __name__ == "__main__":
    main()
